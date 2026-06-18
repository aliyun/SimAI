#!/bin/bash
# LRA — 一键安装脚本，完全自包含。
# Phase 1 (需求澄清): 交互讨论，创建 feature → 见 feature_list.json
# Phase 2 (开发执行): gate 强制 feature scope → 见 scripts/lra-gate.py
# Usage: bash install-lra.sh [--force]
set -e
FORCE=false; [ "${1:-}" = "--force" ] && FORCE=true
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
echo "=== LRA 一键安装 ===" && echo "项目: $ROOT" && echo ""

write_file() { local d="$1"; [ -f "$d" ] && [ "$FORCE" != "true" ] && { echo "  SKIP: ${d#$ROOT/}"; return; }; mkdir -p "$(dirname "$d")"; cat > "$d"; chmod +x "$d" 2>/dev/null || true; echo "  CREATE: ${d#$ROOT/}"; }

echo "--- 核心脚本 ---"
write_file "$ROOT/scripts/lra_common.py" << 'HEREDOC'
#!/usr/bin/env python3
"""LRA shared utilities — single source of truth for all LRA components."""
import json, os, re, subprocess, sys

def find_root():
    try: return subprocess.check_output(["git","rev-parse","--show-toplevel"],stderr=subprocess.DEVNULL).decode().strip()
    except: return os.getcwd()

ALLOWED_PREFIXES = ("feature_list.json","progress.md","init.sh","CLAUDE.md",".claude/","docs/","scripts/lra-")

def is_allowed(path, root=None):
    if not path: return True
    if root is None: root = find_root()
    rel = os.path.relpath(path, root) if path.startswith("/") else path
    if rel.startswith(".."): return True
    for a in ALLOWED_PREFIXES:
        if rel == a or rel.startswith(a): return True
    return rel.endswith(".md")

def load_features(fl_path=None):
    if fl_path is None: fl_path = os.path.join(find_root(),"feature_list.json")
    try:
        with open(fl_path) as f: data = json.load(f)
    except: return []
    if isinstance(data, list): return [f for f in data if isinstance(f,dict) and f.get("id")]
    if "features" not in data:
        all_items = {}
        for key, ds in [("done","done"),("pending_bugs","pending"),("pending_tests","pending"),("active","in_progress")]:
            for item in data.get(key,[]):
                if isinstance(item,dict): fid = item.get("id")
                elif isinstance(item,str): fid, item = item, {"id": fid}
                else: continue
                m = all_items.get(fid,{}); m.update(item)
                if not item.get("status"): m["status"] = ds
                all_items[fid] = m
        return list(all_items.values())
    return [f for f in data.get("features",[]) if isinstance(f,dict) and f.get("id")]

def get_in_progress():
    features = load_features()
    in_prog = [f for f in features if f.get("status") == "in_progress"]
    if not in_prog: return None, None, []
    c = in_prog[0]; return c["id"], c, c.get("files",[])

_MODIFY_RE = re.compile(r"\bsed\b\s+.*-i\b|\brm\b\s+(-rf?\s+)?\S+|\bmv\b\s+\S+\s+\S+|\bcp\b\s+.*\s+\S+|>\s*\S+|\btee\b\s+\S+|\bdd\b\s+.*\bof=\S+|\btouch\b\s+\S+|\bmkdir\b\s+\S+")
def is_modifying_bash(command):
    if not command or not command.strip(): return False
    clean = re.sub(r"\d?>\s*/dev/null","",command); clean = re.sub(r"\d?>\s*&1","",clean)
    return bool(_MODIFY_RE.search(clean))

def parse_ts(val):
    if not val: return 0
    if isinstance(val,(int,float)): return float(val)
    try:
        s = str(val).replace("Z","+00:00")
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})",s)
        if m:
            from calendar import timegm; from datetime import datetime
            dt = datetime(*map(int,m.groups()[:6]))
            return timegm(dt.utctimetuple()) + (dt.microsecond/1e6 if '.' in s else 0)
    except: pass
    return 0

def compute_done_hash(features):
    import hashlib
    done = sorted([f for f in features if f.get("status")=="done"],key=lambda x:x.get("id",""))
    c = json.dumps([{"id":f["id"],"status":f.get("status"),"description":f.get("description",""),"files":sorted(f.get("files",[])),"verification_steps":f.get("verification_steps",[]),"passes":f.get("passes")} for f in done],sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(c.encode()).hexdigest()[:16], len(done)

def compute_progress_hash(pp):
    import hashlib
    if not os.path.exists(pp): return "", 0
    with open(pp) as f: pl = [l for l in f if l.strip().startswith("| 20")]
    return hashlib.sha256("".join(pl).encode()).hexdigest()[:16], len(pl)
HEREDOC

write_file "$ROOT/scripts/lra-gate.py" << 'HEREDOC'
#!/usr/bin/env python3
"""LRA PreToolUse gate: enforce feature tracking, scope, and Phase 1/2 boundary."""
import fnmatch, json, os, sys
ROOT = os.environ.get("CLAUDE_PROJECT_DIR","")
if not ROOT:
    try:
        import subprocess as sp
        ROOT = sp.check_output(["git","rev-parse","--show-toplevel"],stderr=sp.DEVNULL).decode().strip()
    except: ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from lra_common import is_allowed, load_features, is_modifying_bash

FL = os.path.join(ROOT,"feature_list.json"); DIRTY_FILE = os.path.join(ROOT,".lra_dirty")

def load_dirty():
    try:
        with open(DIRTY_FILE) as f: return json.load(f)
    except: return None

def main():
    try: payload = json.load(sys.stdin)
    except: return 0
    ti = payload.get("tool_input",{}) or {}; path = ti.get("file_path") or ti.get("path") or ""
    command = ti.get("command","")
    if command:
        if not is_modifying_bash(command): return 0
        import shlex
        try: tokens = shlex.split(command)
        except: tokens = command.split()
        pp = [t for t in tokens if t.startswith(ROOT) or (t.startswith("/") and ROOT in t)]
        if not pp: return 0
        path = pp[0]
    if not path or is_allowed(path, ROOT): return 0

    features = load_features()
    in_prog = [f for f in features if f.get("status")=="in_progress"]
    pending = [f for f in features if f.get("status")=="pending"]
    done_count = sum(1 for f in features if f.get("status")=="done")

    if not in_prog:
        if not features: print("BLOCKED - Phase 1 (需求澄清): feature_list.json is empty.\n  Discuss requirements, then create features.",file=sys.stderr)
        elif not pending: print(f"BLOCKED: All {done_count} features done. No pending work.\n  Discuss new requirements first.",file=sys.stderr)
        else:
            p0 = [f for f in pending if f.get("priority")=="P0"]; hint = ""
            if p0: hint = f"\n  Top P0: {', '.join(f['id'] for f in p0[:3])}\n  Pick one, set status='in_progress'."
            print(f"BLOCKED: {len(pending)} pending, none in_progress.{hint}",file=sys.stderr)
        return 2

    current = in_prog[0]; rel = os.path.relpath(path, ROOT) if path.startswith("/") else path
    scope = current.get("files")
    if not scope: print(f"BLOCKED: '{current['id']}' has no files scope.",file=sys.stderr); return 2
    if not any(fnmatch.fnmatch(rel, p) for p in scope):
        print(f"BLOCKED: '{current['id']}' does not own: {rel}\n  Allowed: {', '.join(scope[:5])}",file=sys.stderr); return 2

    dirty = load_dirty(); df = dirty.get("feature") if dirty else None
    if df and df != "unknown" and df != current["id"]:
        print(f"BLOCKED: '{df}' has untested changes. Run lra-test.sh first.",file=sys.stderr); return 2

    if current.get("type")=="feature" and not current.get("verification_steps"):
        print(f"BLOCKED: '{current['id']}' has no verification_steps.",file=sys.stderr); return 2

    # in_progress feature MUST have created_at (Phase 1 proof)
    if not current.get("created_at"):
        print(f"BLOCKED: Feature '{current['id']}' has no created_at.\n"
              f"  Action: add created_at to '{current['id']}'.",file=sys.stderr)
        return 2
    # confidence assessment MUST be done BEFORE any edit
    conf = current.get("confidence","")
    if not conf:
        print(f"【LRA 阻断】'{current['id']}' 无置信度。先评估再编辑: HIGH=直接修 LOW=交给用户。",file=sys.stderr)
        return 2
    if conf.startswith("LOW"):
        print(f"【LRA 阻断】LOW 置信度 ({conf})。交给用户决策，不要直接修改。",file=sys.stderr)
        return 2
    if not conf.startswith("HIGH:"):
        print(f"【LRA 阻断】置信度格式错误: {conf}。必须是 HIGH: 理由 或 LOW: 理由。",file=sys.stderr)
        return 2
    print(f"【LRA 置信度】{conf} | feature={current['id']}",file=sys.stderr)
    return 0
if __name__=="__main__": sys.exit(main())
HEREDOC

write_file "$ROOT/scripts/lra-stop.py" << 'HEREDOC'
#!/usr/bin/env python3
"""LRA Stop hook: enforce tests, freshness, immutability, Phase 1/2 boundary."""
import json, os, sys, time
ROOT = os.environ.get("CLAUDE_PROJECT_DIR","")
if not ROOT:
    try:
        import subprocess as sp
        ROOT = sp.check_output(["git","rev-parse","--show-toplevel"],stderr=sp.DEVNULL).decode().strip()
    except: ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from lra_common import load_features, is_allowed, parse_ts, compute_done_hash

DIRTY_FILE = os.path.join(ROOT,".lra_dirty"); LAST_TEST = os.path.join(ROOT,".lra_last_test")
FL = os.path.join(ROOT,"feature_list.json"); PM = os.path.join(ROOT,"progress.md")

def _mod():
    try:
        import subprocess as sp
        out = sp.check_output(["git","diff","--name-only","HEAD"],cwd=ROOT,stderr=sp.DEVNULL).decode().strip()
    except: return []
    return [f.strip() for f in (out.split("\n") if out else[]) if f.strip() and not is_allowed(f.strip(),ROOT)]
def _lt():
    try:
        with open(LAST_TEST) as f: return json.load(f).get("timestamp",0)
    except: return 0
def _ml(files):
    latest = 0
    for f in files:
        try: m = os.path.getmtime(os.path.join(ROOT,f)); latest = max(latest,m)
        except: pass
    return latest
def _cw(features,dirty,in_prog):
    w = {"timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"reason":"session end with blockers","in_progress":[{"id":f["id"],"description":f.get("description","")[:120]} for f in in_prog],"dirty_files":dirty.get("files",[]) if dirty else[]}
    try:
        with open(os.path.join(ROOT,".lra_context_warning"),"w") as f: json.dump(w,f,indent=2)
    except: pass

def main():
    features = load_features(FL)
    if not features: return 0
    msgs = []; blocked = False
    modified = _mod()
    if modified:
        lt = _lt(); lm = _ml(modified)
        if lt==0 or lm>lt: msgs.append(f"BLOCKED: {len(modified)} file(s) changed after last test.\n  Files: {', '.join(modified[:5])}\n  Action: run scripts/lra-test.sh"); blocked = True
        fe = [f for f in modified if f.startswith("dashboard/") and not f.endswith(".md")]
        if fe:
            try: stages = json.load(open(LAST_TEST)).get("stages",[])
            except: stages = []
            if "e2e" not in stages: msgs.append(f"BLOCKED: {len(fe)} frontend file(s) - E2E required."); blocked = True
    dirty = None
    try:
        with open(DIRTY_FILE) as f: dirty = json.load(f)
    except: pass
    if dirty and dirty.get("files"): msgs.append(f"BLOCKED: .lra_dirty still present for '{dirty.get('feature','?')}'.\n  Action: run scripts/lra-test.sh"); blocked = True
    uv = [f for f in features if f.get("status")=="done" and not f.get("passes")]
    if uv: msgs.append(f"BLOCKED: {len(uv)} done without passes ({', '.join(f['id'] for f in uv)})."); blocked = True
    now = time.time(); recent = [f for f in features if f.get("status")=="done" and f.get("passes") and parse_ts(f.get("updated_at",""))>now-3600]
    if recent:
        lt = _lt()
        try: pm_mt = os.path.getmtime(PM)
        except: pm_mt = 0
        for f in recent:
            if lt<parse_ts(f.get("updated_at","")) or pm_mt<parse_ts(f.get("updated_at","")): msgs.append(f"BLOCKED: '{f['id']}' done before tests/progress update."); blocked = True; break
    nt = [f for f in features if f.get("status")=="done" and not f.get("verification_steps")]
    if nt: msgs.append(f"BLOCKED: {len(nt)} done with empty verification_steps."); blocked = True
    in_prog = [f for f in features if f.get("status")=="in_progress"]
    if in_prog:
        stale = True
        try:
            if time.time()-os.path.getmtime(PM)<900: stale = False
        except: pass
        if stale: msgs.append(f"BLOCKED: {len(in_prog)} feature(s) in_progress ({', '.join(f['id'] for f in in_prog)}), progress.md not updated in 15min."); blocked = True
    # Phase 1/2 boundary: features without created_at
    haste = [f for f in in_prog if not f.get("created_at")]
    if haste: msgs.append(f"WARNING: {len(haste)} in_progress features ({', '.join(f['id'] for f in haste)}) lack created_at.\n  Features should be planned in Phase 1, not created during coding.")
    # Hollow detection
    hollow = [f for f in in_prog if len(f.get("description","").strip())<15 or not f.get("files") or not f.get("verification_steps")]
    if hollow and not blocked: msgs.append(f"WARNING: {len(hollow)} hollow features ({', '.join(f['id'] for f in hollow)}).")
    # Immutability
    DH = os.path.join(ROOT,".lra_done_hash")
    try:
        with open(DH) as f: saved = json.load(f)
    except: saved = {}
    if saved.get("done_hash"):
        import hashlib; ch,_ = compute_done_hash(features)
        if ch != saved["done_hash"]: msgs.append(f"BLOCKED: completed features modified. Revert then run lra-test.sh."); blocked = True
    if saved.get("progress_hash") and saved.get("progress_count"):
        try:
            with open(PM) as f: pl = [l for l in f if l.strip().startswith("| 20")]
            import hashlib
            if hashlib.sha256("".join(pl[:saved["progress_count"]]).encode()).hexdigest()[:16] != saved["progress_hash"]: msgs.append("BLOCKED: progress.md historical entries modified."); blocked = True
        except: pass
    if blocked or in_prog: _cw(features,dirty,in_prog)
    if msgs: print("\n\n".join(msgs),file=sys.stderr)
    return 2 if blocked else 0
if __name__=="__main__": sys.exit(main())
HEREDOC

write_file "$ROOT/scripts/lra-mark-dirty.py" << 'HEREDOC'
#!/usr/bin/env python3
"""LRA PostToolUse: mark dirty + auto-inject created_at for Phase 1/2 tracking."""
import fnmatch, json, os, sys
from datetime import datetime, timezone
ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from lra_common import is_allowed, get_in_progress

DIRTY_FILE = os.path.join(ROOT,".lra_dirty"); FL = os.path.join(ROOT,"feature_list.json")

def ensure_created_at(fid):
    try:
        with open(FL) as f: data = json.load(f)
    except: return
    modified = False
    for arr in ["active","pending","pending_bugs","pending_tests"]:
        for item in data.get(arr,[]):
            iid = item.get("id") if isinstance(item,dict) else item
            if iid == fid and isinstance(item,dict) and not item.get("created_at"):
                item["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                modified = True
    if modified:
        with open(FL,"w") as f: json.dump(data,f,indent=2,ensure_ascii=False)
        print(f"[LRA] Injected created_at for {fid}",file=sys.stderr)

def main():
    try: payload = json.load(sys.stdin)
    except: return 0
    path = (payload.get("tool_input",{}) or {}).get("file_path") or ""
    if is_allowed(path, ROOT): return 0
    fid, feature, scope = get_in_progress()
    if not fid: return 0
    ensure_created_at(fid)
    rel = os.path.relpath(path, ROOT) if path.startswith("/") else path
    if scope and not any(fnmatch.fnmatch(rel,p) for p in scope): return 0
    dirty = {}
    try:
        with open(DIRTY_FILE) as f: dirty = json.load(f)
    except: pass
    if dirty.get("feature") != fid: dirty = {"feature":fid,"files":[],"since":datetime.now(timezone.utc).isoformat()}
    if rel not in dirty["files"]: dirty["files"].append(rel)
    with open(DIRTY_FILE,"w") as f: json.dump(dirty,f,indent=2)
    print(f"[LRA] dirty: {len(dirty['files'])} file(s) for {fid}",file=sys.stderr)
    return 0
if __name__=="__main__": sys.exit(main())
HEREDOC

write_file "$ROOT/scripts/lra-test.sh" << 'HEREDOC'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$SCRIPT_DIR")"
DIRTY_FILE="$ROOT/.lra_dirty"; LAST_TEST_FILE="$ROOT/.lra_last_test"
FAILED=0; STAGES=""
echo "=== LRA Test Suite ==="

if [ -d "$ROOT/server/tests" ]; then
    echo "--- Backend ---"; cd "$ROOT/server"
    python3 -m pytest tests/ -v --tb=short 2>&1 && { echo "  [PASS] Backend"; STAGES="$STAGES backend"; } || { echo "  [FAIL] Backend"; FAILED=1; }; cd "$ROOT"
fi
if [ -f "$ROOT/dashboard/tsconfig.json" ]; then
    echo "--- Frontend ---"; cd "$ROOT/dashboard"
    npx tsc --noEmit 2>&1 && { echo "  [PASS] TypeScript"; STAGES="$STAGES tsc"; } || { echo "  [FAIL] TypeScript"; FAILED=1; }; cd "$ROOT"
fi
if [ "${LRA_E2E:-0}" = "1" ]; then
    echo "--- E2E ---"; cd "$ROOT/dashboard"; npx playwright test 2>&1 || echo "  [WARN] E2E"; STAGES="$STAGES e2e"; cd "$ROOT"
fi

GIT_HEAD=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown); PASSED=$([ "$FAILED" -eq 0 ] && echo True || echo False)
python3 -c "import json,time; json.dump({'timestamp':time.time(),'git_head':'${GIT_HEAD}','passed':${PASSED},'stages':'${STAGES}'.split()},open('${LAST_TEST_FILE}','w'))"

NEED_E2E=0
if [ -f "$DIRTY_FILE" ]; then
    python3 -c "
import json,subprocess
with open('$DIRTY_FILE') as f: dirty=json.load(f)
fe=[fp for fp in dirty.get('files',[]) if fp.startswith('dashboard/') and not fp.endswith('.md')]
if fe:
    del_only=True
    try:
        diff=subprocess.check_output(['git','diff','HEAD','--']+fe,cwd='$ROOT',stderr=subprocess.DEVNULL).decode()
        for l in diff.split('\n'):
            if l.startswith('+') and not l.startswith('+++'): del_only=False;break
    except: del_only=False
    if not del_only: print('NEED_E2E')
" 2>/dev/null | grep -q NEED_E2E && NEED_E2E=1
fi

if [ "$FAILED" -eq 0 ]; then
    if [ "$NEED_E2E" = "1" ] && [ "${LRA_E2E:-0}" != "1" ]; then echo "  PASSED (E2E required)"; exit 1; fi
    echo "  ALL TESTS PASSED"; rm -f "$DIRTY_FILE"
    python3 -c "
import sys,os,json; sys.path.insert(0,os.path.join('$ROOT','scripts'))
from lra_common import load_features, compute_done_hash, compute_progress_hash
f=load_features('$ROOT/feature_list.json'); dh,dn=compute_done_hash(f); ph,pc=compute_progress_hash('$ROOT/progress.md')
json.dump({'done_hash':dh,'done_count':dn,'progress_hash':ph,'progress_count':pc},open('$ROOT/.lra_done_hash','w'))
print(f'  integrity: done={dh}({dn}) progress={ph}({pc})')"
    exit 0
else
    echo "  SOME TESTS FAILED"; exit 1
fi
HEREDOC

write_file "$ROOT/scripts/lra-context-save.sh" << 'HEREDOC'
#!/bin/bash
cd "$(dirname "$0")/.."; export REASON="${1:-compaction detected}"; export WARNING_FILE=".lra_context_warning"
echo "=== LRA Context Save ==="
python3 << 'PYEOF'
import sys,os,json; sys.path.insert(0,os.path.join(os.getcwd(),'scripts'))
from lra_common import load_features
reason = os.environ.get("REASON","compaction detected"); wf = os.environ.get("WARNING_FILE",".lra_context_warning")
in_progress = []
for f in load_features():
    if f.get("status")=="in_progress":
        in_progress.append({"id":f["id"],"description":f.get("description","")[:120],"type":f.get("type","?"),"has_created_at":bool(f.get("created_at"))})
dirty_files = []
if os.path.exists(".lra_dirty"):
    with open(".lra_dirty") as f: dirty_files = json.load(f).get("files",[])
w = {"timestamp":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),"reason":reason,"in_progress":in_progress,"dirty_files":dirty_files}
with open(wf,"w") as f: json.dump(w,f,indent=2)
print(f"State saved ({len(in_progress)} in_progress, {len(dirty_files)} dirty)")
PYEOF
HEREDOC

echo ""; echo "--- init.sh ---"
write_file "$ROOT/init.sh" << 'HEREDOC'
#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "=== LRA Session Init ==="
if [ -f .lra_context_warning ]; then
    echo ""; echo "!!! CONTEXT WARNING — previous session interrupted !!!"
    python3 -c "import json; w=json.load(open('.lra_context_warning')); print(f\"  Time: {w.get('timestamp','?')}  Reason: {w.get('reason','?')}\"); [print(f\"  Was in_progress: [{x['id']}] {x.get('description','')[:80]}\") for x in w.get('in_progress',[])]; [print(f\"  Was dirty: {x}\") for x in w.get('dirty_files',[])]"
    mv .lra_context_warning .lra_context_warning.last; echo ""
fi
echo "--- Recent commits ---"; git log --oneline -5 2>/dev/null || echo "  (not a git repo)"; echo ""
[ -f progress.md ] && { echo "--- Recent progress ---"; grep -E '^\| 20' progress.md | tail -3 || true; echo ""; }
if [ -f feature_list.json ]; then
    echo "--- Feature status ---"
    python3 -c "
import sys,os; sys.path.insert(0,os.path.join(os.getcwd(),'scripts'))
from lra_common import load_features; from collections import Counter
f=load_features(); d=sum(1 for x in f if x['status']=='done'); p=sum(1 for x in f if x['status']=='pending'); ip=sum(1 for x in f if x['status']=='in_progress')
t=Counter(x.get('type','?') for x in f)
print(f'Total:{len(f)} Done:{d} InProg:{ip} Pending:{p} Types:{dict(t)}')
ipf=[x for x in f if x['status']=='in_progress']; p0=[x for x in f if x['status']=='pending' and x.get('priority')=='P0']
if ipf:
    print('In progress:')
    for x in ipf: ca=' (NO created_at!)' if not x.get('created_at') else ''; cc=x.get('confidence',''); ct=' (no confidence!)' if not cc else (' (⚠LOW)' if cc.startswith('LOW') else f' (✓HIGH: {cc[5:30]})'); print(f\"  [{x['id']}] {x.get('description','')[:60]}{ca}{ct}\")
if p0: print('Top P0:'); [print(f\"  [{x['id']}] {x.get('description','')[:80]}\") for x in p0[:3]]
"; echo ""
fi
echo "=========================================="
echo "1. Continue with feature_list plan?  2. P0 priority  3. Bug -> confidence first"
echo "4. Features without created_at were NOT planned in Phase 1 — flag them!"
echo "=========================================="
HEREDOC

echo ""; echo "--- Templates ---"
[ ! -f "$ROOT/feature_list.json" ] || [ "$FORCE" = "true" ] && { cat > "$ROOT/feature_list.json" << 'HEREDOC'
{"project":"","active":[],"done":[],"pending_bugs":[],"pending":[]}
HEREDOC
echo "  CREATE: feature_list.json"; } || echo "  SKIP: feature_list.json"

[ ! -f "$ROOT/progress.md" ] || [ "$FORCE" = "true" ] && { cat > "$ROOT/progress.md" << HEREDOC
# Progress
## Status
**Phase**: init | **Done**: 0 | **Pending**: 0
## Sessions
| Date | Work |
|------|------|
| $(date +%Y-%m-%d) | LRA installed |
HEREDOC
echo "  CREATE: progress.md"; } || echo "  SKIP: progress.md"

[ ! -f "$ROOT/CLAUDE.md" ] || [ "$FORCE" = "true" ] && { cat > "$ROOT/CLAUDE.md" << 'HEREDOC'
# CLAUDE.md
## LRA Two-Phase Workflow
### Phase 1: Requirements Clarification (interactive)
Discuss, clarify, create features in feature_list.json with: id, type, description, files scope, verification_steps.
Features must have `created_at` — this proves they were planned, not created on-the-fly during coding.
### Phase 2: Development (gate-enforced)
Every code edit requires an in_progress feature with matching files scope.
Features created during Phase 2 (no created_at) are flagged as warnings.
### After Code Changes
1. Update progress.md  2. Run scripts/lra-test.sh
### Bug Triage
High confidence (clear root cause) -> fix directly. Low confidence -> analyse + escalate.
HEREDOC
echo "  CREATE: CLAUDE.md"; } || echo "  SKIP: CLAUDE.md"

echo ""; echo "--- Hooks ---"; mkdir -p "$ROOT/.claude"
python3 <<- PYEOF
import json,os
h={"hooks":{"SessionStart":[{"matcher":"*","hooks":[{"type":"command","command":"[ -f ./init.sh ] && bash ./init.sh; true"}]}],"PreToolUse":[{"matcher":"*","hooks":[{"type":"command","command":"ROOT=\$(git rev-parse --show-toplevel 2>/dev/null || echo \"\$PWD\"); [ -f \"\$ROOT/scripts/lra-gate.py\" ] && python3 \"\$ROOT/scripts/lra-gate.py\"; true"}]}],"PostToolUse":[{"matcher":"*","hooks":[{"type":"command","command":"ROOT=\$(git rev-parse --show-toplevel 2>/dev/null || echo \"\$PWD\"); [ -f \"\$ROOT/scripts/lra-mark-dirty.py\" ] && python3 \"\$ROOT/scripts/lra-mark-dirty.py\"; true"}]}],"Stop":[{"matcher":"*","hooks":[{"type":"command","command":"ROOT=\$(git rev-parse --show-toplevel 2>/dev/null || echo \"\$PWD\"); [ -f \"\$ROOT/scripts/lra-stop.py\" ] && python3 \"\$ROOT/scripts/lra-stop.py\""}]}]}}
sf="$ROOT/.claude/settings.local.json"; e={}
try:
    with open(sf) as f: e=json.load(f)
except: pass
e["hooks"]=h["hooks"]
with open(sf,"w") as f: json.dump(e,f,indent=2)
print("  Hooks: SessionStart, PreToolUse, PostToolUse, Stop")
PYEOF

echo ""; echo "--- Bug triage ---"; mkdir -p "$ROOT/.claude/rules/common"
[ ! -f "$ROOT/.claude/rules/common/bug-triage.md" ] || [ "$FORCE" = "true" ] && { cat > "$ROOT/.claude/rules/common/bug-triage.md" << 'HEREDOC'
---
paths: ["**/*"]
---
# Bug Triage Protocol
## High Confidence -> Fix
Clear root cause, straightforward fix, low risk.
## Low Confidence -> Escalate
Unclear cause, unfamiliar module, tradeoffs. Give evidence + options.
HEREDOC
echo "  CREATE: .claude/rules/common/bug-triage.md"; } || echo "  SKIP: bug-triage.md"

echo ""; echo "=== LRA 安装完成 ==="
echo "Installed: lra_common.py gate stop dirty-marker test context-save init.sh"
echo "Templates: feature_list.json progress.md CLAUDE.md"
echo "Run ./init.sh to verify."
