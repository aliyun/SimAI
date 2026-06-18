#!/bin/bash
# LRA (Long-Running Agent) — 一键卸载脚本
# 清理 install-lra.sh 创建的所有文件和配置。
#
# Usage: bash uninstall-lra.sh [--all]
#   --all: 同时删除 feature_list.json / progress.md

set -e

ALL=false
if [ "${1:-}" = "--all" ]; then
  ALL=true
  shift
fi

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
echo "=== LRA 一键卸载 ==="
echo "项目根目录: $ROOT"
echo ""

removed=0
skipped=0

remove() {
  local target="$1"
  local desc="${2:-$1}"
  if [ -e "$target" ]; then
    rm -rf "$target"
    echo "  REMOVED: $desc"
    ((removed++)) || true
  else
    echo "  absent: $desc"
    ((skipped++)) || true
  fi
}

# ── 1. Core scripts ──
echo "--- LRA 脚本 ---"
remove "$ROOT/scripts/lra_common.py"        "scripts/lra_common.py"
remove "$ROOT/scripts/lra-gate.py"          "scripts/lra-gate.py"
remove "$ROOT/scripts/lra-stop.py"           "scripts/lra-stop.py"
remove "$ROOT/scripts/lra-mark-dirty.py"     "scripts/lra-mark-dirty.py"
remove "$ROOT/scripts/lra-test.sh"           "scripts/lra-test.sh"
remove "$ROOT/scripts/lra-context-save.sh"   "scripts/lra-context-save.sh"
remove "$ROOT/scripts/lra-loop.sh"           "scripts/lra-loop.sh (legacy)"
if [ -d "$ROOT/scripts" ] && [ -z "$(ls -A "$ROOT/scripts" 2>/dev/null)" ]; then
  rmdir "$ROOT/scripts"
  echo "  REMOVED: scripts/ (empty dir)"
fi

# ── 2. init.sh ──
echo ""
echo "--- init.sh ---"
remove "$ROOT/init.sh" "init.sh"

# ── 3. Runtime state ──
echo ""
echo "--- Runtime state ---"
remove "$ROOT/.lra_dirty"                ".lra_dirty"
remove "$ROOT/.lra_last_test"            ".lra_last_test"
remove "$ROOT/.lra_context_warning"      ".lra_context_warning"
remove "$ROOT/.lra_context_warning.last" ".lra_context_warning.last"
remove "$ROOT/.lra_done_hash"            ".lra_done_hash"
remove "$ROOT/.lra_loop.log"            ".lra_loop.log (legacy)"

# ── 4. Data files ──
echo ""
echo "--- Data files ---"
if [ "$ALL" = "true" ]; then
  remove "$ROOT/feature_list.json" "feature_list.json"
  remove "$ROOT/progress.md"       "progress.md"
else
  echo "  KEPT: feature_list.json  (use --all to remove)"
  echo "  KEPT: progress.md        (use --all to remove)"
  skipped=$((skipped + 2))
fi

# ── 5. Hook configuration ──
echo ""
echo "--- Hook configuration ---"

SETTINGS="$ROOT/.claude/settings.local.json"
if [ -f "$SETTINGS" ]; then
  export SETTINGS_FILE="$SETTINGS"
  python3 << 'PYEOF'
import json, os

sf = os.environ["SETTINGS_FILE"]
with open(sf) as f:
    data = json.load(f)

hooks = data.get("hooks", {})
lra_cmds = [
    "lra-gate.py",
    "lra-stop.py",
    "lra-mark-dirty.py",
    "./init.sh"
]

removed_count = 0
for event in list(hooks.keys()):
    configs = hooks[event]
    new_configs = []
    for cfg in configs:
        h_list = cfg.get("hooks", [])
        new_hooks = [h for h in h_list
                     if not any(c in h.get("command", "") for c in lra_cmds)]
        if len(new_hooks) < len(h_list):
            removed_count += len(h_list) - len(new_hooks)
        if new_hooks:
            cfg["hooks"] = new_hooks
            new_configs.append(cfg)
        else:
            removed_count += 1
    if new_configs:
        hooks[event] = new_configs
    else:
        del hooks[event]

if hooks:
    data["hooks"] = hooks
else:
    data.pop("hooks", None)

with open(sf, "w") as f:
    json.dump(data, f, indent=2)
print(f"  Removed {removed_count} LRA hook entries")
PYEOF
  echo "  UPDATED: .claude/settings.local.json"
else
  echo "  absent: .claude/settings.local.json"
fi

# ── 6. CLAUDE.md LRA section ──
echo ""
echo "--- CLAUDE.md ---"
CLAUDE_MD="$ROOT/CLAUDE.md"
if [ -f "$CLAUDE_MD" ]; then
  if grep -qE "LRA (强制执行工作流|Workflow|Workflow)" "$CLAUDE_MD" 2>/dev/null; then
    if [ "$ALL" = "true" ]; then
      remove "$CLAUDE_MD" "CLAUDE.md (--all)"
    else
      export CLAUDE_FILE="$CLAUDE_MD"
      python3 << 'PYEOF'
import os
cf = os.environ["CLAUDE_FILE"]
with open(cf) as f:
    lines = f.readlines()

start = end = -1
for i, line in enumerate(lines):
    if "LRA" in line and ("强制执行工作流" in line or "Workflow" in line):
        for j in range(i, -1, -1):
            if lines[j].strip().startswith("## "):
                start = j; break
        if start < 0: start = i
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("## ") and "LRA" not in lines[j]:
                end = j; break
        if end < 0: end = len(lines)
        break

if start >= 0:
    del lines[start:end]
    # Remove trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append("")
    with open(cf, "w") as f:
        f.writelines(lines)
    print("  STRIPPED: LRA section from CLAUDE.md")
else:
    print("  absent: LRA section")
PYEOF
    fi
  else
    echo "  absent: LRA section in CLAUDE.md"
  fi
else
  echo "  absent: CLAUDE.md"
fi

# ── 7. Global rule ──
echo ""
echo "--- Bug-triage rule ---"
RULE="$ROOT/.claude/rules/common/bug-triage.md"
if [ -f "$RULE" ]; then
  if [ "$ALL" = "true" ]; then
    remove "$RULE" ".claude/rules/common/bug-triage.md"
  else
    echo "  KEPT: $RULE (use --all to remove)"
  fi
else
  echo "  absent"
fi

echo ""
echo "=========================================="
echo "  LRA 卸载完成"
echo "  Removed: $removed  Kept: $skipped"
echo "=========================================="
