#!/usr/bin/env python3
"""LRA shared utilities — single source of truth for all LRA components."""
import json, os, re, subprocess, sys


# ═══════════════════════════════════════════
# Path resolution
# ═══════════════════════════════════════════

def find_root():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.getcwd()


# ═══════════════════════════════════════════
# File allowlist (not tracked for dirty/scope)
# ═══════════════════════════════════════════

ALLOWED_PREFIXES = (
    "feature_list.json", "progress.md", "init.sh", "CLAUDE.md",
    ".claude/", "docs/", "scripts/lra-",
)


def is_allowed(path, root=None):
    if not path:
        return True
    if root is None:
        root = find_root()
    rel = os.path.relpath(path, root) if path.startswith("/") else path
    if rel.startswith(".."):
        return True
    for a in ALLOWED_PREFIXES:
        if rel == a or rel.startswith(a):
            return True
    return rel.endswith(".md")


# ═══════════════════════════════════════════
# Unified feature_list.json parser
#  Supports: flat list, dict {active,done,pending_*}, legacy {features}
# ═══════════════════════════════════════════

def load_features(fl_path=None):
    if fl_path is None:
        fl_path = os.path.join(find_root(), "feature_list.json")
    try:
        with open(fl_path) as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict) and f.get("id")]

    if "features" not in data:
        all_items = {}
        for key, default_status in [
            ("done", "done"),
            ("pending_bugs", "pending"),
            ("pending_tests", "pending"),
            ("active", "in_progress"),
        ]:
            for item in data.get(key, []):
                if isinstance(item, dict):
                    fid = item.get("id")
                elif isinstance(item, str):
                    fid, item = item, {"id": fid}
                else:
                    continue
                m = all_items.get(fid, {})
                m.update(item)
                # Respect explicit status override, otherwise use array default
                if not item.get("status"):
                    m["status"] = default_status
                all_items[fid] = m
        return list(all_items.values())

    return [f for f in data.get("features", [])
            if isinstance(f, dict) and f.get("id")]


def get_in_progress():
    """Return (feature_id, feature_dict, files_scope) or (None, None, [])."""
    features = load_features()
    in_prog = [f for f in features if f.get("status") == "in_progress"]
    if not in_prog:
        return None, None, []
    current = in_prog[0]
    return current["id"], current, current.get("files", [])


# ═══════════════════════════════════════════
# Bash command analysis
# ═══════════════════════════════════════════

_MODIFY_RE = re.compile(
    r"\bsed\b\s+.*-i\b|"
    r"\brm\b\s+(-rf?\s+)?\S+|"
    r"\bmv\b\s+\S+\s+\S+|"
    r"\bcp\b\s+.*\s+\S+|"
    r">\s*\S+|"
    r"\btee\b\s+\S+|"
    r"\bdd\b\s+.*\bof=\S+|"
    r"\btouch\b\s+\S+|"
    r"\bmkdir\b\s+\S+"
)


def is_modifying_bash(command):
    if not command or not command.strip():
        return False
    clean = re.sub(r"\d?>\s*/dev/null", "", command)
    clean = re.sub(r"\d?>\s*&1", "", clean)
    return bool(_MODIFY_RE.search(clean))


# ═══════════════════════════════════════════
# Timestamp parser
# ═══════════════════════════════════════════

def parse_ts(val):
    if not val:
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).replace("Z", "+00:00")
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", s)
        if m:
            from calendar import timegm
            from datetime import datetime
            dt = datetime(*map(int, m.groups()[:6]))
            return timegm(dt.utctimetuple()) + (dt.microsecond / 1e6 if '.' in s else 0)
    except (ValueError, TypeError):
        pass
    return 0


# ═══════════════════════════════════════════
# Integrity hashes (done features + progress.md)
# ═══════════════════════════════════════════

def compute_done_hash(features):
    import hashlib
    done = sorted(
        [f for f in features if f.get("status") == "done"],
        key=lambda x: x.get("id", ""),
    )
    canonical = json.dumps([{
        "id": f["id"],
        "status": f.get("status"),
        "description": f.get("description", ""),
        "files": sorted(f.get("files", [])),
        "verification_steps": f.get("verification_steps", []),
        "passes": f.get("passes"),
    } for f in done], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16], len(done)


def compute_progress_hash(progress_path):
    import hashlib
    if not os.path.exists(progress_path):
        return "", 0
    with open(progress_path) as f:
        plines = [l for l in f if l.strip().startswith("| 20")]
    count = len(plines)
    h = hashlib.sha256("".join(plines).encode()).hexdigest()[:16]
    return h, count
