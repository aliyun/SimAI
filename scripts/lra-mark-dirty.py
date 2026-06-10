#!/usr/bin/env python3
"""LRA PostToolUse hook: mark dirty when code files are edited + auto-inject created_at."""
import fnmatch, json, os, sys
from datetime import datetime, timezone

# Add scripts/ to path for lra_common import
ROOT = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lra_common import is_allowed, get_in_progress, find_root, load_features, parse_ts

DIRTY_FILE = os.path.join(ROOT, ".lra_dirty")
FL = os.path.join(ROOT, "feature_list.json")


def ensure_created_at(feature_id):
    """If a feature in 'active' lacks created_at, inject it now.
    Flags features created hastily during Phase 2 instead of planned in Phase 1."""
    try:
        with open(FL) as f:
            data = json.load(f)
    except Exception:
        return

    modified = False
    for item in data.get("active", []):
        fid = item.get("id") if isinstance(item, dict) else item
        if fid == feature_id and isinstance(item, dict) and not item.get("created_at"):
            item["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            modified = True
    for arr_name in ("pending", "pending_bugs", "pending_tests"):
        for item in data.get(arr_name, []):
            fid = item.get("id") if isinstance(item, dict) else item
            if fid == feature_id and isinstance(item, dict) and not item.get("created_at"):
                item["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                modified = True

    if modified:
        with open(FL, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[LRA] Injected created_at for {feature_id}", file=sys.stderr)


def check_phase_violation():
    """After feature_list.json was edited, check for Phase 2 violations:
    new features without created_at while development is active (dirty exists)."""
    if not os.path.exists(DIRTY_FILE):
        return  # No active development — Phase 1 ok
    try:
        features = load_features(FL)
    except Exception:
        return
    new_without_ca = [f for f in features
                      if f.get("status") in ("in_progress", "pending")
                      and not f.get("created_at")]
    if new_without_ca:
        ids = ", ".join(f["id"] for f in new_without_ca[:5])
        print(
            f"[LRA] PHASE VIOLATION: {len(new_without_ca)} feature(s) ({ids}) "
            f"added during active development.\n"
            f"       Features MUST be created in Phase 1 (需求澄清), "
            f"not during Phase 2 (开发执行).\n"
            f"       These features lack created_at and will be flagged by stop hook.",
            file=sys.stderr,
        )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    path = (payload.get("tool_input", {}) or {}).get("file_path") or ""

    # Surveillance: detect feature_list.json edits during Phase 2
    rel = os.path.relpath(path, ROOT) if path.startswith("/") else path
    if rel == "feature_list.json":
        check_phase_violation()

    if is_allowed(path, ROOT):
        return 0

    fid, feature, scope = get_in_progress()
    if not fid:
        return 0

    # Auto-inject created_at on first edit of a feature
    ensure_created_at(fid)

    rel = os.path.relpath(path, ROOT) if path.startswith("/") else path
    if scope and not any(fnmatch.fnmatch(rel, pat) for pat in scope):
        return 0

    dirty = {}
    try:
        with open(DIRTY_FILE) as f:
            dirty = json.load(f)
    except Exception:
        pass

    if dirty.get("feature") != fid:
        dirty = {"feature": fid, "files": [],
                 "since": datetime.now(timezone.utc).isoformat()}

    if rel not in dirty["files"]:
        dirty["files"].append(rel)

    with open(DIRTY_FILE, "w") as f:
        json.dump(dirty, f, indent=2)

    print(f"[LRA] dirty: {len(dirty['files'])} file(s) for {fid}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
