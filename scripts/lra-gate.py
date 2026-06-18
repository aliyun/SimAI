#!/usr/bin/env python3
"""LRA PreToolUse gate: enforce feature tracking, TDD, and test-before-switch."""
import fnmatch, json, os, sys

# ── Bootstrap: add scripts/ to path for lra_common ──
ROOT = os.environ.get("CLAUDE_PROJECT_DIR", "")
if not ROOT:
    try:
        import subprocess
        ROOT = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lra_common import (
    find_root, is_allowed, load_features, is_modifying_bash, parse_ts
)

FL = os.path.join(ROOT, "feature_list.json")
DIRTY_FILE = os.path.join(ROOT, ".lra_dirty")


def load_dirty():
    try:
        with open(DIRTY_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def is_test_file(path):
    rel = os.path.relpath(path, ROOT) if path.startswith("/") else path
    return (rel.startswith("tests/") or rel.startswith("test/")
            or ".test." in rel or "_test." in rel
            or rel.startswith("__tests__/"))


def check_phase_violation():
    """If dirty files exist (Phase 2 is active), check for features without created_at."""
    dirty = load_dirty()
    if not dirty:
        return None  # No active development — Phase 1 is fine
    features = load_features()
    recent = [f for f in features
              if f.get("status") in ("in_progress", "pending")
              and not f.get("created_at")]
    if recent:
        ids = ", ".join(f["id"] for f in recent)
        return (f"PHASE VIOLATION: {len(recent)} feature(s) ({ids}) lack created_at.\n"
                "  Features should be created during 需求澄清 (Phase 1), not during development.\n"
                "  Action: move these back to planning, or ensure they were created before coding.")
    return None


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    ti = payload.get("tool_input", {}) or {}
    path = ti.get("file_path") or ti.get("path") or ""

    # ── Bash command gate ──
    command = ti.get("command", "")
    if command:
        if not is_modifying_bash(command):
            return 0
        import shlex
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        project_paths = [t for t in tokens
                         if t.startswith(ROOT)
                         or (t.startswith("/") and ROOT in t)]
        if not project_paths:
            return 0
        path = project_paths[0]

    if not path or is_allowed(path, ROOT):
        return 0

    # ── Feature checks ──
    features = load_features()
    in_prog = [f for f in features if f.get("status") == "in_progress"]
    done_count = sum(1 for f in features if f.get("status") == "done")
    pending = [f for f in features if f.get("status") == "pending"]

    # Rule 1: must have an in_progress feature
    if not in_prog:
        if not features:
            print(
                "BLOCKED — 需求澄清 (Phase 1): feature_list.json is empty.\n"
                f"  Target: {path}\n"
                "  Action: discuss requirements, then create features.",
                file=sys.stderr,
            )
        elif not pending:
            print(
                f"BLOCKED: All {done_count} features done. No pending work.\n"
                f"  Target: {path}\n"
                "  Action: discuss new requirements first.",
                file=sys.stderr,
            )
        else:
            p0 = [f for f in pending if f.get("priority") == "P0"]
            hint = ""
            if p0:
                hint = f"\n  Top P0: {', '.join(f['id'] for f in p0[:3])}\n  Pick one and set status='in_progress'."
            print(
                f"BLOCKED: {len(pending)} feature(s) pending, none in_progress.\n"
                f"  Target: {path}"
                + hint,
                file=sys.stderr,
            )
        return 2

    rel = os.path.relpath(path, ROOT) if path.startswith("/") else path

    # Find first in_progress feature that owns this file
    current = None
    for f in in_prog:
        scope = f.get("files")
        if scope and any(fnmatch.fnmatch(rel, pat) for pat in scope):
            current = f
            break

    if current is None:
        # No in_progress feature owns this file — block unless allowed
        current = in_prog[0]

    # Rule 2: files scope required
    scope = current.get("files")
    if not scope:
        print(
            f"BLOCKED (no files scope): Feature '{current['id']}' has no 'files' list.\n"
            f"  Target: {rel}\n"
            f"  Action: add files to '{current['id']}'.files, or create a new feature.",
            file=sys.stderr,
        )
        return 2

    if not any(fnmatch.fnmatch(rel, pat) for pat in scope):
        print(
            f"BLOCKED (wrong feature): '{current['id']}' does not own: {rel}\n"
            f"  Allowed: {', '.join(scope)}\n"
            f"  Action: update '{current['id']}'.files or create a new feature.",
            file=sys.stderr,
        )
        return 2

    # Rule 3: dirty switch — must run tests before switching features
    dirty = load_dirty()
    dirty_feature = dirty.get("feature") if dirty else None
    if dirty_feature and dirty_feature != "unknown" and dirty_feature != current["id"]:
        print(
            f"BLOCKED (untested changes): '{dirty_feature}' has dirty files.\n"
            f"  Action: run scripts/lra-test.sh, then switch to '{current['id']}'.",
            file=sys.stderr,
        )
        return 2

    # Rule 4: in_progress feature MUST have created_at (Phase 1 proof)
    if not current.get("created_at"):
        print(
            f"BLOCKED: Feature '{current['id']}' has no created_at.\n"
            "  Features MUST be created in Phase 1 (需求澄清), not during coding.\n"
            f"  Action: add 'created_at' to '{current['id']}' in feature_list.json\n"
            "    with the timestamp from when this feature was originally planned.",
            file=sys.stderr,
        )
        return 2

    # Rule 5: 必须输出置信度才能编辑
    conf = current.get("confidence", "")
    if not conf:
        print(
            f"【LRA 阻断】Feature '{current['id']}' 没有置信度评估。\n"
            "  编辑代码前必须先评估置信度：\n"
            "    【置信度: HIGH】+ 理由 → 可以直接修\n"
            "    【置信度: LOW】+ 理由 → 交给用户决策\n"
            f"  操作: 在 feature_list.json 中为 '{current['id']}' 添加 'confidence' 字段",
            file=sys.stderr,
        )
        return 2
    if conf.startswith("LOW"):
        print(
            f"【LRA 阻断】Feature '{current['id']}' 置信度为 LOW: {conf}\n"
            "  LOW 置信度的问题必须交给用户决策，不能直接修改。\n"
            "  操作: 给出分析+证据+选项，让用户决定。",
            file=sys.stderr,
        )
        return 2
    if not conf.startswith("HIGH:"):
        print(
            f"【LRA 阻断】Feature '{current['id']}' 置信度格式错误: {conf}\n"
            "  必须是 'HIGH: <理由>' 或 'LOW: <理由>'。\n"
            f"  操作: 修正 'confidence' 字段。",
            file=sys.stderr,
        )
        return 2

    # Rule 6: type=feature requires verification_steps
    if current.get("type") == "feature" and not current.get("verification_steps"):
        print(
            f"BLOCKED: Feature '{current['id']}' has no verification_steps.",
            file=sys.stderr,
        )
        return 2

    # Surface: always print confidence so user can see it
    print(
        f"[LRA] 置信度: {conf} | feature={current['id']}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
