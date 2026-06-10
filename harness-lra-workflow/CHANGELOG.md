# Changelog

## 1.0.0 (2026-06-04)

Initial release.

### Features
- Feature tracking via `feature_list.json` with typed entries (feature/bugfix/test/refactor)
- Atomic task scoping with file-level gating
- Confidence-gated editing (HIGH/LOW decision tree)
- Append-only `progress.md` with integrity hash
- Dirty file tracking via `.lra_dirty` (auto-cleared on test pass)
- Hook gates: PreToolUse (scope/confidence/verification), PostToolUse (dirty record), Stop (dirty block)
- `quick_status.py` for instant project status
- `lra-gate.py` for hook integration and health checks
- One-click `install.sh` for project setup
- Reference docs: install guide, quick start, advanced (version mgmt + logging)

### Compatibility
- Claude Code CLI with hooks support
- Python 3.8+
- Any git-tracked project
