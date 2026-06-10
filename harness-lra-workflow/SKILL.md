---
name: harness-lra-workflow
description: Harness 长周期自主开发工作流 — Agent + Hook + 文档 + 测试闭环。通过原子化任务拆分、三阶段门禁拦截、不可变进度日志和置信度分级决策，让 AI 跨越多个会话持续自主推进复杂项目。直接说“新会话继续开发”即可自动恢复。
---

# Harness 长周期自主开发工作流

Harness = Agent + Hook + 文档 + 测试闭环，让 AI 在长周期项目中持续自主编码，解决跨会话记忆断裂、重复开发、幻觉漂移等核心痛点。

## Core Concepts

**Feature list** (`feature_list.json`): Structured tracking of all work items with IDs, types, statuses, verification steps, and file scopes. Only code within an in-progress feature's scope can be edited.

**Progress log** (`progress.md`): Append-only human-readable session history. Previous records are immutable — append new entries only.

**Dirty file tracking** (`.lra_dirty`): Auto-managed file that tracks uncommitted changes per feature. Cleared when `lra-test.sh` passes.

**Confidence gating**: Before editing code, must declare `【置信度: HIGH】` or `【置信度: LOW】` with reasoning. HIGH allows direct fix; LOW requires user decision.

**Integrity hashes** (`.lra_done_hash`): SHA-256 hashes of completed features and progress log to detect tampering.

## File Structure

```
project/
├── feature_list.json     # Structured feature tracking
├── progress.md            # Append-only progress log
├── .lra_dirty             # Uncommitted change tracking (auto)
├── .lra_done_hash         # Integrity hashes (auto)
├── scripts/
│   ├── lra-test.sh        # Full test suite runner
│   └── lra-gate.py        # Pre/PostToolUse gate
└── .claude/
    └── settings.local.json # Hook configuration
```

## Feature Schema

Each feature in `feature_list.json`:

```json
{
  "id": "F001",
  "type": "feature|bugfix|test|refactor",
  "category": "subsystem-tag",
  "description": "What this does",
  "status": "pending|in_progress|done",
  "priority": "P0|P1|P2|P3",
  "files": ["path/to/file.py", "path/to/dir/**"],
  "verification_steps": ["step 1", "step 2"],
  "passes": false,
  "confidence": "HIGH: reason|LOW: reason",
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp"
}
```

## Workflow

### Session Start
1. `git log --oneline -10`
2. Read `progress.md` and `feature_list.json`
3. Continue in-progress features

### Making Changes
1. Declare confidence: `【置信度: HIGH】one-line reason` or `【置信度: LOW】one-line reason`
2. Feature must be `in_progress` and own the target file
3. After changes: run `lra-test.sh`
4. Update `progress.md` (append only)

### Feature Lifecycle
```
pending → in_progress → done
              ↑            │
              └── reopen ──┘
```

- `pending`: Not yet started
- `in_progress`: Currently working (only ONE at a time per file scope)
- `done`: Verified and passing

### Bug Triage
```
【置信度: HIGH】→ 直接修 (scope covers file, low risk)
【置信度: LOW】 → 给分析+证据+选项，交给用户决策
```

## Hook Gates

| Hook | Rule |
|------|------|
| PreToolUse | Edit non-whitelist file → feature must be `in_progress` |
| PreToolUse | `.lra_dirty` non-empty + feature mismatch → block (run tests first) |
| PreToolUse | Type=`feature` without verification_steps → block |
| PostToolUse | Edit non-whitelist file → write to `.lra_dirty` |
| Stop | `.lra_dirty` non-empty → block |
| Stop | Done but passes=false → block |

## Test Strategy

- Backend: `python3 -m pytest tests/ -v`
- Frontend: `npx tsc --noEmit`
- E2E: Playwright or similar
- All: `lra-test.sh`

Minimum coverage: 80%. Tests must pass before marking feature done.

## Resume Protocol

When resuming after interruption:
1. Read `CLAUDE.md` and project memory
2. Read `feature_list.json` — note in_progress items
3. Read `progress.md` — last few entries for context
4. Check `.lra_dirty` — files needing test before continue
5. Run `lra-test.sh` to clear dirty state

## Examples

```
# Session start after interruption
$ cat .lra_dirty
{"feature": "F005", "files": ["src/auth.py"]}
$ bash scripts/lra-test.sh
ALL TESTS PASSED — .lra_dirty cleared

# Declaring confidence
【置信度: HIGH】确定性字段重命名，3个文件，无逻辑变更

# Appending progress
## 2026-06-04 | F005: auth refactor — tests pass
```

## Guidelines

- Never modify `feature_list.json` format or schema
- Never modify historical `progress.md` entries — append only
- Never manually delete `.lra_dirty` — run tests instead
- Never skip hooks with `--no-verify`
- One feature `in_progress` per file scope at a time
- Feature ID format: `F` + 3-digit zero-padded number
