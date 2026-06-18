---
paths:
  - "**/*"
---

# Bug Triage Protocol

When encountering a bug or problem, you MUST:

## Step 1: Output Confidence (MANDATORY)
```
【置信度: HIGH】or【置信度: LOW】+ one-line reason
```

## Step 2: Act on Decision Tree

**HIGH confidence** (all true):
- Root cause clearly identified
- Fix path straightforward, familiar module
- Low risk of side effects or regressions
- No architectural tradeoffs needed

→ Fix directly if scope covers the file. If no feature exists, create a bugfix entry explaining why it's high confidence, then fix.

**LOW confidence** (any true):
- Root cause unclear or only partially understood
- Unfamiliar modules or core algorithms involved
- Multiple valid approaches with non-obvious tradeoffs
- Fix could introduce subtle regressions

→ Give analysis + evidence + options. Escalate to user. Do NOT fix.

## Trigger Keywords
bug, problem, crash, wrong, broken, error, fail, 问题, 不工作, 报错, 不对, 为什么, 分析
