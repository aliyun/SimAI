# Harness LRA 快速上手

## 5 分钟体验

### 1. 初始化项目

在新项目目录执行安装后，得到：

```
project/
├── feature_list.json     ← 功能追踪
├── progress.md           ← 进度日志
├── .lra_version          ← 协议版本
├── scripts/
│   ├── lra-test.sh       ← 测试入口
│   ├── lra-gate.py       ← Hook 门禁
│   └── quick_status.py   ← 状态查看
└── .claude/
    └── settings.local.json ← Hook 配置
```

### 2. 创建第一个 Feature

编辑 `feature_list.json`，在 `active` 数组中添加：

```json
{
  "id": "F001",
  "type": "feature",
  "category": "core",
  "description": "Add user login endpoint",
  "status": "in_progress",
  "priority": "P0",
  "files": ["src/auth/login.py", "tests/test_login.py"],
  "verification_steps": ["pytest tests/test_login.py -v"],
  "passes": false,
  "confidence": "HIGH: standard REST endpoint, isolated scope",
  "created_at": "2026-06-04T00:00:00Z",
  "updated_at": "2026-06-04T00:00:00Z"
}
```

### 3. 开始编码

在 Claude Code 中直接说 "实现 F001 的 login endpoint"。Agent 会：

1. 自动读取 `feature_list.json` 了解范围和验证步骤
2. 写出置信度声明后开始编辑
3. 每次编辑后被 Hook 记录到 `.lra_dirty`
4. 结束时被 Stop Hook 强制要求跑测试

### 4. 完成 Feature

```bash
# 运行测试
bash scripts/lra-test.sh
# 检查状态
python3 scripts/quick_status.py
# 查看进度
cat progress.md
```

所有测试通过后，将 feature 状态改为 `done`，`passes` 设为 `true`。

### 5. 会话恢复

新开一个 Claude Code 会话，直接说 "继续开发"。Agent 自动：

- 读 `feature_list.json` → 知道 F001 done, F002 in_progress
- 读 `progress.md` → 知道上次做到哪
- 检查 `.lra_dirty` → 确认无待测试变更
- 继续推进 F002

## 日常循环

```
新会话 → Agent 读取状态 → 继续开发 → Hook 记录变更
  → 跑测试 → 更新进度 → 标记完成 → 下一个 feature
```
