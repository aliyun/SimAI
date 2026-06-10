# Harness LRA 高级特性

## 版本管理

### 协议版本

`.lra_version` 文件记录当前协议版本：

```json
{
  "protocol": "1.0.0",
  "installed_at": "2026-06-04T00:00:00Z",
  "updated_at": "2026-06-04T00:00:00Z",
  "compatible": ["1.0.x"]
}
```

升级时向后兼容：`compatible` 字段声明支持的版本范围。`lra-gate.py --check-version` 在 SessionStart 时校验版本兼容性。

### Feature 变更日志

每次 feature 状态变更自动追加到 `.lra_changelog`：

```json
{"ts":"2026-06-04T10:00:00Z","feature":"F001","from":"in_progress","to":"done","trigger":"test_pass"}
```

通过 `quick_status.py --history F001` 查看完整变更链。

### 多分支协作

每个 git 分支独立维护 `feature_list.json`。合并时：

```bash
python3 scripts/lra-gate.py --merge feature_list.json main_feature_list.json
```

自动检测冲突、去重、合并 `done` 条目。

## 调用日志

### Session 日志

每个会话自动记录到 `.lra_sessions/`：

```json
{
  "session_id": "uuid",
  "started": "ISO timestamp",
  "ended": "ISO timestamp",
  "features_touched": ["F001", "F002"],
  "files_edited": 12,
  "tests_run": 98,
  "tests_passed": 98,
  "dirty_cleared": true
}
```

### 操作审计

每次 Hook 拦截记录到 `.lra_audit.jsonl`：

```json
{"ts":"...","hook":"PreToolUse","action":"block","reason":"file not in scope","feature":"F089","file":"src/other.py"}
{"ts":"...","hook":"Stop","action":"block","reason":".lra_dirty non-empty","feature":"F090","dirty_files":["src/auth.py"]}
```

### 统计面板

```bash
python3 scripts/quick_status.py --stats
```

输出：
- 总 feature 数、完成率
- 平均开发周期（从 in_progress 到 done）
- Hook 拦截统计（阻止了多少次越权编辑）
- 测试通过率趋势

## 运营运维

### 健康检查

```bash
python3 scripts/lra-gate.py --health
```

检查项：
- [ ] `feature_list.json` 格式有效
- [ ] `progress.md` 完整性 hash 匹配
- [ ] `.lra_dirty` 状态一致
- [ ] Hook 配置语法正确
- [ ] 测试脚本可执行
- [ ] 所有 done feature 的 `passes` 为 true

### 自动修复

```bash
python3 scripts/lra-gate.py --repair
```

自动修复常见问题：
- 重新计算 integrity hash
- 清理孤立的 `.lra_dirty` 条目
- 补充缺失的 `updated_at` 时间戳
- 修正 `summary` 计数

### 备份恢复

```bash
tar czf lra-backup-$(date +%Y%m%d).tar.gz feature_list.json progress.md .lra_done_hash .lra_sessions/
```

恢复时校验 hash 确保数据未被篡改。

## 扩展接口

### 自定义 Hook 规则

在 `feature_list.json` 顶层添加：

```json
{
  "rules": {
    "require_review_for": ["src/database/**", "src/auth/**"],
    "auto_approve_patterns": ["tests/**", "docs/**"],
    "max_files_per_feature": 5
  }
}
```

### 集成外部工具

`lra-gate.py` 支持 `--on-pass` 和 `--on-fail` 回调：

```bash
python3 scripts/lra-gate.py --on-pass "curl -X POST webhook.example.com/deploy"
```
