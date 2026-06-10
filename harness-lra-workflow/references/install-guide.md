# Harness LRA 安装指南

## 前置条件

- Claude Code CLI 已安装
- 项目已初始化 git 仓库

## 一键安装

```bash
bash scripts/install.sh
```

安装脚本自动完成：
1. 创建 `feature_list.json` 模板
2. 创建 `progress.md` 模板
3. 生成 `scripts/lra-test.sh` 测试框架
4. 写入 `.claude/settings.local.json` Hook 配置
5. 生成 `.lra_version` 版本文件

## 手动安装

### 1. 初始化数据文件

```bash
cp references/feature_list.json ./feature_list.json
cp references/progress.md ./progress.md
```

编辑 `feature_list.json`，将 `"project"` 改为实际项目名。

### 2. 部署 Hook 脚本

```bash
cp scripts/lra-gate.py ./scripts/lra-gate.py
cp scripts/quick_status.py ./scripts/quick_status.py
chmod +x scripts/lra-gate.py scripts/quick_status.py
```

### 3. 创建测试入口

```bash
cat > scripts/lra-test.sh << 'EOF'
#!/bin/bash
set -e
# Backend tests
python3 -m pytest tests/ -v
# Frontend type check (如适用)
npx tsc --noEmit 2>/dev/null || true
# LRA integrity update
python3 scripts/lra-gate.py --update
echo "ALL TESTS PASSED"
EOF
chmod +x scripts/lra-test.sh
```

### 4. 配置 Hook

在 `.claude/settings.local.json` 中添加：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python3 scripts/lra-gate.py pre"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "python3 scripts/lra-gate.py post ${FILE_PATH}"
      }
    ],
    "Stop": [
      {
        "command": "python3 scripts/lra-gate.py stop"
      }
    ]
  }
}
```

## 验证安装

```bash
bash scripts/lra-test.sh
python3 scripts/quick_status.py
```

应显示项目状态且所有测试通过。
