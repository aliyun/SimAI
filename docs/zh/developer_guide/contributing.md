# 贡献指南

感谢您对 SimAI 项目的关注！本指南介绍完整的开发工作流。

> **完整版**：详见项目根目录下的 [CONTRIBUTING.md](../../../CONTRIBUTING.md)。

---

## 贡献方式

1. **新功能** — 添加模型支持、并行策略、调度策略
2. **Bug 修复** — 修复仿真不准确、崩溃或结果错误
3. **性能优化** — 提升仿真速度、内存使用或可扩展性
4. **文档** — 改进教程、添加示例、修正错误
5. **基准测试与验证** — 添加对比真实硬件的验证结果
6. **问题报告** — 报告 Bug、请求功能或分享反馈

---

## 开发工作流

### 步骤 1：Fork 和克隆

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/SimAI.git
cd SimAI
git remote add upstream https://github.com/aliyun/SimAI.git
```

### 步骤 2：创建功能分支

```bash
git fetch upstream
git checkout -b feature/your-feature-name upstream/master

# 分支命名规范：
#   feature/xxx  — 新功能
#   fix/xxx      — Bug 修复
#   docs/xxx     — 文档
#   perf/xxx     — 性能优化
#   refactor/xxx — 代码重构
```

### 步骤 3：开发和测试

```bash
# C++ 修改需重新编译：
bash scripts/build.sh -c analytical  # 或 ns3

# Python 修改：
python -c "from aicb import ..."
```

### 步骤 4：提交

```bash
git add -A
git commit -m "feat(aicb): add Llama-4 model workload generation"
```

### 步骤 5：推送并创建 PR

```bash
git push origin feature/your-feature-name
# 然后在 GitHub 上创建 Pull Request
```

---

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>
```

### 类型

| 类型 | 说明 |
|------|-------------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 仅文档修改 |
| `refactor` | 代码重构 |
| `test` | 添加或更新测试 |
| `perf` | 性能优化 |
| `chore` | 构建流程、工具 |

### 范围

`aicb`、`vidur`、`astra-sim`、`ns3`、`simccl`、`docs`、`docker`、`scripts`

### 示例

```
feat(aicb): add DeepSeek-V3 inference workload generation
fix(astra-sim): correct AllReduce latency calculation for ring algorithm
docs: update build instructions for NS-3 mode
perf(vidur): reduce memory allocation in request scheduler
```

---

## 代码风格

### Python

- **格式化**: [black](https://github.com/psf/black)（默认设置）
- **Import 排序**: [isort](https://pycqa.github.io/isort/)
- **Linter**: [flake8](https://flake8.pycqa.org/)
- **最大行宽**: 120 字符

```bash
black --line-length 120 your_file.py
isort your_file.py
flake8 your_file.py --max-line-length 120
```

### C++

- 遵循 `astra-sim-alibabacloud/` 中现有代码风格
- 4 空格缩进
- 函数和变量使用 `snake_case`
- 非显而易见的逻辑需添加注释

### 通用规则

- 注释使用**英文**编写
- 所有新函数/类应有文档字符串或头部注释
- 避免硬编码路径；使用相对路径或配置变量
- 每个 PR 只包含一个功能/修复

---

## 子模块操作

SimAI 使用 Git submodule，关键要点：

| 子模块 | 仓库 | 语言 |
|-----------|------------|----------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | Python |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | Python |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | C++ |
| `astra-sim-alibabacloud` | 项目内 | C++ |
| `vidur-alibabacloud` | 项目内 | Python |

### 跨子模块修改

如果您的贡献涉及多个子模块：

1. 在每个子模块中分别进行修改并提交
2. 更新父仓库指向新的子模块 commit
3. 为有独立远程仓库的子模块创建单独的 PR
4. 在 PR 描述中引用相关 PR

---

## Pull Request 指南

### PR 标题

使用与 commit message 相同的格式：`<type>(<scope>): <description>`

### PR 检查清单

- [ ] 代码编译无错误
- [ ] 现有仿真结果不变（无精度退化）
- [ ] 新代码有适当注释
- [ ] 新功能添加了测试
- [ ] 必要时更新了文档

---

## 提交前质量检查

```bash
# 1. C++ 编译检查
bash scripts/build.sh -c analytical
bash scripts/build.sh -c ns3

# 2. Python lint 检查
black --check --line-length 120 your_changed_files.py
flake8 your_changed_files.py --max-line-length 120

# 3. 基本仿真测试
cd bin && ./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml

# 4. 子模块状态检查
git submodule status
```

---

## 验收标准

| 标准 | 要求 |
|-----------|-------------|
| **编译** | 编译无错误 |
| **精度** | 不降低现有仿真精度 |
| **测试** | 关键代码路径有覆盖 |
| **文档** | 新功能有注释/文档更新 |
| **风格** | 遵循代码风格规范 |
| **范围** | 修改集中且解释清晰 |

---

## 审核时间线

1. **初审**：3-5 个工作日
2. **反馈**：建设性评论和可操作建议
3. **迭代**：处理反馈并更新 PR
4. **合并**：批准的 PR 合入主分支

---

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/aliyun/SimAI/issues)
- **讨论**: 创建 Issue 并以 "Question:" 为前缀
- **文档**: 参见 [Tutorial](../../../docs/Tutorial.md)
