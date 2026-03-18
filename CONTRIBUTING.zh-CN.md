# SimAI 贡献指南

[English Version](CONTRIBUTING.md)

感谢你对 SimAI 项目的关注！本指南将帮助你了解如何贡献代码、文档和想法。

---

## 项目愿景与目标

**愿景**：打造业界首个全栈、高精度的 AI 大规模推理与训练仿真器。

**目标**：提供端到端的 AI 训练/推理过程建模与仿真——涵盖框架层、集合通信层、网络层等——使研究人员无需真实硬件即可分析性能、评估优化方案、探索基础设施设计。

**当前进展**：SimAI 1.5 已发布（2025年12月），支持端到端多请求推理仿真、DeepSeek/Qwen3 模型、Prefill/Decode 分离。

**学术背景**：已被 NSDI'25 Spring 接收。技术细节请参阅我们的[论文](https://arxiv.org/abs/2410.07346)。

---

## 贡献方式

### 你可以通过以下方式参与贡献

1. **新功能开发** — 添加模型支持、并行策略、调度策略等
2. **Bug 修复** — 修复仿真精度问题、崩溃或错误结果
3. **性能优化** — 提升仿真速度、内存使用或可扩展性
4. **文档改进** — 完善教程、添加示例、修正错误
5. **基准测试与验证** — 添加与真实硬件结果的对比验证
6. **Issue 报告** — 报告 Bug、提出需求或分享反馈

---

## 项目架构

SimAI 是一个模块化项目，由 5 个核心子模块（Git submodules）和若干辅助目录组成：

```
SimAI/
├── aicb/                        # AI 计算基准 — 工作负载生成（Python）
│   ├── workload_generator/      #   训练/推理工作负载生成器
│   └── aicb.py                  #   主入口
├── astra-sim-alibabacloud/      # 仿真引擎 — 核心仿真器（C++）
│   ├── astra-sim/               #   基于 astra-sim 1.0 扩展
│   └── build.sh                 #   编译脚本
├── ns-3-alibabacloud/           # NS-3 网络仿真后端（C++）
├── vidur-alibabacloud/          # LLM 推理仿真框架（Python）
│   ├── vidur/                   #   核心仿真框架
│   └── setup.py                 #   Python 包配置
├── SimCCL/                      # 集合通信变换库
├── docs/                        # 文档与教程
├── example/                     # 示例工作负载和配置
├── scripts/                     # 构建和工具脚本
│   └── build.sh                 #   主编译脚本
├── results/                     # 仿真结果输出目录
├── bin/                         # 编译产物目录
├── Dockerfile                   # Docker 容器定义
└── README.md                    # 项目文档
```

---

## 开发环境搭建

### 前置依赖

- **Python** 3.8+（Docker 镜像中推荐 3.12）
- **CMake** 3.16+
- **GCC/G++** 9.4+
- **Git**（支持 submodule）

### 方式一：Docker（推荐）

```bash
# 构建 Docker 镜像
docker build -t simai:latest .

# 启动容器（支持 GPU）
docker run --gpus all -it --rm \
    -v $(pwd)/results:/workspace/SimAI/results \
    simai:latest /bin/bash
```

### 方式二：源码编译

```bash
# 1. 克隆仓库（含子模块）
git clone --recurse-submodules https://github.com/aliyun/SimAI.git
cd SimAI

# 2. 编译 C++ 组件（选择一种模式）
# 分析模式（快速仿真，不含网络细节）：
bash scripts/build.sh -c analytical

# NS-3 仿真模式（全栈，详细网络建模）：
bash scripts/build.sh -c ns3

# 物理模式（Beta，RDMA 集群）：
bash scripts/build.sh -c phy

# 3. 安装 Python 依赖
pip install -r aicb/requirements.txt
pip install -r vidur-alibabacloud/requirements.txt

# 4. 验证编译结果
ls bin/  # 应包含 SimAI_analytical 或 SimAI_simulator
```

### 验证安装

```bash
# 快速测试：运行一个简单的分析仿真
cd bin
./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml
```

---

## 子模块开发指南

SimAI 使用 Git submodule 管理核心组件。理解子模块的工作方式对于贡献至关重要。

### 子模块概览

| 子模块 | 仓库 | 语言 | 说明 |
|--------|------|------|------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | Python | 工作负载生成 |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | Python | 集合通信变换 |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | C++ | 网络仿真 |
| `astra-sim-alibabacloud` | 仓库内 | C++ | 仿真引擎 |
| `vidur-alibabacloud` | 仓库内 | Python | 推理仿真 |

### 关键规则

1. **子模块有独立的 Git 历史。** 子模块目录内的更改由该子模块自身的仓库跟踪，而非父仓库。
2. **父仓库只跟踪子模块的 commit hash。** 修改子模块后，需在子模块和父仓库中分别提交。
3. **克隆后务必初始化子模块：**
   ```bash
   git submodule update --init --recursive
   ```

### 跨子模块修改

如果你的贡献涉及多个子模块（例如在 `aicb` 中添加新模型，同时在 `astra-sim-alibabacloud` 中添加仿真支持）：

1. 在每个子模块中分别修改并提交
2. 更新父仓库，指向子模块的新 commit
3. 如果子模块有独立的远程仓库，需分别创建 PR
4. 在 PR 描述中互相引用关联的 PR

---

## 开发工作流

### 第一步：Fork 和 Clone

```bash
# 先在 GitHub 上 Fork 仓库，然后：
git clone --recurse-submodules https://github.com/YOUR_USERNAME/SimAI.git
cd SimAI

# 添加上游远程仓库
git remote add upstream https://github.com/aliyun/SimAI.git
```

### 第二步：创建功能分支

```bash
# 先同步上游代码
git fetch upstream
git checkout -b feature/your-feature-name upstream/master

# 分支命名约定：
#   feature/xxx  — 新功能
#   fix/xxx      — Bug 修复
#   docs/xxx     — 文档更新
#   perf/xxx     — 性能优化
#   refactor/xxx — 代码重构
```

### 第三步：开发与测试

```bash
# 进行修改...
# 运行相关测试（见下方"测试要求"章节）

# C++ 代码修改后需重新编译：
bash scripts/build.sh -c analytical  # 或 ns3

# Python 代码修改后，验证导入和基本功能：
python -c "from aicb import ..."
```

### 第四步：提交变更

```bash
# 暂存更改
git add -A

# 使用规范的提交消息（见下方 Commit 规范）
git commit -m "feat(aicb): add Llama-4 model workload generation"
```

### 第五步：推送并创建 PR

```bash
# 推送到你的 Fork
git push origin feature/your-feature-name

# 然后在 GitHub 上创建 Pull Request
```

---

## 代码规范

### Python

- **格式化工具**：[black](https://github.com/psf/black)（默认设置）
- **导入排序**：[isort](https://pycqa.github.io/isort/)（兼容 black）
- **静态检查**：[flake8](https://flake8.pycqa.org/)
- **最大行宽**：120 字符

```bash
# 格式化 Python 代码
black --line-length 120 your_file.py
isort your_file.py
flake8 your_file.py --max-line-length 120
```

### C++

- 遵循 `astra-sim-alibabacloud/` 中现有的代码风格
- 使用 4 空格缩进
- 函数和变量名使用 `snake_case`
- 为非平凡逻辑添加注释

### Shell 脚本

- 使用 `#!/bin/bash` 声明
- 变量一律加引号：`"${VAR}"` 而非 `$VAR`
- 适当使用 `set -e` 进行错误处理

### 通用规则

- 代码注释使用**英文**
- 所有新函数/类应有文档字符串或头注释
- 避免硬编码路径，使用相对路径或配置变量
- 保持改动聚焦——每个 PR 只做一件事

---

## Commit 消息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[可选的正文]

[可选的脚注]
```

### 类型（Type）

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 仅文档变更 |
| `refactor` | 代码重构（非新功能/修复） |
| `test` | 添加或更新测试 |
| `perf` | 性能优化 |
| `chore` | 构建流程、工具、依赖 |

### 作用域（Scope）

`aicb`、`vidur`、`astra-sim`、`ns3`、`simccl`、`docs`、`docker`、`scripts`

### 示例

**正确：**
```
feat(aicb): add DeepSeek-V3 inference workload generation
fix(astra-sim): correct AllReduce latency calculation for ring algorithm
docs: update build instructions for NS-3 mode
perf(vidur): reduce memory allocation in request scheduler
```

**错误：**
```
update code                          # 太模糊
fix bug                              # 无作用域，无描述
feat(aicb): Add DeepSeek-V3 inference workload generation support for the new model architecture  # 太长
```

---

## Pull Request 规范

### PR 标题

与 Commit 消息格式一致：`<type>(<scope>): <description>`

### PR 描述模板

```markdown
## 概述
简要描述本 PR 的内容。

## 变更内容
- 变更 1
- 变更 2

## 测试方式
描述你如何测试了这些变更。

## 关联 Issue
Closes #xxx（如适用）

## 自检清单
- [ ] 代码编译无错误
- [ ] 现有仿真结果不受影响（无精度回归）
- [ ] 新代码有适当的注释
- [ ] 为新功能添加了测试
- [ ] 必要时更新了文档
```

---

## 测试要求

### 训练仿真测试（分析模式）

```bash
# 生成训练工作负载
cd aicb
python aicb.py -m training --model_name GPT-175B

# 运行分析仿真
cd ../bin
./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml
```

### 训练仿真测试（NS-3 模式）

```bash
# 先编译 NS-3 后端
bash scripts/build.sh -c ns3

# 运行全栈仿真
cd bin
./SimAI_simulator [仿真参数]
```

### 推理仿真测试

```bash
# 通过 vidur 运行推理仿真
cd vidur-alibabacloud
python -m vidur.main [配置选项]
```

### 精度回归验证

修改仿真逻辑时，务必与已知基线结果进行对比：

```bash
# 修改前保存基线结果
cp results/output_baseline.csv /tmp/baseline.csv

# 修改后对比
diff results/output_baseline.csv /tmp/baseline.csv
```

---

## 提交前质量检查清单

提交 PR 之前，请逐项检查：

```bash
# 1. C++ 编译检查（如果修改了 C++ 代码）
bash scripts/build.sh -c analytical
bash scripts/build.sh -c ns3

# 2. Python 代码检查
black --check --line-length 120 your_changed_files.py
flake8 your_changed_files.py --max-line-length 120

# 3. 基本仿真测试
cd bin && ./SimAI_analytical \
    --workload_path=../example/workload_analytical.txt \
    --comm_group_type=TP_GROUP \
    --busbw_path=../example/busbw.yaml

# 4. 子模块状态检查
git submodule status  # 确保没有意外的子模块变更

# 5. 确认无遗漏文件
git diff --stat  # 提交前审查所有变更
```

**检查清单总结：**
- [ ] C++ 代码编译无错误或警告
- [ ] Python 代码通过 lint 检查（black、flake8）
- [ ] 基本仿真运行成功
- [ ] 无意外的子模块指针变更
- [ ] Commit 消息符合规范
- [ ] PR 描述完整

---

## 审查流程与接受标准

### 接受标准

你的贡献需要满足以下标准才能被接受：

| 标准 | 要求 |
|------|------|
| **编译** | C++ 和 Python 代码均无编译/语法错误 |
| **精度** | 不降低现有仿真精度 |
| **测试** | 关键代码路径有测试或验证覆盖 |
| **文档** | 新功能有注释和/或文档更新 |
| **风格** | 遵循上述代码规范 |
| **范围** | 变更聚焦且解释清晰 |

### 拒绝原因

- 编译失败
- 仿真精度回归且无合理解释
- 新功能缺少测试
- PR 过大且混合了不相关的变更
- 描述不充分

### 审查时间线

1. **初始审查**：3-5 个工作日内
2. **反馈**：建设性的意见和可操作的建议
3. **迭代**：根据反馈更新 PR
4. **合并**：审批通过后合入主分支

---

## 不接受的贡献

- 引入私有或闭源依赖
- 未经讨论就破坏向后兼容性的变更
- 大规模格式化变更（请先开 Issue 讨论）
- 仿真关键路径上的未测试代码
- 包含敏感信息的提交（API 密钥、内部 URL 等）

---

## 致谢与认可

贡献者将获得以下认可：
- 在发布说明中致谢
- 在项目文档中列名
- 在 commit 历史中署名

重要贡献者可能被邀请加入维护者团队。

---

## 获取帮助

- **Issue**：[GitHub Issues](https://github.com/aliyun/SimAI/issues)
- **讨论**：以 "Question:" 前缀开一个 Issue
- **文档**：参阅 [docs/Tutorial.md](docs/Tutorial.md) 获取详细使用指南
- **社区活动**：查看 [README.md](README.md) 了解近期活动和研讨会

---

## 感谢

SimAI 由一个不断壮大的研究者和工程师社区共同构建。你的每一份贡献都在推动 AI 系统研究的发展。

**让我们一起创造更好的未来！**
