<p align="left">
    中文&nbsp ｜ &nbsp<a href="SimAI_1.6_Tech_Report.md">English</a>
</p>

# SimAI 1.6 技术报告

> 本报告涵盖 SimAI 1.5 全部功能及 SimAI 1.6 新增特性。

## 1. 概述

**SimAI** 是业界首个全栈高精度 AI 大规模推理与训练模拟器（**Sim**ulator for **AI**），由阿里云开源。SimAI 对 LLM 推理与训练全流程进行详细建模和仿真，涵盖框架层、集合通信层、网络传输层等，提供端到端的性能数据。SimAI 论文已被 NSDI'25 Spring 接收 [1]。

SimAI 1.6 在 SimAI 1.5 的基础上进一步增强，主要新增了 **GPU 显存计算模块**（支持 DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B 三种 MoE 模型的精确参数量计算与 KV Cache 管理）、**四场景端到端测试套件**，以及全面的代码质量改进（双语文档、日志系统、死代码清理等）。

### 组件构成

```
        |--- AICB                        (工作负载生成与计算 profiling)
SimAI --|--- SimCCL                      (集合通信算法分析)
        |--- astra-sim-alibabacloud      (仿真引擎：Analytical / Simulation / Physical)
        |--- ns-3-alibabacloud           (NS-3 网络后端)
        |--- vidur-alibabacloud          (多请求推理调度与显存管理)
```

---

## 2. 关键里程碑

以下为 2025 年 11 月至 2026 年 3 月的关键开发事件：

| 时间 | 事件 | 说明 |
|------|------|------|
| 2025/11 | AICB PR [#58](https://github.com/aliyun/aicb/pull/58) | AICB 新增推理工作负载生成能力，区分 prefill/decode 阶段，支持 DeepSeek、Qwen3-MoE、Qwen3-Next |
| 2025/12 | AICB PR [#60](https://github.com/aliyun/aicb/pull/60) | AICB 进一步更新，完善推理工作负载生成 |
| 2025/12 | SimAI PR [#203](https://github.com/aliyun/SimAI/pull/203) | SimAI 1.5 核心更新：端到端推理仿真、PD 分离、Vidur 调度集成、现代模型支持 |
| 2025/12 | ns-3 commit [7e3cb5b](https://github.com/aliyun/ns-3-alibabacloud/commit/7e3cb5b88c99abcb582c5abc3919484a4805111b) | ns-3-alibabacloud README 文档增强，详细说明 NS3 网络后端修改 |
| 2026/01 | 显存模块系列 commit | 完成 DeepSeek-V3-671B、Qwen3-Next-80B、Qwen3-MoE-235B 的精确显存计算 |
| 2026/02 | PD 分离显存规划 | 实现 Prefill/Decode 阶段独立的参数显存与 KV Cache 预算计算 |
| 2026/03 | 代码质量改进 | 双语注释/文档/日志全面改进，死代码清理，TODO 标准化，类型注解补充 |

---

## 3. 端到端推理仿真

SimAI 支持完整的多请求 LLM 推理仿真，核心特性如下：

### 3.1 Prefill–Decode（PD）分离架构

推理过程分为两个阶段：

- **Prefill 阶段**：处理输入 prompt 的全部 token，生成第一个输出 token（计算密集型）
- **Decode 阶段**：逐 token 自回归生成后续输出（访存密集型）

PD 分离允许将 Prefill 和 Decode 阶段部署在不同的 GPU 节点上，实现：
- 弹性资源分配（Prefill 节点可配置更多计算资源，Decode 节点可配置更多显存）
- 性能隔离（避免 Prefill 和 Decode 之间的资源争用）
- 灵活的 P:D 节点比例配置（通过 `--replica_config_pd_node_ratio` 控制）

该设计参考了 [splitwise-sim](https://github.com/Mutinifni/splitwise-sim) [6]。

### 3.2 多请求推理调度

请求调度组件基于微软 [Vidur](https://github.com/microsoft/vidur) [5] 改编（vidur-alibabacloud），支持以下调度策略：

| 调度器类型 | 级别 | 说明 |
|-----------|------|------|
| `split_wise` | 全局 | PD 分离场景下的全局调度，将请求分配到 Prefill 和 Decode 副本 |
| `lor` | 全局 | Least Outstanding Requests，将请求分配到负载最轻的副本 |
| `round_robin` | 全局 | 轮询分配 |
| `sarathi` | 副本级 | 单副本内的批处理调度 |
| `split_wise` | 副本级 | PD 分离场景下的副本级调度 |

### 3.3 灵活的并行策略

支持多种并行策略的组合：

- **数据并行（DP）** — 通过 `--cluster_config_num_replicas` 控制
- **张量并行（TP）** — 通过 `--replica_config_tensor_parallel_size` 控制
- **流水线并行（PP）** — 通过 `--replica_config_num_pipeline_stages` 控制
- **专家并行（EP）** — 通过 `--replica_config_expert_model_parallel_size` 控制

同时支持 Dense 模型和 MoE（混合专家）模型。

### 3.4 多种执行时间预测后端

| 后端 | 说明 |
|------|------|
| **AICB/AIOB** | 部分支持 DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B 的计算核与 TP/DP/PP/EP 通信量建模 |
| **SimAI Simulation** | 基于 SimAI NS-3 的网络通信全栈仿真（当前支持 TP） |
| **SimAI Analytical** | SimAI 解析性能模型（当前支持 TP） |
| **Native Vidur** | 原版 Vidur 后端，支持 TP、DP、PP |

---

## 4. 现代模型支持

SimAI 1.6 支持以下三种前沿 MoE 大模型，模型配置文件位于 `vidur-alibabacloud/data/hf_configs/`：

### 4.1 DeepSeek-V3-671B

| 属性 | 值 |
|------|-----|
| 总层数 | 61 |
| 注意力类型 | MLA（Multi-head Latent Attention） |
| 注意力头数 | 128 |
| 隐藏维度 | 7168 |
| KV LoRA 秩 | 512 |
| Q LoRA 秩 | 1536 |
| QK RoPE 头维度 | 64 |
| QK NoPE 头维度 | 128 |
| V 头维度 | 128 |
| MoE 路由专家数 | 256 |
| 每 token 激活专家数 | 8 |
| 共享专家数 | 1 |
| Dense 层（前 3 层） | 固定激活 8 个路由专家 + 1 个共享专家 |
| Sparse 层（第 3-60 层） | 从 256 个路由专家中动态选择 8 个 + 1 个共享专家 |

配置文件：`data/hf_configs/deepseek_v3_config.json`

### 4.2 Qwen3-MoE-235B

| 属性 | 值 |
|------|-----|
| 总层数 | 94 |
| 注意力类型 | MHA/GQA |
| 注意力头数 | 64 |
| KV 头数 | 4 |
| 隐藏维度 | 4096 |
| 头维度 | 128 |
| MoE 路由专家数 | 128 |
| 每 token 激活专家数 | 8 |
| MoE 中间维度 | 1536 |

配置文件：`data/hf_configs/qwen3_moe_config.json`

### 4.3 Qwen3-Next-80B

| 属性 | 值 |
|------|-----|
| 总层数 | 48 |
| 注意力类型 | 混合（全注意力 + 线性注意力，每 4 层交替） |
| 全注意力头数 | 16 |
| KV 头数 | 2 |
| 隐藏维度 | 2048 |
| 头维度 | 256 |
| 线性注意力键头数 | 16 |
| 线性注意力值头数 | 32 |
| MoE 路由专家数 | 512 |
| 每 token 激活专家数 | 10 |
| MoE 中间维度 | 512 |

配置文件：`data/hf_configs/qwen3-next-80B-A3B_config.json`

---

## 5. GPU 显存计算模块

这是 SimAI 1.6 的核心新增特性。该模块为推理仿真提供精确的 GPU 显存估算，覆盖模型参数显存、KV Cache 显存和最大批处理量计算，并在 PD 分离架构下分别计算 Prefill 和 Decode 阶段的显存预算。

### 5.1 参数量计算（ParamCounter）

**文件路径**：`vidur-alibabacloud/vidur/utils/param_counter.py`

ParamCounter 支持按层、按设备计算模型参数量，并在 PD 分离架构下返回三元组 `(total_params, prefill_params, decode_params)`。

#### MLA 参数量（DeepSeek-V3-671B）

单层 MLA 参数量由以下部分组成：

- **Q LoRA 下投影**：`wq_down = hidden_size * q_lora_rank` = 7168 * 1536
- **Q LoRA 上投影**：`wq_up = q_lora_rank * num_attention_heads * qk_head_dim` = 1536 * 128 * 192，其中 `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- **KV LoRA 下投影**：`wkv_down = hidden_size * kv_lora_rank` = 7168 * 512
- **KV LoRA 上投影**：`wkv_up = kv_lora_rank * num_attention_heads * (qk_nope_head_dim + v_head_dim)` = 512 * 128 * 256
- **输出投影**：`wo = hidden_size * num_attention_heads * v_head_dim` = 7168 * 128 * 128

FP8 量化下每个参数元素占 1 字节；FP16/BF16 下每个占 2 字节。

参考：[3] [4]

#### MHA/GQA 参数量（Qwen3-MoE-235B）

单层 MHA 参数量：

```
wq = hidden_size * num_attention_heads * head_dim
wk = hidden_size * num_key_value_heads * head_dim
wv = hidden_size * num_key_value_heads * head_dim
wo = hidden_size * num_attention_heads * head_dim
total = (wq + wk + wv + wo) * bytes_per_element
```

#### 线性注意力参数量（Qwen3-Next-80B）

Qwen3-Next-80B 采用混合注意力架构，每 4 层交替使用全注意力和线性（GDN）注意力。线性注意力层使用独立的键/值头配置（`linear_key_head_dim`、`linear_num_key_heads` 等）。

#### MoE 专家参数量

每个专家的 FFN 参数量（3 个权重矩阵 W1、W2、W3）：

```
expert_params = 3 * hidden_size * moe_intermediate_size * bytes_per_element
```

#### PD 分离下的参数量计算

在 PD 分离架构下，Prefill 和 Decode 集群的专家并行度（EP）可能不同：

- **Prefill 集群**：使用 `prefill_world_size` 作为 EP，每设备加载的专家数 = `num_routed_experts / prefill_world_size`
- **Decode 集群**：使用 `decode_world_size` 作为 EP，每设备加载的专家数 = `num_routed_experts / decode_world_size`

这导致 Prefill 和 Decode 集群的参数显存不同，进而影响各自可用的 KV Cache 容量。

### 5.2 KV Cache 显存管理

**文件路径**：`vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`、`vidur-alibabacloud/vidur/entities/replica.py`

#### MHA/GQA KV Cache 计算

```
kv_cache_per_token = 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
```

其中因子 2 代表 K（Key）和 V（Value）两个缓存。

#### MLA KV Cache 计算（DeepSeek-V3-671B）

MLA 架构使用压缩的 KV 表示。与 MHA 分别存储 K 和 V 缓存不同，MLA 存储一个联合编码 K 和 V 的压缩潜向量（`kv_lora_rank`），外加 RoPE 位置键（`qk_rope_head_dim`）。每 token 的 KV Cache 大小为：

```
kv_cache_per_token = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
```

其中 `kv_lora_rank = 512`，`qk_rope_head_dim = 64`。相比 MHA 每 token 缓存量 `2 * num_kv_heads * head_dim` = 2 * 128 * 128 = 32768 个元素，MLA 减少至 576 个元素——约 **57 倍**压缩。

#### 逐请求 KV Cache 追踪

`Replica` 实体（`vidur/entities/replica.py`）维护以下状态：

- `_allocated_kv_cache_memory`：已分配的 KV Cache 显存（字节）
- `_max_kv_cache_memory`：最大 KV Cache 容量（首次调用时由 MemoryPlanner 计算）
- `_kv_cache_allocation_map`：每请求 KV Cache 分配映射

支持的操作：
- `allocate_request_kv_cache_memory(request, num_blocks, block_size)` — 为请求分配 KV Cache
- `release_request_kv_cache_memory(request)` — 释放已完成请求的 KV Cache
- `get_remaining_kv_cache_capacity()` — 查询剩余 KV Cache 容量和可服务请求数

### 5.3 MemoryPlanner 显存规划

**文件路径**：`vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`

MemoryPlanner 是显存管理的核心组件，计算流程如下：

1. **计算可用 GPU 显存**：`available_memory = total_GPU_memory * (1 - memory_margin_fraction)`
2. **获取模型参数显存**：通过 ParamCounter 计算，PD 分离下返回 `(total, prefill, decode)` 三元组
3. **计算 KV Cache 可用显存**：`kv_cache_available = available_memory - param_memory`
4. **计算最大并发请求数**：`max_requests = kv_cache_available / kv_cache_per_request`

在 PD 分离架构下：
- Prefill 副本使用 `prefill_param_mem` 计算 KV Cache 预算
- Decode 副本使用 `decode_param_mem` 计算 KV Cache 预算

包含 OOM 检测：当参数显存超过可用显存时，输出错误信息并建议增加 TP/EP、使用更大 GPU 或启用 FP8 量化。

---

## 6. AICB 推理工作负载生成

[AICB](https://github.com/aliyun/aicb) 新增了推理工作负载生成能力（PR [#58](https://github.com/aliyun/aicb/pull/58)、[#60](https://github.com/aliyun/aicb/pull/60)），主要特性：

- **Prefill/Decode 阶段分离**：分别生成 Prefill 和 Decode 阶段的计算与通信工作负载
- **计算核 Profiling**：依赖以下硬件加速库（需要 Hopper SM90 或 Blackwell SM100 GPU）：
  - [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — FP8 矩阵乘法
  - [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — MLA 注意力加速
  - [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — 高性能推理内核
- **通信量建模**：支持 TP、DP、PP、EP 四种并行策略下的通信量计算
- **模型支持**：DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B

---

## 7. 四场景端到端测试套件

**文件路径**：`vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh`

提供了 4 个预配置的端到端测试场景，覆盖不同模型、并行策略和 PD 分离配置。

### 共用硬件配置

- GPU：H20 (h20_dgx)
- NVLink 带宽：1600 Gbps
- RDMA 带宽：800 Gbps
- PD P2P 带宽：800 Gbps
- 数据类型：fp8
- 请求：Poisson QPS=100，4 个请求，固定 prefill=100 / decode=8 tokens

### 场景配置表

| 场景 | 模型 | PD 分离 | World Size | TP | PP | EP | 全局调度 |
|------|------|---------|-----------|----|----|-----|---------|
| 1 | Qwen3-Next-80B (MoE) | 否 | 32 (dp=32) | 1 | 1 | 1 | lor |
| 2 | Qwen3-Next-80B (MoE) | 是 (P=2, D=6) | 8 | 1 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B (MoE) | 是 (P=2, D=6) | 8 | 8 | 1 | 8 | split_wise |
| 4 | Qwen3-MoE-235B (MoE) | 是 (P=2, D=6) | 8 | 4 | 1 | 4 | split_wise |

### 运行方式

```bash
# 运行全部 4 个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# 运行单个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 3
```

详细性能数据请运行测试套件获取。每次运行产生的输出文件包括 `request_metrics.csv`（逐请求指标）、`chrome_trace.json`（时间线追踪）、`config.json`（配置快照）以及 `plots/` 目录下的指标文件。

---

## 8. 代码质量改进

SimAI 1.6 在代码质量方面进行了系统性改进：

### 8.1 双语注释与文档

- 所有公共 API 添加中英双语 docstring
- 配置模块（config）、调度器（scheduler）、预测器（predictor）、工具类（utils）添加双语注释
- 实体模块（entities）添加双语注释
- Shell 脚本输出和 Python 运行时输出均为双语格式

### 8.2 日志系统改进

- 全面将 `print` 语句替换为 `logging` 模块（涉及 ~12 个文件）
- 统一日志格式，使用括号式双语格式（如 `"GPU总内存 (Total GPU mem): 96.00 GB"`）

### 8.3 死代码清理

- 移除约 390 行无效代码块
- 清理个人调试标记

### 8.4 TODO 标准化

- 统一为 `TODO(author): description` 格式
- 补充缺失的类型注解

---

## 9. 系统架构

### 推理仿真数据流

```
请求生成器 (Request Generator)
    │  生成合成/真实 trace 请求
    ▼
全局调度器 (Global Scheduler)
    │  将请求分配到 Prefill/Decode 副本
    ▼
副本调度器 (Replica Scheduler)
    │  批处理组装与调度
    ▼
显存管理 (MemoryPlanner + Replica)
    │  KV Cache 分配与容量检查
    ▼
执行时间预测 (Execution Time Predictor)
    │  AICB / SimAI Simulation / SimAI Analytical / Vidur
    ▼
指标收集 (Metrics Store)
    │  TTFT, TBT, E2E, 通信/计算开销
    ▼
输出 (request_metrics.csv, chrome_trace.json, plots/)
```

---

## 10. 快速开始

### 环境搭建

#### 方式一：Docker（推荐）

```bash
# 从项目根目录构建
docker build -t simai:latest .
docker run --gpus all -it --rm simai:latest
```

> 若使用 Hopper GPU，请在 Dockerfile 中添加 `ENV FLASH_MLA_DISABLE_SM100=1`。

#### 方式二：Conda

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 运行四场景测试

```bash
# 前置条件：conda activate vidur
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all
```

### 编译与运行 SimAI 训练仿真

```bash
# 编译 SimAI-Analytical
./scripts/build.sh -c analytical

# 运行
./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

---

## 11. 参考文献

[1] SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision. NSDI'25 Spring. [[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)]

[2] InferSim — Alibaba. Parameter counting and KV cache estimation. [[GitHub](https://github.com/alibaba/InferSim)]

[3] DeepSeek V3 参数推导详解. 知乎. [[link](https://zhuanlan.zhihu.com/p/21455638257)]

[4] DeepSeek V3 Parameter Size Analysis. Yang Wenbo. [[link](https://yangwenbo.com/articles/deepseek-v3-parameter-size.html)]

[5] Vidur: A Large-Scale Simulation Framework For LLM Inference. Microsoft Research. [[GitHub](https://github.com/microsoft/vidur)]

[6] splitwise-sim — Prefill-Decode Disaggregation Simulation. [[GitHub](https://github.com/Mutinifni/splitwise-sim)]
