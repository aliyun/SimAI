# 多请求推理仿真

SimAI 支持完整的多请求 LLM 推理仿真，提供端到端的推理服务系统性能评估，支持 Prefill-Decode (PD) 分离架构。

---

## 概述

推理仿真流水线组合了多个 SimAI 组件：

- **[AICB](../components/aicb.md)** — 生成推理工作负载和计算时间分析
- **[vidur-alibabacloud](../components/vidur.md)** — 请求调度、显存管理和指标收集
- **[astra-sim-alibabacloud](../components/astra_sim.md)** — 仿真引擎（Analytical 或 Simulation 模式）
- **[SimCCL](../components/simccl.md)** — 集合通信转换

---

## Prefill-Decode (PD) 分离

推理过程分为两个阶段：

| 阶段 | 特征 | 说明 |
|------|------|------|
| **Prefill** | 计算密集 | 处理所有输入 prompt token，生成第一个输出 token |
| **Decode** | 内存带宽密集 | 逐个自回归生成后续输出 token |

PD 分离允许将这两个阶段部署在不同的 GPU 节点上：

- **弹性资源分配** — Prefill 节点可配置更多算力，Decode 节点可配置更多显存
- **性能隔离** — 避免阶段间的资源竞争
- **灵活 P:D 比例** — 通过 `--replica_config_pd_node_ratio` 配置

---

## 请求调度

调度组件改编自微软 [Vidur](https://github.com/microsoft/vidur)，支持多种策略：

| 调度器 | 级别 | 说明 |
|--------|------|------|
| `split_wise` | 全局 | PD 分离感知调度，将请求分发到 Prefill 和 Decode 副本 |
| `lor` | 全局 | 最少未完成请求——分发到负载最轻的副本 |
| `round_robin` | 全局 | 轮询分发 |
| `sarathi` | 副本级 | 副本内批量调度 |
| `split_wise` | 副本级 | PD 分离的副本级调度 |

---

## 并行策略

支持多种并行策略组合：

| 策略 | 参数 | 说明 |
|------|------|------|
| **数据并行 (DP)** | `--cluster_config_num_replicas` | 副本数量 |
| **张量并行 (TP)** | `--replica_config_tensor_parallel_size` | 节点内并行 |
| **流水线并行 (PP)** | `--replica_config_num_pipeline_stages` | 阶段间并行 |
| **专家并行 (EP)** | `--replica_config_expert_model_parallel_size` | MoE 专家并行 |

适用于稠密模型和 MoE（混合专家）模型。

---

## 执行时间预测后端

| 后端 | 参数值 | 说明 |
|------|--------|------|
| **AICB/AIOB** | `aicb` | 支持 DeepSeek-V3、Qwen3-MoE、Qwen3-Next 的计算核和 TP/DP/PP/EP 通信量建模 |
| **SimAI Simulation** | `simai_simulation` | 基于 NS-3 的全栈网络仿真（当前支持 TP） |
| **SimAI Analytical** | `simai_analytical` | 分析性能模型（当前支持 TP） |
| **原生 Vidur** | `vidur` | 原版 Vidur 后端，支持 TP、DP、PP |

通过 `--random_forrest_execution_time_predictor_config_backend` 设置。

---

## 快速开始

### 前置条件

- **AICB 后端**：SimAI Docker 环境 + Hopper (SM90) 或 Blackwell (SM100) GPU
- **SimAI 后端**：先编译 SimAI-Analytical 或 SimAI-Simulation
- **Vidur 后端**：Conda 环境 + profiling 数据

### 使用 AICB 后端运行

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_expert_model_parallel_size 8 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

### 运行四场景测试套件

```bash
# 运行全部 4 个预配置场景
bash examples/vidur-ali-scenarios/run_scenarios.sh --all

# 运行单个场景
bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

---

## 四场景配置

**共享硬件**：H20 GPU (h20_dgx)，NVLink 1600 Gbps，RDMA 800 Gbps，PD P2P 800 Gbps（fp8）

| 场景 | 模型 | PD 分离 | World Size | TP | EP | 调度器 |
|------|------|---------|-----------|----|----|--------|
| 1 | Qwen3-Next-80B | 否 | 32 (dp=32) | 1 | 1 | lor |
| 2 | Qwen3-Next-80B | 是（P=2, D=6） | 8 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B | 是（P=2, D=6） | 8 | 8 | 8 | split_wise |
| 4 | Qwen3-MoE-235B | 是（P=2, D=6） | 8 | 4 | 4 | split_wise |

---

## 输出

每次仿真运行生成：

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # 逐请求指标
├── chrome_trace.json       # Chrome DevTools 时间线追踪
├── config.json             # 配置快照
└── plots/                  # 各指标 CSV/JSON 文件
```

输出解读请参见[结果分析](result_analysis.md)。

---

## 相关文档

- [vidur-alibabacloud 组件](../components/vidur.md) — 完整推理仿真文档
- [支持的模型](supported_models.md) — 模型兼容性矩阵
- [结果分析](result_analysis.md) — 输出解读指南
