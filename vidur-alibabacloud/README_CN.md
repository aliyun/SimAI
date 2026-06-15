<p align="left">
    中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>

# Vidur-AlibabaCloud

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Vidur（[原版](https://github.com/microsoft/vidur)）是一个大语言模型（LLM）推理系统的模拟框架。
**Vidur-AlibabaCloud**（本仓库）是针对阿里云 **SimAI** 场景优化的定制版本。支持 **Prefill–Decode（PD）分离**等高级特性，并针对 **DeepSeek-V3-671B**、**Qwen3-MoE-235B**、**Qwen3-Next-80B** 等 SOTA 大模型进行了专门适配。

---

## 目录

- [主要特性](#主要特性)
- [GPU 显存计算模块](#gpu-显存计算模块)
- [支持的模型](#支持的模型)
- [📦 环境配置](#-环境配置)
- [▶️ 运行示例](#️-运行示例)
  - [四场景配置说明](#四场景配置说明)
  - [输出文件说明](#输出文件说明)
- [🔧 关键输入参数参考](#-关键输入参数参考)
- [📊 输出结果解读](#-输出结果解读)
- [⚠️ 已知问题](#️-已知问题)
- [📚 帮助](#-帮助)

---

## 主要特性

- **Prefill–Decode（PD）分离** — 支持 prefill 和 decode 阶段在不同节点运行，实现弹性资源分配和性能隔离。
  （参考 [splitwise-sim](https://github.com/Mutinifni/splitwise-sim)）
- **灵活的并行策略** — 支持：
  - **数据并行（DP）**
  - **张量并行（TP）**
  - **流水线并行（PP）**
  - **专家并行（EP）**（自动设为 cluster world_size，不支持手动指定）

  同时支持 **Dense** 模型和 **混合专家（MoE）** 模型（MoE 适配中）。
- **多种执行时间预测后端** — 可选：
  - **AICB/AIOB** — 部分支持 DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B 的计算核与 TP、DP、PP、EP 通信量建模
  - **SimAI 仿真（Simulation）** — 基于 SimAI NS-3 的网络通信全栈仿真（支持 TP）
  - **SimAI 解析（Analytical）** — SimAI 解析性能模型（支持 TP）
  - **原版 Vidur [original]** — 支持 TP、DP、PP
- **负载生成与回放** — 支持真实 trace 回放，或使用固定/泊松分布生成合成请求。
- **细粒度指标** — 记录：
  - TTFT — 首 token 时延
  - TBT / TPOT — 相邻 token 时延 / 每输出 token 耗时
  - 端到端延迟
  - 通信开销
  - 计算开销
  - 调度延迟

---

## GPU 显存计算模块

本模块为现代 MoE（混合专家）模型的推理仿真提供精确的 GPU 显存估算，涵盖**模型参数显存**、**KV Cache 显存**以及 Prefill–Decode（PD）分离架构下的**最大批处理量**计算。

### 支持的注意力架构

| 架构 | 模型 | 说明 |
|---|---|---|
| **MLA**（多头潜在注意力） | DeepSeek-V3-671B | 使用 LoRA 压缩的 KV Cache（`kv_lora_rank` + `qk_rope_head_dim`），显著降低显存占用 |
| **MHA / GQA**（多头 / 分组查询注意力） | Qwen3-MoE-235B | 标准 KV Cache，每 token 每层使用 `num_kv_heads * head_dim` |
| **混合全注意力 + 线性注意力** | Qwen3-Next-80B | 每 4 层交替使用全注意力和线性（GDN）注意力 |

### 核心组件

- **`ParamCounter`**（`vidur/utils/param_counter.py`）— 计算每层和每设备的参数量，支持 MLA、MHA/GQA、线性注意力和 MoE 专家权重，支持 FP8 量化。在 PD 分离架构下，根据 `prefill_world_size` / `decode_world_size` 分别返回 `(total_params, prefill_params, decode_params)` 三元组。
- **`MemoryPlanner`**（`vidur/scheduler/utils/memory_planner.py`）— 规划 GPU 显存预算：`available = GPU_mem * (1 - margin) - param_mem`，计算 KV Cache 容量和最大并发请求数，包含 OOM 检测与建议输出。
- **逐请求 KV Cache 追踪**（`vidur/entities/replica.py`）— 按请求粒度分配和释放 KV Cache 显存，支持运行时精确查询剩余容量。

### 参考与致谢

本 GPU 显存计算模块的开发参考了以下工作：

- [InferSim](https://github.com/alibaba/InferSim) — 参数量计算与 KV Cache 估算方法论
- [DeepSeek V3 Parameter Size Analysis](https://yangwenbo.com/articles/deepseek-v3-parameter-size.html) — DeepSeek V3 MLA 参数推导
- [DeepSeek V3 参数推导详解](https://zhuanlan.zhihu.com/p/21455638257) — MLA 权重分解详细分析

衷心感谢以上资源为我们的实现提供了基础性的分析与指导。

---

## 支持的模型

- **DeepSeek-V3-671B**（SimAI PP 通信模块适配中；EP 自动设为 world_size；GPU 显存管理已支持）
- **Qwen3-MoE-235B**、**Qwen3-Next-80B**（SimAI PP 通信模块适配中；EP 自动设为 world_size；GPU 显存管理已支持）
- **meta-llama/Meta-Llama-3-8B** / **Meta-Llama-3-70B**
- **meta-llama/Llama-2-7b-hf** / **Llama-2-70b-hf**
- **codellama/CodeLlama-34b-Instruct-hf**
- **internlm/internlm-20b**
- **Qwen/Qwen-72B**

---

## 📦 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -p ./env -f ./environment.yml
```

### 2.（可选）更新开发依赖

```bash
conda env update -f environment-dev.yml
```

### 3. 激活环境

```bash
conda activate vidur
```

### 4. 安装 Python 依赖（使用阿里云 PyPI 镜像）

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple/
```

---

### 5. 数据准备

下面的示例使用 `data/processed_traces/` 中的 trace 文件。这些文件来自上游 [microsoft/vidur](https://github.com/microsoft/vidur) 项目。

**方式一**：从上游 vidur 克隆并拷贝 trace 文件：

```bash
git clone https://github.com/microsoft/vidur.git /tmp/vidur
cp -r /tmp/vidur/data/processed_traces ./data/
```

**方式二**：如果本地已有 vidur 数据：

```bash
cp -r /path/to/vidur/data/processed_traces ./data/
```

准备完成后，目录结构应如下：

```
data/
├── processed_traces/
│   ├── splitwise_conv.csv
│   ├── splitwise_code.csv
│   └── arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv
└── hf_configs/   # 本仓库已包含
```

---

## ▶️ 运行示例

### 使用 AICB 运行 DeepSeek-671B

**前置条件：** 需要 SimAI 和 AICB Docker 环境（参见 [README](../README.md) 了解搭建方法）。

完成环境配置后，运行以下命令：

#### DeepSeek-671B + AICB（固定长度生成器）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

#### DeepSeek-671B + AICB（Trace 长度生成器）

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 1024 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

> ✅ 完整参数说明可通过 `python -m vidur.main -h` 查看。

### 使用 SimAI 仿真运行 Llama-3-8B

```bash
cd SimAI

# 编译 SimAI-Simulation（ns3）
./scripts/build.sh -c ns3

# 生成网络拓扑（Spectrum-X_128g_8gps_100Gbps_A100）
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend simai_simulation \
  --random_forrest_execution_time_predictor_config_simai_dir ../ \
  --random_forrest_execution_time_predictor_config_simai_simulation_topo ../Spectrum-X_128g_8gps_100Gbps_A100 \
  --random_forrest_execution_time_predictor_config_simai_simulation_config ../astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### 使用 SimAI 解析模型运行 Llama-3-8B

```bash
cd SimAI

# 编译 SimAI-Analytical
./scripts/build.sh -c analytical

cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend simai_analytical
```

### 使用原版 Vidur 运行 Llama-3-8B

```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend vidur
```

### 运行四场景套件

使用内置脚本快速验证所有支持的配置：

```bash
bash examples/vidur-ali-scenarios/run_scenarios.sh --all
```

详细信息请运行 `bash examples/vidur-ali-scenarios/run_scenarios.sh --help`。

#### 四场景配置说明

以下场景已在 `run_scenarios.sh` 中预配置，所有场景共享下方硬件配置。

**共用硬件配置：**
- GPU：H20（h20_dgx），NVLink：1600 Gbps，RDMA：800 Gbps
- PD P2P 带宽：800 Gbps，数据类型：fp8
- 请求生成：Poisson QPS=100，4 requests，固定 prefill=100 / decode=8 tokens

| 场景 | 模型 | PD 分离 | World Size | TP | PP | EP | 全局调度器 |
|------|------|---------|------------|----|----|------------|------------|
| 1 | Qwen3-Next-80B (MoE) | 无 | 32 (dp=32) | 1 | 1 | auto (=world_size) | lor |
| 2 | Qwen3-Next-80B (MoE) | 是（P=2, D=6） | 8 | 1 | 1 | auto (=world_size) | split_wise |
| 3 | DeepSeek-671B (MoE) | 是（P=2, D=6） | 8 | 8 | 1 | auto (=world_size) | split_wise |
| 4 | Qwen3-MoE-235B (MoE) | 是（P=2, D=6） | 8 | 4 | 1 | auto (=world_size) | split_wise |

> **说明：** 四个模型均使用混合专家（MoE）架构。EP 在运行时自动设为 cluster world_size，不支持手动指定。

#### run_scenarios.sh 使用方法

```bash
# 激活环境
conda activate vidur

# 运行单个场景（1~4）
bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1

# 顺序运行所有场景
bash examples/vidur-ali-scenarios/run_scenarios.sh --all

# 查看帮助
bash examples/vidur-ali-scenarios/run_scenarios.sh --help
```

#### 手动运行命令（逐场景）

以下为四个场景的完整 CLI 命令，可直接复制运行。所有命令均在 `vidur-alibabacloud/` 目录下执行。

**场景 1：Qwen3-Next-80B 无PD分离（ws=32, lor）**

```bash
python -m vidur.main \
    --replica_config_pd_p2p_comm_bandwidth 800 \
    --replica_config_nvlink_bandwidth 1600 \
    --replica_config_rdma_bandwidth 800 \
    --replica_config_pd_p2p_comm_dtype fp8 \
    --replica_config_network_device h20_dgx \
    --replica_config_device h20 \
    --request_generator_config_type synthetic \
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 100 \
    --synthetic_request_generator_config_num_requests 4 \
    --length_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 100 \
    --fixed_request_length_generator_config_decode_tokens 8 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --random_forrest_execution_time_predictor_config_backend aicb \
    --metrics_config_output_dir examples/vidur-ali-scenarios/simulator_output \
    --cluster_config_num_replicas 32 \
    --replica_config_pd_node_ratio 1 \
    --global_scheduler_config_type lor \
    --replica_scheduler_config_type sarathi \
    --replica_config_model_name qwen3-next-80B \
    --replica_config_tensor_parallel_size 1 \
    --replica_config_num_pipeline_stages 1
```

**场景 2：Qwen3-Next-80B PD分离（P=2, D=6, split_wise）**

```bash
python -m vidur.main \
    --replica_config_pd_p2p_comm_bandwidth 800 \
    --replica_config_nvlink_bandwidth 1600 \
    --replica_config_rdma_bandwidth 800 \
    --replica_config_pd_p2p_comm_dtype fp8 \
    --replica_config_network_device h20_dgx \
    --replica_config_device h20 \
    --request_generator_config_type synthetic \
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 100 \
    --synthetic_request_generator_config_num_requests 4 \
    --length_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 100 \
    --fixed_request_length_generator_config_decode_tokens 8 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --random_forrest_execution_time_predictor_config_backend aicb \
    --metrics_config_output_dir examples/vidur-ali-scenarios/simulator_output \
    --cluster_config_num_replicas 8 \
    --replica_config_pd_node_ratio 0.25 \
    --replica_config_num_prefill_replicas 2 \
    --global_scheduler_config_type split_wise \
    --replica_scheduler_config_type split_wise \
    --replica_config_model_name qwen3-next-80B \
    --replica_config_tensor_parallel_size 1 \
    --replica_config_num_pipeline_stages 1 \
    --replica_config_prefill_tensor_parallel_size 1 \
    --replica_config_prefill_num_pipeline_stages 1 \
    --replica_config_decode_tensor_parallel_size 1 \
    --replica_config_decode_num_pipeline_stages 1
```

**场景 3：DeepSeek-671B PD分离（tp=8, EP=auto, split_wise）**

```bash
python -m vidur.main \
    --replica_config_pd_p2p_comm_bandwidth 800 \
    --replica_config_nvlink_bandwidth 1600 \
    --replica_config_rdma_bandwidth 800 \
    --replica_config_pd_p2p_comm_dtype fp8 \
    --replica_config_network_device h20_dgx \
    --replica_config_device h20 \
    --request_generator_config_type synthetic \
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 100 \
    --synthetic_request_generator_config_num_requests 4 \
    --length_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 100 \
    --fixed_request_length_generator_config_decode_tokens 8 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --random_forrest_execution_time_predictor_config_backend aicb \
    --metrics_config_output_dir examples/vidur-ali-scenarios/simulator_output \
    --cluster_config_num_replicas 8 \
    --replica_config_pd_node_ratio 0.25 \
    --global_scheduler_config_type split_wise \
    --replica_scheduler_config_type split_wise \
    --replica_config_model_name deepseek-671B \
    --replica_config_tensor_parallel_size 8 \
    --replica_config_num_pipeline_stages 1
```

**场景 4：Qwen3-MoE-235B PD分离（tp=4, EP=auto, split_wise）**

```bash
python -m vidur.main \
    --replica_config_pd_p2p_comm_bandwidth 800 \
    --replica_config_nvlink_bandwidth 1600 \
    --replica_config_rdma_bandwidth 800 \
    --replica_config_pd_p2p_comm_dtype fp8 \
    --replica_config_network_device h20_dgx \
    --replica_config_device h20 \
    --request_generator_config_type synthetic \
    --interval_generator_config_type poisson \
    --poisson_request_interval_generator_config_qps 100 \
    --synthetic_request_generator_config_num_requests 4 \
    --length_generator_config_type fixed \
    --fixed_request_length_generator_config_prefill_tokens 100 \
    --fixed_request_length_generator_config_decode_tokens 8 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --random_forrest_execution_time_predictor_config_backend aicb \
    --metrics_config_output_dir examples/vidur-ali-scenarios/simulator_output \
    --cluster_config_num_replicas 8 \
    --replica_config_pd_node_ratio 0.25 \
    --global_scheduler_config_type split_wise \
    --replica_scheduler_config_type split_wise \
    --replica_config_model_name qwen3-moe-235B \
    --replica_config_tensor_parallel_size 4 \
    --replica_config_num_pipeline_stages 1
```

#### 输出文件说明

**输出路径取决于运行方式：**

- **`run_scenarios.sh`** --- 输出到 `examples/vidur-ali-scenarios/simulator_output/`
- **直接 `python -m vidur.main`** --- 输出到 `./simulator_output/`（或通过 `--metrics_config_output_dir` 指定的路径）

每次运行产生如下目录：

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # 逐请求指标（参见"输出结果解读"）
├── chrome_trace.json       # Chrome DevTools 时间轴 trace（可在 chrome://tracing 打开）
├── config.json             # 本次仿真的全部参数快照
└── plots/                  # 逐指标 CSV / JSON 文件（包括但不限于）
    ├── request_e2e_time.csv
    ├── prefill_e2e_time.csv
    ├── pd_p2p_comm_time.csv
    ├── replica_N_memory_usage.json
    └── ...
```

> **说明：** `plots/` 中的具体文件列表可能因版本不同而变化。
> 使用 `run_scenarios.sh` 时，运行日志另存于 `examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log`。

---

## 🔧 关键输入参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--replica_config_pd_p2p_comm_bandwidth` | 800 | PD 分离中 Prefill 节点与 Decode 节点间 P2P 通信带宽（Gbps） |
| `--replica_config_nvlink_bandwidth` | 1600 | TP/EP 通信使用的 NVLink 带宽（Gbps） |
| `--replica_config_rdma_bandwidth` | 800 | 节点间通信使用的 RDMA 带宽（Gbps） |
| `--replica_config_pd_p2p_comm_dtype` | float16 | PD 通信数据类型（`float16`、`float32` 等） |
| `--poisson_request_interval_generator_config_qps` | 0.5 | 泊松请求生成器的 QPS（每秒请求数） |
| `--synthetic_request_generator_config_num_requests` | 128 | 合成请求总数 |
| `--length_generator_config_type` | fixed | 请求长度生成器类型（`fixed`、`trace` 等） |
| `--fixed_request_length_generator_config_prefill_tokens` | 2048 | 每请求的 prefill token 数（仅在 `--length_generator_config_type=fixed` 时生效） |
| `--fixed_request_length_generator_config_decode_tokens` | 512 | 每请求的 decode token 数（仅在 `--length_generator_config_type=fixed` 时生效） |
| `--trace_request_length_generator_config_max_tokens` | 4096 | 使用 trace 长度生成器时的最大 token 数 |
| `--trace_request_length_generator_config_trace_file` | `data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv` | trace 文件路径 |
| `--interval_generator_config_type` | poisson | 请求到达间隔生成器类型 |
| `--cluster_config_num_replicas` | 1 | replica 总数（即数据并行度） |
| `--replica_config_pd_node_ratio` | 0.5 | 分配为 Prefill（P）节点的 replica 比例，其余为 Decode（D）节点。例如 0.5 表示 P:D = 1:1。 |
| `--global_scheduler_config_type` | round_robin | 全局调度器类型（`split_wise`、`round_robin` 等） |
| `--replica_scheduler_config_type` | sarathi | 单 replica 调度器类型 |
| `--replica_config_model_name` | meta-llama/Llama-2-7b-hf | 模型名称（DeepSeek-671B、Qwen3-MoE-235B、Qwen3-Next-80B 等） |
| `--replica_config_tensor_parallel_size` | 1 | 张量并行大小（TP） |
| `--replica_config_num_pipeline_stages` | 1 | 流水线阶段数（PP） |
| `--replica_config_expert_model_parallel_size` | 1 | 专家并行大小（EP）— 内部自动设为 world_size，用户传入非 world_size 值将报错。不建议手动指定。 |
| `--random_forrest_execution_time_predictor_config_backend` | vidur | 执行时间预测后端（`vidur`、`simai_simulation`、`simai_analytical`、`aicb` 等）。**注意：** `simai_simulation` 和 `simai_analytical` 当前仅建模 TP 通信，不支持流水线或专家并行。 |
| `--random_forrest_execution_time_predictor_config_simai_dir` | `../` | SimAI 模拟器根目录（仅在 backend = `simai_simulation` 时生效） |
| `--random_forrest_execution_time_predictor_config_simai_simulation_topo` | `../example/topo` | SimAI 拓扑文件路径（仅在 backend = `simai_simulation` 时生效） |
| `--random_forrest_execution_time_predictor_config_simai_simulation_config` | `../astra-sim-alibabacloud/inputs/config/SimAI.conf` | SimAI 配置文件路径（仅在 backend = `simai_simulation` 时生效） |

---

## 📊 输出结果解读

仿真结果保存于：

```
./simulator_output/YYYY-MM-DD_HH-MM-SS-XXXXXX/request_metrics.csv
```

### `request_metrics.csv` 关键列说明

| 列名 | 含义 |
|------|------|
| `arrived_at` / `prefill_arrived_at` | 请求进入系统的时间戳（秒）。 |
| `scheduled_at` | 请求首次被调度并开始执行的时间戳（秒）。 |
| `prefill_completed_at` | Prefill 阶段完成、生成第一个输出 token 的时间戳。 |
| `decode_arrived_at` | Decode 阶段开始的时间戳。非 PD 分离场景下通常等于 `prefill_completed_at`；PD 分离场景下为 `prefill_completed_at + pd_p2p_comm_time`。 |
| `decode_time` | Decode 阶段持续时间（秒），计算公式：`completed_at - decode_arrived_at`。 |
| `prefill_replica_id` | 执行 Prefill 阶段的 replica ID（PD 分离场景下）。 |
| `decode_replica_id` | 执行 Decode 阶段的 replica ID（PD 分离场景下）。 |
| `request_num_prefill_tokens` | 输入 token 数（即 prompt 长度）。 |
| `request_num_decode_tokens` | 输出 token 数（即生成长度）。 |
| `pd_p2p_comm_size` | PD 分离场景下，从 Prefill 节点传输至 Decode 节点的数据量（字节，含 KV Cache 等）。 |
| `pd_p2p_comm_time` | PD 分离场景下，Prefill 节点与 Decode 节点间 P2P 通信耗时（秒）。 |
| `completed_at` | 请求处理完成的时间戳。 |
| `request_execution_time` | 实际执行总时间（秒），不含抢占或流水线气泡导致的等待。 |
| `request_preemption_time` | 因调度器抢占、流水线气泡或其他非执行间隔导致的等待时间（秒）。 |
| `request_scheduling_delay` | 执行前的调度延迟：`scheduled_at - arrived_at`（秒）。 |
| `request_e2e_time` | 端到端延迟：`completed_at - arrived_at`（秒）。 |
| `prefill_e2e_time` | 首 token 时延（TTFT）：`prefill_completed_at - arrived_at`（秒）。 |
| `tbt` | 相邻 token 时延（TBT / TPOT）。计算公式：`decode_time / request_num_decode_tokens`（秒/token）。 |

**说明：**

- 所有时间字段单位均为**秒（s）**，基于单调时钟或 Unix 时间戳。
- 非 PD 分离部署中，`prefill_replica_id` 与 `decode_replica_id` 通常相同。
- 若 `request_num_decode_tokens = 0`，则 `tbt` 未定义（可能记录为 `NaN` 或 `0`）。
- `tbt` 暂未写入 `request_metrics.csv`，目前需手动计算。

### 示例行（`request_metrics.csv`）

```
Request Id,request_e2e_time,...,arrived_at,prefill_arrived_at,scheduled_at,prefill_completed_at,decode_arrived_at,completed_at,...,prefill_replica_id,decode_replica_id,pd_p2p_comm_size,pd_p2p_comm_time,...
0,0.03607,...,0.0102006,0.0102006,0.0102006,0.0102265,0.0433997,0.0462744,...,0,2,3561947136,0.0331732,...
```

---

## ⚠️ 已知问题

### 绘图警告

退出时可能出现以下错误：

```
RuntimeError: Kaleido requires Google Chrome to be installed.
```

这是因为模拟器尝试生成 PNG 图表但缺少 Chrome。
**重要：** 此问题**不影响** `request_metrics.csv` 的生成。

**解决方案：**

1. **忽略** — CSV 输出不受影响。
2. **安装 Chrome：**
   ```bash
   plotly_get_chrome
   ```
3. **禁用绘图**（不推荐）：注释掉 `vidur/simulator.py` 中的以下行：
   ```python
   # self._metric_store.plot()
   # logger.info("Metrics written")
   ```
   > 禁用绘图将跳过所有可视化输出及 `request_metrics.csv`。

---

## 📚 帮助

查看所有 CLI 选项：

```bash
python -m vidur.main -h
```
