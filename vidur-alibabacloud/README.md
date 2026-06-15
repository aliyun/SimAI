<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

# Vidur-AlibabaCloud

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Vidur ([original](https://github.com/microsoft/vidur)) is a simulation framework for large language model (LLM) inference systems.
**Vidur-AlibabaCloud** (this repository) is a customized version optimized for Alibaba Cloud **SimAI** scenarios. It supports advanced features such as **Prefill–Decode (PD) disaggregation** and includes dedicated adaptations for SOTA LLM models including **DeepSeek-V3-671B**, **Qwen3-MoE-235B**, **Qwen3-Next-80B**, and others.


---

## Table of Contents

- [Key Features](#key-features)
- [GPU Memory Calculation](#gpu-memory-calculation)
- [Supported Models](#supported-models)
- [Environment Setup](#-environment-setup)
- [Running Examples](#%EF%B8%8F-running-examples)
  - [4-Scenario Configuration](#4-scenario-configuration)
  - [Output Files](#output-files)
- [Key Input Parameters](#-key-input-parameter-reference)
- [Key Output Interpretation](#-key-output-interpretation)
- [Known Issues](#%EF%B8%8F-known-issues)
- [Help](#-help)

---

## Key Features

- **Prefill–Decode (PD) Disaggregation** — Enables running the prefill and decode stages on different nodes, allowing elastic resource allocation and performance isolation.
  (Inspired by [splitwise-sim](https://github.com/Mutinifni/splitwise-sim))
- **Flexible Parallelism** — Supports:
  - **Data Parallel (DP)**
  - **Tensor Parallel (TP)**
  - **Pipeline Parallel (PP)**
  - **Expert Parallel (EP)** (auto-set to cluster world_size, manual override not supported)

  Works for both **dense** and **Mixture-of-Experts (MoE)** models (MoE support in progress).
- **Multiple Execution-Time Prediction Backends** — Choose from:
  - **AICB/AIOB** — Partially supports computation kernels and TP, DP, PP, EP communication size for DeepSeek-V3-671B, Qwen3-MoE-235B, Qwen3-Next-80B
  - **SimAI Simulation** — SimAI NS-3-based network simulation (supports TP)
  - **SimAI Analytical** — SimAI analytical performance model (supports TP)
  - **Native Vidur [original]** — Supports TP, DP, PP
- **Workload Generation & Replay** — Replay real-world traces or generate synthetic requests using fixed or Poisson distributions.
- **Fine-Grained Metrics** — Records:
  - TTFT — Time to First Token
  - TBT / TPOT — Time Between Tokens / Time Per Output Token
  - End-to-end latency
  - Communication cost
  - Computation cost
  - Scheduling delay

---

## GPU Memory Calculation

This module provides accurate GPU memory estimation for modern MoE (Mixture-of-Experts) models during inference simulation, covering **model parameter memory**, **KV cache memory**, and **maximum batch size** calculation under Prefill–Decode (PD) disaggregation.

### Supported Attention Architectures

| Architecture | Model | Description |
|---|---|---|
| **MLA** (Multi-head Latent Attention) | DeepSeek-V3-671B | Uses LoRA-compressed KV cache (`kv_lora_rank` + `qk_rope_head_dim`) for reduced memory footprint |
| **MHA / GQA** (Multi-Head / Grouped-Query Attention) | Qwen3-MoE-235B | Standard KV cache with `num_kv_heads * head_dim` per token per layer |
| **Hybrid Full + Linear Attention** | Qwen3-Next-80B | Alternates between full attention and linear (GDN) attention every 4 layers |

### Key Components

- **`ParamCounter`** (`vidur/utils/param_counter.py`) — Computes per-layer and per-device parameter counts for MLA, MHA/GQA, linear attention, and MoE expert weights, with FP8 quantization support. Under PD disaggregation, it returns separate `(total_params, prefill_params, decode_params)` based on `prefill_world_size` / `decode_world_size`.
- **`MemoryPlanner`** (`vidur/scheduler/utils/memory_planner.py`) — Plans GPU memory budget: `available = GPU_mem * (1 - margin) - param_mem`, then computes KV cache capacity and maximum concurrent requests. Includes OOM detection with actionable suggestions.
- **Per-request KV cache tracking** (`vidur/entities/replica.py`) — Allocates and releases KV cache memory on a per-request basis, enabling accurate remaining-capacity queries at runtime.

### References & Acknowledgments

The GPU memory calculation module was developed with reference to the following works:

- [InferSim](https://github.com/alibaba/InferSim) — Parameter counting and KV cache estimation methodology
- [DeepSeek V3 Parameter Size Analysis](https://yangwenbo.com/articles/deepseek-v3-parameter-size.html) — DeepSeek V3 MLA parameter derivation
- [DeepSeek V3 Parameter Derivation (Chinese)](https://zhuanlan.zhihu.com/p/21455638257) — Detailed MLA weight decomposition

We gratefully acknowledge these resources for providing the foundational analysis that guided our implementation.

---

## Supported Models

- **DeepSeek-V3-671B** (SimAI PP communication module in progress; EP auto-set to world_size; GPU memory management supported)
- **Qwen3-MoE-235B**, **Qwen3-Next-80B** (SimAI PP communication module in progress; EP auto-set to world_size; GPU memory management supported)
- **meta-llama/Meta-Llama-3-8B** / **Meta-Llama-3-70B**
- **meta-llama/Llama-2-7b-hf** / **Llama-2-70b-hf**
- **codellama/CodeLlama-34b-Instruct-hf**
- **internlm/internlm-20b**
- **Qwen/Qwen-72B**

---

## 📦 Environment Setup

### 1. Create Conda Environment

```bash
conda env create -p ./env -f ./environment.yml
```

### 2. (Optional) Update Dev Dependencies

```bash
conda env update -f environment-dev.yml
```

### 3. Activate Environment

```bash
conda activate vidur
```

### 4. Install Python Dependencies (Using Alibaba Cloud PyPI Mirror)

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple/
```

---

### 5. Data Preparation

The examples below use trace files from `data/processed_traces/`. These files are provided by the upstream [microsoft/vidur](https://github.com/microsoft/vidur) project.

**Option A**: Clone upstream vidur and copy the trace files:

```bash
git clone https://github.com/microsoft/vidur.git /tmp/vidur
cp -r /tmp/vidur/data/processed_traces ./data/
```

**Option B**: If you already have the vidur data locally:

```bash
cp -r /path/to/vidur/data/processed_traces ./data/
```

After preparation, your directory structure should look like:

```
data/
├── processed_traces/
│   ├── splitwise_conv.csv
│   ├── splitwise_code.csv
│   └── arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv
└── hf_configs/   # Already included in this repo
```

---

## ▶️ Running Examples

### Run DeepSeek-671B with AICB

**Requirements:** SimAI and AICB Docker environment (see [README](../README.md) for setup instructions).

After setting up the environment, run the following commands:

#### DeepSeek-671B with AICB (Fixed Length Generator)

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

#### DeepSeek-671B with AICB (Trace Length Generator)

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

> ✅ Full parameter descriptions are available via `python -m vidur.main -h`.

### Run Llama-3-8B with SimAI Simulation

```bash
cd SimAI

# Compile SimAI-Simulation (ns3)
./scripts/build.sh -c ns3

# Create network topo (Spectrum-X_128g_8gps_100Gbps_A100)
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

### Run Llama-3-8B with SimAI Analytical

```bash
cd SimAI

# Compile SimAI-Analytical
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

### Run Llama-3-8B with Native Vidur [original]

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

### Run 4-Scenario Suite

For a quick validation of all supported configurations, use the bundled test script:

```bash
bash examples/vidur-ali-scenarios/run_scenarios.sh --all
```

See `bash examples/vidur-ali-scenarios/run_scenarios.sh --help` for details.

#### 4-Scenario Configuration

The following scenarios are pre-configured in `run_scenarios.sh`. All scenarios share the hardware configuration below.

**Shared Hardware Configuration:**
- GPU: H20 (h20_dgx), NVLink: 1600 Gbps, RDMA: 800 Gbps
- PD P2P bandwidth: 800 Gbps, dtype: fp8
- Request: Poisson QPS=100, 4 requests, fixed prefill=100 / decode=8 tokens

| Scenario | Model | PD Disaggregation | World Size | TP | PP | EP | Global Scheduler |
|----------|-------|---------------|------------|----|----|------------|------------------|
| 1 | Qwen3-Next-80B (MoE) | No | 32 (dp=32) | 1 | 1 | auto (=world_size) | lor |
| 2 | Qwen3-Next-80B (MoE) | Yes (P=2, D=6) | 8 | 1 | 1 | auto (=world_size) | split_wise |
| 3 | DeepSeek-671B (MoE) | Yes (P=2, D=6) | 8 | 8 | 1 | auto (=world_size) | split_wise |
| 4 | Qwen3-MoE-235B (MoE) | Yes (P=2, D=6) | 8 | 4 | 1 | auto (=world_size) | split_wise |

> **Note:** All four models use Mixture-of-Experts (MoE) architecture. EP is automatically set to the cluster world_size at runtime and cannot be manually overridden.

#### Usage

```bash
# Activate environment
conda activate vidur

# Run a single scenario (1~4)
bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1

# Run all scenarios sequentially
bash examples/vidur-ali-scenarios/run_scenarios.sh --all

# Show help
bash examples/vidur-ali-scenarios/run_scenarios.sh --help
```

#### Manual Commands (Per Scenario)

**Scenario 1: Qwen3-Next-80B without PD Disaggregation (ws=32, lor)**

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

**Scenario 2: Qwen3-Next-80B with PD Disaggregation (P=2, D=6, split_wise)**

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

**Scenario 3: DeepSeek-671B with PD Disaggregation (tp=8, EP=auto, split_wise)**

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

**Scenario 4: Qwen3-MoE-235B with PD Disaggregation (tp=4, EP=auto, split_wise)**

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

#### Output Files

**Output path depends on how you run the simulation:**

- **`run_scenarios.sh`** --- outputs to `examples/vidur-ali-scenarios/simulator_output/`
- **Direct `python -m vidur.main`** --- outputs to `./simulator_output/` (or the path specified by `--metrics_config_output_dir`)

Each run produces the following directory:

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # per-request metrics (see Key Output Interpretation)
├── chrome_trace.json       # Chrome DevTools timeline trace (open at chrome://tracing)
├── config.json             # snapshot of all simulation parameters
└── plots/                  # per-metric CSV / JSON files (including but not limited to)
    ├── request_e2e_time.csv
    ├── prefill_e2e_time.csv
    ├── pd_p2p_comm_time.csv
    ├── replica_N_memory_usage.json
    └── ...
```

> **Note:** The exact file list in `plots/` may vary across versions.
> Run-time logs (when using `run_scenarios.sh`) are saved separately to `examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log`.

---

## 🔧 Key Input Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--replica_config_pd_p2p_comm_bandwidth` | 800 | Bandwidth (Gbps) for P2P communication between Prefill and Decode nodes in PD disaggregation |
| `--replica_config_nvlink_bandwidth` | 1600 | NVLink bandwidth (Gbps) for TP/EP communications |
| `--replica_config_rdma_bandwidth` | 800 | RDMA bandwidth (Gbps) for inter-node communication |
| `--replica_config_pd_p2p_comm_dtype` | float16 | Data type for PD communication (`float16`, `float32`, etc.) |
| `--poisson_request_interval_generator_config_qps` | 0.5 | Queries per second (QPS) for Poisson request generator |
| `--synthetic_request_generator_config_num_requests` | 128 | Number of synthetic requests to generate |
| `--length_generator_config_type` | fixed | Request length generator type (`fixed`, `trace`, etc.) |
| `--fixed_request_length_generator_config_prefill_tokens` | 2048 | Number of prefill tokens per request (only effective when `--length_generator_config_type=fixed`) |
| `--fixed_request_length_generator_config_decode_tokens` | 512 | Number of decode tokens per request (only effective when `--length_generator_config_type=fixed`) |
| `--trace_request_length_generator_config_max_tokens` | 4096 | Max tokens when using trace-based length generator |
| `--trace_request_length_generator_config_trace_file` | `data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv` | Path to trace file for request lengths |
| `--interval_generator_config_type` | poisson | Inter-arrival time generator type |
| `--cluster_config_num_replicas` | 1 | Total number of replicas (i.e., data parallelism degree) |
| `--replica_config_pd_node_ratio` | 1 | Fraction of replicas allocated as prefill (P) nodes. 1 = MIXED mode (no PD disaggregation). (0, 1) = PD disaggregation enabled. E.g., 0.5 means P:D = 1:1. |
| `--global_scheduler_config_type` | round_robin | Global scheduler type (`split_wise`, `round_robin`, etc.) |
| `--replica_scheduler_config_type` | sarathi | Per-replica scheduler type |
| `--replica_config_model_name` | meta-llama/Llama-2-7b-hf | Model name (DeepSeek-671B, Qwen3-MoE-235B, Qwen3-Next-80B, etc.) |
| `--replica_config_tensor_parallel_size` | 1 | Tensor parallelism size (TP) |
| `--replica_config_num_pipeline_stages` | 1 | Number of pipeline stages (PP) |
| `--replica_config_expert_model_parallel_size` | 1 | Expert model parallelism size (EP) — auto-set to world_size internally. Passing a value != world_size raises ValueError. Manual override not recommended. |
| `--random_forrest_execution_time_predictor_config_backend` | vidur | Backend for execution time prediction (`vidur`, `simai_simulation`, `simai_analytical`, `aicb`, etc.). **Note:** `simai_simulation` and `simai_analytical` currently only model TP communication and do not support pipeline or expert parallelism. |
| `--random_forrest_execution_time_predictor_config_simai_dir` | `../` | Root directory of the SimAI simulator (only effective when backend = `simai_simulation`) |
| `--random_forrest_execution_time_predictor_config_simai_simulation_topo` | `../example/topo` | Path to SimAI topology file (only effective when backend = `simai_simulation`) |
| `--random_forrest_execution_time_predictor_config_simai_simulation_config` | `../astra-sim-alibabacloud/inputs/config/SimAI.conf` | Path to SimAI configuration file (only effective when backend = `simai_simulation`) |

### PD Disaggregation Parameters

When `pd_node_ratio` < 1, the following optional parameters become effective:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--replica_config_prefill_tensor_parallel_size` | None | Prefill-specific TP size. Falls back to `tensor_parallel_size` if not set. |
| `--replica_config_prefill_num_pipeline_stages` | None | Prefill-specific PP size. Falls back to `num_pipeline_stages` if not set. |
| `--replica_config_decode_tensor_parallel_size` | None | Decode-specific TP size. Falls back to `tensor_parallel_size` if not set. |
| `--replica_config_decode_num_pipeline_stages` | None | Decode-specific PP size. Falls back to `num_pipeline_stages` if not set. |
| `--replica_config_num_prefill_replicas` | None | Directly specify prefill replica count (takes priority over `pd_node_ratio`). |

**Example: DeepSeek-671B with PD disaggregation (P:D = 2:6)**

```bash
python -m vidur.main \
    --replica_config_model_name deepseek-671B \
    --replica_config_device h20 \
    --replica_config_network_device h20_dgx \
    --cluster_config_num_replicas 8 \
    --replica_config_pd_node_ratio 0.25 \
    --replica_config_tensor_parallel_size 8 \
    --replica_config_num_pipeline_stages 1 \
    --global_scheduler_config_type split_wise \
    --replica_scheduler_config_type split_wise \
    --random_forrest_execution_time_predictor_config_backend aicb
```

---

## 📊 Key Output Interpretation

Simulation results are saved to:

```
./simulator_output/YYYY-MM-DD_HH-MM-SS-XXXXXX/request_metrics.csv
```

### Key Columns in `request_metrics.csv`

| Column | Meaning |
|--------|---------|
| `arrived_at` / `prefill_arrived_at` | Timestamp when the request entered the system (in seconds). |
| `scheduled_at` | Timestamp when the request was first scheduled and began execution (in seconds). |
| `prefill_completed_at` | Timestamp when the Prefill phase completed and the first output token was generated. |
| `decode_arrived_at` | Timestamp when the Decode phase started. In non-PD-disaggregated setup, this typically equals `prefill_completed_at`. In PD-disaggregated setup, it is `prefill_completed_at + pd_p2p_comm_time`. |
| `decode_time` | Duration of the Decode phase (in seconds), computed as `completed_at - decode_arrived_at`. |
| `prefill_replica_id` | Replica ID that executed the Prefill phase (in PD-disaggregated setup). |
| `decode_replica_id` | Replica ID that executed the Decode phase (in PD-disaggregated setup). |
| `request_num_prefill_tokens` | Number of input tokens (i.e., prompt length). |
| `request_num_decode_tokens` | Number of output tokens (i.e., generation length). |
| `pd_p2p_comm_size` | P2P communication size (in bytes) of data transferred from the Prefill node to the Decode node (KV cache, etc.) in PD-disaggregated setup. |
| `pd_p2p_comm_time` | P2P communication time (in seconds) between Prefill and Decode nodes in PD-disaggregated setup. |
| `completed_at` | Timestamp when the request finished processing. |
| `request_execution_time` | Total actual execution time (in seconds), excluding delays due to preemption or pipeline bubbles. |
| `request_preemption_time` | Time (in seconds) spent waiting due to scheduler preemption, pipeline bubbles, or other non-execution gaps. |
| `request_scheduling_delay` | Scheduling delay before execution: `scheduled_at - arrived_at` (in seconds). |
| `request_e2e_time` | End-to-end latency: `completed_at - arrived_at` (in seconds). |
| `prefill_e2e_time` | Time To First Token (TTFT): `prefill_completed_at - arrived_at` (in seconds). |
| `tbt` | Time Between Tokens (TBT), also known as TPOT. Computed as: `decode_time / request_num_decode_tokens` (in seconds/token). |

**Notes:**

- All time-related fields are in **seconds (s)**, based on monotonic clock or Unix timestamps.
- In non-PD-disaggregated deployments, `prefill_replica_id` and `decode_replica_id` are typically identical.
- If `request_num_decode_tokens = 0`, `tbt` is undefined (may be recorded as `NaN` or `0`).
- TBT is not yet logged in `request_metrics.csv`; it can be computed manually for now.

### Sample Row (`request_metrics.csv`)

```
Request Id,request_e2e_time,...,arrived_at,prefill_arrived_at,scheduled_at,prefill_completed_at,decode_arrived_at,completed_at,...,prefill_replica_id,decode_replica_id,pd_p2p_comm_size,pd_p2p_comm_time,...
0,0.03607,...,0.0102006,0.0102006,0.0102006,0.0102265,0.0433997,0.0462744,...,0,2,3561947136,0.0331732,...
```

---

## ⚠️ Known Issues

### Plotting Warning

You may see this error at exit:

```
RuntimeError: Kaleido requires Google Chrome to be installed.
```

This occurs because the simulator tries to generate PNG plots but lacks Chrome.
**Important:** This does **NOT** affect the generation of `request_metrics.csv`.

**Solutions:**

1. **Ignore it** — CSV output is unaffected.
2. **Install Chrome:**
   ```bash
   plotly_get_chrome
   ```
3. **Disable plotting** (not recommended): Comment out these lines in `vidur/simulator.py`:
   ```python
   # self._metric_store.plot()
   # logger.info("Metrics written")
   ```
   > Disabling plotting will skip all visual outputs and `request_metrics.csv`.

---

## 📚 Help

View all CLI options:

```bash
python -m vidur.main -h
```
