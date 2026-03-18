<p align="left">
    <a href="SimAI_1.6_Tech_Report_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

# SimAI 1.6 Technical Report

> This report covers all features from SimAI 1.5 as well as the new enhancements introduced in SimAI 1.6.

## 1. Overview

**SimAI** is the industry's first full-stack, high-precision **Sim**ulator for **AI** large-scale inference and training, open-sourced by Alibaba Cloud. SimAI provides detailed modeling and simulation of the entire LLM inference and training process, encompassing the framework layer, collective communication layer, and network transport layer, delivering end-to-end performance data. The SimAI paper was accepted by NSDI'25 Spring [1].

SimAI 1.6 builds upon SimAI 1.5 with further enhancements, primarily introducing the **GPU Memory Calculation Module** (supporting accurate parameter counting and KV cache management for DeepSeek-V3-671B, Qwen3-MoE-235B, and Qwen3-Next-80B), a **4-Scenario End-to-End Test Suite**, and comprehensive code quality improvements (bilingual documentation, logging system, dead code cleanup, etc.).

### Component Overview

```
        |--- AICB                        (Workload generation & compute profiling)
SimAI --|--- SimCCL                      (Collective communication algorithm analysis)
        |--- astra-sim-alibabacloud      (Simulation engine: Analytical / Simulation / Physical)
        |--- ns-3-alibabacloud           (NS-3 network backend)
        |--- vidur-alibabacloud          (Multi-request inference scheduling & memory management)
```

---

## 2. Key Milestones

The following are the key development events from November 2025 to March 2026:

| Date | Event | Description |
|------|-------|-------------|
| 2025/11 | AICB PR [#58](https://github.com/aliyun/aicb/pull/58) | AICB adds inference workload generation with prefill/decode phase separation, supporting DeepSeek, Qwen3-MoE, and Qwen3-Next |
| 2025/12 | AICB PR [#60](https://github.com/aliyun/aicb/pull/60) | AICB further update, refining inference workload generation |
| 2025/12 | SimAI PR [#203](https://github.com/aliyun/SimAI/pull/203) | SimAI 1.5 core update: end-to-end inference simulation, PD disaggregation, Vidur scheduling integration, modern model support |
| 2025/12 | ns-3 commit [7e3cb5b](https://github.com/aliyun/ns-3-alibabacloud/commit/7e3cb5b88c99abcb582c5abc3919484a4805111b) | ns-3-alibabacloud README documentation enhancement with detailed NS3 backend modifications |
| 2026/01 | Memory module commits | Completed accurate memory calculation for DeepSeek-V3-671B, Qwen3-Next-80B, and Qwen3-MoE-235B |
| 2026/02 | PD disaggregation memory planning | Implemented independent parameter memory and KV cache budget calculation for Prefill/Decode phases |
| 2026/03 | Code quality improvements | Comprehensive bilingual comments/docs/logs, dead code cleanup, TODO standardization, type annotations |

---

## 3. End-to-End Inference Simulation

SimAI supports complete multi-request LLM inference simulation with the following core features:

### 3.1 Prefill-Decode (PD) Disaggregation Architecture

The inference process is divided into two phases:

- **Prefill phase**: Processes all input prompt tokens and generates the first output token (compute-intensive)
- **Decode phase**: Autoregressively generates subsequent output tokens one at a time (memory-bandwidth-intensive)

PD disaggregation allows deploying Prefill and Decode phases on different GPU nodes, enabling:
- Elastic resource allocation (Prefill nodes can be configured with more compute, Decode nodes with more memory)
- Performance isolation (avoiding resource contention between Prefill and Decode)
- Flexible P:D node ratio configuration (via `--replica_config_pd_node_ratio`)

This design was inspired by [splitwise-sim](https://github.com/Mutinifni/splitwise-sim) [6].

### 3.2 Multi-Request Inference Scheduling

The request scheduling component is adapted from Microsoft's [Vidur](https://github.com/microsoft/vidur) [5] (vidur-alibabacloud), supporting the following scheduling strategies:

| Scheduler Type | Level | Description |
|---------------|-------|-------------|
| `split_wise` | Global | Global scheduling for PD disaggregation, dispatching requests to Prefill and Decode replicas |
| `lor` | Global | Least Outstanding Requests, dispatching to the least-loaded replica |
| `round_robin` | Global | Round-robin dispatch |
| `sarathi` | Per-replica | Intra-replica batch scheduling |
| `split_wise` | Per-replica | Per-replica scheduling for PD disaggregation |

### 3.3 Flexible Parallelism

Supports combinations of multiple parallelism strategies:

- **Data Parallel (DP)** — via `--cluster_config_num_replicas`
- **Tensor Parallel (TP)** — via `--replica_config_tensor_parallel_size`
- **Pipeline Parallel (PP)** — via `--replica_config_num_pipeline_stages`
- **Expert Parallel (EP)** — via `--replica_config_expert_model_parallel_size`

Works for both dense and MoE (Mixture-of-Experts) models.

### 3.4 Multiple Execution-Time Prediction Backends

| Backend | Description |
|---------|-------------|
| **AICB/AIOB** | Partially supports compute kernels and TP/DP/PP/EP communication size modeling for DeepSeek-V3-671B, Qwen3-MoE-235B, Qwen3-Next-80B |
| **SimAI Simulation** | SimAI NS-3-based full-stack network simulation (currently supports TP) |
| **SimAI Analytical** | SimAI analytical performance model (currently supports TP) |
| **Native Vidur** | Original Vidur backend, supports TP, DP, PP |

---

## 4. Modern Model Support

SimAI 1.6 supports the following three state-of-the-art MoE large models, with configuration files located in `vidur-alibabacloud/data/hf_configs/`:

### 4.1 DeepSeek-V3-671B

| Attribute | Value |
|-----------|-------|
| Total Layers | 61 |
| Attention Type | MLA (Multi-head Latent Attention) |
| Attention Heads | 128 |
| Hidden Size | 7168 |
| KV LoRA Rank | 512 |
| Q LoRA Rank | 1536 |
| QK RoPE Head Dim | 64 |
| QK NoPE Head Dim | 128 |
| V Head Dim | 128 |
| MoE Routed Experts | 256 |
| Experts Per Token | 8 |
| Shared Experts | 1 |
| Dense Layers (first 3) | Fixed activation of 8 routed experts + 1 shared expert |
| Sparse Layers (layers 3-60) | Dynamically select 8 from 256 routed experts + 1 shared expert |

Configuration file: `data/hf_configs/deepseek_v3_config.json`

### 4.2 Qwen3-MoE-235B

| Attribute | Value |
|-----------|-------|
| Total Layers | 94 |
| Attention Type | MHA/GQA |
| Attention Heads | 64 |
| KV Heads | 4 |
| Hidden Size | 4096 |
| Head Dim | 128 |
| MoE Routed Experts | 128 |
| Experts Per Token | 8 |
| MoE Intermediate Size | 1536 |

Configuration file: `data/hf_configs/qwen3_moe_config.json`

### 4.3 Qwen3-Next-80B

| Attribute | Value |
|-----------|-------|
| Total Layers | 48 |
| Attention Type | Hybrid (full + linear attention, alternating every 4 layers) |
| Full Attention Heads | 16 |
| KV Heads | 2 |
| Hidden Size | 2048 |
| Head Dim | 256 |
| Linear Attention Key Heads | 16 |
| Linear Attention Value Heads | 32 |
| MoE Routed Experts | 512 |
| Experts Per Token | 10 |
| MoE Intermediate Size | 512 |

Configuration file: `data/hf_configs/qwen3-next-80B-A3B_config.json`

---

## 5. GPU Memory Calculation Module

This is the core new feature in SimAI 1.6. The module provides accurate GPU memory estimation for inference simulation, covering model parameter memory, KV cache memory, and maximum batch size calculation, with separate memory budget computation for Prefill and Decode phases under PD disaggregation.

### 5.1 Parameter Counting (ParamCounter)

**File path**: `vidur-alibabacloud/vidur/utils/param_counter.py`

ParamCounter supports per-layer and per-device parameter counting, returning a triple `(total_params, prefill_params, decode_params)` under PD disaggregation.

#### MLA Parameters (DeepSeek-V3-671B)

Per-layer MLA parameter components:

- **Q LoRA down-projection**: `wq_down = hidden_size * q_lora_rank` = 7168 * 1536
- **Q LoRA up-projection**: `wq_up = q_lora_rank * num_attention_heads * qk_head_dim` = 1536 * 128 * 192, where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- **KV LoRA down-projection**: `wkv_down = hidden_size * kv_lora_rank` = 7168 * 512
- **KV LoRA up-projection**: `wkv_up = kv_lora_rank * num_attention_heads * (qk_nope_head_dim + v_head_dim)` = 512 * 128 * 256
- **Output projection**: `wo = hidden_size * num_attention_heads * v_head_dim` = 7168 * 128 * 128

Under FP8 quantization, each parameter element uses 1 byte; under FP16/BF16, each uses 2 bytes.

References: [3] [4]

#### MHA/GQA Parameters (Qwen3-MoE-235B)

Per-layer MHA parameters:

```
wq = hidden_size * num_attention_heads * head_dim
wk = hidden_size * num_key_value_heads * head_dim
wv = hidden_size * num_key_value_heads * head_dim
wo = hidden_size * num_attention_heads * head_dim
total = (wq + wk + wv + wo) * bytes_per_element
```

#### Linear Attention Parameters (Qwen3-Next-80B)

Qwen3-Next-80B uses a hybrid attention architecture, alternating between full attention and linear (GDN) attention every 4 layers. Linear attention layers use independent key/value head configurations (`linear_key_head_dim`, `linear_num_key_heads`, etc.).

#### MoE Expert Parameters

Per-expert FFN parameters (3 weight matrices W1, W2, W3):

```
expert_params = 3 * hidden_size * moe_intermediate_size * bytes_per_element
```

#### PD Disaggregation Parameter Calculation

Under PD disaggregation, the expert parallelism (EP) may differ between Prefill and Decode clusters:

- **Prefill cluster**: Uses `prefill_world_size` as EP, experts per device = `num_routed_experts / prefill_world_size`
- **Decode cluster**: Uses `decode_world_size` as EP, experts per device = `num_routed_experts / decode_world_size`

This results in different parameter memory for Prefill and Decode clusters, which in turn affects their respective available KV cache capacity.

### 5.2 KV Cache Memory Management

**File path**: `vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`, `vidur-alibabacloud/vidur/entities/replica.py`

#### MHA/GQA KV Cache Calculation

```
kv_cache_per_token = 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
```

The factor of 2 represents the K (Key) and V (Value) caches.

#### MLA KV Cache Calculation (DeepSeek-V3-671B)

The MLA architecture uses compressed KV representations. Unlike MHA which stores separate K and V caches, MLA stores a single compressed latent vector (`kv_lora_rank`) that jointly encodes K and V, plus the RoPE position keys (`qk_rope_head_dim`). Per-token KV cache size:

```
kv_cache_per_token = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
```

Where `kv_lora_rank = 512` and `qk_rope_head_dim = 64`. Compared to MHA's per-token cache of `2 * num_kv_heads * head_dim` = 2 * 128 * 128 = 32768 elements, MLA reduces this to 576 elements — a **~57x** reduction.

#### Per-Request KV Cache Tracking

The `Replica` entity (`vidur/entities/replica.py`) maintains the following state:

- `_allocated_kv_cache_memory`: Currently allocated KV cache memory (bytes)
- `_max_kv_cache_memory`: Maximum KV cache capacity (computed on first call by MemoryPlanner)
- `_kv_cache_allocation_map`: Per-request KV cache allocation mapping

Supported operations:
- `allocate_request_kv_cache_memory(request, num_blocks, block_size)` — Allocate KV cache for a request
- `release_request_kv_cache_memory(request)` — Release KV cache for a completed request
- `get_remaining_kv_cache_capacity()` — Query remaining KV cache capacity and serviceable request count

### 5.3 MemoryPlanner

**File path**: `vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`

MemoryPlanner is the central component for memory management, with the following calculation flow:

1. **Compute available GPU memory**: `available_memory = total_GPU_memory * (1 - memory_margin_fraction)`
2. **Get model parameter memory**: Computed via ParamCounter; under PD disaggregation returns `(total, prefill, decode)` triple
3. **Compute KV cache available memory**: `kv_cache_available = available_memory - param_memory`
4. **Compute maximum concurrent requests**: `max_requests = kv_cache_available / kv_cache_per_request`

Under PD disaggregation:
- Prefill replicas use `prefill_param_mem` for KV cache budget calculation
- Decode replicas use `decode_param_mem` for KV cache budget calculation

Includes OOM detection: when parameter memory exceeds available memory, error messages are output with suggestions to increase TP/EP, use larger GPUs, or enable FP8 quantization.

---

## 6. AICB Inference Workload Generation

[AICB](https://github.com/aliyun/aicb) introduces inference workload generation capabilities (PR [#58](https://github.com/aliyun/aicb/pull/58), [#60](https://github.com/aliyun/aicb/pull/60)), with key features:

- **Prefill/Decode phase separation**: Generates separate compute and communication workloads for Prefill and Decode phases
- **Compute kernel profiling**: Relies on the following hardware-accelerated libraries (requires Hopper SM90 or Blackwell SM100 GPUs):
  - [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — FP8 matrix multiplication
  - [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — MLA attention acceleration
  - [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — High-performance inference kernels
- **Communication size modeling**: Supports communication size calculation for TP, DP, PP, EP parallelism strategies
- **Model support**: DeepSeek-V3-671B, Qwen3-MoE-235B, Qwen3-Next-80B

---

## 7. Four-Scenario End-to-End Test Suite

**File path**: `vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh`

Provides 4 pre-configured end-to-end test scenarios covering different models, parallelism strategies, and PD disaggregation configurations.

### Shared Hardware Configuration

- GPU: H20 (h20_dgx)
- NVLink bandwidth: 1600 Gbps
- RDMA bandwidth: 800 Gbps
- PD P2P bandwidth: 800 Gbps
- Data type: fp8
- Requests: Poisson QPS=100, 4 requests, fixed prefill=100 / decode=8 tokens

### Scenario Configuration

| Scenario | Model | PD Separation | World Size | TP | PP | EP | Global Scheduler |
|----------|-------|---------------|-----------|----|----|-----|-----------------|
| 1 | Qwen3-Next-80B (MoE) | No | 32 (dp=32) | 1 | 1 | 1 | lor |
| 2 | Qwen3-Next-80B (MoE) | Yes (P=2, D=6) | 8 | 1 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B (MoE) | Yes (P=2, D=6) | 8 | 8 | 1 | 8 | split_wise |
| 4 | Qwen3-MoE-235B (MoE) | Yes (P=2, D=6) | 8 | 4 | 1 | 4 | split_wise |

### Running

```bash
# Run all 4 scenarios
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# Run a single scenario
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 3
```

For detailed performance data, please run the test suite. Each run produces output files including `request_metrics.csv` (per-request metrics), `chrome_trace.json` (timeline trace), `config.json` (configuration snapshot), and metric files under the `plots/` directory.

---

## 8. Code Quality Improvements

SimAI 1.6 includes systematic code quality improvements:

### 8.1 Bilingual Comments and Documentation

- Added bilingual (Chinese/English) docstrings to all public APIs
- Added bilingual comments to config, scheduler, predictor, and utils modules
- Added bilingual comments to entity modules
- Shell script outputs and Python runtime outputs use bilingual format

### 8.2 Logging System Improvements

- Comprehensive replacement of `print` statements with the `logging` module (~12 files)
- Unified log format using parenthetical bilingual style (e.g., `"GPU总内存 (Total GPU mem): 96.00 GB"`)

### 8.3 Dead Code Cleanup

- Removed approximately 390 lines of dead code blocks
- Cleaned up personal debug markers

### 8.4 TODO Standardization

- Unified to `TODO(author): description` format
- Added missing type annotations

---

## 9. System Architecture

### Inference Simulation Data Flow

```
Request Generator
    |  Generate synthetic / real-trace requests
    v
Global Scheduler
    |  Dispatch requests to Prefill / Decode replicas
    v
Replica Scheduler
    |  Batch assembly and scheduling
    v
Memory Management (MemoryPlanner + Replica)
    |  KV cache allocation and capacity checking
    v
Execution Time Predictor
    |  AICB / SimAI Simulation / SimAI Analytical / Vidur
    v
Metrics Store
    |  TTFT, TBT, E2E, communication / compute cost
    v
Output (request_metrics.csv, chrome_trace.json, plots/)
```

---

## 10. Quick Start

### Environment Setup

#### Option 1: Docker (Recommended)

```bash
# Build from project root
docker build -t simai:latest .
docker run --gpus all -it --rm simai:latest
```

> If using Hopper GPUs, add `ENV FLASH_MLA_DISABLE_SM100=1` to the Dockerfile.

#### Option 2: Conda

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Run 4-Scenario Test Suite

```bash
# Prerequisites: conda activate vidur
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all
```

### Compile and Run SimAI Training Simulation

```bash
# Compile SimAI-Analytical
./scripts/build.sh -c analytical

# Run
./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

---

## 11. References

[1] SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision. NSDI'25 Spring. [[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)]

[2] InferSim — Alibaba. Parameter counting and KV cache estimation. [[GitHub](https://github.com/alibaba/InferSim)]

[3] DeepSeek V3 Parameter Derivation (Chinese). Zhihu. [[link](https://zhuanlan.zhihu.com/p/21455638257)]

[4] DeepSeek V3 Parameter Size Analysis. Yang Wenbo. [[link](https://yangwenbo.com/articles/deepseek-v3-parameter-size.html)]

[5] Vidur: A Large-Scale Simulation Framework For LLM Inference. Microsoft Research. [[GitHub](https://github.com/microsoft/vidur)]

[6] splitwise-sim — Prefill-Decode Disaggregation Simulation. [[GitHub](https://github.com/Mutinifni/splitwise-sim)]
