# Multi-Request Inference Simulation

SimAI supports complete multi-request LLM inference simulation, enabling end-to-end performance evaluation of inference serving systems with support for Prefill-Decode (PD) disaggregation.

---

## Overview

The inference simulation pipeline combines several SimAI components:

- **[AICB](../components/aicb.md)** — Generates inference workloads and profiles computation time
- **[vidur-alibabacloud](../components/vidur.md)** — Request scheduling, memory management, and metrics collection
- **[astra-sim-alibabacloud](../components/astra_sim.md)** — Simulation engine (Analytical or Simulation mode)
- **[SimCCL](../components/simccl.md)** — Collective communication transformation

---

## Prefill-Decode (PD) Disaggregation

The inference process is divided into two distinct phases:

| Phase | Characteristic | Description |
|-------|---------------|-------------|
| **Prefill** | Compute-intensive | Processes all input prompt tokens and generates the first output token |
| **Decode** | Memory-bandwidth-intensive | Autoregressively generates subsequent output tokens one at a time |

PD disaggregation allows deploying these phases on different GPU nodes, enabling:

- **Elastic resource allocation** — Prefill nodes can be configured with more compute, Decode nodes with more memory
- **Performance isolation** — Avoiding resource contention between phases
- **Flexible P:D ratio** — Configurable via `--replica_config_pd_node_ratio`

---

## Request Scheduling

The scheduling component is adapted from Microsoft's [Vidur](https://github.com/microsoft/vidur), supporting multiple strategies:

| Scheduler | Level | Description |
|-----------|-------|-------------|
| `split_wise` | Global | PD disaggregation-aware dispatch to Prefill and Decode replicas |
| `lor` | Global | Least Outstanding Requests — dispatch to the least-loaded replica |
| `round_robin` | Global | Round-robin dispatch |
| `sarathi` | Per-replica | Intra-replica batch scheduling |
| `split_wise` | Per-replica | Per-replica scheduling for PD disaggregation |

---

## Parallelism Strategies

Supports combinations of multiple parallelism strategies:

| Strategy | Flag | Description |
|----------|------|-------------|
| **Data Parallel (DP)** | `--cluster_config_num_replicas` | Number of replicas |
| **Tensor Parallel (TP)** | `--replica_config_tensor_parallel_size` | Intra-node parallelism |
| **Pipeline Parallel (PP)** | `--replica_config_num_pipeline_stages` | Inter-stage parallelism |
| **Expert Parallel (EP)** | `--replica_config_expert_model_parallel_size` | MoE expert parallelism |

Works for both dense and MoE (Mixture-of-Experts) models.

---

## Execution-Time Prediction Backends

| Backend | Flag Value | Description |
|---------|-----------|-------------|
| **AICB/AIOB** | `aicb` | Supports compute kernels and TP/DP/PP/EP communication size for DeepSeek-V3, Qwen3-MoE, Qwen3-Next |
| **SimAI Simulation** | `simai_simulation` | NS-3-based full-stack network simulation (currently supports TP) |
| **SimAI Analytical** | `simai_analytical` | Analytical performance model (currently supports TP) |
| **Native Vidur** | `vidur` | Original Vidur backend, supports TP, DP, PP |

Set via `--random_forrest_execution_time_predictor_config_backend`.

---

## Quick Start

### Prerequisites

- **AICB backend**: SimAI Docker environment with Hopper (SM90) or Blackwell (SM100) GPUs
- **SimAI backends**: Compile SimAI-Analytical or SimAI-Simulation first
- **Vidur backend**: Conda environment with profiling data

### Run with AICB Backend

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

### Run with SimAI Simulation Backend

```bash
cd SimAI

# Compile and generate topology
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

cd vidur-alibabacloud

python -m vidur.main \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --random_forrest_execution_time_predictor_config_backend simai_simulation \
  --random_forrest_execution_time_predictor_config_simai_dir ../ \
  --random_forrest_execution_time_predictor_config_simai_simulation_topo ../Spectrum-X_128g_8gps_100Gbps_A100 \
  --random_forrest_execution_time_predictor_config_simai_simulation_config ../astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### Run the 4-Scenario Test Suite

```bash
# Run all 4 pre-configured scenarios
bash examples/vidur-ali-scenarios/run_scenarios.sh --all

# Run a single scenario
bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

---

## 4-Scenario Configuration

**Shared Hardware**: H20 GPU (h20_dgx), NVLink 1600 Gbps, RDMA 800 Gbps, PD P2P 800 Gbps (fp8)

| Scenario | Model | PD Separation | World Size | TP | EP | Scheduler |
|----------|-------|---------------|------------|----|----|-----------|
| 1 | Qwen3-Next-80B | No | 32 (dp=32) | 1 | 1 | lor |
| 2 | Qwen3-Next-80B | Yes (P=2, D=6) | 8 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B | Yes (P=2, D=6) | 8 | 8 | 8 | split_wise |
| 4 | Qwen3-MoE-235B | Yes (P=2, D=6) | 8 | 4 | 4 | split_wise |

---

## Output

Each simulation run produces:

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # Per-request metrics
├── chrome_trace.json       # Chrome DevTools timeline trace
├── config.json             # Configuration snapshot
└── plots/                  # Per-metric CSV/JSON files
```

See [Result Analysis](result_analysis.md) for output interpretation.

---

## See Also

- [vidur-alibabacloud Component](../components/vidur.md) — Full inference simulation documentation
- [Supported Models](supported_models.md) — Model compatibility matrix
- [Result Analysis](result_analysis.md) — Output interpretation guide
