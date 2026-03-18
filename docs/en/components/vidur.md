# vidur-alibabacloud

**vidur-alibabacloud** is the LLM inference simulation component of SimAI, adapted from Microsoft's [Vidur](https://github.com/microsoft/vidur). It provides multi-request inference scheduling, GPU memory management, and Prefill-Decode (PD) disaggregation support.

- **Repository**: In-tree (`vidur-alibabacloud/`)
- **Language**: Python
- **License**: MIT

---

## Key Features

- **Prefill-Decode (PD) Separation** — Run prefill and decode stages on different nodes for elastic resource allocation and performance isolation. Inspired by [splitwise-sim](https://github.com/Mutinifni/splitwise-sim).
- **Flexible Parallelism** — Data Parallel (DP), Tensor Parallel (TP), Pipeline Parallel (PP), Expert Parallel (EP)
- **Multiple Execution Backends** — AICB/AIOB, SimAI Simulation (NS-3), SimAI Analytical, Native Vidur
- **Workload Generation & Replay** — Synthetic (fixed/Poisson) or real-trace request replay
- **Fine-Grained Metrics** — TTFT, TBT/TPOT, E2E latency, communication cost, compute cost, scheduling delay

---

## GPU Memory Calculation Module

This module provides accurate GPU memory estimation for MoE models during inference.

### Components

| Component | File | Description |
|-----------|------|-------------|
| **ParamCounter** | `vidur/utils/param_counter.py` | Per-layer and per-device parameter counting for MLA, MHA/GQA, linear attention, and MoE experts. Returns `(total_params, prefill_params, decode_params)` under PD disaggregation |
| **MemoryPlanner** | `vidur/scheduler/utils/memory_planner.py` | Plans GPU memory budget: `available = GPU_mem * (1 - margin) - param_mem`, computes KV cache capacity and max concurrent requests. Includes OOM detection |
| **Per-request KV Cache Tracking** | `vidur/entities/replica.py` | Allocates/releases KV cache memory per request, enabling runtime remaining-capacity queries |

### Supported Attention Architectures

| Architecture | Model | Description |
|---|---|---|
| **MLA** (Multi-head Latent Attention) | DeepSeek-V3-671B | LoRA-compressed KV cache (`kv_lora_rank` + `qk_rope_head_dim`), ~57x memory savings vs MHA |
| **MHA / GQA** | Qwen3-MoE-235B | Standard KV cache with `num_kv_heads * head_dim` per token per layer |
| **Hybrid Full + Linear Attention** | Qwen3-Next-80B | Alternates between full attention and linear (GDN) attention every 4 layers |

---

## Supported Models

| Model | Attention | Experts | Status |
|-------|-----------|---------|--------|
| DeepSeek-V3-671B | MLA | 256 routed + 1 shared | PP/EP adaptation in progress |
| Qwen3-MoE-235B | MHA/GQA | 128 routed | PP/EP adaptation in progress |
| Qwen3-Next-80B | Hybrid | 512 routed | PP/EP adaptation in progress |
| Meta-Llama-3-8B / 70B | MHA | Dense | Supported |
| Llama-2-7b / 70b | MHA | Dense | Supported |
| CodeLlama-34b | MHA | Dense | Supported |
| InternLM-20B | MHA | Dense | Supported |
| Qwen-72B | MHA | Dense | Supported |

---

## Environment Setup

### Docker (Recommended)

```bash
docker build -t simai:latest .
docker run --gpus all -it --rm simai:latest
```

> Add `ENV FLASH_MLA_DISABLE_SM100=1` to Dockerfile when using Hopper GPUs.

### Conda

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt
```

---

## Key Input Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--replica_config_model_name` | HuggingFace model ID or config path | Required |
| `--cluster_config_num_replicas` | Number of replicas (DP) | 1 |
| `--replica_config_tensor_parallel_size` | TP degree | 1 |
| `--replica_config_num_pipeline_stages` | PP stages | 1 |
| `--replica_config_expert_model_parallel_size` | EP degree | 1 |
| `--replica_config_pd_node_ratio` | P:D node ratio (e.g., `"2:6"`) | `""` (no PD) |
| `--cluster_config_global_scheduler_type` | Global scheduler: `lor` / `round_robin` / `split_wise` | `lor` |
| `--cluster_config_replica_scheduler_type` | Per-replica scheduler: `sarathi` / `split_wise` | `sarathi` |
| `--request_generator_config_type` | `synthetic` / `trace_replay` | `synthetic` |
| `--synthetic_request_generator_config_num_requests` | Number of requests to generate | 100 |
| `--poisson_request_generator_config_qps` | Queries per second (Poisson mode) | 1.0 |
| `--replica_config_device` | GPU type (e.g., `h20_dgx`) | Required |
| `--replica_config_network_device` | Network type | Same as device |
| `--execution_time_predictor_config_type` | Backend: `aicb` / `simai_simulation` / `simai_analytical` / `random_forrest` | `random_forrest` |
| `--nvlink_bandwidth_gbps` | NVLink bandwidth | 1600 |
| `--rdma_bandwidth_gbps` | RDMA bandwidth | 800 |
| `--pd_p2p_bandwidth_gbps` | PD inter-node P2P bandwidth | 800 |
| `--replica_config_fp8_enabled` | Enable FP8 quantization | false |
| `--replica_config_memory_margin_fraction` | GPU memory safety margin | 0.1 |

---

## Output Files

Each run produces the following outputs:

| File | Description |
|------|-------------|
| `request_metrics.csv` | Per-request metrics with 17 columns |
| `chrome_trace.json` | Timeline trace for Chrome `chrome://tracing` visualization |
| `config.json` | Configuration snapshot |
| `plots/` | Metric visualization plots |

### request_metrics.csv Columns

| Column | Description |
|--------|-------------|
| `request_id` | Unique request identifier |
| `arrived_at` | Request arrival time |
| `scheduled_at` | First schedule time |
| `completed_at` | Request completion time |
| `prefill_completed_at` | Prefill completion time (first token) |
| `num_prefill_tokens` | Number of input tokens |
| `num_decode_tokens` | Number of generated tokens |
| `scheduling_delay` | Wait time before scheduling |
| `e2e_time` | End-to-end latency |
| `e2e_time_normalized` | E2E latency / num_decode_tokens |
| `execution_time` | Actual GPU execution time |
| `preemption_time` | Time spent preempted |
| `num_restarts` | Number of restarts |
| `prefill_e2e_time` | TTFT (Time to First Token) |
| `decode_time_normalized` | Average TBT (Time Between Tokens) |
| `total_comm_cost` | Total communication time |
| `total_compute_cost` | Total compute time |

---

## Simulation Metrics (23 Items)

The simulator logs the following metrics (see `vidur-alibabacloud/docs/metrics.md` for details):

1. `request_inter_arrival_delay_histogram` — Request inter-arrival delay distribution
2. `request_num_tokens_histogram` — Token count distribution (prefill + decode)
3. `request_num_restarts_histogram` — Restart count distribution
4. `request_e2e_time_cdf` — End-to-end latency CDF
5. `request_e2e_time_normalised_cdf` — Normalized E2E latency CDF
6. `request_execution_plus_preemption_times_cdf` — Execution + preemption time CDF
7. `request_scheduling_delay_cdf` — Scheduling delay CDF
8. `request_execution_time_cdf` — Pure execution time CDF
9. `request_preempted_time_cdf` — Preemption time CDF
10. `decode_token_execution_plus_preemption_times` — Per-token inter-token delay CDF
11. `batch_num_tokens_cdf` — Batch total token count CDF
12. `batch_sizes_cdf` — Batch size CDF
13. `prefill_time_e2e_cdf` — TTFT CDF
14. `prefill_time_execution_plus_preemption_cdf` — Prefill processing time CDF
15. `prefill_time_execution_plus_preemption_normalized_cdf` — Normalized prefill time CDF
16. `decode_time_execution_plus_preemption_normalized_cdf` — Normalized decode time CDF
17. `request_completions_time_series` — Request completion time series
18. `prefill_completions_time_series` — Prefill completion time series
19. `decode_completions_time_series` — Decode completion time series
20. `replica_{id}_memory_usage_weighted_mean` — Per-replica memory utilization
21. `replica_{id}_stage_{id}_busy_time_percent_weighted_mean` — Per-stage busy time percentage
22. `replica_{id}_stage_{id}_mfu_weighted_mean` — Per-stage MFU
23. `request_arrivals_time_series` — Request arrival time series

---

## 4-Scenario Test Suite

Run all pre-configured scenarios:

```bash
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all
# Or a single scenario:
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 3
```

For detailed scenario configuration, see [Benchmarking — Test Suite](../benchmarking/test_suite.md).

---

## Adding New Models

To add a new model to vidur-alibabacloud, see the [Adding Models Guide](../developer_guide/adding_models.md) and the upstream documentation at `vidur-alibabacloud/docs/profiling.md`.

---

## Related Documentation

- [Inference Simulation User Guide](../user_guide/inference_simulation.md) — End-to-end inference simulation workflow
- [Result Analysis](../user_guide/result_analysis.md) — How to interpret output files
- [GPU Memory Module Technical Reference](../technical_reference/memory_module.md) — Detailed memory calculation formulas
- [Benchmarking Test Suite](../benchmarking/test_suite.md) — 4-scenario configuration details
