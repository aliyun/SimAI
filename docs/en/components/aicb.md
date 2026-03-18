# AICB — AI Communication Benchmark

**Repository**: [aliyun/aicb](https://github.com/aliyun/aicb) | **Language**: Python

AICB is a specialized communication benchmarking suite for AI scenarios. It generates realistic communication workloads aligned to real-world LLM training and inference processes.

---

## Introduction

AICB (Artificial Intelligence Communication Benchmark) produces communication workloads with precise patterns aligned to real-world applications. It supports:

- Benchmarking and tuning GPU cluster communication systems
- Investigating communication patterns of specific model configurations
- Generating workloads for simulators like SimAI

---

## Benchmark Suite

AICB provides 10 pre-configured benchmark cases covering typical LLM configurations:

| ID | Model | Seq Length | Framework | TP | PP | SP | MoE |
|----|-------|-----------|-----------|----|----|-----|-----|
| 1 | LLaMA-7B | 2048 | Megatron | 1 | 1 | - | - |
| 2 | GPT-13B | 2048 | Megatron | 2 | 1 | Yes | - |
| 3 | GPT-22B | 2048 | Megatron | 4 | 1 | - | - |
| 4 | LLaMA-65B | 4096 | Megatron | 8 | 2 | Yes | - |
| 5 | GPT-175B | 2048 | Megatron | 8 | 8 | Yes | - |
| 6 | GPT-175B | 2048 | Megatron | 8 | 8 | - | - |
| 7 | Llama3-405B | 8192 | Megatron | 8 | 16 | Yes | - |
| 8 | LLaMA-7B | 4096 | DeepSpeed | 1 | 1 | - | Zero-2 |
| 9 | LLaMA-65B | 4096 | DeepSpeed | 1 | 1 | - | Zero-3 |
| 10 | Mistral-8x7B | 2048 | Megatron | 2 | 1 | Yes | 8 experts |

---

## Environment Setup

### Docker

```bash
docker build -t aicb:v0.0.1 .
docker run --gpus all --net host --shm-size 16g -it --rm aicb:v0.0.1
```

### Local Environment

Requirements: Python >= 3.8, CUDA >= 11.8, PyTorch >= 2.0.0, NVIDIA APEX

### NGC Container

```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/aicb:/workspace/aicb nvcr.io/nvidia/pytorch:xx.xx-py3
```

> **Note**: Inference workload profiling requires NVIDIA Hopper (SM90) or Blackwell (SM100) GPUs.

---

## Physical Execution on GPU Clusters

### Environment Variables

| Parameter | Description |
|-----------|-------------|
| `nnodes` | Number of nodes |
| `node_rank` | Rank of the node |
| `nproc_per_node` | Number of GPUs per node |
| `master_addr` | Master node address |
| `master_port` | Master node port |

### Running Megatron Workloads

```bash
sh scripts/megatron_gpt.sh \
  --nnodes 1 --node_rank 0 --nproc_per_node 8 \
  --master_addr localhost --master_port 29500 \
  -m 7 --world_size 8 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --seq_length 2048 --swiglu --use_flash_attn --aiob_enable
```

### Running MoE Workloads

```bash
sh scripts/megatron_gpt.sh \
  -m moe --world_size 8 --tensor_model_parallel_size 4 \
  --moe_enable --expert_model_parallel_size 1 \
  --num_experts 4 --moe_router_topk 2 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --sp --grouped_gemm --aiob_enable --swiglu --use_flash_attn
```

### Running DeepSeek Workloads

```bash
sh scripts/megatron_gpt.sh \
  --frame DeepSeek -m deepseek \
  --tensor_model_parallel_size 4 --moe_enable \
  --expert_model_parallel_size 1 --num_experts 4 \
  --global_batch 4 --micro_batch 1 --world_size 4 \
  --num_layers 10 --sp --swiglu --aiob_enable
```

---

## Training Workload Generation

Generate workload files for SimAI simulation:

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu
```

Output saved in `results/mocked_workload/`.

---

## Inference Workload Generation

AICB generates inference workloads with prefill/decode phase separation for:

| Model | Attention | MoE Experts |
|-------|-----------|-------------|
| DeepSeek-V3-671B | MLA | 256 routed + 1 shared |
| Qwen3-MoE-235B | MHA/GQA | 128 routed |
| Qwen3-Next-80B | Hybrid (full + linear) | 512 routed |

Requires hardware-accelerated libraries: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

---

## AIOB: Computation Profiling

AIOB profiles actual GPU computation times and embeds them into workloads:

- `--aiob_enable` — Profile on current GPU
- `--comp_filepath <path>` — Use pre-existing profiling data

Output saved in `results/aiob_outputs/`.

---

## Custom Model Development

AICB supports creating workloads for custom model architectures using `MockedParam` and `MockedModel` base classes.

The training process is abstracted into: `init → forward → backward → step`

Each workload item consists of:
1. **Communication info**: `comm_type`, `comm_group`, `comm_group_size`, `msg_size`
2. **Additional info**: source node (broadcast), compute time
3. **Runtime info**: `elapsed_time`, `algo_bw`, `bus_bw`

Refer to existing `MockedMegatron` and `MockedDeepSpeed` implementations for examples.

---

## Key Parameters

| Category | Parameter | Description |
|----------|-----------|-------------|
| Framework | `frame` | Megatron / DeepSpeed / DeepSeek |
| Model | `model_size` | Pre-configured size (7/13/22/175/moe/deepseek) |
| Training | `world_size` | Total GPU count |
| | `global_batch` | Total batch size |
| | `micro_batch` | Micro-batch size |
| | `seq_length` | Sequence length |
| Parallelism | `tensor_model_parallel_size` | TP degree |
| | `pipeline_model_parallel` | PP degree |
| | `expert_model_parallel_size` | EP degree |
| MoE | `moe_enable` | Enable MoE |
| | `num_experts` | Number of experts |
| | `moe_router_topk` | Experts per token |
| DeepSeek | `qk_rope_dim` | RoPE dimension for QK |
| | `kv_lora_rank` | KV compression LoRA dimension |
| | `q_lora_rank` | Q compression LoRA dimension |
| | `n_shared_expert` | Number of shared experts |
| Optimization | `use_flash_attn` | FlashAttention |
| | `swiglu` | SwiGLU activation |
| | `aiob_enable` | AIOB compute profiling |
| | `comp_filepath` | Pre-existing computation file |

---

## Result Output

### Physical Execution

- Per-communication logs: type, group, message size, execution time, throughput
- Per-iteration timing analysis
- CSV outputs in `results/comm_logs/`

### Workload Files

- Training workloads: `results/mocked_workload/` or `results/workload/`
- AIOB profiles: `results/aiob_outputs/`

---

## See Also

- [Workload Generation Guide](../user_guide/workload_generation.md) — User-facing workload generation guide
- [Supported Models](../user_guide/supported_models.md) — Full model list
- [Tutorial](https://github.com/aliyun/aicb/blob/master/training/tutorial.md) — Detailed AICB tutorial
