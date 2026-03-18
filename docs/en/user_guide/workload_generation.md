# Workload Generation

AICB (AI Communication Benchmark) provides workload generation capabilities for both training and inference simulation in SimAI.

---

## Overview

AICB generates workload description files (`.txt`) that describe the communication and computation patterns of LLM training/inference processes. These workloads are consumed by SimAI's simulation engine.

Two types of workload generation are supported:

| Type | Description | Models Supported |
|------|-------------|------------------|
| **Training** | Generates training communication/computation patterns | GPT (7B/13B/22B/175B), LLaMA (7B/65B/405B), DeepSeek (16B/236B/671B), MoE |
| **Inference** | Generates prefill/decode phase workloads | DeepSeek-V3-671B, Qwen3-MoE-235B, Qwen3-Next-80B |

---

## Training Workload Generation

### Quick Start with Pre-configured Models

```bash
# Generate workload for Megatron GPT-7B
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
  --world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
  --frame Megatron --global_batch 8192 \
  --micro_batch 1 --seq_length 4096 --swiglu \
  --use_flash_attn --aiob_enable \
  --comp_filepath workload/aiob_inputs/Example.txt
```

Available pre-configured model sizes: `7`, `13`, `22`, `175` (GPT/LLaMA), `moe`, `deepseek` (16/236/671).

### Generating for Different Frameworks

#### Megatron

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu
```

#### MoE

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name MoE --frame=Megatron \
  --world_size=32 --tensor_model_parallel_size=4 --pipeline_model_parallel=1 \
  --expert_model_parallel_size=2 --moe_enable --num_experts=8 --moe_router_topk=2 \
  --global_batch=32 --micro_batch=1 --seq_length=2048 \
  --aiob_enable --swiglu --use_flash_attn
```

#### DeepSeek

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --frame=DeepSeek \
  --world_size=32 --tensor_model_parallel_size=4 \
  --expert_model_parallel_size=2 --moe_enable --num_experts=4 --moe_router_topk=2 \
  --global_batch=16 --micro_batch=1 --seq_length=4096 \
  --aiob_enable --swiglu -m deepseek
```

#### DeepSpeed

```bash
python -m workload_generator.generate_deepspeed_stage3_workload \
  --world_size=64 --global_batch=64 \
  --num_layers=40 --hidden_size=5120 --seq_length=4096 \
  --zero_stage=3 --reduce_bucket_size=1000000000
```

### Output

Generated workload files are saved in:
- Training: `results/mocked_workload/` or `results/workload/`

---

## Inference Workload Generation

SimAI uses AICB to generate inference workloads with prefill/decode phase separation.

> **Note**: Inference compute profiling requires NVIDIA Hopper (SM90) or Blackwell (SM100) GPUs due to dependencies on [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

### Supported Inference Models

| Model | Attention | MoE Experts | Experts/Token |
|-------|-----------|-------------|---------------|
| DeepSeek-V3-671B | MLA | 256 routed + 1 shared | 8 |
| Qwen3-MoE-235B | MHA/GQA | 128 routed | 8 |
| Qwen3-Next-80B | Hybrid (full + linear) | 512 routed | 10 |

Inference workloads are automatically generated and consumed by the vidur-alibabacloud scheduling framework. See [Inference Simulation](inference_simulation.md) for end-to-end usage.

---

## AIOB: Computation Time Embedding

AIOB (AI Operation Benchmark) is a sub-module within AICB that profiles actual GPU computation times and embeds them into workloads.

### Usage Options

| Option | Description |
|--------|-------------|
| `--aiob_enable` | Enable AIOB to profile computation times on the current GPU |
| `--comp_filepath <path>` | Use a pre-existing computation time description file |
| Neither | Use fixed default computation times |

### Example: Profile and Embed

```bash
sh scripts/megatron_gpt.sh \
  -m 7 --world_size 8 --tensor_model_parallel_size 2 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --seq_length 2048 --swiglu --use_flash_attn --aiob_enable
```

Computation description files are saved in `results/aiob_outputs/`.

---

## Key Parameters

| Category | Parameter | Description |
|----------|-----------|-------------|
| **Framework** | `--frame` | Megatron / DeepSpeed / DeepSeek |
| **Model** | `--model_size` or `-m` | Pre-configured model size |
| **Training** | `--world_size` | Total number of GPUs |
| | `--global_batch` | Total batch size |
| | `--micro_batch` | Micro-batch size |
| | `--seq_length` | Sequence length |
| | `--epoch_num` | Number of iterations |
| **Parallelism** | `--tensor_model_parallel_size` | TP degree |
| | `--pipeline_model_parallel` | PP degree |
| | `--expert_model_parallel_size` | EP degree |
| **MoE** | `--moe_enable` | Enable MoE |
| | `--num_experts` | Number of experts |
| | `--moe_router_topk` | Experts per token |
| **Optimization** | `--use_flash_attn` | Use FlashAttention |
| | `--swiglu` | Use SwiGLU activation |
| | `--aiob_enable` | Enable AIOB computation profiling |
| | `--comp_filepath` | Path to computation time file |

For the full parameter list, see the [AICB component documentation](../components/aicb.md) or the [CLI Reference](../technical_reference/cli_reference.md).

---

## See Also

- [AICB Component](../components/aicb.md) — Complete AICB documentation
- [Inference Simulation](inference_simulation.md) — End-to-end inference simulation guide
- [Supported Models](supported_models.md) — Full model compatibility list
