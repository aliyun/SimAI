# Supported Models

SimAI supports a range of LLM models for both training and inference simulation.

---

## Inference Models (SimAI 1.5+)

These models are supported for multi-request inference simulation via vidur-alibabacloud, with GPU memory calculation, PD disaggregation, and workload generation support.

### DeepSeek-V3-671B

| Attribute | Value |
|-----------|-------|
| **Total Layers** | 61 |
| **Attention Type** | MLA (Multi-head Latent Attention) |
| **Attention Heads** | 128 |
| **Hidden Size** | 7168 |
| **KV LoRA Rank** | 512 |
| **Q LoRA Rank** | 1536 |
| **QK RoPE Head Dim** | 64 |
| **QK NoPE Head Dim** | 128 |
| **V Head Dim** | 128 |
| **MoE Routed Experts** | 256 |
| **Experts Per Token** | 8 |
| **Shared Experts** | 1 |
| **Dense Layers** | First 3 layers (fixed activation of 8 routed + 1 shared expert) |
| **Sparse Layers** | Layers 3-60 (dynamically select 8 from 256 routed + 1 shared expert) |
| **Config File** | `vidur-alibabacloud/data/hf_configs/deepseek_v3_config.json` |

### Qwen3-MoE-235B

| Attribute | Value |
|-----------|-------|
| **Total Layers** | 94 |
| **Attention Type** | MHA / GQA |
| **Attention Heads** | 64 |
| **KV Heads** | 4 |
| **Hidden Size** | 4096 |
| **Head Dim** | 128 |
| **MoE Routed Experts** | 128 |
| **Experts Per Token** | 8 |
| **MoE Intermediate Size** | 1536 |
| **Config File** | `vidur-alibabacloud/data/hf_configs/qwen3_moe_config.json` |

### Qwen3-Next-80B

| Attribute | Value |
|-----------|-------|
| **Total Layers** | 48 |
| **Attention Type** | Hybrid (full + linear attention, alternating every 4 layers) |
| **Full Attention Heads** | 16 |
| **KV Heads** | 2 |
| **Hidden Size** | 2048 |
| **Head Dim** | 256 |
| **Linear Attention Key Heads** | 16 |
| **Linear Attention Value Heads** | 32 |
| **MoE Routed Experts** | 512 |
| **Experts Per Token** | 10 |
| **MoE Intermediate Size** | 512 |
| **Config File** | `vidur-alibabacloud/data/hf_configs/qwen3-next-80B-A3B_config.json` |

### Legacy Inference Models (via Vidur Backend)

These models are supported using the original Vidur profiling-based backend:

| Model | TP Support | PP Support |
|-------|-----------|-----------|
| meta-llama/Meta-Llama-3-8B | Yes | Yes |
| meta-llama/Meta-Llama-3-70B | Yes | Yes |
| meta-llama/Llama-2-7b-hf | Yes | Yes |
| meta-llama/Llama-2-70b-hf | Yes | Yes |
| codellama/CodeLlama-34b-Instruct-hf | Yes | Yes |
| internlm/internlm-20b | Yes | Yes |
| Qwen/Qwen-72B | Yes | Yes |

---

## Training Models (AICB)

The following models are supported for training workload generation:

### AICB Benchmark Suite

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

### Training Framework Support

| Framework | Models | AIOB Support |
|-----------|--------|-------------|
| **Megatron** | GPT, LLaMA, MoE | Yes |
| **DeepSpeed** | LLaMA (Zero Stage 1/2/3) | No (fixed times) |
| **DeepSeek** | DeepSeek (16B/236B/671B) | Yes |

---

## Attention Architecture Comparison

| Architecture | Model | KV Cache Strategy | Memory Efficiency |
|-------------|-------|-------------------|-------------------|
| **MLA** | DeepSeek-V3-671B | Compressed latent vector (`kv_lora_rank` + `qk_rope_head_dim`) | ~57x reduction vs MHA |
| **MHA / GQA** | Qwen3-MoE-235B | Standard KV cache (`num_kv_heads * head_dim`) | Standard |
| **Hybrid Full + Linear** | Qwen3-Next-80B | Full attention layers + linear (GDN) attention alternating every 4 layers | Reduced (linear layers have no KV cache) |

---

## Hardware Requirements

### Inference Profiling (AICB Backend)

| Requirement | Details |
|-------------|---------|
| **GPU Architecture** | NVIDIA Hopper (SM90) or Blackwell (SM100) |
| **Reason** | Dependency on DeepGEMM, FlashMLA, FlashInfer |
| **Hopper Note** | Add `ENV FLASH_MLA_DISABLE_SM100=1` to Dockerfile |

### Training Simulation

- **SimAI-Analytical**: Any CPU (no GPU required)
- **SimAI-Simulation**: Any CPU (no GPU required)
- **AICB Physical Execution**: Requires GPU cluster with NCCL support

---

## See Also

- [Inference Simulation](inference_simulation.md) — Multi-request inference guide
- [Workload Generation](workload_generation.md) — AICB workload generation
- [GPU Memory Module](../technical_reference/memory_module.md) — Memory calculation details
- [vidur-alibabacloud](../components/vidur.md) — Inference scheduling component
