# AICB Workload Generator — CLAUDE.md

> **Research reference**: `RESEARCH.md` contains the definitive architecture
> comparison (LLaMA 3 vs Qwen3 vs Qwen3.5), verified model parameter tables
> for all 22 supported models, and the complete research methodology (28 sources
> across 5 sub-questions). This CLAUDE.md is the operational reference; consult
> RESEARCH.md for the "why" behind the architecture decisions.
>
> Primary external sources:
> - Qwen3 Technical Report: ar5iv.labs.arxiv.org/html/2505.09388 (140KB HTML)
> - Qwen3.5 official blog: qwen-ai.com/qwen-3-5 (38KB, March 2026)
> - Model configs: hf-mirror.com (16 models), modelscope.cn (1 gated model)
> - HF Transformers source: ghproxy.net (modeling_qwen3.py, modeling_qwen3_5.py,
>   modeling_llama.py, modular_qwen3.py)

## Architecture

The AICB workload generator has two independent layers that share model configs:

```
Layer 1: Workload Mock  (training/*.py, inference/*.py)
  └── Declarative communication trace: builds a list of LogItem objects
      describing TP/EP/DP/PP collective operations. No tensors allocated,
      no GPUs touched. Output: CSV consumed by the SimAI simulator.

Layer 2: AIOB Compute Benchmark  (Aiob*.py)
  └── Actual CUDA kernel timing: runs real PyTorch GEMMs, attention ops,
      MoE routing on GPU and records microsecond wall-clock times.
      Output: timing maps merged into the CSV for compute-aware simulation.
```

Both layers are dispatched by `generate_megatron_workload.py` based on `--frame`.

### Entry Point

```
python -m workload_generator.generate_megatron_workload \
  --frame <Megatron|DeepSeek|Qwen3|Qwen3.5> \
  [--config config.json] [--hidden_size 4096 ...]
```

Flow: `get_params()` parsers CLI args → `MegatronWorkload(args, model)` calls
`model.forward()` and `model.backward()` recursively → produces `Workload` object
→ dumped as CSV.

The `WorkloadGenerator` base class (`workload_generator.py`) handles rank mapping,
pipeline parallelism scheduling, and optimizer step collectives.

### How the Mock Works

Each module (attention, MLP, MoE) implements `forward()` and `backward()` that
return a `Workload` object (list of `LogItem`). A `LogItem` is declarative:

```python
LogItem(
    comm_type=CommType.all_gather,       # what kind of communication
    comm_group=CommGroup.tp_group,       # which process group
    comm_group_size=8,                   # how many ranks
    msg_size=67108864,                   # bytes transferred
    stage="forward.Qwen3ColumnLinear"
)
```

Modules compose recursively: `Qwen3Model.forward()` calls `embedding.forward()`,
then each `layer.forward()`, then `lm_head.forward()`. Each layer's `forward()`
calls `attention.forward()` then `mlp.forward()`. The result is a flat list of
all communication events across the entire forward/backward pass.

**Critical rule: the mock ONLY models events that cross GPU boundaries.**
Pure local compute (RMSNorm, SiLU, softmax, RoPE, QK-Norm, GatedDeltaNet
recurrence) produces ZERO LogItems. Only TP/EP/DP/PP collectives generate
entries. GatedDeltaNet layers return empty Workload -- all projections are
replicated (not TP-sharded); parameters are tracked via raw MockedParam for
DP gradient sync sizing.

---

## Supported Model Families

### Training Workload Mocks

| Frame Name | File | Status | Models |
|---|---|---|---|
| `Megatron` | `training/MockedMegatron.py` | COMPLETE | LLaMA 3.1 8B/70B/405B |
| `DeepSeek` | `training/MockedDeepSeek.py` | COMPLETE | DeepSeek V3 (671B MoE), V3.1 |
| `Qwen3` | `training/MockedQwen3.py` | COMPLETE | Dense 0.6B-32B, MoE 30B-A3B, 235B-A22B |
| `Qwen3.5` | `training/MockedQwen3_5.py` | COMPLETE | Dense 0.8B-27B, MoE 35B-A3B, 122B-A10B, 397B-A17B |

### Inference Workload Mocks

| File | Status |
|---|---|
| `inference/MockedDeepSeek.py` | COMPLETE |
| `inference/MockedQwen3Moe.py` | SKELETAL (no forward/backward) |
| `inference/MockedQwen3Next.py` | SKELETAL (no forward/backward) |

### AIOB Compute Benchmarks

| File | Status |
|---|---|
| `training/AiobMegatron.py` (1107L) | COMPLETE |
| `training/AiobDeepSeek.py` (917L) | COMPLETE |
| `inference/AiobDeepSeek.py` (495L) | COMPLETE |
| `inference/AiobQwen3Moe.py` (427L) | COMPLETE |
| `inference/AiobQwen3Next.py` (697L) | COMPLETE |

---

## Key Design Patterns

### The ColumnLinear / RowLinear Comm Abstraction

All TP communication is encapsulated in two dimension-agnostic classes:

- **ColumnLinear**: shards output across TP. Forward = all-gather input, then
  local matmul. Backward = local grad matmul, then reduce-scatter grad.
- **RowLinear**: shards input across TP. Forward = local matmul, then
  reduce-scatter output. Backward = all-gather grad, then local matmul.

These classes accept arbitrary `input_size` and `output_size` — the
communication message sizes are computed from `seq_len`, `batch_size`, and
`input_size` (for ColumnLinear all-gather) or `output_size` (for RowLinear
reduce-scatter). This makes them reusable across all architectures with zero
modification.

### MoE Communication Pattern

The `MOEMLP` class models the standard Megatron MoE pattern used by all
supported MoE models (Qwen3, Qwen3.5, DeepSeek, Megatron):

```
Forward:
  1. Shared expert MLP (if present): ColumnLinear all-gather + RowLinear reduce-scatter
  2. EP all-to-all dispatch: ship tokens to expert-owning ranks
  3. TP all-gather: gather full token batch within TP for grouped GEMM
  4. [expert FFN computation — not modeled]
  5. TP reduce-scatter: reduce partial expert outputs across TP
  6. EP all-to-all combine: ship results back to original token ranks

Backward: same operations in reverse with gradient data
```

Message size formulas (all divide by ep_size after dispatch, fixed 2025-06-15):
- EP dispatch: `seq_len * hidden_size * batch_size * topk / tp / ep * 2` bytes
- TP all-gather: `2 * hidden_size * topk * batch_size * seq_len / ep` bytes
- TP reduce-scatter: same as TP all-gather
- EP combine: same as EP dispatch

### Why Qwen3/Qwen3.5 Don't Need New MoE Classes

- **Qwen3 MoE**: 128 experts, top-8, NO shared experts. Uses stock `MOEMLP`
  directly — zero modifications. (Qwen3 Technical Report explicitly removed
  shared experts from Qwen2.5-MoE.)

- **Qwen3.5 MoE**: 256-512 experts, top-8/10, WITH shared experts (1
  always-active dense MLP). Uses `Qwen3_5MoEMLP` which extends the `MOEMLP`
  pattern with a `shared_expert` MegatronMlp instance.

### Qwen3 Attention vs Megatron Attention

The only code difference is in `__init__` dimension computation:

```
MegatronAttention:
  kv_channels = hidden_size // num_attention_heads
  query_projection = kv_channels * num_attention_heads     # = hidden_size
  kv_projection = kv_channels * num_attention_heads        # = hidden_size
  qkv_output = 3 * hidden_size

Qwen3Attention:
  head_dim = 128  (from config, fixed across all models)
  query_projection = head_dim * num_attention_heads        # may ≠ hidden_size!
  kv_projection = head_dim * num_key_value_heads           # GQA-aware!
  qkv_output = query_projection + 2 * kv_projection
```

`forward()` and `backward()` are byte-for-byte identical to MegatronAttention.
This correctly handles models where `num_heads * head_dim != hidden_size`
(e.g., Qwen3-4B: 32×128=4096 ≠ hidden=2560; Qwen3-0.6B: 16×128=2048 ≠ 1024).

### Qwen3.5 GatedDeltaNet: Communication-Accurate Mock

GatedDeltaNet is a linear attention mechanism (O(L) complexity):

```
S_t = S_{t-1} * α_t * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T
```

All GatedDeltaNet operations are local compute -- the QKVZ/BA input
projections and output projection are replicated across TP ranks (not
TP-sharded) because their parameter count is small relative to the MLP/MoE
that follows, and the recurrent state makes TP complex. The mock models
this correctly: `Qwen3_5GatedDeltaNet.forward()` returns `Workload()` (empty).

Raw `MockedParam` objects track full-size parameters for accurate DP gradient
sync sizing in the `step()` method. This achieves both communication accuracy
(zero TP collectives from DeltaNet layers) and parameter-count accuracy
(full-size weights counted for DP all-reduce sizing).

Verified end-to-end (2025-06-15):

```
Qwen3.5-9B (32L, h=4096, TP=8):  82 fwd ops   (24 DeltaNet x 2 MLP + 8 FullAttn x 4 + 2)
Qwen3-8B   (36L, h=4096, TP=8):  146 fwd ops  (36 layers x 4 + 2)
Observed reduction: 44% (64 fewer ops: 48 from DeltaNet + 16 from 4 fewer layers)
```

Per-layer comparison:

| Layer type | Count | Attn comms | MLP comms | Per-layer | Total |
|---|---|---|---|---|---|
| Qwen3-8B (all full-attn) | 36 | 2 | 2 | 4 | 144 |
| Qwen3.5 FullAttention | 8 | 2 | 2 | 4 | 32 |
| Qwen3.5 GatedDeltaNet | 24 | **0** | 2 | 2 | 48 |
| Embedding + LM head | 2 | -- | -- | 1 | 2 |
| **Qwen3.5-9B total** | | | | | **82** |

---

## Qwen3.5 Architecture Details

### Hybrid Layer Layout

```
full_attention_interval = 4 → pattern: [L, L, L, F, L, L, L, F, ...]

layer_types (from config.json):
  ["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]

Layer selection: (layer_id + 1) % full_attention_interval == 0 → Full Attention
```

### Full Attention Layer Features

- head_dim = 256 (2× Qwen3's 128)
- QK-Norm: per-head RMSNorm on Q and K before RoPE (same as Qwen3)
- partial_rotary_factor = 0.25 (only 64 of 256 dims receive RoPE)
- MRoPE: multimodal RoPE with mrope_section = [11, 11, 10]
- attn_output_gate = True: sigmoid gate on attention output
- Q projection doubled: half for query, half for gating signal

All these are local compute — zero communication impact.

### GatedDeltaNet Features

- linear_key_head_dim = 128, linear_value_head_dim = 128
- linear_num_key_heads = 16 (CONSTANT across all model sizes)
- linear_num_value_heads varies per model (16-64)
- linear_conv_kernel_dim = 4 (causal depthwise conv + SiLU)
- Uses RMSNormGated (not standard RMSNorm, not QK-Norm)

### MoE Differences from Qwen3

| Feature | Qwen3 MoE | Qwen3.5 MoE |
|---|---|---|
| Experts | 128 | 256-512 |
| Top-K | 8 | 8 or 10 |
| Shared experts | None (removed) | 1 (always active) |
| Per-expert FFN dim | intermediate_size | moe_intermediate_size |
| Shared expert FFN | N/A | shared_expert_intermediate_size |

---

## Verified Model Configs

### Qwen3 Dense

| Model | hidden | layers | Q heads | KV heads | intermediate | head_dim | tie_emb |
|---|---|---|---|---|---|---|---|
| 0.6B | 1024 | 28 | 16 | 8 | 3072 | 128 | true |
| 1.7B | 2048 | 28 | 16 | 8 | 6144 | 128 | true |
| 4B | 2560 | 36 | 32 | 8 | 9728 | 128 | true |
| 8B | 4096 | 36 | 32 | 8 | 12288 | 128 | false |
| 14B | 5120 | 40 | 40 | 8 | 17408 | 128 | false |
| 32B | 5120 | 64 | 64 | 8 | 25600 | 128 | false |

Common: vocab_size=151936, rope_theta=1M, max_position=40960, qk_norm=hardcoded,
use_sliding_window=false, attention_bias=false, rms_norm_eps=1e-6.

### Qwen3 MoE

| Model | hidden | layers | Q heads | KV heads | intermediate | experts | topk | shared |
|---|---|---|---|---|---|---|---|---|
| 30B-A3B | 2048 | 48 | 32 | 4 | 6144 | 128 | 8 | none |
| 235B-A22B | 4096 | 94 | 64 | 4 | 12288 | 128 | 8 | none |

### Qwen3.5 Dense

| Model | hidden | layers | Q heads | KV heads | intermediate | head_dim | full:lin | tie_emb |
|---|---|---|---|---|---|---|---|---|
| 0.8B | 1024 | 24 | 8 | 2 | 3584 | 256 | 6:18 | true |
| 2B | 2048 | 24 | 8 | 2 | 6144 | 256 | 6:18 | true |
| 4B | 2560 | 32 | 16 | 4 | 9216 | 256 | 8:24 | true |
| 9B | 4096 | 32 | 16 | 4 | 12288 | 256 | 8:24 | false |
| 27B | 5120 | 64 | 24 | 4 | 17408 | 256 | 16:48 | false |

Common: vocab_size=248320, rope_theta=10M, max_position=262144,
full_attention_interval=4, linear_key_head_dim=128, linear_value_head_dim=128,
linear_num_key_heads=16, linear_conv_kernel_dim=4, partial_rotary=0.25,
attn_output_gate=true, MRoPE mrope_section=[11,11,10].

### Qwen3.5 MoE

| Model | hidden | layers | Q heads | KV heads | experts | topk | moe_ffn | shared_ffn | full:lin |
|---|---|---|---|---|---|---|---|---|---|
| 35B-A3B | 2048 | 40 | 16 | 2 | 256 | 8 | 512 | 512 | 10:30 |
| 122B-A10B | 3072 | 48 | 32 | 2 | 256 | 8 | 1024 | 1024 | 12:36 |
| 397B-A17B | 4096 | 60 | 32 | 2 | 512 | 10 | 1024 | 1024 | 15:45 |

---

## Usage Examples

```bash
# Qwen3-8B dense, TP=8
python -m workload_generator.generate_megatron_workload \
  --frame Qwen3 --model_name Qwen3-8B \
  --hidden_size 4096 --num_hidden_layers 36 \
  --num_attention_heads 32 --num_key_value_heads 8 --head_dim 128 \
  --ffn_hidden_size 12288 --vocab_size 151936 \
  --world_size 8 --tensor_model_parallel_size 8 \
  --seq_length 4096 --micro_batch 2 \
  --enable_sequence_parallel --swiglu

# Qwen3.5-9B dense, TP=8
python -m workload_generator.generate_megatron_workload \
  --frame Qwen3.5 --model_name Qwen3.5-9B \
  --hidden_size 4096 --num_hidden_layers 32 \
  --num_attention_heads 16 --num_key_value_heads 4 --head_dim 256 \
  --ffn_hidden_size 12288 --vocab_size 248320 \
  --world_size 8 --tensor_model_parallel_size 8 \
  --seq_length 4096 --micro_batch 2 \
  --enable_sequence_parallel --swiglu

# Qwen3-235B-A22B MoE, TP=8, EP=8
python -m workload_generator.generate_megatron_workload \
  --frame Qwen3 --model_name Qwen3-235B-A22B \
  --hidden_size 4096 --num_hidden_layers 94 \
  --num_attention_heads 64 --num_key_value_heads 4 --head_dim 128 \
  --ffn_hidden_size 12288 --vocab_size 151936 \
  --world_size 64 --tensor_model_parallel_size 8 \
  --expert_model_parallel_size 8 \
  --num_experts 128 --moe_router_topk 8 \
  --moe_enable --enable_sequence_parallel --swiglu

# Qwen3.5-397B-A17B MoE via config file
python -m workload_generator.generate_megatron_workload \
  --frame Qwen3.5 \
  --config path/to/qwen3_5_397b_config.json \
  --world_size 128 --tensor_model_parallel_size 8 \
  --expert_model_parallel_size 16 \
  --moe_enable --enable_sequence_parallel --swiglu
```

### Config JSON Format

```json
{
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "moe_enable": false,
    "model_name": "Qwen3-8B"
}
```

For Qwen3.5, add: `"full_attention_interval": 4`, `"linear_key_head_dim": 128`,
`"linear_value_head_dim": 128`, `"linear_num_key_heads": 16`,
`"linear_num_value_heads": 32`, `"linear_conv_kernel_dim": 4`,
`"moe_intermediate_size": 1024`, `"shared_expert_intermediate_size": 1024`.

For HF Qwen3.5 multimodal format, `text_config` is unpacked automatically by
`Qwen3_5Params.__init__`.

---

## Testing

```bash
# Qwen3 mock smoke test (standalone)
cd aicb
.venv/bin/python workload_generator/mocked_model/training/MockedQwen3.py /path/to/config.json

# Qwen3.5 mock smoke test (standalone)
.venv/bin/python workload_generator/mocked_model/training/MockedQwen3_5.py /path/to/config.json

# Full workload generation (requires pandas, torch optional)
.venv/bin/python -m workload_generator.generate_megatron_workload \
  --frame Qwen3 --config config.json --workload_only
```

Config file keys use HuggingFace naming: `num_hidden_layers`, `intermediate_size`,
`vocab_size`. Training-specific keys (`tensor_model_parallel_size`, `seq_length`,
`micro_batch`) are layered on top by `Qwen3Params` / `Qwen3_5Params`.

---

## Known Limitations

1. **MTP (Multi-Token Prediction) not modeled.** Qwen3.5 has MTP heads after the
   backbone. These add ~2% extra compute but negligible communication. Documented
   as known gap.

2. **MoE backward communication was undercounted** (fixed 2025-06-15).
   MOEMLP.backward() in MockedMegatron.py was missing two `workloads.extend()`
   calls on the return values of `self.permutation()` and `self.unpermutation()`.
   Pre-existing since original commit. Fix restored backward-forward parity.

3. **Inference workload mocks for Qwen3/Qwen3.5 are skeletal.** Only the
   AIOB compute benchmarks exist for inference. Training workload mocks are
   complete.

4. **AIOB training benchmarks for Qwen3/Qwen3.5 do not exist.** Only inference
   AIOB benchmarks are implemented (`inference/AiobQwen3Moe.py`,
   `inference/AiobQwen3Next.py`). Training AIOB would require implementing
   backward-pass kernels not present in the inference benchmarks.

---

## Adding a New Model Family

1. Create `training/Mocked<Family>.py` with:
   - `<Family>Attention`: attention module (reuse ColumnLinear/RowLinear)
   - `<Family>Mlp`: SwiGLU MLP (reuse ColumnLinear/RowLinear)
   - `<Family>TransformerLayer`: attention + MLP composition
   - `<Family>Model`: embedding + N layers + lm_head assembly
   - `<Family>Params(MockedParamsBase)`: config loading with defaults

2. Add Qwen3-specific CLI args to `utils/utils.py` via a `get_<family>_params`
   function called from `get_params()`.

3. Add `"<Family>"` to `--frame` choices in `utils/utils.py`.

4. Add import and dispatch in `generate_megatron_workload.py`:
   ```python
   from workload_generator.mocked_model.training.Mocked<Family> import <Family>Model
   ...
   elif args.frame == "<Family>":
       model = <Family>Model(args)
   ```

5. Test with config file and verify workload counts. The definitive formulas
   (verified 2025-06-15 across all configs, TP=8, SP enabled):

   ```
   DENSE (TP only):
     Megatron:            fwd = 1 + L * 4         bwd = fwd * 1.50
       (MegatronModel.forward omits final_norm ColumnLinear; no +1 at end)
       Megatron-8B: 1 + 32*4 = 129

     Qwen3:               fwd = 1 + L * 4 + 1     bwd = fwd * 1.50
       (Qwen3Model.forward includes lm_head ColumnLinear; +1 at end)
       (4 per layer: attn QKV col + attn out row + MLP gate_up col + MLP down row)
       Qwen3-8B: 1 + 36*4 + 1 = 146

     Qwen3.5:             fwd = 1 + L_f * 4 + L_d * 2 + 1     bwd = fwd * 1.50
       L_f = full-attention layers (L // full_attention_interval)
       L_d = DeltaNet layers (L - L_f)
       Qwen3.5-9B: 1 + 8*4 + 24*2 + 1 = 82

   MOE (TP + EP, no shared experts):
     Megatron:            fwd = 1 + L * 7         bwd ≈ fwd
     Qwen3:               fwd = 1 + L * 7 + 1     bwd ≈ fwd
       (7 per layer: 2 attn + 5 MoE dispatch/combine; MoE=preprocess+dispatch+perm+unperm+combine)
       Qwen3-235B: 1 + 94*7 + 1 = 660

   MOE (TP + EP, with shared experts — Qwen3.5):
     fwd = 1 + L_f * 9 + L_d * 7 + 1     bwd ≈ fwd * 1.04
       FullAttn layers: 9 (2 attn + 2 shared expert + 5 MoE)
       DeltaNet layers: 7 (0 attn + 2 shared expert + 5 MoE)
       Qwen3.5-397B: 1 + 15*9 + 45*7 + 1 = 452
   ```

   MoE per-layer breakdown (5 ops): preprocess all-gather + dispatch all-to-all
   + TP all-gather + TP reduce-scatter + combine all-to-all.
   Shared expert adds 2 ops: ColumnLinear all-gather + RowLinear reduce-scatter.
   All MoE msg_size formulas divide by ep_size (fixed 2025-06-15).
