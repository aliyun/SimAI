# AICB Workload Generator — CLAUDE.md

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
entries. Note: the current mock conservatively routes GatedDeltaNet through
ColumnLinear/RowLinear for accurate parameter counting; a future
communication-accurate mock would have GatedDeltaNet return empty workloads.

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

Message size formulas:
- EP dispatch: `seq_len * hidden_size * batch_size * topk / tp * 2` bytes
- TP all-gather: `2 * hidden_size * topk * batch_size * seq_len` bytes
  (NOTE: does not divide by ep_size — known conservative overestimate, same in
  Megatron and DeepSeek mocks. TODO in code acknowledges this.)

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

### Qwen3.5 GatedDeltaNet: Implementation Note

GatedDeltaNet is a linear attention mechanism (O(L) complexity):

```
S_t = S_{t-1} * α_t * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T
```

In a real Qwen3.5 model, all GatedDeltaNet operations are local compute
(causal conv1d, gated delta rule recurrence, output projection — no TP
collectives needed). However, the current mock implementation routes
GatedDeltaNet attention through the standard ColumnLinear/RowLinear
primitives for accurate parameter counting (DP gradient sync sizing).
This means GatedDeltaNet layers contribute 4 comm ops per layer — the same
as full-attention layers — rather than the 2 comm ops (MLP-only) that a
communication-accurate model would produce.

This results in a conservative overestimate of Qwen3.5 communication.
Verified end-to-end on 2025-06-15:

```
Qwen3.5-9B (32L, h=4096, TP=8):  130 fwd ops  (NOT 82)
Qwen3-8B   (36L, h=4096, TP=8):  146 fwd ops
Observed reduction: 11% (16 ops from 4 fewer layers), not 44% (64 ops)
```

Future work: implement a true communication-free GatedDeltaNet mock to
capture the real 44% reduction in attention communication.

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

2. **TP all-gather message size in MoE permutation** overestimates by `ep_size`
   factor. The formula `2 * hidden * topk * batch * seq` computes total tokens
   across all EP ranks, but after EP dispatch each rank holds ~1/ep of tokens.
   This is a pre-existing issue in both `MOEMLP` and `DeepSeekMoE` (noted by
   TODO comments). Not Qwen3-specific.

3. **GatedDeltaNet communication is conservatively overestimated.** The current
   mock routes GatedDeltaNet attention through ColumnLinear/RowLinear primitives
   (4 comm ops per layer). A real Qwen3.5 model would have only 2 comm ops
   (MLP-only) for GatedDeltaNet layers since all linear attention operations are
   local compute. Verified 2025-06-15: Qwen3.5-9B produces 130 fwd ops vs
   expected 82 with communication-free GatedDeltaNet. This is a deliberate
   trade-off for accurate DP gradient sync sizing. Future work: implement a
   communication-free GatedDeltaNet mock to capture the full 44% reduction.

4. **MoE backward communication was undercounted** (fixed 2025-06-15).
   MOEMLP.backward() in MockedMegatron.py was missing two `workloads.extend()`
   calls on the return values of `self.permutation()` and `self.unpermutation()`.
   This caused all MoE models (Megatron, Qwen3, Qwen3.5, DeepSeek) to report
   ~43-57% of the correct backward communication. The fix (2 lines) restored
   backward-forward parity for MoE layers. Pre-existing since original commit.

5. **Inference workload mocks for Qwen3/Qwen3.5 are skeletal.** Only the
   AIOB compute benchmarks exist for inference. Training workload mocks are
   complete.

6. **AIOB training benchmarks for Qwen3/Qwen3.5 do not exist.** Only inference
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

5. Test with config file and verify workload counts. The key sanity check:
   ```
   forward_items = layers * (attn_comm_per_layer + mlp_comm_per_layer) + embedding + lm_head
   ```
   where `attn_comm_per_layer` is 2 for Megatron/Qwen3 (all-gather + reduce-scatter),
   2 for Qwen3.5 full-attention layers, and 2 for Qwen3.5 GatedDeltaNet layers
   (current mock routes GatedDeltaNet through ColumnLinear/RowLinear for accurate
   parameter counting; see Implementation Note above). Future work: reduce
   GatedDeltaNet to 0 attn comms for communication-accurate modeling.
