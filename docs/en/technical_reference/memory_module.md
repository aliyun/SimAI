# GPU Memory Calculation Module

The GPU memory calculation module (introduced in SimAI 1.6) provides accurate GPU memory estimation for inference simulation, covering model parameter memory, KV cache memory, and maximum batch size calculation.

---

## Architecture

```
ParamCounter (param_counter.py)
    |-- Computes per-layer, per-device parameter counts
    |-- Returns (total_params, prefill_params, decode_params) under PD
    |
MemoryPlanner (memory_planner.py)
    |-- Plans GPU memory budget
    |-- Computes KV cache capacity
    |-- Detects OOM conditions
    |
Replica KV Cache Tracker (replica.py)
    |-- Per-request allocation/release
    |-- Runtime capacity queries
```

---

## ParamCounter

**File**: `vidur-alibabacloud/vidur/utils/param_counter.py`

### MLA Parameters (DeepSeek-V3-671B)

Per-layer MLA parameter components:

| Component | Formula | DeepSeek-V3 Value |
|-----------|---------|-------------------|
| Q LoRA down-projection | `hidden_size * q_lora_rank` | 7168 * 1536 |
| Q LoRA up-projection | `q_lora_rank * num_heads * qk_head_dim` | 1536 * 128 * 192 |
| KV LoRA down-projection | `hidden_size * kv_lora_rank` | 7168 * 512 |
| KV LoRA up-projection | `kv_lora_rank * num_heads * (qk_nope_dim + v_head_dim)` | 512 * 128 * 256 |
| Output projection | `hidden_size * num_heads * v_head_dim` | 7168 * 128 * 128 |

Where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`

### MHA/GQA Parameters (Qwen3-MoE-235B)

```
wq = hidden_size * num_attention_heads * head_dim
wk = hidden_size * num_key_value_heads * head_dim
wv = hidden_size * num_key_value_heads * head_dim
wo = hidden_size * num_attention_heads * head_dim
total = (wq + wk + wv + wo) * bytes_per_element
```

### Linear Attention Parameters (Qwen3-Next-80B)

Qwen3-Next-80B uses hybrid attention: full attention and linear (GDN) attention alternating every 4 layers. Linear attention layers use independent `linear_key_head_dim` / `linear_num_key_heads` configurations.

### MoE Expert Parameters

Per-expert FFN (3 weight matrices W1, W2, W3):

```
expert_params = 3 * hidden_size * moe_intermediate_size * bytes_per_element
```

### PD Disaggregation

Under PD disaggregation, expert parallelism differs between clusters:

- **Prefill cluster**: `experts_per_device = num_routed_experts / prefill_world_size`
- **Decode cluster**: `experts_per_device = num_routed_experts / decode_world_size`

Returns triple: `(total_params, prefill_params, decode_params)`

---

## KV Cache Calculation

### MHA/GQA KV Cache

```
kv_cache_per_token = 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
```

Factor of 2 = K (Key) + V (Value) caches.

### MLA KV Cache (DeepSeek-V3-671B)

MLA uses compressed KV representations — a single latent vector encoding both K and V:

```
kv_cache_per_token = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
```

Where `kv_lora_rank = 512`, `qk_rope_head_dim = 64`.

**Comparison**: MHA would need `2 * 128 * 128 = 32768` elements per token. MLA needs only `576` elements — a **~57x reduction**.

### Per-Request KV Cache Tracking

The `Replica` entity maintains:

| State | Description |
|-------|-------------|
| `_allocated_kv_cache_memory` | Currently allocated KV cache (bytes) |
| `_max_kv_cache_memory` | Maximum KV cache capacity |
| `_kv_cache_allocation_map` | Per-request allocation mapping |

Operations:
- `allocate_request_kv_cache_memory(request, num_blocks, block_size)`
- `release_request_kv_cache_memory(request)`
- `get_remaining_kv_cache_capacity()`

---

## MemoryPlanner

**File**: `vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`

### Calculation Flow

1. **Available GPU memory**: `available = total_GPU_memory * (1 - memory_margin_fraction)`
2. **Parameter memory**: Via ParamCounter; PD returns `(total, prefill, decode)`
3. **KV cache budget**: `kv_available = available - param_memory`
4. **Max concurrent requests**: `max_requests = kv_available / kv_cache_per_request`

### PD Disaggregation

- Prefill replicas: use `prefill_param_mem` for budget
- Decode replicas: use `decode_param_mem` for budget

### OOM Detection

When `param_memory > available_memory`, outputs error with suggestions:
- Increase TP/EP size
- Use larger GPU (more VRAM)
- Enable FP8 quantization

---

## Quantization Support

| Precision | Bytes per Element | Use Case |
|-----------|-------------------|----------|
| FP32 | 4 | Reference |
| FP16/BF16 | 2 | Default inference |
| FP8 | 1 | Reduced memory, supported by ParamCounter |

---

## See Also

- [vidur-alibabacloud Component](../components/vidur.md) — Full component documentation
- [Supported Models](../user_guide/supported_models.md) — Model specifications
- [SimAI 1.6 Tech Report](../../SimAI_1.6_Tech_Report.md) — Detailed technical report
