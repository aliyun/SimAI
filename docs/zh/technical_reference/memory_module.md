# GPU 显存计算模块

GPU 显存计算模块（SimAI 1.6 引入）为推理仿真提供精确的 GPU 显存估算，涵盖模型参数显存、KV Cache 显存和最大批量大小计算。

---

## 架构

```
ParamCounter (param_counter.py)
    |-- 按层、按设备计算参数量
    |-- PD 分离下返回 (total_params, prefill_params, decode_params)
    |
MemoryPlanner (memory_planner.py)
    |-- 规划 GPU 显存预算
    |-- 计算 KV Cache 容量
    |-- 检测 OOM 条件
    |
Replica KV Cache Tracker (replica.py)
    |-- 按请求分配/释放
    |-- 运行时容量查询
```

---

## ParamCounter

**文件**: `vidur-alibabacloud/vidur/utils/param_counter.py`

### MLA 参数（DeepSeek-V3-671B）

每层 MLA 参数组成：

| 组件 | 公式 | DeepSeek-V3 值 |
|-----------|---------|-------------------|
| Q LoRA 下投影 | `hidden_size * q_lora_rank` | 7168 * 1536 |
| Q LoRA 上投影 | `q_lora_rank * num_heads * qk_head_dim` | 1536 * 128 * 192 |
| KV LoRA 下投影 | `hidden_size * kv_lora_rank` | 7168 * 512 |
| KV LoRA 上投影 | `kv_lora_rank * num_heads * (qk_nope_dim + v_head_dim)` | 512 * 128 * 256 |
| 输出投影 | `hidden_size * num_heads * v_head_dim` | 7168 * 128 * 128 |

其中 `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`

### MHA/GQA 参数（Qwen3-MoE-235B）

```
wq = hidden_size * num_attention_heads * head_dim
wk = hidden_size * num_key_value_heads * head_dim
wv = hidden_size * num_key_value_heads * head_dim
wo = hidden_size * num_attention_heads * head_dim
total = (wq + wk + wv + wo) * bytes_per_element
```

### 线性注意力参数（Qwen3-Next-80B）

Qwen3-Next-80B 使用混合注意力：全注意力和线性（GDN）注意力每 4 层交替。线性注意力层使用独立的 `linear_key_head_dim` / `linear_num_key_heads` 配置。

### MoE 专家参数

每专家 FFN（3 个权重矩阵 W1、W2、W3）：

```
expert_params = 3 * hidden_size * moe_intermediate_size * bytes_per_element
```

### PD 分离

PD 分离下，专家并行在不同集群间有差异：

- **Prefill 集群**: `experts_per_device = num_routed_experts / prefill_world_size`
- **Decode 集群**: `experts_per_device = num_routed_experts / decode_world_size`

返回三元组：`(total_params, prefill_params, decode_params)`

---

## KV Cache 计算

### MHA/GQA KV Cache

```
kv_cache_per_token = 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
```

因子 2 = K（Key）+ V（Value）缓存。

### MLA KV Cache（DeepSeek-V3-671B）

MLA 使用压缩的 KV 表示——单个潜向量同时编码 K 和 V：

```
kv_cache_per_token = (kv_lora_rank + qk_rope_head_dim) * num_layers * bytes_per_element
```

其中 `kv_lora_rank = 512`、`qk_rope_head_dim = 64`。

**对比**：MHA 每 Token 需要 `2 * 128 * 128 = 32768` 个元素。MLA 仅需 `576` 个元素——**约 57 倍压缩**。

### 按请求 KV Cache 追踪

`Replica` 实体维护：

| 状态 | 说明 |
|-------|-------------|
| `_allocated_kv_cache_memory` | 当前已分配的 KV Cache（字节） |
| `_max_kv_cache_memory` | 最大 KV Cache 容量 |
| `_kv_cache_allocation_map` | 按请求的分配映射 |

操作：
- `allocate_request_kv_cache_memory(request, num_blocks, block_size)`
- `release_request_kv_cache_memory(request)`
- `get_remaining_kv_cache_capacity()`

---

## MemoryPlanner

**文件**: `vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py`

### 计算流程

1. **可用 GPU 显存**: `available = 总GPU显存 * (1 - memory_margin_fraction)`
2. **参数显存**: 通过 ParamCounter 计算；PD 返回 `(total, prefill, decode)`
3. **KV Cache 预算**: `kv_available = available - param_memory`
4. **最大并发请求**: `max_requests = kv_available / kv_cache_per_request`

### PD 分离

- Prefill 副本：使用 `prefill_param_mem` 计算预算
- Decode 副本：使用 `decode_param_mem` 计算预算

### OOM 检测

当 `param_memory > available_memory` 时，输出错误并给出建议：
- 增加 TP/EP 度
- 使用更大 GPU（更多显存）
- 启用 FP8 量化

---

## 量化支持

| 精度 | 每元素字节数 | 使用场景 |
|-----------|-------------------|----------|
| FP32 | 4 | 参考基准 |
| FP16/BF16 | 2 | 默认推理 |
| FP8 | 1 | 降低显存，ParamCounter 支持 |

---

## 相关文档

- [vidur-alibabacloud 组件](../components/vidur.md) — 完整组件文档
- [支持的模型](../user_guide/supported_models.md) — 模型规格
- [SimAI 1.6 技术报告](../../SimAI_1.6_Tech_Report.md) — 详细技术报告
