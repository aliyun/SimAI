# Adding New Models

This guide covers how to add new model support to SimAI, including both the Vidur inference simulation side (GPU memory, profiling) and the AICB workload generation side.

---

## Overview

Adding a new model typically involves two components:

| Component | What to add | Required hardware |
|-----------|-------------|-------------------|
| **vidur-alibabacloud** | Model config, profiling data (compute + network) | GPU (for profiling only) |
| **AICB** | Workload generation parameters (`MockedParam` / `MockedModel`) | None |

---

## Part 1: Vidur — Model Configuration and Profiling

### Step 1: Add Model Configuration

Create a YAML/JSON model config in `vidur-alibabacloud/data/model_configs/` or `vidur-alibabacloud/data/hf_configs/`:

- Use the model's HuggingFace model ID as filename (e.g., `meta-llama/Llama-2-70b-hf.yml`)
- Reference the model's HuggingFace `config.json` for parameter values
- Ensure the correct parameters are set so the reference transformer model closely resembles the new model

**Example config parameters:**

```yaml
num_layers: 80
hidden_size: 8192
num_attention_heads: 64
num_key_value_heads: 8       # For GQA models
head_dim: 128
intermediate_size: 28672
vocab_size: 128256
max_position_embeddings: 8192
```

For MoE models, also include:

```yaml
num_routed_experts: 256
num_experts_per_tok: 8
num_shared_experts: 1
moe_intermediate_size: 2048
```

### Step 2: Profiling Data Structure

Profiling data is stored in `vidur-alibabacloud/data/profiling/`:

```
profiling/
├── compute/
│   ├── a100/
│   │   └── model-name/
│   │       ├── mlp.csv
│   │       └── attention.csv
│   └── h100/
│       └── model-name/
│           ├── mlp.csv
│           └── attention.csv
└── network/
    ├── a100_pair_nvlink/
    │   ├── allreduce.csv
    │   └── send_recv.csv
    └── h100_dgx/
        ├── allreduce.csv
        └── send_recv.csv
```

**Key distinction:**
- **Compute profiling**: Only GPU SKU matters (e.g., `a100`, `h100`), not network topology
- **Network profiling**: Network configuration matters (e.g., `a100_pair_nvlink` vs `a100_dgx`)

### Step 3: Compute Profiling (MLP)

Requires actual GPUs. 1 GPU is sufficient even for TP > 1.

```bash
# Install sarathi-serve (vidur branch) for profiling
# Then run MLP profiling:
python vidur/profiling/mlp/main.py \
    --models your-model/model-name \
    --num_gpus 4

# Copy output to data directory:
cp profiling_outputs/mlp/<timestamp>/your-model/model-name/mlp.csv \
   data/profiling/compute/<gpu_sku>/your-model/model-name/mlp.csv
```

### Step 4: Compute Profiling (Attention)

```bash
python vidur/profiling/attention/main.py \
    --models your-model/model-name \
    --num_gpus 4

# Copy output:
cp profiling_outputs/attention/<timestamp>/your-model/model-name/attention.csv \
   data/profiling/compute/<gpu_sku>/your-model/model-name/attention.csv
```

### Step 5: Network Profiling (if needed)

Network profiling is **model-independent** — same data works for all models on the same hardware configuration.

```bash
# AllReduce profiling (for TP):
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective all_reduce

# Send/Recv profiling (for PP, requires multi-node):
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective send_recv
```

**Available network device profiles:**
- `a100_pair_nvlink` — Azure Standard_NC96ads_A100_v4 (4x A100 PCIe + NVLink pairs)
- `h100_pair_nvlink` — Azure internal (4x H100 NVL + NVLink pairs)
- `a100_dgx` — A100 DGX (8x A100)
- `h100_dgx` — H100 DGX (8x H100)

---

## Part 2: AICB — Workload Generation

### Custom Model Parameters (MockedParam)

To add a new model for workload generation in AICB, create a `MockedParam` subclass:

```python
# In aicb/workload_generator/mocked_params/
class YourModelParam(MockedParam):
    def __init__(self):
        super().__init__()
        self.num_layers = 80
        self.hidden_size = 8192
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.ffn_hidden_size = 28672
        self.vocab_size = 128256
        self.seq_length = 8192
        # MoE parameters (if applicable)
        self.num_experts = 256
        self.topk = 8
        self.moe_intermediate_size = 2048
```

### Custom Model Workflow (MockedModel)

For full control over the workload generation process, create a `MockedModel` subclass that defines the compute and communication operations for each layer.

See [AICB Component Documentation](../components/aicb.md#custom-model-development) for detailed examples.

### Inference Workload Generation

For inference workloads with prefill/decode separation:

```bash
# Generate inference workload
python -m aicb.main \
    --model_name your-model-name \
    --workload_type inference \
    --num_prefill_tokens 1024 \
    --num_decode_tokens 128
```

---

## Part 3: GPU Memory Module

If your model uses a non-standard attention architecture, you may need to extend the `ParamCounter` in `vidur/utils/param_counter.py`:

1. Add attention parameter calculation for your architecture
2. Add KV cache per-token size calculation
3. Test with the MemoryPlanner to verify OOM detection works correctly

See [GPU Memory Module Technical Reference](../technical_reference/memory_module.md) for calculation formulas.

---

## Verification Checklist

- [ ] Model config file added to `data/model_configs/` or `data/hf_configs/`
- [ ] Compute profiling data (MLP + attention) added
- [ ] Network profiling data available for target hardware
- [ ] AICB `MockedParam` created (if workload generation needed)
- [ ] GPU memory calculation works correctly (ParamCounter + MemoryPlanner)
- [ ] End-to-end inference simulation produces reasonable results
- [ ] Documentation updated

---

## Related Documentation

- [vidur-alibabacloud Component](../components/vidur.md) — Full vidur documentation
- [AICB Component](../components/aicb.md) — AICB workload generation
- [GPU Memory Module](../technical_reference/memory_module.md) — Memory calculation formulas
- [Supported Models](../user_guide/supported_models.md) — Current model support status
