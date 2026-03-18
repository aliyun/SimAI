# CLI Reference

Complete command-line parameter reference for all SimAI tools.

---

## SimAI-Analytical

**Binary**: `bin/SimAI_analytical`

### Required Parameters

| Flag | Long Form | Description |
|------|-----------|-------------|
| `-w` | `--workload` | Path to workload file |
| `-g` | `--gpus` | Simulation GPU scale |
| `-g_p_s` | `--gpus-per-server` | Scale-up size (GPUs per server) |
| `-r` | `--result` | Output file path and prefix (default: `./results/`) |
| `-busbw` | `--bus-bandwidth` | Path to busbw.yaml file |

### Optional Parameters

| Flag | Long Form | Description |
|------|-----------|-------------|
| `-v` | `--visual` | Generate visualization files |
| `-dp_o` | `--dp-overlap-ratio` | DP overlap ratio [0.0-1.0] |
| `-ep_o` | `--ep-overlap-ratio` | EP overlap ratio [0.0-1.0] |
| `-tp_o` | `--tp-overlap-ratio` | TP overlap ratio [0.0-1.0] |
| `-pp_o` | `--pp-overlap-ratio` | PP overlap ratio [0.0-1.0] |

### Auto Busbw Calculation

| Flag | Description |
|------|-------------|
| `-nv` | NVLink bandwidth (GB/s) |
| `-nic` | NIC bandwidth (GB/s) |
| `-n_p_s` | NICs per server |

---

## SimAI-Simulation

**Binary**: `bin/SimAI_simulator`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AS_LOG_LEVEL` | Log level: DEBUG/INFO/WARNING/ERROR | `INFO` |
| `AS_PXN_ENABLE` | Enable PXN | `0` |
| `AS_NVLS_ENABLE` | Enable NVLS | `0` |
| `AS_SEND_LAT` | Send latency (us) | `6` |
| `AS_NVLSTREE_ENABLE` | Enable NVLS Tree | `false` |

### Parameters

| Flag | Long Form | Description | Default |
|------|-----------|-------------|---------|
| `-t` | `--thread` | Number of threads | `1` |
| `-w` | `--workload` | Path to workload | Required |
| `-n` | `--network-topo` | Topology file path | Required |
| `-c` | `--config` | SimAI.conf path | Required |

---

## SimAI-Physical

**Binary**: `bin/SimAI_phynet`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hostlist` | Path to host IP list | Required |
| `-w` / `--workload` | Workload file path | `./microAllReduce.txt` |
| `-i` / `--gid_index` | GID index for RDMA | `0` |
| `-g` / `--gpus` | Number of GPUs | `8` |

---

## Topology Generator

**Script**: `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py`

| Level | Flag | Description |
|-------|------|-------------|
| Global | `-topo` | Template: Spectrum-X / AlibabaHPN / DCN+ |
| | `-g` | Number of GPUs |
| | `--dp` | Enable dual plane |
| | `--ro` | Enable rail-optimized |
| | `--dt` | Enable dual ToR |
| | `-er` | Error rate |
| Intra-Host | `-gps` | GPUs per server |
| | `-gt` | GPU type (A100/H100) |
| | `-nsps` | NV switches per server |
| | `-nvbw` | NVLink bandwidth |
| | `-nl` | NVLink latency |
| | `-l` | NIC latency |
| Intra-Segment | `-bw` | NIC to ASW bandwidth |
| | `-asw` | ASW switch count |
| | `-nps` | NICs per switch |
| Intra-Pod | `-psn` | PSW switch count |
| | `-apbw` | ASW to PSW bandwidth |
| | `-app` | ASW per PSW |

---

## AICB Workload Generator

**Script**: `scripts/megatron_workload_with_aiob.sh` or `python -m workload_generator.SimAI_training_workload_generator`

### Core Parameters

| Parameter | Description |
|-----------|-------------|
| `--frame` | Framework: Megatron / DeepSpeed / DeepSeek |
| `-m` / `--model_size` | Model size: 7/13/22/175/moe/deepseek |
| `--world_size` | Total GPU count |
| `--global_batch` | Total batch size |
| `--micro_batch` | Micro-batch size |
| `--seq_length` | Sequence length |
| `--epoch_num` | Number of iterations |

### Parallelism Parameters

| Parameter | Description |
|-----------|-------------|
| `--tensor_model_parallel_size` | TP degree |
| `--pipeline_model_parallel` | PP degree |
| `--expert_model_parallel_size` | EP degree |
| `--enable_sequence_parallel` | Enable SP |

### Model Parameters

| Parameter | Description |
|-----------|-------------|
| `--num_layers` | Transformer layers |
| `--hidden_size` | Hidden size |
| `--num_attention_heads` | Attention heads |
| `--ffn_hidden_size` | FFN hidden size |
| `--vocab_size` | Vocabulary size |

### MoE Parameters

| Parameter | Description |
|-----------|-------------|
| `--moe_enable` | Enable MoE |
| `--num_experts` | Number of experts |
| `--moe_router_topk` | Experts per token |
| `--moe_grouped_gemm` | Enable grouped GEMM |

### DeepSeek Parameters

| Parameter | Description |
|-----------|-------------|
| `--qk_rope_dim` | RoPE dimension for QK |
| `--qk_nope_dim` | Non-RoPE dimension for QK |
| `--q_lora_rank` | Q LoRA rank |
| `--kv_lora_rank` | KV LoRA rank |
| `--v_head_dim` | V head dimension |
| `--n_shared_expert` | Shared experts per MoE layer |
| `--n_dense_layer` | Dense layers count |

### Optimization Parameters

| Parameter | Description |
|-----------|-------------|
| `--use_flash_attn` | FlashAttention |
| `--swiglu` | SwiGLU activation |
| `--aiob_enable` | AIOB computation profiling |
| `--comp_filepath` | Pre-computed times file |

---

## Vidur Inference Simulation

**Command**: `python -m vidur.main`

Run `python -m vidur.main -h` for the full parameter list. Key parameters are documented in the [vidur component page](../components/vidur.md).

---

## See Also

- [Configuration Reference](configuration.md) — Config file formats
- [SimAI-Analytical Guide](../user_guide/simai_analytical.md) — Usage examples
- [AICB Component](../components/aicb.md) — Full parameter details
