# CLI 参考

SimAI 所有工具的完整命令行参数参考。

---

## SimAI-Analytical

**二进制**: `bin/SimAI_analytical`

### 必需参数

| 标志 | 长格式 | 说明 |
|------|-----------|-------------|
| `-w` | `--workload` | 工作负载文件路径 |
| `-g` | `--gpus` | 仿真 GPU 规模 |
| `-g_p_s` | `--gpus-per-server` | Scale-up 大小（每服务器 GPU 数） |
| `-r` | `--result` | 输出文件路径和前缀（默认：`./results/`） |
| `-busbw` | `--bus-bandwidth` | busbw.yaml 文件路径 |

### 可选参数

| 标志 | 长格式 | 说明 |
|------|-----------|-------------|
| `-v` | `--visual` | 生成可视化文件 |
| `-dp_o` | `--dp-overlap-ratio` | DP 重叠比例 [0.0-1.0] |
| `-ep_o` | `--ep-overlap-ratio` | EP 重叠比例 [0.0-1.0] |
| `-tp_o` | `--tp-overlap-ratio` | TP 重叠比例 [0.0-1.0] |
| `-pp_o` | `--pp-overlap-ratio` | PP 重叠比例 [0.0-1.0] |

### 自动 Busbw 计算

| 标志 | 说明 |
|------|-------------|
| `-nv` | NVLink 带宽（GB/s） |
| `-nic` | NIC 带宽（GB/s） |
| `-n_p_s` | 每服务器 NIC 数 |

---

## SimAI-Simulation

**二进制**: `bin/SimAI_simulator`

### 环境变量

| 变量 | 说明 | 默认值 |
|----------|-------------|---------|
| `AS_LOG_LEVEL` | 日志级别：DEBUG/INFO/WARNING/ERROR | `INFO` |
| `AS_PXN_ENABLE` | 启用 PXN | `0` |
| `AS_NVLS_ENABLE` | 启用 NVLS | `0` |
| `AS_SEND_LAT` | 发送延迟（us） | `6` |
| `AS_NVLSTREE_ENABLE` | 启用 NVLS Tree | `false` |

### 参数

| 标志 | 长格式 | 说明 | 默认值 |
|------|-----------|-------------|---------|
| `-t` | `--thread` | 线程数 | `1` |
| `-w` | `--workload` | 工作负载路径 | 必需 |
| `-n` | `--network-topo` | 拓扑文件路径 | 必需 |
| `-c` | `--config` | SimAI.conf 路径 | 必需 |

---

## SimAI-Physical

**二进制**: `bin/SimAI_phynet`

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `hostlist` | 主机 IP 列表路径 | 必需 |
| `-w` / `--workload` | 工作负载文件路径 | `./microAllReduce.txt` |
| `-i` / `--gid_index` | RDMA 的 GID 索引 | `0` |
| `-g` / `--gpus` | GPU 数量 | `8` |

---

## 拓扑生成器

**脚本**: `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py`

| 层级 | 标志 | 说明 |
|-------|------|-------------|
| 全局 | `-topo` | 模板：Spectrum-X / AlibabaHPN / DCN+ |
| | `-g` | GPU 数量 |
| | `--dp` | 启用双 Plane |
| | `--ro` | 启用 Rail-optimized |
| | `--dt` | 启用双 ToR |
| | `-er` | 错误率 |
| 服务器内 | `-gps` | 每服务器 GPU 数 |
| | `-gt` | GPU 型号（A100/H100） |
| | `-nsps` | 每服务器 NV Switch 数 |
| | `-nvbw` | NVLink 带宽 |
| | `-nl` | NVLink 延迟 |
| | `-l` | NIC 延迟 |
| Segment 内 | `-bw` | NIC 到 ASW 带宽 |
| | `-asw` | ASW 交换机数量 |
| | `-nps` | 每交换机 NIC 数 |
| Pod 内 | `-psn` | PSW 交换机数量 |
| | `-apbw` | ASW 到 PSW 带宽 |
| | `-app` | 每 PSW 的 ASW 数 |

---

## AICB 工作负载生成器

**脚本**: `scripts/megatron_workload_with_aiob.sh` 或 `python -m workload_generator.SimAI_training_workload_generator`

### 核心参数

| 参数 | 说明 |
|-----------|-------------|
| `--frame` | 框架：Megatron / DeepSpeed / DeepSeek |
| `-m` / `--model_size` | 模型大小：7/13/22/175/moe/deepseek |
| `--world_size` | 总 GPU 数量 |
| `--global_batch` | 总批量大小 |
| `--micro_batch` | 微批量大小 |
| `--seq_length` | 序列长度 |
| `--epoch_num` | 迭代次数 |

### 并行参数

| 参数 | 说明 |
|-----------|-------------|
| `--tensor_model_parallel_size` | TP 度 |
| `--pipeline_model_parallel` | PP 度 |
| `--expert_model_parallel_size` | EP 度 |
| `--enable_sequence_parallel` | 启用 SP |

### 模型参数

| 参数 | 说明 |
|-----------|-------------|
| `--num_layers` | Transformer 层数 |
| `--hidden_size` | 隐藏层大小 |
| `--num_attention_heads` | 注意力头数 |
| `--ffn_hidden_size` | FFN 隐藏层大小 |
| `--vocab_size` | 词表大小 |

### MoE 参数

| 参数 | 说明 |
|-----------|-------------|
| `--moe_enable` | 启用 MoE |
| `--num_experts` | 专家数量 |
| `--moe_router_topk` | 每 Token 专家数 |
| `--moe_grouped_gemm` | 启用分组 GEMM |

### DeepSeek 参数

| 参数 | 说明 |
|-----------|-------------|
| `--qk_rope_dim` | QK 的 RoPE 维度 |
| `--qk_nope_dim` | QK 的非 RoPE 维度 |
| `--q_lora_rank` | Q LoRA 秩 |
| `--kv_lora_rank` | KV LoRA 秩 |
| `--v_head_dim` | V Head 维度 |
| `--n_shared_expert` | 每 MoE 层共享专家数 |
| `--n_dense_layer` | 稠密层数 |

### 优化参数

| 参数 | 说明 |
|-----------|-------------|
| `--use_flash_attn` | FlashAttention |
| `--swiglu` | SwiGLU 激活函数 |
| `--aiob_enable` | AIOB 计算 Profiling |
| `--comp_filepath` | 预计算时间文件 |

---

## Vidur 推理仿真

**命令**: `python -m vidur.main`

运行 `python -m vidur.main -h` 查看完整参数列表。关键参数见 [vidur 组件页面](../components/vidur.md)。

---

## 相关文档

- [配置文件参考](configuration.md) — 配置文件格式
- [SimAI-Analytical 指南](../user_guide/simai_analytical.md) — 使用示例
- [AICB 组件](../components/aicb.md) — 完整参数详情
