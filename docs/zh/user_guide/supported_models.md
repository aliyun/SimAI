# 支持的模型

SimAI 支持多种 LLM 模型的训练和推理仿真。

---

## 推理模型（SimAI 1.5+）

以下模型通过 vidur-alibabacloud 支持多请求推理仿真，包含 GPU 显存计算、PD 分离和工作负载生成支持。

### DeepSeek-V3-671B

| 属性 | 值 |
|------|------|
| **总层数** | 61 |
| **注意力类型** | MLA（多头潜在注意力） |
| **注意力头数** | 128 |
| **隐藏维度** | 7168 |
| **KV LoRA 秩** | 512 |
| **Q LoRA 秩** | 1536 |
| **QK RoPE Head Dim** | 64 |
| **QK NoPE Head Dim** | 128 |
| **V Head Dim** | 128 |
| **MoE 路由专家数** | 256 |
| **每 token 激活专家数** | 8 |
| **共享专家数** | 1 |
| **稠密层** | 前 3 层（固定激活 8 个路由 + 1 个共享专家） |
| **稀疏层** | 第 3-60 层（从 256 个路由专家中动态选择 8 个 + 1 个共享专家） |
| **配置文件** | `vidur-alibabacloud/data/hf_configs/deepseek_v3_config.json` |

### Qwen3-MoE-235B

| 属性 | 值 |
|------|------|
| **总层数** | 94 |
| **注意力类型** | MHA / GQA |
| **注意力头数** | 64 |
| **KV 头数** | 4 |
| **隐藏维度** | 4096 |
| **Head Dim** | 128 |
| **MoE 路由专家数** | 128 |
| **每 token 激活专家数** | 8 |
| **MoE 中间维度** | 1536 |
| **配置文件** | `vidur-alibabacloud/data/hf_configs/qwen3_moe_config.json` |

### Qwen3-Next-80B

| 属性 | 值 |
|------|------|
| **总层数** | 48 |
| **注意力类型** | 混合（全注意力 + 线性注意力，每 4 层交替） |
| **全注意力头数** | 16 |
| **KV 头数** | 2 |
| **隐藏维度** | 2048 |
| **Head Dim** | 256 |
| **线性注意力 Key 头数** | 16 |
| **线性注意力 Value 头数** | 32 |
| **MoE 路由专家数** | 512 |
| **每 token 激活专家数** | 10 |
| **MoE 中间维度** | 512 |
| **配置文件** | `vidur-alibabacloud/data/hf_configs/qwen3-next-80B-A3B_config.json` |

### 传统推理模型（通过 Vidur 后端）

以下模型使用原版 Vidur 基于 profiling 的后端：

| 模型 | TP 支持 | PP 支持 |
|------|---------|---------|
| meta-llama/Meta-Llama-3-8B | 是 | 是 |
| meta-llama/Meta-Llama-3-70B | 是 | 是 |
| meta-llama/Llama-2-7b-hf | 是 | 是 |
| meta-llama/Llama-2-70b-hf | 是 | 是 |
| codellama/CodeLlama-34b-Instruct-hf | 是 | 是 |
| internlm/internlm-20b | 是 | 是 |
| Qwen/Qwen-72B | 是 | 是 |

---

## 训练模型（AICB）

以下模型支持训练工作负载生成：

### AICB 基准测试套件

| ID | 模型 | 序列长度 | 框架 | TP | PP | SP | MoE |
|----|------|----------|------|----|----|-----|-----|
| 1 | LLaMA-7B | 2048 | Megatron | 1 | 1 | - | - |
| 2 | GPT-13B | 2048 | Megatron | 2 | 1 | 是 | - |
| 3 | GPT-22B | 2048 | Megatron | 4 | 1 | - | - |
| 4 | LLaMA-65B | 4096 | Megatron | 8 | 2 | 是 | - |
| 5 | GPT-175B | 2048 | Megatron | 8 | 8 | 是 | - |
| 6 | GPT-175B | 2048 | Megatron | 8 | 8 | - | - |
| 7 | Llama3-405B | 8192 | Megatron | 8 | 16 | 是 | - |
| 8 | LLaMA-7B | 4096 | DeepSpeed | 1 | 1 | - | Zero-2 |
| 9 | LLaMA-65B | 4096 | DeepSpeed | 1 | 1 | - | Zero-3 |
| 10 | Mistral-8x7B | 2048 | Megatron | 2 | 1 | 是 | 8 专家 |

---

## 注意力架构对比

| 架构 | 模型 | KV Cache 策略 | 内存效率 |
|------|------|-------------|----------|
| **MLA** | DeepSeek-V3-671B | 压缩潜向量（`kv_lora_rank` + `qk_rope_head_dim`） | 相比 MHA 约 57 倍缩减 |
| **MHA / GQA** | Qwen3-MoE-235B | 标准 KV 缓存（`num_kv_heads * head_dim`） | 标准 |
| **混合全注意力 + 线性注意力** | Qwen3-Next-80B | 全注意力层 + 线性 (GDN) 注意力每 4 层交替 | 减少（线性注意力层无 KV 缓存） |

---

## 硬件要求

### 推理性能分析（AICB 后端）

| 要求 | 详情 |
|------|------|
| **GPU 架构** | NVIDIA Hopper (SM90) 或 Blackwell (SM100) |
| **原因** | 依赖 DeepGEMM、FlashMLA、FlashInfer |
| **Hopper 注意** | 在 Dockerfile 中添加 `ENV FLASH_MLA_DISABLE_SM100=1` |

### 训练仿真

- **SimAI-Analytical**：任意 CPU（无需 GPU）
- **SimAI-Simulation**：任意 CPU（无需 GPU）
- **AICB 物理执行**：需要支持 NCCL 的 GPU 集群

---

## 相关文档

- [推理仿真](inference_simulation.md) — 多请求推理指南
- [工作负载生成](workload_generation.md) — AICB 工作负载生成
- [GPU 显存模块](../technical_reference/memory_module.md) — 显存计算详情
- [vidur-alibabacloud](../components/vidur.md) — 推理调度组件
