# AICB — AI 通信基准测试

**仓库**: [aliyun/aicb](https://github.com/aliyun/aicb) | **语言**: Python

AICB 是面向 AI 场景的专用通信基准测试套件。它能生成与真实 LLM 训练和推理流程对齐的通信工作负载。

---

## 简介

AICB（Artificial Intelligence Communication Benchmark）生成与真实应用精准对齐的通信工作负载模式，支持：

- 基准测试和调优 GPU 集群通信系统
- 研究特定模型配置的通信模式
- 为 SimAI 等仿真器生成工作负载

---

## 基准测试套件

AICB 提供 10 个预配置的基准测试案例，覆盖典型 LLM 配置：

| 编号 | 模型 | 序列长度 | 框架 | TP | PP | SP | MoE |
|----|-------|-----------|-----------|----|----|-----|-----|
| 1 | LLaMA-7B | 2048 | Megatron | 1 | 1 | - | - |
| 2 | GPT-13B | 2048 | Megatron | 2 | 1 | 是 | - |
| 3 | GPT-22B | 2048 | Megatron | 4 | 1 | - | - |
| 4 | LLaMA-65B | 4096 | Megatron | 8 | 2 | 是 | - |
| 5 | GPT-175B | 2048 | Megatron | 8 | 8 | 是 | - |
| 6 | GPT-175B | 2048 | Megatron | 8 | 8 | - | - |
| 7 | Llama3-405B | 8192 | Megatron | 8 | 16 | 是 | - |
| 8 | LLaMA-7B | 4096 | DeepSpeed | 1 | 1 | - | Zero-2 |
| 9 | LLaMA-65B | 4096 | DeepSpeed | 1 | 1 | - | Zero-3 |
| 10 | Mistral-8x7B | 2048 | Megatron | 2 | 1 | 是 | 8 experts |

---

## 环境搭建

### Docker

```bash
docker build -t aicb:v0.0.1 .
docker run --gpus all --net host --shm-size 16g -it --rm aicb:v0.0.1
```

### 本地环境

要求：Python >= 3.8、CUDA >= 11.8、PyTorch >= 2.0.0、NVIDIA APEX

### NGC 容器

```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/aicb:/workspace/aicb nvcr.io/nvidia/pytorch:xx.xx-py3
```

> **注意**：推理工作负载 Profiling 需要 NVIDIA Hopper (SM90) 或 Blackwell (SM100) GPU。

---

## 物理集群执行

### 环境变量

| 参数 | 说明 |
|-----------|-------------|
| `nnodes` | 节点数量 |
| `node_rank` | 当前节点编号 |
| `nproc_per_node` | 每节点 GPU 数 |
| `master_addr` | 主节点地址 |
| `master_port` | 主节点端口 |

### 运行 Megatron 工作负载

```bash
sh scripts/megatron_gpt.sh \
  --nnodes 1 --node_rank 0 --nproc_per_node 8 \
  --master_addr localhost --master_port 29500 \
  -m 7 --world_size 8 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --seq_length 2048 --swiglu --use_flash_attn --aiob_enable
```

### 运行 MoE 工作负载

```bash
sh scripts/megatron_gpt.sh \
  -m moe --world_size 8 --tensor_model_parallel_size 4 \
  --moe_enable --expert_model_parallel_size 1 \
  --num_experts 4 --moe_router_topk 2 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --sp --grouped_gemm --aiob_enable --swiglu --use_flash_attn
```

### 运行 DeepSeek 工作负载

```bash
sh scripts/megatron_gpt.sh \
  --frame DeepSeek -m deepseek \
  --tensor_model_parallel_size 4 --moe_enable \
  --expert_model_parallel_size 1 --num_experts 4 \
  --global_batch 4 --micro_batch 1 --world_size 4 \
  --num_layers 10 --sp --swiglu --aiob_enable
```

---

## 训练工作负载生成

为 SimAI 仿真生成工作负载文件：

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu
```

输出保存在 `results/mocked_workload/`。

---

## 推理工作负载生成

AICB 为以下模型生成带 Prefill/Decode 阶段分离的推理工作负载：

| 模型 | 注意力架构 | MoE 专家数 |
|-------|-----------|-------------|
| DeepSeek-V3-671B | MLA | 256 路由 + 1 共享 |
| Qwen3-MoE-235B | MHA/GQA | 128 路由 |
| Qwen3-Next-80B | 混合（全注意力 + 线性注意力） | 512 路由 |

需要硬件加速库：[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)。

---

## AIOB：计算性能分析

AIOB 可采集实际 GPU 计算耗时并嵌入工作负载：

- `--aiob_enable` — 在当前 GPU 上进行 Profiling
- `--comp_filepath <path>` — 使用已有 Profiling 数据

输出保存在 `results/aiob_outputs/`。

---

## 自定义模型开发

AICB 支持使用 `MockedParam` 和 `MockedModel` 基类为自定义模型架构创建工作负载。

训练过程被抽象为：`init → forward → backward → step`

每条工作负载项包含：
1. **通信信息**：`comm_type`、`comm_group`、`comm_group_size`、`msg_size`
2. **附加信息**：源节点（broadcast 场景）、计算耗时
3. **运行时信息**：`elapsed_time`、`algo_bw`、`bus_bw`

可参考现有 `MockedMegatron` 和 `MockedDeepSpeed` 实现。

---

## 关键参数

| 类别 | 参数 | 说明 |
|----------|-----------|-------------|
| 框架 | `frame` | Megatron / DeepSpeed / DeepSeek |
| 模型 | `model_size` | 预配置大小（7/13/22/175/moe/deepseek） |
| 训练 | `world_size` | 总 GPU 数量 |
| | `global_batch` | 总批量大小 |
| | `micro_batch` | 微批量大小 |
| | `seq_length` | 序列长度 |
| 并行策略 | `tensor_model_parallel_size` | TP 度 |
| | `pipeline_model_parallel` | PP 度 |
| | `expert_model_parallel_size` | EP 度 |
| MoE | `moe_enable` | 启用 MoE |
| | `num_experts` | 专家数量 |
| | `moe_router_topk` | 每 Token 专家数 |
| DeepSeek | `qk_rope_dim` | QK 的 RoPE 维度 |
| | `kv_lora_rank` | KV 压缩 LoRA 维度 |
| | `q_lora_rank` | Q 压缩 LoRA 维度 |
| | `n_shared_expert` | 共享专家数 |
| 优化 | `use_flash_attn` | FlashAttention |
| | `swiglu` | SwiGLU 激活函数 |
| | `aiob_enable` | AIOB 计算 Profiling |
| | `comp_filepath` | 预有计算文件 |

---

## 结果输出

### 物理执行

- 每次通信日志：类型、分组、消息大小、执行时间、吞吐量
- 每次迭代耗时分析
- CSV 输出在 `results/comm_logs/`

### 工作负载文件

- 训练工作负载：`results/mocked_workload/` 或 `results/workload/`
- AIOB Profiling：`results/aiob_outputs/`

---

## 相关文档

- [工作负载生成指南](../user_guide/workload_generation.md) — 用户指南中的工作负载生成
- [支持的模型](../user_guide/supported_models.md) — 完整模型列表
- [Tutorial](https://github.com/aliyun/aicb/blob/master/training/tutorial.md) — AICB 详细教程
