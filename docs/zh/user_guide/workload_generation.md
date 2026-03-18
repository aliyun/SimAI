# 工作负载生成

AICB（AI Communication Benchmark）为 SimAI 的训练和推理仿真提供工作负载生成功能。

---

## 概述

AICB 生成工作负载描述文件（`.txt`），描述 LLM 训练/推理过程的通信和计算模式。这些工作负载由 SimAI 仿真引擎消费。

支持两类工作负载生成：

| 类型 | 说明 | 支持的模型 |
|------|------|-----------|
| **训练** | 生成训练通信/计算模式 | GPT (7B/13B/22B/175B)、LLaMA (7B/65B/405B)、DeepSeek (16B/236B/671B)、MoE |
| **推理** | 生成 prefill/decode 阶段工作负载 | DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B |

---

## 训练工作负载生成

### 使用预配置模型快速开始

```bash
# 生成 Megatron GPT-7B 工作负载
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
  --world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
  --frame Megatron --global_batch 8192 \
  --micro_batch 1 --seq_length 4096 --swiglu \
  --use_flash_attn --aiob_enable \
  --comp_filepath workload/aiob_inputs/Example.txt
```

可用预配置模型大小：`7`、`13`、`22`、`175`（GPT/LLaMA）、`moe`、`deepseek`（16/236/671）。

### 不同框架的生成方法

#### Megatron

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 \
  --global_batch=16 --micro_batch=1 --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu
```

#### MoE

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --model_name MoE --frame=Megatron \
  --world_size=32 --tensor_model_parallel_size=4 --pipeline_model_parallel=1 \
  --expert_model_parallel_size=2 --moe_enable --num_experts=8 --moe_router_topk=2 \
  --global_batch=32 --micro_batch=1 --seq_length=2048 \
  --aiob_enable --swiglu --use_flash_attn
```

#### DeepSeek

```bash
python -m workload_generator.SimAI_training_workload_generator \
  --frame=DeepSeek \
  --world_size=32 --tensor_model_parallel_size=4 \
  --expert_model_parallel_size=2 --moe_enable --num_experts=4 --moe_router_topk=2 \
  --global_batch=16 --micro_batch=1 --seq_length=4096 \
  --aiob_enable --swiglu -m deepseek
```

#### DeepSpeed

```bash
python -m workload_generator.generate_deepspeed_stage3_workload \
  --world_size=64 --global_batch=64 \
  --num_layers=40 --hidden_size=5120 --seq_length=4096 \
  --zero_stage=3 --reduce_bucket_size=1000000000
```

### 输出

生成的工作负载文件保存在：
- 训练：`results/mocked_workload/` 或 `results/workload/`

---

## 推理工作负载生成

SimAI 使用 AICB 生成带有 prefill/decode 阶段分离的推理工作负载。

> **注意**：推理计算性能分析需要 NVIDIA Hopper (SM90) 或 Blackwell (SM100) GPU，因为依赖 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 和 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)。

### 支持的推理模型

| 模型 | 注意力 | MoE 专家数 | 每 token 激活专家数 |
|------|--------|-----------|-------------------|
| DeepSeek-V3-671B | MLA | 256 路由 + 1 共享 | 8 |
| Qwen3-MoE-235B | MHA/GQA | 128 路由 | 8 |
| Qwen3-Next-80B | 混合（全注意力 + 线性注意力） | 512 路由 | 10 |

推理工作负载由 vidur-alibabacloud 调度框架自动生成和消费。端到端用法请参见[推理仿真](inference_simulation.md)。

---

## AIOB：计算时间嵌入

AIOB（AI Operation Benchmark）是 AICB 的子模块，用于分析实际 GPU 计算时间并将其嵌入工作负载。

### 使用选项

| 选项 | 说明 |
|------|------|
| `--aiob_enable` | 启用 AIOB，在当前 GPU 上分析计算时间 |
| `--comp_filepath <path>` | 使用已有的计算时间描述文件 |
| 均不指定 | 使用固定默认计算时间 |

### 示例：分析并嵌入

```bash
sh scripts/megatron_gpt.sh \
  -m 7 --world_size 8 --tensor_model_parallel_size 2 \
  --frame Megatron --global_batch 16 --micro_batch 1 \
  --seq_length 2048 --swiglu --use_flash_attn --aiob_enable
```

计算描述文件保存在 `results/aiob_outputs/`。

---

## 关键参数

| 类别 | 参数 | 说明 |
|------|------|------|
| **框架** | `--frame` | Megatron / DeepSpeed / DeepSeek |
| **模型** | `--model_size` 或 `-m` | 预配置模型大小 |
| **训练** | `--world_size` | GPU 总数 |
| | `--global_batch` | 全局批大小 |
| | `--micro_batch` | 微批大小 |
| | `--seq_length` | 序列长度 |
| | `--epoch_num` | 迭代次数 |
| **并行** | `--tensor_model_parallel_size` | TP 并行度 |
| | `--pipeline_model_parallel` | PP 并行度 |
| | `--expert_model_parallel_size` | EP 并行度 |
| **MoE** | `--moe_enable` | 启用 MoE |
| | `--num_experts` | 专家数量 |
| | `--moe_router_topk` | 每 token 激活专家数 |
| **优化** | `--use_flash_attn` | 使用 FlashAttention |
| | `--swiglu` | 使用 SwiGLU 激活函数 |
| | `--aiob_enable` | 启用 AIOB 计算分析 |
| | `--comp_filepath` | 计算时间文件路径 |

完整参数列表请参见 [AICB 组件文档](../components/aicb.md) 或 [CLI 参考](../technical_reference/cli_reference.md)。

---

## 相关文档

- [AICB 组件](../components/aicb.md) — 完整 AICB 文档
- [推理仿真](inference_simulation.md) — 端到端推理仿真指南
- [支持的模型](supported_models.md) — 完整模型兼容列表
