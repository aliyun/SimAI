# 添加新模型

本指南介绍如何为 SimAI 添加新模型支持，包括 Vidur 推理仿真侧（GPU 显存、Profiling）和 AICB 工作负载生成侧。

---

## 概述

添加新模型通常涉及两个组件：

| 组件 | 需要添加的内容 | 硬件要求 |
|-----------|-------------|-------------------|
| **vidur-alibabacloud** | 模型配置、Profiling 数据（计算 + 网络） | GPU（仅 Profiling 阶段需要） |
| **AICB** | 工作负载生成参数（`MockedParam` / `MockedModel`） | 无 |

---

## 第一部分：Vidur — 模型配置与 Profiling

### 步骤 1：添加模型配置

在 `vidur-alibabacloud/data/model_configs/` 或 `vidur-alibabacloud/data/hf_configs/` 中创建 YAML/JSON 模型配置：

- 使用模型的 HuggingFace 模型 ID 作为文件名（如 `meta-llama/Llama-2-70b-hf.yml`）
- 参考模型的 HuggingFace `config.json` 获取参数值
- 确保正确设置参数，使参考 Transformer 模型尽可能接近新模型

**配置参数示例：**

```yaml
num_layers: 80
hidden_size: 8192
num_attention_heads: 64
num_key_value_heads: 8       # GQA 模型
head_dim: 128
intermediate_size: 28672
vocab_size: 128256
max_position_embeddings: 8192
```

MoE 模型还需包含：

```yaml
num_routed_experts: 256
num_experts_per_tok: 8
num_shared_experts: 1
moe_intermediate_size: 2048
```

### 步骤 2：Profiling 数据结构

Profiling 数据存储在 `vidur-alibabacloud/data/profiling/`：

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

**关键区别：**
- **计算 Profiling**：仅依赖 GPU 型号（如 `a100`、`h100`），不依赖网络拓扑
- **网络 Profiling**：依赖网络配置（如 `a100_pair_nvlink` vs `a100_dgx`）

### 步骤 3：计算 Profiling（MLP）

需要实际 GPU。TP > 1 时仅需 1 块 GPU 即可。

```bash
# 安装 sarathi-serve（vidur 分支）用于 Profiling
# 然后运行 MLP Profiling：
python vidur/profiling/mlp/main.py \
    --models your-model/model-name \
    --num_gpus 4

# 将输出复制到数据目录：
cp profiling_outputs/mlp/<timestamp>/your-model/model-name/mlp.csv \
   data/profiling/compute/<gpu_sku>/your-model/model-name/mlp.csv
```

### 步骤 4：计算 Profiling（Attention）

```bash
python vidur/profiling/attention/main.py \
    --models your-model/model-name \
    --num_gpus 4

# 复制输出：
cp profiling_outputs/attention/<timestamp>/your-model/model-name/attention.csv \
   data/profiling/compute/<gpu_sku>/your-model/model-name/attention.csv
```

### 步骤 5：网络 Profiling（如需）

网络 Profiling 是**与模型无关**的——相同硬件配置的数据可用于所有模型。

```bash
# AllReduce Profiling（用于 TP）：
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective all_reduce

# Send/Recv Profiling（用于 PP，需要多节点）：
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective send_recv
```

**可用网络设备 Profile：**
- `a100_pair_nvlink` — Azure Standard_NC96ads_A100_v4（4x A100 PCIe + NVLink pairs）
- `h100_pair_nvlink` — Azure 内部（4x H100 NVL + NVLink pairs）
- `a100_dgx` — A100 DGX（8x A100）
- `h100_dgx` — H100 DGX（8x H100）

---

## 第二部分：AICB — 工作负载生成

### 自定义模型参数（MockedParam）

在 AICB 中添加新模型的工作负载生成，需创建 `MockedParam` 子类：

```python
# 在 aicb/workload_generator/mocked_params/ 中
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
        # MoE 参数（如适用）
        self.num_experts = 256
        self.topk = 8
        self.moe_intermediate_size = 2048
```

### 自定义模型工作流（MockedModel）

如需完全控制工作负载生成过程，可创建 `MockedModel` 子类，定义每层的计算和通信操作。

详见 [AICB 组件文档](../components/aicb.md#自定义模型开发)。

### 推理工作负载生成

生成带 Prefill/Decode 分离的推理工作负载：

```bash
# 生成推理工作负载
python -m aicb.main \
    --model_name your-model-name \
    --workload_type inference \
    --num_prefill_tokens 1024 \
    --num_decode_tokens 128
```

---

## 第三部分：GPU 显存模块

如果您的模型使用非标准注意力架构，可能需要扩展 `vidur/utils/param_counter.py` 中的 `ParamCounter`：

1. 添加您的架构的注意力参数计算
2. 添加 KV Cache 每 Token 大小计算
3. 使用 MemoryPlanner 测试验证 OOM 检测正确工作

详见 [GPU 显存模块技术参考](../technical_reference/memory_module.md)。

---

## 验证清单

- [ ] 模型配置文件已添加到 `data/model_configs/` 或 `data/hf_configs/`
- [ ] 计算 Profiling 数据（MLP + Attention）已添加
- [ ] 目标硬件的网络 Profiling 数据可用
- [ ] AICB `MockedParam` 已创建（如需工作负载生成）
- [ ] GPU 显存计算正确（ParamCounter + MemoryPlanner）
- [ ] 端到端推理仿真产生合理结果
- [ ] 文档已更新

---

## 相关文档

- [vidur-alibabacloud 组件](../components/vidur.md) — 完整 vidur 文档
- [AICB 组件](../components/aicb.md) — AICB 工作负载生成
- [GPU 显存模块](../technical_reference/memory_module.md) — 显存计算公式
- [支持的模型](../user_guide/supported_models.md) — 当前模型支持状态
