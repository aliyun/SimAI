# 配置文件参考

本文档涵盖 SimAI 使用的所有配置文件。

---

## SimAI.conf

**路径**: `astra-sim-alibabacloud/inputs/config/SimAI.conf`

SimAI-Analytical 和 SimAI-Simulation 模式共用的主仿真配置文件，控制通信算法、缓冲区大小和时序参数。

---

## busbw.yaml

**路径**: `example/busbw.yaml`

SimAI-Analytical 使用，用于指定不同通信组和集合操作的总线带宽。

### 格式

```yaml
test
TP:
  allreduce,: 300      # TP 组 AllReduce busbw 300GB/s
  allgather,: 280
  reducescatter,: 280
  alltoall,: 230
DP:
  allreduce,: null      # null = 该组不使用此操作
  allgather,: 380
  reducescatter,: 380
  alltoall,: null
EP:
  allreduce,: null
  allgather,: 45
  reducescatter,: 45
  alltoall,: 80
```

### 通信组

| 组 | 说明 |
|-------|-------------|
| `TP` | 张量并行 — 服务器内 NVLink 通信 |
| `DP` | 数据并行 — 服务器间 RDMA 通信 |
| `EP` | 专家并行 — MoE 专家通信 |

### 集合操作

| 操作 | 说明 |
|-----------|-------------|
| `allreduce` | 归约 + 广播到所有 Rank |
| `allgather` | 从所有 Rank 收集数据 |
| `reducescatter` | 归约并分发 |
| `alltoall` | 全对全个性化交换 |

对特定组中不使用的操作，将值设为 `null`。

---

## 拓扑文件

由 `gen_Topo_Template.py` 生成，拓扑文件为 SimAI-Simulation 定义网络结构。

### 生成

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps
```

输出文件以参数命名，如 `Spectrum-X_128g_8gps_100Gbps_A100`。

### 模板默认值

| 模板 | GPU 数 | 拓扑 | 带宽 | GPU 型号 |
|----------|------|------|-----------|----------|
| Spectrum-X | 4096 | Rail-optimized，单 ToR | 400Gbps | H100 |
| AlibabaHPN（单 Plane） | 15360 | Rail-optimized，双 ToR | 200Gbps | H100 |
| AlibabaHPN（双 Plane） | 15360 | Rail-optimized，双 ToR，双 Plane | 200Gbps | H100 |
| DCN+（单 ToR） | 512 | 非 Rail-optimized | 400Gbps | A100 |
| DCN+（双 ToR） | 512 | 非 Rail-optimized，双 ToR | 200Gbps | H100 |

---

## 模型配置文件

### 推理模型配置

位于 `vidur-alibabacloud/data/hf_configs/`：

| 模型 | 配置文件 |
|-------|------------|
| DeepSeek-V3-671B | `deepseek_v3_config.json` |
| Qwen3-MoE-235B | `qwen3_moe_config.json` |
| Qwen3-Next-80B | `qwen3-next-80B-A3B_config.json` |

这些文件遵循 HuggingFace `config.json` 格式，定义模型架构参数。

### Profiling 数据

位于 `vidur-alibabacloud/data/profiling/`：

```
profiling/
├── compute/
│   ├── a100/
│   │   └── <model_name>/
│   │       ├── mlp.csv
│   │       └── attention.csv
│   └── h100/
│       └── <model_name>/
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

- **计算 Profiling**：仅依赖 GPU 型号（如 `a100`、`h100`），不依赖网络拓扑
- **网络 Profiling**：依赖网络配置（如 `a100_pair_nvlink` vs `a100_dgx`）

---

## 工作负载文件

### 训练工作负载格式

```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 8 ga: 1 all_gpus: 32 checkpoints: 0 checkpoint_initiates: 0
6
embedding_layer  -1 556000  ALLREDUCE  16777216  1  NONE 0  1  NONE 0  1
...
```

头部字段：
- `model_parallel_NPU_group`：TP 大小
- `ep`：EP 大小
- `pp`：PP 大小
- `vpp`：虚拟流水线并行
- `ga`：梯度累积
- `all_gpus`：总 GPU 数量

### 请求 Trace 文件

用于推理仿真，位于 `vidur-alibabacloud/data/processed_traces/`：

- `splitwise_conv.csv` — 对话式 Trace
- `sharegpt_8k_filtered_stats_llama2_tokenizer.csv` — ShareGPT Trace

---

## 相关文档

- [CLI 参考](cli_reference.md) — 命令行参数
- [SimAI-Analytical 指南](../user_guide/simai_analytical.md) — busbw 配置使用
- [SimAI-Simulation 指南](../user_guide/simai_simulation.md) — 拓扑配置使用
