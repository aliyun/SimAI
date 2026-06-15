# AICB Workload Generator 模型可扩展性研究报告

**日期:** 2026-06-15
**研究范围:** aliyun/aicb workload generator 的模型架构扩展能力分析
**方法:** 多源交叉验证（论文、代码库、GitHub API、社区讨论）

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [当前 AICB 架构分析](#2-当前-aicb-架构分析)
3. [扩展到其他模型架构的可行性](#3-扩展到其他模型架构的可行性)
4. [2025-2026 年新并行策略与通信 Collective](#4-2025-2026-年新并行策略与通信-collective)
5. [与同类工具的比较](#5-与同类工具的比较)
6. [aliyun/aicb 仓库近期动态](#6-aliyunaicb-仓库近期动态)
7. [可操作的扩展建议](#7-可操作的扩展建议)
8. [参考资料](#8-参考资料)

---

## 1. 执行摘要

AICB (aliyun/aicb) 是 SimAI 生态中的 workload generator，当前通过 `mocked_model` 机制支持 Megatron-LM（GPT 风格稠密 Transformer）和 DeepSeek（MLA + MoE）两种训练 workload，以及 Qwen3 MoE/Next 的推理 workload（结构骨架，未完成 workload 生成逻辑）。

**核心发现:**

1. **AICB 可以扩展到 LLaMA、Mistral/Mixtral、Qwen、Gemma、DBRX**，每模型需 200-600 行 Python 代码。Mamba/SSM 和 Falcon（并行 attention+MLP）需要重大重构。
2. **2025 年 DeepSeek 开源周**引入了三项新并行策略（DualPipe、DeepEP、EPLB），NVIDIA 和 MLSys 2026 进一步推进了 MoE 通信优化，AICB 当前均不支持。
3. **AICB 仓库活跃**（233 stars, 55 forks），AICB 2.0 于 2025 年 12 月合并，社区贡献者 MXtremist 主导。
4. **STAGE 是最接近的可比工具**，其声明式 CSV 驱动方法可消除 AICB 的逐模型编码需求。PARAM 和 Chakra 是 trace 采集工具，不是 workload 生成器。

**优先级建议:**

| 优先级 | 行动 | 预期影响 |
|--------|------|----------|
| 立即 | 添加模型注册表替代硬编码 if/elif | 消除核心代码修改需求 |
| 高 | 实现 LLaMA 模型类（GQA + SwiGLU） | 覆盖最大模型家族 |
| 高 | 参数化 MoE 路由配置 | 支持 Mixtral/DBRX 无需新代码 |
| 中 | 添加 Context Parallelism 支持 | 覆盖 2025+ 长序列训练 |
| 中 | 采用 Chakra 输出格式 | 与 MLCommons 生态互操作 |
| 战略 | 采用 STAGE 的声明式 tensor graph 方法 | 消除所有逐模型编码 |

---

## 2. 当前 AICB 架构分析

### 2.1 MockedModel 基础架构

**文件:** `aicb/workload_generator/mocked_model/MockedModel.py`（169 行）

```
MockedModel (基类)
  +-- parameters() -> List[MockedParam]
  +-- child_modules() -> List[MockedModel]
  +-- register_forward_pre_hook / register_backward_pre_hook
  +-- register_forward_post_hook / register_backward_post_hook

MockedParam(shape: Tuple, elem_size=2, name=None)
  +-- numel(), elem_size(), msg_size(), get_shape()

Linear(MockedModel)  # 简单的线性层包装
```

### 2.2 训练模型覆盖

| 模型 | 文件 | 架构特征 | 代码量 |
|------|------|----------|--------|
| MegatronModel | `MockedMegatron.py` | 标准 MHA + Up-down FFN + 可选 MoE | ~676 行 |
| DeepSeekV3Model | `MockedDeepSeek.py` | MLA (q_lora + kv_lora) + DeepSeekMoE + 共享专家 + FP8 | ~600+ 行 |
| MockedDeepspeed | `MockedDeepspeed.py` | ZeRO Stage 1/2/3 梯度分片 | 独立文件 |

### 2.3 推理模型覆盖（AICB 2.0 新增）

| 模型 | 文件 | 状态 |
|------|------|------|
| Qwen3MoeModel | `MockedQwen3Moe.py` | 结构骨架，标记 `#TODO support Workload` |
| Qwen3NextModel | `MockedQwen3Next.py` | 结构骨架 |
| AiobDeepSeek | `AiobDeepSeek.py` | 完整推理 workload |
| AiobQwen3Moe | `AiobQwen3Moe.py` | 基于 AIOB 的推理 |
| AiobQwen3Next | `AiobQwen3Next.py` | 基于 AIOB 的推理 |

### 2.4 支持的通信 Collective

**文件:** `aicb/utils/utils.py`，`CommType` 枚举（第 544-566 行）

| Collective | 用途 |
|------------|------|
| `all_reduce` | TP 同步、DP 梯度聚合 |
| `all_gather` | TP Column Linear 前向、序列并行 |
| `reduce_scatter` | TP Row Linear 前向、序列并行 |
| `all_to_all` | MoE Expert Parallelism dispatch/combine |
| `broadcast` | 参数广播 |
| `isend` / `irecv` | P2P 通信（PP 边界） |
| `barrier` | 同步点 |
| `reduce` | 归约操作 |
| `computation` | 计算操作 |
| `epoch_end` | Epoch 边界标记 |

**通信组 (`CommGroup`):** `dp_group`, `pp_group`, `tp_group`, `ep_group`, `ep_dp_group`, `ep_tp_group`, `embedding_group`, `all`

### 2.5 关键架构局限

**局限 1: 硬编码模型调度**

`generate_megatron_workload.py` 第 437-440 行:
```python
if args.frame == "DeepSeek":
    model = DeepSeekV3Model(args)
elif args.frame == "Megatron":
    model = MegatronModel(args)
```

添加任何新模型都需要修改核心调度代码，没有插件注册机制。

**局限 2: 命令式 LogItem 构造**

每个操作的前向/反向传播都需要手动构造 `LogItem`，指定 `comm_type`、`comm_group`、`msg_size`、`stage`。这与 STAGE 的声明式 einsum 方法形成对比。

**局限 3: 模型与并行策略耦合**

并行策略（TP、SP、EP）的选择逻辑嵌入在 `forward()`/`backward()` 方法内部（如 `if self.tensor_model_parallel_size > 1: ...`），无法独立切换或组合。

**局限 4: 框架选项受限**

argparse 的 `--frame` choices 只有 4 个: `["Megatron", "DeepSpeed", "collective_test", "DeepSeek"]`

---

## 3. 扩展到其他模型架构的可行性

### 3.1 各架构扩展难度与工作量估算

| 架构 | 与现有 AICB 的差异 | 难度 | 估算工作量 |
|------|-------------------|------|-----------|
| **GPT (经典)** | Post-norm, GELU, 无 GQA | 低 | ~100 行（MegatronModel 已接近） |
| **LLaMA 2/3/4** | GQA, SwiGLU Gate-up-down FFN, RMSNorm pre-norm, RoPE | 中 | ~300 行 |
| **Mistral** | GQA + sliding window attention | 中 | ~200 行（在 LLaMA 基础上加 sliding window） |
| **Mixtral** | Mistral + MoE (top-2, 8 experts/layer) | 中 | ~200 行（复用 MoE 基础设施） |
| **Qwen 2.5/3** | GQA + SwiGLU + MoE | 中-高 | ~400 行（AICB 2.0 已有推理骨架） |
| **Gemma** | GeGLU 激活, 不同 norm 位置 | 中 | ~250 行 |
| **Falcon** | 并行 attention+MLP（非串行） | 高 | ~600 行（完全不同层组合） |
| **DBRX** | 细粒度 MoE (16 experts, top-4) | 中 | ~200 行（参数化复用 MoE） |
| **Mamba/SSM** | 无 attention, 线性状态空间 | 非常高 | ~800+ 行（全新算子图） |
| **Jamba/Zamba** | 混合 Transformer + Mamba 层 | 非常高 | ~1000+ 行（组合两种范式） |

### 3.2 LLaMA 模型实现要点（最重要的扩展目标）

LLaMA 与 Megatron 的关键架构差异:

1. **Attention:** Group Query Attention (GQA) -- KV head 数 < Q head 数。AICB 当前 `MegatronAttention` 假设 Q/K/V 维度相等。
2. **FFN:** SwiGLU (Gate-up-down) -- 3 个线性投影而非 2 个。`MegatronMlp` 只有 `dense_h_to_4h` + `dense_4h_to_h`，需要添加 gate 投影。
3. **Normalization:** RMSNorm (pre-norm) vs FusedLayernorm。RMSNorm 无 bias 参数且无均值中心化。
4. **Position Encoding:** Rotary Position Embedding (RoPE) -- 影响 Q/K 投影的维度划分。

实现策略:
- 新建 `MockedLlama.py`，复用 `MegatronRowLinear` / `MegatronColumnLinear` 作为基础 TP 模块
- 添加 `SwiGLUFFN` 类（3 个 ColumnLinear: gate, up, down）
- 添加 `GQAttention` 类（Q 投影保持，K/V 投影使用 `num_kv_heads`）
- 重用 `FusedLayernorm` 作为 RMSNorm 近似

### 3.3 为什么当前方法不够

STAGE 论文（arXiv:2511.10480, Nov 2025）直接指出了模板方法的局限:

> "Several simulation frameworks (such as Calculon, MadMax, SimAI) rely on customized templates or analytical first-order equations to address the challenge of describing AI workloads. While this approach enables fast analysis and can be easily evaluated on a CPU-only system, they are often over-optimized for specific target workloads and require deep understanding with the codebase for extensions."

STAGE 的替代方案: CSV 驱动的 tensor graph 定义 + 声明式组合:

```python
# STAGE: 声明式 -- 从数据文件加载，按名称连接
def feed_forward_network(ffn_path=None):
    ffn_path = "./sharding_spreadsheets/module3/tpsp/llama_feed_forward_network.csv"
    ffn = ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path), "ffn.%s")
    return ffn
```

---

## 4. 2025-2026 年新并行策略与通信 Collective

### 4.1 DeepSeek 开源周（2025 年 2 月）

**DualPipe -- 双向流水线并行**
- 前向/反向计算-通信完全重叠
- 微批次拆分为 4 个子块: Attention -> All-to-All Dispatch -> MLP -> All-to-All Combine
- 显著减少流水线气泡（相比 1F1B 和 ZB1P/2P）
- AICB 当前仅建模标准 1F1B pipeline -- 需新增 PP 调度逻辑

**DeepEP -- MoE 专家并行通信库**
- 首个开源 MoE EP 通信库
- 双模式内核: 训练/prefill 高吞吐量模式 + 推理 decode 低延迟模式 (163 微秒 RDMA)
- 原生 FP8 数据分发支持
- 灵活 GPU 资源控制（SM 数量可配，实现计算-通信重叠）
- AICB 仅支持基础 `all_to_all`，无 FP8 dispatch 或双模式建模

**EPLB -- 专家并行负载均衡器**
- 分层负载均衡（预填充阶段，小 EP 规模）
- 全局负载均衡（解码阶段，大 EP 规模）
- 冗余专家策略: 复制高负载专家到多 GPU
- AICB 仅支持静态 MoE 路由

### 4.2 NVIDIA Hybrid-EP（2026 年 2 月）
- 结合 NVLink（节点内）+ InfiniBand（跨节点）MoE 通信
- DeepSeek V3 训练加速 14%（943 TFLOPS）
- Qwen 3 加速 9.9%
- 仅需 4 个 SM 即可饱和 NVLink 带宽

### 4.3 FarSkip-Collective（MLSys 2026）
- 修改模型架构实现通信-计算重叠
- Llama 4 Scout 转换后精度损失 < 1%
- DeepSeek-V3 prefill 实现 32.6% TTFT 加速
- 97.3% 通信重叠率

### 4.4 对 AICB 的影响总结

| 创新 | AICB 当前支持? | 需要的变更 |
|------|--------------|-----------|
| DualPipe 双向流水线 | 否（仅 1F1B） | 新 PP 调度器 |
| DeepEP 双模式 all-to-all | 否（仅基础 all_to_all） | CommType 扩展 + FP8 因子 |
| EPLB 动态负载均衡 | 否（静态路由） | 新路由策略模块 |
| Hybrid-EP 层次化通信 | 部分（有 ep_group） | 节点内/跨节点带宽差异化 |
| Context Parallelism | 否 | 新 CommGroup.cp_group + ring attention P2P |

---

## 5. 与同类工具的比较

### 5.1 全面对比

| 维度 | AICB | STAGE | Chakra | PARAM |
|------|------|-------|--------|-------|
| **类型** | 合成 workload 生成器 | 符号化 tensor graph 生成器 | 执行 trace 标准化 schema | 真实系统 profiler/benchmark |
| **生成方式** | Python 模板代码 | CSV tensor graph + 声明式组合 | 生成式 AI 统计合成 | 真实 PyTorch job 采集 |
| **模型覆盖** | Megatron, DeepSeek, DeepSpeed, Qwen3(骨架) | Dense(LLaMA,GPT), MoE(DeepSeek,Mixtral), SSM(Mamba) | 任何 PyTorch 模型（通过 NeMo profiler） | 任何 PyTorch 模型 |
| **可扩展性** | 逐模型 Python 类 + if/elif | CSV 配置 + 声明式组合 | 不适用（trace schema） | 不适用（profiler） |
| **输出格式** | 自定义 CSV | Chakra protobuf schema | Chakra protobuf schema | PyTorch profiler traces |
| **并行策略** | DP, TP, PP, EP, SP, FSDP | DP, TP, PP, EP, SP, FSDP, CP, 任意组合 | 依赖于采集 trace 的模型配置 | 依赖于采集 trace 的模型配置 |
| **新并行策略** | 需修改所有模型类 | 新 sharding spreadsheet (CSV) | N/A | N/A |
| **仓库** | github.com/aliyun/aicb (233 stars) | github.com/astra-sim/symbolic_tensor_graph (42 stars) | github.com/mlcommons/chakra | Meta 内部/开源工具链 |
| **标准化** | 无 | MLCommons Chakra WG 合作 | MLCommons 标准化进行中 | PyTorch 生态 |

### 5.2 关键区别

**AICB vs STAGE:**
- AICB: 命令式 -- 每个操作用 Python 硬编码 LogItem；新模型 = 新 Python 文件
- STAGE: 声明式 -- 从 CSV tensor graph 定义加载组件，按名称连接；新模型 = 新 wiring + 现有 CSV 组件重用
- STAGE 的 sharding spreadsheets 将并行策略从模型架构中解耦 -- 同一模型定义可用于 TP、TP+SP、FSDP 等

**AICB vs Chakra:**
- Chakra 是 **格式/协议**，不是 workload 生成器
- Chakra 的生成式 AI 合成功能学习真实 trace 的统计属性 -- 它不理解模型架构
- AICB 理解模型架构但输出专有格式
- STAGE 桥接了两者: 理解架构 + 输出 Chakra 格式

**AICB vs PARAM:**
- PARAM 是 Meta 的 profiler 工具 -- 采集真实运行的 GPU traces
- 通过 Dynolog daemon + Kineto/CUPTI 实现零代码注入 profiling
- 与 Chakra 生态合作，用于分析集体通信性能
- 不生成合成 workload -- 完全不同的工具类别

---

## 6. aliyun/aicb 仓库近期动态

### 6.1 仓库状态
- **Stars:** 233 | **Forks:** 55
- **Open Issues:** 30
- **默认分支:** master
- **最近推送:** 2026-06-10

### 6.2 近期 Commits（最近 10 个）

| SHA | 日期 | 描述 |
|-----|------|------|
| `23eec3c` | 2025-12-27 | Merge PR #58: AICB 2.0 Full Version (from MXtremist/Next) |
| `3a9e8b1` | 2025-12-26 | Update group QR code |
| `9d51993` | 2025-12-26 | feat(moe): add routing strategy to expected_m_per_group |
| `508cd8f` | 2025-12-05 | Merge branch 'AICB2.0_Pre' into Next |
| `26b1345` | 2025-12-05 | delete useless |
| `94f5cb4` | 2025-12-05 | feat(aiob): update deepgemm to a newer version |
| `c8a4588` | 2025-11-28 | fix(aiob): use contiguous in prefill |
| `3d4d33b` | 2025-11-26 | fix(aiob): delete unused XXXQwen3.py, fix shared_expert error |
| `f182db7` | 2025-11-18 | Delete scripts/inference_configs/config_gen.py |
| `47c6205` | 2025-11-15 | update readme and wechat group QR code |

### 6.3 近期 Issues

| # | 状态 | 标题 | 创建日期 |
|---|------|------|----------|
| 65 | open | Bump vllm from 0.11.0 to 0.22.0 | 2026-06-10 |
| 63 | open | Bump transformers from 4.57.1 to 5.0.0rc3 | 2026-04-08 |
| 62 | open | fix(utils): add GPU warm-up for profiling | 2026-02-02 |
| 61 | open | Code example in readme.md not updated | 2025-12-16 |
| 56 | open | change forward_comm1 in moelayer to ALLGATHER_EP | 2025-10-22 |

### 6.4 关键观察

1. **AICB 2.0** 是重大更新，由外部贡献者 MXtremist 主导 -- 表明社区参与活跃
2. MoE 通信模式正在被重新审视（Issue #56: ALLGATHER_EP vs ALLTOALL_EP）
3. 推理 workload 支持是 2.0 的核心新增功能（Qwen3 系列模型）
4. 依赖管理活跃（vllm, transformers bump）
5. README 示例代码已过时（Issue #61）-- 文档维护有滞后

---

## 7. 可操作的扩展建议

### 7.1 短期（低工作量，AICB 2.0 兼容）

#### 建议 1: 添加模型注册表

替换 `generate_megatron_workload.py` 的硬编码 if/elif:

```python
MODEL_REGISTRY = {
    "Megatron": MegatronModel,
    "DeepSeek": DeepSeekV3Model,
    "DeepSpeed": DeepspeedModel,
    "LLaMA": LlamaModel,          # 新增
    "Mixtral": MixtralModel,      # 新增
    # ...
}

# 替换:
# if args.frame == "DeepSeek": model = DeepSeekV3Model(args)
# elif args.frame == "Megatron": model = MegatronModel(args)

model_cls = MODEL_REGISTRY.get(args.frame)
if model_cls is None:
    raise ValueError(f"Unknown model: {args.frame}")
model = model_cls(args)
```

**工作量:** ~20 行修改 | **影响:** 消除核心框架代码修改需求

#### 建议 2: 实现 LLaMA MockedModel

创建 `mocked_model/training/MockedLlama.py`:

```
需要新增的类:
  - SwiGLUFFN(MockedModel)     # 3 个投影: gate_proj, up_proj, down_proj
  - GQAttention(MockedModel)    # Q 全头投影, K/V 减少头数
  - LlamaTransformerLayer       # pre-norm + GQA + post-norm + SwiGLU
  - LlamaModel                  # 组装完整模型

可复用的现有类:
  - MegatronRowLinear           # TP 行线性投影
  - MegatronColumnLinear        # TP 列线性投影
  - FusedLayernorm              # RMSNorm 近似
  - MOEMLP (可选 MoE)           # 现有 MoE 基础设施
```

**工作量:** ~300 行代码 | **影响:** 覆盖最大的模型家族（LLaMA 2/3/4, Mistral, Qwen 基础架构）

#### 建议 3: 参数化 MoE 路由

当前 MoE 路由是静态的。添加配置字段:

```python
# 现有的 MRoE 配置
config.moe_enable = True/False

# 建议新增的配置字段
config.num_experts = 8          # 专家数量 (Mixtral=8, DBRX=16, DeepSeek=256)
config.moe_topk = 2             # 每 token 激活专家数
config.num_shared_experts = 0   # 共享专家数 (DeepSeek=1)
config.routing_strategy = "static"  # static | load_balanced | hierarchical
```

**工作量:** ~50 行修改 | **影响:** 无需新代码即可支持 Mixtral、DBRX、Qwen3 MoE 配置

### 7.2 中期（结构变更）

#### 建议 4: 添加 Context Parallelism (CP)

- 新增 `CommGroup.cp_group` 通信组
- 实现 ring attention 的 P2P send/recv 模式
- 适配长序列训练场景（2025+ 模型标准特性）

**工作量:** ~200 行 | **影响:** 支持 LLaMA 4、DeepSeek-V3 风格长序列训练

#### 建议 5: 采用 Chakra 输出格式

- 实现 `ChakraWriter` 类，将 `Workload` 对象转换为 Chakra protobuf
- 使 AICB 生成的 workload 可与 ASTRA-sim、Chakra 生态工具互操作
- STAGE 已证明这条路可行

**工作量:** ~300 行 | **影响:** 与 MLCommons 标准化生态对齐

#### 建议 6: 完成 Qwen3 推理 Workload

`MockedQwen3Moe.py` 中所有类都标记了 `#TODO support Workload`。完成:
- `Qwen3MoeAttention.forward()` / `.backward()`
- `Qwen3MoeRoute.forward()` / `.backward()`
- `Qwen3MoeExpert.forward()` / `.backward()`

**工作量:** ~200 行 | **影响:** 首个完整的非 DeepSeek 推理 workload

### 7.3 战略（长期架构演进）

#### 建议 7: 采用声明式 Tensor Graph 定义（借鉴 STAGE）

核心思路: 将当前命令式的 `forward()`/`backward()` 方法替换为声明式的 tensor graph 定义。

**Phase 1: CSV 格式的操作定义**

```csv
# ffns/llama_swiglu.csv -- Gate-up-down FFN 的 operator graph
op_id,op_type,inputs,output,attrs
0,einsum,[x,gate_weight],gate_hidden,"bm,mn->bn"
1,einsum,[x,up_weight],up_hidden,"bm,mn->bn"
2,activation,[gate_hidden],gate_act,"swiglu"
3,multiply,[gate_act,up_hidden],gated_hidden,""
4,einsum,[gated_hidden,down_weight],ffn_out,"bm,mn->bn"
```

**Phase 2: 声明式模型组装**

```python
# 从 CSV 加载组件，按名称连接 -- 无需为每个新模型写 Python
def llama_model(hidden_size, num_layers, num_kv_heads, ...):
    gqa = TensorGraph.load_tensor_graph("templates/attention/gqa.csv")
    swiglu = TensorGraph.load_tensor_graph("templates/ffn/swiglu.csv")
    rmsnorm = TensorGraph.load_tensor_graph("templates/norm/rmsnorm.csv")
    
    decoder_layer = ConnectGraph.apply([rmsnorm, gqa, rmsnorm, swiglu], links={...})
    model = StackGraph.apply(decoder_layer, num_layers)
    return model
```

**Phase 3: 分离并行策略**

并行策略（TP、SP、CP、EP）通过单独的 sharding spreadsheet 定义，与模型架构解耦:

```
sharding_spreadsheets/
  +-- tp/        # Tensor Parallelism 的分片配置
  +-- tpsp/      # TP + Sequence Parallelism 组合
  +-- fsdp/      # Fully Sharded Data Parallel
  +-- cp/        # Context Parallelism
```

---

## 8. 参考资料

### 学术论文

1. **STAGE: A Symbolic Tensor Graph Generator for Distributed AI System Co-Design**
   - Man et al., Georgia Tech, arXiv:2511.10480, November 2025
   - 证明了符号化 tensor 表示可统一支持 dense、MoE、SSM 架构

2. **Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces**
   - Sridharan et al., MLCommons, arXiv:2305.14516, May 2023
   - 定义了 MLCommons 标准化的执行 trace schema

3. **DeepSeek-V3 Technical Report**
   - DeepSeek-AI, arXiv:2412.19437, December 2024
   - 引入了 MLA attention 和 DeepSeekMoE 架构

### 工具与仓库

4. **aliyun/aicb** -- https://github.com/aliyun/aicb
   - AICB workload generator，SimAI 生态的核心组件

5. **astra-sim/symbolic_tensor_graph** -- https://github.com/astra-sim/symbolic_tensor_graph
   - STAGE 框架实现，包含 GPT 和 LLaMA 模型模板

6. **deepseek-ai/DualPipe** -- https://github.com/deepseek-ai/DualPipe
   - 双向流水线并行算法，2025 年 2 月开源

7. **mlcommons/chakra** -- https://github.com/mlcommons/chakra
   - MLCommons 标准化的执行 trace schema

### 技术博客 / 报告

8. DeepSeek Open Source Week (February 2025): FlashMLA, DeepEP, DeepGEMM, DualPipe+EPLB
9. NVIDIA Hybrid-EP (February 2026): Slashing MoE Training Communication Overhead by 14%
10. FarSkip-Collective (MLSys 2026): Unhobbling Blocking Communication in MoE Models

### 代码引用

11. `aicb/workload_generator/generate_megatron_workload.py` lines 437-440: Model dispatch logic
12. `aicb/workload_generator/mocked_model/MockedModel.py`: Base MockedModel class (169 lines)
13. `aicb/workload_generator/mocked_model/training/MockedMegatron.py`: Megatron model implementation (676 lines)
14. `aicb/workload_generator/mocked_model/inference/MockedQwen3Moe.py`: Qwen3 MoE inference skeleton
15. `aicb/utils/utils.py` lines 544-579: CommType and CommGroup enums
16. STAGE `models/stage1/gpt_model.py`: GPT model definition (declarative, CSV-driven)
17. STAGE `models/stage1/llama_model.py`: LLaMA model definition (declarative, CSV-driven)
