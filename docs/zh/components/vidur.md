# vidur-alibabacloud — LLM 推理仿真

**位置**: 项目内（`vidur-alibabacloud/`） | **语言**: Python | **许可**: MIT

vidur-alibabacloud 是 SimAI 的 LLM 推理仿真组件，改编自微软 [Vidur](https://github.com/microsoft/vidur)。提供多请求推理调度、GPU 显存管理和 Prefill-Decode（PD）分离支持。

---

## 核心特性

- **Prefill-Decode（PD）分离** — 在不同节点上运行 Prefill 和 Decode 阶段，实现弹性资源分配和性能隔离。灵感来自 [splitwise-sim](https://github.com/Mutinifni/splitwise-sim)
- **灵活的并行策略** — 数据并行（DP）、张量并行（TP）、流水线并行（PP）、专家并行（EP）
- **多种执行后端** — AICB/AIOB、SimAI Simulation (NS-3)、SimAI Analytical、原生 Vidur
- **工作负载生成与回放** — 合成请求（固定/泊松分布）或真实 Trace 回放
- **细粒度指标** — TTFT、TBT/TPOT、端到端延迟、通信开销、计算开销、调度延迟

---

## GPU 显存计算模块

该模块为推理场景下的 MoE 模型提供精确的 GPU 显存估算。

### 组件

| 组件 | 文件 | 说明 |
|-----------|------|-------------|
| **ParamCounter** | `vidur/utils/param_counter.py` | 按层和按设备的参数计数，支持 MLA、MHA/GQA、线性注意力和 MoE 专家。PD 分离下返回 `(total_params, prefill_params, decode_params)` |
| **MemoryPlanner** | `vidur/scheduler/utils/memory_planner.py` | 规划 GPU 显存预算：`可用 = GPU显存 * (1 - margin) - 参数显存`，计算 KV Cache 容量和最大并发请求数。含 OOM 检测 |
| **按请求 KV Cache 追踪** | `vidur/entities/replica.py` | 按请求分配/释放 KV Cache 显存，支持运行时剩余容量查询 |

### 支持的注意力架构

| 架构 | 模型 | 说明 |
|---|---|---|
| **MLA**（多头潜注意力） | DeepSeek-V3-671B | LoRA 压缩 KV Cache（`kv_lora_rank` + `qk_rope_head_dim`），相比 MHA 节省约 57 倍显存 |
| **MHA / GQA** | Qwen3-MoE-235B | 标准 KV Cache，每 Token 每层 `num_kv_heads * head_dim` |
| **混合全注意力 + 线性注意力** | Qwen3-Next-80B | 全注意力与线性（GDN）注意力每 4 层交替 |

---

## 支持的模型

| 模型 | 注意力 | 专家 | 状态 |
|-------|-----------|---------|--------|
| DeepSeek-V3-671B | MLA | 256 路由 + 1 共享 | PP/EP 适配中 |
| Qwen3-MoE-235B | MHA/GQA | 128 路由 | PP/EP 适配中 |
| Qwen3-Next-80B | 混合 | 512 路由 | PP/EP 适配中 |
| Meta-Llama-3-8B / 70B | MHA | 稠密 | 已支持 |
| Llama-2-7b / 70b | MHA | 稠密 | 已支持 |
| CodeLlama-34b | MHA | 稠密 | 已支持 |
| InternLM-20B | MHA | 稠密 | 已支持 |
| Qwen-72B | MHA | 稠密 | 已支持 |

---

## 环境搭建

### Docker（推荐）

```bash
docker build -t simai:latest .
docker run --gpus all -it --rm simai:latest
```

> 在 Hopper GPU 上使用时，在 Dockerfile 中添加 `ENV FLASH_MLA_DISABLE_SM100=1`。

### Conda

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt
```

---

## 关键输入参数

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `--replica_config_model_name` | HuggingFace 模型 ID 或配置路径 | 必需 |
| `--cluster_config_num_replicas` | 副本数量（DP） | 1 |
| `--replica_config_tensor_parallel_size` | TP 度 | 1 |
| `--replica_config_num_pipeline_stages` | PP 阶段数 | 1 |
| `--replica_config_expert_model_parallel_size` | EP 度 | 1 |
| `--replica_config_pd_node_ratio` | P:D 节点比例（如 `"2:6"`） | `""`（无 PD） |
| `--cluster_config_global_scheduler_type` | 全局调度器：`lor` / `round_robin` / `split_wise` | `lor` |
| `--cluster_config_replica_scheduler_type` | 副本调度器：`sarathi` / `split_wise` | `sarathi` |
| `--request_generator_config_type` | `synthetic` / `trace_replay` | `synthetic` |
| `--synthetic_request_generator_config_num_requests` | 生成请求数 | 100 |
| `--poisson_request_generator_config_qps` | 每秒请求数（泊松模式） | 1.0 |
| `--replica_config_device` | GPU 型号（如 `h20_dgx`） | 必需 |
| `--replica_config_network_device` | 网络类型 | 与 device 相同 |
| `--execution_time_predictor_config_type` | 后端：`aicb` / `simai_simulation` / `simai_analytical` / `random_forrest` | `random_forrest` |
| `--nvlink_bandwidth_gbps` | NVLink 带宽 | 1600 |
| `--rdma_bandwidth_gbps` | RDMA 带宽 | 800 |
| `--pd_p2p_bandwidth_gbps` | PD 节点间 P2P 带宽 | 800 |
| `--replica_config_fp8_enabled` | 启用 FP8 量化 | false |
| `--replica_config_memory_margin_fraction` | GPU 显存安全余量 | 0.1 |

---

## 输出文件

每次运行产生以下输出：

| 文件 | 说明 |
|------|-------------|
| `request_metrics.csv` | 每请求指标（17 列） |
| `chrome_trace.json` | 时间线 Trace，可在 Chrome `chrome://tracing` 中可视化 |
| `config.json` | 配置快照 |
| `plots/` | 指标可视化图表 |

### request_metrics.csv 列说明

| 列名 | 说明 |
|--------|-------------|
| `request_id` | 请求唯一标识 |
| `arrived_at` | 请求到达时间 |
| `scheduled_at` | 首次调度时间 |
| `completed_at` | 请求完成时间 |
| `prefill_completed_at` | Prefill 完成时间（首 Token） |
| `num_prefill_tokens` | 输入 Token 数 |
| `num_decode_tokens` | 生成 Token 数 |
| `scheduling_delay` | 调度前等待时间 |
| `e2e_time` | 端到端延迟 |
| `e2e_time_normalized` | E2E 延迟 / num_decode_tokens |
| `execution_time` | 实际 GPU 执行时间 |
| `preemption_time` | 被抢占时间 |
| `num_restarts` | 重启次数 |
| `prefill_e2e_time` | TTFT（首 Token 时间） |
| `decode_time_normalized` | 平均 TBT（Token 间隔时间） |
| `total_comm_cost` | 总通信耗时 |
| `total_compute_cost` | 总计算耗时 |

---

## 仿真指标（23 项）

仿真器记录以下指标（详见 `vidur-alibabacloud/docs/metrics.md`）：

1. `request_inter_arrival_delay_histogram` — 请求到达间隔分布
2. `request_num_tokens_histogram` — Token 数量分布（Prefill + Decode）
3. `request_num_restarts_histogram` — 重启次数分布
4. `request_e2e_time_cdf` — 端到端延迟 CDF
5. `request_e2e_time_normalised_cdf` — 归一化 E2E 延迟 CDF
6. `request_execution_plus_preemption_times_cdf` — 执行 + 抢占时间 CDF
7. `request_scheduling_delay_cdf` — 调度延迟 CDF
8. `request_execution_time_cdf` — 纯执行时间 CDF
9. `request_preempted_time_cdf` — 抢占时间 CDF
10. `decode_token_execution_plus_preemption_times` — 按 Token 的 inter-token 延迟 CDF
11. `batch_num_tokens_cdf` — 批次总 Token 数 CDF
12. `batch_sizes_cdf` — 批次大小 CDF
13. `prefill_time_e2e_cdf` — TTFT CDF
14. `prefill_time_execution_plus_preemption_cdf` — Prefill 处理时间 CDF
15. `prefill_time_execution_plus_preemption_normalized_cdf` — 归一化 Prefill 时间 CDF
16. `decode_time_execution_plus_preemption_normalized_cdf` — 归一化 Decode 时间 CDF
17. `request_completions_time_series` — 请求完成时间序列
18. `prefill_completions_time_series` — Prefill 完成时间序列
19. `decode_completions_time_series` — Decode 完成时间序列
20. `replica_{id}_memory_usage_weighted_mean` — 按副本显存利用率
21. `replica_{id}_stage_{id}_busy_time_percent_weighted_mean` — 按阶段忙碌时间百分比
22. `replica_{id}_stage_{id}_mfu_weighted_mean` — 按阶段 MFU
23. `request_arrivals_time_series` — 请求到达时间序列

---

## 4 场景测试套件

运行所有预配置场景：

```bash
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all
# 或运行单个场景：
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 3
```

场景配置详情请参阅 [基准测试 — 测试套件](../benchmarking/test_suite.md)。

---

## 添加新模型

如需为 vidur-alibabacloud 添加新模型支持，请参阅 [添加新模型指南](../developer_guide/adding_models.md) 和上游文档 `vidur-alibabacloud/docs/profiling.md`。

---

## 相关文档

- [多请求推理仿真](../user_guide/inference_simulation.md) — 端到端推理仿真工作流
- [结果分析](../user_guide/result_analysis.md) — 输出文件解读
- [GPU 显存模块技术参考](../technical_reference/memory_module.md) — 详细显存计算公式
- [基准测试套件](../benchmarking/test_suite.md) — 4 场景配置详情
