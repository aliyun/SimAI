# 结果分析与可视化

本指南介绍如何解读和分析 SimAI 各模式的仿真输出。

---

## SimAI-Analytical 输出

### CSV 输出

运行 SimAI-Analytical 后会在 `results/` 目录生成 CSV 文件，包含：

- **汇总行**：暴露时间、各通信组的计算时间（绝对值和百分比）、端到端迭代时间
- **逐层行**：每层的详细操作时间

关键列包含各通信组（TP、DP、EP、PP）的分解，展示时间分配和重叠效果。

### 可视化

使用 `-v` 参数运行时，SimAI-Analytical 会生成额外的可视化文件，展示各通信组的时间分解。

```bash
# 启用可视化运行
./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml -v
```

---

## SimAI-Simulation 输出

SimAI-Simulation（NS-3 模式）生成详细的追踪数据，捕获细粒度的网络行为。NS-3 后端输出 `.tr` 追踪文件，可使用提供的分析工具进行分析。

### 分析工具

位于 `ns-3-alibabacloud/analysis/`：

| 工具 | 说明 |
|------|------|
| `fct_analysis.py` | 流完成时间（FCT）分析——读取 FCT 输出文件并生成统计数据 |
| `trace_reader` | 解析 `.tr` 追踪文件，支持过滤 |

### 使用 trace_reader

```bash
# 编译
cd ns-3-alibabacloud/analysis
make trace_reader

# 解析追踪文件
./trace_reader <.tr 文件> [过滤表达式]

# 示例：
./trace_reader output.tr "time > 2000010000"
./trace_reader output.tr "sip=0x0b000101&dip=0x0b000201"
```

### 追踪输出格式

追踪输出每行格式如下：

```
2000055540 n:338 4:3 100608 Enqu ecn:0 0b00d101 0b012301 10000 100 U 161000 0 3 1048(1000)
```

字段：时间戳（ns）、节点 ID、端口:队列、队列长度（字节）、事件类型、ECN 标志、源 IP、目的 IP、源端口、目的端口、包类型、序列号、发送时间戳、优先级组、包大小（有效负载）。

---

## 推理仿真输出

### 输出目录结构

每次推理仿真运行生成：

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # 逐请求指标
├── chrome_trace.json       # Chrome DevTools 时间线追踪
├── config.json             # 配置快照
└── plots/                  # 各指标 CSV/JSON 文件
    ├── request_e2e_time.csv
    ├── prefill_e2e_time.csv
    ├── pd_p2p_comm_time.csv
    ├── replica_N_memory_usage.json
    └── ...
```

### request_metrics.csv 列说明

| 列名 | 含义 |
|------|------|
| `arrived_at` | 请求进入系统的时间戳（秒） |
| `scheduled_at` | 请求首次被调度的时间戳（秒） |
| `prefill_completed_at` | Prefill 完成并生成第一个 token 的时间戳 |
| `decode_arrived_at` | Decode 阶段开始的时间戳 |
| `decode_time` | Decode 阶段持续时间（秒） |
| `prefill_replica_id` | 执行 Prefill 的副本 ID（PD 模式） |
| `decode_replica_id` | 执行 Decode 的副本 ID（PD 模式） |
| `request_num_prefill_tokens` | 输入 token 数（prompt 长度） |
| `request_num_decode_tokens` | 输出 token 数（生成长度） |
| `pd_p2p_comm_size` | Prefill 到 Decode 节点的 P2P 通信大小（字节） |
| `pd_p2p_comm_time` | P2P 通信时间（秒） |
| `completed_at` | 请求完成时间戳 |
| `request_execution_time` | 总执行时间（不含延迟，秒） |
| `request_preemption_time` | 因抢占/气泡导致的等待时间（秒） |
| `request_scheduling_delay` | 调度延迟：`scheduled_at - arrived_at`（秒） |
| `request_e2e_time` | 端到端延迟：`completed_at - arrived_at`（秒） |
| `prefill_e2e_time` | 首 token 时间（TTFT）：`prefill_completed_at - arrived_at`（秒） |
| `tbt` | token 间时间：`decode_time / request_num_decode_tokens`（秒/token） |

### Chrome Trace 可视化

在 Chrome DevTools 中打开 `chrome_trace.json` 进行可视化时间线分析：

1. 打开 Chrome 浏览器
2. 访问 `chrome://tracing`
3. 加载 `chrome_trace.json` 文件

### 仿真指标（23 项）

仿真器记录 23 项细粒度指标：

| 类别 | 指标 |
|------|------|
| **请求延迟** | E2E 时间 CDF、归一化 E2E CDF、执行+抢占 CDF |
| **调度** | 调度延迟 CDF |
| **执行** | 执行时间 CDF、抢占时间 CDF |
| **Token 级** | Decode token 执行+抢占时间、token 间延迟 |
| **批次** | 批次 token 数 CDF、批次大小 CDF |
| **Prefill** | Prefill E2E CDF、Prefill 执行+抢占 CDF（归一化） |
| **Decode** | Decode 执行+抢占归一化 CDF |
| **时间序列** | 请求/Prefill/Decode 完成、请求到达 |
| **逐副本** | 显存使用（加权均值）、繁忙时间百分比、MFU |

详细指标定义请参见 [vidur 指标文档](../components/vidur.md)。

---

## AICB 物理执行输出

### 日志输出

每次通信后，AICB 输出：
- 通信类型和组
- 消息大小
- 执行时间
- 吞吐量（algbw 和 busbw）

### 迭代汇总

所有通信完成后，汇总显示：
- 总运行时间和每次迭代的时间
- 按通信类型的统计（消息大小、频率、延迟最小/最大/平均值）

### CSV 输出

结果保存在 `results/comm_logs/`：
- `<模型>_<配置>_log.csv` — 执行日志（包含时间、阶段、algbw、busbw 等）
- `<模型>_<配置>_workload.csv` — 生成的工作负载描述

### 编程分析

```python
# 读取工作负载日志
from log_analyzer.log import Workload
workload, args = Workload.load("results/comm_logs/megatron_gpt_13B_8n_workload.csv")

# 读取执行日志
from log_analyzer.log import Log
log = Log.load("results/comm_logs/megatron_gpt_13B_8n_log.csv")
# log.comm_logs: List[LogItem]
# log.epoch_times: List[int]
# log.comm_log_each_epoch: List[List[LogItem]]
```

---

## 相关文档

- [SimAI-Analytical](simai_analytical.md) — Analytical 模式用法
- [SimAI-Simulation](simai_simulation.md) — NS-3 仿真模式用法
- [推理仿真](inference_simulation.md) — 推理仿真指南
- [NS-3 组件](../components/ns3.md) — NS-3 分析工具
