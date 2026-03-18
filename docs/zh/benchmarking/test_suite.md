# 4 场景端到端测试套件

SimAI 提供预配置的测试套件，覆盖 4 个典型推理场景，可快速验证所有支持的配置。

---

## 概述

测试套件位于 `vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh`，覆盖不同模型、并行策略和 PD 分离配置的组合。

---

## 运行

```bash
# 运行所有 4 个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# 运行单个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1

# 显示帮助
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --help
```

> **前置条件**：需激活 `conda activate vidur` 环境。

---

## 共享硬件配置

所有场景共享以下硬件设置：

| 参数 | 值 |
|-----------|-------|
| GPU | H20 (h20_dgx) |
| NVLink 带宽 | 1600 Gbps |
| RDMA 带宽 | 800 Gbps |
| PD P2P 带宽 | 800 Gbps |
| PD P2P 数据类型 | fp8 |
| 请求生成器 | 泊松分布，QPS=100 |
| 请求数量 | 4 |
| Prefill Tokens | 100（固定） |
| Decode Tokens | 8（固定） |

---

## 场景配置

| 场景 | 模型 | PD 分离 | World Size | TP | PP | EP | 全局调度器 |
|----------|-------|---------------|-----------|----|----|-----|-----------------|
| **1** | Qwen3-Next-80B (MoE) | 否 | 32 (dp=32) | 1 | 1 | 1（默认） | lor |
| **2** | Qwen3-Next-80B (MoE) | 是 (P=2, D=6) | 8 | 1 | 1 | 1（默认） | split_wise |
| **3** | DeepSeek-671B (MoE) | 是 (P=2, D=6) | 8 | 8 | 1 | 8 | split_wise |
| **4** | Qwen3-MoE-235B (MoE) | 是 (P=2, D=6) | 8 | 4 | 1 | 4 | split_wise |

### 场景详情

- **场景 1**：大规模 DP 无 PD 分离 — 测试基线吞吐量
- **场景 2**：同模型加 PD 分离 — 测试 PD 分离开销
- **场景 3**：DeepSeek-671B 大 TP/EP — 测试 MoE + MLA 注意力
- **场景 4**：Qwen3-MoE-235B 中等 TP/EP — 测试 MHA/GQA 注意力模型

---

## 输出

### 输出目录

- **通过 run_scenarios.sh**：`examples/vidur-ali-scenarios/simulator_output/`
- **直接 Python 运行**：`./simulator_output/`

### 输出文件

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # 每请求指标
├── chrome_trace.json       # Chrome DevTools 时间线
├── config.json             # 配置快照
└── plots/                  # 指标 CSV/JSON 文件
```

### 日志

运行日志保存在 `examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log`。

---

## 架构对比示例

### RING vs NVLS（SimAI-Simulation）

```bash
# NVLS 拓扑和运行
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 32 -gt H100 -bw 400Gbps -nvbw 1360Gbps
AS_SEND_LAT=12 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100 -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# RING 拓扑和运行
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 32 -gt H100 -bw 400Gbps -nvbw 1440Gbps
AS_SEND_LAT=2 AS_PXN_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100 -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

**结果**（busbw 单位 GB/s）：

| 消息大小 | NVLS | RING |
|-------------|------|------|
| 16M | 148.88 | 141.84 |
| 32M | 178.04 | 153.68 |
| 64M | 197.38 | 160.60 |
| 128M | 208.70 | 163.85 |
| 256M | 214.87 | 165.72 |
| 512M | 218.09 | 166.68 |

### Spectrum-X vs DCN+（SimAI-Simulation）

```bash
# 生成拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo DCN+ -g 256 -psn 64 -bw 400Gbps
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 256
```

**结果**（busbw 单位 GB/s）：

| 消息大小 | Spectrum-X | DCN+ SingleToR |
|-------------|------------|----------------|
| 16M | 33.10 | 23.33 |
| 64M | 42.05 | 23.68 |
| 256M | 45.10 | 36.21 |
| 512M | 45.65 | 36.24 |

---

## 相关文档

- [多请求推理仿真](../user_guide/inference_simulation.md) — 完整推理仿真指南
- [vidur-alibabacloud](../components/vidur.md) — 组件文档
- [结果分析](../user_guide/result_analysis.md) — 输出解读
