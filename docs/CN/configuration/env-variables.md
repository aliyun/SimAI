# SimAI 环境变量参考

> [English Version](../../configuration/env-variables.md)

## 仿真环境变量

以下变量控制 `SimAI_simulator`（ns3 模式）的行为。

| 变量 | 默认值 | 单位 | 作用域 | 说明 |
|---|---|---|---|---|
| `AS_SEND_LAT` | 未设置（使用表查询） | **纳秒 (ns)** | entry.h（ns3 前端） | 覆盖每个 flow 的发送延迟。最高优先级——覆盖所有 send_lat_table 查询。 |
| `AS_NVLS_ENABLE` | `0` | - | MockNcclGroup.cc | 启用 NVLS 算法用于 AllReduce（H20/H100/H800） |
| `AS_NVLSTREE_ENABLE` | `0` | - | MockNcclGroup.cc | 启用 NVLS Tree 算法 |
| `AS_PXN_ENABLE` | `0` | - | MockNcclGroup.cc | 启用 PXN 跨节点代理 |
| `AS_LOG_LEVEL` | `INFO` | - | 系统全局 | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |

### AS_SEND_LAT 详细说明

**单位：纳秒 (ns)**。send_lat_table 默认值范围为 6000 到 22000 ns，取决于算法、协议和链路类型。

| 示例值 | 含义 |
|---|---|
| 未设置 | 使用 send_lat_table[algo][proto]（Ring+LL+NVLINK=7200ns，PAT+Simple+NET=22000ns 等） |
| `AS_SEND_LAT=6` | 6 ns — 实际上等于禁用发送延迟（用于快速测试） |
| `AS_SEND_LAT=7200` | 7200 ns = 7.2 μs — 匹配 Ring+LL+NVLINK 表值 |
| `AS_SEND_LAT=22000` | 22000 ns = 22 μs — 匹配 PAT+Simple+NET 表值 |

> **警告**：部分旧文档（Tutorial.md）将 AS_SEND_LAT 描述为“单位 μs，默认 6”。这是不准确的。实际单位为**纳秒**，已通过代码分析和实验验证确认。

#### 快速命令

```bash
# 前置
cd SimAI/ && ./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# 基线（使用 send_lat_table）
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# 覆盖发送延迟（7200ns = Ring+LL+NVLINK 表值）
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

完整 A/B 实验指南请见 [快速入门](../getting_started/quickstart.md#5-验证-as_send_lat-效果ab-实验)。

## SimCCL 集成变量

SimCCL（集合通信建模层）有其自身的环境变量。详见：
- [SimCCL 环境变量](../../../SimCCL/docs/CN/configuration/env-variables.md) — Standalone 变量
- [SimCCL 集成指南](../../../SimCCL/docs/CN/integration/integration-with-simai.md) — 跨模块变量

## 与 send_lat_table 的关系

`entry.h` 中的 `send_lat_table` 为每个（算法、协议、链路类型）组合提供发送延迟值。当未设置 `AS_SEND_LAT` 时，每个 flow 使用与其算法和协议匹配的表值。设置后，所有 flow 统一使用覆盖值。

表机制的详细分析请参见 [send-lat-analysis.md](send-lat-analysis.md)。

---

> 最后编辑：2026-06-25
