# SimAI-Simulation

SimAI-Simulation 使用 NS-3 作为网络后端，提供高保真的全栈仿真和细粒度网络通信建模。适用于集合通信算法、网络协议和新型网络架构的深入研究。

## 适用场景

- **集合通信算法研究**：设计和优化非交换机架构下的流量模式
- **网络协议研究**：评估拥塞控制、路由机制和底层协议
- **新型网络架构设计**：探索创新的网络拓扑和配置

## 工作负载生成

使用与 SimAI-Analytical 相同的工作负载，由 [AICB](workload_generation.md) 生成。

## 网络拓扑配置

运行 SimAI-Simulation 之前，需要生成 ns-3-alibabacloud 识别的拓扑文件。

### 拓扑模板

SimAI 提供 5 种常见架构模板：

| 模板 | 说明 | 默认 GPU 数 |
|------|------|-------------|
| `Spectrum-X` | Rail-optimized，单 ToR，单 Plane | 4096 |
| `AlibabaHPN`（单 Plane） | Rail-optimized，双 ToR，单 Plane | 15360 |
| `AlibabaHPN`（双 Plane） | Rail-optimized，双 ToR，双 Plane | 15360 |
| `DCN+`（单 ToR） | 非 Rail-optimized，单 ToR | 512 |
| `DCN+`（双 ToR） | 非 Rail-optimized，双 ToR | 512 |

![Spectrum-X](../../images/Spectrum-X.jpg)

### 生成拓扑

```bash
# 8 GPU 的 Spectrum-X 拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo Spectrum-X -g 8 -psn 1

# 64 GPU 的双 Plane AlibabaHPN 拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo AlibabaHPN --dp -g 64 -asn 16 -psn 16

# 128 GPU 的双 ToR DCN+ 拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo DCN+ --dt -g 128 -asn 2 -psn 8

# 自定义 Rail-optimized 拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -g 32 -bw 200Gbps -gt A100 -psn 8 --ro
```

### 拓扑参数

| 层级 | 参数 | 说明 |
|------|------|------|
| 整体结构 | `-topo` | 模板名称 |
| | `-g` | GPU 数量 |
| | `--dp` | 启用双 Plane |
| | `--ro` | 启用 Rail-optimized |
| | `--dt` | 启用双 NIC 和双 ToR |
| | `-er` | 错误率 |
| 服务器内 | `-gps` | 每服务器 GPU 数 |
| | `-gt` | GPU 类型 |
| | `-nvbw` | NVLink 带宽 |
| | `-nl` | NVLink 延迟 |
| | `-l` | NIC 延迟 |
| Segment 内 | `-bw` | NIC 到 ASW 带宽 |
| | `-asw` | ASW 交换机数量 |
| | `-nps` | 每交换机 NIC 数 |
| Pod 内 | `-psn` | PSW 交换机数量 |
| | `-apbw` | ASW 到 PSW 带宽 |
| | `-app` | 每 PSW 的 ASW 数 |

> 详细拓扑参数和各模板默认值请参见 [astra-sim 组件文档](../components/astra_sim.md)。

## 运行 NS-3 仿真

```bash
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
    -t 16 \
    -w ./example/microAllReduce.txt \
    -n ./Spectrum-X_8g_8gps_400Gbps_H100 \
    -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AS_LOG_LEVEL` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` | `INFO` |
| `AS_PXN_ENABLE` | 启用 PXN（`0`/`1`） | `0` |
| `AS_NVLS_ENABLE` | 启用 NVLS（`0`/`1`） | `0` |
| `AS_SEND_LAT` | 数据包发送延迟（us） | `6` |
| `AS_NVLSTREE_ENABLE` | 启用 NVLSTREE | `false` |

### 仿真参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t` / `--thread` | 线程数（推荐 8-16） | `1` |
| `-w` / `--workload` | 工作负载文件路径 | `./microAllReduce.txt` |
| `-n` / `--network-topo` | 网络拓扑路径 | 无 |
| `-c` / `--config` | SimAI 配置文件 | 无 |

## 示例：RING vs NVLS 对比

请参阅 [Tutorial](../../docs/Tutorial.md#ring-vs-nvls) 了解 RING 和 NVLS 算法在不同消息大小下的完整对比。

## 延伸阅读

- [NS-3 组件文档](../components/ns3.md) — NS-3 模块详细参考
- [结果分析](result_analysis.md) — 仿真输出分析
