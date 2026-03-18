# astra-sim-alibabacloud — 仿真引擎

**位置**: 项目内（`astra-sim-alibabacloud/`） | **语言**: C++

SimAI 的核心仿真引擎，扩展自 [astra-sim 1.0](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0)。支持三种运行模式，并集成了 NCCL 算法与自定义增强。

---

## 概述

astra-sim-alibabacloud 是 SimAI 仿真的中央调度器：

- 接收 AICB 生成的工作负载
- 使用 SimCCL 将集合操作分解为 P2P 传输
- 通过 NS-3（仿真模式）或直接 RDMA（物理模式）驱动网络仿真
- 使用 busbw 参数进行时间计算（分析模式）

---

## 三种运行模式

### SimAI-Analytical

使用总线带宽（busbw）进行快速分析仿真，估算集合通信耗时。

**编译**: `./scripts/build.sh -c analytical`
**二进制**: `bin/SimAI_analytical`

### SimAI-Simulation

使用 NS-3 网络后端的全栈仿真，实现细粒度网络建模。

**编译**: `./scripts/build.sh -c ns3`
**二进制**: `bin/SimAI_simulator`

### SimAI-Physical

在真实硬件上使用 RDMA 生成物理流量。

**编译**: `./scripts/build.sh -c phy`
**二进制**: `bin/SimAI_phynet`

---

## 核心组件

| 组件 | 说明 |
|-----------|-------------|
| **AstraComputeAPI** | 管理计算时序和调度 |
| **MemoryAPI** | 处理内存分配和追踪 |
| **NetworkAPI** | 网络后端接口（NS-3、物理网络） |
| **MockNcclGroup** | 模拟 NCCL 通信组 |
| **MockNcclChannel** | 管理单个通信通道 |
| **SimAiFlowModelRdma** | RDMA 流量模型 |

---

## 配置

### SimAI.conf

主配置文件位于 `astra-sim-alibabacloud/inputs/config/SimAI.conf`，控制以下仿真参数：

- 通信算法
- 缓冲区大小
- 时序参数
- 网络后端设置

### 环境变量（仿真模式）

| 变量 | 说明 | 默认值 |
|----------|-------------|---------|
| `AS_LOG_LEVEL` | 日志级别：DEBUG、INFO、WARNING、ERROR | `INFO` |
| `AS_PXN_ENABLE` | 启用 PXN（Proxied NVLINK） | `0`（禁用） |
| `AS_NVLS_ENABLE` | 启用 NVLS（NVLink Sharp） | `0`（禁用） |
| `AS_SEND_LAT` | 包发送延迟（us） | `6` |
| `AS_NVLSTREE_ENABLE` | 启用 NVLS Tree 算法 | `false` |

### 仿真参数

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `-t` / `--thread` | 加速线程数 | `1`（建议 8-16） |
| `-w` / `--workload` | 工作负载文件路径 | 必需 |
| `-n` / `--network-topo` | 网络拓扑文件路径 | 必需（仿真模式） |
| `-c` / `--config` | SimAI 配置文件 | 必需 |

---

## 拓扑生成

astra-sim 通过 `gen_Topo_Template.py` 提供 5 种拓扑模板：

### 可用模板

| 模板 | 架构 | 说明 |
|----------|-------------|-------------|
| `Spectrum-X` | NVIDIA Spectrum-X | Rail-optimized，单 ToR，单 Plane |
| `AlibabaHPN`（单 Plane） | Alibaba HPN 7.0 | 双 ToR，Rail-optimized，单 Plane |
| `AlibabaHPN`（双 Plane） | Alibaba HPN 7.0 | 双 ToR，Rail-optimized，双 Plane |
| `DCN+`（单 ToR） | DCN+ | 单 ToR，非 Rail-optimized |
| `DCN+`（双 ToR） | DCN+ | 双 ToR，非 Rail-optimized |

### 拓扑参数

| 层级 | 参数 | 说明 |
|-------|-----------|-------------|
| **全局** | `-topo` | 模板名称 |
| | `-g` | GPU 数量 |
| | `--dp` | 启用双 Plane |
| | `--ro` | 启用 Rail-optimized |
| | `--dt` | 启用双 ToR |
| **服务器内** | `-gps` | 每服务器 GPU 数 |
| | `-gt` | GPU 型号（A100/H100） |
| | `-nvbw` | NVLink 带宽 |
| | `-nl` | NVLink 延迟 |
| **Segment 内** | `-bw` | NIC 到 ASW 带宽 |
| | `-asw` | ASW 交换机数量 |
| | `-nps` | 每交换机 NIC 数 |
| **Pod 内** | `-psn` | PSW 交换机数量 |
| | `-apbw` | ASW 到 PSW 带宽 |

### 示例

```bash
# Spectrum-X 128 GPU
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# 双 Plane AlibabaHPN 64 GPU
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo AlibabaHPN --dp -g 64 -asn 16 -psn 16

# 双 ToR DCN+ 128 GPU
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo DCN+ --dt -g 128 -asn 2 -psn 8
```

---

## 相关文档

- [SimAI-Analytical 使用指南](../user_guide/simai_analytical.md) — 分析模式使用
- [SimAI-Simulation 使用指南](../user_guide/simai_simulation.md) — NS-3 仿真使用
- [SimAI-Physical 使用指南](../user_guide/simai_physical.md) — 物理模式使用
- [NS-3 组件](ns3.md) — 网络后端详情
- [SimCCL 组件](simccl.md) — 集合通信分解
