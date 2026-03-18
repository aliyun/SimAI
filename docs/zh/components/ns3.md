# ns-3-alibabacloud — 网络仿真后端

**仓库**: [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | **语言**: C++

基于 NS-3 的网络仿真器，作为 SimAI 的网络后端，扩展了面向数据中心/RDMA 的端到端建模能力。

---

## 概述

相比上游 [NS-3](https://www.nsnam.org/)，ns-3-alibabacloud 在点对点模块上扩展了全面的数据中心网络特性：

- **QBB/PFC + 多优先级队列** — 8 个优先级队列，支持 PAUSE/RESUME 处理
- **ECN + CNP 反馈** — 交换机侧 ECN 标记和接收端拥塞通知
- **RDMA 主机协议栈（QP 级别）** — 完整 QP 建模，支持 5 种拥塞控制算法
- **交换机和 NVSwitch 建模** — ECMP 转发、缓冲区管理、PFC 逻辑

### dev/qp 分支

[dev/qp](https://github.com/aliyun/ns-3-alibabacloud/tree/dev/qp) 分支包含额外增强：

1. 基于实际 RDMA 逻辑的 QP 创建/销毁支持
2. 按 IP 或按 QP 的 NIC CC 配置
3. 优化的 Max-Min 调度逻辑
4. 解耦的 CC 模块，提升模块化程度

---

## 核心模块

### QBB 网络设备（`qbb-net-device`）

基于 `PointToPointNetDevice` 构建的支持 QBB 的网络设备，具有 8 个优先级。特性：

- PFC PAUSE/RESUME 处理
- `RdmaEgressQueue`：高优先级 ACK/NACK 队列 + QP 间轮询
- `BEgressQueue`：交换机端口轮询
- NVSwitch 发送路径支持（NVLS 模式）

**关键属性**: `QbbEnabled`、`QcnEnabled`、`DynamicThreshold`、`PauseTime`、`NVLS_enable`

### RDMA 主机协议栈（`rdma-hw`）

主机 RDMA 核心实现：

- QP 创建/删除生命周期
- 报文构造（PPP + IPv4 + UDP + SeqTs 头）
- ACK/NACK/CNP 处理
- 按 QP 的拥塞控制算法
- NVSwitch 路由表

**拥塞控制算法**：

| 算法 | 说明 |
|-----------|-------------|
| **DCQCN** | 数据中心量化拥塞通知 |
| **HPCC** | 高精度拥塞控制 |
| **TIMELY** | 基于 RTT 的拥塞控制 |
| **DCTCP** | 数据中心 TCP |
| **HPCC-PINT** | HPCC + 概率 INT |

**协议号（IPv4 Protocol 字段）**：

| 协议 | 编号 | 说明 |
|----------|--------|-------------|
| UDP 数据 | `0x11` | 普通数据报文 |
| CNP | `0xFF` | 拥塞通知报文 |
| PFC | `0xFE` | 优先级流控 |
| ACK | `0xFC` | 确认报文 |
| NACK | `0xFD` | 否定确认报文 |

### 交换机节点（`switch-node`）

交换机流水线实现：
- ECMP 转发（5 元组哈希）
- 通过 MMU 进行准入控制
- PFC Pause/Resume 生成
- ECN 标记
- INT/PINT 注入（用于 HPCC/HPCC-PINT）

### 交换机 MMU（`switch-mmu`）

交换机缓冲区/MMU 模型：
- 入口/出口记账
- 共享缓冲区和 Headroom 管理
- PFC 触发/恢复逻辑
- ECN 标记概率曲线（`kmin/kmax/pmax`）

### NVSwitch 节点（`nvswitch-node`）

用于服务器内 GPU 通信的 NVSwitch 模型，配合 `RdmaHw`/`QbbNetDevice` 中的 NVLS 路由逻辑。

### QP 状态（`rdma-queue-pair`）

按 QP 和按 RxQP 的状态管理，包括：
- 窗口和速率控制
- 已确认序列号追踪
- 按 CC 算法的状态（DCQCN alpha/targetRate、HPCC hop state、TIMELY RTT、DCTCP alpha/ecnCnt、PINT state）

---

## 分析工具

位于 `ns-3-alibabacloud/analysis/`：

### FCT 分析

```bash
python fct_analysis.py -h  # 查看使用帮助
```

读取 FCT 输出文件，生成流完成时间（FCT）分析统计。

### Trace 阅读器

```bash
# 编译
make trace_reader

# 使用
./trace_reader <.tr 文件> [过滤表达式]

# 过滤示例
./trace_reader output.tr "time > 2000010000"
./trace_reader output.tr "sip=0x0b000101&dip=0x0b000201"
```

### Trace 输出格式

```
2000055540 n:338 4:3 100608 Enqu ecn:0 0b00d101 0b012301 10000 100 U 161000 0 3 1048(1000)
```

字段：时间戳、节点、端口:队列、队列长度、事件、ECN、源 IP、目的 IP、源端口、目的端口、报文类型、序列号、发送时间、优先级、大小(载荷)

---

## 头部和工具

| 文件 | 说明 |
|------|-------------|
| `qbb-header` | ACK/NACK 头（含可选 INT 头） |
| `cn-header` | CNP 头（反馈字段） |
| `pause-header` | PFC Pause 头 |
| `pint` | PINT 编解码工具 |
| `trace-format.h` | 用于离线分析的二进制 Trace 记录结构 |

---

## 扩展指南

### 添加新拥塞控制算法

1. **主要修改**: `rdma-hw.{h,cc}` — 添加 `HandleAckX`/`UpdateRateX` 方法，按 `m_cc_mode` 分发
2. **通常需要**: `rdma-queue-pair.h` — 添加新的按 QP 状态变量
3. **如需交换机反馈**: `switch-node.cc` — 添加 INT/PINT 或新标记

### 修改交换机行为

1. **主要修改**: `switch-mmu.{h,cc}` — 修改阈值、曲线、公式
2. **标记/注入**: `switch-node.cc::SwitchNotifyDequeue()`
3. **准入/优先级**: `switch-node.cc::SendToDev()`

### 添加新控制报文

1. 在 `model/` 中创建新 `*Header`（参照 `CnHeader`/`PauseHeader` 模式）
2. 在 `QbbNetDevice::Receive()` 或 `RdmaHw::Receive()` 中添加解析

---

## 相关文档

- [SimAI-Simulation 使用指南](../user_guide/simai_simulation.md) — 全栈仿真使用
- [astra-sim 组件](astra_sim.md) — 仿真引擎
- [NS-3 扩展指南](../developer_guide/extending_ns3.md) — 详细扩展指南
