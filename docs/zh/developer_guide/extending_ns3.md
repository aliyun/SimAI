# NS-3 网络后端扩展指南

本指南介绍如何扩展 `ns-3-alibabacloud`，包括新增拥塞控制算法、交换机行为、控制报文和 NVSwitch 特性。

> **源码参考**：详见 `astra-sim-alibabacloud/extern/network_backend/ns3-interface/README.md` 获取完整模块映射。

---

## 模块概览

所有关键源文件位于 `ns-3-alibabacloud/simulation/src/point-to-point/model/`：

| 文件 | 类 | 用途 |
|------|-------|---------|
| `qbb-net-device.{h,cc}` | `QbbNetDevice`, `RdmaEgressQueue` | 支持 QBB 的 NIC，8 优先级队列，PFC 处理，NVSwitch 发送路径 |
| `rdma-hw.{h,cc}` | `RdmaHw` | 主机 RDMA 核心：QP 管理、报文构造、ACK/NACK、CC 算法 |
| `rdma-queue-pair.{h,cc}` | `RdmaQueuePair`, `RdmaRxQueuePair` | 按 QP 状态（窗口、速率、CC 特定状态） |
| `switch-node.{h,cc}` | `SwitchNode` | 交换机流水线：ECMP 转发、ECN 标记、PFC、INT/PINT 注入 |
| `switch-mmu.{h,cc}` | `SwitchMmu` | 交换机缓冲区/MMU：入口/出口记账、PFC 阈值、ECN 曲线 |
| `nvswitch-node.{h,cc}` | `NVSwitchNode` | 服务器内 GPU 通信的 NVSwitch 模型 |
| `rdma-driver.{h,cc}` | `RdmaDriver` | Node/NIC 与 RdmaHw 之间的连接层 |
| `qbb-header.{h,cc}` | — | ACK/NACK 头（PG/seq/CNP-flag + INT 头） |
| `cn-header.{h,cc}` | — | CNP 头（反馈字段） |
| `pause-header.{h,cc}` | — | PFC Pause 头 |
| `pint.{h,cc}` | — | PINT 编解码工具 |
| `trace-format.h` | `TraceFormat` | 用于离线分析的二进制 Trace 记录结构 |

---

## 添加新拥塞控制算法

NS-3 后端内置 5 种 CC 算法：**DCQCN**、**HPCC**、**TIMELY**、**DCTCP** 和 **HPCC-PINT**。添加新算法步骤：

### 步骤 1：定义 CC 模式

在 `rdma-hw.h` 中添加新的 `CcMode` 值：

```cpp
// 现有模式：1=DCQCN, 3=HPCC, 7=TIMELY, 8=DCTCP, 10=HPCC-PINT
static const uint32_t CC_MODE_YOUR_ALG = 11;
```

### 步骤 2：添加按 QP 状态（如需）

在 `rdma-queue-pair.h` 中为 `RdmaQueuePair` 添加新状态变量：

```cpp
// 您的 CC 算法状态
double m_your_alg_rate;
double m_your_alg_alpha;
// ...
```

### 步骤 3：实现算法逻辑

在 `rdma-hw.cc` 中添加两个关键函数：

```cpp
void RdmaHw::HandleAckYourAlg(Ptr<RdmaQueuePair> qp, ...) {
    // 处理 ACK 并更新速率/窗口
}

void RdmaHw::UpdateRateYourAlg(Ptr<RdmaQueuePair> qp, ...) {
    // 速率更新逻辑
}
```

### 步骤 4：注册分发

在 `rdma-hw.cc` 的 `ReceiveAck()` 和/或 `ReceiveCnp()` 中添加分发：

```cpp
switch (m_cc_mode) {
    // ... 现有 case ...
    case CC_MODE_YOUR_ALG:
        HandleAckYourAlg(qp, ...);
        break;
}
```

### 步骤 5：添加交换机反馈（如需）

如果您的 CC 算法需要交换机侧信息（如 INT/PINT 元数据）：

- 修改 `switch-node.cc::SwitchNotifyDequeue()` 注入元数据
- 在 `RdmaHw::Receive()` 或 `QbbNetDevice::Receive()` 中添加头部解析

---

## 修改交换机行为

### 缓冲区管理 / PFC 阈值

**主要文件**: `switch-mmu.{h,cc}`

关键修改方法：

| 方法 | 用途 |
|--------|---------|
| `ConfigBufferSize()` | 总缓冲池大小 |
| `ConfigHdrm()` | Headroom 分配 |
| `ConfigEcn()` | ECN 标记阈值（`kmin`、`kmax`、`pmax`） |
| `CheckIngressAdmission()` | 入口准入控制 |
| `CheckEgressAdmission()` | 出口准入控制 |
| `GetPfcThreshold()` | PFC 触发阈值公式 |

### ECN 标记 / INT 注入

**文件**: `switch-node.cc`

修改 `SwitchNotifyDequeue()` 实现：
- 基于自定义队列占用公式的 ECN 标记
- 用于高级 CC 算法的 INT/PINT 元数据注入
- 自定义报文标记

### 转发 / ECMP

**文件**: `switch-node.cc`

路由修改：
- `GetOutDev()` — 输出端口选择
- `EcmpHash()` — ECMP 哈希函数（当前为 5 元组）
- `AddTableEntry()` — 路由表管理

---

## 引入新控制报文

### 步骤 1：创建头部

在 `model/` 中创建新头部文件，参照 `CnHeader` 或 `PauseHeader` 模式：

```cpp
// your-header.h
class YourHeader : public Header {
public:
    static TypeId GetTypeId();
    // 序列化/反序列化方法
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    
    // 头部字段
    uint32_t m_your_field;
};
```

### 步骤 2：定义协议号

添加新协议号（遵循现有约定）：

```cpp
// 现有协议号（IPv4 Protocol 字段）：
// UDP 数据:  0x11
// CNP:       0xFF
// PFC:       0xFE
// ACK:       0xFC
// NACK:      0xFD
// 新协议:    0xFB（示例）
```

### 步骤 3：添加解析/分发

在以下位置添加报文处理：
- `QbbNetDevice::Receive()` — 设备级解析
- `RdmaHw::Receive()` — 主机协议栈处理

---

## NVSwitch / NVLS 扩展

**文件**: `nvswitch-node.{h,cc}`、`qbb-net-device.{h,cc}`（NVLS 发送路径）、`rdma-hw.{h,cc}`（NVLS 路由）

`NVSwitchNode` 模拟通过 NVSwitch 的服务器内 GPU 通信。扩展方式：

- **转发**：类似 `SwitchNode` 但不包含 ECN/INT 注入
- **NVLS 路由**：修改 `RdmaHw::GetNicIdxOfQp()` 和 `GetNicIdxOfRxQp()` 以适配 NVSwitch 路由表
- **QP 重分配**：`RdmaHw::RedistributeQp()` 用于 NVSwitch 链路间负载均衡

---

## 分析工具

`ns-3-alibabacloud/analysis/` 目录包含 Trace 分析工具：

| 工具 | 用途 |
|------|---------|
| FCT 分析 | 从仿真 Trace 分析流完成时间 |
| Trace 阅读器 | 解析二进制 `TraceFormat` 记录 |
| 带宽分析 | 按链路的带宽利用率随时间变化 |
| 队列分析 | 队列占用和 PFC 事件分析 |
| QP 分析 | 按 QP 的性能指标 |

### Trace 格式

二进制 Trace 记录结构（`trace-format.h`）捕获按报文的事件。使用离线分析工具：

1. 解析仿真输出的 Trace 文件
2. 计算 FCT、吞吐量、队列深度统计
3. 识别拥塞热点和 PFC 事件

---

## dev/qp 分支增强

[dev/qp](https://github.com/aliyun/ns-3-alibabacloud/tree/dev/qp) 分支包含：

1. **QP 逻辑支持** — 基于实际 RDMA 逻辑的 QP 创建/销毁
2. **NIC CC 配置** — 按 IP 或按 QP 的 CC 设置
3. **优化调度** — Max-Min 原则的公平资源分配
4. **解耦 CC 模块** — 提升模块化程度

---

## 相关文档

- [NS-3 组件](../components/ns3.md) — 完整 NS-3 后端文档
- [SimAI-Simulation 使用指南](../user_guide/simai_simulation.md) — NS-3 仿真模式使用
- [配置文件参考](../technical_reference/configuration.md) — 拓扑和配置文件
