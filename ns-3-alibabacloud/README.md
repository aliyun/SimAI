# NS-3-ALIBABACLOUD

This repository contains an NS3-based network simulator that acts as a network backend for [SimAI](https://github.com/aliyun/SimAI).

We have released a new dev branch [**dev/qp**](https://github.com/aliyun/ns-3-alibabacloud/tree/dev/qp) featuring the following enhancements (From maintainer [**@MXtremist**](https://github.com/MXtremist)):
1. **QP Logic Support**: Enables creation and destruction of QPs based on actual RDMA logic, allowing multiple messages to be carried by a pair of QPs.
2. **NIC CC Configuration**: Supports perIP or perQP settings for enhanced flexibility.
3. **Optimized Scheduling Logic**: Adheres to the Max-Min principle, resolving issues of underutilization and unfairness in network resource allocation.
4. **Decoupling of the CC Module**: For improved modularity and efficiency.

## Key differences vs upstream ns-3 (focus: `simulation/src/point-to-point/model/`)

Compared to the original [ns-3](https://www.nsnam.org/), this repo extends the point-to-point module with a **datacenter / RDMA-oriented** end-to-end model. The main additions live in `simulation/src/point-to-point/model/` and include:

- **QBB/PFC + multi-priority queues**: 8 priority queues, PAUSE/RESUME (PFC-like) handling, and priority-aware scheduling on each port/NIC.
- **ECN + CNP (QCN-style) feedback**: switch-side ECN marking based on queue occupancy and receiver-side ECN accounting; congestion feedback is carried via CNP packets.
- **RDMA host stack (QP-level)**: QP/RxQP modeling, window/on-the-fly control, ACK/NACK handling, and multiple NIC congestion-control (CC) modes (e.g., DCQCN/HPCC/TIMELY/DCTCP/HPCC-PINT).
- **Switch and NVSwitch modeling**: ECMP forwarding, buffer/MMU admission control, PFC trigger/resume logic, and (optional) INT/PINT-style metadata injection for HPCC(-PINT).

## Module map (what each file/class does)

- **`qbb-net-device.{h,cc}` (`QbbNetDevice`, `RdmaEgressQueue`)**
  - **What it does**: A QBB-capable net device on top of `PointToPointNetDevice` with 8 priorities. It intercepts receive to honor PFC, schedules transmissions from either:
    - **host/NIC**: `RdmaEgressQueue` (high-priority ACK/NACK queue + round-robin across QPs), or
    - **switch port**: `BEgressQueue` round-robin across priority queues.
    It also supports an NVSwitch “switch acts as host” send path when NVLS is enabled.
  - **Key attributes**: `QbbEnabled`, `QcnEnabled`, `DynamicThreshold`, `PauseTime`, `NVLS_enable`.
  - **Key integration callbacks**: `m_rdmaReceiveCb` (deliver non-PFC packets to `RdmaHw`), `m_rdmaSentCb` (per-packet send completion), `m_rdmaPktSent` (update QP pacing/next-available), `m_rdmaLinkDownCb`.
  - **Where to extend**:
    - **Scheduling / priority rules**: `DequeueAndTransmit()` and `RdmaEgressQueue::GetNextQindex()`
    - **PFC behavior**: `Receive()` and `SendPfc()`

- **`qbb-channel.{h,cc}` / `qbb-remote-channel.{h,cc}`**
  - **What it does**: point-to-point channel for `QbbNetDevice`; `QbbRemoteChannel` uses MPI (`MpiInterface::SendPacket`) for distributed simulations.
  - **Where to extend**: link behavior/delivery path in `TransmitStart()`.

- **`switch-node.{h,cc}` (`SwitchNode`)**
  - **What it does**: switch pipeline (`nodeType = 1`): ECMP forwarding (5-tuple hash), admission control via MMU, PFC pause/resume generation, optional ECN marking, and INT/PINT injection on dequeue (used by HPCC / HPCC-PINT).
  - **Key attributes**: `EcnEnabled`, `CcMode`, `AckHighPrio`, `MaxRtt`.
  - **Where to extend**:
    - **Forwarding / ECMP**: `GetOutDev()`, `EcmpHash()`, `AddTableEntry()`
    - **ECN / PFC / INT-PINT injection**: `SwitchNotifyDequeue()`

- **`switch-mmu.{h,cc}` (`SwitchMmu`)**
  - **What it does**: switch buffer/MMU model: ingress/egress accounting, shared buffer & headroom, pause/resume decisions, ECN marking probability curve (`kmin/kmax/pmax`), and PFC threshold computation.
  - **Key config APIs**: `ConfigBufferSize()`, `ConfigHdrm()`, `ConfigNPort()`, `ConfigEcn()`.
  - **Where to extend**: implement new buffer management / PFC threshold formula / ECN curve here.

- **`nvswitch-node.{h,cc}` (`NVSwitchNode`)**
  - **What it does**: NVSwitch node model (`nodeType = 2`) used for intra-server GPU communication (paired with the NVLS routing logic in `RdmaHw` / `QbbNetDevice`).
  - **Where to extend**: NVSwitch forwarding/admission/monitoring (similar entry points to `SwitchNode`, but currently without ECN/INT injection).

- **`rdma-hw.{h,cc}` (`RdmaHw`)**
  - **What it does**: host RDMA core: QP create/delete, packet construction (PPP + IPv4 + UDP + SeqTs), ACK/NACK processing, CNP processing, per-QP CC algorithms, and routing to NIC (including NVSwitch routing tables).
  - **Key attributes**: `CcMode`, `Mtu`, `MinRate`, `L2ChunkSize`, `L2AckInterval`, `L2BackToZero`, plus CC-specific knobs for DCQCN/TIMELY/DCTCP/HPCC/PINT (see `GetTypeId()`).
  - **Protocol numbers used (IPv4 Protocol field)**:
    - **UDP data**: `0x11`
    - **CNP**: `0xFF`
    - **PFC**: `0xFE`
    - **ACK**: `0xFC`
    - **NACK**: `0xFD`
  - **Where to extend**:
    - **Add a new CC algorithm**: add `HandleAckX/UpdateRateX` (and optionally CNP hooks) and dispatch by `m_cc_mode` in `ReceiveAck()`/`ReceiveCnp()`.
    - **Add/modify routing (including NVSwitch/NVLS)**: `GetNicIdxOfQp()`, `GetNicIdxOfRxQp()`, `AddTableEntry()`, `RedistributeQp()`.

- **`rdma-driver.{h,cc}` (`RdmaDriver`)**
  - **What it does**: wiring layer between `Node`/NICs and `RdmaHw`: builds NIC/QP groups and exposes QP lifecycle traces (`QpComplete`, `SendComplete`).
  - **Where to extend**: add higher-level observability or app-facing callbacks around QP lifecycle here.

- **`rdma-queue-pair.{h,cc}` (`RdmaQueuePair`, `RdmaRxQueuePair`, `RdmaQueuePairGroup`)**
  - **What it does**: per-QP and per-RxQP state (window, rate, ACKed seq, plus per-CC algorithm state: DCQCN alpha/targetRate, HPCC hop state, TIMELY RTT tracking, DCTCP alpha/ecnCnt, HPCC-PINT state).
  - **Where to extend**: if your new CC needs extra per-QP state, add it here.

- **Headers / utilities**
  - **`qbb-header.{h,cc}`**: ACK/NACK header (PG/seq/CNP-flag + optional INT header).
  - **`cn-header.{h,cc}`**: CNP header (feedback fields: `fid/qIndex/ecnbits/qfb/total`).
  - **`pause-header.{h,cc}`**: PFC pause header (`time/qlen/qindex`).
  - **`pint.{h,cc}`**: PINT encode/decode utilities.
  - **`trace-format.h`**: binary trace record structure `TraceFormat` used by offline analyzers.

## Where to implement new features (quick guide)

- **Add a new host-side congestion control (CC)**
  - **Primary**: `rdma-hw.{h,cc}` (algorithm + dispatch by `CcMode`)
  - **Often needed**: `rdma-queue-pair.h` (new per-QP state)
  - **If switch feedback is required**: `switch-node.cc` (INT/PINT or new markings)

- **Change switch behavior (buffer/ECN/PFC)**
  - **Primary**: `switch-mmu.{h,cc}` (thresholds/curves/formulas)
  - **Where marking/injection happens**: `switch-node.cc::SwitchNotifyDequeue()`
  - **Where admission/priority is applied**: `switch-node.cc::SendToDev()`

- **Introduce a new control packet/header**
  - **Primary**: add a new `*Header` in `model/` (follow `CnHeader` / `PauseHeader`)
  - **Parsing/dispatch**: usually in `QbbNetDevice::Receive()` (device-level) or `RdmaHw::Receive()` (host stack)
  - **Note**: if you need it parsed by `CustomHeader`, you’ll also need to extend the `custom-header` implementation (outside this folder).

# Contact us

Please email Gang Lu (yunding.lg@alibaba-inc.com), Feiyang Xue (xuefeiyang.xfy@alibaba-inc.com) or Qingxu Li (qingxu.lqx@alibaba-inc.com) if you have any questions.

Welcome to join the SimAI community chat groups, with the DingTalk group on the left and the WeChat group on the right.

<div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-left: 20px;">
    <img src="./docs/images/simai_dingtalk.jpg" alt="SimAI DingTalk" style="width: 300px; height: auto;">
    <img src="./docs/images/simai_wechat.jpg" alt="SimAI WeChat" style="width: 300px; height: auto;">
</div>

<br/>