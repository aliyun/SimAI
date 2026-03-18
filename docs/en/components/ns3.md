# ns-3-alibabacloud — Network Simulation Backend

**Repository**: [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | **Language**: C++

An NS-3-based network simulator acting as the network backend for SimAI, extended with datacenter/RDMA-oriented end-to-end modeling.

---

## Overview

Compared to upstream [NS-3](https://www.nsnam.org/), ns-3-alibabacloud extends the point-to-point module with comprehensive datacenter networking features:

- **QBB/PFC + multi-priority queues** — 8 priority queues with PAUSE/RESUME handling
- **ECN + CNP feedback** — Switch-side ECN marking and receiver-side congestion notification
- **RDMA host stack (QP-level)** — Full QP modeling with 5 congestion control algorithms
- **Switch and NVSwitch modeling** — ECMP forwarding, buffer management, PFC logic

### dev/qp Branch

The [dev/qp](https://github.com/aliyun/ns-3-alibabacloud/tree/dev/qp) branch includes additional enhancements:

1. QP logic support with creation/destruction based on actual RDMA logic
2. Per-IP or per-QP NIC CC configuration
3. Optimized Max-Min scheduling logic
4. Decoupled CC module for modularity

---

## Core Modules

### QBB Net Device (`qbb-net-device`)

A QBB-capable net device with 8 priorities built on top of `PointToPointNetDevice`. Features:

- PFC PAUSE/RESUME handling
- `RdmaEgressQueue` with high-priority ACK/NACK queue + round-robin across QPs
- `BEgressQueue` for switch port round-robin
- NVSwitch send path support (NVLS mode)

**Key attributes**: `QbbEnabled`, `QcnEnabled`, `DynamicThreshold`, `PauseTime`, `NVLS_enable`

### RDMA Host Stack (`rdma-hw`)

Host RDMA core implementing:

- QP create/delete lifecycle
- Packet construction (PPP + IPv4 + UDP + SeqTs headers)
- ACK/NACK/CNP processing
- Per-QP congestion control algorithms
- NVSwitch routing tables

**Congestion Control Algorithms**:

| Algorithm | Description |
|-----------|-------------|
| **DCQCN** | Data Center Quantized Congestion Notification |
| **HPCC** | High Precision Congestion Control |
| **TIMELY** | RTT-based congestion control |
| **DCTCP** | Data Center TCP |
| **HPCC-PINT** | HPCC with Probabilistic INT |

**Protocol Numbers (IPv4 Protocol field)**:

| Protocol | Number | Description |
|----------|--------|-------------|
| UDP Data | `0x11` | Normal data packets |
| CNP | `0xFF` | Congestion Notification Packet |
| PFC | `0xFE` | Priority Flow Control |
| ACK | `0xFC` | Acknowledgment |
| NACK | `0xFD` | Negative Acknowledgment |

### Switch Node (`switch-node`)

Switch pipeline implementing:
- ECMP forwarding (5-tuple hash)
- Admission control via MMU
- PFC pause/resume generation
- ECN marking
- INT/PINT injection for HPCC/HPCC-PINT

### Switch MMU (`switch-mmu`)

Switch buffer/MMU model:
- Ingress/egress accounting
- Shared buffer and headroom management
- PFC trigger/resume logic
- ECN marking probability curve (`kmin/kmax/pmax`)

### NVSwitch Node (`nvswitch-node`)

NVSwitch model for intra-server GPU communication, paired with NVLS routing logic in `RdmaHw`/`QbbNetDevice`.

### QP State (`rdma-queue-pair`)

Per-QP and per-RxQP state management including:
- Window and rate control
- ACKed sequence tracking
- Per-CC algorithm state (DCQCN alpha/targetRate, HPCC hop state, TIMELY RTT, DCTCP alpha/ecnCnt, PINT state)

---

## Analysis Tools

Located in `ns-3-alibabacloud/analysis/`:

### FCT Analysis

```bash
python fct_analysis.py -h  # See help for usage
```

Reads FCT output files and produces statistics for flow completion time analysis.

### Trace Reader

```bash
# Build
make trace_reader

# Usage
./trace_reader <.tr file> [filter_expr]

# Filter examples
./trace_reader output.tr "time > 2000010000"
./trace_reader output.tr "sip=0x0b000101&dip=0x0b000201"
```

### Trace Output Format

```
2000055540 n:338 4:3 100608 Enqu ecn:0 0b00d101 0b012301 10000 100 U 161000 0 3 1048(1000)
```

Fields: timestamp, node, port:queue, queue_length, event, ecn, src_ip, dst_ip, src_port, dst_port, pkt_type, seq, tx_time, priority, size(payload)

---

## Headers and Utilities

| File | Description |
|------|-------------|
| `qbb-header` | ACK/NACK header with optional INT header |
| `cn-header` | CNP header (feedback fields) |
| `pause-header` | PFC pause header |
| `pint` | PINT encode/decode utilities |
| `trace-format.h` | Binary trace record structure for offline analysis |

---

## Extension Guide

### Adding a New CC Algorithm

1. **Primary**: `rdma-hw.{h,cc}` — Add `HandleAckX`/`UpdateRateX` methods, dispatch by `m_cc_mode`
2. **Often needed**: `rdma-queue-pair.h` — Add new per-QP state variables
3. **If switch feedback required**: `switch-node.cc` — Add INT/PINT or new markings

### Changing Switch Behavior

1. **Primary**: `switch-mmu.{h,cc}` — Modify thresholds, curves, formulas
2. **Marking/injection**: `switch-node.cc::SwitchNotifyDequeue()`
3. **Admission/priority**: `switch-node.cc::SendToDev()`

### Adding New Control Packets

1. Create new `*Header` in `model/` (follow `CnHeader`/`PauseHeader` pattern)
2. Add parsing in `QbbNetDevice::Receive()` or `RdmaHw::Receive()`

---

## See Also

- [SimAI-Simulation Guide](../user_guide/simai_simulation.md) — Full-stack simulation usage
- [astra-sim Component](astra_sim.md) — Simulation engine
- [Extending NS-3 Guide](../developer_guide/extending_ns3.md) — Detailed extension guide
