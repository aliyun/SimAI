# Extending the NS-3 Network Backend

This guide covers how to extend `ns-3-alibabacloud` with new congestion control algorithms, switch behaviors, control packets, and NVSwitch features.

> **Source reference**: See `astra-sim-alibabacloud/extern/network_backend/ns3-interface/README.md` for the detailed module map.

---

## Module Overview

All key source files are located in `ns-3-alibabacloud/simulation/src/point-to-point/model/`:

| File | Class | Purpose |
|------|-------|---------|
| `qbb-net-device.{h,cc}` | `QbbNetDevice`, `RdmaEgressQueue` | QBB-capable NIC with 8 priority queues, PFC handling, NVSwitch send path |
| `rdma-hw.{h,cc}` | `RdmaHw` | Host RDMA core: QP management, packet construction, ACK/NACK, CC algorithms |
| `rdma-queue-pair.{h,cc}` | `RdmaQueuePair`, `RdmaRxQueuePair` | Per-QP state (window, rate, CC-specific state) |
| `switch-node.{h,cc}` | `SwitchNode` | Switch pipeline: ECMP forwarding, ECN marking, PFC, INT/PINT injection |
| `switch-mmu.{h,cc}` | `SwitchMmu` | Switch buffer/MMU: ingress/egress accounting, PFC thresholds, ECN curves |
| `nvswitch-node.{h,cc}` | `NVSwitchNode` | NVSwitch model for intra-server GPU communication |
| `rdma-driver.{h,cc}` | `RdmaDriver` | Wiring layer between Node/NICs and RdmaHw |
| `qbb-header.{h,cc}` | — | ACK/NACK header (PG/seq/CNP-flag + INT header) |
| `cn-header.{h,cc}` | — | CNP header (feedback fields) |
| `pause-header.{h,cc}` | — | PFC pause header |
| `pint.{h,cc}` | — | PINT encode/decode utilities |
| `trace-format.h` | `TraceFormat` | Binary trace record structure for offline analysis |

---

## Adding a New Congestion Control Algorithm

The NS-3 backend supports 5 built-in CC algorithms: **DCQCN**, **HPCC**, **TIMELY**, **DCTCP**, and **HPCC-PINT**. To add a new CC algorithm:

### Step 1: Define CC Mode

Add a new `CcMode` value in `rdma-hw.h`:

```cpp
// Existing modes: 1=DCQCN, 3=HPCC, 7=TIMELY, 8=DCTCP, 10=HPCC-PINT
static const uint32_t CC_MODE_YOUR_ALG = 11;
```

### Step 2: Add Per-QP State (if needed)

In `rdma-queue-pair.h`, add new state variables to `RdmaQueuePair`:

```cpp
// Your CC algorithm state
double m_your_alg_rate;
double m_your_alg_alpha;
// ...
```

### Step 3: Implement Algorithm Logic

In `rdma-hw.cc`, add two key functions:

```cpp
void RdmaHw::HandleAckYourAlg(Ptr<RdmaQueuePair> qp, ...) {
    // Process ACK and update rate/window
}

void RdmaHw::UpdateRateYourAlg(Ptr<RdmaQueuePair> qp, ...) {
    // Rate update logic
}
```

### Step 4: Register Dispatch

Add dispatch cases in `ReceiveAck()` and/or `ReceiveCnp()` in `rdma-hw.cc`:

```cpp
switch (m_cc_mode) {
    // ... existing cases ...
    case CC_MODE_YOUR_ALG:
        HandleAckYourAlg(qp, ...);
        break;
}
```

### Step 5: Add Switch Feedback (if needed)

If your CC algorithm requires switch-side information (like INT/PINT metadata):

- Modify `switch-node.cc::SwitchNotifyDequeue()` to inject your metadata
- Add header parsing in `RdmaHw::Receive()` or `QbbNetDevice::Receive()`

---

## Modifying Switch Behavior

### Buffer Management / PFC Thresholds

**Primary file**: `switch-mmu.{h,cc}`

Key methods to modify:

| Method | Purpose |
|--------|---------|
| `ConfigBufferSize()` | Total buffer pool size |
| `ConfigHdrm()` | Headroom allocation |
| `ConfigEcn()` | ECN marking thresholds (`kmin`, `kmax`, `pmax`) |
| `CheckIngressAdmission()` | Ingress admission control |
| `CheckEgressAdmission()` | Egress admission control |
| `GetPfcThreshold()` | PFC trigger threshold formula |

### ECN Marking / INT Injection

**File**: `switch-node.cc`

Modify `SwitchNotifyDequeue()` for:
- ECN marking based on custom queue occupancy formulas
- INT/PINT metadata injection for advanced CC algorithms
- Custom packet tagging

### Forwarding / ECMP

**File**: `switch-node.cc`

Modify for routing changes:
- `GetOutDev()` — Output port selection
- `EcmpHash()` — ECMP hash function (currently 5-tuple)
- `AddTableEntry()` — Routing table management

---

## Introducing New Control Packets

### Step 1: Create Header

Create new header files in `model/` following the pattern of `CnHeader` or `PauseHeader`:

```cpp
// your-header.h
class YourHeader : public Header {
public:
    static TypeId GetTypeId();
    // Serialize/Deserialize methods
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    
    // Your header fields
    uint32_t m_your_field;
};
```

### Step 2: Define Protocol Number

Add a new protocol number (following existing conventions):

```cpp
// Existing protocol numbers (IPv4 Protocol field):
// UDP data:  0x11
// CNP:       0xFF
// PFC:       0xFE
// ACK:       0xFC
// NACK:      0xFD
// Your new:  0xFB (example)
```

### Step 3: Add Parsing/Dispatch

Add packet handling in:
- `QbbNetDevice::Receive()` — Device-level parsing
- `RdmaHw::Receive()` — Host stack processing

---

## NVSwitch / NVLS Extensions

**Files**: `nvswitch-node.{h,cc}`, `qbb-net-device.{h,cc}` (NVLS send path), `rdma-hw.{h,cc}` (NVLS routing)

The `NVSwitchNode` models intra-server GPU communication via NVSwitch. To extend:

- **Forwarding**: Similar to `SwitchNode` but without ECN/INT injection
- **NVLS routing**: Modify `RdmaHw::GetNicIdxOfQp()` and `GetNicIdxOfRxQp()` for NVSwitch routing tables
- **QP redistribution**: `RdmaHw::RedistributeQp()` for load balancing across NVSwitch links

---

## Analysis Tools

The `ns-3-alibabacloud/analysis/` directory contains trace analysis tools:

| Tool | Purpose |
|------|---------|
| FCT Analysis | Flow Completion Time analysis from simulation traces |
| Trace Reader | Parse binary `TraceFormat` records |
| Bandwidth Analysis | Per-link bandwidth utilization over time |
| Queue Analysis | Queue occupancy and PFC event analysis |
| QP Analysis | Per-QP performance metrics |

### Trace Format

The binary trace record structure (`trace-format.h`) captures per-packet events. Use the offline analysis tools to:

1. Parse trace files from simulation output
2. Compute FCT, throughput, queue depth statistics
3. Identify congestion hotspots and PFC events

---

## dev/qp Branch Enhancements

The [dev/qp](https://github.com/aliyun/ns-3-alibabacloud/tree/dev/qp) branch includes:

1. **QP Logic Support** — QP creation/destruction based on actual RDMA logic
2. **NIC CC Configuration** — PerIP or perQP CC settings
3. **Optimized Scheduling** — Max-Min principle for fair resource allocation
4. **Decoupled CC Module** — Improved modularity

---

## Related Documentation

- [NS-3 Component](../components/ns3.md) — Full NS-3 backend documentation
- [SimAI-Simulation User Guide](../user_guide/simai_simulation.md) — Using NS-3 simulation mode
- [Configuration Reference](../technical_reference/configuration.md) — Topology and configuration files
