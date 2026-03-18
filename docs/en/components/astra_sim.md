# astra-sim-alibabacloud — Simulation Engine

**Location**: In-tree (`astra-sim-alibabacloud/`) | **Language**: C++

The core simulation engine of SimAI, extended from [astra-sim 1.0](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0). It supports three operation modes and integrates NCCL algorithms with custom enhancements.

---

## Overview

astra-sim-alibabacloud serves as the central orchestrator for SimAI simulations. It:

- Receives workloads from AICB
- Uses SimCCL to decompose collective operations into P2P transfers
- Drives network simulation via NS-3 (simulation mode) or direct RDMA (physical mode)
- Computes timing using busbw parameters (analytical mode)

---

## Three Operation Modes

### SimAI-Analytical

Fast analytical simulation using bus bandwidth (busbw) to estimate collective communication times.

**Build**: `./scripts/build.sh -c analytical`
**Binary**: `bin/SimAI_analytical`

### SimAI-Simulation

Full-stack simulation with NS-3 network backend for fine-grained network modeling.

**Build**: `./scripts/build.sh -c ns3`
**Binary**: `bin/SimAI_simulator`

### SimAI-Physical

Physical traffic generation using RDMA on real hardware.

**Build**: `./scripts/build.sh -c phy`
**Binary**: `bin/SimAI_phynet`

---

## Core Components

| Component | Description |
|-----------|-------------|
| **AstraComputeAPI** | Manages computation timing and scheduling |
| **MemoryAPI** | Handles memory allocation and tracking |
| **NetworkAPI** | Interface to network backends (NS-3, physical) |
| **MockNcclGroup** | Simulates NCCL communication groups |
| **MockNcclChannel** | Manages individual communication channels |
| **SimAiFlowModelRdma** | RDMA flow model for traffic simulation |

---

## Configuration

### SimAI.conf

The main configuration file is located at `astra-sim-alibabacloud/inputs/config/SimAI.conf`. It controls simulation parameters including:

- Communication algorithms
- Buffer sizes
- Timing parameters
- Network backend settings

### Environment Variables (Simulation Mode)

| Variable | Description | Default |
|----------|-------------|---------|
| `AS_LOG_LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR | `INFO` |
| `AS_PXN_ENABLE` | Enable PXN (Proxied NVLINK) | `0` (false) |
| `AS_NVLS_ENABLE` | Enable NVLS (NVLink Sharp) | `0` (false) |
| `AS_SEND_LAT` | Packet sending latency (us) | `6` |
| `AS_NVLSTREE_ENABLE` | Enable NVLS Tree algorithm | `false` |

### Simulation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-t` / `--thread` | Number of threads for acceleration | `1` (recommended: 8-16) |
| `-w` / `--workload` | Path to workload file | Required |
| `-n` / `--network-topo` | Network topology file path | Required (simulation mode) |
| `-c` / `--config` | SimAI configuration file | Required |

---

## Topology Generation

astra-sim provides 5 topology templates via `gen_Topo_Template.py`:

### Available Templates

| Template | Architecture | Description |
|----------|-------------|-------------|
| `Spectrum-X` | NVIDIA Spectrum-X | Rail-optimized, single ToR, single plane |
| `AlibabaHPN` (single plane) | Alibaba HPN 7.0 | Dual ToR, rail-optimized, single plane |
| `AlibabaHPN` (dual plane) | Alibaba HPN 7.0 | Dual ToR, rail-optimized, dual plane |
| `DCN+` (single ToR) | DCN+ | Single ToR, non rail-optimized |
| `DCN+` (dual ToR) | DCN+ | Dual ToR, non rail-optimized |

### Topology Parameters

| Level | Parameter | Description |
|-------|-----------|-------------|
| **Global** | `-topo` | Template name |
| | `-g` | Number of GPUs |
| | `--dp` | Enable dual plane |
| | `--ro` | Enable rail-optimized |
| | `--dt` | Enable dual ToR |
| **Intra-Host** | `-gps` | GPUs per server |
| | `-gt` | GPU type (A100/H100) |
| | `-nvbw` | NVLink bandwidth |
| | `-nl` | NVLink latency |
| **Intra-Segment** | `-bw` | NIC to ASW bandwidth |
| | `-asw` | ASW switch count |
| | `-nps` | NICs per switch |
| **Intra-Pod** | `-psn` | PSW switch count |
| | `-apbw` | ASW to PSW bandwidth |

### Examples

```bash
# Spectrum-X with 128 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# Dual-Plane AlibabaHPN with 64 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo AlibabaHPN --dp -g 64 -asn 16 -psn 16

# Dual-ToR DCN+ with 128 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo DCN+ --dt -g 128 -asn 2 -psn 8
```

---

## See Also

- [SimAI-Analytical Guide](../user_guide/simai_analytical.md) — Analytical mode usage
- [SimAI-Simulation Guide](../user_guide/simai_simulation.md) — NS-3 simulation usage
- [SimAI-Physical Guide](../user_guide/simai_physical.md) — Physical mode usage
- [NS-3 Component](ns3.md) — Network backend details
- [SimCCL Component](simccl.md) — Collective communication decomposition
