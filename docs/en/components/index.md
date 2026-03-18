# Components Overview

SimAI is a modular project composed of 5 core components. Each component can be used independently or combined to achieve different simulation scenarios.

---

## Architecture

```
        |--- AICB                        (Workload generation & compute profiling)
SimAI --|--- SimCCL                      (Collective communication algorithm analysis)
        |--- astra-sim-alibabacloud      (Simulation engine: Analytical / Simulation / Physical)
        |--- ns-3-alibabacloud           (NS-3 network backend)
        |--- vidur-alibabacloud          (Multi-request inference scheduling & memory management)
```

---

## Component Summary

| Component | Language | Repository | Description |
|-----------|----------|------------|-------------|
| [AICB](aicb.md) | Python | [aliyun/aicb](https://github.com/aliyun/aicb) | AI Communication Benchmark — workload generation for training and inference |
| [SimCCL](simccl.md) | Python | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | Collective communication to point-to-point transformation |
| [astra-sim-alibabacloud](astra_sim.md) | C++ | In-tree | Core simulation engine supporting 3 modes (Analytical/Simulation/Physical) |
| [ns-3-alibabacloud](ns3.md) | C++ | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | NS-3 network simulator backend with RDMA/datacenter extensions |
| [vidur-alibabacloud](vidur.md) | Python | In-tree | LLM inference simulation with PD disaggregation and request scheduling |

---

## Component Combinations by Scenario

| Scenario | AICB | SimCCL | astra-sim | ns-3 | vidur |
|----------|------|--------|-----------|------|-------|
| AICB Test Suite (physical GPU) | Required | - | - | - | - |
| Workload Generation | Required | - | - | - | - |
| Collective Comm Analysis | - | Required | - | - | - |
| SimAI-Analytical | Required | - | Required (analytical) | - | - |
| SimAI-Simulation | Required | Required | Required (simulation) | Required | - |
| SimAI-Physical | Required | Required | Required (physical) | - | - |
| Inference Simulation | Required | Required | Required (analytical/simulation) | Optional | Required |

---

## Data Flow

```
AICB (Workload Generation)
    |
    |-- Training workload (.txt) --> astra-sim-alibabacloud
    |-- Inference workload -------> vidur-alibabacloud
    |
SimCCL (Collective → P2P)
    |
    |--> astra-sim-alibabacloud (Simulation/Physical mode)
    |
astra-sim-alibabacloud (Simulation Engine)
    |
    |-- Analytical mode: busbw-based estimation
    |-- Simulation mode: NS-3 backend
    |-- Physical mode: RDMA traffic injection
    |
ns-3-alibabacloud (Network Backend)
    |
    |--> Fine-grained network simulation results
    |
vidur-alibabacloud (Inference Scheduling)
    |
    |--> request_metrics.csv, chrome_trace.json, plots/
```

---

## Detailed Component Documentation

- **[AICB](aicb.md)** — Workload generation, benchmark suite, AIOB compute profiling
- **[SimCCL](simccl.md)** — Collective communication decomposition
- **[astra-sim-alibabacloud](astra_sim.md)** — Core simulation engine, configuration, topology generation
- **[ns-3-alibabacloud](ns3.md)** — RDMA network simulation, CC algorithms, analysis tools
- **[vidur-alibabacloud](vidur.md)** — Inference simulation, PD disaggregation, GPU memory management
