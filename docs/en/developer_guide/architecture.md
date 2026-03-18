# System Architecture

This document describes SimAI's modular architecture, component interactions, and data flow for both training and inference simulation.

---

## Project Structure

```
SimAI/
├── aicb/                        # AI Computation Benchmark — workload generation (Python)
│   ├── workload_generator/      #   Generates training/inference workloads
│   └── aicb.py                  #   Main entry point
├── astra-sim-alibabacloud/      # Simulation engine — core simulator (C++)
│   ├── astra-sim/               #   Extended from astra-sim 1.0
│   └── build.sh                 #   Build script
├── ns-3-alibabacloud/           # NS-3 network simulator backend (C++)
├── vidur-alibabacloud/          # LLM inference simulation (Python)
│   ├── vidur/                   #   Core simulation framework
│   └── setup.py                 #   Python package config
├── SimCCL/                      # Collective communication transformation
├── docs/                        # Documentation and tutorials
├── example/                     # Example workloads and configurations
├── scripts/                     # Build and utility scripts
├── results/                     # Simulation output directory
├── bin/                         # Compiled binary output
└── Dockerfile                   # Docker container definition
```

---

## Component Architecture

```
        |--- AICB                        (Workload generation & compute profiling)
SimAI --|--- SimCCL                      (Collective communication algorithm analysis)
        |--- astra-sim-alibabacloud      (Simulation engine: Analytical / Simulation / Physical)
        |--- ns-3-alibabacloud           (NS-3 network backend)
        |--- vidur-alibabacloud          (Multi-request inference scheduling & memory management)
```

![SimAI Architecture](../../images/SimAI_Arc.png)

### Component Responsibilities

| Component | Role | Language |
|-----------|------|----------|
| **AICB** | Generates training/inference workloads, profiles compute kernels, runs physical benchmarks | Python |
| **SimCCL** | Transforms collective communication operations (AllReduce, AllGather, etc.) into point-to-point communication sets | Python |
| **astra-sim-alibabacloud** | Core simulation engine supporting 3 modes; manages compute/memory/network APIs | C++ |
| **ns-3-alibabacloud** | Packet-level network simulation with RDMA, datacenter topology, and CC algorithms | C++ |
| **vidur-alibabacloud** | Multi-request inference scheduling with PD disaggregation and GPU memory management | Python |

---

## Three Operation Modes

### SimAI-Analytical

```
AICB (workload.txt) → astra-sim (analytical) → busbw estimation → CSV results
```

- **Use case**: Fast performance analysis, parallel parameter sweeps
- **Components**: AICB + astra-sim-alibabacloud (analytical mode)
- **Network model**: Bus bandwidth (busbw) abstraction

### SimAI-Simulation

```
AICB (workload.txt) → SimCCL (collective→P2P) → astra-sim (simulation) → NS-3 → detailed traces
```

- **Use case**: Full-stack network research, CC algorithm evaluation
- **Components**: AICB + SimCCL + astra-sim-alibabacloud (simulation) + ns-3-alibabacloud
- **Network model**: Packet-level NS-3 simulation

### SimAI-Physical

```
AICB (workload.txt) → SimCCL (collective→P2P) → astra-sim (physical) → RDMA traffic on real NICs
```

- **Use case**: NIC behavior study, physical traffic analysis
- **Components**: AICB + SimCCL + astra-sim-alibabacloud (physical)
- **Network model**: Real RDMA traffic via MPI

---

## Inference Simulation Data Flow

```
Request Generator
    |  Generate synthetic / real-trace requests
    v
Global Scheduler
    |  Dispatch requests to Prefill / Decode replicas
    v
Replica Scheduler
    |  Batch assembly and scheduling
    v
Memory Management (MemoryPlanner + Replica)
    |  KV cache allocation and capacity checking
    v
Execution Time Predictor
    |  AICB / SimAI Simulation / SimAI Analytical / Vidur
    v
Metrics Store
    |  TTFT, TBT, E2E, communication / compute cost
    v
Output (request_metrics.csv, chrome_trace.json, plots/)
```

### Key Inference Components

| Component | File | Description |
|-----------|------|-------------|
| Request Generator | `vidur/request_generator/` | Generates synthetic or trace-based requests |
| Global Scheduler | `vidur/scheduler/global_scheduler/` | Dispatches requests across replicas (`lor`, `round_robin`, `split_wise`) |
| Replica Scheduler | `vidur/scheduler/replica_scheduler/` | Per-replica batch scheduling (`sarathi`, `split_wise`) |
| MemoryPlanner | `vidur/scheduler/utils/memory_planner.py` | GPU memory budget computation |
| ParamCounter | `vidur/utils/param_counter.py` | Model parameter counting (MLA/MHA/GQA/linear/MoE) |
| Execution Predictor | `vidur/execution_time_predictor/` | Execution time estimation via multiple backends |
| Metrics Store | `vidur/metrics/` | Collects and exports 23 simulation metrics |

---

## Submodule Structure

SimAI uses Git submodules for its core components:

| Submodule | Repository | Branch |
|-----------|------------|--------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | master |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | master |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | master / dev/qp |
| `astra-sim-alibabacloud` | In-tree | — |
| `vidur-alibabacloud` | In-tree | — |

**Key rules:**
1. Submodules have independent Git histories
2. The parent repo only tracks the commit hash of each submodule
3. Always initialize after cloning: `git submodule update --init --recursive`

---

## Build System

### Build Scripts

```bash
# Analytical mode (fast, busbw-based)
bash scripts/build.sh -c analytical

# NS-3 simulation mode (full-stack)
bash scripts/build.sh -c ns3

# Physical mode (beta, RDMA)
bash scripts/build.sh -c phy
```

### Build Outputs

| Mode | Binary | Location |
|------|--------|----------|
| Analytical | `SimAI_analytical` | `bin/` |
| Simulation | `SimAI_simulator` | `bin/` |
| Physical | `SimAI_physical` | `bin/` |

---

## Related Documentation

- [Components Overview](../components/index.md) — Detailed documentation for each component
- [Contributing Guide](contributing.md) — How to contribute code
- [Configuration Reference](../technical_reference/configuration.md) — Configuration files and parameters
