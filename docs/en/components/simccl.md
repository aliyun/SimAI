# SimCCL — Collective Communication Library

**Repository**: [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | **Language**: Python/C++

SimCCL enables the transformation of collective communication operations into point-to-point communications, serving as a critical bridge between the workload layer and the simulation engine.

---

## Overview

In distributed LLM training, collective communication operations (AllReduce, AllGather, ReduceScatter, AlltoAll, etc.) are fundamental building blocks. SimCCL breaks down these high-level collective operations into sequences of point-to-point communications that can be precisely simulated by the network backend.

---

## Role in SimAI

SimCCL sits between AICB (workload generation) and astra-sim-alibabacloud (simulation engine):

```
AICB generates workload with collective ops
    |
    v
SimCCL decomposes collective → point-to-point
    |
    v
astra-sim sends P2P traffic to NS-3 or physical network
```

SimCCL is required for:
- **SimAI-Simulation** — Full-stack NS-3 simulation
- **SimAI-Physical** — Physical RDMA traffic generation
- **Inference Simulation** — When using SimAI Simulation backend

SimCCL is NOT required for:
- **SimAI-Analytical** — Uses busbw-based estimation directly

---

## Versions

### Basic Version (mocknccl)

A basic implementation is currently available in the [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud) repository. Files are prefixed with `mocknccl` and provide fundamental collective-to-P2P conversion.

### Complete Version

The full SimCCL library with advanced collective communication algorithms is available in the [SimCCL repository](https://github.com/aliyun/SimCCL).

---

## Supported Collective Operations

| Operation | Description |
|-----------|-------------|
| AllReduce | Reduce data across all ranks, result available on all ranks |
| AllGather | Gather data from all ranks, result available on all ranks |
| ReduceScatter | Reduce and scatter data across all ranks |
| AlltoAll | All-to-all personalized communication |
| Broadcast | Broadcast from one rank to all others |

---

## Integration with astra-sim

SimCCL integrates with astra-sim-alibabacloud through the `MockNcclGroup` and `MockNcclChannel` interfaces:

- **MockNcclGroup**: Manages a group of ranks participating in a collective operation
- **MockNcclChannel**: Handles the actual point-to-point data transfer for a specific channel within a collective operation

The decomposition considers:
- Network topology (ring, tree, etc.)
- Number of participating ranks
- Message size
- Available communication channels

---

## See Also

- [Components Overview](index.md) — SimAI component architecture
- [astra-sim Component](astra_sim.md) — Simulation engine that consumes SimCCL output
- [NS-3 Component](ns3.md) — Network backend for P2P simulation
