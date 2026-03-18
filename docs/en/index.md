# Welcome to SimAI Documentation

<p align="left">
    <a href="../zh/index.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)
[![NSDI'25](https://img.shields.io/badge/NSDI'25-SimAI-blue.svg)](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)

**SimAI** is the industry's first full-stack, high-precision **Sim**ulator for **AI** large-scale **inference** and **training**, open-sourced by Alibaba Cloud. It provides detailed modeling and simulation of the entire LLM training and inference process, encompassing the framework layer, collective communication layer, and network transport layer, delivering end-to-end performance data.

SimAI enables researchers to:

- Analyze inference/training process details
- Evaluate the time consumption of AI tasks under specific conditions
- Evaluate E2E performance gains from various algorithmic optimizations (framework parameters, collective communication algorithms, network protocols, congestion control, routing, topology, etc.)

---

## Documentation Overview

| Section | Description |
|---------|-------------|
| [Getting Started](getting_started/index.md) | Installation, environment setup, and quickstart guide |
| [User Guide](user_guide/index.md) | Detailed usage for SimAI-Analytical, SimAI-Simulation, SimAI-Physical, and Inference Simulation |
| [Components](components/index.md) | In-depth documentation for each submodule: AICB, SimCCL, astra-sim, ns-3, vidur |
| [Technical Reference](technical_reference/index.md) | GPU memory module, CLI parameters, and configuration reference |
| [Benchmarking](benchmarking/index.md) | 4-scenario end-to-end test suite and benchmark results |
| [Developer Guide](developer_guide/index.md) | Architecture, contributing guide, adding models, and extending NS-3 |
| [Community](community/index.md) | Events, contact information, and citation |

---

## Architecture

```
        |--- AICB                        (Workload generation & compute profiling)
SimAI --|--- SimCCL                      (Collective communication algorithm analysis)
        |--- astra-sim-alibabacloud      (Simulation engine: Analytical / Simulation / Physical)
        |--- ns-3-alibabacloud           (NS-3 network backend)
        |--- vidur-alibabacloud          (Multi-request inference scheduling & memory management)
```

![SimAI Architecture](../images/SimAI_Arc.png)

---

## Three Operation Modes

| Mode | Description | Use Cases |
|------|-------------|-----------|
| **SimAI-Analytical** | Fast simulation using bus bandwidth (busbw) to estimate collective communication time | Performance analysis, parallel parameter optimization, scale-up exploration |
| **SimAI-Simulation** | Full-stack simulation with NS-3 network backend for fine-grained network modeling | CC algorithm research, network protocol evaluation, novel architecture design |
| **SimAI-Physical** *(Beta)* | Physical traffic generation on CPU RDMA clusters | NIC behavior study during LLM training |

---

## Supported Models

- **DeepSeek-V3-671B** — MLA attention, 256 routed experts
- **Qwen3-MoE-235B** — MHA/GQA, 128 routed experts
- **Qwen3-Next-80B** — Hybrid full + linear attention, 512 routed experts
- **Meta-Llama-3-8B / 70B**, **Llama-2-7b / 70b**, **CodeLlama-34b**, **InternLM-20B**, **Qwen-72B**

---

## Quick Links

- [GitHub Repository](https://github.com/aliyun/SimAI)
- [NSDI'25 Paper (PDF)](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)
- [Slides](../../docs/SimAI_Intro_Online.pdf)
- [Technical Report (1.6)](../SimAI_1.6_Tech_Report.md)
- [Contributing Guide](../../CONTRIBUTING.md)
