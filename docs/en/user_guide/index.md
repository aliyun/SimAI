# User Guide

This section provides detailed usage instructions for all SimAI operation modes.

## Contents

| Page | Description |
|------|-------------|
| [SimAI-Analytical](simai_analytical.md) | Fast analytical simulation using bus bandwidth |
| [SimAI-Simulation](simai_simulation.md) | Full-stack NS-3 network simulation with topology configuration |
| [SimAI-Physical](simai_physical.md) | Physical RDMA traffic generation on real clusters |
| [Inference Simulation](inference_simulation.md) | Multi-request LLM inference simulation with PD disaggregation |
| [Workload Generation](workload_generation.md) | Generate training and inference workloads using AICB |
| [Supported Models](supported_models.md) | Complete list of supported models and configurations |
| [Result Analysis](result_analysis.md) | Analyze and visualize simulation results |

## Workflow Overview

A typical SimAI workflow involves three steps:

1. **Generate workloads** using [AICB](workload_generation.md) — defines the computation and communication patterns
2. **Run simulation** using one of the three modes (Analytical, Simulation, or Physical)
3. **Analyze results** using built-in tools or custom scripts

For inference simulation, the workflow uses Vidur for request scheduling and memory management, with AICB or SimAI as the execution-time prediction backend.
