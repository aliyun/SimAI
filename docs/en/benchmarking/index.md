# Benchmarking

This section covers benchmarking and validation approaches for SimAI.

---

## Contents

| Document | Description |
|----------|-------------|
| [4-Scenario End-to-End Test Suite](test_suite.md) | Pre-configured test scenarios covering different models, parallelism strategies, and PD configurations |

---

## Benchmarking Approaches

SimAI supports several benchmarking methodologies:

### Architecture Comparison

Compare different network architectures (e.g., Spectrum-X vs DCN+) under identical workloads to evaluate their performance characteristics.

### Algorithm Comparison

Compare different collective communication algorithms (e.g., RING vs NVLS) to understand their performance trade-offs at various message sizes.

### Parameter Optimization

Use SimAI-Analytical for rapid exploration of parallel parameter combinations (TP, PP, EP, DP) to find optimal configurations.

### Validation Against Real Hardware

Use AICB physical execution results as ground truth to validate simulation accuracy.
