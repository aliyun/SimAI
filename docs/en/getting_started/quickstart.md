# Quickstart

This guide walks you through running your first simulation with SimAI.

## 1. SimAI-Analytical

The fastest way to get started. Abstracts network details using bus bandwidth (busbw).

```bash
# Run analytical simulation
./bin/SimAI_analytical \
    -w example/workload_analytical.txt \
    -g 9216 \
    -g_p_s 8 \
    -r test- \
    -busbw example/busbw.yaml
```

For automatic bus bandwidth calculation:

```bash
./bin/SimAI_analytical \
    -w ./example/workload_analytical.txt \
    -g 9216 -nv 360 -nic 48.5 \
    -n_p_s 8 -g_p_s 8 -r example-
```

For detailed parameter descriptions, see [SimAI-Analytical User Guide](../user_guide/simai_analytical.md).

## 2. SimAI-Simulation

Full-stack simulation with NS-3 network backend.

```bash
# Step 1: Create network topology
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# Step 2: Run simulation
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
    -t 16 \
    -w ./example/microAllReduce.txt \
    -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
    -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

For detailed parameter descriptions, see [SimAI-Simulation User Guide](../user_guide/simai_simulation.md).

## 3. Multi-Request Inference Simulation

End-to-end inference simulation using the Vidur framework.

### Prerequisites

```bash
# Activate the vidur conda environment
conda activate vidur
```

### Run the 4-Scenario Test Suite

```bash
# Run all 4 scenarios
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# Or run a single scenario
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

### Scenarios Overview

| Scenario | Model | PD Separation | World Size | TP | EP | Scheduler |
|----------|-------|---------------|-----------|----|----|-----------|
| 1 | Qwen3-Next-80B | No | 32 | 1 | 1 | lor |
| 2 | Qwen3-Next-80B | Yes (P=2, D=6) | 8 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B | Yes (P=2, D=6) | 8 | 8 | 8 | split_wise |
| 4 | Qwen3-MoE-235B | Yes (P=2, D=6) | 8 | 4 | 4 | split_wise |

For detailed information, see [Inference Simulation User Guide](../user_guide/inference_simulation.md).

## What's Next

- [User Guide](../user_guide/index.md) — Deep dive into each simulation mode
- [Components](../components/index.md) — Learn about each submodule
- [Benchmarking](../benchmarking/index.md) — Run the full test suite
