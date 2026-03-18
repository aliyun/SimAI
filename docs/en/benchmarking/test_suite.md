# 4-Scenario End-to-End Test Suite

SimAI provides a pre-configured test suite covering 4 representative inference scenarios, enabling quick validation of all supported configurations.

---

## Overview

The test suite is located at `vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh` and covers different combinations of models, parallelism strategies, and PD disaggregation configurations.

---

## Running

```bash
# Run all 4 scenarios
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# Run a single scenario
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1

# Show help
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --help
```

> **Prerequisites**: `conda activate vidur` environment must be active.

---

## Shared Hardware Configuration

All scenarios share the following hardware settings:

| Parameter | Value |
|-----------|-------|
| GPU | H20 (h20_dgx) |
| NVLink Bandwidth | 1600 Gbps |
| RDMA Bandwidth | 800 Gbps |
| PD P2P Bandwidth | 800 Gbps |
| PD P2P Data Type | fp8 |
| Request Generator | Poisson, QPS=100 |
| Request Count | 4 |
| Prefill Tokens | 100 (fixed) |
| Decode Tokens | 8 (fixed) |

---

## Scenario Configuration

| Scenario | Model | PD Separation | World Size | TP | PP | EP | Global Scheduler |
|----------|-------|---------------|-----------|----|----|-----|-----------------|
| **1** | Qwen3-Next-80B (MoE) | No | 32 (dp=32) | 1 | 1 | 1 (default) | lor |
| **2** | Qwen3-Next-80B (MoE) | Yes (P=2, D=6) | 8 | 1 | 1 | 1 (default) | split_wise |
| **3** | DeepSeek-671B (MoE) | Yes (P=2, D=6) | 8 | 8 | 1 | 8 | split_wise |
| **4** | Qwen3-MoE-235B (MoE) | Yes (P=2, D=6) | 8 | 4 | 1 | 4 | split_wise |

### Scenario Details

- **Scenario 1**: Large-scale DP without PD separation — tests baseline throughput
- **Scenario 2**: Same model with PD separation — tests PD disaggregation overhead
- **Scenario 3**: DeepSeek-671B with large TP/EP — tests MoE with MLA attention
- **Scenario 4**: Qwen3-MoE-235B with moderate TP/EP — tests MHA/GQA attention model

---

## Output

### Output Directory

- **Via run_scenarios.sh**: `examples/vidur-ali-scenarios/simulator_output/`
- **Direct python**: `./simulator_output/`

### Output Files

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # Per-request metrics
├── chrome_trace.json       # Chrome DevTools timeline
├── config.json             # Configuration snapshot
└── plots/                  # Metric CSV/JSON files
```

### Logs

Run logs are saved to `examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log`.

---

## Architecture Comparison Examples

### RING vs NVLS (SimAI-Simulation)

```bash
# NVLS topology and run
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 32 -gt H100 -bw 400Gbps -nvbw 1360Gbps
AS_SEND_LAT=12 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100 -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# RING topology and run
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 32 -gt H100 -bw 400Gbps -nvbw 1440Gbps
AS_SEND_LAT=2 AS_PXN_ENABLE=1 ./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_32g_8gps_400Gbps_H100 -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

**Results** (busbw in GB/s):

| Message Size | NVLS | RING |
|-------------|------|------|
| 16M | 148.88 | 141.84 |
| 32M | 178.04 | 153.68 |
| 64M | 197.38 | 160.60 |
| 128M | 208.70 | 163.85 |
| 256M | 214.87 | 165.72 |
| 512M | 218.09 | 166.68 |

### Spectrum-X vs DCN+ (SimAI-Simulation)

```bash
# Generate topologies
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo DCN+ -g 256 -psn 64 -bw 400Gbps
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 256
```

**Results** (busbw in GB/s):

| Message Size | Spectrum-X | DCN+ SingleToR |
|-------------|------------|----------------|
| 16M | 33.10 | 23.33 |
| 64M | 42.05 | 23.68 |
| 256M | 45.10 | 36.21 |
| 512M | 45.65 | 36.24 |

---

## See Also

- [Inference Simulation](../user_guide/inference_simulation.md) — Full inference simulation guide
- [vidur-alibabacloud](../components/vidur.md) — Component documentation
- [Result Analysis](../user_guide/result_analysis.md) — Output interpretation
