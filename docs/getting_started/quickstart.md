# SimAI Quick Start Guide

> [中文版](../CN/getting_started/quickstart.md)

## End-to-End Running Guide

### 1. Compile SimAI-Simulation (ns3)

```bash
cd SimAI/
./scripts/build.sh -c ns3
```

Produces: `bin/SimAI_simulator` (symlink to the ns3 build output).

### 2. Generate Topology

Use the topology generator script:

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

#### Key Parameters

| Parameter | Description | Unit/Values | Example |
|---|---|---|---|
| `-topo` | Template name | `Spectrum-X`, `AlibabaHPN`, `DCN+` | `-topo Spectrum-X` |
| `-g` | Total GPU count | Integer | `-g 8` |
| `-gt` | GPU type | `A100`, `H100`, `H800`, `H20` | `-gt H20` |
| `-gps` | GPUs per server | Integer (default: 8) | `-gps 8` |
| `-bw` | NIC bandwidth (scale-out) | e.g., `100Gbps`, `200Gbps`, `400Gbps` | `-bw 200Gbps` |
| `-nvbw` | NVLink bandwidth (scale-up) | e.g., `2400Gbps`, `2880Gbps` | `-nvbw 2400Gbps` |
| `--ro` | Rail-optimized topology | Flag (no value) | `--ro` |
| `-psn` | PSW switch number | Integer | `-psn 64` |
| `--dp` | Dual-plane | Flag | `--dp` |
| `--dt` | Dual-ToR | Flag | `--dt` |
| `-nl` | NVLink latency | e.g., `0.000025ms` | `-nl 0.000025ms` |
| `-l` | NIC latency | e.g., `0.0005ms` | `-l 0.0005ms` |

#### Output Filename Convention

Output file is named: `{template}_{g}g_{gps}gps_{bw}_{gt}`

Examples:
- `--ro -g 8 -gt H20 -bw 200Gbps` → `Rail_Opti_SingleToR_8g_8gps_200Gbps_H20`
- `-topo Spectrum-X -g 128 -gt A100 -bw 100Gbps` → `Spectrum-X_128g_8gps_100Gbps_A100`

### 3. Run Simulation

```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

#### Simulator Parameters

| Parameter | Description | Default |
|---|---|---|
| `-t` | Number of threads (multi-threading acceleration) | 1 |
| `-w` | Path to workload file | Required |
| `-n` | Path to network topology file | Required |
| `-c` | Path to config file | Required |

#### Environment Variables (set before command)

| Variable | Unit | Default | Description |
|---|---|---|---|
| `AS_SEND_LAT` | **nanoseconds (ns)** | Not set (use send_lat_table) | Override per-flow send latency. Overrides all table lookups. |
| `AS_NVLS_ENABLE` | - | `0` | Enable NVLS algorithm for AllReduce |
| `AS_PXN_ENABLE` | - | `0` | Enable PXN cross-node proxy |
| `AS_LOG_LEVEL` | - | `INFO` | Log verbosity: DEBUG/INFO/WARNING/ERROR |

> **Important**: `AS_SEND_LAT` unit is **nanoseconds**, not microseconds. The send_lat_table default values range from 6000-22000 ns (6-22 us). Setting `AS_SEND_LAT=6` means 6 ns (effectively disabling send latency).

### 4. Expected Output

Simulation generates these files in the current working directory:

| File | Description |
|---|---|
| `ncclFlowModel_EndToEnd.csv` | End-to-end iteration timing summary |
| `ncclFlowModel_detailed_N.csv` | Per-layer detailed timing (N = node count) |
| `ncclFlowModel_detailed_flows.csv` | SimCCL point-to-point flow decomposition |
| `ncclFlowModel_*_dimension_utilization_*.csv` | Communication group utilization |

Verify output:
```bash
ls ncclFlowModel_*.csv
head -5 ncclFlowModel_EndToEnd.csv
```

### 5. Verify AS_SEND_LAT Effect (A/B Experiment)

**Design principle**: Single-variable experiment. Same topo, same workload, only change `AS_SEND_LAT`.

```bash
# Baseline: use send_lat_table (no override)
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# Record: total time from EndToEnd.csv

# Experiment: override with AS_SEND_LAT=6000 (6 us, close to default Ring+LL+NVLINK=7200ns)
AS_SEND_LAT=6000 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# Record: total time from EndToEnd.csv
```

#### Measured Results (8 GPU, H20, microAllReduce.txt)

| Condition | Total Time | Notes |
|---|---|---|
| Baseline (send_lat_table) | 389,404 | Table values: Ring+LL+NVLINK=7200ns, etc. |
| AS_SEND_LAT=6 (6 ns) | 1,772 | Negligible send latency |
| AS_SEND_LAT=6000 (6 us) | 169,604 | Uniform 6us, lower than table average |
| AS_SEND_LAT=7200 (7.2 us) | 203,204 | Matches Ring+LL+NVLINK table value |

**Analysis**:
- The baseline uses per-(algorithm, protocol, link_type) table values, averaging higher than any single value
- `AS_SEND_LAT` overrides ALL flows to one value, losing per-type differentiation
- End-to-end time is affected by topology, congestion, protocol, algorithm, and payload size — not just send_lat
- Use this experiment to verify the relative impact of send latency changes

---

## Smoke Test (8 GPU, Single Node)

Complete single-node test in under 1 minute:

```bash
# 1. Generate topology
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# 2. Run
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# 3. Verify
cat ncclFlowModel_EndToEnd.csv | head -3
```

This tests intra-node communication (NVLINK). All flows stay within one node.

---

## Cross-Node Test (16 GPU, 2 Nodes)

To verify cross-node (NET/IB) behavior:

### 1. Create 16 GPU Workload

Modify `example/microAllReduce.txt` to set `all_gpus: 16`:

```bash
# Copy and modify
cp example/microAllReduce.txt example/microAllReduce_16g.txt
# Edit: change "all_gpus: 8" to "all_gpus: 16"
sed -i 's/all_gpus: 8/all_gpus: 16/' example/microAllReduce_16g.txt
```

### 2. Generate 16 GPU Topology (2 nodes x 8 GPUs)

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 16 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

Output: `Rail_Opti_SingleToR_16g_8gps_200Gbps_H20`

### 3. Run Cross-Node Simulation

```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce_16g.txt \
  -n ./Rail_Opti_SingleToR_16g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

This exercises both NVLINK (intra-node) and NET (inter-node) communication paths. The send_lat_table applies different values for each link type.

---

## Related Documentation

- [Installation Guide](installation.md) — Build instructions for all modes
- [Environment Variables](../configuration/env-variables.md) — Complete variable reference
- [Build Options](../configuration/build-options.md) — Build modes and parameters
- [send_lat Analysis](../configuration/send-lat-analysis.md) — Deep dive into send_lat mechanism
- [SimCCL Integration](../../SimCCL/docs/integration/integration-with-simai.md) — SimCCL + SimAI integration
- [SimCCL Standalone Guide](../../SimCCL/docs/getting_started/quickstart.md) — Independent flow analysis

---

> Last edited: 2026-08-14
