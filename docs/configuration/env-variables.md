# SimAI Environment Variables Reference

> [中文版](../CN/configuration/env-variables.md)

## Simulation Environment Variables

These variables control the behavior of `SimAI_simulator` (ns3 mode).

| Variable | Default | Unit | Scope | Description |
|---|---|---|---|---|
| `AS_SEND_LAT` | Not set (use table) | **nanoseconds (ns)** | entry.h (ns3 frontend) | Override per-flow send latency. Highest priority — overrides all send_lat_table lookups. |
| `AS_NVLS_ENABLE` | `0` | - | MockNcclGroup.cc | Enable NVLS algorithm for AllReduce (H20/H100/H800) |
| `AS_NVLSTREE_ENABLE` | `0` | - | MockNcclGroup.cc | Enable NVLS Tree algorithm |
| `AS_PXN_ENABLE` | `0` | - | MockNcclGroup.cc | Enable PXN cross-node proxy |
| `AS_LOG_LEVEL` | `INFO` | - | System-wide | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### AS_SEND_LAT Details

**Unit: nanoseconds (ns)**. The send_lat_table default values range from 6000 to 22000 ns depending on algorithm, protocol, and link type.

| Example Value | Meaning |
|---|---|
| Not set | Use send_lat_table[algo][proto] (7200ns for Ring+LL+NVLINK, 22000ns for PAT+Simple+NET, etc.) |
| `AS_SEND_LAT=6` | 6 ns — effectively disables send latency (for fast testing) |
| `AS_SEND_LAT=7200` | 7200 ns = 7.2 us — matches Ring+LL+NVLINK table value |
| `AS_SEND_LAT=22000` | 22000 ns = 22 us — matches PAT+Simple+NET table value |

> **Warning**: Some older documentation (Tutorial.md) describes AS_SEND_LAT as "unit is us, default 6". This is inaccurate. The actual unit is **nanoseconds** as confirmed by code analysis and experiment.

#### Quick Commands

```bash
# Prerequisites
cd SimAI/ && ./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# Baseline (use send_lat_table)
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# Override send latency (7200ns = Ring+LL+NVLINK table value)
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

For full A/B experiment guide, see [Quick Start](../getting_started/quickstart.md#5-verify-as_send_lat-effect-ab-experiment).

## SimCCL Integration Variables

SimCCL (the collective communication modeling layer) has its own environment variables. See:
- [SimCCL Environment Variables](../../SimCCL/docs/configuration/env-variables.md) — Standalone variables
- [SimCCL Integration Guide](../../SimCCL/docs/integration/integration-with-simai.md) — Cross-module variables

## Relationship with send_lat_table

The `send_lat_table` in `entry.h` provides per-(algorithm, protocol, link_type) send latency values. When `AS_SEND_LAT` is not set, each flow uses the table value matching its algorithm and protocol. When set, ALL flows use the override value.

For detailed analysis of the table mechanism, see [send-lat-analysis.md](send-lat-analysis.md).

---

> Last edited: 2026-06-25
