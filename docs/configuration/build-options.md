# SimAI Build Options

> [中文版](../CN/configuration/build-options.md)

## Build Modes

| Mode | Command | Output Binary | Description |
|---|---|---|---|
| analytical | `./scripts/build.sh -c analytical` | `bin/SimAI_analytical` | Fast busbw-based simulation (no network modeling) |
| ns3 | `./scripts/build.sh -c ns3` | `bin/SimAI_simulator` | Full ns3 network simulation (high fidelity) |
| phy | `./scripts/build.sh -c phy` | `bin/SimAI_phynet` | Physical RDMA traffic generation (requires IB hardware) |

## Clean Build

```bash
./scripts/build.sh -l ns3         # Clean ns3 build artifacts
./scripts/build.sh -l analytical  # Clean analytical build
./scripts/build.sh -l phy         # Clean physical build
```

## SimAI_simulator Parameters (ns3 mode)

| Parameter | Long Form | Description | Default |
|---|---|---|---|
| `-t` | `--thread` | Number of threads for multi-threading acceleration | 1 |
| `-w` | `--workload` | Path to workload file | Required |
| `-n` | `--network-topo` | Path to network topology file | Required |
| `-c` | `--config` | Path to simulation config file | Required |

Typical usage:
```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

> **Threading**: Recommended 8-16 threads for multi-threaded mode. Higher thread count reduces simulation time for large topologies.

## SimAI_analytical Parameters

| Parameter | Long Form | Description | Default |
|---|---|---|---|
| `-w` | `--workload` | Path to workload file | Required |
| `-g` | `--gpus` | Total GPU count | Required |
| `-g_p_s` | `--gpus-per-server` | GPUs per server (scale-up size) | Required |
| `-r` | `--result` | Output file path/prefix | `./results/` |
| `-busbw` | `--bus-bandwidth` | Path to busbw.yaml | Required |
| `-v` | `--visual` | Generate visualization files | Off |
| `-nv` | - | NVLink bandwidth (GB/s) for auto busbw | Optional |
| `-nic` | - | NIC bandwidth (GB/s) for auto busbw | Optional |
| `-n_p_s` | - | NICs per server for auto busbw | Optional |

Typical usage:
```bash
./bin/SimAI_analytical \
  -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 \
  -r test- \
  -busbw example/busbw.yaml
```

## Topology Generator Parameters

See [Quick Start - Generate Topology](../getting_started/quickstart.md#2-generate-topology) for full parameter reference.

---

> Last edited: 2026-06-25
