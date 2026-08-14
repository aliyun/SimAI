# SimAI Installation Guide

> [中文版](../CN/getting_started/installation.md)

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) | Tested on Ubuntu 20.04/22.04 |
| GCC/G++ | 9.4.0+ | C++17 support required |
| Python3 | 3.8+ | For topology generation scripts |
| CMake | 3.14+ | Used by ns3 build |
| GPU/CUDA | **Not required** | Simulation is CPU-only |
| ninja | **Must NOT be installed** | ns3 build conflicts with ninja |

> **Note**: If using NGC container images (recommended for AICB workload generation), remove ninja first:
> ```bash
> apt remove ninja-build && pip uninstall ninja
> ```

## Clone and Initialize

```bash
git clone https://github.com/aliyun/SimAI.git
cd SimAI/

# Initialize all submodules (SimCCL, ns-3-alibabacloud, aicb, etc.)
git submodule update --init --recursive
git submodule update --remote
```

## Compile SimAI-Analytical

Fast busbw-based simulation (no network modeling):

```bash
./scripts/build.sh -c analytical
```

Output: `bin/SimAI_analytical`

## Compile SimAI-Simulation (ns3)

Full-stack ns3 network simulation:

```bash
./scripts/build.sh -c ns3
```

Output: `bin/SimAI_simulator`

> **Note**: Default mock version is v2.30 (protocol-aware, PAT support). To use legacy v2.20: `SIMAI_NCCL_VERSION=v2.20 ./scripts/build.sh -c ns3`

> **Note**: This command removes and rebuilds `astra-sim-alibabacloud/extern/network_backend/ns3-interface/`. First build takes 5-15 minutes depending on hardware.

## Compile SimAI-Physical

Physical RDMA traffic generation (requires InfiniBand hardware + MPI):

```bash
# Install MPI dependencies
sudo yum install openmpi openmpi-devel  # or apt install libopenmpi-dev

# Set MPI paths
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++

# Build
./scripts/build.sh -c phy
```

Output: `bin/SimAI_phynet`

## Docker Environment (Recommended)

```bash
# Enter container
docker exec -it <container_name> bash

# Inside container: compile ns3 mode
cd /path/to/SimAI
./scripts/build.sh -c ns3

# Verify
ls -la bin/SimAI_simulator
```

## Clean Build

To remove build artifacts and start fresh:

```bash
./scripts/build.sh -l ns3         # Clean ns3 build
./scripts/build.sh -l analytical  # Clean analytical build
./scripts/build.sh -l phy         # Clean physical build
```

## Verify Installation

```bash
# Check binaries exist
ls bin/SimAI_simulator bin/SimAI_analytical

# Quick smoke test (8 GPU, single node)
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# Expected: simulation completes, generates ncclFlowModel_EndToEnd.csv
ls ncclFlowModel_EndToEnd.csv
```

---

## Compile SimCCL Standalone (Optional)

For independent collective operation analysis without ns3:

```bash
cd SimCCL/standalone
bash build.sh v2.30
```

Output: `build/simccl-standalone` (~300KB, no ns3 dependency, CPU-only)

---

> Last edited: 2026-08-14
