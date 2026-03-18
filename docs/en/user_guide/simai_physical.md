# SimAI-Physical Mode

> **Status**: Beta — Currently in internal testing phase.

SimAI-Physical enables physical traffic generation for CPU RDMA cluster environments. This mode generates NCCL-like traffic patterns, allowing in-depth study of NIC behaviors during LLM training.

---

## Overview

Unlike SimAI-Analytical and SimAI-Simulation which run entirely in software simulation, SimAI-Physical injects real RDMA traffic into a physical network. This enables:

- Studying actual NIC behavior under realistic LLM training traffic patterns
- Validating network configurations on real hardware
- Benchmarking RDMA performance with representative collective communication workloads

**Component Combination**: [AICB](../../components/aicb.md) + [SimCCL](../../components/simccl.md) + [astra-sim-alibabacloud](../../components/astra_sim.md) (physical mode)

---

## Prerequisites

SimAI-Physical uses the RoCEv2 protocol for traffic generation. Before compilation, ensure your environment meets:

- **RDMA Support**: Working `libibverbs` / RDMA device drivers
- **MPI**: OpenMPI installed and functional
- **Verification**: Successfully run `ib_write_bw` or similar RDMA perftest tools

---

## Compilation

```bash
# Clone and initialize
git clone https://github.com/aliyun/SimAI.git
cd SimAI/
git submodule update --init --recursive
git submodule update --remote

# Install MPI (CentOS/RHEL)
sudo yum install openmpi openmpi-devel

# Set MPI paths
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++

# Build SimAI-Physical
./scripts/build.sh -c phy
```

---

## Workload Generation

SimAI-Physical uses the same workload format as SimAI-Simulation, generated through [AICB](../../components/aicb.md). See [Workload Generation](workload_generation.md) for details.

### Example Workload

```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 2 ep: 1 pp: 1 vpp: 8 ga: 1 all_gpus: 2 checkpoints: 0 checkpoint_initiates: 0
10
mlp_norm    -1  1055000  ALLGATHER  1073741824  1055000  NONE  0  1055000  NONE  0  100
mlp_norm    -1  1055000  ALLGATHER  1073741824  1055000  NONE  0  1055000  NONE  0  100
...
```

---

## Prepare the Host List

Prepare an IP list file for the MPI program. The number of IPs should match the number of NICs involved in physical traffic generation (not the number of nodes).

```
33.255.199.130
33.255.199.129
```

---

## Running

### MPI Execution

```bash
/usr/lib64/openmpi/bin/mpirun -np 2 \
  -host 33.255.199.130,33.255.199.129 \
  --allow-run-as-root \
  -x AS_LOG_LEVEL=0 \
  ./bin/SimAI_phynet ./hostlist -g 2 -w ./example/microAllReduce.txt
```

### MPI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-np` | Number of processes | Required |
| `-host` | Comma-separated IP list | Required |
| `--allow-run-as-root` | Allow running as root | `FALSE` |

### SimAI-Physical Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hostlist` | Path to host IP list file | Required |
| `-w` / `--workload` | Path to workload file | `./microAllReduce.txt` |
| `-i` / `--gid_index` | GID index for RDMA device | `0` |
| `-g` / `--gpus` | Number of GPUs (must match IP count in hostlist) | `8` |

---

## Notes

- The number of GPUs (`-g`) must be consistent with the number of IPs in the host IP list
- Ensure all nodes have network connectivity and RDMA is properly configured
- SimAI-Physical is currently in beta; some features may change in future releases

---

## See Also

- [AICB Component](../components/aicb.md) — Workload generation
- [SimCCL Component](../components/simccl.md) — Collective communication transformation
- [astra-sim Component](../components/astra_sim.md) — Simulation engine details
