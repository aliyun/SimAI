# Installation

This guide covers how to install SimAI and its dependencies.

## Option A: Docker (Recommended)

```bash
# Build the Docker image
docker build -t simai:latest .

# Run a container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd)/results:/workspace/SimAI/results \
    simai:latest /bin/bash
```

> **Note:** If using Hopper GPUs, add `ENV FLASH_MLA_DISABLE_SM100=1` to the Dockerfile.

## Option B: Build from Source

The following instructions have been tested on GCC/G++ 9.4.0, Python 3.8.10, Ubuntu 20.04.

> **Important:** Do not install ninja (it is pre-installed in NGC images and must be removed for SimAI-Simulation compilation).
> ```bash
> apt remove ninja-build && pip uninstall ninja
> ```

### Step 1: Clone the Repository

```bash
git clone https://github.com/aliyun/SimAI.git
cd ./SimAI/

# Initialize submodules
git submodule update --init --recursive
# Update to latest commits
git submodule update --remote
```

### Step 2: Compile C++ Components

Choose the mode(s) you need:

```bash
# SimAI-Analytical (fast, abstracts network details)
./scripts/build.sh -c analytical

# SimAI-Simulation (full-stack with NS-3 network backend)
./scripts/build.sh -c ns3

# SimAI-Physical (beta, requires RDMA environment)
sudo yum install openmpi openmpi-devel
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++
./scripts/build.sh -c phy
```

### Step 3: Install Python Dependencies

```bash
pip install -r aicb/requirements.txt
pip install -r vidur-alibabacloud/requirements.txt
```

### Step 4: Verify the Build

```bash
ls bin/  # Should contain SimAI_analytical and/or SimAI_simulator
```

## Option C: Conda Environment (for Inference Simulation)

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## NGC Container (for Workload Generation)

For generating workloads with computation profiling (AIOB), we recommend using NGC container images directly:

```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm \
    -v /path/to/SimAI:/workspace/SimAI \
    nvcr.io/nvidia/pytorch:xx.xx-py3
```

> **Note:** Use PyTorch >= 23.08 NGC images.

## What's Next

- [Quickstart Guide](quickstart.md) — Run your first simulation
- [User Guide](../user_guide/index.md) — Detailed usage for each mode
