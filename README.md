# SimAI Components

<pre>
        |--- <a href="https://github.com/aliyun/aicb">AICB</a>
SimAI --|--- <a href="https://github.com/aliyun/SimCCL">SimCCL</a>
        |--- <a href="https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud">astra-sim-alibabacloud</a>
        |--- <a href="https://github.com/aliyun/ns-3-alibabacloud">ns-3-alibabacloud</a>
</pre>

# Table of Contents
- [SimAI Overview](#simai-overview)
  - [Introduction](#introduction)
  - [Citation](#citation)
- [Usage](#usage)
  - [Setup](#setup)
    - [From Source Code](#from-source-code)
  - [Use SimAI-Analytical](#use-simai-analytical)
  - [Use SimAI-Simulation](#use-simai-simulation)

# SimAI Overview
## Introduction

**SimAI** is the industry's first full-stack, high-precision **Sim**ulator for **AI** large-scale training. It provides detailed modeling and simulation of the entire LLM training process, encompassing framework, collective communication, network layers, and more. This comprehensive approach offers end-to-end performance data, enabling researchers to:

- Analyze training process details
- Evaluate the time consumption of AI tasks under specific conditions
- Evaluate E2E performance gains from various algorithmic optimizations including:
  - Framework parameters settings
  - Collective communication algorithms
  - NCCL environment variables
  - Network transmission protocols
  - Congestion control algorithms
  - Adaptive routing algorithms
  - Scale-up/out network topology modifications
  - ...

Building on pure simulation capabilities, SimAI has evolved into a versatile full-stack toolkit comprising four components ([aicb](https://github.com/aliyun/aicb), [SimCCL](https://github.com/aliyun/SimCCL), [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud), [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud)). These components can be combined in various ways to achieve different functionalities. Below, we present the six main usage scenarios for SimAI. We encourage users to explore even more possibilities with this powerful tool.

| Scenario | Description | Component Combination |
|----------|-------------|------------------------|
| 1. AICB Test Suite | Run communication patterns on GPU clusters using AICB Test suite | [AICB](https://github.com/aliyun/aicb) |
| 2. AICB/AIOB Workload | Model compute/communication patterns of training process to generate workload | [AICB](https://github.com/aliyun/aicb) |
| 3. Collective Comm Analyze | Break down collective communication operations into point-to-point communication sets | [SimCCL](https://github.com/aliyun/SimCCL) |
| 4. Collective Comm w/o GPU | Perform RDMA collective communication traffic on non-GPU clusters | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(physical) |
| 5. SimAI-Analytical | Conduct rapid AICB workload analysis and simulation on any server (ignoring underlying network details) | [AICB](https://github.com/aliyun/aicb) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(analytical) |
| 6. SimAI-Simulation | Perform full simulation on any server | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(simulation) + [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) |

Below is the architecture diagram of the SimAI Simulator:
![SimAI_Arc](./docs/images/SimAI_Arc.png)

astra-sim-alibabacloud is extended from [astra-sim](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0), we have integrated NCCL algorithms and added some new features.

## Citation

SimAI work has been accepted by NSDI'25 Spring, and it'll be showcased soon.

Paper name is "SimAI: Unifying Architecture Design and Performance Tunning for Large-Scale Large Language Model Training with Scalability and Precision"  [pdf]() / [slides]() / [video]()

# Usage

Here are some simple examples, SimAI full tutorials can be found here: [**SimAI@Tutorial**](docs/Tutorial.md), [**aicb@Tutorial**](https://github.com/aliyun/aicb/blob/master/training/tutorial.md), [SimCCL@Tutorial], [ns-3-alibabacloud@Tutorial]

## Setup

You can follow the instrucitons below to quickly set up the environtments and run SimAI

### From Source Code

The following code has been successfully tested on GCC/G++ 9.4.0, python 3.8.10 in Ubuntu 20.04

```bash
# Clone the repository
$ git clone https://github.com/aliyun/SimAI.git
$ cd ./SimAI/

# Clone submodules
$ git submodule update --init --recursive
# Make sure use the newest commit
$ git submodule update --remote

# Compile SimAI-Analytical
$ ./scripts/build.sh -c analytical

# Compile SimAI-Simulation (ns3)
$ ./scripts/build.sh -c ns3

```

## Use SimAI-Analytical

```bash
$  ./bin/SimAI_analytical -w example/workload_analytical.txt -g 1024 -g_p_s 8 -r results/test- -busbw example/busbw.yaml
```

## Use SimAI-Simulation

```bash
# Create network topo
$ python3 ./astra-sim-alibabacloud/inputs/topo/gen_HPN_7.0_topo_mulgpus_one_link.py -g 128 -gt A100 -bw 400Gbps -nvbw 2400Gbps

# Running
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./HPN_7_0_128_gpus_8_in_one_server_with_400Gbps_A100

```

