# Lastest News

<div align="center">
🎯 <b>Events & Community Engagement</b> 🎯

### 📅 Upcoming Events

| Date | Event | Location | Content | Type |
|:----:|:------|:---------|:--------|:----:|
| **Dec 27, 2024** | **SimAI Technical Presentation** | 📍 Beihang University | SimAI Technical Sharing & Discussion | 🎓 On-site |

### 🌟 Past Events

| Date | Event | Location | Content | Type |
|:----:|:------|:---------|:--------|:----:|
| Dec 6, 2024 | HKUST Technical Workshop | 📍 HKUST(GZ) | SimAI Technical Sharing & Discussion | 🎓 On-site |
| Dec 5, 2024 | [Bench'24 Conference](https://mp.weixin.qq.com/s/STic_E12xMhZRxhzK9wRnw) | 📍 Guangzhou | SimAI Tutorial & Deep-dive Session | 🎓 On-site |
| Nov 26, 2024 | SimAI Community Live Stream | 🌐 Online | Interactive Technical Discussion & Demo (400+ Attendees) | 💻 Virtual |
| Nov 15, 2024 | Technical Workshop | 📍 Thousand Island Lake | SimAI Offline Technical Exchange | 🎯 On-site |
| Oct 18, 2024 | Guest Lecture | 📍 Fudan University | SimAI Tutorial & Public Course | 🎓 On-site |
| Sept 24-26, 2024 | CCF HPC China 2024 | 📍 Wuhan | SimAI Introduction & Technical Presentation | 🎤 Conference |
</div>

---

# Table of Contents
- [SimAI Overview](#simai-overview)
  - [Introduction](#introduction)
  - [Components](#components)
  - [Scenario](#scenario)
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

## Components

<pre>
        |--- <a href="https://github.com/aliyun/aicb">AICB</a>
SimAI --|--- <a href="https://github.com/aliyun/SimCCL">SimCCL</a>
        |--- <a href="https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud">astra-sim-alibabacloud</a>
        |--- <a href="https://github.com/aliyun/ns-3-alibabacloud">ns-3-alibabacloud</a>
</pre>

Building on pure simulation capabilities, SimAI has evolved into a versatile full-stack toolkit comprising four components ([aicb](https://github.com/aliyun/aicb), [SimCCL](https://github.com/aliyun/SimCCL), [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud), [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud)). These components can be combined in various ways to achieve different functionalities. Below, we present the six main usage scenarios for SimAI. We encourage users to explore even more possibilities with this powerful tool.

Below is the architecture diagram of the SimAI Simulator:
![SimAI_Arc](./docs/images/SimAI_Arc.png)

astra-sim-alibabacloud is extended from [astra-sim](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0). We are grateful to the astra-sim team for their excellent work and open-source contribution. We have integrated NCCL algorithms and added some new features.

## Scenario

SimAI supports three major operation modes to meet different simulation requirements:

**SimAI-Analytical** offers fast simulation by abstracting network communication details using bus bandwidth (busbw) to estimate collective communication time. While it currently supports user-defined busbw, automatic busbw calculation feature is coming soon.

**SimAI-Simulation** provides full-stack simulation with fine-grained network communication modeling. It leverages NS3 or other network simulators (NS3 currently open-sourced) to achieve detailed simulation of all communication behaviors, aiming for high-fidelity reproduction of actual training environments.

**SimAI-Physical** *(Beta)* enables physical traffic generation for CPU RDMA cluster environments. This mode generates NCCL-like traffic patterns, allowing in-depth study of NIC behaviors during LLM training. It is currently in internal testing phase.

| Scenario | Description | Component Combination |
|----------|-------------|------------------------|
| 1. AICB Test Suite | Run communication patterns on GPU clusters using AICB Test suite | [AICB](https://github.com/aliyun/aicb) |
| 2. AICB/AIOB Workload | Model compute/communication patterns of training process to generate workload | [AICB](https://github.com/aliyun/aicb) |
| 3. Collective Comm Analyze | Break down collective communication operations into point-to-point communication sets | [SimCCL](https://github.com/aliyun/SimCCL) |
| 4. Collective Comm w/o GPU | Perform RDMA collective communication traffic on non-GPU clusters | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(physical) |
| 5. SimAI-Analytical | Conduct rapid AICB workload analysis and simulation on any server (ignoring underlying network details) | [AICB](https://github.com/aliyun/aicb) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(analytical) |
| 6. SimAI-Simulation | Perform full simulation on any server | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(simulation) + [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) |


## Citation

SimAI work has been accepted by NSDI'25 Spring, for more details, please refer to our paper below:

*SimAI: Unifying Architecture Design and Performance Tunning for Large-Scale Large Language Model Training with Scalability and Precision.*

[[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)] / [[slides](./docs/SimAI_Intro_Online.pdf)] / [video]

We encourage innovative research and extensions based on SimAI. Welcome to join our community group or reach out via email for discussion. We may provide technical support.

# Quick Start

Here are some simple examples, SimAI full tutorials can be found here: [**SimAI@Tutorial**](./docs/Tutorial.md), [**aicb@Tutorial**](https://github.com/aliyun/aicb/blob/master/training/tutorial.md), [SimCCL@Tutorial], [ns-3-alibabacloud@Tutorial]

## Setup

You can follow the instrucitons below to quickly set up the environtments and run SimAI

### From Source Code

The following code has been successfully tested on GCC/G++ 9.4.0, python 3.8.10 in Ubuntu 20.04

You can use the official Ubuntu 20.04 image, and do not install ninja.

(For generation workloads, it's recommended to leverage NGC container images directly.)

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
$  ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

## Use SimAI-Simulation

```bash
# Create network topo
$ python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# Running
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf

```

# Contact us

Please email Gang Lu (yunding.lg@alibaba-inc.com) or Qingxu Li (qingxu.lqx@alibaba-inc.com) if you have any questions.

Welcome to join the SimAI community chat groups, with the DingTalk group on the left and the WeChat group on the right.

<div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-left: 20px;">
    <img src="./docs/images/simai_dingtalk.jpg" alt="SimAI DingTalk" style="width: 300px; height: auto;">
    <img src="./docs/images/simai_wechat.jpg" alt="SimAI WeChat" style="width: 300px; height: auto;">
</div>

<br/>
