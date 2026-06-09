<p align="left">
    中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp ｜ &nbsp<a href="README.ja.md">日本語</a>
</p>

# SimAI

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NSDI'25](https://img.shields.io/badge/NSDI'25-SimAI-blue.svg)](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)

# 最新动态

### 近期更新

- [2026/04] **SimAI 1.6 正式发布！** 主要更新：
  - 推理仿真 GPU 显存建模（参数计数与 KV Cache 管理）。
  - Decode 耗时线性插值估算（替代最近邻查找）。
  - PD 分离内存规划（Prefill/Decode 独立预算）。

- [2025/12] **SimAI 1.5 正式发布！** 本版本新增对多请求**推理**工作负载的端到端仿真支持，主要特性包括：

  - **高级推理仿真：** 支持 Prefill/Decode 分离等复杂场景建模。
  - **主流模型支持：** 新增 DeepSeek、Qwen3Moe 和 Qwen3Next 模型。详见 [AICB README](./aicb/README.md)。
  - **请求调度：** 请求调度组件基于微软 [Vidur](https://github.com/microsoft/vidur) 适配，详见 [Vidur-Alibabacloud README](./vidur-alibabacloud/README_CN.md)。

- [2025/11] [AICB](https://github.com/aliyun/aicb/tree/master) 新增对 **DeepSeek**、**Qwen3-MoE** 和 **Qwen3-Next** 的 **prefill/decode** 推理工作负载生成支持。

- [2025/09] [AICB](https://github.com/aliyun/aicb/tree/master) 新增 DeepSeek 训练工作负载生成支持。感谢 [@parthpower](https://github.com/parthpower) 的贡献。

- [2025/06] SimCCL 代码首次在 [SimCCL](https://github.com/aliyun/SimAI/tree/SimCCL) 分支发布，后续将在独立仓库正式开源。

**欢迎社区贡献！** 如有想法，欢迎提交 Issue 讨论或发起 Pull Request。

<div align="center">
🎯 <b>活动与社区</b> 🎯

### 📅 即将举办

| 日期 | 活动 | 地点 | 内容 | 形式 |
|:----:|:----- |:-------- |:------- |:----:|
| --   |       |          |         |      |

### 🌟 往期活动

| 日期             | 活动                                                                     | 地点                    | 内容                                                     | 形式          |
|:----------------:|:------------------------------------------------------------------------ |:----------------------- |:-------------------------------------------------------- |:-------------:|
| Apr 23, 2026     | SimAI 1.6                                                                | 🌐 线上                 | SimAI 1.6 正式发布                                       | 💻 线上直播   |
| Dec 30, 2025     | SimAI 1.5                                                                | 🌐 线上                 | SimAI 1.5 正式发布                                       | 💻 线上直播   |
| Jun 4, 2025      | SimAI 社区第一届研讨会                                                   | 📍 北京大学             | 三场社区贡献者演讲                                       | 🎓 线下       |
| May 24, 2025     | 第 28 届 Chinasys 研讨会                                                 | 📍 重庆大学             | SimAI 受邀演讲                                           | 🎓 线下       |
| Dec 27, 2024     | SimAI 技术分享                                                           | 📍 北京航空航天大学     | SimAI 技术分享与交流                                     | 🎓 线下       |
| Dec 6, 2024      | 香港科技大学技术研讨会                                                   | 📍 香港科技大学（广州） | SimAI 技术分享与交流                                     | 🎓 线下       |
| Dec 5, 2024      | [Bench'24 会议](https://mp.weixin.qq.com/s/STic_E12xMhZRxhzK9wRnw)      | 📍 广州                 | SimAI 教程与深度技术专场                                 | 🎓 线下       |
| Nov 26, 2024     | SimAI 社区直播                                                           | 🌐 线上                 | 互动技术交流与演示（400+ 参与者）                        | 💻 线上直播   |
| Nov 15, 2024     | 技术研讨会                                                               | 📍 千岛湖               | SimAI 线下技术交流                                       | 🎯 线下       |
| Oct 18, 2024     | 嘉宾讲座                                                                 | 📍 复旦大学             | SimAI 教程与公开课                                       | 🎓 线下       |
| Sept 24-26, 2024 | CCF HPC China 2024                                                       | 📍 武汉                 | SimAI 介绍与技术报告                                     | 🎤 会议       |

</div>

---

## 文档

详见 [Tutorial](./docs/Tutorial.md) 获取完整文档。

---

# 目录

- [SimAI 概述](#simai-概述)
  - [简介](#简介)
  - [组件](#组件)
  - [应用场景](#应用场景)
  - [引用](#引用)
- [快速开始](#快速开始)
  - [环境搭建](#环境搭建)
  - [使用 SimAI-Analytical](#使用-simai-analytical)
  - [使用 SimAI-Simulation](#使用-simai-simulation)
  - [使用多请求推理仿真](#使用多请求推理仿真)

# SimAI 概述

## 简介

**SimAI** 是业界首个全栈高精度 AI 大规模**推理**与**训练**模拟器（**Sim**ulator for **AI**）。它对 LLM 训练全流程进行详细建模和仿真，涵盖框架、集合通信、网络层等，提供端到端的性能数据，帮助研究人员：

- 分析推理/训练过程细节
- 评估特定条件下 AI 任务的耗时
- 评估各类算法优化带来的 E2E 性能收益，包括：
  - 框架参数配置
  - 集合通信算法
  - NCCL 环境变量
  - 网络传输协议
  - 拥塞控制算法
  - 自适应路由算法
  - 扩展/集合网络拓扑调整
  - ……

## 组件

<pre>
        |--- <a href="https://github.com/aliyun/aicb">AICB</a>
SimAI --|--- <a href="https://github.com/aliyun/SimCCL">SimCCL</a>
        |--- <a href="https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud">astra-sim-alibabacloud</a>
        |--- <a href="https://github.com/aliyun/ns-3-alibabacloud">ns-3-alibabacloud</a>
        |--- vidur-alibabacloud
</pre>

在纯仿真能力基础上，SimAI 已演进为一个由四个组件（[aicb](https://github.com/aliyun/aicb)、[SimCCL](https://github.com/aliyun/SimCCL)、[astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)、[ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud)）构成的全栈工具套件。这些组件可以灵活组合以实现不同功能。我们鼓励用户探索更多可能性。

下图为 SimAI 模拟器架构图：
![SimAI_Arc](./docs/images/SimAI_Arc.png)

astra-sim-alibabacloud 基于 [astra-sim](https://github.com/astra-sim/astra-sim/tree/ASTRA-sim-1.0) 扩展开发。感谢 astra-sim 团队的优秀工作和开源贡献。我们在其基础上集成了 NCCL 算法并添加了若干新特性。

## 应用场景

SimAI 支持三种主要运行模式：

**SimAI-Analytical** 通过使用总线带宽（busbw）抽象网络通信细节来估算集合通信时间，实现快速仿真。目前支持用户自定义 busbw，自动计算 busbw 功能即将推出。

**SimAI-Simulation** 提供基于细粒度网络通信建模的全栈仿真。利用 NS-3 或其他网络模拟器（当前 NS-3 已开源）实现对所有通信行为的详细仿真，力求高保真还原真实训练环境。

**SimAI-Physical** *(Beta)* 支持在 CPU RDMA 集群环境下生成物理流量，通过生成类 NCCL 的流量模式深入研究 LLM 训练中的 NIC 行为。当前处于内测阶段。

| 场景 | 描述 | 组件组合 |
|------|------|----------|
| 1. AICB 测试套件 | 在 GPU 集群上使用 AICB 测试套件运行通信模式 | [AICB](https://github.com/aliyun/aicb) |
| 2. AICB/AIOB 工作负载 | 建模**推理**/训练过程的计算/通信模式以生成工作负载 | [AICB](https://github.com/aliyun/aicb) |
| 3. 集合通信分析 | 将集合通信操作分解为点对点通信集合 | [SimCCL](https://github.com/aliyun/SimCCL) |
| 4. 无 GPU 集合通信 | 在非 GPU 集群上执行 RDMA 集合通信流量 | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(physical) |
| 5. SimAI-Analytical | 在任意服务器上快速进行 AICB 工作负载分析与仿真（忽略底层网络细节） | [AICB](https://github.com/aliyun/aicb) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(analytical) |
| 6. SimAI-Simulation | 在任意服务器上进行全栈仿真 | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(simulation) + [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) |
| 7. 多请求推理仿真 | 在单 GPU 服务器上进行多请求**推理**全栈仿真 | [AICB](https://github.com/aliyun/aicb) + [SimCCL](https://github.com/aliyun/SimCCL) + [vidur-alibabacloud](./vidur-alibabacloud) + [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud)(analytical/simulation) |

## 引用

SimAI 论文已被 NSDI'25 Spring 接收，详情请参阅：

*SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision.*

[[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)] / [[slides](./docs/SimAI_Intro_Online.pdf)] / [[video](https://n.dingtalk.com/dingding/live-room/index.html?roomId=OF5BkBUXVxmgsK7x&liveUuid=305736cd-aa70-498b-8003-2b471a53decd)]

欢迎基于 SimAI 开展创新研究和功能扩展。欢迎加入社区群或通过邮件联系我们交流，我们可提供技术支持。

# 快速开始

以下为简单示例。完整教程请参见：[**SimAI@Tutorial**](./docs/Tutorial.md)、[**aicb@Tutorial**](https://github.com/aliyun/aicb/blob/master/training/tutorial.md)、[SimCCL@Tutorial]、[ns-3-alibabacloud@Tutorial]

## 环境搭建

请按照以下步骤快速搭建环境并运行 SimAI。

### 从源码安装

以下步骤已在 Ubuntu 20.04 的 GCC/G++ 9.4.0、python 3.8.10 环境下验证。

可使用官方 Ubuntu 20.04 镜像，**不要安装 ninja**。

（对于工作负载生成场景，推荐直接使用 NGC 容器镜像。）

```bash
# 克隆仓库
$ git clone https://github.com/aliyun/SimAI.git
$ cd ./SimAI/

# 初始化子模块
$ git submodule update --init --recursive
# 更新到最新提交
$ git submodule update --remote

# 编译 SimAI-Analytical
$ ./scripts/build.sh -c analytical

# 编译 SimAI-Simulation (ns3)
$ ./scripts/build.sh -c ns3
```

## 使用 SimAI-Analytical

```bash
$ ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

若需自动计算总线带宽，请尝试：

```bash
$ ./bin/SimAI_analytical -w ./example/workload_analytical.txt -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 -r example-
```

## 使用 SimAI-Simulation

```bash
# 生成网络拓扑
$ python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# 运行仿真
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

## 使用多请求推理仿真

详情请参见 `vidur-alibabacloud` 目录下的 [README](./vidur-alibabacloud/README_CN.md)。该模块利用 AICB 对**推理**工作负载的计算时间进行 profiling。由于依赖 DeepGEMM 和 FlashMLA 等特定硬件加速库，目前仅兼容基于 **Hopper（SM90）** 和 **Blackwell（SM100）** 架构的 NVIDIA GPU。

```bash
# 从 Dockerfile 构建
docker build -t image:latest .
docker run --gpus all -it --rm image:latest
```

**注意：** 若使用 Hopper GPU，请在 Dockerfile 中添加 `ENV FLASH_MLA_DISABLE_SM100=1`。

如需快速验证所有支持的推理场景（Qwen3-Next-80B、DeepSeek-671B、Qwen3-MoE-235B），可使用内置的四场景测试套件：

```bash
# 前置条件：conda activate vidur
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all
# 或单独运行某个场景：
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

> **前置条件：** 需先激活 `conda activate vidur` 环境。详见 [环境配置](./vidur-alibabacloud/README_CN.md#-环境配置)。
>
> 完整场景配置表与输出文件说明请参见 [Vidur-AlibabaCloud README](./vidur-alibabacloud/README_CN.md#四场景配置说明)。

# 致谢

衷心感谢以下人员和机构对本项目的贡献：

<!-- keep-english: contributor names and affiliations must remain in English -->
- TianHao Fu (Peking University) and [TELOS-syslab](https://github.com/TELOS-syslab/)
- Parth Parikh (KEYSIGHT)
- Sarah-Michelle Hammer & Ziyi Wang (TU-Berlin)
- Xinyue Li (BUPT)
- Tong Chen (Zhejiang University)
- Ming Wang (BUPT)
- Tao Jiang (Institute of Computing Technology, Chinese Academy of Sciences)

……以及众多来自社区的个人贡献者（详见 [Contributors to aliyun/SimAI](https://github.com/aliyun/SimAI/graphs/contributors)）。

同时感谢 Chenning Li（MIT CSAIL）发起了将 SimAI 集成到 [M4](https://github.com/netiken/m4) 的合作——M4 是一个新型创新模拟器。

**本项目持续欢迎更多贡献与建议。**

# 贡献指南

欢迎参与贡献！开始前请阅读以下指引：

| | |
|---|---|
| [贡献指南](./CONTRIBUTING.zh-CN.md) | 如何提交 Issue 和 Pull Request |
| [安全政策](./SECURITY_CN.md) | 如何报告安全漏洞 |
| [行为准则](./CODE_OF_CONDUCT_CN.md) | 社区行为规范 |
| [更新日志](./CHANGELOG_CN.md) | v1.5 起的版本历史 |

# 联系我们

如有任何问题，欢迎发送邮件至：Gang Lu（yunding.lg@alibaba-inc.com）、Feiyang Xue（xuefeiyang.xfy@alibaba-inc.com）或 Qingxu Li（qingxu.lqx@alibaba-inc.com）。

欢迎加入 SimAI 社区交流群，左侧为钉钉群，右侧为微信群。

<div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-left: 20px;">
    <img src="./docs/images/simai_dingtalk.jpg" alt="SimAI 钉钉群" style="width: 300px; height: auto;">
    <img src="./docs/images/simai_wechat.jpeg" alt="SimAI 微信群" style="width: 300px; height: auto;">
</div>
