# 欢迎使用 SimAI 文档

<p align="left">
    English&nbsp ｜ &nbsp<a href="../en/index.md">English</a>
</p>

[![License](https://img.shields.io/badge/license-MIT-green.svg)](../../LICENSE)
[![NSDI'25](https://img.shields.io/badge/NSDI'25-SimAI-blue.svg)](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)

**SimAI** 是阿里云开源的业界首个全栈高精度 AI 大规模**推理**与**训练**仿真器。它提供了对 LLM 训练和推理全流程的详细建模与仿真，涵盖框架层、集合通信层和网络传输层，提供端到端的性能数据。

SimAI 使研究人员能够：

- 分析推理/训练过程细节
- 评估特定条件下 AI 任务的时间消耗
- 评估各种算法优化带来的端到端性能提升（框架参数、集合通信算法、网络协议、拥塞控制、路由、拓扑等）

---

## 文档概览

| 章节 | 说明 |
|------|------|
| [快速入门](getting_started/index.md) | 安装、环境搭建、快速开始 |
| [用户指南](user_guide/index.md) | SimAI-Analytical、SimAI-Simulation、SimAI-Physical、推理仿真的详细使用方法 |
| [组件详情](components/index.md) | 各子模块详细文档：AICB、SimCCL、astra-sim、ns-3、vidur |
| [技术参考](technical_reference/index.md) | GPU 显存模块、CLI 参数、配置文件参考 |
| [基准测试](benchmarking/index.md) | 四场景端到端测试套件及基准测试结果 |
| [开发者指南](developer_guide/index.md) | 架构、贡献指南、添加模型、扩展 NS-3 |
| [社区](community/index.md) | 活动、联系方式、引用 |

---

## 系统架构

```
        |--- AICB                        （工作负载生成 & 计算性能分析）
SimAI --|--- SimCCL                      （集合通信算法分析）
        |--- astra-sim-alibabacloud      （仿真引擎：Analytical / Simulation / Physical）
        |--- ns-3-alibabacloud           （NS-3 网络后端）
        |--- vidur-alibabacloud          （多请求推理调度 & 显存管理）
```

![SimAI 架构图](../images/SimAI_Arc.png)

---

## 三种运行模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **SimAI-Analytical** | 使用总线带宽（busbw）估算集合通信时间的快速仿真 | 性能分析、并行参数优化、scale-up 探索 |
| **SimAI-Simulation** | 使用 NS-3 网络后端的全栈仿真，提供细粒度网络建模 | CC 算法研究、网络协议评估、新架构设计 |
| **SimAI-Physical** *(Beta)* | 在 CPU RDMA 集群上生成物理流量 | NIC 行为研究 |

---

## 支持的模型

- **DeepSeek-V3-671B** — MLA 注意力，256 个路由专家
- **Qwen3-MoE-235B** — MHA/GQA，128 个路由专家
- **Qwen3-Next-80B** — 混合全注意力 + 线性注意力，512 个路由专家
- **Meta-Llama-3-8B / 70B**、**Llama-2-7b / 70b**、**CodeLlama-34b**、**InternLM-20B**、**Qwen-72B**

---

## 快速链接

- [GitHub 仓库](https://github.com/aliyun/SimAI)
- [NSDI'25 论文 (PDF)](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)
- [演示文稿](../../docs/SimAI_Intro_Online.pdf)
- [技术报告 (1.6)](../SimAI_1.6_Tech_Report.md)
- [贡献指南](../../CONTRIBUTING.md)
