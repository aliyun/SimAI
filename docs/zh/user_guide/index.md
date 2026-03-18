# 用户指南

本章节提供 SimAI 各运行模式的详细使用说明。

## 目录

| 页面 | 说明 |
|------|------|
| [SimAI-Analytical](simai_analytical.md) | 使用总线带宽的快速分析仿真 |
| [SimAI-Simulation](simai_simulation.md) | 使用 NS-3 网络后端的全栈仿真与拓扑配置 |
| [SimAI-Physical](simai_physical.md) | 在真实集群上生成物理 RDMA 流量 |
| [推理仿真](inference_simulation.md) | 支持 PD 分离的多请求 LLM 推理仿真 |
| [工作负载生成](workload_generation.md) | 使用 AICB 生成训练和推理工作负载 |
| [支持的模型](supported_models.md) | 支持的模型完整列表及配置 |
| [结果分析](result_analysis.md) | 仿真结果分析与可视化 |

## 工作流程概览

典型的 SimAI 工作流程包含三个步骤：

1. 使用 [AICB](workload_generation.md) **生成工作负载** — 定义计算和通信模式
2. 使用三种模式之一（Analytical、Simulation 或 Physical）**运行仿真**
3. 使用内置工具或自定义脚本**分析结果**

推理仿真的工作流程使用 Vidur 进行请求调度和内存管理，AICB 或 SimAI 作为执行时间预测后端。
