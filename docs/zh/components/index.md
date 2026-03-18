# 组件概述

SimAI 是一个模块化项目，由 5 个核心组件组成。各组件可独立使用，也可组合使用以实现不同的仿真场景。

---

## 架构

```
        |--- AICB                        （工作负载生成 & 计算性能分析）
SimAI --|--- SimCCL                      （集合通信算法分析）
        |--- astra-sim-alibabacloud      （仿真引擎：Analytical / Simulation / Physical）
        |--- ns-3-alibabacloud           （NS-3 网络后端）
        |--- vidur-alibabacloud          （多请求推理调度 & 显存管理）
```

---

## 组件摘要

| 组件 | 语言 | 仓库 | 说明 |
|------|------|------|------|
| [AICB](aicb.md) | Python | [aliyun/aicb](https://github.com/aliyun/aicb) | AI 通信基准测试——训练和推理工作负载生成 |
| [SimCCL](simccl.md) | Python | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | 集合通信到点对点通信转换 |
| [astra-sim-alibabacloud](astra_sim.md) | C++ | In-tree | 支持 3 种模式的核心仿真引擎 |
| [ns-3-alibabacloud](ns3.md) | C++ | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | 带 RDMA/数据中心扩展的 NS-3 网络仿真后端 |
| [vidur-alibabacloud](vidur.md) | Python | In-tree | 支持 PD 分离和请求调度的 LLM 推理仿真 |

---

## 场景与组件组合

| 场景 | AICB | SimCCL | astra-sim | ns-3 | vidur |
|------|------|--------|-----------|------|-------|
| AICB 测试套件（物理 GPU） | 必需 | - | - | - | - |
| 工作负载生成 | 必需 | - | - | - | - |
| 集合通信分析 | - | 必需 | - | - | - |
| SimAI-Analytical | 必需 | - | 必需（analytical） | - | - |
| SimAI-Simulation | 必需 | 必需 | 必需（simulation） | 必需 | - |
| SimAI-Physical | 必需 | 必需 | 必需（physical） | - | - |
| 推理仿真 | 必需 | 必需 | 必需 | 可选 | 必需 |

---

## 数据流

```
AICB（工作负载生成）
    |
    |-- 训练工作负载 (.txt) --> astra-sim-alibabacloud
    |-- 推理工作负载 -------> vidur-alibabacloud
    |
SimCCL（集合 → P2P）
    |
    |--> astra-sim-alibabacloud（Simulation/Physical 模式）
    |
astra-sim-alibabacloud（仿真引擎）
    |
    |-- Analytical 模式：busbw 估算
    |-- Simulation 模式：NS-3 后端
    |-- Physical 模式：RDMA 流量注入
    |
ns-3-alibabacloud（网络后端）
    |
    |--> 细粒度网络仿真结果
    |
vidur-alibabacloud（推理调度）
    |
    |--> request_metrics.csv, chrome_trace.json, plots/
```

---

## 组件详细文档

- **[AICB](aicb.md)** — 工作负载生成、基准测试套件、AIOB 计算分析
- **[SimCCL](simccl.md)** — 集合通信分解
- **[astra-sim-alibabacloud](astra_sim.md)** — 核心仿真引擎、配置、拓扑生成
- **[ns-3-alibabacloud](ns3.md)** — RDMA 网络仿真、CC 算法、分析工具
- **[vidur-alibabacloud](vidur.md)** — 推理仿真、PD 分离、GPU 显存管理
