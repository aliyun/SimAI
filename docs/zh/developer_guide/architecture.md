# 系统架构

本文档描述 SimAI 的模块化架构、组件交互以及训练和推理仿真的数据流。

---

## 项目结构

```
SimAI/
├── aicb/                        # AI 计算基准——工作负载生成（Python）
│   ├── workload_generator/      #   训练/推理工作负载生成器
│   └── aicb.py                  #   主入口
├── astra-sim-alibabacloud/      # 仿真引擎——核心仿真器（C++）
│   ├── astra-sim/               #   扩展自 astra-sim 1.0
│   └── build.sh                 #   编译脚本
├── ns-3-alibabacloud/           # NS-3 网络仿真后端（C++）
├── vidur-alibabacloud/          # LLM 推理仿真（Python）
│   ├── vidur/                   #   核心仿真框架
│   └── setup.py                 #   Python 包配置
├── SimCCL/                      # 集合通信转换
├── docs/                        # 文档和教程
├── example/                     # 示例工作负载和配置
├── scripts/                     # 编译和工具脚本
├── results/                     # 仿真输出目录
├── bin/                         # 编译二进制输出
└── Dockerfile                   # Docker 容器定义
```

---

## 组件架构

```
        |--- AICB                        （工作负载生成 & 计算性能分析）
SimAI --|--- SimCCL                      （集合通信算法分析）
        |--- astra-sim-alibabacloud      （仿真引擎：Analytical / Simulation / Physical）
        |--- ns-3-alibabacloud           （NS-3 网络后端）
        |--- vidur-alibabacloud          （多请求推理调度 & 显存管理）
```

![SimAI 架构](../../images/SimAI_Arc.png)

### 组件职责

| 组件 | 角色 | 语言 |
|-----------|------|----------|
| **AICB** | 生成训练/推理工作负载、采集计算内核性能、运行物理基准测试 | Python |
| **SimCCL** | 将集合通信操作（AllReduce、AllGather 等）转换为点对点通信集合 | Python |
| **astra-sim-alibabacloud** | 支持 3 种模式的核心仿真引擎；管理计算/内存/网络 API | C++ |
| **ns-3-alibabacloud** | 带 RDMA、数据中心拓扑和 CC 算法的包级网络仿真 | C++ |
| **vidur-alibabacloud** | 支持 PD 分离和 GPU 显存管理的多请求推理调度 | Python |

---

## 三种运行模式

### SimAI-Analytical

```
AICB (workload.txt) → astra-sim (analytical) → busbw 估算 → CSV 结果
```

- **适用场景**：快速性能分析、并行参数扫描
- **组件**：AICB + astra-sim-alibabacloud（分析模式）
- **网络模型**：总线带宽（busbw）抽象

### SimAI-Simulation

```
AICB (workload.txt) → SimCCL (集合→P2P) → astra-sim (simulation) → NS-3 → 详细 Trace
```

- **适用场景**：全栈网络研究、CC 算法评估
- **组件**：AICB + SimCCL + astra-sim-alibabacloud (simulation) + ns-3-alibabacloud
- **网络模型**：包级 NS-3 仿真

### SimAI-Physical

```
AICB (workload.txt) → SimCCL (集合→P2P) → astra-sim (physical) → 真实 NIC 上的 RDMA 流量
```

- **适用场景**：NIC 行为研究、物理流量分析
- **组件**：AICB + SimCCL + astra-sim-alibabacloud（物理模式）
- **网络模型**：通过 MPI 的真实 RDMA 流量

---

## 推理仿真数据流

```
请求生成器
    |  生成合成/真实 Trace 请求
    v
全局调度器
    |  将请求分发到 Prefill / Decode 副本
    v
副本调度器
    |  批次组装和调度
    v
显存管理（MemoryPlanner + Replica）
    |  KV Cache 分配和容量检查
    v
执行时间预测器
    |  AICB / SimAI Simulation / SimAI Analytical / Vidur
    v
指标存储
    |  TTFT、TBT、E2E、通信/计算开销
    v
输出（request_metrics.csv, chrome_trace.json, plots/）
```

### 推理关键组件

| 组件 | 文件 | 说明 |
|-----------|------|-------------|
| 请求生成器 | `vidur/request_generator/` | 生成合成或基于 Trace 的请求 |
| 全局调度器 | `vidur/scheduler/global_scheduler/` | 跨副本分发请求（`lor`、`round_robin`、`split_wise`） |
| 副本调度器 | `vidur/scheduler/replica_scheduler/` | 副本内批次调度（`sarathi`、`split_wise`） |
| MemoryPlanner | `vidur/scheduler/utils/memory_planner.py` | GPU 显存预算计算 |
| ParamCounter | `vidur/utils/param_counter.py` | 模型参数计数（MLA/MHA/GQA/线性/MoE） |
| 执行预测器 | `vidur/execution_time_predictor/` | 通过多种后端估算执行时间 |
| 指标存储 | `vidur/metrics/` | 采集并导出 23 项仿真指标 |

---

## 子模块结构

SimAI 使用 Git submodule 管理核心组件：

| 子模块 | 仓库 | 分支 |
|-----------|------------|--------|
| `aicb` | [aliyun/aicb](https://github.com/aliyun/aicb) | master |
| `SimCCL` | [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | master |
| `ns-3-alibabacloud` | [aliyun/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud) | master / dev/qp |
| `astra-sim-alibabacloud` | 项目内 | — |
| `vidur-alibabacloud` | 项目内 | — |

**关键规则：**
1. 子模块拥有独立的 Git 历史
2. 父仓库仅追踪每个子模块的 commit hash
3. 克隆后务必初始化：`git submodule update --init --recursive`

---

## 构建系统

### 编译脚本

```bash
# 分析模式（快速，基于 busbw）
bash scripts/build.sh -c analytical

# NS-3 仿真模式（全栈）
bash scripts/build.sh -c ns3

# 物理模式（Beta，RDMA）
bash scripts/build.sh -c phy
```

### 编译产物

| 模式 | 二进制 | 位置 |
|------|--------|----------|
| Analytical | `SimAI_analytical` | `bin/` |
| Simulation | `SimAI_simulator` | `bin/` |
| Physical | `SimAI_physical` | `bin/` |

---

## 相关文档

- [组件概述](../components/index.md) — 各组件详细文档
- [贡献指南](contributing.md) — 如何贡献代码
- [配置文件参考](../technical_reference/configuration.md) — 配置文件和参数
