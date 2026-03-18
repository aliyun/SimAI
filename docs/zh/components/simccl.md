# SimCCL — 集合通信库

**仓库**: [aliyun/SimCCL](https://github.com/aliyun/SimCCL) | **语言**: Python/C++

SimCCL 将集合通信操作转换为点对点通信，是工作负载层与仿真引擎之间的关键桥梁。

---

## 概述

在分布式 LLM 训练中，集合通信操作（AllReduce、AllGather、ReduceScatter、AlltoAll 等）是基础构建块。SimCCL 将这些高层集合操作分解为点对点通信序列，以便网络后端精确仿真。

---

## 在 SimAI 中的角色

SimCCL 位于 AICB（工作负载生成）和 astra-sim-alibabacloud（仿真引擎）之间：

```
AICB 生成包含集合操作的工作负载
    |
    v
SimCCL 分解集合操作 → 点对点通信
    |
    v
astra-sim 将 P2P 流量发送到 NS-3 或物理网络
```

SimCCL 在以下场景中**必需**：
- **SimAI-Simulation** — 全栈 NS-3 仿真
- **SimAI-Physical** — 物理 RDMA 流量生成
- **推理仿真** — 使用 SimAI Simulation 后端时

SimCCL 在以下场景中**不需要**：
- **SimAI-Analytical** — 直接使用 busbw 估算

---

## 版本

### 基础版（mocknccl）

基础实现目前位于 [astra-sim-alibabacloud](https://github.com/aliyun/SimAI/tree/master/astra-sim-alibabacloud) 仓库中。文件以 `mocknccl` 为前缀，提供基本的集合→P2P 转换功能。

### 完整版

具备高级集合通信算法的完整 SimCCL 库可在 [SimCCL 仓库](https://github.com/aliyun/SimCCL) 获取。

---

## 支持的集合操作

| 操作 | 说明 |
|-----------|-------------|
| AllReduce | 跨所有 Rank 进行归约，结果在所有 Rank 上可用 |
| AllGather | 从所有 Rank 收集数据，结果在所有 Rank 上可用 |
| ReduceScatter | 跨所有 Rank 进行归约并分发 |
| AlltoAll | 全对全个性化通信 |
| Broadcast | 从一个 Rank 广播到所有其他 Rank |

---

## 与 astra-sim 的集成

SimCCL 通过 `MockNcclGroup` 和 `MockNcclChannel` 接口与 astra-sim-alibabacloud 集成：

- **MockNcclGroup**：管理参与集合操作的一组 Rank
- **MockNcclChannel**：处理集合操作中特定 Channel 的实际点对点数据传输

分解过程考虑：
- 网络拓扑（Ring、Tree 等）
- 参与的 Rank 数量
- 消息大小
- 可用通信通道

---

## 相关文档

- [组件概述](index.md) — SimAI 组件架构
- [astra-sim 组件](astra_sim.md) — 消费 SimCCL 输出的仿真引擎
- [NS-3 组件](ns3.md) — P2P 仿真的网络后端
