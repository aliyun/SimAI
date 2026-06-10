# 蓝区架构定位图

**定位**：基于开源仿真项目 SimAI，打造全面适配 OXC-HCCL 算法、OXC 组网拓扑、端口分配算法的 OXC 仿真平台，同时具备面向用户的可视化前端页面。

---

## 系统架构图

```mermaid
flowchart LR
    USER([用户])

    subgraph BLUE["蓝区 · OXC 仿真平台"]
        direction TB
        FE["可视化前端<br/>React · 9 页面 · 7.1K 行"]
        BE["Dashboard 后端<br/>Flask · 5.3K 行"]
        ADAPT["OXC 适配层<br/>C++ · 8.8K 行"]
        CORE[("SimAI 仿真内核<br/>开源基座")]

        FE --> BE
        BE --> ADAPT
        ADAPT --> CORE
    end

    subgraph YELLOW["黄区 · 外部系统"]
        direction TB
        HCCL["OXC-HCCL 算法"]
        PORT["端口分配 Solver<br/>:9000"]
        TOPO["OXC 组网实况<br/>lld.json · ranktable"]
    end

    USER --> FE
    ADAPT -. "契约级对接 · Mock 驱动<br/>（蓝区网络隔离，不连真实服务）" .-> HCCL
    BE -. "契约级对接 · Mock 驱动" .-> PORT
    TOPO -. "lld.json 样本文件上传" .-> FE

    classDef blueNode fill:#DCEEFF,stroke:#1B5E9E,stroke-width:2px,color:#0D3F6E
    classDef yellowNode fill:#FFF2CC,stroke:#D6A000,stroke-width:2px,color:#5A3F00
    classDef coreNode fill:#F0F0F0,stroke:#888,stroke-width:1px,stroke-dasharray:6 4,color:#333
    classDef userNode fill:#FFFFFF,stroke:#444,stroke-width:1px,color:#222

    class FE,BE,ADAPT blueNode
    class CORE coreNode
    class HCCL,PORT,TOPO yellowNode
    class USER userNode

    style BLUE fill:#F5FAFF,stroke:#1B5E9E,stroke-width:2px,color:#0D3F6E
    style YELLOW fill:#FFFBEF,stroke:#D6A000,stroke-width:2px,color:#5A3F00
```

---

## 分层职责

| 层 | 归属 | 组件 | 行数 | 职责 |
|---|---|---|---|---|
| 1. 前端 | 蓝区自研 | React + TypeScript | 7.1K | 用户工作流（9 页面） |
| 2. API 网关 | 蓝区自研 | Flask | 5.3K | workspace / process / edg / auth / monitor |
| 3. OXC 适配层 | 蓝区自研 | C++ | 8.8K | OXC-HCCL 集成、拓扑建模、端口分配消费端 |
| 4. 仿真内核 | 开源基座 | C++（SimAI 上游） | — | analytical / ns3 / phy 三模式 |

## 对外集成接口

| 接口 | 方向 | 蓝区实现 | 说明 |
|---|---|---|---|
| OXC-HCCL | 蓝区 →（契约）→ 黄区 | HTTP 客户端 + Mock | 蓝区按契约实现客户端，开发期走 `AS_OXC_MOCK=1`；真实联调在集成环境 |
| 端口分配（EDG） | 蓝区 →（契约）→ 黄区 | `edg_client.py` + Mock | 蓝区按契约实现客户端，开发期走 `AS_EDG_MOCK=1`；真实联调在集成环境 |
| OXC 组网实况 | 用户 → 蓝区 | 文件上传 | 用户在 TopologyPage 上传 `lld.json` 样本文件，蓝区按文件规约解析 |
