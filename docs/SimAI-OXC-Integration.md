# SimAI-OXC 集成技术文档

## 1. 架构概览

OXC（光交叉连接）集成允许 SimAI 通过外部 OXC 优化服务生成跨机架集合通信的最优流调度方案。当前仅支持 AllReduce。

### 组件关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimAI 仿真器 (NS3)                            │
│                                                                   │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │AstraSimNetwork│───→│  Sys / Workload│───→│  MockNcclGroup   │  │
│  │  (main 入口)  │    │  (层调度)      │    │  (集合通信生成)   │  │
│  └──────┬───────┘    └───────────────┘    └────────┬─────────┘  │
│         │                                           │             │
│         │ 初始化                          是否跨机架？│             │
│         ▼                                    ┌──────┴──────┐     │
│  ┌──────────────┐                    否      │             │ 是   │
│  │ OxcAdapter   │◄──────────────────────     │             │     │
│  │ (全局单例)    │                      │     ▼             ▼     │
│  └──────┬───────┘                      │  原生 Ring    OXC 路径  │
│         │                              │  算法          │         │
│         │ HTTP POST                    │               │         │
│         ▼                              │               ▼         │
│  ┌──────────────┐              ┌───────┴──────────────────┐     │
│  │ OXC Java 后端│              │    NcclTreeFlowModel     │     │
│  │ (外部服务)    │              │    (流执行引擎)           │     │
│  └──────────────┘              └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 核心源文件

| 文件 | 作用 |
|------|------|
| `OxcIntegration.h/.cc` | OXC 适配器：配置、API 调用、流生成 |
| `OxcTypes.h` | 数据结构定义（RankTable、请求/响应格式） |
| `MockNcclGroup.cc` | 决策入口：OXC 还是原生算法 |
| `MockNcclChannel.h` | `SingleFlow` 结构体定义 |
| `NcclTreeFlowModel.cc` | 流执行：发送/接收调度 |
| `AstraSimNetwork.cc` | 程序入口，OXC 初始化 |

---

## 2. 初始化时序

```
程序启动 (main)
    │
    ├── 1. main1(network_topo, network_conf)     // NS3 网络拓扑初始化
    │
    ├── 2. OxcIntegration::initializeGlobalOxcAdapter()
    │       │
    │       ├── OxcConfig::fromEnvironment()      // 读取环境变量
    │       │     AS_OXC_ENABLE, AS_OXC_URL, AS_OXC_ALGO,
    │       │     AS_OXC_GPUS_PER_SERVER, AS_OXC_RANKTABLE, ...
    │       │
    │       ├── curl_global_init()                // libcurl 初始化（once）
    │       │
    │       ├── loadRankRackMapFromFile()          // 可选：加载 rank→rack 映射
    │       │
    │       └── loadRankTableFromFile()            // 可选：加载 RankTable JSON
    │             └── 自动生成 rank→rack 映射（若未手动指定）
    │
    ├── 3. new AstraSim::Sys(...)                 // 创建仿真系统（×N 个 GPU）
    │
    └── 4. workload->fire()                       // 开始仿真
            └── genAllReduceFlowModels()          // 首次触发 OXC 流生成
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AS_OXC_ENABLE` | `0` | 启用 OXC（设为 `1` 或 `true`） |
| `AS_OXC_URL` | `http://localhost:8080` | OXC 后端服务地址 |
| `AS_OXC_ALGO` | `ALGO_OXC_RING` | 请求的算法名称 |
| `AS_OXC_GPUS_PER_SERVER` | `8` | 每台服务器的 GPU 数量 |
| `AS_OXC_RANKTABLE` | 空 | RankTable JSON 文件路径 |
| `AS_OXC_RANK_RACK_MAP` | 空 | rank→rack 映射 JSON 文件路径 |
| `AS_OXC_HTTP_TIMEOUT` | `30` | HTTP 请求超时（秒） |
| `AS_OXC_CONNECT_TIMEOUT` | `5` | HTTP 连接超时（秒） |

---

## 3. 跨机架检测逻辑

```
shouldUseOxc(group_ranks, comm_type)
    │
    ├── 未初始化或未启用？ → return false
    ├── 不是 AllReduce？   → return false
    └── isCrossRack(group_ranks)
          │
          ├── 对每个 rank：
          │     ├── rank_rack_map 中有映射？→ 使用映射的 rack_id
          │     └── 没有？→ 默认: "rack_" + rank / gpus_per_server
          │
          └── 统计不同的 rack 数量
                ├── > 1 个 rack → return true（使用 OXC）
                └── = 1 个 rack → return false（使用原生算法）
```

**实际运行示例（16 GPU, 2 racks）：**

```
ranks=[0,1,2,3,4,5,6,7]  → superpod1_0 → 1 个 rack → 原生 ring
ranks=[8,9,10,11,12,13,14,15] → superpod1_1 → 1 个 rack → 原生 ring
ranks=[7,15]              → superpod1_0 + superpod1_1 → 2 个 rack → OXC
```

---

## 4. SingleFlow 字段语义

```cpp
struct SingleFlow {
    int flow_id;              // 全局唯一流 ID
    int src;                  // 发送端 rank
    int dest;                 // 接收端 rank
    uint64_t flow_size;       // 数据量（字节）
    vector<int> prev;         // 在当前 step 中，谁发数据给 src
    vector<int> parent_flow_id; // 前置依赖流 ID（必须先完成）
    vector<int> child_flow_id;  // 后续依赖流 ID（等待本流完成）
    int channel_id;           // 通道 ID（OXC 固定为 0）
    int chunk_id;             // 时间步（OXC API 的 step）
    int chunk_count;          // 总步数
    string conn_type;         // 连接类型（OXC 固定为 "RING"）
};
```

### 字段详解

**`prev` — 接收源 rank**

NcclTreeFlowModel 用 `prev[0]` 作为 `sim_recv()` 的源 rank。含义是"在当前 step 中，谁把数据发给了 src，使得 src 才有数据可以发给 dest"。

```
Step 0: A→B, C→A     ← 流 A→B 的 prev = [C]（C 发给 A）
Step 1: B→C, A→B     ← 流 B→C 的 prev = [A]（A 发给 B）
```

**`parent_flow_id` / `child_flow_id` — 依赖链**

表示数据依赖关系：step N 的流必须等 step N-1 中发送到 src 的流完成。

```
Flow 0 (step=0, A→B)
  parent_flow_id = []     ← 第一步，无前置依赖
  child_flow_id = [2]     ← Flow 2 依赖 Flow 0

Flow 2 (step=1, B→C)
  parent_flow_id = [0]    ← 依赖 Flow 0 完成
  child_flow_id = []      ← 最后一步
```

**`conn_type` — 为什么必须是 "RING"**

NcclTreeFlowModel 根据 `conn_type` 选择不同的 `tag_id` 计算公式：

```cpp
if (parent_flow_id.size() == 0 || conn_type == "RING") {
    tag_id = layer_num * chunk_count * m_channels
           + chunk_count * channel_id
           + chunk_id;
} else {
    // Tree 类型：tag_id 有 +1 偏移
}
```

如果 OXC 流的 `conn_type` 不是 `"RING"`，sender 和 receiver 的 tag_id 不匹配，数据永远无法送达。

**Map Key: `pair(channel_id, flow_id)`**

FlowModels 的 key 必须是 `pair(channel_id, flow_id)`。NcclTreeFlowModel 中所有查找都按此格式：

```cpp
_flow_models[std::make_pair(channel_id, flow_id)]
```

---

## 5. OXC 流生成详解

### 5.1 与原生 Ring 算法对比

| 维度 | 原生 Ring | OXC |
|------|----------|-----|
| 入口 | `genAllReduceRingFlowModels()` | `generateAllReduceFlows()` |
| chunk_count | `2*(nRanks-1)` 固定公式 | OXC API 返回的 `max(step)+1` |
| 通道数 | 多通道（`ringchannels`） | 单通道（`channel_id=0`） |
| prev 来源 | ring 拓扑中的前驱 rank | `step_dst_to_src[step][src]` |
| conn_type | `"RING"` 或 `"PXN_INIT"` | 始终 `"RING"` |
| PXN 支持 | 有（跨节点优化） | 无 |
| 数据分片 | `data_size / nranks / channels` | OXC API 返回的 `datasize` |

### 5.2 OXC 生成流程（三遍扫描）

**第 0 遍：准备阶段**

```
1. 调用 OXC API → 获得 entries: [{src, dst, step, datasize}, ...]
2. 计算 chunk_count = max(step) + 1
3. 构建辅助映射：
   step_dst_to_src[step][dst] = src    // 用于计算 prev
   step_src_to_flow_id[(step, src)] = fid  // 用于计算依赖
```

**第 1 遍：创建 SingleFlow**

```
对每个 entry:
  sf.flow_id = base_flow_id + i
  sf.src = entry.src_rank
  sf.dest = entry.dst_rank
  sf.flow_size = entry.datasize
  sf.channel_id = 0
  sf.chunk_id = entry.step
  sf.chunk_count = chunk_count
  sf.conn_type = "RING"
  sf.prev = [step_dst_to_src[step][src]]   // 谁发给 src

  result[pair(0, flow_id)] = sf
```

**第 2 遍：设置依赖链**

```
对每个 sf:
  if step > 0:
    // 找 step-1 中发送到 sf.src 的流
    sender = step_dst_to_src[step-1][sf.src]
    parent_fid = step_src_to_flow_id[(step-1, sender)]
    sf.parent_flow_id = [parent_fid]

  if step < chunk_count - 1:
    // 找 step+1 中从 sf.dest 发出的流
    child_fid = step_src_to_flow_id[(step+1, sf.dest)]
    sf.child_flow_id = [child_fid]
```

### 5.3 防御性检查

| 检查 | 触发条件 | 日志级别 |
|------|----------|---------|
| 重复 dst_rank | 同一 step 中两个流发往同一 dst | WARNING |
| 重复 src_rank | 同一 step 中同一 src 发两个流 | WARNING |
| 空 prev | src 在当前 step 没有接收源 | ERROR |

---

## 6. 流执行（NcclTreeFlowModel）

### 6.1 流的生命周期

```
                        ┌──────────────┐
                        │   StreamInit │
                        └──────┬───────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                      │
    parent_flow_id 为空                    parent_flow_id 非空
    且 chunk_id == 0                      （等待前置完成）
            │                                      │
            ▼                                      ▼
    ┌───────────────┐                      init_recv_ready()
    │insert_packets │                      注册接收事件
    │  用 prev[0]    │                             │
    │  发起 sim_recv │                      前置流完成
    │  + sim_send   │                      indegree → 0
    └───────┬───────┘                             │
            │                                      ▼
            │                              ┌───────────────┐
            │                              │    ready()     │
            │                              │  发起 sim_recv │
            │                              │  + sim_send   │
            │                              └───────┬───────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           │
                    PacketSentFinished
                           │
                    ┌──────┴──────┐
                    │   reduce()  │
                    │ 递减子流    │
                    │ indegree    │
                    └──────┬──────┘
                           │
                    indegree == 0?
                    ├── 是 → ready(子流)
                    └── 否 → 等待
```

### 6.2 关键代码路径

**初始启动（step 0 的流）：**

```cpp
// NcclTreeFlowModel.cc:243-254
if (parent_list.size() == 0 && chunk_id == 0) {
    insert_packets(channel_id, flow_id);
    // → 使用 prev[0] 发起接收
    // → 发起发送到 dest
}
```

**依赖满足后（step > 0 的流）：**

```cpp
// NcclTreeFlowModel.cc:186-189
if (--indegree_mapping[next_flow_id] == 0) {
    ready(channel_id, next_flow_id);
    // → 对 prev 中的每个 rank 发起 sim_recv
    // → 发起 sim_send 到 dest
}
```

---

## 7. OXC API 交互

### 7.1 请求格式

```
POST /api/oxc/allreduce
Content-Type: application/json
```

```json
{
  "ranktable": {
    "version": "2.0",
    "status": "completed",
    "rank_count": 16,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,
        "local_id": 0,
        "level_list": [{
          "net_layer": 0,
          "net_instance_id": "superpod1_0",
          "net_type": "TOPO_FILE_DESC",
          "rank_addr_list": [{
            "addr_type": "EID",
            "addr": "000000000000002000100000df001001",
            "ports": ["0/0"],
            "plane_id": "plane0"
          }]
        }]
      }
    ]
  },
  "dpCommDomain": [[7, 15]],
  "commDomainVolume": 134217728,
  "rankIdRackIdMap": {
    "7": "superpod1_0",
    "15": "superpod1_1"
  },
  "algName": "ALGO_OXC_RING"
}
```

### 7.2 响应格式

```json
[[7, 15, 0, 67108864],
 [15, 7, 0, 67108864],
 [15, 7, 1, 67108864],
 [7, 15, 1, 67108864]]
```

每个元素：`[src_rank, dst_rank, step, datasize]`

### 7.3 错误处理与降级

| 故障场景 | 行为 |
|----------|------|
| OXC 服务不可达 | 打印警告，降级为原生 ring 算法 |
| HTTP 超时（30s） | 打印错误，降级为原生 ring 算法 |
| 响应格式错误 | 打印错误，降级为原生 ring 算法 |
| 响应包含 "error" | 打印错误，降级为原生 ring 算法 |
| RankTable 文件加载失败 | 禁用 OXC，所有通信走原生算法 |
| 空流列表 | 打印警告，降级为原生 ring 算法 |

---

## 8. 正确性验证

### 8.1 对比测试结果（16 GPU, 2 racks, microAllReduce_dp）

| 指标 | OXC 启用 | OXC 禁用 | 差值 |
|------|---------|---------|------|
| 总完成时间 (cycles) | 5,374,224 | 5,374,216 | +8 |
| embedding fwd comm | 85,068 | 85,068 | 0 |
| transformer fwd comm | 85,068 | 85,068 | 0 |
| transformer wg comm | 2,724,589 | 2,724,583 | +6 |
| embedding wg comm | 4,092,079 | 4,092,071 | +8 |
| 每节点数据量 | 234,881,332 | 234,881,024 | +308 |
| 流完成率 | 100% | 100% | — |

**分析：** 小规模（2 跨 rack 节点）下 OXC ring 和原生 ring 等价，差异来自浮点精度和分片粒度。

### 8.2 正确性要点

- 机架内通信（ranks [0..7] 和 [8..15]）正确走原生 ring 路径
- 跨机架通信（ranks [7,15]）正确走 OXC 路径
- 所有 16 个节点完成仿真，无段错误
- 没有 ERROR/WARNING 日志输出
- 100% 的 stream 完成

---

## 9. 已知限制

| 限制 | 影响 | 优先级 |
|------|------|--------|
| 仅支持 AllReduce | AllGather/ReduceScatter/AllToAll 无法使用 OXC | 中 |
| 单通道（channel_id=0） | 无法利用多通道并行，降低吞吐 | 高 |
| 每次 AllReduce 都调 HTTP API | 增加仿真延迟，无响应缓存 | 中 |
| 不支持 PXN 优化 | 跨节点但同服务器内的优化缺失 | 低 |
| 手写 JSON 解析器 | 不支持转义/Unicode，易出错 | 低 |
| step 必须从 0 开始连续 | 如 OXC 返回非连续 step 会导致依赖缺失 | 低 |

---

## 10. 配置示例

### 基本使用

```bash
AS_OXC_ENABLE=1 \
AS_OXC_URL=http://localhost:8080 \
AS_OXC_ALGO=ALGO_OXC_RING \
AS_OXC_GPUS_PER_SERVER=8 \
AS_OXC_RANKTABLE=/path/to/ranktable.json \
AS_NVLS_ENABLE=1 \
AS_SEND_LAT=3 \
./bin/SimAI_simulator_oxc -t 8 -w workload.txt -n topology -c config.conf
```

### RankTable JSON 格式

```json
{
  "version": "2.0",
  "status": "completed",
  "rank_count": 16,
  "rank_list": [
    {
      "rank_id": 0,
      "device_id": 0,
      "local_id": 0,
      "level_list": [{
        "net_layer": 0,
        "net_instance_id": "superpod1_0",
        "net_type": "TOPO_FILE_DESC",
        "net_attr": "",
        "rank_addr_list": [{
          "addr_type": "EID",
          "addr": "000000000000002000100000df001001",
          "ports": ["0/0"],
          "plane_id": "plane0"
        }]
      }]
    }
  ]
}
```

### Rank-Rack 映射 JSON 格式

```json
{
  "0": "rack_0",
  "1": "rack_0",
  "8": "rack_1",
  "9": "rack_1"
}
```

如不提供此文件，将根据 `rank / gpus_per_server` 自动生成默认映射。
