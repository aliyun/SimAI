# send_lat 深入分析：send_latency_table vs AS_SEND_LAT

> [English Version](../../configuration/send-lat-analysis.md)

## 概述

SimAI 使用 `send_lat`（发送延迟）值来延迟 ns3 仿真中的数据包发送。这模拟了发起 NCCL 集合通信操作的软件开销。有两种机制控制此值。

---

## 1. send_latency_table（表查询）

### 位置
`astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h:134-152`

### 工作原理

两个 7x3 表，按 `[algorithm][protocol]` 索引：
- `send_lat_table_nvlink[7][3]` — 用于同节点（NVLINK）通信
- `send_lat_table_net[7][3]` — 用于跨节点（NET/IB）通信

算法索引：0=Tree, 1=Ring, 2=CollNetDirect, 3=CollNetChain, 4=NVLS, 5=NVLS_TREE, 6=PAT
协议索引：0=LL, 1=LL128, 2=Simple

### 表值（单位：纳秒）

#### NVLINK 表（节点内）
| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 7400 | 15250 | 12400 |
| Ring | 7200 | 15900 | 11800 |
| CollNetDirect | 0 | 0 | 3700 |
| CollNetChain | 0 | 0 | 2800 |
| NVLS | 0 | 0 | 25000 |
| NVLS_TREE | 0 | 0 | 25000 |
| PAT | 0 | 0 | 12000 |

#### NET 表（节点间）
| 算法 | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 11800 | 22500 | 22400 |
| Ring | 9300 | 18000 | 22400 |
| CollNetDirect | 0 | 0 | 31000 |
| CollNetChain | 0 | 0 | 30000 |
| NVLS | 0 | 0 | 18000 |
| NVLS_TREE | 0 | 0 | 20900 |
| PAT | 0 | 0 | 22000 |

值 `0` 表示不支持的组合 → 回退到默认值 6000 ns。

### 链路类型检测（entry.h:159-162）

```cpp
int gpus_per_node = request->flowTag.gpus_per_node;
bool same_node = (src / gpus_per_node) == (dst / gpus_per_node);
const int (*table)[3] = same_node ? send_lat_table_nvlink : send_lat_table_net;
```

### 调用栈

```
NcclFlowModel::insert_packets()
  → front_end_sim_send()
    → entry.h: SendFlow()
      → 读取 flowTag.algorithm, protocol, gpus_per_node
      → 判断 same_node → 选择 nvlink 或 net 表
      → table[algo][proto] → send_lat (ns)
      → AS_SEND_LAT 覆盖检查（最高优先级）
      → send_lat *= 1000 → 将 ns 转换为 ps（ns3 Time 单位）
      → appCon.Start(Time(send_lat))
```

---

## 2. AS_SEND_LAT（环境变量覆盖）

### 位置
`entry.h:168-177`

### 工作原理

```cpp
const char* send_lat_env = std::getenv("AS_SEND_LAT");
if (send_lat_env) {
    send_lat = std::stoi(send_lat_env);  // 覆盖表查询
}
send_lat *= 1000; // ns → ps
```

### 作用域

**AS_SEND_LAT 不在 SimCCL 中**。它作用于 ns3 网络前端层。

| 属性 | send_latency_table | AS_SEND_LAT |
|---|---|---|
| 位置 | entry.h（ns3 前端） | entry.h（ns3 前端） |
| 粒度 | 按 (algorithm, protocol, link_type) | 全局单一值 |
| 优先级 | 较低 | **最高**（覆盖表） |
| 用例 | 生产环境仿真 | A/B 实验、快速校准 |
| 单位 | 纳秒 | 纳秒 |

### send_latency_table 能否完全替代 AS_SEND_LAT？

**不能。** 它们服务于不同目的：

- `send_latency_table`：细粒度的按 (algo, proto, link) 延迟。用于精确的生产仿真。
- `AS_SEND_LAT`：快速全局覆盖，用于 A/B 测试。设置 `AS_SEND_LAT=6` 使所有通信使用 6000 ps，无论算法/协议。

**建议**：生产环境使用表，AS_SEND_LAT 仅用于调试/对比实验。

---

## 3. 与 NCCL 2.30 延迟模型的比较

### NCCL 源码
`nccl-2.30/src/graph/tuning.cc:150-174`（基础/硬件延迟表），`L380-435`（延迟计算）

### 关键差异

SimAI 的 `send_lat_table` 是 NCCL 延迟模型的 **分桶近似**：

| 方面 | SimAI entry.h | NCCL tuning.cc |
|---|---|---|
| 模型类型 | 固定 3D 表查询 | 每算法动态计算 |
| Ring 延迟 | 固定值（如 LL-NET 9300 ns） | `baseLat + nsteps * (intraLat + netOverhead)` |
| PAT 延迟 | 固定 22000 ns（Simple-NET） | `log2(nNodes) * (interLat/3.5) + nRanks * 2.8` |
| Tree 延迟 | 固定值 | `baseLat + 2*log2(nNodes)*interLat` |
| 依赖拓扑 | 仅链路类型（同/跨节点） | nNodes, nRanks, nsteps, netOverhead, cpuArch |
| 硬件特定 | 否（所有 GPU 使用同一表） | 按 CPU 厂商调整 `netOverhead` |

**含义**：SimAI 的表提供合理但近似的 send_lat。对于 PAT 具体而言，表使用固定的 22us（NET-Simple），而 NCCL 计算 `log2(N)*(interLat/3.5)+N*2.8`，取决于实际节点数和节点间延迟。

---

## 4. 如何实验

### 前置条件

```bash
cd SimAI/
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

### 使用 send_lat_table 运行（默认）

```bash
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# 查看结果
head -3 ncclFlowModel_EndToEnd.csv
```

### 使用 AS_SEND_LAT 覆盖运行

```bash
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### 对比结果

| 条件 | Total Time | 备注 |
|---|---|---|
| 基线（send_lat_table） | 389,404 | 按(算法,协议,链路)查表 |
| AS_SEND_LAT=6 (6ns) | 1,772 | 发送延迟可忽略 |
| AS_SEND_LAT=6000 (6us) | 169,604 | 统一 6us |
| AS_SEND_LAT=7200 (7.2us) | 203,204 | 匹配 Ring+LL+NVLINK |

详细分析请见 [快速入门 - A/B 实验](../getting_started/quickstart.md#5-验证-as_send_lat-效果ab-实验)。

---

> 最后编辑：2026-06-25
