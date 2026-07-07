# send_lat Deep Dive: send_latency_table vs AS_SEND_LAT

> [中文版](../CN/configuration/send-lat-analysis.md)

## Overview

SimAI uses a `send_lat` (send latency) value to delay packet transmission in ns3 simulation. This models the software overhead of initiating a NCCL collective communication operation. Two mechanisms control this value.

---

## 1. send_latency_table (Table Lookup)

### Location
`astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h:134-152`

### How It Works

Two 7x3 tables indexed by `[algorithm][protocol]`:
- `send_lat_table_nvlink[7][3]` — for same-node (NVLINK) communication
- `send_lat_table_net[7][3]` — for cross-node (NET/IB) communication

Algorithm index: 0=Tree, 1=Ring, 2=CollNetDirect, 3=CollNetChain, 4=NVLS, 5=NVLS_TREE, 6=PAT
Protocol index: 0=LL, 1=LL128, 2=Simple

### Table Values (unit: nanoseconds)

#### NVLINK Table (intra-node)
| Algorithm | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 7400 | 15250 | 12400 |
| Ring | 7200 | 15900 | 11800 |
| CollNetDirect | 0 | 0 | 3700 |
| CollNetChain | 0 | 0 | 2800 |
| NVLS | 0 | 0 | 25000 |
| NVLS_TREE | 0 | 0 | 25000 |
| PAT | 0 | 0 | 12000 |

#### NET Table (inter-node)
| Algorithm | LL | LL128 | Simple |
|---|---|---|---|
| Tree | 11800 | 22500 | 22400 |
| Ring | 9300 | 18000 | 22400 |
| CollNetDirect | 0 | 0 | 31000 |
| CollNetChain | 0 | 0 | 30000 |
| NVLS | 0 | 0 | 18000 |
| NVLS_TREE | 0 | 0 | 20900 |
| PAT | 0 | 0 | 22000 |

Value `0` means unsupported combination -> falls back to default 6000 ns.

### Link Type Detection (entry.h:159-162)

```cpp
int gpus_per_node = request->flowTag.gpus_per_node;
bool same_node = (src / gpus_per_node) == (dst / gpus_per_node);
const int (*table)[3] = same_node ? send_lat_table_nvlink : send_lat_table_net;
```

### Call Stack

```
NcclFlowModel::insert_packets()
  → front_end_sim_send()
    → entry.h: SendFlow()
      → Read flowTag.algorithm, protocol, gpus_per_node
      → Determine same_node → select nvlink or net table
      → table[algo][proto] → send_lat (ns)
      → AS_SEND_LAT override check (highest priority)
      → send_lat *= 1000 → convert ns to ps (ns3 Time unit)
      → appCon.Start(Time(send_lat))
```

---

## 2. AS_SEND_LAT (Environment Variable Override)

### Location
`entry.h:168-177`

### How It Works

```cpp
const char* send_lat_env = std::getenv("AS_SEND_LAT");
if (send_lat_env) {
    send_lat = std::stoi(send_lat_env);  // overrides table lookup
}
send_lat *= 1000; // ns → ps
```

### Scope

**AS_SEND_LAT is NOT in SimCCL**. It operates in the ns3 network frontend layer.

| Property | send_latency_table | AS_SEND_LAT |
|---|---|---|
| Location | entry.h (ns3 frontend) | entry.h (ns3 frontend) |
| Granularity | Per-(algorithm, protocol, link_type) | Global single value |
| Priority | Lower | **Highest** (overrides table) |
| Use case | Production simulation | A/B experiments, quick calibration |
| Unit | Nanoseconds | Nanoseconds |

### Can send_latency_table Fully Replace AS_SEND_LAT?

**No.** They serve different purposes:

- `send_latency_table`: Fine-grained per-(algo, proto, link) latency. For accurate production simulations.
- `AS_SEND_LAT`: Quick global override for A/B testing. Setting `AS_SEND_LAT=6` makes ALL communications use 6000 ps regardless of algorithm/protocol.

**Recommendation**: Use table for production, AS_SEND_LAT for debugging/comparison experiments only.

---

## 3. Comparison with NCCL 2.30 Latency Model

### NCCL Source
`nccl-2.30/src/graph/tuning.cc:150-174` (base/hw latency tables), `L380-435` (latency calculation)

### Key Differences

SimAI's `send_lat_table` is a **bucketed approximation** of NCCL's latency model:

| Aspect | SimAI entry.h | NCCL tuning.cc |
|---|---|---|
| Model type | Fixed 3D table lookup | Dynamic calculation per-algorithm |
| Ring latency | Fixed value (e.g., 9300 ns for LL-NET) | `baseLat + nsteps * (intraLat + netOverhead)` |
| PAT latency | Fixed 22000 ns (Simple-NET) | `log2(nNodes) * (interLat/3.5) + nRanks * 2.8` |
| Tree latency | Fixed value | `baseLat + 2*log2(nNodes)*interLat` |
| Depends on topology | Only link type (same/cross node) | nNodes, nRanks, nsteps, netOverhead, cpuArch |
| Hardware-specific | Not (same table for all GPUs) | Per-CPU-vendor `netOverhead` adjustment |

**Implication**: SimAI's table gives reasonable but approximate send_lat. For PAT specifically, the table uses a fixed 22us for NET-Simple, while NCCL computes `log2(N)*(interLat/3.5)+N*2.8` which depends on the actual node count and inter-node latency.

---

## 4. How to Experiment

### Prerequisites

```bash
cd SimAI/
./scripts/build.sh -c ns3
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

### Running with send_lat_table (default)

```bash
./bin/SimAI_simulator -t 8 -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# Check results
head -3 ncclFlowModel_EndToEnd.csv
```

### Running with AS_SEND_LAT override

```bash
AS_SEND_LAT=7200 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

### Comparing Results

| Condition | Total Time | Notes |
|---|---|---|
| Baseline (send_lat_table) | 389,404 | Per-(algo,proto,link) table values |
| AS_SEND_LAT=6 (6ns) | 1,772 | Negligible send latency |
| AS_SEND_LAT=6000 (6us) | 169,604 | Uniform 6us |
| AS_SEND_LAT=7200 (7.2us) | 203,204 | Matches Ring+LL+NVLINK |

For detailed analysis of what these numbers mean, see [Quick Start - A/B Experiment](../getting_started/quickstart.md#5-verify-as_send_lat-effect-ab-experiment).

---

> Last edited: 2026-06-25
