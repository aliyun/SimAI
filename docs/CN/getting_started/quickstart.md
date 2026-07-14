# SimAI 快速入门指南

> [English Version](../../getting_started/quickstart.md)

## 端到端运行指南

### 1. 编译 SimAI-Simulation (ns3)

```bash
cd SimAI/
./scripts/build.sh -c ns3
```

产物：`bin/SimAI_simulator`（指向 ns3 构建输出的符号链接）。

### 2. 生成拓扑

使用拓扑生成脚本：

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

#### 关键参数

| 参数 | 说明 | 单位/取值 | 示例 |
|---|---|---|---|
| `-topo` | 拓扑模板名称 | `Spectrum-X`, `AlibabaHPN`, `DCN+` | `-topo Spectrum-X` |
| `-g` | GPU 总数 | 整数 | `-g 8` |
| `-gt` | GPU 类型 | `A100`, `H100`, `H800`, `H20` | `-gt H20` |
| `-gps` | 每台服务器 GPU 数 | 整数（默认：8） | `-gps 8` |
| `-bw` | NIC 带宽（横向扩展） | 如 `100Gbps`, `200Gbps`, `400Gbps` | `-bw 200Gbps` |
| `-nvbw` | NVLink 带宽（纵向扩展） | 如 `2400Gbps`, `2880Gbps` | `-nvbw 2400Gbps` |
| `--ro` | Rail-Optimized 拓扑 | 标志位（无需值） | `--ro` |
| `-psn` | PSW 交换机数量 | 整数 | `-psn 64` |
| `--dp` | 双平面 | 标志位 | `--dp` |
| `--dt` | 双 ToR | 标志位 | `--dt` |
| `-nl` | NVLink 延迟 | 如 `0.000025ms` | `-nl 0.000025ms` |
| `-l` | NIC 延迟 | 如 `0.0005ms` | `-l 0.0005ms` |

#### 输出文件名规则

输出文件命名格式：`{模板}_{g}g_{gps}gps_{bw}_{gt}`

示例：
- `--ro -g 8 -gt H20 -bw 200Gbps` → `Rail_Opti_SingleToR_8g_8gps_200Gbps_H20`
- `-topo Spectrum-X -g 128 -gt A100 -bw 100Gbps` → `Spectrum-X_128g_8gps_100Gbps_A100`

### 3. 运行仿真

```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

#### 仿真器参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-t` | 线程数（多线程加速） | 1 |
| `-w` | Workload 文件路径 | 必填 |
| `-n` | 网络拓扑文件路径 | 必填 |
| `-c` | 配置文件路径 | 必填 |

#### 环境变量（命令前设置）

| 变量 | 单位 | 默认值 | 说明 |
|---|---|---|---|
| `AS_SEND_LAT` | **纳秒 (ns)** | 未设置（使用 send_lat_table） | 覆盖每个 flow 的发送延迟。覆盖所有表查询。 |
| `AS_NVLS_ENABLE` | - | `0` | 启用 NVLS 算法用于 AllReduce |
| `AS_PXN_ENABLE` | - | `0` | 启用 PXN 跨节点代理 |
| `AS_LOG_LEVEL` | - | `INFO` | 日志级别：DEBUG/INFO/WARNING/ERROR |

> **重要提示**：`AS_SEND_LAT` 的单位是**纳秒**，不是微秒。send_lat_table 默认值范围为 6000-22000 ns（6-22 μs）。设置 `AS_SEND_LAT=6` 意味着 6 ns（实际上等于禁用发送延迟）。

### 4. 预期输出

仿真在当前工作目录生成以下文件：

| 文件 | 说明 |
|---|---|
| `ncclFlowModel_EndToEnd.csv` | 端到端迭代时序汇总 |
| `ncclFlowModel_detailed_N.csv` | 逐层详细时序（N = 节点数） |
| `ncclFlowModel_detailed_flows.csv` | SimCCL 点对点流分解 |
| `ncclFlowModel_*_dimension_utilization_*.csv` | 通信组利用率 |

验证输出：
```bash
ls ncclFlowModel_*.csv
head -5 ncclFlowModel_EndToEnd.csv
```

### 5. 验证 AS_SEND_LAT 效果（A/B 实验）

**设计原则**：单变量实验。相同拓扑、相同 workload，仅改变 `AS_SEND_LAT`。

```bash
# 基线：使用 send_lat_table（无覆盖）
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# 记录：EndToEnd.csv 中的 total time

# 实验组：使用 AS_SEND_LAT=6000 覆盖（6 μs，接近默认 Ring+LL+NVLINK=7200ns）
AS_SEND_LAT=6000 ./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
# 记录：EndToEnd.csv 中的 total time
```

#### 实测结果（8 GPU, H20, microAllReduce.txt）

| 条件 | Total Time | 备注 |
|---|---|---|
| 基线（send_lat_table 查表） | 389,404 | 表值：Ring+LL+NVLINK=7200ns 等 |
| AS_SEND_LAT=6（6 ns） | 1,772 | 发送延迟可忽略不计 |
| AS_SEND_LAT=6000（6 μs） | 169,604 | 统一 6μs，低于表平均值 |
| AS_SEND_LAT=7200（7.2 μs） | 203,204 | 匹配 Ring+LL+NVLINK 表值 |

**分析**：
- 基线使用按（算法、协议、链路类型）分类的表值，平均值高于任何单一值
- `AS_SEND_LAT` 将所有 flow 覆盖为同一个值，失去了按类型差异化的能力
- 端到端时间受拓扑、拥塞、协议、算法和数据量等多因素影响——不仅仅是 send_lat
- 使用此实验验证发送延迟变化的相对影响

---

## 冒烟测试（8 GPU，单节点）

完整单节点测试可在 1 分钟内完成：

```bash
# 1. 生成拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

# 2. 运行
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# 3. 验证
cat ncclFlowModel_EndToEnd.csv | head -3
```

此测试验证节点内通信（NVLINK）。所有 flow 在单个节点内完成。

---

## 跨节点测试（16 GPU，2 节点）

验证跨节点（NET/IB）行为：

### 1. 创建 16 GPU Workload

修改 `example/microAllReduce.txt`，设置 `all_gpus: 16`：

```bash
# 复制并修改
cp example/microAllReduce.txt example/microAllReduce_16g.txt
# 编辑：将 "all_gpus: 8" 改为 "all_gpus: 16"
sed -i 's/all_gpus: 8/all_gpus: 16/' example/microAllReduce_16g.txt
```

### 2. 生成 16 GPU 拓扑（2 节点 x 8 GPU）

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 16 -gt H20 -bw 200Gbps -nvbw 2400Gbps
```

输出文件：`Rail_Opti_SingleToR_16g_8gps_200Gbps_H20`

### 3. 运行跨节点仿真

```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce_16g.txt \
  -n ./Rail_Opti_SingleToR_16g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

此命令同时测试 NVLINK（节点内）和 NET（节点间）两种通信路径。send_lat_table 对不同链路类型使用不同的延迟值。

---

## 相关文档

- [安装指南](installation.md) — 所有模式的构建说明
- [环境变量](../configuration/env-variables.md) — 完整变量参考
- [构建选项](../configuration/build-options.md) — 构建模式和参数
- [send_lat 分析](../configuration/send-lat-analysis.md) — send_lat 机制深入分析
- [SimCCL 集成](../../../SimCCL/docs/CN/integration/integration-with-simai.md) — SimCCL + SimAI 集成

---

> 最后编辑：2026-06-25
