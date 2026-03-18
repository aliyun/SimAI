# SimAI-Physical 模式

> **状态**：Beta — 目前处于内部测试阶段。

SimAI-Physical 在 CPU RDMA 集群环境中生成物理流量。该模式生成类 NCCL 的流量模式，用于深入研究 LLM 训练过程中的 NIC 行为。

---

## 概述

与 SimAI-Analytical 和 SimAI-Simulation 完全在软件中仿真不同，SimAI-Physical 将真实的 RDMA 流量注入物理网络。可用于：

- 研究真实 LLM 训练流量模式下的 NIC 行为
- 在真实硬件上验证网络配置
- 使用典型集合通信工作负载进行 RDMA 性能基准测试

**组件组合**：[AICB](../components/aicb.md) + [SimCCL](../components/simccl.md) + [astra-sim-alibabacloud](../components/astra_sim.md)（物理模式）

---

## 前置条件

SimAI-Physical 使用 RoCEv2 协议生成流量。编译前请确保：

- **RDMA 支持**：可用的 `libibverbs` / RDMA 设备驱动
- **MPI**：已安装并可运行 OpenMPI
- **验证**：能成功运行 `ib_write_bw` 等 RDMA 性能测试工具

---

## 编译

```bash
# 克隆和初始化
git clone https://github.com/aliyun/SimAI.git
cd SimAI/
git submodule update --init --recursive
git submodule update --remote

# 安装 MPI（CentOS/RHEL）
sudo yum install openmpi openmpi-devel

# 设置 MPI 路径
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++

# 编译 SimAI-Physical
./scripts/build.sh -c phy
```

---

## 工作负载生成

SimAI-Physical 使用与 SimAI-Simulation 相同的工作负载格式，通过 [AICB](../components/aicb.md) 生成。详见[工作负载生成](workload_generation.md)。

---

## 准备主机列表

为 MPI 程序准备 IP 列表文件。IP 数量需与参与物理流量生成的 NIC 数量一致（非节点数）。

```
33.255.199.130
33.255.199.129
```

---

## 运行

### MPI 执行

```bash
/usr/lib64/openmpi/bin/mpirun -np 2 \
  -host 33.255.199.130,33.255.199.129 \
  --allow-run-as-root \
  -x AS_LOG_LEVEL=0 \
  ./bin/SimAI_phynet ./hostlist -g 2 -w ./example/microAllReduce.txt
```

### MPI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-np` | 进程数 | 必填 |
| `-host` | 逗号分隔的 IP 列表 | 必填 |
| `--allow-run-as-root` | 允许以 root 运行 | `FALSE` |

### SimAI-Physical 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `hostlist` | 主机 IP 列表文件路径 | 必填 |
| `-w` / `--workload` | 工作负载文件路径 | `./microAllReduce.txt` |
| `-i` / `--gid_index` | RDMA 设备 GID 索引 | `0` |
| `-g` / `--gpus` | GPU 数量（须与 hostlist 中 IP 数一致） | `8` |

---

## 注意事项

- GPU 数量（`-g`）必须与主机 IP 列表中的 IP 数一致
- 确保所有节点具有网络连通性且 RDMA 已正确配置
- SimAI-Physical 目前为 Beta 版本；部分功能可能在后续版本中变更

---

## 相关文档

- [AICB 组件](../components/aicb.md) — 工作负载生成
- [SimCCL 组件](../components/simccl.md) — 集合通信转换
- [astra-sim 组件](../components/astra_sim.md) — 仿真引擎详情
