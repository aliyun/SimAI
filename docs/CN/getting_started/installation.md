# SimAI 安装指南

> [English Version](../../getting_started/installation.md)

## 前置条件

| 要求 | 版本 | 备注 |
|---|---|---|
| 操作系统 | Linux (Ubuntu 20.04+) | 在 Ubuntu 20.04/22.04 上测试通过 |
| GCC/G++ | 9.4.0+ | 需要 C++17 支持 |
| Python3 | 3.8+ | 用于拓扑生成脚本 |
| CMake | 3.14+ | ns3 构建使用 |
| GPU/CUDA | **不需要** | 仿真只需 CPU |
| ninja | **不能安装** | ns3 构建与 ninja 冲突 |

> **注意**：如使用 NGC 容器镜像（推荐用于 AICB workload 生成），需先移除 ninja：
> ```bash
> apt remove ninja-build && pip uninstall ninja
> ```

## 克隆和初始化

```bash
git clone https://github.com/aliyun/SimAI.git
cd SimAI/

# 初始化所有子模块（SimCCL、ns-3-alibabacloud、aicb 等）
git submodule update --init --recursive
git submodule update --remote
```

## 编译 SimAI-Analytical

快速 busbw 仿真（无网络建模）：

```bash
./scripts/build.sh -c analytical
```

输出：`bin/SimAI_analytical`

## 编译 SimAI-Simulation (ns3)

完整 ns3 网络仿真：

```bash
./scripts/build.sh -c ns3
```

输出：`bin/SimAI_simulator`

> **注意**：此命令会删除并重建 `astra-sim-alibabacloud/extern/network_backend/ns3-interface/`。首次构建需 5-15 分钟。

## 编译 SimAI-Physical

物理 RDMA 流量生成（需要 InfiniBand 硬件 + MPI）：

```bash
# 安装 MPI 依赖
sudo yum install openmpi openmpi-devel  # 或 apt install libopenmpi-dev

# 设置 MPI 路径
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++

# 构建
./scripts/build.sh -c phy
```

输出：`bin/SimAI_phynet`

## Docker 环境（推荐）

```bash
# 进入容器
docker exec -it <container_name> bash

# 容器内：编译 ns3 模式
cd /path/to/SimAI
./scripts/build.sh -c ns3

# 验证
ls -la bin/SimAI_simulator
```

## 清理构建

```bash
./scripts/build.sh -l ns3         # 清理 ns3 构建产物
./scripts/build.sh -l analytical  # 清理 analytical 构建
./scripts/build.sh -l phy         # 清理 physical 构建
```

## 验证安装

```bash
# 检查二进制是否存在
ls bin/SimAI_simulator bin/SimAI_analytical

# 快速冒烟测试（8 GPU，单节点）
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H20 -bw 200Gbps -nvbw 2400Gbps

./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf

# 预期：仿真完成，生成 ncclFlowModel_EndToEnd.csv
ls ncclFlowModel_EndToEnd.csv
```

---

> 最后编辑：2026-06-25
