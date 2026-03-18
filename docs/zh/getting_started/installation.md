# 安装指南

本指南介绍如何安装 SimAI 及其依赖。

## 方式一：Docker（推荐）

```bash
# 构建 Docker 镜像
docker build -t simai:latest .

# 运行容器（带 GPU 支持）
docker run --gpus all -it --rm \
    -v $(pwd)/results:/workspace/SimAI/results \
    simai:latest /bin/bash
```

> **注意：** 如使用 Hopper GPU，请在 Dockerfile 中添加 `ENV FLASH_MLA_DISABLE_SM100=1`。

## 方式二：从源码编译

以下步骤已在 GCC/G++ 9.4.0、Python 3.8.10、Ubuntu 20.04 环境下测试通过。

> **重要：** 请勿安装 ninja（NGC 镜像中已预装，需移除以兼容 SimAI-Simulation 编译）。
> ```bash
> apt remove ninja-build && pip uninstall ninja
> ```

### 第一步：克隆仓库

```bash
git clone https://github.com/aliyun/SimAI.git
cd ./SimAI/

# 初始化子模块
git submodule update --init --recursive
# 更新到最新提交
git submodule update --remote
```

### 第二步：编译 C++ 组件

根据需要选择编译模式：

```bash
# SimAI-Analytical（快速，抽象网络细节）
./scripts/build.sh -c analytical

# SimAI-Simulation（使用 NS-3 网络后端的全栈仿真）
./scripts/build.sh -c ns3

# SimAI-Physical（Beta，需要 RDMA 环境）
sudo yum install openmpi openmpi-devel
export MPI_INCLUDE_PATH=/usr/include/openmpi-x86_64/
export MPI_BIN_PATH=/usr/lib64/openmpi/bin/mpic++
./scripts/build.sh -c phy
```

### 第三步：安装 Python 依赖

```bash
pip install -r aicb/requirements.txt
pip install -r vidur-alibabacloud/requirements.txt
```

### 第四步：验证编译结果

```bash
ls bin/  # 应包含 SimAI_analytical 和/或 SimAI_simulator
```

## 方式三：Conda 环境（推理仿真专用）

```bash
cd vidur-alibabacloud
conda env create -p ./env -f ./environment.yml
conda activate vidur
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## NGC 容器（工作负载生成）

使用 AIOB 进行计算性能分析生成工作负载时，建议直接使用 NGC 容器镜像：

```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm \
    -v /path/to/SimAI:/workspace/SimAI \
    nvcr.io/nvidia/pytorch:xx.xx-py3
```

> **注意：** 请使用 PyTorch >= 23.08 版本的 NGC 镜像。

## 下一步

- [快速开始](quickstart.md) — 运行第一次仿真
- [用户指南](../user_guide/index.md) — 各模式详细使用方法
