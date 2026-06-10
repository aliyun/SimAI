# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimAI is a full-stack, high-precision simulator for AI large-scale training. It provides detailed modeling and simulation of the entire LLM training process, encompassing framework, collective communication, and network layers. The project has been accepted by NSDI'25 Spring.

## Architecture

SimAI consists of four main components that can be combined in different ways:

1. **AICB** (`aicb/`) - AI Communication Benchmark for workload generation and testing
2. **SimCCL** (`SimCCL/`) - Collective communication library simulation
3. **astra-sim-alibabacloud** (`astra-sim-alibabacloud/`) - Extended from astra-sim with NCCL algorithms
4. **ns-3-alibabacloud** (`ns-3-alibabacloud/`) - Network simulator backend

### Three Operation Modes

- **SimAI-Analytical**: Fast simulation using bus bandwidth (busbw) abstraction to estimate collective communication time
- **SimAI-Simulation**: Full-stack simulation with fine-grained network communication modeling using NS3
- **SimAI-Physical** (Beta): Physical traffic generation for CPU RDMA cluster environments

## Build Commands

The project uses GCC/G++ 9.4.0 and Python 3.8.10 on Ubuntu 20.04.

### Initial Setup
```bash
# Clone with submodules
git clone https://github.com/aliyun/SimAI.git
cd SimAI
git submodule update --init --recursive
git submodule update --remote
```

### Compilation
```bash
# Compile SimAI-Analytical
./scripts/build.sh -c analytical

# Compile SimAI-Simulation (ns3)
# IMPORTANT: Remove ninja first if in NGC container
# apt remove ninja-build && pip uninstall ninja
./scripts/build.sh -c ns3

# Compile SimAI-Physical
./scripts/build.sh -c phy

# Clean builds
./scripts/build.sh -l analytical  # or ns3, phy
```

Compiled binaries are symlinked to `./bin/`:
- `SimAI_analytical`
- `SimAI_simulator`
- `SimAI_phynet`

## Running Simulations

### SimAI-Analytical

Basic usage:
```bash
./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

With automatic bus bandwidth calculation:
```bash
./bin/SimAI_analytical -w ./example/workload_analytical.txt -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 -r example-
```

Key parameters:
- `-w/--workload`: Path to workload file
- `-g/--gpus`: Number of GPUs to simulate
- `-g_p_s/--gpus-per-server`: Scale-up size (GPUs per server)
- `-r/--result`: Output file path/prefix (default: `./results/`)
- `-busbw/--bus-bandwidth`: Path to busbw.yaml file
- `-v/--visual`: Generate visualization files
- `-dp_o/--dp-overlap-ratio`: DP overlap ratio [0.0-1.0]
- `-ep_o/--ep-overlap-ratio`: EP overlap ratio [0.0-1.0]
- `-tp_o/--tp-overlap-ratio`: TP overlap ratio [0.0-1.0]
- `-pp_o/--pp-overlap-ratio`: PP overlap ratio [0.0-1.0]

### SimAI-Simulation (NS3)

Generate network topology first:
```bash
# Example: Spectrum-X topology with 128 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps
```

Run simulation:
```bash
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./Spectrum-X_128g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

Environment variables:
- `AS_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR, UNKNOWN; default: INFO)
- `AS_PXN_ENABLE`: Enable PXN (0/1; default: false)
- `AS_NVLS_ENABLE`: Enable NVLS (0/1; default: false)
- `AS_SEND_LAT`: Packet sending latency in microseconds (default: 6)
- `AS_NVLSTREE_ENABLE`: Enable NVLSTREE (default: false)

Parameters:
- `-t/--thread`: Number of threads (8-16 recommended for multithreading)
- `-w/--workload`: Path to workload file
- `-n/--network-topo`: Network topology path
- `-c`: Configuration file path

### Network Topology Templates

Five built-in templates available:
- `Spectrum-X`: Single plane, rail-optimized, single ToR
- `AlibabaHPN`: Dual-plane or single-plane, rail-optimized, dual ToR
- `DCN+`: Single plane, non-rail-optimized, dual ToR or single ToR

Example topology generation:
```bash
# Dual-Plane AlibabaHPN with 64 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo AlibabaHPN --dp -g 64 -asn 16 -psn 16

# Dual-ToR DCN+ with 128 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo DCN+ --dt -g 128 -asn 2 -psn 8

# Custom rail-optimized single ToR with 32 GPUs
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -g 32 -bw 200Gbps -gt A100 -psn 8 --ro
```

## Workload Generation

Workloads are generated using AICB. The workload files specify:
- `model_parallel_NPU_group`: Tensor Parallelism size
- `ep`: Expert model parallelism size
- `pp`: Pipeline model parallelism size
- `vpp`: Virtual Pipeline Parallelism

For detailed workload generation, refer to the [AICB tutorial](https://github.com/aliyun/aicb/blob/master/training/tutorial.md).

## Key Python Components

### AICB Main Entry (`aicb/aicb.py`)
Main script for running AICB workload generation and testing. Supports:
- Megatron framework workloads
- DeepSpeed Stage 1/2/3 workloads
- Collective communication tests
- AIOB (AI Operator Benchmark) integration for compute timing

### Workload Generators (`aicb/workload_generator/`)
- `generate_megatron_workload.py`: Megatron-LM workload generation
- `generate_deepspeed_stage1_2_workload.py`: DeepSpeed ZeRO Stage 1/2
- `generate_deepspeed_stage3_workload.py`: DeepSpeed ZeRO Stage 3
- `generate_collective_test.py`: Collective communication patterns
- `AIOB_simAI_workload_generator.py`: AIOB-based workload generation

### Analysis Tools (`aicb/log_analyzer/`)
- `analyze_res_csv.py`: Analyze simulation results
- `ds_comm_log_analyzer.py`: DeepSpeed communication log analysis
- `plot.py`: Visualization utilities

## Directory Structure

```
SimAI/
├── aicb/                    # AI Communication Benchmark
├── astra-sim-alibabacloud/  # Core simulator (extended from astra-sim)
├── ns-3-alibabacloud/       # NS3 network simulator backend
├── SimCCL/                  # Collective communication library
├── scripts/                 # Build scripts
│   └── build.sh            # Main build script
├── example/                 # Example workloads and configs
│   ├── workload_analytical.txt
│   ├── microAllReduce.txt
│   └── busbw.yaml
├── docs/                    # Documentation
│   └── Tutorial.md         # Comprehensive tutorial
├── bin/                     # Compiled binaries (created after build)
└── results/                 # Simulation output (created after build)
```

## Important Notes

- The project uses git submodules extensively. Always run `git submodule update --init --recursive` after cloning.
- NGC container images are recommended for workload generation with GPU timing (AIOB feature).
- When compiling SimAI-Simulation in NGC containers, remove ninja first.
- Results are output as CSV files with detailed timing breakdowns by layer and communication group.
- Visualization can be enabled with the `-v` flag for analytical simulations.
- The simulator supports multithreading (8-16 threads recommended) for faster simulation.

## LRA 强制执行工作流

本项目采用长时间运行代理（LRA）协议，通过 Hook 硬约束 + Claude 行为指令确保跨会话的连续性和代码质量。

### 你的职责（必须执行）

### 两阶段工作流

**阶段 1 — 需求澄清（交互）**
- 用户提出需求 → Claude 追问澄清
- 明确后 → 在 feature_list.json 中创建条目（id、type、description、files 范围、verification_steps）
- type=`feature` 必须同步创建 test 条目

**阶段 2 — 开发执行**
- 编码必须在 in_progress feature 范围内
- Gate 强制：无 feature → BLOCK，文件不在 scope → BLOCK
- 每次代码变更后：更新 progress.md + 跑 lra-test.sh
- Claude 自己管理 session 生命周期（compaction + 必要时 exit）
- LRA 只负责状态持久化：compaction 时 save context，exit 时 stop check，下次启动 init.sh 恢复

**遇到 Bug/问题时 — 必须先输出置信度（强制）：**

第一步必须明确输出：
   `【置信度: HIGH】` 或 `【置信度: LOW】` + 一句话理由

第二步按决策树执行：
1. HIGH + 有 in_progress feature，scope 覆盖？→ 直接修
2. HIGH + 没 feature 或 scope 不覆盖？→ 解释为什么高置信，然后"我建个 bugfix 条目"
3. LOW → 给出分析+证据+选项，丢给用户决策。不修。
4. 低置信度的信号：根因不明确、触及不熟悉模块、多方案需权衡、可能引入回归、涉及核心算法/数据结构变更

**致命禁区：禁止程序化修改 hook 配置。**
- `.claude/settings.local.json` 中的 hook command 只能通过 install-lra.sh 生成或手动编辑
- 禁止用 Python/sed/Bash 脚本批量替换 hook 命令文本
- 引号转义错误会导致整个 LRA 工具链失效

### Hook 自动强制规则

| Hook | 规则 |
|------|------|
| PreToolUse | 编辑非白名单文件时，必须有 feature 处于 `in_progress` |
| PreToolUse | `.lra_dirty` 非空且 feature 不匹配 → 阻止（先跑测试） |
| PreToolUse | type=`feature` 且无 verification_steps → 阻止 |
| PostToolUse | 编辑非白名单文件 → 写入 `.lra_dirty` |
| Stop | `.lra_dirty` 非空 → 阻止会话结束 |
| Stop | done 但 passes=false → 阻止 |

### feature_list.json 规范

每个条目包含：
- `id`: 功能编号 (F001, F002, ..., T001, T002, ...)
- `type`: `bugfix` | `feature` | `test` | `refactor`
- `category`: 子分类
- `description`: 功能描述
- `status`: `pending` | `in_progress` | `done`
- `priority`: P0 / P1 / P2 / P3
- `phase`: phase0-4
- `verification_steps`: 验证步骤列表（**禁止删除或修改已有步骤**）
- `passes`: 是否通过验证
- `created_at` / `updated_at`: 时间戳

### 测试策略

- 后端: `cd server && python3 -m pytest tests/ -v`
- 前端类型: `cd dashboard && npx tsc --noEmit`
- E2E: `cd dashboard && npx playwright test`
- 全部测试: `scripts/lra-test.sh`

### 进度文件

- `progress.md`: 人类可读进度 + Session Change Log
- `feature_list.json`: 结构化功能追踪
- `.lra_dirty`: 未测试变更标记（Hook 管理，不可手动删除）

## Related Documentation

- Full tutorial: `docs/Tutorial.md`
- AICB documentation: https://github.com/aliyun/aicb
- SimCCL documentation: https://github.com/aliyun/SimCCL
- Paper: [NSDI'25 Spring - SimAI](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)
