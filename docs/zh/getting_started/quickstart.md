# 快速开始

本指南帮助你运行第一次 SimAI 仿真。

## 1. SimAI-Analytical

最快的入门方式。使用总线带宽（busbw）抽象网络细节。

```bash
# 运行 Analytical 仿真
./bin/SimAI_analytical \
    -w example/workload_analytical.txt \
    -g 9216 \
    -g_p_s 8 \
    -r test- \
    -busbw example/busbw.yaml
```

自动计算总线带宽：

```bash
./bin/SimAI_analytical \
    -w ./example/workload_analytical.txt \
    -g 9216 -nv 360 -nic 48.5 \
    -n_p_s 8 -g_p_s 8 -r example-
```

详细参数说明请参考 [SimAI-Analytical 用户指南](../user_guide/simai_analytical.md)。

## 2. SimAI-Simulation

使用 NS-3 网络后端的全栈仿真。

```bash
# 第一步：创建网络拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
    -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# 第二步：运行仿真
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
    -t 16 \
    -w ./example/microAllReduce.txt \
    -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
    -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

详细参数说明请参考 [SimAI-Simulation 用户指南](../user_guide/simai_simulation.md)。

## 3. 多请求推理仿真

使用 Vidur 框架的端到端推理仿真。

### 前置条件

```bash
# 激活 vidur conda 环境
conda activate vidur
```

### 运行四场景测试套件

```bash
# 运行全部 4 个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --all

# 或运行单个场景
bash vidur-alibabacloud/examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

### 场景概览

| 场景 | 模型 | PD 分离 | World Size | TP | EP | 调度器 |
|------|------|---------|-----------|----|----|--------|
| 1 | Qwen3-Next-80B | 否 | 32 | 1 | 1 | lor |
| 2 | Qwen3-Next-80B | 是（P=2, D=6） | 8 | 1 | 1 | split_wise |
| 3 | DeepSeek-671B | 是（P=2, D=6） | 8 | 8 | 8 | split_wise |
| 4 | Qwen3-MoE-235B | 是（P=2, D=6） | 8 | 4 | 4 | split_wise |

详细信息请参考[推理仿真用户指南](../user_guide/inference_simulation.md)。

## 下一步

- [用户指南](../user_guide/index.md) — 深入了解各仿真模式
- [组件详情](../components/index.md) — 了解各子模块
- [基准测试](../benchmarking/index.md) — 运行完整测试套件
