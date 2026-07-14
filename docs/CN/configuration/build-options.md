# SimAI 构建选项

> [English Version](../../configuration/build-options.md)

## 构建模式

| 模式 | 命令 | 输出二进制 | 说明 |
|---|---|---|---|
| analytical | `./scripts/build.sh -c analytical` | `bin/SimAI_analytical` | 快速 busbw 仿真（无网络建模） |
| ns3 | `./scripts/build.sh -c ns3` | `bin/SimAI_simulator` | 完整 ns3 网络仿真（高保真度） |
| phy | `./scripts/build.sh -c phy` | `bin/SimAI_phynet` | 物理 RDMA 流量生成（需要 IB 硬件） |

## 清理构建

```bash
./scripts/build.sh -l ns3         # 清理 ns3 构建产物
./scripts/build.sh -l analytical  # 清理 analytical 构建
./scripts/build.sh -l phy         # 清理 physical 构建
```

## SimAI_simulator 参数（ns3 模式）

| 参数 | 长格式 | 说明 | 默认值 |
|---|---|---|---|
| `-t` | `--thread` | 多线程加速的线程数 | 1 |
| `-w` | `--workload` | Workload 文件路径 | 必填 |
| `-n` | `--network-topo` | 网络拓扑文件路径 | 必填 |
| `-c` | `--config` | 仿真配置文件路径 | 必填 |

典型用法：
```bash
./bin/SimAI_simulator -t 8 \
  -w ./example/microAllReduce.txt \
  -n ./Rail_Opti_SingleToR_8g_8gps_200Gbps_H20 \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

> **线程建议**：多线程模式推荐使用 8-16 个线程。更高的线程数可减少大规模拓扑的仿真时间。

## SimAI_analytical 参数

| 参数 | 长格式 | 说明 | 默认值 |
|---|---|---|---|
| `-w` | `--workload` | Workload 文件路径 | 必填 |
| `-g` | `--gpus` | GPU 总数 | 必填 |
| `-g_p_s` | `--gpus-per-server` | 每台服务器 GPU 数（纵向扩展大小） | 必填 |
| `-r` | `--result` | 输出文件路径/前缀 | `./results/` |
| `-busbw` | `--bus-bandwidth` | busbw.yaml 文件路径（用户自定义 busbw）。**当前开源二进制未接线** — 见下方说明 | 不适用 |
| `-v` | `--visual` | 生成可视化文件 | 关闭 |
| `-nv` | - | NVLink 带宽（GB/s），用于自动 busbw（默认方法） | 推荐 |
| `-nic` | - | NIC 带宽（GB/s），用于自动 busbw（默认方法） | 推荐 |
| `-n_p_s` | - | 每台服务器 NIC 数，用于自动 busbw（默认方法） | 推荐 |

典型用法（自动 busbw —— 默认方式，当前二进制支持）：
```bash
./bin/SimAI_analytical \
  -w ./example/workload_analytical.txt \
  -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 \
  -r example-
```

> 注意：下面的 `-busbw example/busbw.yaml`（用户自定义 busbw）用法在当前开源 analytical 二进制中**不受支持**——`-busbw` 参数不会被解析，命令会打印用法并退出。此处仅作保留参考；手动 `busbw.yaml` 用法请参考早期 SimAI 版本。
```bash
./bin/SimAI_analytical \
  -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 \
  -r test- \
  -busbw example/busbw.yaml
```

## 拓扑生成器参数

详见 [快速入门 - 生成拓扑](../getting_started/quickstart.md#2-生成拓扑)。

---

> 最后编辑：2026-06-25
