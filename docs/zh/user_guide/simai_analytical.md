# SimAI-Analytical

SimAI-Analytical 通过总线带宽（busbw）抽象网络通信细节来估算集合通信时间，提供快速仿真。适用于快速场景验证和性能分析。

## 适用场景

- **性能分析**：比较不同模型的完成时间（如专家数对 MoE 训练的影响）
- **并行参数优化**：平衡和优化 TP/EP/PP 参数
- **Scale-up 探索**：研究不同 scale-up 域下的并行参数性能
- **Scale-out 带宽选择**：研究高性价比的带宽配置

## 工作负载生成

使用 [AICB](workload_generation.md) 生成工作负载：

```bash
sh ./aicb/scripts/megatron_workload_with_aiob.sh \
    -m 7 --world_size 4096 \
    --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
    --frame Megatron --global_batch 8192 \
    --micro_batch 1 --seq_length 4096 \
    --swiglu --use_flash_attn --aiob_enable
```

生成的 `.txt` 工作负载文件包含：
- `model_parallel_NPU_group`：张量并行度
- `ep`：专家模型并行度
- `pp`：流水线并行度
- `vpp`：虚拟流水线并行度

> 更多信息参见 [AICB 工作负载生成](workload_generation.md) 和 [AICB 组件文档](../components/aicb.md)。

## Busbw 配置

SimAI-Analytical 使用 `busbw.yaml` 文件为不同通信组指定总线带宽：

```yaml
test
TP:
  allreduce,: 300      # TP 组内 AllReduce busbw 300GB/s
  allgather,: 280
  reducescatter,: 280
  alltoall,: 230
DP:
  allreduce,: null
  allgather,: 380      # DP 组内 AllGather busbw 380GB/s
  reducescatter,: 380
  alltoall,: null
EP:
  allreduce,: null
  allgather,: 45
  reducescatter,: 45
  alltoall,: 80        # EP 组内 AlltoAll busbw 80GB/s
```

## 运行 Analytical 仿真

```bash
./bin/SimAI_analytical \
    -w example/workload_analytical.txt \
    -g 9216 \
    -g_p_s 8 \
    -r test- \
    -busbw example/busbw.yaml
```

### 必选参数

| 参数 | 长格式 | 说明 |
|:----:|:-------|:-----|
| `-w` | `--workload` | 工作负载文件路径 |
| `-g` | `--gpus` | 仿真 GPU 规模 |
| `-g_p_s` | `--gpus-per-server` | Scale-up 大小（每服务器 GPU 数） |
| `-r` | `--result` | 输出文件路径和前缀（默认：`./results/`） |
| `-busbw` | `--bus-bandwidth` | busbw 文件路径 |

### 可选参数

| 参数 | 长格式 | 说明 |
|:----:|:-------|:-----|
| `-v` | `--visual` | 生成可视化文件 |

### 重叠比例

| 参数 | 长格式 | 说明 | 范围 |
|:----:|:-------|:-----|:-----|
| `-dp_o` | `--dp-overlap-ratio` | DP 重叠比例 | [0.0-1.0] |
| `-ep_o` | `--ep-overlap-ratio` | EP 重叠比例 | [0.0-1.0] |
| `-tp_o` | `--tp-overlap-ratio` | TP 重叠比例 | [0.0-1.0] |
| `-pp_o` | `--pp-overlap-ratio` | PP 重叠比例 | [0.0-1.0] |

## 结果分析

运行 SimAI-Analytical 后生成的 CSV 文件包含：
- 汇总行：暴露时间、各通信组的计算时间百分比、端到端迭代时间
- 逐层操作详情

![原始输出](../../images/simai_raw.png)

指定 `-v` 参数后还会生成可视化文件：

![可视化](../../images/simai_visual.png)

更多结果分析方法请参考[结果分析](result_analysis.md)。
