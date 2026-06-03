# Deep Interview Spec: AICB 通信域范围信息增强

## Metadata
- Interview ID: a2c8e1f4-6b3d-4e9a-8c7f-1d5e9b3a7c2f
- Rounds: 6
- Final Ambiguity Score: 15.5%
- Type: brownfield
- Generated: 2026-06-02
- Threshold: 0.2
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.9 | 35% | 0.315 |
| Constraint Clarity | 0.85 | 25% | 0.213 |
| Success Criteria | 0.7 | 25% | 0.175 |
| Context Clarity | 0.95 | 15% | 0.143 |
| **Total Clarity** | | | **0.845** |
| **Ambiguity** | | | **15.5%** |

## Goal
在 AICB 的 LogItem CSV 输出中，为每条通信操作增加**显式的通信域类型标注**和**参与 rank 列表**，同时在 CSV header 段输出完整的 rank mapping table（各并行维度的 group 到 rank 映射）。新格式用于外部分析工具的消费，不改变 SimAI C++ 模拟器的输入格式。

## Constraints
- **仅修改 Python 侧**：只改 `aicb/` 目录下的 Python 代码，不动 `astra-sim-alibabacloud/` 中的 C++ 代码
- **仅修改 LogItem CSV 格式**：SimAI TXT 格式（`Work_Item`）保持不变
- **不要求 SimAI 模拟器兼容**：新 CSV 仅供外部工具使用
- **向后兼容**：旧 CSV 文件不再需要被新版工具消费（专门用途，独立流程）
- **现有 LogItem 字段不变**：只在末尾增加新列，已有解析器不受影响

## Non-Goals
- 不修改 `Work_Item` / SimAI TXT 格式
- 不修改 C++ `Workload.cc` 解析器
- 不改变 SimAI 模拟器的行为
- 不要求新版 CSV 能被现有 SimAI 工具链消费

## Acceptance Criteria
- [ ] LogItem dataclass 增加 `ranks: list[int]` 字段（或等效字段）
- [ ] CSV 输出每条通信操作行末尾包含参与 rank 列表（如 `"0,1,2,3"`）
- [ ] CSV header 段包含完整的 rank mapping table（tp_size, dp_size, ep_size, pp_size 以及每个 group 的 rank 成员列表）
- [ ] 独立验证脚本：对比 CSV 中的 rank 列表与 Python `RankGenerator` 的输出，确保逐位匹配
- [ ] 所有现有 workload generator（megatron, deepspeed stage1/2/3, collective_test）在生成 CSV 时正确填充 ranks 字段
- [ ] `Workload.dump()` 输出格式正确，新增的列和 header 信息完整

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| 通信域信息需要同时加在 SimAI TXT 和 CSV 两种格式 | "你用来做什么？" | 仅 CSV 格式，用于外部分析工具 |
| 需要保持 SimAI C++ 模拟器的向后兼容 | "旧文件在新版解析器中的行为？" | 不需要，新格式有独立用途 |
| 加一个 ranks 字符串列就够了 | "输出格式偏好？" | CSV 加 ranks 列 + header mapping table |

## Technical Context

### 涉及的源文件
- `aicb/log_analyzer/log.py:24-39` — `LogItem` dataclass 定义，需要添加 `ranks` 字段
- `aicb/log_analyzer/log.py:282-302` — `Workload.dump()` 方法，CSV 输出逻辑
- `aicb/utils/utils.py:141-220` — `RankGenerator` 类，已有的 rank 分组逻辑
- `aicb/utils/utils.py:569-580` — `CommGroup` enum
- `aicb/workload_generator/workload_generator.py:19-43` — 基类 `WorkloadGenerator`
- `aicb/workload_generator/generate_megatron_workload.py` — Megatron 训练 workload
- `aicb/workload_generator/generate_deepspeed_stage1_2_workload.py` — DeepSpeed Stage 1/2
- `aicb/workload_generator/generate_deepspeed_stage3_workload.py` — DeepSpeed Stage 3
- `aicb/workload_generator/generate_collective_test.py` — 集合通信测试

### 关键代码关系
```
RankGenerator (utils/utils.py:141)
  |-- generate_masked_orthogonal_rank_groups() (line 36)
  |-- get_ranks() — 返回各 group 的 rank 列表
  |
  v
WorkloadGenerator.__call__() (workload_generator.py:28)
  |-- 调用 init/forward/backward/step
  |-- 生成 LogItem 列表（当前无 ranks 信息）
  |
  v
Workload.dump() (log.py:282)
  |-- 输出 CSV header
  |-- 逐行输出 LogItem（需要增加 ranks 列）
```

### LogItem 当前字段
```python
comm_type, comm_group, comm_group_size, msg_size, stage,
dst, src, additional, _elapsed_time, algbw, busbw, count
```
其中 `comm_group` 已经是显式的 group type（`tp_group`, `dp_group` 等），只需补充实际的 rank 列表。

### 设计要点
- `ranks` 字段：`list[int]`，CSV 序列化为逗号分隔字符串 `"0,1,2,3"` 或范围格式 `"0-3"`
- Header mapping table：在 CSV 开头增加注释行（`#` 前缀），描述各 group 的 rank 组成
- RankGenerator 已被各 workload generator 使用，只需在生成 LogItem 时查询 ranks 并填入

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| LogItem | core domain | comm_type, comm_group, comm_group_size, msg_size, stage, ranks | LogItem belongs to Workload; LogItem references CommGroup |
| Workload | core domain | items, dump() | Workload contains many LogItem |
| RankGenerator | supporting | tp_size, dp_size, ep_size, pp_size, world_size, get_ranks() | RankGenerator produces rank-to-group mapping used by LogItem |
| CommGroup | supporting | tp_group, dp_group, ep_group, pp_group, ep_dp_group, ep_tp_group | CommGroup categorizes LogItem |
| CSV Output | core domain | header mapping table, per-row ranks column | CSV Output serializes Workload + LogItem + rank mapping |
| Verification Script | supporting | compare RankGenerator vs CSV ranks | Verification Script validates CSV Output against RankGenerator |
| External Analysis Tool | external system | consumes CSV with rank information | External Analysis Tool reads CSV Output |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 5 | 5 | - | - | N/A |
| 2 | 7 | 2 | 0 | 5 | 71% |
| 3 | 7 | 0 | 0 | 7 | 100% |
| 4 | 6 | 0 | 0 | 6 | 86% (SimAI Parser removed) |
| 5 | 7 | 1 | 0 | 6 | 86% |
| 6 | 7 | 0 | 0 | 7 | 100% |

Ontology 在 Round 3 后已收敛稳定。Round 4 因明确不涉及 SimAI C++ 而移除 SimAI C++ Parser 实体，此后 entity 集合不再变化。

## Interview Transcript
<details>
<summary>Full Q&A (6 rounds)</summary>

### Round 1
**Q:** 你希望在 workload 中加入的"通信域范围"具体是什么信息？
**A:** 两者都要（显式 group type + 实际 rank ID 列表）
**Ambiguity:** 56% (Goal: 0.7, Constraints: 0.2, Criteria: 0.1, Context: 0.8)

### Round 2
**Q:** 你期望如何验证 rank 列表信息是正确加入的？
**A:** 两者都要（仿真结果一致性 + 独立验证脚本）
**Ambiguity:** 35% (Goal: 0.8, Constraints: 0.25, Criteria: 0.75, Context: 0.8)

### Round 3
**Q:** 你希望修改哪种 workload 格式？是否需要向后兼容？
**A:** 两种格式都改（后续推翻了此回答）
**Ambiguity:** 29% (Goal: 0.85, Constraints: 0.4, Criteria: 0.75, Context: 0.85)

### Round 4
**Q:** 旧版 workload 文件在新版模拟器中运行时的行为？
**A:** 新版 workload 有专门用途，不用于 SimAI
**Ambiguity:** 34% (Goal: 0.6, Constraints: 0.75, Criteria: 0.5, Context: 0.9)

### Round 5
**Q:** 新版带通信域信息的 workload 的专门用途是什么？
**A:** 给其他模拟器/分析工具用
**Ambiguity:** 29% (Goal: 0.8, Constraints: 0.8, Criteria: 0.4, Context: 0.9)

### Round 6
**Q:** 你期望的输出格式？CSV 加 ranks 列？独立配置文件？全新格式？
**A:** CSV 加 ranks + header mapping
**Ambiguity:** 15.5% (Goal: 0.9, Constraints: 0.85, Criteria: 0.7, Context: 0.95)

</details>
