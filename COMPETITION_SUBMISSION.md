# 【公司优质SKILL大赛—Harness 长周期自主开发工作流】

## 一、Skill简介

**Skill名称**：`harness-lra-workflow`（Harness 长周期自主开发工作流）

**简单描述**：一套面向 AI 辅助开发的工程化协议，通过 Hook 门禁 + 原子化任务拆分 + 不可变进度日志 + 测试闭环，让 Agent 跨越多个会话持续自主推进长周期项目。安装脚本一键部署，新会话自动恢复上下文，解决 Agent 容易失焦、跨会话记忆断裂、重复开发的核心痛点。

**SkillHub地址**：（待上架后填写）

**Skill 包内容**（10 个文件）：

```
harness-lra-workflow.skill
├── SKILL.md                        # Agent 行为指令 + 工作流描述
├── CHANGELOG.md                    # 版本变更记录
├── .lra_version                    # 协议版本号 (1.0.0)
├── scripts/
│   ├── install.sh                  # 一键安装脚本
│   ├── lra-gate.py                 # Hook 门禁引擎 (pre/post/stop/health)
│   └── quick_status.py             # 项目状态检查
└── references/
    ├── feature_list.json           # 功能追踪模板
    ├── progress.md                 # 进度日志模板
    ├── install-guide.md            # 安装指南
    ├── quick-start.md              # 快速上手
    └── advanced.md                 # 高级特性（版本管理/调用日志/运维）
```

**效果演示要点**：
1. 一键安装：`bash scripts/install.sh`，自动创建所有模板、Hook 配置、测试入口
2. 会话恢复：新会话自动读取 `feature_list.json` + `progress.md`，恢复项目上下文
3. 门禁拦截：编辑未授权文件被 PreToolUse Hook 拦截；Stop 时 `.lra_dirty` 非空强制阻断
4. 测试闭环：运行 `lra-test.sh` 后 dirty 自动清除，integrity hash 更新
5. 健康检查：`lra-gate.py --health` 一键诊断所有配置和数据文件

---

## 二、业务背景

**场景**：大型软件项目的长周期 AI 辅助开发，涉及多语言（C++/Python/TypeScript）、多模块（内核/后端/前端），项目规模 100+ 文件、50+ 功能点，开发周期跨数周。

**核心痛点**：

| 痛点 | 表现 | 根因 |
|------|------|------|
| 无法长时间自主编码 | Agent 处理宏观任务时修改范围失控、幻觉骤增 | 缺乏任务拆分和边界约束机制 |
| 跨会话记忆断裂 | 已完成功能被重复开发、已修复 Bug 被重复引入 | 缺乏结构化的持久记忆和自动状态恢复 |
| 验证仍需人工设计 | Agent 能补测试但不能规划关键业务路径 | 缺乏需求→验证步骤的结构化追踪 |
| 纠偏成本高 | 项目越复杂，Agent 越容易擅自重构、偏离设计 | 缺乏 Hook 级别的硬约束 |

**洞察**：让 Agent 一直自动写代码不等于提效。如果没有工程化协议约束，开发者会花大量时间在审查、纠偏、返工和重跑测试上。提效的关键不是让模型无限自由发挥，而是把任务拆小、把边界定义清楚、把验证自动化。

---

## 三、Skill产生价值

**定量效果**（基于 50+ 功能点的实际使用数据）：

| 指标 | 使用前 | 使用后 |
|------|--------|--------|
| 跨会话上下文恢复 | 5-10 分钟（人工回顾 git log + 代码） | 自动（SessionStart Hook 注入摘要） |
| 重复开发/Bug 回退 | 每 2-3 次会话出现 1 次 | 0（feature_list.json 追踪 + test 保护） |
| 未测试代码合入 | 高（Agent 易跳过测试结束会话） | 0（Stop Hook 强制拦截 .lra_dirty） |
| 人工审查投入 | 每功能点 30-60 分钟 | 每功能点 10-15 分钟（门禁拦截约 80% 低级错误） |
| 单功能开发周期 | 2-3 天（含多轮返工） | 0.5-1 天 |

**覆盖范围**：协议已在复杂项目中验证 50+ 功能点（feature + bugfix + test + refactor），跨 7+ 开发会话，99 个存量测试持续保护，E2E 测试覆盖全链路。

**可推广性**：安装脚本自动适配项目结构，适用于任何 10 文件以上、跨 3+ 会话的中大型项目。

---

## 四、Skill构建思路

### 4.1 核心设计原则

**"不是让模型一次记住所有东西，而是让模型每次只专注一个百行级任务"**。

复杂任务 → 原子化拆分 → 每个任务有明确的输入（feature scope）、输出（passes）、验收标准（verification_steps）→ Hook 强制执行。

### 4.2 架构：Agent + Hook + 文档 + 测试闭环

```
SessionStart Hook          → 注入 feature_list 摘要 + 进度 + 上下文
  ↓
PreToolUse Hook            → 文件不在 scope → 拦截
                          → feature 无 verification_steps → 拦截
                          → 置信度未声明 → 拦截
  ↓
PostToolUse Hook           → 写入 .lra_dirty
  ↓
Stop Hook                  → .lra_dirty 非空 → 拦截
                          → passes=false → 拦截
  ↓
lra-test.sh                → backend + tsc + e2e → update hashes
```

### 4.3 关键数据模型

| 文件 | 用途 | 管理方式 |
|------|------|----------|
| `feature_list.json` | 结构化功能树，每个条目含 scope/verification/confidence | 手动创建，Hook 读取 |
| `progress.md` | 不可变进度日志 | 仅追加，hash 防篡改 |
| `.lra_dirty` | 会话级变更追踪 | Hook 自动管理，测试通过清除 |
| `.lra_done_hash` | 完整性校验 | 测试通过后自动更新 |
| `.lra_version` | 协议版本（语义化版本） | install.sh 写入，gate 校验兼容性 |
| `.lra_sessions/` | 会话日志 | 每次会话记录，支持统计分析 |

### 4.4 版本管理与调用日志

**版本管理**：`.lra_version` 记录协议版本和兼容范围（如 `"compatible": ["1.0.x"]`），`lra-gate.py --check-version` 在 SessionStart 校验。`CHANGELOG.md` 记录每个版本的新增特性。

**调用日志**：每个会话自动记录到 `.lra_sessions/`（session_id、features_touched、tests_run、dirty_cleared）。Hook 拦截事件记录到 `.lra_audit.jsonl`（详细记录每次 block 的原因和上下文）。

**运营运维**：`lra-gate.py --health` 一键诊断配置和数据完整性；`lra-gate.py --repair` 自动修复 hash 不一致、孤立的 dirty 条目等常见问题。

### 4.5 独创性总结

| 机制 | 来源 | 说明 |
|------|------|------|
| 原子化任务拆分 | 原创 | 将功能拆为百行级，每个 feature 绑定具体 files + verification_steps |
| Hook 三阶段门禁 | 原创 | PreToolUse/PostToolUse/Stop 全生命周期强制检查 |
| 置信度分级决策 | 原创 | HIGH→直接修，LOW→交给用户，减少无效纠偏 |
| 不可变进度日志 | 借鉴 git append-only | progress.md 历史不可修改，hash 校验防止篡改 |
| 测试闭环 | 借鉴 TDD | 非测试通过不能标记 done，Stop Hook 强制 |
| 协议版本管理 | 借鉴 semver | .lra_version + CHANGELOG，向后兼容声明 |
| 调用审计日志 | 原创 | 会话级 + Hook 拦截级双层日志，支持统计和回溯 |

---

## 六、经验总结及建议

### 6.1 构建心得

1. **协议先于能力**：不要指望更强的模型解决所有问题。把工程化约束（Hook + 文档 + 测试）做扎实，中等模型也能在长周期项目中稳定产出。

2. **门禁比提示词更可靠**：Prompt 层面的"请先跑测试"大概率被忽略。Hook 级别的硬拦截（Stop 时检查 .lra_dirty）100% 生效。

3. **原子化是提效的关键**：百行级 feature + 明确的文件 scope，Agent 不会"想太多"，幻觉率显著下降。

4. **可运营是可持续的前提**：版本管理 + 调用日志 + 健康检查，让协议本身成为可维护的系统，而不是一次性脚本。

### 6.2 给其他 Skill 作者的建议

- **先定义边界，再写实现**：feature 的 `files` 和 `verification_steps` 在写代码之前就明确。
- **Hook 要少而精**：3-4 个关键节点足矣，过多 Hook 会拖慢交互速度。
- **示例数据 > 文档说明**：提供可运行的模板（feature_list.json、install.sh），比纯文字更易理解。
- **考虑可运营性**：Skill 不是"写完就完"的，加入版本号、健康检查、调用日志，让它能持续演进。
