<p align="left">
    中文&nbsp ｜ &nbsp<a href="CHANGELOG.md">English</a>
</p>

# 更新日志

SimAI 的所有重要变更均记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。

> **注意**：本更新日志涵盖 v1.0（首次开源发布）及之后的版本。

## [未发布]

## [1.6.0] - 2026-03-16

### 新增

- GPU 内存计算模块：支持 DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B 的精确参数计数与 KV Cache 管理
- PD 分离内存规划：Prefill/Decode 阶段独立的内存预算计算
- 改进 AICB decode 时间估算（首尾线性插值 + 全局缓存）
- 4 场景端到端推理测试套件（`run_scenarios.sh`）
- SimAI 1.6 技术报告（EN/ZH）
- 完整双语文档系统（`docs/en/`、`docs/zh/` 下 30+ 文件）
- GitHub 社区规范文件：Issue/PR 模板、行为准则、安全政策、贡献指南

### 变更

- vidur-alibabacloud 各模块 print 输出替换为 logging
- 公开 API 添加双语 docstring
- TODO 注释格式统一规范化

### 移除

- 清理 vidur-alibabacloud 中约 390 行死代码
- 清理 8 个文件中的个人调试标记

## [1.5.0] - 2025-12-30

### 新增

- **端到端多请求推理仿真**：全面支持多请求推理工作负载的端到端仿真。
- **Prefill/Decode 分离**：支持 Prefill/Decode 阶段分离等复杂推理场景建模。
- **主流模型支持**：新增对 DeepSeek、Qwen3-MoE 和 Qwen3-Next 模型的支持。
- **基于 Vidur 的请求调度**：集成了基于微软 [Vidur](https://github.com/microsoft/vidur) 适配的请求调度组件（详见 [vidur-alibabacloud](./vidur-alibabacloud/)）。
- **AICB 推理工作负载生成**：AICB 现已支持为 DeepSeek、Qwen3-MoE 和 Qwen3-Next 生成 prefill/decode 推理工作负载。
- **DeepSeek 训练工作负载支持**：AICB 新增 DeepSeek 训练工作负载生成支持（由 [@parthpower](https://github.com/parthpower) 贡献）。
- **SimCCL 首次发布**：SimCCL 集合通信转换模块首次对外公开发布。

## [1.0.0] - 2024-10-18

### 新增

- SimAI 首次开源发布：业界首个全栈高精度 AI 大规模训练模拟器
- 核心组件：AICB、SimCCL、astra-sim-alibabacloud、ns-3-alibabacloud
- SimAI-Analytical：基于总线带宽抽象的快速仿真
- SimAI-Simulation：基于 NS3 的全栈网络仿真
- SimAI-Physical（Beta）：CPU RDMA 集群物理流量生成

### 学术

- SimAI 论文被 **NSDI'25 Spring** 接收。详见 [论文](https://arxiv.org/abs/2410.07346)。
