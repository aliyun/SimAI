<p align="left">
    <a href="CHANGELOG_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

# Changelog

All notable changes to SimAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Note**: This changelog covers v1.0 (initial open-source release) and later versions.

## [Unreleased]

## [1.6.0] - 2026-03-16

### Added

- GPU memory calculation module: accurate parameter counting and KV cache management for DeepSeek-V3-671B, Qwen3-MoE-235B, and Qwen3-Next-80B
- PD-separation memory planning for independent Prefill/Decode memory budgets
- Improved AICB decode time estimation with linear interpolation and global cache
- 4-scenario end-to-end inference test suite (`run_scenarios.sh`)
- SimAI 1.6 Technical Report (EN/ZH)
- Complete bilingual documentation system (30+ files under `docs/en/`, `docs/zh/`)
- GitHub community health files: issue/PR templates, Code of Conduct, Security Policy, Contributing Guide

### Changed

- Replaced print statements with logging across vidur-alibabacloud modules
- Added bilingual docstrings for public APIs
- Standardized TODO comments format

### Removed

- Removed ~390 lines of dead code in vidur-alibabacloud
- Cleaned personal debug markers across 8 files

## [1.5.0] - 2025-12-30

### Added

- **End-to-end multi-request inference simulation**: Full simulation support for multi-request inference workloads.
- **Prefill/Decode separation**: Model complex inference scenarios with Prefill/Decode phase separation.
- **Modern model support**: Added support for DeepSeek, Qwen3-MoE, and Qwen3-Next models.
- **Request scheduling via Vidur**: Integrated request scheduling component adapted from Microsoft's [Vidur](https://github.com/microsoft/vidur) (see [vidur-alibabacloud](./vidur-alibabacloud/)).
- **AICB inference workload generation**: AICB now supports generating prefill/decode inference workloads for DeepSeek, Qwen3-MoE, and Qwen3-Next.
- **DeepSeek training workload support**: AICB now supports generating training workloads for DeepSeek (contributed by [@parthpower](https://github.com/parthpower)).
- **SimCCL initial release**: First public release of the SimCCL collective communication transformation module.

## [1.0.0] - 2024-10-18

### Added

- Initial open-source release of SimAI: full-stack simulator for AI large-scale training
- Core components: AICB, SimCCL, astra-sim-alibabacloud, ns-3-alibabacloud
- SimAI-Analytical: fast simulation using bus bandwidth abstraction
- SimAI-Simulation: full-stack NS3-based network simulation
- SimAI-Physical (Beta): CPU RDMA cluster physical traffic generation

### Academic

- SimAI paper accepted by **NSDI'25 Spring**. See [paper](https://arxiv.org/abs/2410.07346).
