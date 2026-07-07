# Changelog

All notable changes to SimAI are documented in this file.

## [1.7] - 2026-07-07

- [SimCCL](https://github.com/aliyun/SimCCL) v2.30 mock: NCCL-style collective flow decomposition with Ring, PAT, and NVLS algorithm support.
- Protocol-aware selection (LL/LL128/Simple based on message size).
- SimCCL standalone binary for independent collective operation analysis (no GPU required).
- Per-(algorithm, protocol, link_type) send latency table for higher simulation fidelity.

## [1.6] - 2026-04-23

- GPU memory modeling for inference simulation (parameter counting & KV cache).
- Linear interpolation for decode time estimation (replacing nearest-neighbor).
- Prefill-Decode Disaggregation memory planning (independent budgets for Prefill/Decode).

## [1.5] - 2025-12-30

- End-to-end simulation for multi-request **inference** workloads.
- Advanced Inference Simulation: Model complex scenarios with Prefill/Decode separation.
- Modern Model Support: DeepSeek, Qwen3Moe and Qwen3Next.
- Request Scheduling: Adapted from Microsoft's [Vidur](https://github.com/microsoft/vidur).
