# SimAI-OXC 开发进度

## 状态

**当前阶段**: Phase 3 — 功能扩展  
**完成**: 33 features | **待开发**: 13 items | **LLD→OXC→NS3 端到端**: ✅

## 近期 Session

| 日期 | 工作 |
|------|------|
| 2026-06-09 | v3 lld NPU-Leaf 连接分析: server#0 36端口每端口连2 leaf(滑窗), 共64边, 非用户的8NPU×8端口×1leaf拓扑. + F092 localStorage migration fix (旧serverIps映射到serverIds) |
| 2026-06-08 | F092: serverIps→serverIds rename — v3 lld server node_id is name not IP. 10 files: network.ts, network-store.ts, edg-api.ts, HomePage.tsx, EdgPage.tsx, EdgDiffGraph.tsx, routes.py, merger.py, conftest.py, test_merger.py, test_e2e_ns3_oxc.py. npu_match server_ip preserved. 79 tests + tsc + E2E pass |
| 2026-06-08 | Deep Interview: MegatronWorkload vs MegatronModel 选型分析 — 4轮访谈(14.0%), 逐函数对标NCCL trace, 发现PP double-step bug. spec: .omc/specs/deep-interview-megatron-workload-vs-model.md |
| 2026-06-08 | F091: EDG init 全局持久化 — config.py加EDG_DATA_ROOT, routes.py新增_edg_global_dir()/_edg_load()支持global store(优先)/workspace(回退), init双写, baseline-graph+register-task读global. 前端: edg-api.ts传topology_dir, EdgPage.tsx传topologyDir, wizard-store.ts加zustand/persist(EDG graph数据跨refresh存活) |
| 2026-06-08 | EDG init 持久化分析 — 根因: lld.json/init_crosses.json 存于 session 级 workspace, 新 session 丢失需重新 init。方案: 迁移到全局 EDG_DATA_ROOT/{topology_dir}/, topology_dir 已在 localStorage 持久化。改动5文件 |
| 2026-06-08 | Rank 分解 CSV 验证 — megatron_demo_128gpu_sp_False 的 all_groups 列验证 TP/DP/PP 组完全对标 Megatron-LM tp-cp-ep-dp-pp 排序，AICB CSV 已输出正确的 per-rank 数据 |
| 2026-06-04 | F090: LLDP v2→v3 migration — 9 files refactored (deep-interview→ralplan→autopilot pipeline). Field renames: node_ip→node_id, port_infos→port_id_list, server_type→chassis_topo. +_chassis_to_npu_type(). Bandwidth from port_id. 19/19 tests pass. Frontend tsc clean. |
|------|------|
| 2026-06-04 | Deep Interview: Rank 分解架构分析 — AICB vs SimAI 双重实现问题，确认 AICB RankGenerator 和 Megatron-LM 使用相同默认排序 tp-cp-ep-dp-pp，不是理论分法。产出 spec: .omc/specs/deep-interview-rank-decomposition.md (4轮访谈, ambiguity 10.7%)。方案: 显式传递 rank_ordering 参数消除双重实现风险 |
| 2026-06-03 | 128 GPU per-rank workload 生成演示 — tp=8,dp=8,pp=2, 全部128 rank CSV输出; 修复lra-test.sh时间戳在hash后记录避免mtime误判; 验证per-rank CSV唯一差异为ranks列 |
| 2026-06-03 | 3个submodule扁平化+server+dashboard+F089全部改动 -> https://gitcode.com/yanzhenghao/oec-sim |
| 2026-06-03 | generate_per_rank_csv.py(自动检测sidecar中的tp/pp/ep/world_size), 8 rank全部输出, TP/DP/PP各组ranks验证正确 |
| 2026-06-04 | F090 spine+OXC topology: lld.json updated to full v3 ELECTRICAL_2_OPTICAL_1 topology (2 OXC, 2 Spine, 2 Leaf, 1 Server). merger.py _build_edge_maps updated to chain OXC→spine→leaf. edg_client.py _mock_baseline_crosses + _smart_adjustment spine-aware. SimAI.conf: +800Gbps rate map, /etc→/tmp paths. NS3 sim passes 8-GPU ALLREDUCE with 800Gbps links, A5 NPU. |
| 2026-06-04 | Session checkpoint: F090 active (spine topology work), all tests pass. |
| 2026-06-03 | Python RankGenerator vs C++ MockNcclGroup: TP一致✅, DP不一致(Python strided vs C++ 前/后半), PP在C++中空实现❌. 根因: 两套独立rank分解实现缺乏对齐. |
| 2026-06-04 | Session checkpoint: F090 complete, F088/F089 in progress. All tests pass. |
| 2026-06-02 | F089: 新增ranks可视化 — Domain Flow Graph+Domain MsgSize Bar; 交叉验证36 ops ranks与RankGenerator逐位一致 |
| 2026-06-02 | F089: AICB通信域ranks实现完成 — LogItem+ranks字段, rank_mapper.py, _fill_ranks(), dump sidecar, 15/15 tests pass, LRA gate多feature支持 |
| 2026-06-02 | Deep Interview + Ralplan: AICB通信域范围增强 — 6轮Socratic访谈(ambiguity 100%→15.5%), 3轮Architect+Critic共识(12+ findings fixed), 产出spec(.omc/specs/deep-interview-aicb-comm-domain.md)+plan(.omc/plans/ralplan-aicb-comm-domain.md). 方案: LogItem加ranks字段+sidecar rank_mapping.csv, Python-only, hybrid population(Option C). |
| 2026-06-02 | AICB能力分析: aicb workload vs 真实Megatron训练差异 — 无计算通信overlap建模, 无hiding调度, 推理workload可用(SimAI_inference/Vidur生成器, 支持DeepSeek-V3/Qwen3-MoE/Qwen3-Next, prefill+decode). Overlap/hiding完全交给SimAI模拟器的overlap ratio参数(-dp_o/-tp_o/-pp_o/-ep_o). |
| 2026-05-12 | F054 ✅ DONE: 全场景取消 NVSwitch，所有 server 机内 full mesh + NPU→Leaf 直连。single-server 通信占比 33.06%，multi-server 16GPU 仿真通过。修复三处: (1) ns3_emitter.py 移除 NVSwitch 相关拓扑生成； (2) MockNcclGroup.cc 空 NVSwitch vector 越界访问； (3) rdma-hw.cc intra-server 路由强制走 NVSwitch 导致断言失败，改为 NVSwitch 路径可选，无 NVSwitch 时回退到 m_rtTable 直连链路。 |
| 2026-05-12 | T028: NS3 full-mesh E2E dashboard launch 验证. 修复 SimAI_simulator 符号链接路径错误(缺少../). 修复 launch API 400 (workload/ranktable 未注入 auth token). 修复 status polling endpoint (/api/process/logs 非 /api/process/status). E2E test 可成功启动 NS3 仿真. |
| 2026-05-12 | LRA 协议增强：遇到 Bug 时按置信度分级处理 — 高置信度直接修，低置信度给分析+选项丢给用户 |
| 2026-06-04 | F090 spine mapping 修复: spine→leaf 被 OXC→spine 边污染，加 oxc_ips 排除。test_merger.py 改为 spine topology 适配 (7 tests)。test_e2e_ns3_oxc step2 加 spine 支持。99/99 tests pass。LRA .skill 包导出 (lra-simai-snapshot.skill, 71KB, 5 files)。 |
| 2026-05-12 | LRA context-save 机制：compaction 时自动保存现场到 .lra_context_warning，下次 init.sh 顶部大字提醒 |
| 2026-06-04 | F090 收尾: skill 改名 harness-lra-workflow, 竞赛申报书输出至 COMPETITION_SUBMISSION.md, 全部测试通过 |
| 2026-06-04 | F090 N:N topology: OXC port→leaf full fan-out, spine→leaf/leaf→server/oxc→spine all N:N, 8-server 64-leaf lld.json, 1/2/4/8 server NS3 ALLREDUCE all pass, 64 GPU 128n 1312l ✅ |
| 2026-06-04 | F090 multi-leaf 支持: server_to_leaves (1:N), NPU round-robin 分配跨 leaf, OXC 2 leaf-leaf edges 激活, NS3 sim passes, 99 tests |
| 2026-06-05 | F090 AIOB E2E: 8-srv 64-leaf 580-edge v3 topology, AIOB算子表 (gpu_compute_timing.txt) 生成 workload, 8/32/64 GPU NS3 ALLREDUCE all pass. TP busbw=43.9GB/s (NVLink), DP busbw=6.7GB/s (OXC cross), 6.5x gap normal |
| 2026-05-13 | F067-F068: AIOB auto-enable + homepage cleanup, E2E exemption for deletions-only frontend changes |
| 2026-05-13 | F059-F060: install-lra.sh 完全自包含，rule 装项目级 .claude/ |
| 2026-05-13 | F055-F058: LRA 三条防线 — gate scope/dirty/done+progress |
| 2026-05-12 | LRA install/uninstall 脚本：基于 cp 不再嵌 heredoc，uninstall 对称清理 |
| 2026-06-05 | AICB workload 生成流程梳理: 前端 WorkloadPage→API→workload_generator.py→SIMAI_workload, 后端直调 AICB 模块 (非 CLI), AIOB 算子表注入路径完整, 前端 preset 模式 + CLI 命令行两种入口 |
| 2026-06-05 | F090 AIOB workload生成+验证: 修复SimAI_training_workload_generator.py get_model_details() bug (model→self.model), 通过server/workload_generator.py generate_megatron_workload(aiob_enable=True) 生成GPT-13B workload, 8GPU v3拓扑验证通过: compute=137ms(89.7%) comm=15ms(10.3%) busbw=50.6GB/s NVLink. 64GPU 128n 1312l太重需Linux服务器 |
| 2026-06-05 | PP=4 workload生成: aicb.py + gloo backend, 2090 items. 确认megatron_demo_128gpu CSV由aicb.py生成, computationEnable_False跳过了fwd/bwd层的send/recv |
| 2026-06-05 | 两个workload生成器深度对比: MegatronWorkload(1F1B+PP ISEND/IRECV+每层4+LogItem) vs SIMAI_workload(无PP调度+每层1Item). NS3包级仿真必须用MegatronWorkload, SIMAI_workload仅适用于analytical |
| 2026-06-05 | F090 gitcode 4 commits: v3 LLDP + port-level spine + OXC strip fix + srv_leaf_count strip. E2E verified: 16GPU 64 LL edges + NS3 sim OK |
| 2026-06-08 | 回顾 F086-F089 解耦: Layer1(ranks) + Layer2(flow decouple) 均已完成, 两层正交, 无需合并 |
| 2026-06-08 | 流文件解析: AS_DECOUPLED_OUTPUT生成flow_output.txt, 896条RING流, 262KB/chunk×14, 117MB/节点, 按GPU分组(先走完同一GPU所有chunk), 定义在MockNcclGroup.cc:2221+AstraSimNetwork.cc:349+entry.h:171 |
| 2026-05-12 | 38988 根因定位：TP AllReduce 死绑 NCCL_ALGO_RING，全网格 28 条链路只用到 1 条。需加 NCCL_ALGO_TREE 分支 |
| 2026-05-09 | F052 ✅ DONE: HomePage NPU_INTRA_MAP 补齐 A100(2400Gbps)/H100/H800
| 2026-05-11 | F053 ✅ DONE: OXC 4端口修复— _mock_baseline_crosses min(2)→min(all), 每对Leaf 4×400Gbps cross; NS3 config 调优 (BUFFER_SIZE 512→PFC风暴修复, TP加速166x); 信息收集清单(Config+Topo完整参数表) 
| 2026-05-10 | F053 ✅ DONE: 运维工具线上部署 + PFC 风暴根因定位(BUFFER_SIZE=512 修复 TP 加速 166x) |
| 2026-05-09 | F053 ✅ DONE: 运维工具 — 调用记录(EDG+OXC-HCCL来源过滤) + 日志查询(关键词搜索) |
| 2026-05-08 | F051 ✅ DONE: 仿真进度条完成 — 84419 成功跑完 3412 streams，SIGSEGV 修复验证通过。DeepSeek-16B A2 结果：97% 通信（TP 瓶颈 200Gbps）| 仿真进度条 — results_routes.py progress/<pid> API 读取 workload 层数+log 当前层，计算百分比和预估时间；ResultsPage 加 PID 查询+进度条 UI |
| 2026-05-08 | F050 ✅ DONE: DeepSeek-16B 32GPU SIGSEGV 根因 — MockNcclGroup.cc:1658 if(rank_it->first) 漏 != -1 导致 rank 0 prevranks 为空→NcclTreeFlowModel crash。修复验证 44min+ 无崩 | DeepSeek-16B 32GPU SIGSEGV 调查 — PID 45325/44468 (A2, EP=4) 在 all-gather/mlp_moelayer 崩溃，lldb 下时序改变无法复现。新增错误反馈功能：前端显示中文错误标签+详情，后端 _signal_name 映射 |
| 2026-05-07 | F049: 修复 Dashboard 所有任务结果显示同一旧文件 — results_routes.py _find_ns3_result_files_in_workspace: prefix 改为 sim_result_, mtime 排序最新优先; ResultsPage.tsx: storedPrefix 不匹配时 fallback 到最新结果; Flask 重启加载新代码 |
| 2026-05-09 | F052 ✅ DONE: HomePage NPU_INTRA_MAP 补齐 A100(2400Gbps)/H100/H800
| 2026-05-11 | F053 ✅ DONE: OXC 4端口修复— _mock_baseline_crosses min(2)→min(all), 每对Leaf 4×400Gbps cross; NS3 config 调优 (BUFFER_SIZE 512→PFC风暴修复, TP加速166x); 信息收集清单(Config+Topo完整参数表) 
| 2026-05-10 | F053 ✅ DONE: 运维工具线上部署 + PFC 风暴根因定位(BUFFER_SIZE=512 修复 TP 加速 166x) |
| 2026-05-09 | F053 ✅ DONE: 运维工具 — 调用记录(EDG+OXC-HCCL来源过滤) + 日志查询(关键词搜索) |
| 2026-05-08 | F051 ✅ DONE: 仿真进度条完成 — 84419 成功跑完 3412 streams，SIGSEGV 修复验证通过。DeepSeek-16B A2 结果：97% 通信（TP 瓶颈 200Gbps）| 仿真进度条 — results_routes.py progress/<pid> API 读取 workload 层数+log 当前层，计算百分比和预估时间；ResultsPage 加 PID 查询+进度条 UI |
| 2026-05-08 | F050 ✅ DONE: DeepSeek-16B 32GPU SIGSEGV 根因 — MockNcclGroup.cc:1658 if(rank_it->first) 漏 != -1 导致 rank 0 prevranks 为空→NcclTreeFlowModel crash。修复验证 44min+ 无崩 | DeepSeek-16B 32GPU SIGSEGV 调查 — PID 45325/44468 (A2, EP=4) 在 all-gather/mlp_moelayer 崩溃，lldb 下时序改变无法复现。新增错误反馈功能：前端显示中文错误标签+详情，后端 _signal_name 映射 |
| 2026-05-07 | F049: Dashboard 结果页所有任务显示同一文件 — results_routes.py 用 started_at 重建精确 prefix，进程各自匹配结果文件不再 fallback 到随机文件；ResultsPage.tsx storedPrefix 不匹配时自动加载最新结果 |
| 2026-05-07 | F048 ✅ DONE: 修复 ns3_emitter NVSwitch→Leaf 非标准链路，恢复 GPU→Leaf 直连；补全 NPU_INTRA_BW_MAP (A100/H100/H800)；对齐延迟默认值到官方(0.000025ms/0.0005ms)。process_service 增加 _cleanup_ns3_outputs 防止 ghost result；AS_RESULT_PATH 环境变量使每次仿真输出独立文件。AstraSimNetwork.cc RESULT_PATH 改为 get_result_path() 支持 env 覆盖。LRA hooks 修复：feature_list.json dict→flat list，gate/stop 兼容三种格式。GPT-7B A2 200Gbps 实测通信占比 24%。|
| 2026-04-30 | T001: 仿真冒烟测试矩阵 — 4 modes smoke test (analytical/oxc × ns3/oxc), 4/4 pass (22s). feature_list 筛减 75→46 条(F009/F020删除, pendings合并). lra-test E2E 改为 opt-in | — GPU↔NVSwitch=16, NVSwitch↔Leaf=2, GPU↔Leaf=0 ✓. NS3 仿真 Expose TP=1.45% <10% ✓ (Linux workspace). topology 格式和路由正确 |(旧GUI/api_client/deploy)，修复 bin/ 符号链接冲突。76/76 backend pass + tsc pass。feature_list 去重 pending_tests(21条→删除) |
| 2026-04-30 | 文档: 编写 OEC-Sim 软件实现设计说明书（docs/OEC-Sim-Platform-Software-Design.md），参考 VTP 模板，1769 行 31 mermaid 图。覆盖 OXC-HCCL 适配、EDG 接入、前后端分离、结果可视化、DFX/FMEA/安全分析。全文去代码化，伪代码替换为文字描述+流程图。平台名 SimAI-OXC → OEC-Sim。发现 ranktable 生成未考虑 LLD groupid 问题（待修） |
| 2026-04-29 | F046: 修复 EDG ns3_emitter — GPU→Leaf 直连改为 NVSwitch→Leaf，消除 intra-server TP 走 NIC Switch 的 ECMP 混用。路由验证：intra=2跳/2400Gbps，cross=4跳/400Gbps，无 GPU→Leaf 直连 ✓ |
| 2026-04-29 | F046: 修复 EDG NS3 拓扑生成 — GPU→Leaf 直连改为 NVSwitch→Leaf，消除 intra-server TP 通信经 NIC Switch 的 ECMP 混用。验证：2-server 拓扑生成正确(7链路，无 GPU→Leaf 直连) |
| 2026-04-29 | F031-F045_fix: 多组网拆分, OXC 端到端, AIOB timing, Sys.cc 死锁, ncclFlowModel 重命名+prefix匹配+并发防护, 全部参数暴露, LRA gate/stale/Bash强化, 超时移除, progress/feature-list 格式重构, done 前置条件, F045 is_ns3 顺序 fix |
| 2026-04-27 | F015 线程安全, F007 HTTP 超时, F030 参数契约, 结果页分页, NS3 OXC 可观测 |
| 2026-04-22 | F030 process_service 参数契约修复 |

## 架构

- Dashboard: React + Flask, 前端 `dashboard/` :3000, 后端 `server/` :5001
- OXC: OxcAdapter → OxcHttpClient → OXC REST API (`AS_EDG_URL`)
- NS3: `SimAI_simulator_oxc`, topology→EDG emitter→`edg_topo_*`
- 构建: GCC 9.4 / Ubuntu 20.04, macOS 需 `-undefined dynamic_lookup`

## LRA 规则

| Rule | 触发条件 | 动作 |
|------|---------|------|
| Gate 1 | 无 in_progress feature | BLOCK |
| Gate 2 | 文件不在 feature.files 范围 | BLOCK |
| Gate 3 | 切 feature 前有 dirty 文件 | BLOCK |
| Stop 1 | 有未测试的修改文件 | BLOCK |
| Stop 2 | .lra_dirty 未清 | BLOCK |
| Stop 3 | done 且 passes=false | BLOCK |
| **Stop 5** | **in_progress 但 progress.md 5分钟未更新** | **BLOCK** |
| 2026-05-14 | EDG emitter .pyc fixed (dont_write_bytecode), full-mesh confirmed (9 8 0 1 36), LRA ;true removed, gate actually blocks |
| 2026-05-14 | F075: 67198 root cause — Dashboard workload generator produces 2-layer microAllReduce not 141-layer Llama-7B. Backend simulation OK (504K total, 2.7% TP). Frontend workload gen is the bug.
| 2026-05-14 | Backend confirmed OK: 141-layer AIOB workload with full-mesh topology = 2.7% TP comm, 504K total. Dashboard generates 2-layer micro workloads — frontend issue, not backend.
| 2026-05-14 | F076: OxcIntegration.cc curl FD leak on macOS — added circuit breaker after first connection failure |
| 2026-05-14 | F077: Fixed AS_SEND_LAT default from 6000 (6ms) to 6 (6us) — was 1000x too large, caused Dashboard simulations to be 18x slower than direct binary runs |
| 2026-05-15 | SimAI_simulator_oxc FIXED: OXC curl circuit breaker + AS_SEND_LAT default 6μs → 509K (was 9362K). 3 entry.h copies synced. E2E passed.
| 2026-05-18 | F079: Cleaned 2.7GB workspace simulation data for migration
| 2026-05-26 | 27770 GPT-13B ga=1 still running (25:09) |
| 2026-05-30 | F080-F085: SimAI-NS3 decoupling — FlowRecord, ImportFlows, flow file serialization, coupled test (555) ✅ |
| 2026-05-30 | F086 ✅ DONE: E2E decoupled test. 修复4个bug: (1) autoEnableFlowOutput() 实现缺失导致SIGSEGV at 0x0; (2) genAllReduceRingFlowModels 缺流文件写入; (3) exit(0)不刷新ofstream致末尾截断→加f.flush(); (4) ImportFlows空文件stoul抛异常→加try-catch。1792 flows输出到文件, coupled结果 total time=120.798 ✅ |
| 2026-05-30 | F086-F088 ✅ ALL DONE: SimAI-NS3 E2E 解耦完成！修复6个关键bug: (1) autoEnableFlowOutput实现缺失→SIGSEGV; (2) genAllReduceRingFlowModels缺流输出; (3) exit(0)不刷新ofstream; (4) ImportFlows 空文件stoul异常; (5) pending依赖检查导致0 flows注入; (6) flow_size vs maxPacketCount混用(WSize=4bytes非16384)。最终结果: 1792 flows注入, received bytes 5505024/node = coupled IDENTICAL! 涉及改动: MockNcclGroup.{h,cc}, entry.h, AstraSimNetwork.cc |
| 2026-05-30 | F088 验证: 2层ALLREDUCE训练 E2E对比 — 耦合 total time=120.798 (TP 83.44%), 解耦 received bytes 5,505,024/node IDENTICAL. replay模式缺EndToEnd.csv因SimAI stats路径被跳过, next: checkpoint补stats汇总 |
| 2026-06-01 | F088 cleanup: 移除 replay checkpoint 硬编码 EndToEnd.csv; replay 仅输出 nodeHash bytes。 |
| 2026-06-01 | 集中化流输出: getFlowModels() 一处迭代 result map, std::set 按 flow_id 去重, 覆盖所有 collective 类型不再逐个 gen* 打补丁。修复 /tmp 配置文件被 macOS 重启清掉导致 ReadConf 静默失败。1792 flows, E2E IDENTICAL。 |
| 2026-06-01 | Step 1-5 完成: layer_num+type/op/loopstate 加入流文件+FlowRecord+ImportFlows 读取; qp_finish 记录 last_flow_finish_ns; replay checkpoint 动态生成 EndToEnd.csv。bytes IDENTICAL。 |
| 2026-06-01 | 架构重构: loadFlowsFromFile() 预加载 flow_models 缓存, getFlowModels() 命中返回。SimAI event loop 完整保留 — 耦合 120.035 vs replay 118.735 (1.08% init I/O skew)。bytes IDENTICAL, CSV 结构 IDENTICAL。 |
| 2026-06-01 | 根因+修复: 1.08% 因 parent_flow_id/child_flow_id 未序列化→NcclTreeFlowModel 依赖链断裂。_writeFlowRecord +2 字段, loadFlowsFromFile +2 反序列化。3 agent 并行探索确认 boostedTick 不受 IO 影响。结果: 120.035=120.035 IDENTICAL。 |
| 2026-06-01 | 10 workload 矩阵: AR/AG/RS/A2A 全覆盖, 1/2/3层, 多TP, 大64MB。loadFlowsFromFile parent/child 解析补全。10/10 PASS, 全部逐字节一致。 |
| 2026-06-02 | AICB deep-interview: 3 agent 探索 AICB 全貌→补 DP+EP 测试。DP=4(TP=2,WG=ALLREDUCE,448 flows) PASS。EP=2(TP=4,ALLTOALL_EP,296 flows) PASS。Tree/NVLS 在 A100 上不触发, 流写路径相同已覆盖。覆盖 TP/DP/EP + AR/AG/RS/A2A。 |
