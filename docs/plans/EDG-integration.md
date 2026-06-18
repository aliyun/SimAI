# 方案：EDG 组件集成（组网初始化 + AI 训练任务粗调节）

## 背景

SimAI 目前使用预构建的 NS3 拓扑文件（如 `Spectrum-X_16g_8gps_100Gbps_A100`），Dashboard 向导直接从磁盘选取。现在需要接入外部 **EDG** 服务来管理 OXC 光交叉连接状态，新增两个流程：

1. **新建组网** — 用户上传 `lld.json` → SimAI 转发给 EDG（`IMPORT_FULL_TOPO`）→ EDG 返回初始 OXC 端口交叉 → SimAI 存储为该组网的基线 OXC 状态
2. **新建 AI 训练任务** — 用户上传 `npu_match.json`（哪些服务器/NPU 参与）→ SimAI 转发给 EDG（`NOTIFY_NODE_MATRIX`）→ EDG 返回增量 del/add 交叉 patch → SimAI 将 patch 合并到基线，解析 leaf↔leaf 路径，生成 NS3 格式拓扑文件，交给现有 SimAI 启动流程

EDG **已在运行**：`http://127.0.0.1:9000/api/port_allocation`，单一 endpoint，通过 `message_type` 区分。用户已确认真实响应。Mock 仅用于 CI / 离线开发。

## 确定范围（路径 A 简版）

- EDG 任务注册**仅输出子拓扑**（`edg_topo_<task_id>`）。Ranktable 继续走用户现有上传流程。
- 融合器按 `npu_match.json` 中服务器出现顺序，将参与服务器排为 NS3 拓扑的**前 N 个** GPU 节点（符合 SimAI "取前 N 个 rank" 的约定）。不参与的服务器、leaf、OXC cross 全部排除。
- `npu_id != -1` 本轮视为"整台服务器参与"，后续再做单卡粒度。

---

## EDG 接口协议（已确认）

**POST** `http://127.0.0.1:9000/api/port_allocation`

### `IMPORT_FULL_TOPO` — 初始化组网
- 请求：完整 `lld.json`（含 `version`, `request_id`, `message_type`, `topology:{oxc_nodes, server_nodes, leaf_nodes, spine_nodes, edges, full_oxc_crosses}`）
- 响应：`{request_id, message_type, result:"OUTPUT_OXCCROSS", oxc_oper_orders:{del_oper_info_list:[...], add_oper_info_list:[{node_ip, a_port_id, b_port_id},...]}}`

### `NOTIFY_NODE_MATRIX` — 训练任务注册（增量）
- 请求：`{version, request_id, message_type:"NOTIFY_NODE_MATRIX", npu_matrix:[{inst_key:{task_id, k8s_task_id, inst_id, inst_type}, npu_set:[{server_ip, npu_id}]}]}`
- 成功：`{request_id, message_type, result:"OUTPUT_OXCCROSS", oxc_oper_orders:[{del_oper_info_list:[...], add_oper_info_list:[...]},...]}`（多批次数组）
- 失败：`{request_id, message_type, result:"OUTPUT_ERROR"}`

**交叉语义** — `{node_ip:OXC_IP, a_port_id, b_port_id}` 表示"在该 OXC 内部，将端口 a 和端口 b 光学连通"。结合 `lld.json:edges`（OXC 端口→leaf 端口、leaf 端口→server 端口的固定光纤），可推导出有效的 leaf↔leaf 路径，用于构建 NS3 拓扑。

---

## 设计

### 后端 — 新建 `server/edg/` 包

| 文件 | 职责 |
|------|------|
| `server/edg/__init__.py` | Blueprint 导出 |
| `server/edg/edg_client.py` | HTTP 客户端封装。环境变量：`AS_EDG_URL`（默认 `http://127.0.0.1:9000/api/port_allocation`）、`AS_EDG_TIMEOUT`（30s）、`AS_EDG_MOCK=1` 离线模式。公开方法：`import_full_topo(lld)`、`notify_node_matrix(npu_match)`。`OUTPUT_ERROR` 或网络异常时抛 `EdgError`。 |
| `server/edg/crosses.py` | 纯函数。`apply_patch(base_crosses, del_list, add_list) → new_crosses`。`apply_batches(base, batches)` 处理多批次任务 patch。 |
| `server/edg/merger.py` | `resolve_paths(lld, cross_set)` — 遍历 `lld.topology.edges` 固定光纤 + OXC 交叉 → 输出图结构 `{servers, leaves, server_leaf_edges, leaf_leaf_edges}`。 |
| `server/edg/ns3_emitter.py` | `write_ns3_topology(graph, out_path, params)` — 输出 `gen_Topo_Template.py` 同款文本格式：第1行 `<nodes> <gpu_per_server> <nv_switch_num> <switch_nodes> <links> <gpu_type>`，第2行交换机 ID，后续每行 `src dst bw latency err`。NVSwitch 层合成（lld 无 NV 信息）；默认参数从环境变量读取。 |
| `server/edg/routes.py` | Flask blueprint `url_prefix="/api/edg"`，复用 `@require_auth` + `save_file(request.workspace_dir, ...)`（同 [server/app.py:99-129](server/app.py#L99) 和 [server/workspace/workspace_service.py:49-62](server/workspace/workspace_service.py#L49) 的模式）。 |
| `server/edg/tests/test_crosses.py` | 单元测试：patch 幂等性、去重、顺序无关性 |
| `server/edg/tests/test_merger.py` | 用 `lld.json` + 手工构造的 cross set → 断言预期 leaf-leaf 边 |

### 路由

**`POST /api/edg/init`** — 初始化组网
```
body: { lld: <lld.json 对象> }
→ 保存 lld.json 到 workspace ($WS/lld.json)
→ edg_client.import_full_topo(lld)
→ 保存 EDG 响应到 $WS/edg/init_crosses.json
→ 调用 scripts/lld_to_topology.py 生成监控大屏用的 XML
→ 返回 { network_id, lld_path, crosses_count, oxc_count }
```

**`POST /api/edg/register-task`** — 注册训练任务
```
body: { npu_match: <对象>, task_id: "T001" }
→ 保存到 $WS/edg/tasks/<task_id>/npu_match.json
→ 加载 $WS/edg/init_crosses.json（不存在则报错"请先调用 /api/edg/init"）
→ edg_client.notify_node_matrix(npu_match)
→ merged_crosses = crosses.apply_batches(init_crosses, response.oxc_oper_orders)
→ 保存到 $WS/edg/tasks/<task_id>/merged_crosses.json
→ graph = merger.resolve_paths(lld, merged_crosses)  # 仅含参与服务器的子图
→ out_name = f"edg_topo_{task_id}"
→ ns3_emitter.write_ns3_topology(graph, $WS/<out_name>, ...)
→ 返回 { task_id, topology_file: out_name, graph_stats }
```

在 [server/app.py](server/app.py) 注册 blueprint（一行代码）。

### 前端

| 文件 | 改动 |
|------|------|
| `dashboard/src/api/edg-api.ts` | 新建 — `initNetwork(lld)` 和 `registerTask(npuMatch, taskId)`，复用 `ApiResponse` 类型 |
| `dashboard/src/types/edg.ts` | 新建 — `EdgCross`、`EdgInitResponse`、`EdgTaskResponse` 类型定义 |
| [TopologyPage.tsx](dashboard/src/pages/TopologyPage.tsx) | 在目录扫描框上方新增"导入 lld.json"文件选择器。选择后：`FileReader → JSON.parse → initNetwork(lld) → 提示"EDG 初始化完成，OXC 交叉数: N" → 刷新 scanTopologyDir()`。现有扫描 UI 不变。 |
| [LaunchPage.tsx](dashboard/src/pages/LaunchPage.tsx) | 在组网下拉框上方新增可折叠的"AI 训练任务调度 (EDG)"区域。包含：task_id 输入框（默认 `T001`）、npu_match.json 文件选择器。如果选了文件，`handleLaunch` 先调 `registerTask`，用返回的 `topology_file` 替换 `launchConfig.topologyPath`。未选文件则走原有流程。 |
| [wizard-store.ts](dashboard/src/stores/wizard-store.ts) | 新增字段 `npuMatchFile?`、`taskId?`、`edgTopologyPath?` |

### 监控大屏联动

**现状**：监控大屏每 5 秒轮询 `/api/topology/pod/<pod_id>` → 后端读 `topology/pods/<pod_id>.xml` → 前端用 ReactFlow 渲染节点和边。XML 由 `lld_to_topology.py` 生成，边 label 是占位 `0.00%~0.00%`。

**EDG 任务注册后 XML 需要更新**，原因：
1. OXC 交叉状态变了 — EDG 返回的 `add_oper_info_list` 表示哪些 OXC 端口被光学连通，对应 leaf↔leaf 之间有了有效路径。大屏应该**新增 leaf-leaf 逻辑连线**（虚线高亮），让用户看到"粗调后哪些 leaf 之间通了"。
2. 参与任务的服务器应被标记 — `npu_match.json` 指定的 SERVER 节点需要视觉区分（颜色/标签标注 task_id）。

**实现方式**：
- 扩展 [scripts/lld_to_topology.py](scripts/lld_to_topology.py) 新增函数 `generate_pod_xml_with_crosses(lld_data, crosses, npu_match, pod_id)`：
  - 参与任务的 SERVER 节点：`fillColor=#d5e8d4`（绿色背景）+ label 追加 `[T001]`
  - OXC 交叉激活的 leaf-leaf 路径：新增虚线边（`dashed=1;strokeColor=#FF6600`），label 标注端口对
  - 未激活的 OXC 端口对应的边保持原样（灰色）
- `POST /api/edg/register-task` 路由在生成 NS3 拓扑后，**同时重新生成** `topology/pods/<pod_id>.xml`（覆盖写入）
- 大屏 5 秒轮询自动拿到新 XML，无需前端改动
- `POST /api/edg/init` 路由生成的是基线 XML（无高亮），和现在一样

| 新增/修改文件 | 改动 |
|--------------|------|
| [scripts/lld_to_topology.py](scripts/lld_to_topology.py) | 新增 `generate_pod_xml_with_crosses()` 函数 |
| `server/edg/routes.py` | `register-task` 路由末尾调用该函数重新生成 pod XML |

### 融合 / NS3 输出算法

1. 索引 `lld.json:edges` 建两张映射表：
   - `oxc_port_to_leaf[(oxc_ip, port)] = (leaf_ip, leaf_port)` — OXC 端口连到哪个 leaf
   - `leaf_port_to_server[(leaf_ip, port)] = (server_ip, server_port)` — leaf 端口连到哪台服务器
2. 对每个 OXC 交叉 `(oxc_ip, {a, b})`，通过 `oxc_port_to_leaf` 解析出 `(leaf_a, leaf_b)`。两端都可达 → 添加 leaf-leaf 边；悬空端口 → 跳过并记日志。
3. **子图裁剪**：仅保留 `npu_match` 中列出的服务器及其上行 leaf 和相关 OXC cross。
4. NS3 节点编号（与 `gen_Topo_Template.py` 一致）：`0..G-1` GPU，然后 NVSwitch，然后 leaf/ASW 交换机。
5. 每台服务器展开为 `gpu_per_server` 个 GPU；GPU→leaf 连线带 `bandwidth/latency/error_rate`；GPU→NVSwitch 连线用 NVLink 带宽。

---

## 需要修改的关键文件

- **新建**：`server/edg/*`（6 个文件 + 2 个测试）
- **新建**：`dashboard/src/api/edg-api.ts`、`dashboard/src/types/edg.ts`
- **修改**：[server/app.py](server/app.py) — 注册 blueprint（一行）
- **修改**：[TopologyPage.tsx](dashboard/src/pages/TopologyPage.tsx) — 添加 lld.json 上传
- **修改**：[LaunchPage.tsx](dashboard/src/pages/LaunchPage.tsx) — 添加任务注册前置步骤
- **修改**：[wizard-store.ts](dashboard/src/stores/wizard-store.ts) — 新增字段
- **修改**：[scripts/lld_to_topology.py](scripts/lld_to_topology.py) — 新增 `generate_pod_xml_with_crosses()` 函数（监控大屏联动）
- **复用**：[workspace_service.py:save_file](server/workspace/workspace_service.py#L49)、[process_service.py:113-117](server/process/process_service.py#L113)（topology_file → -n 参数）

## 验证方案

1. **单元测试** — `cd server && python3 -m pytest edg/tests/` — crosses patch 数学正确性 + merger 路径解析
2. **集成测试（真实 EDG）** — 直接调 `import_full_topo(lld.json)` 确认返回 `OUTPUT_OXCCROSS`
3. **监控大屏联动** — 调 `register-task` 后，检查 `topology/pods/<pod_id>.xml` 是否包含：参与 SERVER 节点有绿色 `fillColor`、leaf-leaf 虚线边存在且 `strokeColor=#FF6600`。打开大屏确认 5 秒内自动刷新显示高亮。
4. **E2E** — 启动 server + dashboard，在 TopologyPage 上传 `lld.json`，确认监控大屏渲染 POD；在 LaunchPage 上传 `npu_match.json`（task_id=T001），点击启动，确认 `$WORKSPACE/edg_topo_T001` 存在且 NS3 二进制以 `-n edg_topo_T001` 启动
5. **类型检查** — `cd dashboard && npx tsc --noEmit`
6. **回归** — 不上传 EDG 文件、直接选已有组网启动，流程不受影响；监控大屏在无 EDG 任务时显示基线拓扑（无高亮）

## 本轮不做

- 多 POD EDG（当前 lld 只有 1 个 OXC）
- `npu_id != -1` 的单卡粒度（本轮视为整台服务器参与）
- 仿真运行中 EDG 状态变更后自动重新生成拓扑（仅启动时一次性生成）
- Ranktable 自动生成（继续走用户现有上传流程）
