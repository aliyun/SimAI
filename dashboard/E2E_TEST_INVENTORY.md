# SimAI Dashboard - Complete E2E Test Inventory

## OVERVIEW
**Project**: OCS-Sim (Optical Circuit Switched Simulator) Dashboard  
**Tech Stack**: React, TypeScript, Zustand (state management), Recharts (charting), ReactFlow (topology visualization)  
**Routes**: 8 pages, 40+ interactive components, 3 API services  
**Architecture**: Multi-step wizard (Workload → RankTable → Launch → Results)

---

## 1. PAGES (URL Mappings & User Interactions)

### 1.1 HomePage (`/`)
**URL**: `/`  
**Component**: `HomePage.tsx`  
**Purpose**: Main landing page with network management and deployment wizard navigation

#### Interactive Elements:
1. **Create Network Modal**
   - Trigger: "+ 新建组网" button (in Network Management section)
   - Fields:
     - Text input: "组网名称" (network name)
     - Text input: "拓扑目录" (topology directory path)
   - Actions:
     - Button: "取消" (close modal)
     - Button: "创建" (create network - disabled if no inputs)
   - State: `showCreateModal` (boolean), `newName`, `newDir`, `creating`, `createError`
   - API Calls: `setTopologyDir()`, `createNetwork()`

2. **Network List**
   - Display: List of existing networks with name and path
   - Empty State: "暂无组网，点击下方创建" message
   - Per-network actions:
     - Click network name → `handleSelectNetwork()` → navigate to `/networks/{id}`
     - Hover delete button → `deleteNetwork()` on click

3. **Navigation Buttons** (3 sections):
   - Section 1 (Workload): Link to `/deploy/workload`
   - Section 2 (RankTable): Link to `/deploy/ranktable`
   - Section 3 (Results): Link to `/results` and `/monitor`

#### State Management:
- Zustand stores: `useNetworkStore()`, `useNavigate()`
- Local state: `showCreateModal`, `newName`, `newDir`, `creating`, `createError`

#### API Calls:
- `setTopologyDir(topologyDir)` - POST
- `createNetwork({name, topologyDir})` - local store
- `deleteNetwork(id)` - local store

---

### 1.2 NetworkDetailPage (`/networks/:id`)
**URL**: `/networks/:id`  
**Component**: `NetworkDetailPage.tsx`  
**Purpose**: View network topology and configuration files with real-time monitoring

#### Interactive Elements:

1. **Header Section**
   - Network name (editable):
     - Click to enter edit mode
     - Text input for new name
     - "保存" (save) button
     - "取消" (cancel) button
   - Topology directory path (display only)

2. **Tab Buttons**
   - "监控" (Monitor) - tab
   - "拓扑文件" (Files) - tab
   - State: `activeTab` ('monitor' | 'files')

3. **Metrics Bar** (visible in Monitor tab):
   - Display cards:
     - Total PODs: `clusterSummary?.totalPods`
     - Total GPUs: `clusterSummary?.totalGpus`
     - Utilization: `${(overallUtilization * 100).toFixed(1)}%`
     - Active Alerts: count with warning status

4. **Monitor Tab**:
   - ReactFlow network visualization
   - Node click → navigate to `/monitor/pod/{podId}`
   - Nodes: PodNode components (clickable)
   - Edges: MetricEdge components

5. **Files Tab**:
   - File list display with type badges (DIR/FILE)
   - File size display in KB
   - "刷新" (refresh) button (becomes "扫描中..." while scanning)
   - Empty state: "点击刷新扫描拓扑目录"

#### State Management:
- Zustand: `useNetworkStore()`, `useTopologyStore()`
- Local state: `activeTab`, `files`, `isScanning`, `scanError`, `editingName`, `editName`
- Polling hook: `usePolling()` with 5000ms interval for Monitor tab

#### API Calls:
- `setTopologyDir(topologyDir)` - POST (sync when switching to monitor)
- `fetchOverviewTopology()` - GET (polled every 5s in monitor tab)
- `fetchClusterSummary()` - GET (polled every 5s)
- `scanTopologyDir()` - GET (on files tab entry and refresh click)
- `updateNetwork(id, {name})` - local store

---

### 1.3 WorkloadPage (`/deploy/workload`)
**URL**: `/deploy/workload`  
**Component**: `WorkloadPage.tsx`  
**Purpose**: Configure or upload workload file with preset/custom/upload modes

#### Interactive Elements:

1. **Mode Selector** (3 buttons):
   - "Preset Model" - use LLaMA/GPT presets
   - "Custom Layers" - define custom config
   - "Upload File" - upload existing file
   - State: `workloadMode` ('preset' | 'custom' | 'upload')

2. **Preset Mode Fields**:
   - Select: Model Size (options: 7B, 13B, 70B, 175B)
   - Input: Total GPUs (number, min=1)
   - Input: TP Size (Tensor Parallelism, number, min=1)
   - Input: DP Size (Data Parallelism, number, min=1)
   - Input: PP Size (Pipeline Parallelism, number, min=1)
   - Input: EP Size (Expert Parallelism, number, min=1)

3. **Custom Mode Fields**:
   - Input: TP Size
   - Input: DP Size
   - Input: PP Size
   - Input: Total GPUs

4. **Upload Mode**:
   - FileUploadZone component (accepts .txt, .workload)
   - Drag-and-drop or click to upload
   - Displays filename when uploaded

5. **Preview**:
   - CodePreview component showing workload content (max-height: 250px)

6. **Action Buttons**:
   - "下一步 →" (Next) button - validates and generates workload, navigates to `/deploy/ranktable`
   - Loading state: "处理中..." (Processing)

7. **Validation**:
   - ValidationBanner for errors
   - ValidationBanner for success messages

#### State Management:
- Zustand: `useWizardStore()`
- Local state: `isLoading`, `error`, `successMsg`

#### API Calls:
- `generatePresetWorkload(workloadConfig)` - POST (returns content + layers)
- `generateCustomWorkload({tp_size, dp_size, pp_size, ep_size, all_gpus, layers_config})` - POST
- `saveFile('workload.txt', content)` - POST (saves and returns path)

---

### 1.4 RankTablePage (`/deploy/ranktable`)
**URL**: `/deploy/ranktable`  
**Component**: `RankTablePage.tsx`  
**Purpose**: Generate or upload RankTable (GPU rank to rack mapping)

#### Interactive Elements:

1. **Mode Selector** (3 buttons):
   - "Auto Generate" - generate from GPU count and rack size
   - "Custom Topology" - manual mapping
   - "Upload File" - upload ranktable.json

2. **Auto Mode Fields**:
   - Input: Total Ranks (GPUs)
   - Input: GPUs per Rack
   - Input: Rack Prefix
   - Display (read-only): Computed Racks (calculated as `Math.ceil(rank_count / gpus_per_rack)`)

3. **Upload Mode**:
   - FileUploadZone (accepts .json)

4. **Preview**:
   - CodePreview showing JSON (max-height: 300px)

5. **Action Buttons**:
   - "下一步 →" (Next) button - generates/validates ranktable, saves, navigates to `/deploy/launch`
   - Loading state: "处理中..."

6. **Validation**:
   - ValidationBanner for errors
   - ValidationBanner for warnings (validation errors list)

#### State Management:
- Zustand: `useWizardStore()`
- Local state: `isLoading`, `error`, `successMsg`, `validationErrors`

#### API Calls:
- `generateRanktable(ranktableConfig)` - POST (returns ranktable + rank_rack_map)
- `validateRanktable(ranktableData)` - POST (returns {valid, errors[]})
- `saveFile('ranktable.json', JSON.stringify(data, null, 2))` - POST

---

### 1.5 LaunchPage (`/deploy/launch`)
**URL**: `/deploy/launch`  
**Component**: `LaunchPage.tsx`  
**Purpose**: Configure and launch simulation with Analytical or NS-3 mode

#### Interactive Elements:

1. **Network Selector**:
   - Select dropdown (disabled if running)
   - Shows: "{name} ({topologyDir})"
   - Option: "-- 请选择组网 --" (placeholder)
   - Display: Active network topology dir

2. **Configuration Fields**:
   - Select: Simulation Mode ("Analytical (快速)" | "NS3 Simulation (详细)")
   - Input: Threads (number, min=1, max=64, hint: NS3 recommends 8-16, disabled if running)
   - Input: Workload Path (text, disabled if running)
   - Input: RankTable Path (text, disabled if running)
   - Input: Topology Path (text, disabled if running)

3. **Pre-filled Status Boxes**:
   - Workload box: Shows saved path or "未配置 — 请先在 Workload 步骤保存"
   - RankTable box: Shows saved path or "未配置 — 请先在 RankTable 步骤保存"

4. **Launch/Kill Buttons**:
   - "启动仿真" (Launch) - if not running (disabled if no network selected or loading)
     - Loading state: "启动中..."
   - "停止仿真" (Kill) - if running (no disabled state)

5. **Success Banner**:
   - Shows "仿真完成！" with link to `/results`

6. **ProcessList Component**:
   - Displays running processes with:
     - Status badge (running/other)
     - PID in monospace
     - Command (truncated)
     - Kill button (if running)

7. **LogViewer Component**:
   - Shows simulation output logs
   - Auto-scrolls to bottom
   - Status indicator (running/exited/other)

#### State Management:
- Zustand: `useWizardStore()`, `useNetworkStore()`
- Local state: `isLaunching`, `error`
- Hook: `useLogStream({pid, enabled})` - returns `{logs, status}`

#### API Calls:
- `launchSimulation({binary, workload_file, ranktable_file, topology_file, env_vars, extra_args})` - POST
- `listProcesses()` - GET (polled every 5s)
- `fetchProcessLogs(pid)` - GET
- `killProcess(pid)` - DELETE

---

### 1.6 ResultsPage (`/results`)
**URL**: `/results`  
**Component**: `ResultsPage.tsx`  
**Purpose**: Analyze simulation results with multiple visualization tabs

#### Interactive Elements:

1. **Task List Section** (if no data loaded):
   - Displays completed tasks with:
     - Tracking ID badge (if available)
     - PID display (if available)
     - Task label (extracted from result_prefix or binary name)
     - Timestamp
     - Result file name
   - Click task → `handleLoadTask()` → loads EndToEnd data
   - Loading state: "加载中..."
   - Empty state: "暂无已完成的仿真任务"

2. **Toggle Button**:
   - "或者手动上传文件 ↓" / "收起手动上传"
   - Toggles `showUpload` state

3. **Manual Upload Section** (if showUpload):
   - Column 1: EndToEnd CSV
     - FileUploadZone (accepts .csv)
     - Call: `handleEndToEndUpload()`
   - Column 2: Console Output
     - FileUploadZone (accepts .txt, .log)
     - Call: `handleConsoleUpload()`

4. **Tab Bar** (if data loaded):
   - Tabs: Overview, Layer Timing, Bandwidth, Node Transfer
   - "← 返回任务列表" button to clear data

5. **Overview Tab**:
   - OverviewMetrics component
   - ComputeCommBreakdown chart (if data available)
   - DimensionBreakdown chart (if data available)

6. **Layer Timing Tab**:
   - LayerTimingChart component (if endtoendData.layers exists)
   - Empty state: "暂无 Layer Timing 数据"

7. **Bandwidth Tab**:
   - BandwidthChart component (if endtoendData.layers exists)
   - Empty state: "暂无 Bandwidth 数据"

8. **Node Transfer Tab**:
   - NodeTransferChart component (if consoleData exists)
   - Empty state: "暂无 Node Transfer 数据 — 请上传 Console Output 文件"

#### State Management:
- Zustand: `useWizardStore()`
- Local state: `activeTab`, `error`, `isLoading`, `tasks`, `tasksLoading`, `showUpload`

#### API Calls:
- `fetchCompletedTasks()` - GET (on mount)
- `loadResultByFilepath(filepath)` - POST
- `parseEndToEnd(content)` - POST
- `parseConsoleOutput(logLines)` - POST

---

### 1.7 PodDetailPage (`/monitor/pod/:podId`)
**URL**: `/monitor/pod/:podId`  
**Component**: `PodDetailPage.tsx`  
**Purpose**: Detailed network topology view of a specific pod/cluster

#### Interactive Elements:

1. **Header**:
   - Title: "{podId} Detail"
   - Back link: "/monitor"
   - Last updated timestamp

2. **Metrics Bar**:
   - Servers count
   - Spines count
   - Leaves count
   - Links count

3. **Network Visualization** (ReactFlow):
   - Nodes: NetworkDeviceNode (clickable)
   - Edges: MetricEdge (clickable)
   - Click node → `onNodeClick()` → toggles selection
   - Click edge → `onEdgeClick()` → toggles selection
   - Deselect if clicking same item again

4. **Node Detail Panel** (right sidebar):
   - Shows when node selected
   - Displays: nodeId, GPU Utilization, CPU Utilization, Memory, Temperature, Power, Status
   - Close button
   - Polling every 3000ms for metrics

5. **Edge Detail Panel** (right sidebar):
   - Shows when edge selected
   - Header: "{source} → {target}"
   - Summary: Ports count, Avg In%, Avg Out% (if ports available)
   - Chart: Horizontal bar chart showing In Rate / Out Rate per port
   - Close button

#### State Management:
- Zustand: `useTopologyStore()`
- Polling: `usePolling()` for pod detail every 5s

#### API Calls:
- `fetchPodDetail(podId)` - GET (polled every 5s)
- `fetchNodeMetrics(nodeId)` - GET (from NodeDetailPanel, polled every 3s)

---

## 2. COMPONENTS INVENTORY

### 2.1 Layout Components
- **DashboardShell**: Main layout wrapper
- **Header**: Page header with title, back link, last updated display

### 2.2 Wizard Components
- **WizardLayout**: Container for wizard steps
- **ModeSelector**: Generic mode selection (3-button layout)
- **FormField**: Labeled input wrapper with optional hint
- **FileUploadZone**: Drag-drop file upload with click fallback
- **CodePreview**: Read-only code viewer with monospace font
- **ValidationBanner**: Error/warning/success message banner
- **DeployStepper**: Progress indicator for deployment steps
- **WizardStepper**: Generic stepper component

### 2.3 Chart Components
- **OverviewMetrics**: Summary metric cards
- **LayerTimingChart**: Stacked bar chart of layer timing
- **BandwidthChart**: Bandwidth visualization by layer
- **ComputeCommBreakdown**: Pie chart of compute vs communication
- **DimensionBreakdown**: Breakdown by dimensions
- **NodeTransferChart**: Node transfer visualization

### 2.4 Widget Components
- **MetricCard**: Single metric display card
- **LogViewer**: Scrollable log output display (auto-scroll to bottom)
- **ProcessList**: Running process list with kill buttons
- **NodeDetailPanel**: Right sidebar showing node metrics
- **EdgeDetailPanel**: Right sidebar showing edge metrics with port-level bar chart

### 2.5 Graph Components
- **PodNode**: Pod visualization node (ReactFlow)
- **NetworkDeviceNode**: Network device node (ReactFlow)
- **SuperNodeBoundary**: Boundary box for super nodes (ReactFlow)
- **MetricEdge**: Edge visualization (ReactFlow)

### 2.6 Edge Components
- **MetricEdge**: Custom edge with metrics display

---

## 3. STATE MANAGEMENT (Zustand Stores)

### 3.1 Network Store (`useNetworkStore`)
**Purpose**: Manage network configurations  
**Persistent**: Yes (localStorage: 'ocs-sim-networks')

**State**:
```
networks: Network[]
activeNetworkId: string | null
```

**Actions**:
- `createNetwork({name, topologyDir})` → returns Network
- `updateNetwork(id, {name?, topologyDir?})`
- `deleteNetwork(id)`
- `setActiveNetwork(id)`
- `getNetwork(id)` → Network | undefined

---

### 3.2 Wizard Store (`useWizardStore`)
**Purpose**: Multi-step wizard state  
**Persistent**: No (in-memory only)

**State Sections**:

**Navigation**:
- `currentStep: WizardStep`
- `completedSteps: Set<WizardStep>`

**Workload**:
- `workloadMode: 'preset' | 'custom' | 'upload'`
- `workloadConfig: {model_size, tp_size, dp_size, pp_size, ep_size, all_gpus}`
- `workloadContent: string`
- `workloadLayers: WorkloadLayer[]`
- `workloadSaved: boolean`
- `workloadSavedPath: string`

**RankTable**:
- `ranktableMode: 'auto' | 'custom' | 'upload'`
- `ranktableConfig: {rank_count, gpus_per_rack, superpod_prefix}`
- `ranktableData: RankTable | null`
- `rankRackMap: RankRackMap | null`
- `ranktableSaved: boolean`
- `ranktableSavedPath: string`

**Launch**:
- `launchConfig: {mode, workloadPath, ranktablePath, topologyPath, threads, envVars}`
- `activePid: number | null`
- `simulationStatus: 'idle' | 'running' | 'completed' | 'failed'`

**Results**:
- `endtoendData: EndToEndData | null`
- `consoleData: ConsoleData | null`

**Actions**: setters for each state section, `completeStep()`, `goToStep()`, `reset()`

---

### 3.3 Topology Store (`useTopologyStore`)
**Purpose**: Network topology visualization state

**State**:
- `overview: PodOverviewTopology | null`
- `podDetail: PodDetailTopology | null`
- `clusterSummary: ClusterSummary | null`
- `selectedNodeId: string | null`
- `selectedEdgeId: string | null`
- `isLoading: boolean`
- `error: string | null`
- `lastUpdated: number | null`

**Actions**:
- `setOverview()`, `setPodDetail()`, `setClusterSummary()`
- `selectNode()`, `selectEdge()`
- `setLoading()`, `setError()`

---

## 4. API CALLS (Complete Catalog)

### 4.1 Simulation API (`/api/simulation/*`)

**Workload Endpoints**:
- `POST /api/simulation/workload/generate-preset` → `{content, layers, model_size}`
- `POST /api/simulation/workload/generate-custom` → `{content}`
- `POST /api/simulation/workload/parse` → `{config}`
- `GET /api/simulation/workload/presets` → `{presets: Record<string, ModelPreset>}`

**RankTable Endpoints**:
- `POST /api/simulation/ranktable/generate` → `{ranktable, rank_rack_map}`
- `POST /api/simulation/ranktable/generate-custom` → `{ranktable, rank_rack_map}`
- `POST /api/simulation/ranktable/validate` → `{valid, errors[]}`

**Topology Dir Endpoints**:
- `POST /api/simulation/topology-dir` → `{path}`
- `GET /api/simulation/topology-dir` → `{path}`
- `GET /api/simulation/topology-dir/scan` → `{path, files[]}`

**Results Endpoints**:
- `POST /api/simulation/results/parse-endtoend` → `EndToEndData`
- `POST /api/simulation/results/parse-console` → `ConsoleData`
- `GET /api/simulation/results/find-files` → `{files: Record<string, string>}`
- `GET /api/simulation/results/list-tasks` → `{tasks: CompletedTask[]}`

**File Operations**:
- `POST /api/files/save` → `{path, filename}`
- `GET /api/files/load` → `{content, path, filename}`

**Process Operations**:
- `POST /api/process/launch` → `{tracking_id, pid, status}`
- `GET /api/process/list` → `{processes: ProcessEntry[]}`
- `GET /api/process/logs/{pid}` → `{logs[], status, pid}`
- `DELETE /api/process/{pid}`

---

### 4.2 Topology API (`/api/topology/*`)

- `GET /api/topology/overview` → `PodOverviewTopology`
- `GET /api/topology/pod/{podId}` → `PodDetailTopology`

---

### 4.3 Metrics API (`/api/metrics/*`)

- `GET /api/metrics/cluster` → `ClusterSummary`
- `GET /api/metrics/node/{nodeId}` → `NodeLiveMetrics`

---

## 5. INTERACTION FLOWS

### 5.1 Create Network Flow
1. HomePage: Click "+ 新建组网" button
2. Modal opens: user enters name and topology directory
3. Click "创建" button
4. `setTopologyDir()` API call
5. `createNetwork()` Zustand action
6. Auto-navigate to `/networks/{id}`

### 5.2 Edit Network Name Flow
1. NetworkDetailPage: Click on network name (editable)
2. Shows input field with "保存" and "取消" buttons
3. User types new name
4. Click "保存"
5. `updateNetwork(id, {name})` Zustand action
6. Exit edit mode

### 5.3 Deployment Wizard Flow
1. HomePage → Click "Workload 配置" link → `/deploy/workload`
2. WorkloadPage: Select mode (preset/custom/upload)
3. Fill configuration or upload file
4. Click "下一步 →"
5. Generates workload, saves to file, navigates to `/deploy/ranktable`
6. RankTablePage: Select mode (auto/custom/upload)
7. Configure or upload ranktable
8. Click "下一步 →"
9. Generates/validates ranktable, saves, navigates to `/deploy/launch`
10. LaunchPage: Select network, configure simulation
11. Click "启动仿真"
12. Simulation launches, shows logs in real-time
13. On completion: "查看仿真结果 →" link to `/results`

### 5.4 Monitor Network Flow
1. HomePage → NetworkDetailPage (click network)
2. Click "监控" tab
3. Polls `fetchOverviewTopology()` every 5s
4. Shows network topology with pods as nodes
5. Click pod node → navigate to `/monitor/pod/{podId}`
6. PodDetailPage: Shows detailed topology with network devices
7. Click device node → shows NodeDetailPanel with metrics (polled every 3s)
8. Click edge → shows EdgeDetailPanel with port-level analytics

### 5.5 View Results Flow
1. `/results` page loads
2. Fetches completed tasks list
3. Click task → loads EndToEnd CSV data
4. Select tab (Overview/Layers/Bandwidth/Nodes)
5. Or: Click "或者手动上传文件 ↓"
6. Upload EndToEnd CSV or Console Output
7. Parse and visualize

---

## 6. FORM FIELDS & INPUTS (Complete Catalog)

### HomePage
- Text input: Network name (placeholder: "例如: Spectrum-X-128G")
- Text input: Topology directory (placeholder: "/path/to/topology/directory")

### WorkloadPage
- Select: Model Size (options: 7B, 13B, 70B, 175B)
- Input: Total GPUs (type: number, min: 1)
- Input: TP Size (type: number, min: 1)
- Input: DP Size (type: number, min: 1)
- Input: PP Size (type: number, min: 1)
- Input: EP Size (type: number, min: 1)
- File upload: Workload file (accept: .txt, .workload)

### RankTablePage
- Input: Total Ranks (type: number, min: 1)
- Input: GPUs per Rack (type: number, min: 1)
- Input: Rack Prefix (type: text)
- File upload: RankTable JSON (accept: .json)

### LaunchPage
- Select: Network (mapped to activeNetworkId)
- Select: Simulation Mode (options: analytical, ns3)
- Input: Threads (type: number, min: 1, max: 64)
- Input: Workload Path (type: text)
- Input: RankTable Path (type: text)
- Input: Topology Path (type: text)

### PodDetailPage
- Node click interaction (ReactFlow)
- Edge click interaction (ReactFlow)

---

## 7. BUTTONS & CLICKABLE ELEMENTS

### HomePage
- "+ 新建组网" (new network button)
- Network name (click to select)
- "删除" button (per network, hover to reveal)
- "Workload 配置", "RankTable 配置", "启动仿真" (links)
- "结果可视化", "实时监控大屏" (links)

### WorkloadPage
- Mode selector buttons (3 options)
- "下一步 →" button (Next)

### RankTablePage
- Mode selector buttons (3 options)
- "下一步 →" button (Next)

### LaunchPage
- Network dropdown select
- "启动仿真" or "停止仿真" button
- "Kill" buttons in ProcessList

### PodDetailPage
- ReactFlow nodes (clickable)
- ReactFlow edges (clickable)
- Close buttons on detail panels

### ResultsPage
- Task buttons (click to load)
- "或者手动上传文件 ↓" toggle
- Tab buttons (Overview/Layers/Bandwidth/Nodes)
- "← 返回任务列表" button

---

## 8. MODAL DIALOGS & PANELS

### HomePage
- **Create Network Modal**: Overlay with form, 2 input fields, cancel/create buttons

### PodDetailPage
- **NodeDetailPanel**: Right sidebar, metric display, close button
- **EdgeDetailPanel**: Right sidebar, chart visualization, close button

### ResultsPage
- File upload collapsible section

---

## 9. TABS & MODES

### NetworkDetailPage
- "监控" tab (monitor) - ReactFlow topology
- "拓扑文件" tab (files) - file listing

### WorkloadPage
- "Preset Model" mode - form with selects/inputs
- "Custom Layers" mode - form with inputs
- "Upload File" mode - file upload

### RankTablePage
- "Auto Generate" mode - form with inputs
- "Custom Topology" mode - (not implemented)
- "Upload File" mode - file upload

### ResultsPage
- "Overview" tab - metrics + breakdowns
- "Layer Timing" tab - layer timing chart
- "Bandwidth" tab - bandwidth chart
- "Node Transfer" tab - node transfer chart

---

## 10. ERROR HANDLING & VALIDATION

### HomePage
- Modal input validation: name and directory required
- Error display: `createError` in ValidationBanner
- Network deletion (no confirmation dialog)

### WorkloadPage
- Mode-specific validation
- Upload mode: file required before next
- Error/Success messages via ValidationBanner

### RankTablePage
- Config validation via API
- Validation errors displayed as warnings
- Error messages via ValidationBanner

### LaunchPage
- Network required before launch
- Launch disabled if no network selected
- Process list auto-refresh every 5s
- Kill button only for running processes

### ResultsPage
- Task loading with error handling
- File parsing errors displayed
- Empty states for missing data

---

## 11. POLLING & REAL-TIME UPDATES

### Polling Intervals:
1. **NetworkDetailPage (Monitor tab)**: 5000ms
   - `fetchOverviewTopology()`, `fetchClusterSummary()`

2. **PodDetailPage**: 5000ms
   - `fetchPodDetail()`

3. **NodeDetailPanel**: 3000ms
   - `fetchNodeMetrics()`

4. **ProcessList**: 5000ms
   - `listProcesses()`

5. **LogViewer**: Real-time via `useLogStream()` hook

---

## 12. RESPONSIVE DESIGN & GRID LAYOUTS

### Grids Used:
- **HomePage**: 3 columns on lg, 1 column on mobile
- **Quick Start Steps**: 4 columns on md, 2 columns on sm
- **LaunchPage Config**: 2 columns
- **ResultsPage Upload**: 1 column on md, 2 columns otherwise
- **Charts**: ResponsiveContainer (Recharts)

---

## 13. VALIDATION PATTERNS

1. **Text inputs**: Non-empty string check
2. **Number inputs**: min/max bounds (validated in HTML)
3. **File uploads**: Accept MIME type filtering
4. **API responses**: Error handling with try-catch
5. **Form submission**: Disabled state during loading

---

## 14. NAVIGATION PATTERNS

- **Browser back button**: Works (uses React Router)
- **Page links**: Link components to URL paths
- **Programmatic navigation**: `useNavigate()` hook
- **Auto-navigation**: On wizard step completion

---

## 15. TESTING COVERAGE REQUIREMENTS

### Critical User Paths (100% E2E Coverage):
1. Create → Configure → Launch → View Results
2. Edit Network
3. Monitor Network Topology
4. View Pod Details
5. Upload Custom Files
6. Handle Errors & Empty States

### Interaction Types to Test:
- Button clicks (all 40+ buttons)
- Form field input (text, number, select, file upload)
- Mode/tab selection (3-4 modes per page)
- Modal open/close
- Validation & error messages
- API call success/failure
- Real-time polling updates
- Navigation flows
- Sidebar panel open/close
- Chart/graph interactions

---

