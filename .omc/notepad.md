# Notepad
<!-- Auto-managed by OMC. Manual edits preserved in MANUAL section. -->

## Priority Context
<!-- ALWAYS loaded. Keep under 500 chars. Critical discoveries only. -->

## Working Memory
<!-- Session notes. Auto-pruned after 7 days. -->
### 2026-04-15 09:59
SimAI E2E Testing Investigation Complete

Key Findings:
1. Ports: Server (5000/5001), Dashboard (3000)
2. Server: Flask on port 5000 (configurable via SIMAI_SERVER_PORT)
3. Dashboard: Vite dev server on port 3000 (configurable)
4. E2E Tests: Playwright with MSW (Mock Service Worker) - no backend needed for tests
5. Start Script: ./start_dashboard.sh orchestrates both services

Next: Provide detailed report to user


## MANUAL
<!-- User content. Never auto-pruned. -->
### 2026-04-14 08:45
## SimAI Frontend/GUI Exploration Summary

### Project Structure
**Base Path**: `/Users/anthony/PycharmProjects/SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/oxc/gui/`

### Main Entry Point (app.py)
- **Framework**: Streamlit (v1.28.0+)
- **Purpose**: Single-entry application for SimAI-OXC Configuration GUI
- **Login Flow**:
  - Single-user mode: Auto-login with "default" user, redirects to workload page
  - Multi-user mode: Shows login form, validates credentials
- **Page Icon**: 🔐 (security)
- **Title**: "SimAI-OXC - AI Training Simulator — Configuration GUI"
- **Key Features**:
  - Initializes API client from `api_client.py` (SimAIClient)
  - Manages session state for files/configs
  - Uses Chinese page titles for all workflows
  - Version: v1.1 (shown in footer)

### Page Files (5 Steps)

**1_工作负载.py** (📝 Workload Configuration)
- **Purpose**: Configure training workload parameters
- **Modes**:
  - Preset models (7B, 13B, 70B, 175B)
  - Custom configuration with layer-by-layer settings
  - Upload existing workload.txt
- **Parallelism Settings**: TP (Tensor), DP (Data), PP (Pipeline), EP (Expert)
- **Output**: `workload.txt` file
- **Key Functions**: 
  - `generate_megatron_workload()`
  - `generate_custom_workload()`
  - Auto-navigation to Step 2

**2_RankTable.py** (🖥️ RankTable Configuration)
- **Purpose**: Configure GPU topology/rank-to-rack mapping
- **Modes**:
  - Auto-generate (simple rank/GPU/rack counts)
  - Custom topology (manual rack assignment)
  - Upload existing ranktable.json
- **Network Types**: OXC, RDMA, RoCE
- **Outputs**: 
  - `ranktable.json` (GPU rank metadata)
  - `rank_rack_map.json` (rank-to-rack mapping)
- **Visualization**: Topology graph with Plotly

**3_拓扑编辑.py** (🎨 Topology Editor)
- **Purpose**: Embedded Draw.io editor for network topology design
- **Features**:
  - Copy-paste XML workflow
  - Converts Draw.io diagrams to NS3 topology format
  - Node colors: GPU (green), NVSwitch (blue), ASW (orange), PSW (red)
  - Bandwidth/latency on links (e.g., "200Gbps 0.5ms")
- **Output**: NS3-compatible topology file
- **Template**: `assets/simai_template.drawio` (pre-made network diagram)

**4_启动仿真.py** (🚀 Launch Configuration)
- **Purpose**: Configure and launch NS3 simulation with OXC integration
- **Key Settings**:
  - OXC API URL, algorithm type (RING/HD/NB)
  - NVLS/PXN optimization flags
  - Thread count, timeouts
  - Send latency
- **Prerequisites Check**: Validates workload, ranktable, topology files exist
- **Execution**:
  - Single-user: Direct subprocess execution
  - Multi-user: Backend API calls with progress streaming
- **Environment Variables**:
  - AS_OXC_ENABLE, AS_OXC_URL, AS_OXC_ALGO, etc.

**5_可视化.py** (📊 Visualization)
- **Purpose**: Analyze simulation results
- **Input Data**: `ncclFlowModel_EndToEnd.csv` from NS3 output
- **Tabs**:
  1. Overview: Summary metrics, compute vs comm breakdown
  2. Per-layer timing: Layer-by-layer compute/comm/exposed times
  3. Bandwidth analysis: Algorithm and bus bandwidth per layer
  4. Node data transfer: Data sent/received per node
- **Export**: CSV download of layer timing data

### Assets Directory
- **simai_template.drawio**: Draw.io template for network topology
  - Pre-configured with Server 0 example (8 GPUs, 1 NVSwitch)
  - Swimlane layout for servers
  - Color-coded node types (GPU=green, NVSwitch=blue, etc.)
  - Example connections with labels
  
- **simai_shapes.xml**: Custom shape library for Draw.io (node definitions)

### Core Utilities
- **workload_generator.py**: Megatron/custom workload generation
- **ranktable_generator.py**: RankTable and rank-rack mapping creation
- **drawio_to_ns3.py**: Converts Draw.io XML → NS3 topology format
- **visualizer.py**: Plotly charts for results analysis
- **page_auth.py**: Authentication/authorization utilities
- **api_client.py**: Backend API wrapper (Flask integration)

### Branding & Styling
- **Product Name**: SimAI-OXC
- **Tagline**: "AI Training Simulator — Configuration GUI"
- **Version**: v1.1
- **Framework Attribution**: "Powered by Streamlit"
- **Color Scheme**: 
  - Primary gradient: #667eea → #764ba2 (purple)
  - Success green: #10b981 → #059669
  - Warning yellow: #fef3c7 → #fde68a
  - Node colors in topology: green/blue/orange/red
- **Emojis Used**: 🔐 🔒 📝 🖥️ 🎨 🚀 📊 ✅ ❌ ⏳ ⚠️ 📋 ⏱️ 📶 🔄
- **No custom logo file found** (uses Streamlit defaults + emoji icons)

### Key Architecture Notes
1. **Multi-user Support**: API client detects mode via `SIMAI_MULTIUSER` env var
2. **Session State Management**: Persistent file references across steps
3. **Chinese Localization**: All page names and UI in Chinese
4. **Page Navigation**: `st.switch_page()` for linear workflow (Step 1→2→3→4→5)
5. **Error Handling**: Validation before file save, prerequisites checks
6. **Output Organization**: Files saved via API client (central file management)

### Requirements
- streamlit >= 1.28.0
- pandas >= 1.5.0
- plotly >= 5.15.0
- numpy >= 1.24.0
- requests (for API client)



