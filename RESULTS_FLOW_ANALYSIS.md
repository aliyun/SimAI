# SimAI Simulation Results Flow Analysis

## 1. Result Files Produced by SimAI_analytical

### File Location and Naming
**C++ Files:**
- `/Users/anthony/PycharmProjects/SimAI/astra-sim-alibabacloud/astra-sim/network_frontend/analytical/AnalyticalAstra.cc` (line 29)
- `/Users/anthony/PycharmProjects/SimAI/astra-sim-alibabacloud/astra-sim/workload/Workload.cc` (lines 128-131)

**RESULT_PATH:** `./results/` (hardcoded in AnalyticalAstra.cc:29)

**Output Prefix:** Controlled by `-r` or `--result` flag (AstraParamParse.cc:124-125):
```cpp
} else if (arg == "-r" || arg == "--result") {
    if (++i < argc) this->res = argv[i];
```

**Files Generated:**
1. **EndToEnd.csv** - Main performance metrics file
   - Created: `Workload.cc:129` via `CSVWriter(path, "EndToEnd.csv")`
   - Path: `RESULT_PATH + param->res`
   - Example full path: `./results/mymodel-tp2-pp4-dp8-ga1-ep1-NVL8-100.0G-DP0.1`

2. **detailed_*.csv** - Detailed layer-by-layer metrics
   - Created: `Workload.cc:128`
   - Naming: `detailed_<total_nodes>.csv`

3. ***_dimension_utilization_*.csv** - Dimension utilization metrics
   - Created: `Workload.cc:130-131`
   - Naming: `<run_name>_dimension_utilization_<npu_offset>.csv`

### How Results Are Written

**CSVWriter Class** (`/Users/anthony/PycharmProjects/SimAI/astra-sim-alibabacloud/astra-sim/workload/CSVWriter.hh` and `.cc`):

```cpp
// Constructor (CSVWriter.cc:9-12)
CSVWriter::CSVWriter(std::string path, std::string name) {
  this->path = path;
  this->name = name;
}

// Write operation (CSVWriter.cc:13-18)
void CSVWriter::write_line(std::string data) {
  if (!myFile.is_open()) {
    myFile.open(path + name, std::ios::out | std::ios::app);
  }
  myFile << data << std::endl;
}

// Write with header prepend (CSVWriter.cc:19-38)
void CSVWriter::write_res(std::string data) {
  // Reads existing content, writes new header, then existing content
  myFile.close();
  myFile.open(path + name, std::ios::in);
  // ... read current content ...
  myFile.open(path + name, std::ios::out | std::ios::trunc);
  myFile << data << std::endl;
  myFile << current_content;
  myFile.close();
}
```

**Writing Calls** (Layer.cc):
- Line 433, 493, 499, 502: `EndToEnd->write_line(data)`
- Line 535, 765: `EndToEnd->write_res(data)` (header prepend)

---

## 2. Backend Endpoints for Results

### File: `/Users/anthony/PycharmProjects/SimAI/server/simulation/results_routes.py` (lines 1-87)

**Three Main Endpoints:**

#### a) Parse EndToEnd CSV
```python
@results_bp.route("/parse-endtoend", methods=["POST"])
@require_auth
def api_parse_endtoend():
    """Parse EndToEnd CSV content and return structured data."""
    # Accepts either:
    # - content: raw CSV string
    # - filepath: path to CSV file
    # Returns: parsed results dict with keys:
    #   - layers: array of layer metrics
    #   - summary: SUM row
    #   - totals: total_exposed, total_compute, total_time
    #   - dimensions: key-value pairs with value/percentage
    #   - run_name: extracted from data
```

**Parsing:** Uses `parse_endtoend_csv()` from `server/simulation/visualizer.py`

#### b) Parse Console Output
```python
@results_bp.route("/parse-console", methods=["POST"])
@require_auth
def api_parse_console():
    """Parse console output lines and extract metrics."""
    # Input: log_lines (array of strings)
    # Extracts via regex:
    #   - finish_time: "all passes finished at time: X"
    #   - streams_injected: "Total streams injected: N"
    #   - streams_finished: "Total streams finished: N"
    #   - nodes: data sent/received per node
```

**Parsing:** Uses `parse_console_output()` from `visualizer.py`

#### c) Find Result Files
```python
@results_bp.route("/find-files", methods=["GET"])
@require_auth
def api_find_files():
    """Find simulation output files in result directories."""
    # Searches for:
    #   - "EndToEnd.csv" in result path or workspace
    #   - "detailed_*.csv" files
    # Returns: dict with keys "endtoend" and "detailed"
```

**Parsing:** Uses `find_ns3_output_files()` from `visualizer.py` (lines 175-204)

---

## 3. Database Schema (Process Storage)

### File: `/Users/anthony/PycharmProjects/SimAI/server/db/database.py` (lines 14-42)

**Processes Table:**
```sql
CREATE TABLE IF NOT EXISTS processes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pid INTEGER NOT NULL,
    session_token TEXT NOT NULL,
    username TEXT NOT NULL,
    workspace_dir TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    finished_at REAL,
    return_code INTEGER,
    error_message TEXT,
    FOREIGN KEY (session_token) REFERENCES sessions(token)
);

CREATE INDEX IF NOT EXISTS idx_processes_session ON processes(session_token);
CREATE INDEX IF NOT EXISTS idx_processes_status ON processes(status);
CREATE INDEX IF NOT EXISTS idx_processes_username ON processes(username);
```

**Key Fields:**
- `workspace_dir` - relative path to workspace containing results
- `status` - running/finished/error/timeout/killed
- `finished_at` - timestamp when process completed
- `return_code` - exit code (0 = success)

**Note:** DB does NOT store result file paths directly. Results are found at:
```
{workspace_dir}/logs/{pid}.log        # process logs
{workspace_dir}/results/              # simulation output files
```

---

## 4. Server Configuration for Results

### File: `/Users/anthony/PycharmProjects/SimAI/server/config.py` (lines 1-34)

**Relevant Config:**
```python
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_DIR)
WORKSPACE_ROOT = os.path.join(SERVER_DIR, "workspaces")
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
```

**Key Paths:**
- `WORKSPACE_ROOT`: `/Users/anthony/PycharmProjects/SimAI/server/workspaces`
- Result files are stored within each workspace subdirectory

**No explicit RESULT_PATH configuration** — results location is derived from workspace directory.

---

## 5. Process Service & Result File Discovery

### File: `/Users/anthony/PycharmProjects/SimAI/server/process/process_service.py` (lines 71-243)

#### Launch Function (lines 71-184)
```python
def launch(session_token, username, workspace_dir, params):
    # Accepts params like:
    # {
    #   "binary": "SimAI_analytical",
    #   "workload_file": "workload.txt",
    #   "ranktable_file": "ranktable.json",
    #   "topology_file": "topology_16g_...",
    #   "output_prefix": "oxc_output",
    #   "env_vars": { ... },
    #   "extra_args": ["-r", "result_prefix"]
    # }
    
    # Stores in DB:
    # - pid, username, workspace_dir
    # - command line executed
    # - status = 'running', started_at = now()
```

#### List Processes Function (lines 226-242)
```python
def list_processes(username=None, session_token=None, status=None):
    """Query processes with optional filters."""
    # Returns last 50 processes matching filters
    # Fields returned: all columns from processes table
```

**Result Discovery Flow:**
1. Call `list_processes(username=user)` → get running/finished processes
2. For finished processes, `workspace_dir` is known
3. Call backend endpoint `GET /api/simulation/results/find-files?path={workspace_dir}`
4. Searches for `EndToEnd.csv` and `detailed_*.csv`

---

## 6. Frontend Integration

### File: `/Users/anthony/PycharmProjects/SimAI/dashboard/src/pages/ResultsPage.tsx` (lines 1-186)

**Current Flow (Manual Upload):**
```tsx
// Lines 36-48
const handleEndToEndUpload = useCallback(async (content: string) => {
  try {
    const result = await parseEndToEnd(content);  // POST /api/simulation/results/parse-endtoend
    setEndtoendData(result);
    completeStep('results');
  } catch (err) { ... }
}, [setEndtoendData, completeStep]);
```

**Frontend API Client** (`/Users/anthony/PycharmProjects/SimAI/dashboard/src/api/simulation-api.ts`):

```typescript
// Line 141-147
export async function parseEndToEnd(content: string): Promise<EndToEndData> {
  const { data } = await apiClient.post<EndToEndData>(
    '/api/simulation/results/parse-endtoend',
    { content },
  );
  return data;
}

// Line 161-168
export async function findResultFiles(path?: string): Promise<Record<string, string>> {
  const params = path ? { path } : {};
  const { data } = await apiClient.get<FindFilesResponse>(
    '/api/simulation/results/find-files',
    { params },
  );
  return data.files;
}

// Line 206-225
export async function launchSimulation(params: Record<string, unknown>): Promise<LaunchResponse> {
  const { data } = await apiClient.post<LaunchResponse>('/api/process/launch', params);
  return data;
}

export async function listProcesses(): Promise<readonly ProcessEntry[]> {
  const { data } = await apiClient.get<ListProcessesResponse>('/api/process/list');
  return data.processes;
}
```

---

## 7. Complete Flow Summary

### Simulation → Results Discovery → Display

```
┌─────────────────────────────────────────────────────┐
│ 1. SIMULATION EXECUTION                             │
├─────────────────────────────────────────────────────┤
│ Frontend: LaunchPage → POST /api/process/launch     │
│ Backend: process_service.launch()                   │
│   - Stores in DB: processes(pid, username,          │
│     workspace_dir, command, status=running)         │
│   - Spawns subprocess running SimAI binary          │
│   - Streams logs to workspace_dir/logs/{pid}.log    │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 2. SIMULATOR WRITES RESULTS                         │
├─────────────────────────────────────────────────────┤
│ C++: AstraParamParse with -r flag                   │
│   param->res = "model_config_string"                │
│ C++: Workload.cc creates CSVWriter with path:      │
│   path = RESULT_PATH + param->res                   │
│   = "./results/" + param->res                       │
│ Output files:                                       │
│   - EndToEnd.csv                                    │
│   - detailed_{nodes}.csv                           │
│   - {run_name}_dimension_utilization_{offset}.csv  │
│ Written to: workspace_dir/results/                  │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 3. PROCESS COMPLETES                                │
├─────────────────────────────────────────────────────┤
│ Backend: _stream_output() thread detects completion │
│   - Updates DB: processes.status = 'finished'      │
│   - Sets finished_at, return_code                  │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 4. RESULTS PAGE REQUESTS RESULTS                    │
├─────────────────────────────────────────────────────┤
│ Frontend: ResultsPage.tsx                           │
│   (Currently manual file upload only)               │
│ Could query: GET /api/process/list                 │
│   → Returns processes with workspace_dir           │
│   → GET /api/simulation/results/find-files?path... │
│   → Returns {endtoend, detailed} file paths        │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 5. PARSE AND DISPLAY                                │
├─────────────────────────────────────────────────────┤
│ Frontend: POST /api/simulation/results/parse-endtoend
│   → visualizer.parse_endtoend_csv()                │
│   → Returns: layers[], summary, totals, dimensions │
│ Display: OverviewMetrics, LayerTimingChart, etc.  │
└─────────────────────────────────────────────────────┘
```

---

## 8. Missing Piece: Automatic Result Discovery

**Currently:** Results page only supports **manual file upload**

**What's Available But Not Used:**
- Backend has `/api/simulation/results/find-files` endpoint
- Backend can list finished processes with `workspace_dir`
- Backend can parse files directly from disk
- Database tracks workspace_dir for each process

**What's Needed for Auto-Discovery:**
1. Frontend: After launch, poll `GET /api/process/list`
2. Filter for `status='finished'` processes  
3. For each finished process, call:
   `GET /api/simulation/results/find-files?path={workspace_dir}/results/`
4. Automatically fetch and parse EndToEnd.csv
5. Display results without manual upload

---

## Key File Reference

| File Path | Lines | Purpose |
|-----------|-------|---------|
| `astra-sim-alibabacloud/astra-sim/network_frontend/analytical/AnalyticalAstra.cc` | 29, 121 | RESULT_PATH, param->res |
| `astra-sim-alibabacloud/astra-sim/system/AstraParamParse.hh/.cc` | 83-84, 124-125 | -r flag parsing |
| `astra-sim-alibabacloud/astra-sim/workload/Workload.cc` | 128-131 | Create EndToEnd.csv |
| `astra-sim-alibabacloud/astra-sim/workload/CSVWriter.hh/.cc` | 24, 9-38 | File write mechanism |
| `astra-sim-alibabacloud/astra-sim/workload/Layer.cc` | 433, 493, 535 | Write data to CSV |
| `server/db/database.py` | 14-42 | Process DB schema |
| `server/process/process_service.py` | 71, 100-102, 164 | Launch, workspace_dir, logs |
| `server/simulation/results_routes.py` | 20-87 | Parse/find endpoints |
| `server/simulation/visualizer.py` | 22, 124, 175 | Parse functions |
| `server/config.py` | 12-17 | Path configuration |
| `dashboard/src/pages/ResultsPage.tsx` | 36-62 | Current manual upload |
| `dashboard/src/api/simulation-api.ts` | 141-168, 206-225 | API clients |
