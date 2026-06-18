# Implementation Plan: SimAI-NS3 Decoupled Replay (Revised)

**Plan saved to:** `.omc/plans/decoupled-replay.md`
**Spec:** `.omc/specs/deep-interview-decoupled-replay.md`
**Generated:** 2026-06-18 | **Revised:** 2026-06-18 (analyst gap analysis)
**Complexity:** HIGH (brownfield, 7 files modified, ~5 new files, cross-component changes)

---

## Revision Notes (analyst gap analysis)

This revision addresses 4 critical findings and resolves 6 open questions discovered during analyst review.

| Gap | Finding | Resolution |
|-----|---------|------------|
| G1 | Destructor placement unreliable for `finalizeFlowFile()` | Add explicit call in `AstraSimNetwork.cc::main()` between `Simulator::Run()` and `Simulator::Destroy()`; keep destructor as fallback only |
| G2 | 3 mirror copies of common.h, not 2 as originally listed | List all 3 copies explicitly; add CI diff-check script |
| G3 | `ImportFlows()` parses partial format; independent binary must parse complete format | Base `flow_reader.h` on `loadFlowsFromFile()` (complete parser), not `ImportFlows()` (partial parser) |
| G4 | Cross-layer ordering not guaranteed by `prev[]` alone | Add `layer_num` constraint in `DepScheduler`: all Layer N flows must complete before any Layer N+1 flow is scheduled |

---

## RALPLAN-DR Summary

### Principles (5)
1. **Minimal invasiveness** -- SimAI side changes must not alter behavior of existing coupled mode or `AS_REPLAY_MODE=1` path
2. **True independence** -- Independent binary links only against ns3 libraries, zero SimAI symbols
3. **Causality-driven timing** -- Use dependency graph (`prev[]`) + relative delay (`relative_delay_ns`), not absolute wall-clock timestamps
4. **Extract and adapt** -- Copy necessary logic from `common.h`/`entry.h`/`MockNcclGroup.cc` to independent modules; do not rewrite proven RDMA/network code
5. **Verifiability** -- Every phase produces measurable, checkable output (flow file, FCT file, diff report, CI diff-check)

### Decision Drivers (top 3)
| # | Driver | Weight | Explanation |
|---|--------|--------|-------------|
| 1 | Independent binary must compile and run without SimAI libraries | HARD | Spec constraint: NS3 replay binary must not link any SimAI (astra-sim) code |
| 2 | Flow timing must reflect actual SimAI event chain behavior | HIGH | Pre-computed estimates would deviate from coupled-mode timing; goal is to capture real event-chain gaps |
| 3 | Preserve backward compatibility | HIGH | Existing coupled mode, AS_REPLAY_MODE=1, and analytical mode must continue working unchanged |

### Viable Options

#### Option A: SimAI simulation-time instrumentation + independent dependency-graph scheduler [SELECTED]
- SimAI records per-flow send times during coupled simulation (via `ASTRASimNetwork::sim_send()`)
- After simulation, computes `relative_delay_ns = send_time - max(send_time of prev[])` and writes flow file
- Independent binary uses dependency graph: waits for all `prev[]` completion, then schedules after `relative_delay_ns`
- **Pros:** Captures actual SimAI timing from the event chain; clean architecture with clear boundary; verifiable against coupled FCT; dependency graph preserves causality
- **Cons:** Requires moderate SimAI-side instrumentation (~150 lines); two-pass flow file generation

#### Option B: Pre-computed relative delays from LogGP analytical model
- **Rejected because:** Fragile dependency on LogGP internals; analytical estimates differ from event-chain behavior; cannot capture scheduler-level timing (overlap ratios, contention effects)

#### Option C: Absolute-timestamp based replay
- **Rejected because:** Loses causality information; cannot model network-feedback effects; contradicts spec guidance (Round 3 interview explicitly chose dependency-graph over absolute timestamps)

#### Why only Option A survives
- Option B is invalidated by fragility concerns (LogGP model internals are not stable public API) and loss of accuracy
- Option C is invalidated by spec constraints (Round 3 explicitly chose dependency-graph over absolute timestamps) and loss of the core value proposition (network competition feedback)
- Option A is the only approach satisfying all three decision drivers

---

## Architecture

### Data Flow Diagram (text-based)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SimAI Coupled Simulation                  │
│                                                                      │
│  workload → genFlowModels() → [Flow metadata accumulated in buffer]   │
│       ↓                                                              │
│  SimAI Event Chain (LogGP scheduling)                                │
│       ↓                                                              │
│  Sys::sim_send() → ASTRASimNetwork::sim_send()                      │
│       │                    │                                         │
│       │         [record flow_id → Sys::boostedTick()]                │
│       ↓                    ↓                                         │
│  SendFlow() → NS3 RDMA simulation → FCT output (coupled)             │
│                                                                      │
│  SimAI completes (Simulator::Run() returns):                          │
│    GlobalGroup->finalizeFlowFile():                                   │
│      for each buffered flow: compute relative_delay_ns               │
│      write flow file with complete format (all fields)                │
│      write metadata header (total_flows)                              │
│                                                                      │
│  NOTE: finalizeFlowFile() called EXPLICITLY in main(),               │
│        not relying on destructor ordering.                           │
└──────────────────────────────────────────────────────────────────────┘
                              │
                     flow_file.txt
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│               PHASE 2: Independent NS3 Replay Binary                  │
│                                                                      │
│  main()                                                              │
│    ├── ReadConf(topology, config)  ← extracted from common.h         │
│    ├── SetupNetwork()              ← extracted from common.h         │
│    ├── LoadFlows(flow_file)        ← COMPLETE format parsing          │
│    │     (all fields: prev, parent, child, layer, group, op,         │
│    │      loopstate, relative_delay_ns)                              │
│    ├── DepScheduler::Init(flows)   ← build dependency graph          │
│    │     (prev[] + layer_num constraint)                              │
│    │                                                                │
│    ├── DepScheduler::Run():                                        │
│    │     while (pending_flows > 0):                                  │
│    │       ready = flows where all prev[] completed                  │
│    │               AND all prior layer flows completed               │
│    │       for each ready flow:                                      │
│    │         wait relative_delay_ns                                  │
│    │         SendFlow(src, dst, size, ...)                          │
│    │         mark as "in_flight"                                    │
│    │       wait for next completion event                           │
│    │     assert layer ordering invariant                            │
│    │                                                                │
│    ├── NS3 Simulator::Run()                                         │
│    └── Output FCT file (via qp_finish callbacks)                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

```
SimAI Side (astra-sim-alibabacloud/)
├── network_frontend/ns3/common.h     ← FlowRecord + relative_delay_ns [COPY #1]
├── network_frontend/ns3/entry.h      ← (no changes)
├── network_frontend/ns3/AstraSimNetwork.cc ← record send times, explicit finalize
├── system/MockNcclGroup.h            ← deferred flow buffer + send time map
└── system/MockNcclGroup.cc           ← accumulate flows, finalizeFlowFile()
├── extern/.../scratch/common.h       ← [COPY #2] mirror FlowRecord change
└── extern/.../src/.../ns3/common.h   ← [COPY #3] mirror FlowRecord change

Independent Binary (ns-3-alibabacloud/simulation/scratch/decoupled_replay/)
├── CMakeLists.txt                    ← links ns3 ONLY, no astra-sim
├── main.cc                           ← entry point, orchestration
├── flow_reader.h                     ← COMPLETE FlowRecord parsing (from loadFlowsFromFile format)
├── topology_reader.h                 ← ReadConf + SetupNetwork (from common.h)
├── flow_sender.h                     ← SendFlow, RdmaClient (from entry.h)
├── fct_writer.h                      ← qp_finish, FCT output (from entry.h)
├── dep_scheduler.h                   ← dependency graph + relative delay + layer_num constraint
└── common_types.h                    ← shared types, global state
```

---

## Flow File Format (Complete Specification)

The independent binary's `flow_reader.h` must parse the COMPLETE format, NOT the partial format from `ImportFlows()`.

### Write format (from `_writeFlowRecord()` in MockNcclGroup.cc:33-49)

```
flow_id src dest flow_size channel_id chunk_id chunk_count conn_type start_time pg maxPacketCount port dport np prev[0..np-1] npar parent_flow_id[0..npar-1] nchi child_flow_id[0..nchi-1] layer_num group_type op loopstate relative_delay_ns
```

| Field Index | Name | Type | Written by |
|-------------|------|------|------------|
| 1 | flow_id | uint32_t | _writeFlowRecord |
| 2 | src | uint32_t | _writeFlowRecord |
| 3 | dest | uint32_t | _writeFlowRecord |
| 4 | flow_size | uint64_t | _writeFlowRecord |
| 5 | channel_id | int | _writeFlowRecord |
| 6 | chunk_id | int | _writeFlowRecord |
| 7 | chunk_count | int | _writeFlowRecord |
| 8 | conn_type | string | _writeFlowRecord |
| 9 | start_time | double (kept 0.0) | _writeFlowRecord |
| 10 | pg | uint32_t (hardcoded 3) | _writeFlowRecord |
| 11 | maxPacketCount | uint32_t | _writeFlowRecord |
| 12 | port | uint32_t | _writeFlowRecord |
| 13 | dport | uint32_t | _writeFlowRecord |
| 14 | prev[] | vector<uint32_t> | _writeFlowRecord |
| 15 | parent_flow_id[] | vector<int> | _writeFlowRecord |
| 16 | child_flow_id[] | vector<int> | _writeFlowRecord |
| 17 | layer_num | int → uint32_t | _writeFlowRecord |
| 18 | group_type | int → uint32_t | _writeFlowRecord |
| 19 | op | int → uint32_t | _writeFlowRecord |
| 20 | loopstate | int → uint32_t | _writeFlowRecord |
| 21 | relative_delay_ns | uint64_t (NEW) | NEW: appended by revised _writeFlowRecord |

### Fields silently skipped by `ImportFlows()` (entry.h:211-220)
- parent_flow_id[] (field 15)
- child_flow_id[] (field 16)
- group_type (field 18) -- present in FlowRecord struct but not parsed
- op (field 19) -- present in FlowRecord struct but not parsed
- loopstate (field 20) -- present in FlowRecord struct but not parsed

### Fields parsed by `loadFlowsFromFile()` (MockNcclGroup.cc:2245-2265)
- ALL 20 fields (the complete format). This is the CORRECT reference for the independent binary's `flow_reader.h`.

---

## File-Level Changes

### SimAI Side (7 files modified across 3+ locations)

---

#### 1. `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h` [COPY #1 of 3]

**Change:** Add `uint64_t relative_delay_ns` field to `FlowRecord` struct (after `loopstate`)

```cpp
// FlowRecord struct (line 145-167)
struct FlowRecord {
  uint32_t flow_id;
  uint32_t src, dst;
  uint64_t flow_size;
  int channel_id;
  int chunk_id;
  int chunk_count;
  std::string conn_type;
  double start_time;       // ns, relative to sim start (legacy, kept for compat; always 0.0 in decoupled)
  std::vector<uint32_t> prev;
  uint32_t pg;
  uint32_t maxPacketCount;
  uint32_t port, dport;
  uint32_t layer_num;
  uint32_t group_type;
  uint32_t op;
  uint32_t loopstate;
  uint64_t relative_delay_ns; // NEW: delay after all prev[] complete, before sending (ns)
};
```

**Mirror locations (must apply identical change):**
- [COPY #2]: `astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/src/applications/astra-sim/network_frontend/ns3/common.h`
- [COPY #3]: `astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/scratch/common.h`

All 3 copies are byte-identical (verified). A CI diff-check script must be added to prevent drift.

---

#### 2. `astra-sim-alibabacloud/astra-sim/system/MockNcclGroup.h`

**Changes:**
- Add `#include <unordered_map>`
- Add member variables:
  ```cpp
  // Decoupled mode: accumulate flows for deferred write (replaces per-flow immediate write)
  struct FlowWriteEntry {
    SingleFlow sf;
    uint32_t max_pkts, port, dport;
    int layer_num, group_type, op, loopstate;
    uint64_t relative_delay_ns; // computed in finalizeFlowFile()
  };
  std::vector<FlowWriteEntry> _flow_buffer;

  // Map flow_id → SimAI tick (boostedTick) when flow was actually sent via sim_send()
  std::unordered_map<uint32_t, uint64_t> _flow_send_times;
  ```
- Add method declarations:
  ```cpp
  void recordFlowSendTime(uint32_t flow_id);
  void finalizeFlowFile();
  ```
- Remove `~MockNcclGroup()` reliance for `finalizeFlowFile()` -- explicit call replaces destructor-based finalization. Keep destructor as fallback with a warn-if-not-called guard.

---

#### 3. `astra-sim-alibabacloud/astra-sim/system/MockNcclGroup.cc`

**Change A: `_writeFlowRecord()` (line 33-49)** -- Accept `relative_delay_ns` parameter and append it as the last field:

```cpp
static void _writeFlowRecord(std::ofstream &f, const SingleFlow &sf,
                             uint32_t maxPkts, uint32_t port, uint32_t dport,
                             double start_ns, int layer_num,
                             int group_type, int op, int loopstate,
                             uint64_t relative_delay_ns) {  // NEW param
  f << sf.flow_id << " " << sf.src << " " << sf.dest << " "
    << sf.flow_size << " " << sf.channel_id << " " << sf.chunk_id << " "
    << sf.chunk_count << " " << sf.conn_type << " "
    << start_ns << " 3 " << maxPkts << " " << port << " " << dport
    << " " << sf.prev.size();
  for (int pid : sf.prev) f << " " << pid;
  f << " " << sf.parent_flow_id.size();
  for (int pid : sf.parent_flow_id) f << " " << pid;
  f << " " << sf.child_flow_id.size();
  for (int cid : sf.child_flow_id) f << " " << cid;
  f << " " << layer_num << " " << group_type << " " << op << " " << loopstate
    << " " << relative_delay_ns;  // NEW: append relative delay as last field
  f << "\n";
  f.flush();
}
```

**Change B: `genFlowModels()` / `getFlowModels()` (line 364-378)** -- Accumulate flows in `_flow_buffer` instead of writing immediately:

```cpp
// In getFlowModels(), replace the _writeFlowRecord() call in the _flow_file.is_open() block:
if (_flow_file.is_open()) {
    auto& rank2fm = flow_models[flow_model_name];
    std::set<int> written;
    for (auto& rkv : rank2fm)
      for (auto& fkv : *rkv.second)
        if (written.insert(fkv.first.second).second) {
          const SingleFlow& sf = fkv.second;
          // Push to deferred buffer instead of immediate write
          _flow_buffer.push_back({
              sf,
              (uint32_t)((sf.flow_size + 4095) / 4096),
              (uint32_t)(sf.src * 100 + sf.dest),
              100,
              layer_num,
              (int)type,
              (int)op,
              (int)loopstate,
              0  // relative_delay_ns placeholder, computed in finalizeFlowFile()
          });
          _flow_count++;
        }
}
```

**Change C: New method `recordFlowSendTime()`:**

```cpp
void MockNcclGroup::recordFlowSendTime(uint32_t flow_id) {
    // Record the current SimAI tick when the flow is actually sent
    _flow_send_times[flow_id] = Sys::boostedTick();
}
```

**Change D: New method `finalizeFlowFile()`** -- Compute relative delays from send times and write file:

```cpp
void MockNcclGroup::finalizeFlowFile() {
    if (!_flow_file.is_open() || _flow_buffer.empty()) {
        if (_flow_file.is_open()) {
            // Write 0 count to indicate empty flow file
            _flow_file << "0\n";
            _flow_file.close();
        }
        return;
    }

    // Step 1: Build prev-to-flow lookup
    std::unordered_map<uint32_t, size_t> flow_id_to_idx;
    for (size_t i = 0; i < _flow_buffer.size(); i++)
        flow_id_to_idx[_flow_buffer[i].sf.flow_id] = i;

    // Step 2: Compute relative_delay_ns for each flow
    int zero_send_count = 0;
    for (auto& entry : _flow_buffer) {
        const auto& sf = entry.sf;
        uint64_t my_send_time = 0;

        if (_flow_send_times.count(sf.flow_id)) {
            my_send_time = _flow_send_times[sf.flow_id];
        } else {
            zero_send_count++;
        }

        if (sf.prev.empty()) {
            // No dependencies: relative delay is the absolute send time
            entry.relative_delay_ns = my_send_time;
        } else {
            // Find latest predecessor send time
            uint64_t max_prev_time = 0;
            for (int pid : sf.prev) {
                if (_flow_send_times.count(pid))
                    max_prev_time = std::max(max_prev_time, _flow_send_times[pid]);
            }
            // relative delay = my send time - latest predecessor send time
            entry.relative_delay_ns = (my_send_time > max_prev_time)
                ? (my_send_time - max_prev_time) : 0;
        }
    }

    if (zero_send_count > 0) {
        std::cerr << "[Decoupled] WARNING: " << zero_send_count
                  << " flows have send_time=0 (never recorded via sim_send())" << std::endl;
        std::cerr << "[Decoupled] This may indicate an unsupported mode (analytical/physical) "
                  << "or a code path that bypasses ASTRASimNetwork::sim_send()" << std::endl;
    }

    // Step 3: Write all flows to file
    _flow_file.seekp(0);
    _flow_file << _flow_buffer.size() << "\n";  // Write actual count as header

    for (auto& entry : _flow_buffer) {
        _writeFlowRecord(_flow_file, entry.sf,
            entry.max_pkts, entry.port, entry.dport,
            0.0,  // start_time kept as 0.0 for compat (independent binary uses relative_delay_ns)
            entry.layer_num, entry.group_type, entry.op, entry.loopstate,
            entry.relative_delay_ns);
    }

    _flow_file.close();
    std::cout << "[Decoupled] Flow file written: " << _flow_buffer.size()
              << " flows" << std::endl;
}
```

**Change E: Update `loadFlowsFromFile()` to also read `relative_delay_ns`** (for `AS_REPLAY_MODE=1` path, which pre-loads flows from file):

```cpp
// In loadFlowsFromFile(), after reading pf.loopstate (line 2265):
// Add: is >> pf.relative_delay_ns;  (optional, with fallback to 0 for legacy files)
```

**Change F: `~MockNcclGroup()` fallback** -- Add a warn-if-not-called guard:
```cpp
MockNcclGroup::~MockNcclGroup() {
    if (_flow_file.is_open() && !_flow_buffer.empty()) {
        std::cerr << "[Decoupled] WARNING: finalizeFlowFile() was not called explicitly; "
                  << "calling from destructor (this may be too late)" << std::endl;
        finalizeFlowFile();
    }
    if (_flow_file.is_open()) {
        _flow_file.close();
    }
}
```

---

#### 4. `astra-sim-alibabacloud/astra-sim/network_frontend/ns3/AstraSimNetwork.cc`

**Change A: `sim_send()` -- Add send time recording:**

```cpp
virtual int sim_send(void *buffer, uint64_t count, int type, int dst,
                     int tag, AstraSim::sim_request *request,
                     void (*msg_handler)(void *fun_arg), void *fun_arg) {
    // ... existing code (buffer copying, etc.) ...

    // Record send time for decoupled replay
    // Guard: only record when GlobalGroup exists (NS3 mode) and request has a flow_tag
    if (request && GlobalGroup) {
        GlobalGroup->recordFlowSendTime(request->flowTag.current_flow_id);
    }

    SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
}
```

**Change B: `main()` -- Add explicit `finalizeFlowFile()` call:**

Insert between `Simulator::Run()` and `Simulator::Destroy()` (before the existing `delete GlobalGroup`):

```cpp
  Simulator::Run();

  // Decoupled mode: finalize flow file while all data is still valid
  // Must be called BEFORE Simulator::Destroy() to ensure send time data is intact
  if (GlobalGroup != nullptr) {
      GlobalGroup->finalizeFlowFile();
  }

  Simulator::Destroy();
```

Keep the existing `delete GlobalGroup` after `Simulator::Destroy()` -- it now acts as a safety net (destructor fallback warns if `finalizeFlowFile()` was somehow skipped).

---

#### 5. `astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/src/applications/astra-sim/network_frontend/ns3/common.h` [COPY #2 of 3]

Identical change to File #1: add `uint64_t relative_delay_ns` to `FlowRecord` struct.

---

#### 6. `astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/scratch/common.h` [COPY #3 of 3]

Identical change to File #1: add `uint64_t relative_delay_ns` to `FlowRecord` struct.

---

#### 7. `astra-sim-alibabacloud/astra-sim/system/Sys.cc`

**Change:** In `Sys::~Sys()` (for the analytical mode path), add `finalizeFlowFile()` call before `delete GlobalGroup`:

```cpp
// In Sys::~Sys(), before delete GlobalGroup (around line 117):
  if (GlobalGroup != nullptr) {
    GlobalGroup->finalizeFlowFile();  // NEW: ensure flow file is finalized in analytical mode too
    delete GlobalGroup;
    GlobalGroup = nullptr;
  }
```

**Note:** In analytical mode, `AS_DECOUPLED_OUTPUT` won't open a flow file (gated to NS3 mode). This call is a no-op in that case. But if the user forces flow output in analytical mode, this ensures the file is still finalized.

---

### Independent NS3 Binary (new files, all in new directory)

#### Directory: `ns-3-alibabacloud/simulation/scratch/decoupled_replay/`

---

#### 8. `CMakeLists.txt` (NEW)

```cmake
# Decoupled Replay Binary
# Links ONLY against ns3 libraries -- NO astra-sim dependency
# Uses ns3 scratch build conventions

set(SOURCES
    main.cc
)

# Use ns3's create_scratch function for consistent linking
create_scratch(${SOURCES})

# Additional include directories for ns3 internal headers
target_include_directories(scratch_decoupled_replay PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/point-to-point
    ${CMAKE_SOURCE_DIR}/src/applications
)

# NOTE: create_scratch() auto-links ns3 core libraries.
# If additional ns3 modules are needed, add explicit target_link_libraries.
# Verify build with: cmake --build . --target scratch_decoupled_replay
```

**Build verification:**
```bash
# Must return empty: no astra-sim symbols linked
nm scratch_decoupled_replay | grep -i simai
```

---

#### 9. `common_types.h` (NEW) -- Shared types and global state

Extracts from `common.h` (SimAI-independent subset):

- `FlowRecord` struct (with `relative_delay_ns` -- the COMPLETE struct, matching all 3 mirror copies)
- Global configuration variables (`cc_mode`, `enable_qcn`, `packet_payload_size`, `buffer_size`, etc.)
- `node_id_to_ip()`, `ip_to_node_id()`
- `NodeContainer n`, `serverAddress`, `portNumber`
- `pairRtt`, `pairBw`, `maxRtt`, `maxBdp`
- `Ipv4AddressHelper` setup variables
- `Topology` container references

**Replaced / removed from SimAI version:**
- `AstraSim::ncclFlowTag` → local `FlowTag` struct (copy-paste the fields needed)
- `AstraSim::sim_request` → local `FlowRequest` struct
- `MockNcclLog` → `std::cerr` / `NS_LOG`
- `Sys::boostedTick()` → `Simulator::Now().GetNanoSeconds()`
- `#ifdef NS3_MTP`, `MtpInterface` → removed (not needed)

**Local FlowTag struct:**
```cpp
struct FlowTag {
    uint32_t current_flow_id = 0;
    bool nvls_on = false;
};
```

---

#### 10. `flow_reader.h` (NEW) -- COMPLETE flow file parsing

**CRITICAL:** This module parses the COMPLETE flow file format (all 21 fields), NOT the partial `ImportFlows()` format. Reference implementation: `loadFlowsFromFile()` in `MockNcclGroup.cc:2245-2265`.

```cpp
// Flow file format (one line per flow, header line = count):
// flow_id src dest flow_size channel_id chunk_id chunk_count conn_type
// start_time pg maxPacketCount port dport np prev[0..np-1]
// npar parent_flow_id[0..npar-1] nchi child_flow_id[0..nchi-1]
// layer_num group_type op loopstate relative_delay_ns

struct FlowFileRecord {
    uint32_t flow_id, src, dst;
    uint64_t flow_size;
    int channel_id, chunk_id, chunk_count;
    std::string conn_type;
    double start_time;           // always 0.0 in decoupled mode
    uint32_t pg, maxPacketCount, port, dport;
    std::vector<uint32_t> prev;
    std::vector<int> parent_flow_id;
    std::vector<int> child_flow_id;
    uint32_t layer_num, group_type, op, loopstate;
    uint64_t relative_delay_ns;  // NEW FIELD

    // Validation
    bool valid() const { return flow_id != 0 || src != dst || flow_size > 0; }
};

std::vector<FlowFileRecord> LoadFlows(const std::string& flow_file_path) {
    // 1. Open file, read header count
    // 2. Parse each line: all 21 fields (see format above)
    // 3. Handle legacy format (no relative_delay_ns): default to 0
    // 4. Return vector of FlowFileRecord
}
```

**Backward compatibility:** If the last field is missing (legacy format without `relative_delay_ns`), default to `0`. Detection: after reading `loopstate`, check if stream still has data; if yes, read `relative_delay_ns`; if no, set to 0.

---

#### 11. `topology_reader.h` (NEW) -- Topology and config reading

Extracts from `common.h:481-687+` (ReadConf + surrounding setup):
- `ReadConf(network_topo, network_conf)` -- reads SimAI.conf
- `SetupNetwork()` -- builds NS3 network from topology XML + config
- All RDMA/switch/node setup logic
- Global route manager setup

**Removed from SimAI version:**
- `flow_file` handling → moved to flow_reader.h
- SimAI-specific config keys

---

#### 12. `flow_sender.h` (NEW) -- RDMA flow injection

Extracts from `entry.h:109-166` (SendFlow):
- `SendFlow(src, dst, maxPacketCount, ...)` -- creates RdmaClientHelper, installs on node
- `notify_receiver_receive_data()` -- receiver-side completion tracking
- `is_receive_finished()`, `is_sending_finished()`
- Tag/flow_id tracking (`sender_src_port_map`, `waiting_to_notify_receiver`, etc.)
- `_QPS_PER_CONNECTION_` constant (hardcoded to 1 for decoupled replay)

**Key changes from SimAI version:**
- Remove `AstraSim::sim_request*` parameter → use local `FlowRequest*`
- Replace `AstraSim::ncclFlowTag` with local `FlowTag`
- Replace `request->flowTag.current_flow_id` with local flow_id
- Replace `Sys::boostedTick()` with `Simulator::Now().GetNanoSeconds()`
- Handle send latency via `AS_SEND_LAT` env var (keep existing behavior)
- No MockNcclLog dependency
- Assert `_QPS_PER_CONNECTION_ == 1` at startup (see Q6 resolution)

---

#### 13. `fct_writer.h` (NEW) -- FCT output

Extracts from `entry.h:410-457` (qp_finish):
- `qp_finish(FILE* fout, Ptr<RdmaQueuePair> q)` -- writes FCT records
- `send_finish()` callback
- `last_flow_finish_ns` tracking
- FCT file format: `src_ip dst_ip sport dport size start_time fct standalone_fct`
- `expeRecvHash` setup for receive expectation

**Removed from SimAI version:**
- `AstraSim::ncclFlowTag` → local `FlowTag`
- `MockNcclLog` → `NS_LOG` or `std::cout`

**Flow completion notification:** `qp_finish()` must call `DepScheduler::OnFlowCompleted(flow_id)` so the dependency graph scheduler can unblock dependent flows.

---

#### 14. `dep_scheduler.h` (NEW) -- Dependency graph scheduler with layer constraint [CORE NEW LOGIC]

```cpp
class DepScheduler {
public:
    struct FlowState {
        FlowFileRecord record;
        bool completed = false;
        bool scheduled = false;
        uint64_t scheduled_time = 0;   // ns
        uint64_t completed_time = 0;   // ns
        int pending_deps = 0;          // count of uncompleted prev[] flows
        std::vector<uint32_t> dependents;  // flows that list this flow in their prev[]
    };

    void Init(const std::vector<FlowFileRecord>& flows);
    void ScheduleReadyFlows();         // called at sim start and on each completion
    void OnFlowCompleted(uint32_t flow_id);  // callback from qp_finish

    bool AllCompleted() const;
    uint64_t LastCompletionTime() const;

    // Layer tracking (GAP FIX G4)
    int MaxLayer() const;
    bool LayerComplete(int layer_num) const;

private:
    std::unordered_map<uint32_t, FlowState> _states;
    uint32_t _total_flows = 0;
    uint32_t _completed_flows = 0;
    std::set<uint32_t> _in_flight;

    // Layer constraint (GAP FIX G4):
    // Tracks which layers are "unlocked". Layer N+1 is unlocked only when
    // ALL flows in Layer N have completed.
    int _current_unlocked_layer = 0;
    int _max_layer = 0;
    std::map<int, int> _layer_flow_count;    // layer_num → total flows
    std::map<int, int> _layer_completed;     // layer_num → completed flows
};
```

**Algorithm (revised with G4 fix):**

1. `Init()`:
   - Parse all flows, build `_states` map
   - For each flow, set `pending_deps = record.prev.size()`
   - For each `pid` in `record.prev`, add `record.flow_id` to `_states[pid].dependents`
   - Count flows per layer: `_layer_flow_count[record.layer_num]++`
   - Set `_max_layer` to the maximum `layer_num` observed
   - If a `prev[]` reference points to a non-existent flow, emit ERROR and abort

2. `ScheduleReadyFlows()`:
   - Find flows where `pending_deps == 0 && !scheduled`
   - **Layer constraint (G4):** Additionally require `record.layer_num <= _current_unlocked_layer`
     - Layer 0 is always unlocked
     - Higher layers are unlocked only when the previous layer is fully complete
   - For each ready flow: schedule `SendFlow()` after `relative_delay_ns` using `Simulator::Schedule(NanoSeconds(delay), ...)`
   - Mark as `scheduled`, add to `_in_flight`

3. `OnFlowCompleted(flow_id)`:
   - Mark flow as `completed`, record `completed_time`
   - For each `dep` in `_states[flow_id].dependents`: decrement `pending_deps`
   - Remove from `_in_flight`
   - Increment `_layer_completed[record.layer_num]`
   - **Layer unlock check (G4):** If `_layer_completed[_current_unlocked_layer] == _layer_flow_count[_current_unlocked_layer]`, increment `_current_unlocked_layer`
   - Call `ScheduleReadyFlows()` to trigger newly-ready flows

4. `AllCompleted()`: Returns `_completed_flows == _total_flows`

5. `VerifyCompletion()`: Asserts all flows completed, reports per-layer stats.

---

#### 15. `main.cc` (NEW) -- Entry point

```cpp
int main(int argc, char *argv[]) {
    // 1. Parse CLI args
    // 2. ReadConf(topology, config)
    // 3. SetupNetwork()
    // 4. LoadFlows(flow_file) → std::vector<FlowFileRecord>
    // 5. DepScheduler::Init(flows)
    //    -- asserts _QPS_PER_CONNECTION_ == 1
    //    -- optionally runs --verify-dag (cycle detection)
    // 6. DepScheduler::ScheduleReadyFlows()
    // 7. Simulator::Stop(Seconds(sim_stop_time))
    // 8. Simulator::Run()
    // 9. DepScheduler::VerifyCompletion()
    // 10. Output: FCT file + per-layer/per-group timing stats
    // 11. Simulator::Destroy()
}
```

**CLI arguments:**
```
-f, --flow-file      Path to flow file (required)
-t, --topo-dir       Topology directory (e.g., Spectrum-X_128g_...)
-c, --config         SimAI.conf path (default: topo-dir/SimAI.conf)
-o, --fct-output     FCT output file (default: fct.txt)
-s, --stop-time      Simulator stop time in seconds (default: 2000000000 -- same as coupled mode)
--verify-dag         Run DAG cycle detection before simulation
--dump-layer-stats   Output per-layer timing statistics JSON
```

**`sim_stop_time` (Q5 resolution):** Default to `Seconds(2000000000)` (effectively no limit, same as coupled mode). User can override with `-s` for bounded runs.

---

## Implementation Order

### Phase 1: SimAI-side flow recording (Steps 1-5)

| Step | Files | Description | Dependencies | Acceptance Criteria |
|------|-------|-------------|--------------|---------------------|
| **1** | `common.h` (all 3 copies) | Add `relative_delay_ns` to FlowRecord struct | None | Struct compiles in all 3 locations; existing code unaffected |
| **2a** | `MockNcclGroup.h` | Add `_flow_buffer`, `_flow_send_times`, method declarations | Step 1 | Header compiles |
| **2b** | `MockNcclGroup.cc` | Modify `_writeFlowRecord()` to accept `relative_delay_ns` parameter | Step 1 | Signature change compiles; existing callers updated |
| **2c** | `MockNcclGroup.cc` | Modify `genFlowModels()`/`getFlowModels()` to push to `_flow_buffer` instead of immediate write | Steps 2a, 2b | Flow metadata accumulates in `_flow_buffer`; no early file write |
| **2d** | `MockNcclGroup.cc` | Implement `recordFlowSendTime()` | Step 2a | Stores tick values correctly |
| **2e** | `MockNcclGroup.cc` | Implement `finalizeFlowFile()` -- compute relative delays, write complete format | Steps 2c, 2d | Flow file contains correct `relative_delay_ns` values as last field |
| **2f** | `MockNcclGroup.cc` | Update `~MockNcclGroup()` with warn-if-not-called fallback | Step 2e | Warning printed if explicit call was missed |
| **2g** | `MockNcclGroup.cc` | Update `loadFlowsFromFile()` to read `relative_delay_ns` (with legacy fallback) | Step 1 | `AS_REPLAY_MODE=1` can parse new format files |
| **3** | `AstraSimNetwork.cc::sim_send()` | Add `recordFlowSendTime()` call | Step 2d | Per-flow send times captured during NS3 simulation |
| **4a** | `AstraSimNetwork.cc::main()` | Add explicit `GlobalGroup->finalizeFlowFile()` between `Simulator::Run()` and `Simulator::Destroy()` | Steps 2e, 3 | Flow file finalized explicitly, not via destructor |
| **4b** | `Sys.cc::~Sys()` | Add `GlobalGroup->finalizeFlowFile()` before `delete GlobalGroup` (analytical mode path) | Step 2e | Flow file finalized in analytical mode too (no-op when gated) |
| **5** | CI script (NEW) | Add `scripts/check-common-h-consistency.sh` that diffs all 3 common.h copies | Step 1 | CI fails if any copy diverges |

**Phase 1 Verification:**
- Run existing coupled NS3 simulation with `AS_DECOUPLED_OUTPUT=/tmp/test_flows.txt`
- Verify flow file is generated with correct count header and non-zero `relative_delay_ns` values
- Verify `prev[]`, `parent_flow_id[]`, `child_flow_id[]`, `layer_num`, `group_type`, `op`, `loopstate` fields are populated correctly
- Verify coupled simulation FCT output is identical to before (no regression)
- Verify `AS_REPLAY_MODE=1` still works with both old and new format flow files
- Run analytical mode: verify no flow file is generated unless `AS_DECOUPLED_OUTPUT` is set
- Run `scripts/check-common-h-consistency.sh`: verify all 3 common.h copies are byte-identical

### Phase 2: Independent NS3 binary (Steps 6-12)

| Step | Files | Description | Dependencies | Acceptance Criteria |
|------|-------|-------------|--------------|---------------------|
| **6** | `scratch/decoupled_replay/` dir + `CMakeLists.txt` | Create directory structure, CMake using `create_scratch()` | None | CMake configures without errors; `make` finds ns3 libs |
| **7** | `common_types.h` | Extract FlowRecord + global state + local FlowTag/FlowRequest (no SimAI deps) | Step 6 | Compiles; all type definitions complete |
| **8** | `flow_reader.h` | Parse COMPLETE flow file format (21 fields) with legacy fallback | Step 7 | Reads test flow file; all 21 fields parse correctly; handles missing relative_delay_ns |
| **9** | `topology_reader.h` | ReadConf + SetupNetwork (no SimAI deps) | Step 7 | Network topology builds from POD#N.xml + SimAI.conf |
| **10** | `flow_sender.h` + `fct_writer.h` | SendFlow + qp_finish (no SimAI deps) with DepScheduler callback | Steps 7, 9 | Can create RdmaClient; FCT output written; `OnFlowCompleted` triggered |
| **11** | `dep_scheduler.h` | Dependency graph + layer_num constraint + relative delay scheduling | Steps 7, 8 | DAG validates (no cycles); layer ordering enforced; flows scheduled correctly |
| **12** | `main.cc` | Wire everything: init → schedule → run → verify → output | Steps 8-11 | Binary runs end-to-end; FCT file produced |

**Phase 2 Verification:**
- Binary compiles with `cmake --build` without SimAI symbols
- `nm scratch_decoupled_replay | grep -i simai` returns empty (true independence)
- DAG cycle detection passes on flow file from Phase 1
- Layer ordering: scheduler logs confirm Layer N flows all complete before Layer N+1 starts
- FCT file format matches coupled-mode format exactly
- `_QPS_PER_CONNECTION_ == 1` assertion holds

### Phase 3: Integration testing and verification (Steps 13-15)

| Step | Description | Dependencies | Acceptance Criteria |
|------|-------------|--------------|---------------------|
| **13** | Run coupled simulation → generate flow file → run independent binary → compare FCT | Phases 1-2 | Total training time within 20% of coupled |
| **14** | Run diff analysis: per-flow FCT comparison, per-layer timing breakdown | Step 13 | Differences are explainable (network competition effects) |
| **15** | Regression: verify coupled mode + AS_REPLAY_MODE=1 + analytical mode unchanged | Phases 1-2 | All existing tests pass; no behavioral change |

---

## Resolved Open Questions

### Q1: Where should `finalizeFlowFile()` be called?
**Resolution: Option A** -- Call `GlobalGroup->finalizeFlowFile()` explicitly in `AstraSimNetwork.cc::main()` between `Simulator::Run()` and `Simulator::Destroy()`, BEFORE the existing `delete GlobalGroup`. Also add the call in `Sys::~Sys()` for the analytical path. Keep destructor as fallback with a warning. This ensures the flow file is always finalized while the NS3 simulation state is still valid.

### Q2: Should `AS_DECOUPLED_OUTPUT` be gated to NS3 mode only?
**Resolution: Option A+C combined** -- Gate in `autoEnableFlowOutput()` to check if NS3 mode is active (the `ASTRASimNetwork` type). Additionally, `finalizeFlowFile()` warns if zero send times were recorded (catching the case where a user forces it in analytical mode). Document as "NS3 mode only" in the CLI help and README.

### Q3: Does NVLS flow injection go through `ASTRASimNetwork::sim_send()`?
**Resolution: VERIFIED -- no action needed.** In coupled mode, both NVLS and non-NVLS flows go through the same `Sys::sim_send()` → `ASTRASimNetwork::sim_send()` → `SendFlow()` path. The `nvls_on` flag is metadata passed through `sim_request->flowTag` to the RdmaClient, not a separate code path. The `recordFlowSendTime()` call in `sim_send()` will capture NVLS flows correctly.

### Q4: What are the exact ns3 library cmake target names?
**Resolution: Deferred to implementation -- use `create_scratch()`.** The existing ns3 scratch build system uses a `create_scratch()` function (defined in `scratch/CMakeLists.txt`) that auto-links all necessary ns3 libraries. The independent binary CMakeLists.txt uses this function, which avoids hardcoding library target names. If additional modules are needed, verify the exact names during the build step.

### Q5: How should the independent binary determine `simulator_stop_time`?
**Resolution: Option B** -- Use a very large default, `Seconds(2000000000)`, same as the coupled mode. The `-s` CLI flag allows the user to specify a shorter stop time for bounded runs. Rationale: there is no reliable way to compute a tight bound from flow data without running the simulation first, and an over-large stop time is harmless (the scheduler terminates when all flows complete).

### Q6: Should DepScheduler handle `_QPS_PER_CONNECTION_ > 1`?
**Resolution: Option A with assertion** -- Assume `_QPS_PER_CONNECTION_` stays at 1 for decoupled replay. Add a static_assert or runtime assertion in `main.cc` that verifies this. If multiple QPs per connection are ever needed, the scheduler can be extended at that time with a per-flow QP completion counter. Document this assumption in the code and plan.

---

## ADR: Architecture Decision Record

### Decision
Use **SimAI simulation-time instrumentation** to capture per-flow send times from the coupled event chain, compute `relative_delay_ns = send_time - max(send_time of prev[])`, write these to the flow file as the 21st field, and build an **independent NS3 binary** that schedules flows using a dependency graph with relative delay timers AND a secondary `layer_num` constraint.

### Drivers
1. **HARD**: Independent binary must not link any SimAI (astra-sim) library code
2. **HIGH**: Flow timing must reflect actual SimAI event chain behavior, not analytical estimates
3. **HIGH**: Must preserve backward compatibility with existing coupled, replay, and analytical modes
4. **MEDIUM**: The result should enable network competition feedback (the core value proposition of decoupled replay)
5. **MEDIUM**: Cross-layer ordering must be structurally guaranteed, not dependent on `prev[]` alone

### Alternatives Considered
| Alternative | Why Rejected |
|-------------|--------------|
| Pre-compute relative delays from LogGP formulas | Fragile coupling to LogGP internals; analytical estimates diverge from event-chain behavior |
| Absolute timestamp scheduling | Loses causality; cannot model network feedback; contradicts spec guidance (Round 3) |
| Shared library extraction | Would still link SimAI code -- violates hard constraint |
| Rely solely on prev[] for layer ordering (original plan) | prev[] connects flows within a collective group, not across layers. Layer N+1 flows may have no prev[] entries pointing to Layer N flows, creating a gap. |

### Why Chosen
Option A (with G4 layer constraint fix) is the only approach that satisfies all five decision drivers simultaneously. It captures actual timing from the SimAI event chain (unlike LogGP), preserves causality via dependency graph (unlike absolute timestamps), creates a truly independent binary, and guarantees cross-layer ordering via an explicit layer number gate.

### Consequences
**Positive:**
- Independent binary has clean architecture with no SimAI coupling
- Flow file is a self-contained artifact that can be replayed independently
- Dependency graph + layer constraint naturally enables network competition feedback
- Existing modes are completely unaffected (gated behind `_flow_file.is_open()`)
- Explicit call site for `finalizeFlowFile()` eliminates destructor-ordering risk
- CI diff-check script prevents common.h drift across 3 mirror copies

**Negative:**
- ~180 lines of new SimAI-side instrumentation (moderate but contained)
- Flow file generation becomes two-pass (accumulate during gen, write after sim)
- Independent binary duplicates ~400 lines of SendFlow/qp_finish/topology code (justified by independence requirement)
- 3 mirror copies of common.h require synchronized maintenance
- DepScheduler complexity increased by layer constraint tracking

**Risk mitigation:**
- Keep duplicated code in clearly-named files with `// Extracted from: ...` comments
- Use the same FlowRecord definition in all 3 common.h copies (enforced by CI diff-check)
- Explicit `finalizeFlowFile()` call in main() eliminates destructor-ordering risk entirely
- Layer constraint complements (does not replace) prev[] dependencies

### Follow-ups
1. **CI integration**: Add decoupled replay test to the build pipeline (generate flow file, replay, compare FCT)
2. **Flow file format versioning**: Consider adding a format version header for future extensibility (the `relative_delay_ns` field is backward-compatible as the last field)
3. **Multi-threaded scheduler**: If flow count grows large (>100K), the DepScheduler may need optimized data structures
4. **Chakra integration**: Replay from Chakra execution traces as an alternative flow source
5. **common.h consolidation**: Long-term, consider consolidating the 3 mirror copies into a single canonical header via ns3 include path configuration

---

## Test Plan

### Unit Tests (SimAI side)

| Test | What it verifies | How |
|------|-----------------|-----|
| `test_flow_buffer_accumulation` | genFlowModels accumulates flows in `_flow_buffer` when `_flow_file` is open | Run genFlowModels with AS_DECOUPLED_OUTPUT set; assert `_flow_buffer` size matches expected flow count |
| `test_record_flow_send_time` | `recordFlowSendTime()` stores correct tick values | Call recordFlowSendTime with known flow_ids; verify map entries |
| `test_relative_delay_computation` | `finalizeFlowFile()` computes correct relative delays | Set up known send times; verify computed `relative_delay_ns` matches expected values |
| `test_no_dependency_flow` | Flows with empty `prev[]` get relative delay = send_time | Verify `relative_delay_ns == send_time` for flows with no dependencies |
| `test_dependency_chain` | Flows with `prev[]` get delay = send_time - max(prev send times) | A → B → C chain; verify B.delay = B.send - A.send; C.delay = C.send - B.send |
| `test_finalize_empty_buffer` | `finalizeFlowFile()` writes "0\n" header when buffer is empty | Call with empty buffer; verify file contains only "0\n" |
| `test_zero_send_time_warning` | Warning emitted when flows have send_time=0 | Call finalize with flows that have no recorded send time; verify stderr contains warning |

### Unit Tests (Independent binary)

| Test | What it verifies | How |
|------|-----------------|-----|
| `test_flow_reader_complete_format` | Parse all 21 fields correctly | Parse flow file with all fields; verify each field value |
| `test_flow_reader_legacy_format` | Legacy flow file (without relative_delay_ns) parses with default 0 | Parse old-format file; verify `relative_delay_ns == 0` for all records |
| `test_flow_reader_missing_fields` | Graceful error on truncated lines | Parse file with missing fields; verify error handling |
| `test_dag_cycle_detection` | Cycle detection catches A→B→A cycles | Create flow set with cycle; assert detection failure |
| `test_dep_scheduler_init` | pending_deps correctly computed | 5-flow graph; verify each flow's pending_deps count |
| `test_dep_scheduler_ready_flows` | Only flows with pending_deps=0 AND unlocked layer are scheduled | After init, verify only Layer 0 flows with pending_deps=0 are ready |
| `test_layer_constraint` | Layer N+1 flows NOT scheduled until all Layer N complete | 2-layer workload; complete Layer 0; verify Layer 1 becomes unlocked |
| `test_layer_constraint_partial` | Layer N partially complete does NOT unlock N+1 | Complete 50% of Layer 0; verify Layer 1 remains locked |
| `test_dep_scheduler_cascade` | Completing flow unblocks dependents | Complete flow A; verify B's pending_deps decreases; B scheduled when ready |
| `test_dep_scheduler_all_completed` | AllCompleted() true only when all flows done | Partial completion; verify AllCompleted() = false |

### Integration Tests

| Test | What it verifies | How |
|------|-----------------|-----|
| `test_coupled_unchanged` | Coupled mode produces identical FCT after SimAI changes | Run same workload before/after; diff FCT files |
| `test_replay_mode_unchanged` | AS_REPLAY_MODE=1 works with new flow file format | Run with AS_REPLAY_MODE=1 on flow file with relative_delay_ns; verify |
| `test_flow_file_all_fields` | Flow file contains all 21 fields including parent_flow_id, child_flow_id, group_type, op, loopstate | Parse generated flow file; verify all fields present and correct |
| `test_independent_binary_runs` | Independent binary runs to completion with layer ordering | Run with flow file; verify exit code 0 and FCT file exists |
| `test_fct_format_compatibility` | Independent binary FCT matches coupled FCT format | Compare header/structure of both FCT files |
| `test_common_h_consistency` | All 3 common.h copies are byte-identical | Run `scripts/check-common-h-consistency.sh`; assert exit 0 |

### E2E Tests

| Test | What it verifies | How |
|------|-----------------|-----|
| `test_e2e_small_workload` | Full pipeline: SimAI → flow file → independent replay → FCT | Run microAllReduce (8 GPU); generate flow file; replay; verify FCT |
| `test_e2e_dependency_order` | Flows execute in correct dependency order within layers | Instrument scheduler; verify no flow starts before its prev[] complete |
| `test_e2e_layer_order` | Layer N+1 flows start only after ALL Layer N flows complete | 2+ layer workload; verify layer boundaries respected |
| `test_fct_diff_within_tolerance` | Total training time within 20% of coupled | Compare total FCT from coupled vs. decoupled |
| `test_e2e_nvls_workload` | NVLS flows recorded and replayed correctly | Run workload with NVLS tree allreduce; verify send times recorded |
| `test_e2e_backward_compat` | AS_REPLAY_MODE=1 still works | Run replay mode; verify correct operation |

### Observability

| Metric | What to monitor | How |
|--------|----------------|------|
| Flow file write time | Should not significantly slow SimAI | Log time in `finalizeFlowFile()` |
| Zero send-time count | Warn if flows bypass sim_send() recording | Log count in `finalizeFlowFile()` |
| DepScheduler overhead | Scheduler should not bottleneck replay | Log time in `ScheduleReadyFlows()` vs. actual NS3 sim time |
| Per-layer FCT | Compare coupled vs. decoupled per-layer | Output per-layer timing breakdown JSON via `--dump-layer-stats` |
| Flow injection timing | Verify relative delays are honored | Log actual vs. expected injection time for first 100 flows |
| Layer unlock events | Verify layer constraint working | Log layer unlock timestamps |

### CI Diff-Check Script

File: `scripts/check-common-h-consistency.sh`

```bash
#!/bin/bash
# Verify all 3 copies of common.h are byte-identical (modulo whitespace)
set -e

COPIES=(
  "astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h"
  "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/src/applications/astra-sim/network_frontend/ns3/common.h"
  "astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation/scratch/common.h"
)

BASE="${COPIES[0]}"
for ((i=1; i<${#COPIES[@]}; i++)); do
  if ! diff -q "$BASE" "${COPIES[$i]}" > /dev/null 2>&1; then
    echo "ERROR: ${COPIES[$i]} differs from $BASE"
    diff "$BASE" "${COPIES[$i]}"
    exit 1
  fi
done
echo "OK: All ${#COPIES[@]} common.h copies are identical."
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Destructor ordering issues (G1 FIXED) | LOW (was MEDIUM) | HIGH | `finalizeFlowFile()` called explicitly in `main()` between Run and Destroy; destructor is fallback only |
| Flow file format breakage in mirror copies (G2 FIXED) | LOW (was MEDIUM) | MEDIUM | All 3 copies documented; CI diff-check script prevents drift |
| Incomplete flow parsing in independent binary (G3 FIXED) | LOW (was HIGH) | HIGH | `flow_reader.h` based on `loadFlowsFromFile()` complete parser, not `ImportFlows()` partial parser |
| Cross-layer ordering violation (G4 FIXED) | LOW (was MEDIUM) | HIGH | `DepScheduler` enforces layer_num constraint: Layer N+1 locked until all Layer N flows complete |
| send_time recording misses some flows | LOW | HIGH | Warning logged in `finalizeFlowFile()` if any flow has send_time=0; validation during E2E test |
| Independent binary dependency inversion | MEDIUM | HIGH | Strict `nm` check; CI gate that build fails if SimAI symbols detected |
| Code duplication maintenance burden | HIGH | MEDIUM | ~400 lines duplicated with `// Extracted from:` comments; CI diff-check for common.h |
| Performance regression in coupled mode | LOW | MEDIUM | All new code gated behind `_flow_file.is_open()` check |
| DAG validation false positives | LOW | LOW | `--verify-dag` optional; test on known-good workloads first |
| NVLS flow path bypasses recording | LOW | MEDIUM | Verified: NVLS flows go through `sim_send()`; covered by E2E test |

### Pre-mortem (DELIBERATE mode)

**Scenario 1: "The relative delays are all zero"**
- Cause: `recordFlowSendTime()` never called because `request` is null or `GlobalGroup` is null in `sim_send()`. Or the flow file was opened in analytical mode where `Sys::boostedTick()` always returns 0.
- Prevention: `finalizeFlowFile()` warns if zero_send_count > 0. Gate `autoEnableFlowOutput()` to NS3 mode only. E2E test verifies non-zero delays.

**Scenario 2: "Independent binary hangs forever"**
- Cause: Deadlock in dependency graph -- a flow's `prev[]` references a non-existent flow_id, so it never becomes ready. Or layer constraint never unlocks because a Layer N flow was never scheduled due to missing prev[] dependency.
- Prevention: DAG validation at startup (`--verify-dag`) catches missing prev[] references. Layer constraint timeout: if a layer remains incomplete after all its flows are scheduled and the last one completes, log an error and unlock the next layer anyway (with warning).

**Scenario 3: "FCT diff is > 50%, plan invalidated"**
- Cause: The simplified relative delay model (max prev send time, not completion time) doesn't capture enough timing information. Flows in decoupled mode start much earlier or later than in coupled mode.
- Prevention: Run small-scale test first (8 GPU, microAllReduce). If diff exceeds threshold, pivot to recording per-flow completion times (not send times) in SimAI. The `finalizeFlowFile()` architecture supports this -- only the `_flow_send_times` map content changes, not the overall structure.

---

## Success Criteria

1. [ ] **Flow file correctness**: Generated flow file contains valid `relative_delay_ns` for all flows; all 21 fields present; DAG has no cycles
2. [ ] **All common.h copies synchronized**: `scripts/check-common-h-consistency.sh` passes (all 3 copies byte-identical)
3. [ ] **Independent compilation**: `nm scratch_decoupled_replay | grep -i simai` returns empty
4. [ ] **Layer ordering**: All Layer N flows complete before any Layer N+1 flow starts (verified by scheduler logs and `--dump-layer-stats`)
5. [ ] **FCT output**: Independent binary produces `fct.txt` with correct format (src/dst/size/startTime/fct/standalone_fct)
6. [ ] **Diff tolerance**: Total training time within 20% of coupled mode; per-layer FCT differences are explainable
7. [ ] **Backward compatibility**: Coupled mode, AS_REPLAY_MODE=1, and analytical mode produce identical results before/after changes
8. [ ] **Zero SimAI linkage**: Independent binary build log confirms no astra-sim objects linked
9. [ ] **No destructor reliance**: `finalizeFlowFile()` is called explicitly in `main()`; destructor only serves as fallback with warning
10. [ ] **Explicit finalize call site**: Code review confirms `GlobalGroup->finalizeFlowFile()` appears between `Simulator::Run()` and `Simulator::Destroy()` in `AstraSimNetwork.cc::main()`
