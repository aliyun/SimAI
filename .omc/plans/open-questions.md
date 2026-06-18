# Open Questions: SimAI-NS3 Decoupled Replay

Generated from analyst gap analysis on 2026-06-18.
Source plan: `.omc/plans/decoupled-replay.md`
Source spec: `.omc/specs/deep-interview-decoupled-replay.md`
**Status: ALL RESOLVED (2026-06-18 revision)**

---

## Q1: Where should `finalizeFlowFile()` be called? -- RESOLVED

**Resolution: Option A -- Explicit call in `AstraSimNetwork.cc::main()`.**

`GlobalGroup->finalizeFlowFile()` is called explicitly between `Simulator::Run()` and `Simulator::Destroy()` in `AstraSimNetwork.cc::main()`, BEFORE the existing `delete GlobalGroup`. Also added in `Sys::~Sys()` for the analytical mode path. The destructor `~MockNcclGroup()` retains a fallback call with a warning if the explicit call was somehow missed.

**Rationale:** While `delete GlobalGroup` at `AstraSimNetwork.cc:369` does trigger `~MockNcclGroup()`, relying on destructor ordering is fragile. An explicit call site makes the intent clear and ensures the flow file is finalized while NS3 simulation state is still valid (before `Simulator::Destroy()`).

---

## Q2: Should `AS_DECOUPLED_OUTPUT` be gated to NS3 mode only? -- RESOLVED

**Resolution: Option A+C combined -- Gate in `autoEnableFlowOutput()` + warn in `finalizeFlowFile()`.**

The flow file is only opened when NS3 mode is active (checking the network backend type). If a user somehow forces it in analytical/physical mode, `finalizeFlowFile()` will warn about zero send times recorded. Documentation states "NS3 mode only."

**Rationale:** Sending a flow file with all-zero `relative_delay_ns` to the independent binary produces incorrect (but silently so) results. The gating + warning combination makes it fail loudly.

---

## Q3: Does NVLS flow injection go through `ASTRASimNetwork::sim_send()`? -- RESOLVED

**Resolution: VERIFIED -- No action needed.**

In coupled mode, both NVLS and non-NVLS flows go through the same `Sys::sim_send()` → `ASTRASimNetwork::sim_send()` → `SendFlow()` path. The `nvls_on` flag is metadata passed through `sim_request->flowTag`, not a separate code path. The `recordFlowSendTime()` call in `sim_send()` will capture NVLS flows correctly.

---

## Q4: What are the exact ns3 library cmake target names? -- RESOLVED (deferred)

**Resolution: Use `create_scratch()` function.**

The existing ns3 scratch build system (`scratch/CMakeLists.txt`) provides a `create_scratch()` function that auto-links all necessary ns3 libraries. The independent binary CMakeLists.txt uses this function, avoiding hardcoded library target names. If additional modules are needed beyond what `create_scratch()` provides, the exact target names should be verified against the ns3 build at implementation time.

---

## Q5: How should the independent binary determine `simulator_stop_time`? -- RESOLVED

**Resolution: Option B -- Use large default, same as coupled mode.**

Default to `Simulator::Stop(Seconds(2000000000))` (effectively no limit), matching the coupled mode behavior. The `-s` / `--stop-time` CLI flag allows the user to specify a shorter stop time for bounded runs.

**Rationale:** There is no reliable way to compute a tight bound from flow data without running the simulation first. An over-large stop time is harmless -- the `DepScheduler` terminates simulation when all flows complete. A too-short stop time (from `SimAI.conf`'s default of 3.01 seconds) would silently truncate late-scheduled flows.

---

## Q6: Should DepScheduler handle `_QPS_PER_CONNECTION_ > 1`? -- RESOLVED

**Resolution: Option A with assertion -- Assume QPS=1 for decoupled replay.**

The scheduler assumes a single QP per flow. A runtime assertion in `main.cc` verifies `_QPS_PER_CONNECTION_ == 1` at startup. If multiple QPs per connection are ever needed, the scheduler can be extended with a per-flow QP completion counter.

**Rationale:** `_QPS_PER_CONNECTION_` is `#define`d to 1 and has never been changed. Building support for a hypothetical future change would add complexity with no current benefit. The assertion ensures we fail loudly if the assumption is ever violated.
