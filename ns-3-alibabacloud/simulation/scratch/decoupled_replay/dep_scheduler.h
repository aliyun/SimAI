/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Layer-constrained flow scheduler.
 * [CORE NEW LOGIC - not extracted from SimAI]
 *
 * === Scheduling model ===
 *
 * Two gates control when a flow is injected into NS3:
 *
 *   1. Layer constraint (hard gate):
 *      Layer N+1 is locked until ALL flows in Layer N have completed.
 *      This replaces the old per-flow dependency graph. Layer 0 is always
 *      unlocked. The layer-unlock gate fires inside OnFlowCompleted() when
 *      _layer_completed[layer] >= _layer_flow_count[layer].
 *
 *   2. relative_delay_ns (soft gate):
 *      After a flow's layer unlocks and it becomes eligible, it is scheduled
 *      via Simulator::Schedule(NanoSeconds(relative_delay_ns), ...).
 *      relative_delay_ns = send_time - max(prev[] completion times),
 *      computed by SimAI Phase 1 (MockNcclGroup::finalizeFlowFile).
 *
 * === Why no flow-level dependency graph ===
 *
 * Double-counting: the old design used both pending_deps (hard gate:
 * predecessor must complete) AND relative_delay_ns (soft gate: wait N ns
 * after predecessor becomes eligible). Since relative_delay_ns was computed
 * from send times, the hard gate added the predecessor's entire wire time
 * on top of the already-recorded SimAI gap, eliminating flow overlap.
 *
 * Single-gate fix: remove the dependency graph entirely. Causality is fully
 * encoded in relative_delay_ns by using completion times instead of send
 * times in the Phase 1 computation.
 *
 *   relative_delay_ns = send_time - max(prev[] QP completion times)
 *
 *   Predecessor completed before we sent → delay = 0  (overlap preserved)
 *   Predecessor still in-flight when we sent  → delay > 0  (real gap)
 *
 * === Why prev[] was never usable as flow dependencies ===
 *
 * SingleFlow.prev contains rank IDs (0, 1, 2, ...), not flow IDs (10423,
 * 10424, ...). Every flow model generator (ring, tree, NVLS, alltoall)
 * populates prev with ncclTree.rank / ring member indices / group ranks.
 * The SimAI event chain uses these for receive-side rank-based coordination
 * (free_packets[channel_id][rank]), not flow-level sequencing.
 *
 * Flow-level sequencing lives in parent_flow_id / child_flow_id (tree/NVLS)
 * and in the implicit linear order of g_flow_id assignment.
 *
 * === Known limitation: PXN flows ===
 *
 * PXN proxy flows (conn_type="PXN_INIT") have prev = {sender_rank} but
 * their true inter-flow dependency is parent_flow_id. With the dependency
 * graph removed this is now harmless -- the proxy flow's relative_delay_ns
 * computed from its parent's completion time provides correct ordering.
 *
 * === Key invariants ===
 *
 *   - Single QP per flow (SendFlow inlined, no multi-QP loop)
 *   - Layer 0 is always unlocked; higher layers unlock sequentially
 *   - No flow_id circulation -- prev[] entries are ignored
 *   - relative_delay_ns encodes all causality; no secondary ordering needed
 */

#ifndef __DECOUPLED_DEP_SCHEDULER_H__
#define __DECOUPLED_DEP_SCHEDULER_H__

#include "common_types.h"
#include "flow_reader.h"
#include "flow_sender.h"

#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

// ============================================================================
// DepScheduler
// ============================================================================

class DepScheduler {
public:
    struct FlowState {
        FlowFileRecord record;
        bool completed = false;
        bool scheduled = false;
        uint64_t scheduled_time_ns = 0;   // when SendFlow was called
        uint64_t completed_time_ns = 0;   // when qp_finish completed the flow

        // For expeRecvHash: buffer allocated for receive expectation
        char* recv_buffer = nullptr;
    };

    DepScheduler() = default;

    // ------------------------------------------------------------------
    // Init: build dependency graph from loaded flows
    // ------------------------------------------------------------------
    bool Init(const std::vector<FlowFileRecord>& flows) {
        _total_flows = flows.size();
        _completed_flows = 0;
        _states.clear();
        _in_flight.clear();
        _layer_flow_count.clear();
        _layer_completed.clear();
        _requests.clear();
        _request_by_id.clear();
        _current_unlocked_layer = 0;
        _max_layer = 0;

        // Build flow_id -> index in _states
        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < flows.size(); i++) {
            id_to_idx[flows[i].flow_id] = i;
        }

        // Create FlowState for each flow
        // No flow-level dependency graph — causality is encoded in relative_delay_ns
        // computed from send times minus predecessor completion times in Phase 1.
        // Only the layer constraint gates flow scheduling.
        for (const auto& rec : flows) {
            FlowState st;
            st.record = rec;

            _layer_flow_count[rec.layer_num]++;
            if ((int)rec.layer_num > _max_layer) {
                _max_layer = (int)rec.layer_num;
            }

            _states[rec.flow_id] = st;
        }

        // Allocate FlowRequest objects for each flow (heap-allocated, stable pointers
        // for the simulation lifetime -- SendFlow stores the pointer in sender_src_port_map).
        _requests.resize(_total_flows);
        _request_by_id.clear();
        for (size_t i = 0; i < flows.size(); i++) {
            _requests[i] = new FlowRequest();
            const auto& rec = flows[i];
            _requests[i]->flowTag.current_flow_id = rec.flow_id;
            _requests[i]->flowTag.nvls_on =
                (rec.conn_type == "NVLS" || rec.conn_type == "NVLS_TREE");
            _requests[i]->srcRank = rec.src;
            _requests[i]->dstRank = rec.dst;
            _requests[i]->reqCount = rec.flow_size;
            _request_by_id[rec.flow_id] = _requests[i];  // O(1) lookup for DoSendFlow
        }

        // Setup expeRecvHash for each flow (receive expectation)
        for (const auto& rec : flows) {
            int tag = (int)rec.flow_id;
            auto key = std::make_pair(tag, std::make_pair((int)rec.src, (int)rec.dst));
            if (expeRecvHash.find(key) == expeRecvHash.end()) {
                RecvTask t;
                t.src = (int)rec.src;
                t.dest = (int)rec.dst;
                t.count = rec.flow_size;  // expected byte count
                t.type = 1;
                t.msg_handler = [](void*){ /* no-op: completion tracked via g_on_flow_completed */ };
                // Allocate a minimal buffer for the receive handler
                char* buf = new char[168]();
                t.fun_arg = buf;
                expeRecvHash[key] = t;

                // Store buffer pointer for cleanup
                if (id_to_idx.count(rec.flow_id)) {
                    _states[rec.flow_id].recv_buffer = buf;
                }
            }
        }

        std::cout << "[DepScheduler] Init: " << _total_flows << " flows, "
                  << (_max_layer + 1) << " layers" << std::endl;

        return true;
    }

    // ------------------------------------------------------------------
    // ScheduleReadyFlows: find flows with all deps satisfied and schedule them
    // ------------------------------------------------------------------
    void ScheduleReadyFlows() {
        int scheduled_this_round = 0;
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            FlowState& st = kv.second;

            if (st.scheduled || st.completed) continue;

            // Layer constraint (G4): only schedule flows in unlocked layers
            if ((int)st.record.layer_num > _current_unlocked_layer) {
                continue;
            }

            // Schedule this flow after relative_delay_ns
            uint64_t delay_ns = st.record.relative_delay_ns;
            st.scheduled = true;
            st.scheduled_time_ns = Simulator::Now().GetNanoSeconds() + delay_ns;
            _in_flight.insert(fid);

            // Use a member function callback bound with the flow_id
            Simulator::Schedule(NanoSeconds(delay_ns),
                              &DepScheduler::DoSendFlow, this, fid);

            scheduled_this_round++;
        }

        if (scheduled_this_round > 0) {
            NS_LOG_DEBUG("[DepScheduler] Scheduled " << scheduled_this_round
                         << " flows this round, " << _in_flight.size()
                         << " total in-flight");
        }
    }

    // ------------------------------------------------------------------
    // OnFlowCompleted: called from qp_finish via g_on_flow_completed
    // ------------------------------------------------------------------
    void OnFlowCompleted(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) {
            std::cerr << "[DepScheduler] WARNING: OnFlowCompleted for unknown flow "
                      << flow_id << std::endl;
            return;
        }

        FlowState& st = it->second;
        if (st.completed) {
            std::cerr << "[DepScheduler] WARNING: flow " << flow_id
                      << " already completed" << std::endl;
            return;
        }

        st.completed = true;
        st.completed_time_ns = Simulator::Now().GetNanoSeconds();
        _completed_flows++;
        _in_flight.erase(flow_id);
        _layer_completed[st.record.layer_num]++;

        int layer = (int)st.record.layer_num;
        NS_LOG_DEBUG("[DepScheduler] Flow " << flow_id << " completed (layer "
                     << layer << "), " << _completed_flows << "/"
                     << _total_flows << " total, in-flight: "
                     << _in_flight.size());

        // Layer unlock check (G4):
        // If the current unlocked layer is fully complete, unlock the next layer
        while (_current_unlocked_layer <= _max_layer) {
            int total_in_layer = _layer_flow_count[_current_unlocked_layer];
            int completed_in_layer = _layer_completed[_current_unlocked_layer];
            if (total_in_layer > 0 && completed_in_layer >= total_in_layer) {
                NS_LOG_DEBUG("[DepScheduler] Layer " << _current_unlocked_layer
                             << " complete (" << completed_in_layer << "/"
                             << total_in_layer << "), unlocking layer "
                             << (_current_unlocked_layer + 1));
                _current_unlocked_layer++;
            } else {
                break;
            }
        }

        // Schedule newly-ready flows
        ScheduleReadyFlows();
    }

    // ------------------------------------------------------------------
    // DoSendFlow: public callback from Simulator::Schedule (must be public
    // for ns3's MakeEvent / Simulator::Schedule with member-function pointers).
    // ------------------------------------------------------------------
    void DoSendFlow(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) return;

        FlowState& st = it->second;
        // O(1) lookup via map populated in Init()
        auto rit = _request_by_id.find(flow_id);
        if (rit == _request_by_id.end()) {
            std::cerr << "[DepScheduler] ERROR: no request found for flow "
                      << flow_id << std::endl;
            return;
        }
        FlowRequest* req = rit->second;

        NS_LOG_DEBUG("[DepScheduler] Sending flow " << flow_id
                     << " src=" << st.record.src << " dst=" << st.record.dst
                     << " size=" << st.record.flow_size
                     << " delay_ns=" << st.record.relative_delay_ns
                     << " at tick=" << Simulator::Now().GetNanoSeconds());

        SendFlow((int)st.record.src, (int)st.record.dst,
                 st.record.flow_size,
                 [](void*){}, nullptr,  // no-op msg_handler for decoupled replay
                 (int)flow_id, req);
    }

    // ------------------------------------------------------------------
    // AllCompleted
    // ------------------------------------------------------------------
    bool AllCompleted() const {
        return _completed_flows == _total_flows;
    }

    // ------------------------------------------------------------------
    // LastCompletionTime
    // ------------------------------------------------------------------
    uint64_t LastCompletionTime() const {
        uint64_t max_t = 0;
        for (const auto& kv : _states) {
            if (kv.second.completed_time_ns > max_t) {
                max_t = kv.second.completed_time_ns;
            }
        }
        return max_t;
    }

    // ------------------------------------------------------------------
    // VerifyCompletion: assert all flows completed, report per-layer stats
    // ------------------------------------------------------------------
    bool VerifyCompletion() {
        if (!AllCompleted()) {
            std::cerr << "[DepScheduler] WARNING: Not all flows completed! "
                      << _completed_flows << "/" << _total_flows << std::endl;

            // Report which flows are incomplete
            for (const auto& kv : _states) {
                if (!kv.second.completed) {
                    std::cerr << "  Incomplete: flow " << kv.first
                              << " layer=" << kv.second.record.layer_num
                              << " scheduled=" << kv.second.scheduled
                              << std::endl;
                }
            }
            return false;
        }

        std::cout << "[DepScheduler] All " << _total_flows << " flows completed." << std::endl;
        std::cout << "[DepScheduler] Last completion time: "
                  << LastCompletionTime() << " ns" << std::endl;

        return true;
    }

    // ------------------------------------------------------------------
    // DumpLayerStats: output per-layer timing statistics
    // ------------------------------------------------------------------
    void DumpLayerStats() {
        std::cout << "\n========== Per-Layer Timing Statistics ==========\n";
        std::map<int, std::vector<uint64_t>> layer_fcts;  // layer_num -> list of FCTs

        for (const auto& kv : _states) {
            const FlowState& st = kv.second;
            if (st.completed && st.scheduled_time_ns > 0) {
                uint64_t fct = st.completed_time_ns - st.scheduled_time_ns;
                layer_fcts[st.record.layer_num].push_back(fct);
            }
        }

        for (int l = 0; l <= _max_layer; l++) {
            auto it = layer_fcts.find(l);
            if (it == layer_fcts.end() || it->second.empty()) {
                std::cout << "Layer " << l << ": no completed flows\n";
                continue;
            }

            const auto& fcts = it->second;
            uint64_t sum = 0, min_fct = UINT64_MAX, max_fct = 0;
            for (uint64_t f : fcts) {
                sum += f;
                if (f < min_fct) min_fct = f;
                if (f > max_fct) max_fct = f;
            }
            uint64_t avg = sum / fcts.size();

            int total_in_layer = _layer_flow_count[l];
            int completed_in_layer = _layer_completed[l];

            std::cout << "Layer " << l << ": "
                      << completed_in_layer << "/" << total_in_layer << " flows, "
                      << "avg FCT=" << avg << " ns, "
                      << "min=" << min_fct << " ns, "
                      << "max=" << max_fct << " ns\n";
        }
        std::cout << "==================================================\n";
    }

    // ------------------------------------------------------------------
    // VerifyDAG: no-op — no flow-level dependency graph exists.
    // Causality is encoded in relative_delay_ns (completion-based timing).
    // ------------------------------------------------------------------
    bool VerifyDAG() {
        std::cout << "[DepScheduler] DAG verification skipped (completion-based timing)." << std::endl;
        return true;
    }

    // ------------------------------------------------------------------
    // Cleanup: free allocated resources
    // ------------------------------------------------------------------
    ~DepScheduler() {
        for (auto& kv : _states) {
            delete[] kv.second.recv_buffer;
        }
        for (auto* req : _requests) {
            delete req;
        }
    }

private:
    std::unordered_map<uint32_t, FlowState> _states;
    uint32_t _total_flows = 0;
    uint32_t _completed_flows = 0;
    std::set<uint32_t> _in_flight;

    // Layer constraint (GAP FIX G4)
    int _current_unlocked_layer = 0;
    int _max_layer = 0;
    std::map<int, int> _layer_flow_count;    // layer_num -> total flows
    std::map<int, int> _layer_completed;     // layer_num -> completed flows

    // Heap-allocated FlowRequest objects (lifetime = simulation duration)
    // _request_by_id provides O(1) lookup for DoSendFlow.
    std::vector<FlowRequest*> _requests;
    std::unordered_map<uint32_t, FlowRequest*> _request_by_id;
};

#endif // __DECOUPLED_DEP_SCHEDULER_H__
