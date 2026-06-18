/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Layer-constrained flow scheduler with dependency graph.
 * [CORE NEW LOGIC - not extracted from SimAI]
 *
 * === Scheduling model ===
 *
 * Three gates control when a flow is injected into NS3:
 *
 *   1. Flow dependency (hard gate):
 *      A flow is eligible only after ALL flows listed in its prev[] field
 *      have completed.  This restores the per-flow ordering that SimAI's
 *      event chain enforces in coupled mode.
 *
 *   2. Layer constraint (hard gate):
 *      A flow is only eligible when its layer is unlocked.  Layer N+1
 *      unlocks when ALL flows in Layer N have completed.  Layer 0 is
 *      always unlocked.  This guarantees cross-layer ordering.
 *
 *   3. relative_delay_ns (soft gate):
 *      After a flow becomes eligible (both gates above satisfied), it is
 *      scheduled via Simulator::Schedule(NanoSeconds(relative_delay_ns), ...).
 *
 *      relative_delay_ns = send_time - max(prev[] QP completion times),
 *      computed by SimAI Phase 1 (MockNcclGroup::finalizeFlowFile).
 *
 *      Because the delay is measured from the moment the LAST predecessor
 *      completed (when the flow becomes eligible), tightly-coupled flows
 *      get delay ≈ 0 and run with realistic overlap.  Gapped flows get a
 *      positive delay that reproduces the coupled-mode spacing.
 *
 * === Why the dependency graph is necessary ===
 *
 * Delta-based timing: relative_delay_ns is only meaningful when measured
 * from the correct baseline -- the moment the last prev[] flow completed.
 * Without the dependency graph, the baseline would be the layer-unlock
 * instant, which could be much later (if intra-layer dependencies exist)
 * or much earlier (if layer N-1 is very fast but our predecessor is slow).
 *
 * The flow dependency graph paired with completion-based relative_delay_ns
 * reproduces the coupled simulation timeline: eligible_time + delay_ns
 * equals the original send_time from the coupled run.
 *
 * === Key invariants ===
 *
 *   - Single QP per flow (_QPS_PER_CONNECTION_ == 1)
 *   - Layer 0 is always unlocked; higher layers unlock sequentially
 *   - prev[] entries are flow IDs (not rank IDs) in the recorded flow file
 *   - relative_delay_ns is computed from completion times, not send times
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
#include <functional>
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

        // Dependency graph
        int pending_deps = 0;              // count of uncompleted prev[] flows
        std::vector<uint32_t> dependents;  // flows that list this flow in their prev[]

        // For expeRecvHash: buffer allocated for receive expectation
        char* recv_buffer = nullptr;
    };

    DepScheduler() = default;

    // ------------------------------------------------------------------
    // Init: build dependency graph and layer tracking from loaded flows
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

        // Build flow_id -> index
        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < flows.size(); i++) {
            id_to_idx[flows[i].flow_id] = i;
        }

        // Pass 1: create FlowState for each flow
        for (const auto& rec : flows) {
            FlowState st;
            st.record = rec;
            st.pending_deps = (int)rec.prev.size();

            _layer_flow_count[rec.layer_num]++;
            if ((int)rec.layer_num > _max_layer) {
                _max_layer = (int)rec.layer_num;
            }

            _states[rec.flow_id] = st;
        }

        // Pass 2: build dependents (reverse edges) and validate prev[] references
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            FlowState& st = kv.second;
            for (uint32_t pid : st.record.prev) {
                if (_states.find(pid) == _states.end()) {
                    std::cerr << "[DepScheduler] ERROR: flow " << fid
                              << " depends on non-existent flow " << pid << std::endl;
                    return false;
                }
                _states[pid].dependents.push_back(fid);
            }
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
            _request_by_id[rec.flow_id] = _requests[i];
        }

        // Setup expeRecvHash for each flow (receive expectation)
        for (const auto& rec : flows) {
            int tag = (int)rec.flow_id;
            auto key = std::make_pair(tag, std::make_pair((int)rec.src, (int)rec.dst));
            if (expeRecvHash.find(key) == expeRecvHash.end()) {
                RecvTask t;
                t.src = (int)rec.src;
                t.dest = (int)rec.dst;
                t.count = rec.flow_size;
                t.type = 1;
                t.msg_handler = [](void*){ /* no-op: completion tracked via g_on_flow_completed */ };
                char* buf = new char[168]();
                t.fun_arg = buf;
                expeRecvHash[key] = t;

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
    // ScheduleReadyFlows: find eligible flows and schedule them
    // ------------------------------------------------------------------
    void ScheduleReadyFlows() {
        int scheduled_this_round = 0;
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            FlowState& st = kv.second;

            if (st.scheduled || st.completed) continue;

            // Gate 1: all prev[] dependencies must be satisfied
            if (st.pending_deps > 0) continue;

            // Gate 2: layer must be unlocked (G4)
            if ((int)st.record.layer_num > _current_unlocked_layer) {
                continue;
            }

            // Gate 3: relative_delay_ns measured from NOW (the instant
            // the flow became eligible, i.e. when its last predecessor completed
            // and its layer was already unlocked).
            uint64_t delay_ns = st.record.relative_delay_ns;
            st.scheduled = true;
            st.scheduled_time_ns = Simulator::Now().GetNanoSeconds() + delay_ns;
            _in_flight.insert(fid);

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
            return;  // already completed (can happen with multiple QPs)
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

        // Decrement pending_deps for all dependent flows
        for (uint32_t dep_fid : st.dependents) {
            auto dep_it = _states.find(dep_fid);
            if (dep_it != _states.end()) {
                dep_it->second.pending_deps--;
                NS_LOG_DEBUG("[DepScheduler] Flow " << dep_fid
                             << " pending_deps now " << dep_it->second.pending_deps);
            }
        }

        // Layer unlock check (G4):
        // If the current unlocked layer is fully complete, unlock the next layer.
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

        // Schedule newly-ready flows (dependency-unblocked + layer-unlocked)
        ScheduleReadyFlows();
    }

    // ------------------------------------------------------------------
    // DoSendFlow: public callback from Simulator::Schedule
    // ------------------------------------------------------------------
    void DoSendFlow(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) return;

        FlowState& st = it->second;
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
                 [](void*){}, nullptr,
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
    // VerifyCompletion
    // ------------------------------------------------------------------
    bool VerifyCompletion() {
        if (!AllCompleted()) {
            std::cerr << "[DepScheduler] WARNING: Not all flows completed! "
                      << _completed_flows << "/" << _total_flows << std::endl;

            for (const auto& kv : _states) {
                if (!kv.second.completed) {
                    std::cerr << "  Incomplete: flow " << kv.first
                              << " layer=" << kv.second.record.layer_num
                              << " pending_deps=" << kv.second.pending_deps
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
    // DumpLayerStats
    // ------------------------------------------------------------------
    void DumpLayerStats() {
        std::cout << "\n========== Per-Layer Timing Statistics ==========\n";
        std::map<int, std::vector<uint64_t>> layer_fcts;

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
    // VerifyDAG: DFS-based cycle detection on the dependency graph
    // ------------------------------------------------------------------
    bool VerifyDAG() {
        std::unordered_map<uint32_t, int> color; // 0=white, 1=gray, 2=black
        for (const auto& kv : _states) color[kv.first] = 0;

        std::function<bool(uint32_t, std::vector<uint32_t>&)> dfs =
            [&](uint32_t fid, std::vector<uint32_t>& path) -> bool {
            color[fid] = 1; // in progress
            path.push_back(fid);

            const FlowState& st = _states[fid];
            for (uint32_t dep_fid : st.dependents) {
                if (color[dep_fid] == 1) {
                    // Cycle detected
                    std::cerr << "[DepScheduler] ERROR: Cycle detected: ";
                    bool in_cycle = false;
                    for (uint32_t n : path) {
                        if (n == dep_fid) in_cycle = true;
                        if (in_cycle) std::cerr << n << " ";
                    }
                    std::cerr << dep_fid << std::endl;
                    return false;
                }
                if (color[dep_fid] == 0) {
                    if (!dfs(dep_fid, path)) return false;
                }
            }

            path.pop_back();
            color[fid] = 2; // done
            return true;
        };

        for (const auto& kv : _states) {
            if (color[kv.first] == 0) {
                std::vector<uint32_t> path;
                if (!dfs(kv.first, path)) {
                    std::cerr << "[DepScheduler] DAG verification FAILED" << std::endl;
                    return false;
                }
            }
        }

        std::cout << "[DepScheduler] DAG verification PASSED (no cycles)" << std::endl;
        return true;
    }

    // ------------------------------------------------------------------
    // Cleanup
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
    std::vector<FlowRequest*> _requests;
    std::unordered_map<uint32_t, FlowRequest*> _request_by_id;
};

#endif // __DECOUPLED_DEP_SCHEDULER_H__
