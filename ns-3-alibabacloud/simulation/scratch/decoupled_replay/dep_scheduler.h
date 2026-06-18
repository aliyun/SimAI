/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Dependency graph scheduler with layer constraint.
 * [CORE NEW LOGIC - not extracted from SimAI]
 *
 * Schedules flows based on:
 *   1. parent_flow_id / prev[] (rank-ID) dependencies (all predecessors must complete)
 *   2. layer_num constraint (Layer N+1 locked until ALL Layer N flows complete)
 *   3. relative_delay_ns (delay after dependencies satisfied, before sending)
 *
 * Dependency resolution:
 *   - Primary: parent_flow_id[] (contains flow_ids from tree/NVLS/PXN models)
 *   - Fallback: prev[] (rank IDs) → binary-search per-rank flow_id lists
 *
 * Key invariants:
 *   - _QPS_PER_CONNECTION_ == 1 (asserted at startup)
 *   - Layer 0 is always unlocked; higher layers unlock sequentially
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
// Global callback: called by qp_finish when a flow completes
// Set by main.cc to DepScheduler::OnFlowCompleted
// ============================================================================

extern void (*g_on_flow_completed)(uint32_t flow_id);

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
        int pending_deps = 0;             // count of uncompleted dependency flows
        std::vector<uint32_t> dependents; // flows that depend on this flow

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
        _current_unlocked_layer = 0;
        _max_layer = 0;

        // Build flow_id -> index in _states
        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < flows.size(); i++) {
            id_to_idx[flows[i].flow_id] = i;
        }

        // Build per-rank sorted flow_id lists (for prev[] fallback lookups)
        // Key: rank, Value: sorted vector of flow_ids that involve this rank
        std::unordered_map<uint32_t, std::vector<uint32_t>> rank_to_flow_ids;
        for (const auto& rec : flows) {
            rank_to_flow_ids[rec.src].push_back(rec.flow_id);
            rank_to_flow_ids[rec.dst].push_back(rec.flow_id);
        }
        // Sort + dedup each rank's flow_id list for binary search
        for (auto& kv : rank_to_flow_ids) {
            std::sort(kv.second.begin(), kv.second.end());
            auto last = std::unique(kv.second.begin(), kv.second.end());
            kv.second.erase(last, kv.second.end());
        }

        // Pass 1: create FlowState for each flow, resolving dependencies
        // Primary: parent_flow_id (contains flow_ids)
        // Fallback: prev (contains rank IDs) → find predecessor flow by rank
        std::unordered_map<uint32_t, std::vector<uint32_t>> resolved_deps; // flow_id → dep flow_ids
        for (const auto& rec : flows) {
            FlowState st;
            st.record = rec;
            std::vector<uint32_t> deps;

            // Primary: use parent_flow_id entries (they are flow_ids)
            if (!rec.parent_flow_id.empty()) {
                for (int pfid : rec.parent_flow_id) {
                    if (pfid >= 0 && id_to_idx.count((uint32_t)pfid)) {
                        deps.push_back((uint32_t)pfid);
                    }
                }
            }

            // Fallback: prev[] contains rank IDs; find predecessor flow for each rank
            if (deps.empty() && !rec.prev.empty()) {
                for (uint32_t prev_rank : rec.prev) {
                    auto rit = rank_to_flow_ids.find(prev_rank);
                    if (rit == rank_to_flow_ids.end()) continue;
                    const auto& fids = rit->second;
                    // Find the highest flow_id from this rank that is < rec.flow_id
                    auto lb = std::lower_bound(fids.begin(), fids.end(), rec.flow_id);
                    if (lb != fids.begin()) {
                        deps.push_back(*(lb - 1));
                    }
                }
            }

            st.pending_deps = (int)deps.size();
            resolved_deps[rec.flow_id] = std::move(deps);

            // Track max layer and per-layer flow count
            _layer_flow_count[rec.layer_num]++;
            if ((int)rec.layer_num > _max_layer) {
                _max_layer = (int)rec.layer_num;
            }

            _states[rec.flow_id] = st;
        }

        // Pass 2: build dependents (reverse edges) from resolved dependencies
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            auto dit = resolved_deps.find(fid);
            if (dit == resolved_deps.end()) continue;
            for (uint32_t dep_fid : dit->second) {
                auto dep_it = _states.find(dep_fid);
                if (dep_it != _states.end()) {
                    dep_it->second.dependents.push_back(fid);
                }
            }
        }

        // Allocate FlowRequest objects for each flow (heap-allocated, lives for full sim)
        _requests.resize(_total_flows);
        for (size_t i = 0; i < flows.size(); i++) {
            _requests[i] = new FlowRequest();
            const auto& rec = flows[i];
            _requests[i]->flowTag.current_flow_id = rec.flow_id;
            _requests[i]->flowTag.nvls_on =
                (rec.conn_type == "NVLS" || rec.conn_type == "NVLS_TREE");
            _requests[i]->srcRank = rec.src;
            _requests[i]->dstRank = rec.dst;
            _requests[i]->reqCount = rec.flow_size;
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
            if (st.pending_deps > 0) continue;

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
    // DoSendFlow: internal callback from Simulator::Schedule
    // ------------------------------------------------------------------
    void DoSendFlow(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) return;

        FlowState& st = it->second;
        // Find the corresponding FlowRequest
        // (requests are stored in the same order as flows were loaded)
        FlowRequest* req = nullptr;
        for (size_t i = 0; i < _requests.size(); i++) {
            if (_requests[i]->flowTag.current_flow_id == flow_id) {
                req = _requests[i];
                break;
            }
        }

        if (!req) {
            std::cerr << "[DepScheduler] ERROR: no request found for flow "
                      << flow_id << std::endl;
            return;
        }

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
    // VerifyDAG: run cycle detection on the dependency graph
    // ------------------------------------------------------------------
    bool VerifyDAG() {
        // DFS-based cycle detection (3-color: 0=white, 1=gray, 2=black)
        std::unordered_map<uint32_t, int> color; // 0=unvisited, 1=in_progress, 2=done
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
    std::vector<FlowRequest*> _requests;
};

// ============================================================================
// Global callback pointer definition
// ============================================================================

void (*g_on_flow_completed)(uint32_t flow_id) = nullptr;

#endif // __DECOUPLED_DEP_SCHEDULER_H__
