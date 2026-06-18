/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---
 * DECOUPLED REPLAY: Common type definitions extracted from SimAI common.h/entry.h.
 * All SimAI (AstraSim::) types replaced with local equivalents.
 * No dependency on astra-sim libraries.
 *
 * NOTE: Global variables are defined directly (no extern) matching the original
 * common.h pattern. This works because main.cc is the ONLY translation unit.
 */

#ifndef __DECOUPLED_COMMON_TYPES_H__
#define __DECOUPLED_COMMON_TYPES_H__

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <atomic>

// NS3 headers
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"

#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <ns3/nvswitch-node.h>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("DECOUPLED_REPLAY");

// ============================================================================
// Local type replacements for SimAI types
// ============================================================================

// Replaces AstraSim::ncclFlowTag
// Extracted from: astra-sim-alibabacloud/astra-sim/system/... (ncclFlowTag fields)
struct FlowTag {
    uint32_t current_flow_id = 0;
    bool nvls_on = false;
};

// Replaces AstraSim::sim_request
struct FlowRequest {
    FlowTag flowTag;
    int srcRank = 0;
    int dstRank = 0;
    uint64_t reqCount = 0;
};

// ============================================================================
// Global configuration variables
// Extracted from: astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h:51-98
// ============================================================================

uint32_t cc_mode = 1;
bool enable_qcn = true, use_dynamic_pfc_threshold = true;
uint32_t packet_payload_size = 1000, l2_chunk_size = 0, l2_ack_interval = 0;
double pause_time = 5, simulator_stop_time = 3.01;
std::string data_rate, link_delay, topology_file, flow_file, trace_file,
    trace_output_file;
std::string fct_output_file = "fct.txt";
std::string pfc_output_file = "pfc.txt";
std::string send_output_file = "send.txt";

double alpha_resume_interval = 55, rp_timer, ewma_gain = 1 / 16;
double rate_decrease_interval = 4;
uint32_t fast_recovery_times = 5;
std::string rate_ai, rate_hai, min_rate = "100Mb/s";
std::string dctcp_rate_ai = "1000Mb/s";

bool clamp_target_rate = false, l2_back_to_zero = false;
double error_rate_per_link = 0.0;
uint32_t has_win = 1;
uint32_t global_t = 1;
uint32_t mi_thresh = 5;
bool var_win = false, fast_react = true;
bool multi_rate = true;
bool sample_feedback = false;
double pint_log_base = 1.05;
double pint_prob = 1.0;
double u_target = 0.95;
uint32_t int_multi = 1;
bool rate_bound = true;
int nic_total_pause_time = 0;

uint32_t ack_high_prio = 0;
uint64_t link_down_time = 0;
uint32_t link_down_A = 0, link_down_B = 0;

uint32_t enable_trace = 1;
uint32_t buffer_size = 16;

uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
GPUType gpu_type;
std::vector<int> NVswitchs;

uint32_t qp_mon_interval = 100;
uint32_t bw_mon_interval = 10000;
uint32_t qlen_mon_interval = 10000;
uint64_t mon_start = 0, mon_end = 2100000000;

std::string qlen_mon_file;
std::string bw_mon_file;
std::string rate_mon_file;
std::string cnp_mon_file;
std::string total_flow_file = "/tmp/decoupled_replay/monitor_output/";
FILE* total_flow_output = nullptr;

std::unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
std::unordered_map<uint64_t, double> rate2pmax;

std::ifstream topof, flowf, tracef;

// ============================================================================
// Topology / network state
// Extracted from: astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h:112-136
// ============================================================================

NodeContainer n;
uint64_t nic_rate;
uint64_t maxRtt, maxBdp;
std::vector<Ipv4Address> serverAddress;
std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint16_t>> portNumber;

// Extracted from: common.h:122-129
struct Interface {
    uint32_t idx;
    bool up;
    uint64_t delay;
    uint64_t bw;

    Interface() : idx(0), up(false) {}
};

std::map<Ptr<Node>, std::map<Ptr<Node>, Interface>> nbr2if;
std::map<Ptr<Node>, std::map<Ptr<Node>, std::vector<Ptr<Node>>>> nextHop;
std::map<Ptr<Node>, std::map<Ptr<Node>, uint64_t>> pairDelay;
std::map<Ptr<Node>, std::map<Ptr<Node>, uint64_t>> pairTxDelay;
std::map<uint32_t, std::map<uint32_t, uint64_t>> pairBw;
std::map<Ptr<Node>, std::map<Ptr<Node>, uint64_t>> pairBdp;
std::map<uint32_t, std::map<uint32_t, uint64_t>> pairRtt;

// ============================================================================
// Flow-related structs
// ============================================================================

// Extracted from: common.h:138-142
struct FlowInput {
    uint32_t src, dst, pg, maxPacketCount, port, dport;
    double start_time;
    uint32_t idx;
};

FlowInput flow_input = {0};
uint32_t flow_num;

// Extracted from: common.h:145-164
// Decoupled mode: NS3 replays flows from file using this record.
struct FlowRecord {
    uint32_t flow_id;
    uint32_t src, dst;
    uint64_t flow_size;
    int channel_id;
    int chunk_id;
    int chunk_count;
    std::string conn_type;
    double start_time;              // ns, relative to sim start
    std::vector<uint32_t> prev;     // Dependency graph (flow_ids that must complete first)
    uint32_t pg;                     // priority group
    uint32_t maxPacketCount;
    uint32_t port, dport;
    uint32_t layer_num;             // workload layer index
    uint32_t group_type;            // GroupType enum: TP=0, DP=1, EP=2, DP_EP=3
    uint32_t op;                    // ComType enum
    uint32_t loopstate;             // State enum: FWD=0, WG=1, IG=2
    uint64_t relative_delay_ns;     // Decoupled mode: delay after all prev[] complete, before sending (ns)
};

// ============================================================================
// QlenDistribution struct
// Extracted from: common.h:181-191
// ============================================================================

struct QlenDistribution {
    std::vector<uint32_t> cnt;

    void add(uint32_t qlen) {
        uint32_t kb = qlen / 1000;
        if (cnt.size() < kb + 1)
            cnt.resize(kb + 1);
        cnt[kb]++;
    }
};

// ============================================================================
// Task struct for receive/send tracking (replaces SimAI-level task1)
// Extracted from: entry.h:60-68
// ============================================================================

struct RecvTask {
    int src;
    int dest;
    int type;
    uint64_t count;
    void *fun_arg;
    void (*msg_handler)(void *fun_arg);
    double schTime;
};

// ============================================================================
// Helper functions
// Extracted from: common.h:168-173
// ============================================================================

// Extracted from: common.h:168-171
inline Ipv4Address node_id_to_ip(uint32_t id) {
    return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) +
                       ((id % 256) * 0x00000100));
}

// Extracted from: common.h:173
inline uint32_t ip_to_node_id(Ipv4Address ip) {
    return (ip.Get() >> 8) & 0xffff;
}

#endif // __DECOUPLED_COMMON_TYPES_H__
