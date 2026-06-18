/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: RDMA flow injection and receive/send tracking.
 * Extracted from: astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h
 *
 * Functions extracted:
 *   is_sending_finished()              - entry.h:77-89
 *   is_receive_finished()              - entry.h:91-107
 *   SendFlow()                         - entry.h:109-166
 *   notify_receiver_receive_data()     - entry.h:277-348
 *   notify_sender_sending_finished()   - entry.h:350-387
 *
 * Key changes from SimAI version:
 *   - AstraSim::ncclFlowTag → local FlowTag
 *   - AstraSim::sim_request* → local FlowRequest*
 *   - MockNcclLog → removed (was debug-only)
 *   - Sys::boostedTick() → Simulator::Now().GetNanoSeconds()
 *   - #ifdef NS3_MTP / MtpInterface blocks removed
 *   - Single QP per flow (no multi-QP loop)
 */

#ifndef __DECOUPLED_FLOW_SENDER_H__
#define __DECOUPLED_FLOW_SENDER_H__

#include "common_types.h"

#include <cstdlib>
#include <algorithm>

// ============================================================================
// Tracking maps
// Extracted from: entry.h:55-76
// ============================================================================

// Extracted from: entry.h:55
std::map<std::pair<std::pair<int, int>, int>, FlowTag> receiver_pending_queue;

// Extracted from: entry.h:56
uint64_t last_flow_finish_ns = 0;

// Extracted from: entry.h:59
std::map<std::pair<int, std::pair<int, int>>, FlowTag> sender_src_port_map;

// Extracted from: entry.h:69-76
std::map<std::pair<int, std::pair<int, int>>, RecvTask> expeRecvHash;
std::map<std::pair<int, std::pair<int, int>>, uint64_t> recvHash;
std::map<std::pair<int, std::pair<int, int>>, RecvTask> sentHash;
std::map<std::pair<int, int>, int64_t> nodeHash;
std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_sent_callback;
std::map<std::pair<int, std::pair<int, int>>, int> waiting_to_notify_receiver;
std::map<std::pair<int, std::pair<int, int>>, uint64_t> received_chunksize;
std::map<std::pair<int, std::pair<int, int>>, uint64_t> sent_chunksize;

// ============================================================================
// is_sending_finished
// Extracted from: entry.h:77-89
// ============================================================================

// Extracted from: entry.h:77-89
inline bool is_sending_finished(int src, int dst, FlowTag flowTag) {
    int tag_id = flowTag.current_flow_id;
    if (waiting_to_sent_callback.count(
            std::make_pair(tag_id, std::make_pair(src, dst)))) {
        if (--waiting_to_sent_callback[std::make_pair(
                tag_id, std::make_pair(src, dst))] == 0) {
            waiting_to_sent_callback.erase(
                std::make_pair(tag_id, std::make_pair(src, dst)));
            return true;
        }
    }
    return false;
}

// ============================================================================
// is_receive_finished
// Extracted from: entry.h:91-107
// ============================================================================

// Extracted from: entry.h:91-107
inline bool is_receive_finished(int src, int dst, FlowTag flowTag) {
    int tag_id = flowTag.current_flow_id;
    if (waiting_to_notify_receiver.count(
            std::make_pair(tag_id, std::make_pair(src, dst)))) {
        NS_LOG_DEBUG(" is_receive_finished waiting_to_notify_receiver  tag_id  "
                     << tag_id << " src " << src << " dst " << dst << " count "
                     << waiting_to_notify_receiver[std::make_pair(
                            tag_id, std::make_pair(src, dst))]);
        if (--waiting_to_notify_receiver[std::make_pair(
                tag_id, std::make_pair(src, dst))] == 0) {
            waiting_to_notify_receiver.erase(
                std::make_pair(tag_id, std::make_pair(src, dst)));
            return true;
        }
    }
    return false;
}

// ============================================================================
// SendFlow
// Extracted from: entry.h:109-166
// Modified: SimAI types replaced, NS3_MTP removed, MockNcclLog removed
// ============================================================================

// Extracted from: entry.h:109-166
// Simplified: single QP per flow (no multi-QP loop)
inline void SendFlow(int src, int dst, uint64_t maxPacketCount,
                     void (*msg_handler)(void *fun_arg), void *fun_arg,
                     int tag, FlowRequest *request) {
    uint64_t real_PacketCount = maxPacketCount;
    uint32_t port = portNumber[src][dst]++;

    sender_src_port_map[std::make_pair(port, std::make_pair(src, dst))] = request->flowTag;

    int flow_id = request->flowTag.current_flow_id;
    bool nvls_on = request->flowTag.nvls_on;
    int pg = 3, dport = 100;
    int send_lat = 6;  // microseconds (multiplied by 1000 to ns below)
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
        try {
            send_lat = std::stoi(send_lat_env);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[SendFlow] ERROR: AS_SEND_LAT invalid value" << std::endl;
            exit(-1);
        }
    }
    send_lat *= 1000;
    flow_input.idx++;
    if (real_PacketCount == 0) real_PacketCount = 1;

    NS_LOG_DEBUG(" [Packet sending event] " << src << " SendFlow to " << dst
                 << " flow_id " << flow_id
                 << " srcip " << serverAddress[src]
                 << " dstip " << serverAddress[dst]
                 << " size: " << maxPacketCount
                 << " at the tick: " << Simulator::Now().GetNanoSeconds());

    RdmaClientHelper clientHelper(
        pg, serverAddress[src], serverAddress[dst], port, dport, real_PacketCount,
        has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
        global_t == 1 ? maxRtt : pairRtt[src][dst], msg_handler, fun_arg, tag,
        src, dst);
    if (nvls_on) clientHelper.SetAttribute("NVLS_enable", UintegerValue(1));

    ApplicationContainer appCon = clientHelper.Install(n.Get(src));
    appCon.Start(Time(send_lat));
    waiting_to_sent_callback[std::make_pair(request->flowTag.current_flow_id,
                                            std::make_pair(src, dst))]++;
    waiting_to_notify_receiver[std::make_pair(request->flowTag.current_flow_id,
                                              std::make_pair(src, dst))]++;

    NS_LOG_DEBUG("waiting_to_notify_receiver  current_flow_id "
                 << request->flowTag.current_flow_id
                 << " src " << src << " dst " << dst << " count "
                 << waiting_to_notify_receiver[std::make_pair(
                        tag, std::make_pair(src, dst))]);
}

// ============================================================================
// notify_receiver_receive_data
// Extracted from: entry.h:277-348
// Modified: SimAI types replaced, NS3_MTP removed, MockNcclLog removed
// ============================================================================

// Extracted from: entry.h:277-348
inline void notify_receiver_receive_data(int sender_node, int receiver_node,
                                         uint64_t message_size, FlowTag flowTag) {
    NS_LOG_DEBUG(" " << sender_node << " notify receiver: " << receiver_node
                 << " message size: " << message_size);
    int tag = flowTag.current_flow_id;
    if (expeRecvHash.find(std::make_pair(
            tag, std::make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
        RecvTask t2 =
            expeRecvHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))];
        NS_LOG_DEBUG(" " << sender_node << " notify receiver: " << receiver_node
                     << " message size: " << message_size
                     << " t2.count: " << t2.count
                     << " channel id: " << tag);

        if (message_size == t2.count) {
            NS_LOG_DEBUG(" message_size = t2.count expeRecvHash.erase "
                         << sender_node << " notify receiver: " << receiver_node
                         << " message size: " << message_size << " channel_id " << tag);
            expeRecvHash.erase(std::make_pair(tag, std::make_pair(sender_node, receiver_node)));
            t2.msg_handler(t2.fun_arg);
            goto receiver_end;
        } else if (message_size > t2.count) {
            recvHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))] =
                message_size - t2.count;
            NS_LOG_DEBUG("message_size > t2.count expeRecvHash.erase "
                         << sender_node << " notify receiver: " << receiver_node
                         << " message size: " << message_size << " channel_id " << tag);
            expeRecvHash.erase(std::make_pair(tag, std::make_pair(sender_node, receiver_node)));
            t2.msg_handler(t2.fun_arg);
            goto receiver_end;
        } else {
            t2.count -= message_size;
            expeRecvHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))] = t2;
        }
    } else {
        receiver_pending_queue[std::make_pair(
            std::make_pair(receiver_node, sender_node), tag)] = flowTag;
        if (recvHash.find(std::make_pair(tag, std::make_pair(sender_node, receiver_node))) ==
            recvHash.end()) {
            recvHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))] =
                message_size;
        } else {
            recvHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))] +=
                message_size;
        }
    }

receiver_end:
    if (nodeHash.find(std::make_pair(receiver_node, 1)) == nodeHash.end()) {
        nodeHash[std::make_pair(receiver_node, 1)] = message_size;
    } else {
        nodeHash[std::make_pair(receiver_node, 1)] += message_size;
    }
}

// ============================================================================
// notify_sender_sending_finished
// Extracted from: entry.h:350-387
// Modified: SimAI types replaced, NS3_MTP removed, MockNcclLog removed
// ============================================================================

// Extracted from: entry.h:350-387
inline void notify_sender_sending_finished(int sender_node, int receiver_node,
                                           uint64_t message_size, FlowTag flowTag) {
    int tag = flowTag.current_flow_id;
    if (sentHash.find(std::make_pair(tag, std::make_pair(sender_node, receiver_node))) !=
        sentHash.end()) {
        RecvTask t2 = sentHash[std::make_pair(tag, std::make_pair(sender_node, receiver_node))];
        if (t2.count == message_size) {
            sentHash.erase(std::make_pair(tag, std::make_pair(sender_node, receiver_node)));
            if (nodeHash.find(std::make_pair(sender_node, 0)) == nodeHash.end()) {
                nodeHash[std::make_pair(sender_node, 0)] = message_size;
            } else {
                nodeHash[std::make_pair(sender_node, 0)] += message_size;
            }
            t2.msg_handler(t2.fun_arg);
            goto sender_end;
        } else {
            NS_LOG_ERROR("sentHash msg size != sender_node " << sender_node
                         << " receiver_node " << receiver_node
                         << " message_size " << message_size);
        }
    } else {
        NS_LOG_ERROR("sentHash cannot find sender_node " << sender_node
                     << " receiver_node " << receiver_node
                     << " message_size " << message_size);
    }
sender_end:
    return;
}

#endif // __DECOUPLED_FLOW_SENDER_H__
