/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: FCT output and QP completion callbacks.
 * Extracted from: astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h
 *
 * Functions extracted:
 *   qp_finish()   - entry.h:410-457
 *   send_finish() - entry.h:459-484
 *
 * Key changes from SimAI version:
 *   - AstraSim::ncclFlowTag → local FlowTag
 *   - MockNcclLog → NS_LOG
 *   - Sys::boostedTick() → Simulator::Now().GetNanoSeconds()
 *   - #ifdef NS3_MTP blocks removed
 *
 * Dependencies: flow_sender.h (for tracking maps)
 */

#ifndef __DECOUPLED_FCT_WRITER_H__
#define __DECOUPLED_FCT_WRITER_H__

#include "common_types.h"
#include "flow_sender.h"

// Forward declaration: global flow-completion callback (defined in dep_scheduler.h)
extern void (*g_on_flow_completed)(uint32_t flow_id);

// ============================================================================
// qp_finish
// Extracted from: entry.h:410-457
// Modified: SimAI types replaced, NS3_MTP removed, MockNcclLog → NS_LOG
// ============================================================================

// Extracted from: entry.h:410-457
inline void qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
    uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
    uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
    uint32_t total_bytes =
        q->m_size +
        ((q->m_size - 1) / packet_payload_size + 1) *
            (CustomHeader::GetStaticWholeHeaderSize() -
             IntHeader::GetStaticSize());
    uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
    fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
            q->sport, q->dport, q->m_size, q->startTime.GetTimeStep(),
            (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
    fflush(fout);

    FlowTag flowTag;
    uint64_t notify_size;

    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);

    NS_LOG_DEBUG("qp finish, src: " << sid << " did: " << did
                 << " port: " << q->sport << " total bytes: " << q->m_size
                 << " at the tick: " << Simulator::Now().GetNanoSeconds());

    if (sender_src_port_map.find(std::make_pair(q->sport, std::make_pair(sid, did))) ==
        sender_src_port_map.end()) {
        std::cerr << "[qp_finish] ERROR: could not find the tag for sport="
                  << q->sport << " src=" << sid << " dst=" << did << std::endl;
        exit(-1);
    }
    flowTag = sender_src_port_map[std::make_pair(q->sport, std::make_pair(sid, did))];
    sender_src_port_map.erase(std::make_pair(q->sport, std::make_pair(sid, did)));
    received_chunksize[std::make_pair(flowTag.current_flow_id,
                                       std::make_pair(sid, did))] += q->m_size;
    if (!is_receive_finished(sid, did, flowTag)) {
        return;
    }
    notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id,
                                                     std::make_pair(sid, did))];
    received_chunksize.erase(std::make_pair(flowTag.current_flow_id,
                                             std::make_pair(sid, did)));

    notify_receiver_receive_data(sid, did, notify_size, flowTag);
    last_flow_finish_ns = Simulator::Now().GetNanoSeconds();

    // Notify the dependency scheduler that this flow is complete
    if (g_on_flow_completed) {
        g_on_flow_completed(flowTag.current_flow_id);
    }
}

// ============================================================================
// send_finish
// Extracted from: entry.h:459-484
// Modified: SimAI types replaced, NS3_MTP removed, MockNcclLog → NS_LOG
// ============================================================================

// Extracted from: entry.h:459-484
inline void send_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
    uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
    FlowTag flowTag;

    NS_LOG_DEBUG("[Packet sent from NIC] send finish, src: " << sid
                 << " did: " << did << " port: " << q->sport
                 << " srcip " << q->sip << " dstip " << q->dip
                 << " total bytes: " << q->m_size
                 << " at the tick: " << Simulator::Now().GetNanoSeconds());

    uint64_t all_sent_chunksize;

    flowTag = sender_src_port_map[std::make_pair(q->sport, std::make_pair(sid, did))];
    sent_chunksize[std::make_pair(flowTag.current_flow_id,
                                   std::make_pair(sid, did))] += q->m_size;
    if (!is_sending_finished(sid, did, flowTag)) {
        return;
    }
    all_sent_chunksize = sent_chunksize[std::make_pair(flowTag.current_flow_id,
                                                        std::make_pair(sid, did))];
    sent_chunksize.erase(std::make_pair(flowTag.current_flow_id,
                                         std::make_pair(sid, did)));

    notify_sender_sending_finished(sid, did, all_sent_chunksize, flowTag);
}

#endif // __DECOUPLED_FCT_WRITER_H__
