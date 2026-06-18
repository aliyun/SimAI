/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Complete flow file parser.
 * Parses ALL 21 fields (including relative_delay_ns).
 * Based on: loadFlowsFromFile() in MockNcclGroup.cc:2245-2265
 * (NOT ImportFlows() -- that skips parent_flow_id[], child_flow_id[],
 *  group_type, op, loopstate).
 *
 * Flow file format (21 fields per line, header line = total flow count):
 *   flow_id src dest flow_size channel_id chunk_id chunk_count conn_type
 *   start_time pg maxPacketCount port dport
 *   np prev[0..np-1]
 *   npar parent_flow_id[0..npar-1] nchi child_flow_id[0..nchi-1]
 *   layer_num group_type op loopstate relative_delay_ns
 */

#ifndef __DECOUPLED_FLOW_READER_H__
#define __DECOUPLED_FLOW_READER_H__

#include "common_types.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>

// ============================================================================
// FlowFileRecord: complete parsed record with ALL 21 fields
// ============================================================================

struct FlowFileRecord {
    uint32_t flow_id = 0;
    uint32_t src = 0;
    uint32_t dst = 0;
    uint64_t flow_size = 0;
    int channel_id = 0;
    int chunk_id = 0;
    int chunk_count = 0;
    std::string conn_type;
    double start_time = 0.0;          // always 0.0 in decoupled mode
    uint32_t pg = 0;
    uint32_t maxPacketCount = 0;
    uint32_t port = 0;
    uint32_t dport = 0;
    std::vector<uint32_t> prev;
    std::vector<int> parent_flow_id;
    std::vector<int> child_flow_id;
    uint32_t layer_num = 0;
    uint32_t group_type = 0;
    uint32_t op = 0;
    uint32_t loopstate = 0;
    uint64_t relative_delay_ns = 0;   // completion-based: send_time - max(prev completion times)

    // Basic validation
    bool valid() const {
        return flow_size > 0 || (src != dst);
    }
};

// ============================================================================
// LoadFlows: Parse complete flow file
// Extracted from: MockNcclGroup.cc:2245-2265 (loadFlowsFromFile)
// ============================================================================

// Extracted from: MockNcclGroup.cc:2245-2265 (parse loop pattern)
inline std::vector<FlowFileRecord> LoadFlows(const std::string& flow_file_path) {
    std::vector<FlowFileRecord> flows;

    std::ifstream ff(flow_file_path);
    if (!ff.is_open()) {
        std::cerr << "[LoadFlows] ERROR: Cannot open flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    // Read header line (total flow count)
    std::string header_line;
    if (!std::getline(ff, header_line) || header_line.empty()) {
        std::cerr << "[LoadFlows] ERROR: Empty or invalid flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    uint32_t total = 0;
    try {
        total = std::stoul(header_line);
    } catch (...) {
        std::cerr << "[LoadFlows] ERROR: Bad header in flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    if (total == 0) {
        std::cerr << "[LoadFlows] WARNING: Flow file header says 0 flows." << std::endl;
        return flows;
    }

    flows.reserve(total);

    std::string line;
    uint32_t line_num = 1; // line 1 was the header
    while (std::getline(ff, line)) {
        line_num++;
        if (line.empty()) continue;

        std::istringstream is(line);
        FlowFileRecord r;

        // Fields 1-8: fixed scalars
        if (!(is >> r.flow_id >> r.src >> r.dst >> r.flow_size
                  >> r.channel_id >> r.chunk_id >> r.chunk_count
                  >> r.conn_type)) {
            std::cerr << "[LoadFlows] ERROR: Line " << line_num
                      << " truncated (cannot read fields 1-8)" << std::endl;
            continue;
        }

        // Fields 9-13: start_time, pg, maxPacketCount, port, dport, np
        double st; uint32_t pg, mpc, port, dport; uint32_t np;
        if (!(is >> st >> pg >> mpc >> port >> dport >> np)) {
            std::cerr << "[LoadFlows] ERROR: Line " << line_num
                      << " truncated (cannot read fields 9-14)" << std::endl;
            continue;
        }
        r.start_time = st;
        r.pg = pg;
        r.maxPacketCount = mpc;
        r.port = port;
        r.dport = dport;

        // Field 14: prev[] (variable length)
        r.prev.reserve(np);
        for (uint32_t j = 0; j < np; j++) {
            uint32_t pid;
            if (!(is >> pid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << line_num
                          << " truncated (prev[" << j << "/" << np << "])" << std::endl;
                break;
            }
            r.prev.push_back(pid);
        }

        // Fields 15-16: parent_flow_id[], child_flow_id[] (variable length)
        uint32_t npar, nchi;
        if (!(is >> npar)) {
            std::cerr << "[LoadFlows] ERROR: Line " << line_num
                      << " truncated (cannot read npar)" << std::endl;
            continue;
        }
        r.parent_flow_id.reserve(npar);
        for (uint32_t j = 0; j < npar; j++) {
            int pid;
            if (!(is >> pid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << line_num
                          << " truncated (parent_flow_id[" << j << "/" << npar << "])"
                          << std::endl;
                break;
            }
            r.parent_flow_id.push_back(pid);
        }

        if (!(is >> nchi)) {
            std::cerr << "[LoadFlows] ERROR: Line " << line_num
                      << " truncated (cannot read nchi)" << std::endl;
            continue;
        }
        r.child_flow_id.reserve(nchi);
        for (uint32_t j = 0; j < nchi; j++) {
            int cid;
            if (!(is >> cid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << line_num
                          << " truncated (child_flow_id[" << j << "/" << nchi << "])"
                          << std::endl;
                break;
            }
            r.child_flow_id.push_back(cid);
        }

        // Fields 17-20: layer_num, group_type, op, loopstate
        if (!(is >> r.layer_num >> r.group_type >> r.op >> r.loopstate)) {
            std::cerr << "[LoadFlows] ERROR: Line " << line_num
                      << " truncated (cannot read layer_num/group_type/op/loopstate)"
                      << std::endl;
            continue;
        }

        // Field 21 (NEW): relative_delay_ns with legacy fallback
        // Detection: try to read; if stream is exhausted, default to 0
        if (!(is >> r.relative_delay_ns)) {
            // Legacy format -- no relative_delay_ns field
            r.relative_delay_ns = 0;
            // Clear failbit so we can continue to next line
            is.clear();
        }

        flows.push_back(r);
    }

    ff.close();

    std::cout << "[LoadFlows] Parsed " << flows.size() << " flows from "
              << flow_file_path << " (header said " << total << ")" << std::endl;

    return flows;
}

#endif // __DECOUPLED_FLOW_READER_H__
