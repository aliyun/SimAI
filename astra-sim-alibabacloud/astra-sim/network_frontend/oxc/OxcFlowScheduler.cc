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
 */

#include "OxcFlowScheduler.h"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace OXC {

OxcFlowScheduler::OxcFlowScheduler() {
}

OxcFlowScheduler::~OxcFlowScheduler() {
}

void OxcFlowScheduler::buildScheduledFlows(const std::vector<OutputFlow>& flows) {
    scheduled_flows_.clear();
    flow_id_to_index_.clear();
    indegree_mapping_.clear();
    pending_flows_.clear();
    active_flows_.clear();
    completed_flows_.clear();

    // 转换 OutputFlow 到 ScheduledFlow
    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& f = flows[i];
        ScheduledFlow sf;
        sf.flow_id = f.flow_id;
        sf.operation_id = f.operation_id;
        sf.layer_name = f.layer_name;
        sf.phase = f.phase;
        sf.comm_type = f.comm_type;
        sf.group_type = f.group_type;
        sf.src = f.src;
        sf.dst = f.dst;
        sf.flow_size = f.flow_size;
        sf.step = f.step;
        sf.parent_flow_ids = f.depends_on;  // 复制依赖关系
        sf.indegree = 0;
        sf.schedule_tick = -1;
        sf.complete_tick = -1;
        sf.is_active = false;
        sf.is_completed = false;

        flow_id_to_index_[sf.flow_id] = static_cast<int>(scheduled_flows_.size());
        scheduled_flows_.push_back(sf);
        pending_flows_.insert(sf.flow_id);
    }

    // 构建双向链接
    buildBidirectionalLinks();

    // 初始化入度
    initIndegreeMapping();

    std::cout << "[Scheduler] Built " << scheduled_flows_.size() << " scheduled flows" << std::endl;
}

void OxcFlowScheduler::buildBidirectionalLinks() {
    // 根据 parent_flow_ids 构建 child_flow_ids
    for (auto& sf : scheduled_flows_) {
        for (int parent_id : sf.parent_flow_ids) {
            auto it = flow_id_to_index_.find(parent_id);
            if (it != flow_id_to_index_.end()) {
                scheduled_flows_[it->second].child_flow_ids.push_back(sf.flow_id);
            }
        }
    }

    // 统计
    int total_parent_links = 0;
    int total_child_links = 0;
    for (const auto& sf : scheduled_flows_) {
        total_parent_links += static_cast<int>(sf.parent_flow_ids.size());
        total_child_links += static_cast<int>(sf.child_flow_ids.size());
    }
    std::cout << "[Scheduler] Bidirectional links: " << total_parent_links
              << " parent links, " << total_child_links << " child links" << std::endl;
}

void OxcFlowScheduler::initIndegreeMapping() {
    for (auto& sf : scheduled_flows_) {
        sf.indegree = static_cast<int>(sf.parent_flow_ids.size());
        indegree_mapping_[sf.flow_id] = sf.indegree;
    }

    // 统计入度为0的流数量
    int zero_indegree = 0;
    for (const auto& sf : scheduled_flows_) {
        if (sf.indegree == 0) {
            zero_indegree++;
        }
    }
    std::cout << "[Scheduler] Indegree initialized: " << zero_indegree
              << " flows with indegree=0 (ready to start)" << std::endl;
}

std::vector<int> OxcFlowScheduler::activateReadyFlows(int current_tick) {
    std::vector<int> activated;

    // 找出所有入度为0且未激活的流
    std::vector<int> to_activate;
    for (int flow_id : pending_flows_) {
        auto it = flow_id_to_index_.find(flow_id);
        if (it != flow_id_to_index_.end()) {
            ScheduledFlow& sf = scheduled_flows_[it->second];
            if (sf.indegree == 0 && !sf.is_active && !sf.is_completed) {
                to_activate.push_back(flow_id);
            }
        }
    }

    // 激活这些流
    for (int flow_id : to_activate) {
        auto it = flow_id_to_index_.find(flow_id);
        if (it != flow_id_to_index_.end()) {
            ScheduledFlow& sf = scheduled_flows_[it->second];
            sf.is_active = true;
            sf.schedule_tick = current_tick;
            pending_flows_.erase(flow_id);
            active_flows_.insert(flow_id);
            activated.push_back(flow_id);
        }
    }

    return activated;
}

void OxcFlowScheduler::completeFlow(int flow_id, int current_tick) {
    auto it = flow_id_to_index_.find(flow_id);
    if (it == flow_id_to_index_.end()) {
        return;
    }

    ScheduledFlow& sf = scheduled_flows_[it->second];
    sf.is_completed = true;
    sf.is_active = false;
    sf.complete_tick = current_tick;
    active_flows_.erase(flow_id);
    completed_flows_.insert(flow_id);

    // 递减所有子流的入度
    for (int child_id : sf.child_flow_ids) {
        auto child_it = flow_id_to_index_.find(child_id);
        if (child_it != flow_id_to_index_.end()) {
            ScheduledFlow& child_sf = scheduled_flows_[child_it->second];
            if (child_sf.indegree > 0) {
                child_sf.indegree--;
                indegree_mapping_[child_id] = child_sf.indegree;
            }
        }
    }
}

ScheduleStats OxcFlowScheduler::runSimulation() {
    stats_ = ScheduleStats();
    stats_.total_flows = static_cast<int>(scheduled_flows_.size());
    stats_.max_parallel_flows = 0;

    int current_tick = 0;
    const int MAX_TICKS = 100000;  // 防止无限循环

    std::cout << "[Scheduler] Starting simulation with " << stats_.total_flows << " flows" << std::endl;

    while (!pending_flows_.empty() || !active_flows_.empty()) {
        if (current_tick >= MAX_TICKS) {
            std::cerr << "[Scheduler] Warning: Max ticks reached, stopping simulation" << std::endl;
            break;
        }

        // 1. 激活入度为0的流
        std::vector<int> activated = activateReadyFlows(current_tick);

        // 记录本tick激活的流
        if (!activated.empty()) {
            stats_.tick_to_flows[current_tick] = activated;
        }

        // 2. 记录当前并行流数
        int parallel_count = static_cast<int>(active_flows_.size());
        stats_.flows_per_tick[current_tick] = parallel_count;
        if (parallel_count > stats_.max_parallel_flows) {
            stats_.max_parallel_flows = parallel_count;
        }

        // 3. 完成当前所有活跃流（简化模型：每个流执行1个tick）
        std::vector<int> to_complete(active_flows_.begin(), active_flows_.end());
        for (int flow_id : to_complete) {
            completeFlow(flow_id, current_tick);
        }

        current_tick++;
    }

    stats_.total_ticks = current_tick;

    std::cout << "[Scheduler] Simulation completed:" << std::endl;
    std::cout << "  Total ticks: " << stats_.total_ticks << std::endl;
    std::cout << "  Max parallel flows: " << stats_.max_parallel_flows << std::endl;
    std::cout << "  Completed flows: " << completed_flows_.size() << std::endl;

    return stats_;
}

bool OxcFlowScheduler::writeScheduleResult(const std::string& output_prefix) {
    std::string filename = output_prefix + "_schedule.csv";
    std::ofstream ofs(filename);

    if (!ofs.is_open()) {
        std::cerr << "[Scheduler] Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    // 写入CSV头
    ofs << "flow_id,op_id,layer,phase,comm_type,group,src,dst,size,step,"
        << "parent_count,child_count,schedule_tick,complete_tick" << std::endl;

    // 写入每个流的调度信息
    for (const auto& sf : scheduled_flows_) {
        ofs << sf.flow_id << ","
            << sf.operation_id << ","
            << sf.layer_name << ","
            << phaseToString(sf.phase) << ","
            << commTypeToString(sf.comm_type) << ","
            << groupTypeToString(sf.group_type) << ","
            << sf.src << ","
            << sf.dst << ","
            << sf.flow_size << ","
            << sf.step << ","
            << sf.parent_flow_ids.size() << ","
            << sf.child_flow_ids.size() << ","
            << sf.schedule_tick << ","
            << sf.complete_tick << std::endl;
    }

    ofs.close();
    std::cout << "[Scheduler] Schedule result written to " << filename << std::endl;

    // 写入统计摘要
    std::string stats_filename = output_prefix + "_schedule_stats.json";
    std::ofstream stats_ofs(stats_filename);

    if (stats_ofs.is_open()) {
        stats_ofs << "{" << std::endl;
        stats_ofs << "  \"total_flows\": " << stats_.total_flows << "," << std::endl;
        stats_ofs << "  \"total_ticks\": " << stats_.total_ticks << "," << std::endl;
        stats_ofs << "  \"max_parallel_flows\": " << stats_.max_parallel_flows << "," << std::endl;

        // 写入每个tick的并行流数
        stats_ofs << "  \"flows_per_tick\": {" << std::endl;
        bool first = true;
        for (const auto& pair : stats_.flows_per_tick) {
            if (!first) stats_ofs << "," << std::endl;
            first = false;
            stats_ofs << "    \"" << pair.first << "\": " << pair.second;
        }
        stats_ofs << std::endl << "  }," << std::endl;

        // 写入每个tick激活的流ID
        stats_ofs << "  \"tick_to_flows\": {" << std::endl;
        first = true;
        for (const auto& pair : stats_.tick_to_flows) {
            if (!first) stats_ofs << "," << std::endl;
            first = false;
            stats_ofs << "    \"" << pair.first << "\": [";
            for (size_t i = 0; i < pair.second.size(); ++i) {
                if (i > 0) stats_ofs << ", ";
                stats_ofs << pair.second[i];
            }
            stats_ofs << "]";
        }
        stats_ofs << std::endl << "  }" << std::endl;

        stats_ofs << "}" << std::endl;
        stats_ofs.close();
        std::cout << "[Scheduler] Schedule stats written to " << stats_filename << std::endl;
    }

    return true;
}

}  // namespace OXC
