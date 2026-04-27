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

#include "OxcFlowGenerator.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <set>

namespace OXC {

OxcFlowGenerator::OxcFlowGenerator(
    const std::string& oxc_server_url,
    int num_gpus,
    int gpus_per_server,
    int tp_size,
    int dp_size,
    int ep_size,
    int pp_size)
    : oxc_server_url_(oxc_server_url),
      alg_name_("ALGO_OXC_RING"),
      num_gpus_(num_gpus),
      gpus_per_server_(gpus_per_server),
      tp_size_(tp_size),
      dp_size_(dp_size),
      ep_size_(ep_size),
      pp_size_(pp_size),
      global_flow_id_(0),
      global_operation_id_(0),
      comm_domains_set_(false),
      external_ranktable_set_(false) {
    http_client_.initialize(oxc_server_url);
    // 初始化全局通信域（一次性计算）
    initCommDomains();
    // 注意：RankTable 必须通过 setRankTable() 从外部文件加载
}

OxcFlowGenerator::~OxcFlowGenerator() {
}

void OxcFlowGenerator::setAlgorithm(const std::string& alg_name) {
    alg_name_ = alg_name;
}

void OxcFlowGenerator::setRankTable(const RankTable& ranktable) {
    global_ranktable_ = ranktable;
    external_ranktable_set_ = true;
    std::cout << "[OXC] External RankTable set with " << ranktable.rank_count << " ranks" << std::endl;
}

void OxcFlowGenerator::setRankRackMap(const std::map<std::string, std::string>& rank_rack_map) {
    global_rank_rack_map_ = rank_rack_map;
    std::cout << "[OXC] External RankRackMap set with " << rank_rack_map.size() << " entries" << std::endl;
}

bool OxcFlowGenerator::hasExternalRankTable() const {
    return external_ranktable_set_;
}

bool OxcFlowGenerator::isOxcSupported(CommType comm_type) const {
    // 目前OXC只支持AllReduce
    return comm_type == CommType::ALL_REDUCE;
}

int OxcFlowGenerator::getNextFlowId() {
    return global_flow_id_++;
}

int OxcFlowGenerator::getNextOperationId() {
    return global_operation_id_++;
}

const std::vector<OutputFlow>& OxcFlowGenerator::getAllFlows() const {
    return all_flows_;
}

const std::vector<OperationContext>& OxcFlowGenerator::getAllOperations() const {
    return all_operations_;
}

std::vector<std::vector<int>> OxcFlowGenerator::buildCommDomains(
    GroupType group_type,
    int total_gpus) {

    // 如果已经缓存了通信域，直接返回
    if (comm_domains_set_) {
        auto it = comm_domains_.find(group_type);
        if (it != comm_domains_.end()) {
            return it->second;
        }
        // 如果没有找到该类型的通信域，返回空
        return {};
    }

    // 如果没有缓存，动态计算（兼容旧代码）
    return computeCommDomains(group_type, total_gpus);
}

void OxcFlowGenerator::printCommDomainDetails(
    const std::string& name,
    const std::vector<std::vector<int>>& domains,
    int max_groups_to_print) {

    std::cout << "  " << name << ": " << domains.size() << " groups";
    if (!domains.empty()) {
        std::cout << ", " << domains[0].size() << " ranks/group";
    }
    std::cout << std::endl;

    // 打印每个组的详细信息（限制打印数量避免输出过多）
    int groups_to_print = std::min(static_cast<int>(domains.size()), max_groups_to_print);
    for (int i = 0; i < groups_to_print; ++i) {
        std::cout << "    Group " << i << ": [";
        for (size_t j = 0; j < domains[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << domains[i][j];
        }
        std::cout << "]" << std::endl;
    }
    if (static_cast<int>(domains.size()) > max_groups_to_print) {
        std::cout << "    ... (" << (domains.size() - max_groups_to_print)
                  << " more groups)" << std::endl;
    }
}

void OxcFlowGenerator::initCommDomains() {
    // 一次性计算所有类型的通信域并缓存
    comm_domains_[GroupType::TP] = computeCommDomains(GroupType::TP, num_gpus_);
    comm_domains_[GroupType::DP] = computeCommDomains(GroupType::DP, num_gpus_);
    comm_domains_[GroupType::EP] = computeCommDomains(GroupType::EP, num_gpus_);
    comm_domains_[GroupType::DP_EP] = computeCommDomains(GroupType::DP_EP, num_gpus_);
    comm_domains_set_ = true;

    // 打印通信域详细信息
    std::cout << "[OXC] ========== Communication Domains ==========" << std::endl;
    std::cout << "[OXC] Configuration: total_gpus=" << num_gpus_
              << ", gpus_per_server=" << gpus_per_server_
              << ", TP=" << tp_size_
              << ", DP=" << dp_size_
              << ", EP=" << ep_size_
              << ", PP=" << pp_size_ << std::endl;

    // 打印每种通信域的详细信息（最多打印4个组）
    printCommDomainDetails("TP", comm_domains_[GroupType::TP], 4);
    printCommDomainDetails("DP", comm_domains_[GroupType::DP], 4);
    printCommDomainDetails("EP", comm_domains_[GroupType::EP], 4);
    printCommDomainDetails("DP_EP", comm_domains_[GroupType::DP_EP], 4);

    std::cout << "[OXC] ============================================" << std::endl;
}

std::vector<std::vector<int>> OxcFlowGenerator::computeCommDomains(
    GroupType group_type,
    int total_gpus) {

    std::vector<std::vector<int>> domains;

    // 仿照 MockNcclGroup 的通信域创建逻辑
    // 约束: TP_size * DP_size * PP_size = total_gpus
    // 约束: EP_size * DP_EP_size = DP_size

    // 参数保护：防止除零和无效参数
    int tp_size = (tp_size_ > 0 && tp_size_ <= total_gpus) ? tp_size_ : 1;
    int dp_size = (dp_size_ > 0 && dp_size_ <= total_gpus) ? dp_size_ : 1;
    int ep_size = (ep_size_ > 0 && ep_size_ <= total_gpus) ? ep_size_ : 1;

    int TP_nums = total_gpus / tp_size;  // TP 组数量
    int DP_nums = total_gpus / dp_size;  // DP 组数量

    // 计算 DP_EP_size (如果 EP_size > 1)
    int dp_ep_size = (ep_size > 1 && dp_size >= ep_size) ? (dp_size / ep_size) : 1;

    switch (group_type) {
        case GroupType::TP: {
            // TP组：连续的 rank 组成一个 TP 组
            // 例如 TP_size=4, total=16: [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
            if (tp_size > 1) {
                for (int i = 0; i < TP_nums; ++i) {
                    std::vector<int> domain;
                    for (int j = 0; j < tp_size; ++j) {
                        int rank = i * tp_size + j;
                        domain.push_back(rank);
                    }
                    domains.push_back(domain);
                }
            }
            break;
        }
        case GroupType::DP: {
            // DP组：跨 TP 组的相同位置 rank 组成 DP 组
            // 例如 DP_size=4, DP_nums=4, total=16: [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]]
            if (dp_size > 1) {
                for (int i = 0; i < DP_nums; ++i) {
                    std::vector<int> domain;
                    for (int j = 0; j < dp_size; ++j) {
                        int rank = i + j * DP_nums;
                        domain.push_back(rank);
                    }
                    domains.push_back(domain);
                }
            }
            break;
        }
        case GroupType::EP: {
            // EP组：基于 TP 组，跨多个连续 TP 组选择相同位置的 rank
            // 例如 EP_size=2, TP_size=4, total=16:
            // TP组: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
            // EP组: [0,4], [1,5], [2,6], [3,7], [8,12], [9,13], [10,14], [11,15]
            if (ep_size > 1 && TP_nums >= ep_size) {
                // 先构建所有 TP 组
                std::vector<std::vector<int>> tp_groups;
                for (int i = 0; i < TP_nums; ++i) {
                    std::vector<int> tp_group;
                    for (int j = 0; j < tp_size; ++j) {
                        tp_group.push_back(i * tp_size + j);
                    }
                    tp_groups.push_back(tp_group);
                }

                // 每 EP_size 个连续 TP 组形成一组 EP 域
                for (int i = 0; i < TP_nums; i += ep_size) {
                    // 对于 TP 组内的每个位置
                    for (int k = 0; k < tp_size; ++k) {
                        std::vector<int> domain;
                        // 从 EP_size 个连续 TP 组中取相同位置的 rank
                        for (int l = i; l < i + ep_size && l < TP_nums; ++l) {
                            domain.push_back(tp_groups[l][k]);
                        }
                        if (domain.size() > 1) {
                            domains.push_back(domain);
                        }
                    }
                }
            }
            break;
        }
        case GroupType::DP_EP: {
            // DP_EP组：类似 EP，但步长为 EP_size
            // 例如 DP_EP_size=2, EP_size=2, TP_size=4, total=16:
            // DP_EP组: [0,8], [1,9], [2,10], [3,11], [4,12], [5,13], [6,14], [7,15]
            if (dp_ep_size > 1 && ep_size > 0) {
                // 先构建所有 TP 组
                std::vector<std::vector<int>> tp_groups;
                for (int i = 0; i < TP_nums; ++i) {
                    std::vector<int> tp_group;
                    for (int j = 0; j < tp_size; ++j) {
                        tp_group.push_back(i * tp_size + j);
                    }
                    tp_groups.push_back(tp_group);
                }

                // 每隔 EP_size 个 TP 组取一个，共取 DP_EP_size 个
                for (int i = 0; i < ep_size && i < TP_nums; ++i) {
                    for (int k = 0; k < tp_size; ++k) {
                        std::vector<int> domain;
                        for (int l = i; l < TP_nums; l += ep_size) {
                            domain.push_back(tp_groups[l][k]);
                        }
                        if (domain.size() > 1) {
                            domains.push_back(domain);
                        }
                    }
                }
            }
            break;
        }
        default: {
            // 默认：所有GPU在一个组
            std::vector<int> domain;
            for (int i = 0; i < total_gpus; ++i) {
                domain.push_back(i);
            }
            domains.push_back(domain);
            break;
        }
    }

    return domains;
}

std::vector<OutputFlow> OxcFlowGenerator::convertOxcResponse(
    const std::vector<OxcFlowEntry>& entries,
    const OperationContext& ctx) {

    std::vector<OutputFlow> flows;

    // 按step分组，用于建立依赖关系
    // 使用 map<step, map<dst_rank, flow_id>> 优化查找效率 O(1)
    std::map<int, std::map<int, int>> step_dst_to_flow_id;
    int base_flow_id = global_flow_id_;

    for (const auto& entry : entries) {
        OutputFlow flow;
        flow.operation_id = ctx.operation_id;
        flow.layer_name = ctx.layer_name;
        flow.phase = ctx.phase;
        flow.comm_type = ctx.comm_type;
        flow.group_type = ctx.group_type;
        flow.flow_id = getNextFlowId();
        flow.src = entry.src_rank;
        flow.dst = entry.dst_rank;
        flow.flow_size = entry.datasize;
        flow.step = entry.step;

        // 设置依赖：依赖于前一个step中目标为当前源的流
        if (entry.step > 0) {
            auto step_it = step_dst_to_flow_id.find(entry.step - 1);
            if (step_it != step_dst_to_flow_id.end()) {
                auto dst_it = step_it->second.find(entry.src_rank);
                if (dst_it != step_it->second.end()) {
                    flow.depends_on.push_back(dst_it->second);
                }
            }
        }

        // 记录当前流的 dst -> flow_id 映射，供后续 step 查找
        step_dst_to_flow_id[entry.step][entry.dst_rank] = flow.flow_id;
        flows.push_back(flow);
    }

    return flows;
}

std::vector<OutputFlow> OxcFlowGenerator::generateAllReduceViaOxc(
    const OperationContext& ctx,
    const std::vector<int>& comm_group_ranks) {

    // 检查是否跨 rack 通信
    // OXC 算法只适用于跨 rack 的通信，同一 rack 内的通信使用原生算法
    std::set<std::string> racks;
    for (int rank : comm_group_ranks) {
        std::string rank_str = std::to_string(rank);
        auto it = global_rank_rack_map_.find(rank_str);
        if (it != global_rank_rack_map_.end()) {
            racks.insert(it->second);
        } else {
            // 如果 rank_rack_map 中没有该 rank，使用 gpus_per_server_ 计算作为回退
            int rack_id = rank / gpus_per_server_;
            racks.insert("rack_" + std::to_string(rack_id));
        }
    }

    if (racks.size() <= 1) {
        // 所有 rank 都在同一个 rack，使用原生算法
        static int native_log_count = 0;
        if (native_log_count < 5) {
            std::cout << "[OXC] Using NATIVE (same rack): op=" << ctx.operation_id
                      << ", racks=" << racks.size() << std::endl;
            native_log_count++;
        }
        return generateViaNative(ctx, comm_group_ranks);
    }

    // 跨 rack 通信，调用 OXC API
    static int oxc_log_count = 0;
    if (oxc_log_count < 5) {
        std::cout << "[OXC] Calling OXC API (cross-rack): op=" << ctx.operation_id
                  << ", racks=" << racks.size() << ", ranks=[";
        for (size_t i = 0; i < std::min(comm_group_ranks.size(), static_cast<size_t>(4)); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << comm_group_ranks[i];
        }
        std::cout << "]" << std::endl;
        oxc_log_count++;
    }

    OxcAllReduceRequest request;

    // 使用全局 RankTable
    request.ranktable = global_ranktable_;

    // 构建dpCommDomain - 使用传入的通信组
    request.dpCommDomain.push_back(comm_group_ranks);

    // 设置通信量
    request.commDomainVolume = static_cast<double>(ctx.data_size);

    // 使用全局 rank 到 rack 的映射
    request.rankIdRackIdMap = global_rank_rack_map_;

    // 设置算法名称
    request.algName = alg_name_;

    // 调用OXC API
    std::vector<OxcFlowEntry> entries = http_client_.callAllReduceApi(request);

    if (entries.empty()) {
        std::cerr << "[OXC] Warning: Empty response from OXC API for operation "
                  << ctx.operation_id << ", error: " << http_client_.getLastError()
                  << std::endl;
        // 回退到原生实现
        return generateViaNative(ctx, comm_group_ranks);
    }

    std::cout << "[OXC] Received " << entries.size() << " flow entries for "
              << ctx.layer_name << " " << phaseToString(ctx.phase) << std::endl;

    return convertOxcResponse(entries, ctx);
}

std::vector<OutputFlow> OxcFlowGenerator::generateViaNative(
    const OperationContext& ctx,
    const std::vector<int>& comm_group_ranks) {

    std::vector<OutputFlow> flows;
    int num_ranks = static_cast<int>(comm_group_ranks.size());

    if (num_ranks <= 1) {
        return flows;
    }

    // 简单的Ring算法实现
    switch (ctx.comm_type) {
        case CommType::ALL_REDUCE: {
            // AllReduce = ReduceScatter + AllGather
            // 简化：生成Ring通信模式
            uint64_t chunk_size = ctx.data_size / num_ranks;

            // ReduceScatter阶段
            for (int step = 0; step < num_ranks - 1; ++step) {
                for (int i = 0; i < num_ranks; ++i) {
                    OutputFlow flow;
                    flow.operation_id = ctx.operation_id;
                    flow.layer_name = ctx.layer_name;
                    flow.phase = ctx.phase;
                    flow.comm_type = ctx.comm_type;
                    flow.group_type = ctx.group_type;
                    flow.flow_id = getNextFlowId();
                    flow.src = comm_group_ranks[i];
                    flow.dst = comm_group_ranks[(i + 1) % num_ranks];
                    flow.flow_size = chunk_size;
                    flow.step = step;

                    if (step > 0) {
                        // 依赖于前一个step
                        flow.depends_on.push_back(flow.flow_id - num_ranks);
                    }

                    flows.push_back(flow);
                }
            }

            // AllGather阶段
            for (int step = 0; step < num_ranks - 1; ++step) {
                for (int i = 0; i < num_ranks; ++i) {
                    OutputFlow flow;
                    flow.operation_id = ctx.operation_id;
                    flow.layer_name = ctx.layer_name;
                    flow.phase = ctx.phase;
                    flow.comm_type = ctx.comm_type;
                    flow.group_type = ctx.group_type;
                    flow.flow_id = getNextFlowId();
                    flow.src = comm_group_ranks[i];
                    flow.dst = comm_group_ranks[(i + 1) % num_ranks];
                    flow.flow_size = chunk_size;
                    flow.step = (num_ranks - 1) + step;

                    // 依赖于前一个step
                    flow.depends_on.push_back(flow.flow_id - num_ranks);

                    flows.push_back(flow);
                }
            }
            break;
        }

        case CommType::ALL_GATHER: {
            // AllGather: 每个rank发送数据到所有其他rank
            uint64_t chunk_size = ctx.data_size / num_ranks;

            for (int step = 0; step < num_ranks - 1; ++step) {
                for (int i = 0; i < num_ranks; ++i) {
                    OutputFlow flow;
                    flow.operation_id = ctx.operation_id;
                    flow.layer_name = ctx.layer_name;
                    flow.phase = ctx.phase;
                    flow.comm_type = ctx.comm_type;
                    flow.group_type = ctx.group_type;
                    flow.flow_id = getNextFlowId();
                    flow.src = comm_group_ranks[i];
                    flow.dst = comm_group_ranks[(i + 1) % num_ranks];
                    flow.flow_size = chunk_size;
                    flow.step = step;

                    if (step > 0) {
                        flow.depends_on.push_back(flow.flow_id - num_ranks);
                    }

                    flows.push_back(flow);
                }
            }
            break;
        }

        case CommType::REDUCE_SCATTER: {
            // ReduceScatter: 类似AllGather但带reduce
            uint64_t chunk_size = ctx.data_size / num_ranks;

            for (int step = 0; step < num_ranks - 1; ++step) {
                for (int i = 0; i < num_ranks; ++i) {
                    OutputFlow flow;
                    flow.operation_id = ctx.operation_id;
                    flow.layer_name = ctx.layer_name;
                    flow.phase = ctx.phase;
                    flow.comm_type = ctx.comm_type;
                    flow.group_type = ctx.group_type;
                    flow.flow_id = getNextFlowId();
                    flow.src = comm_group_ranks[i];
                    flow.dst = comm_group_ranks[(i + 1) % num_ranks];
                    flow.flow_size = chunk_size;
                    flow.step = step;

                    if (step > 0) {
                        flow.depends_on.push_back(flow.flow_id - num_ranks);
                    }

                    flows.push_back(flow);
                }
            }
            break;
        }

        case CommType::ALL_TO_ALL: {
            // AllToAll: 每个rank发送不同数据到每个其他rank
            uint64_t chunk_size = ctx.data_size / (num_ranks * num_ranks);

            for (int src_idx = 0; src_idx < num_ranks; ++src_idx) {
                for (int dst_idx = 0; dst_idx < num_ranks; ++dst_idx) {
                    if (src_idx == dst_idx) continue;

                    OutputFlow flow;
                    flow.operation_id = ctx.operation_id;
                    flow.layer_name = ctx.layer_name;
                    flow.phase = ctx.phase;
                    flow.comm_type = ctx.comm_type;
                    flow.group_type = ctx.group_type;
                    flow.flow_id = getNextFlowId();
                    flow.src = comm_group_ranks[src_idx];
                    flow.dst = comm_group_ranks[dst_idx];
                    flow.flow_size = chunk_size;
                    flow.step = 0;  // AllToAll可以并行执行

                    flows.push_back(flow);
                }
            }
            break;
        }

        default:
            break;
    }

    return flows;
}

std::vector<OutputFlow> OxcFlowGenerator::generateFlows(
    const OperationContext& ctx,
    const std::vector<int>& comm_group_ranks) {

    std::vector<OutputFlow> flows;

    if (comm_group_ranks.size() <= 1) {
        return flows;
    }

    // 记录操作
    OperationContext op_ctx = ctx;
    op_ctx.operation_id = getNextOperationId();
    op_ctx.base_flow_id = global_flow_id_;

    // 调试输出（每1000个操作输出一次，避免输出过多）
    static int op_log_count = 0;
    bool should_debug = (op_log_count < 10) || (op_log_count % 1000 == 0);

    if (should_debug) {
        std::cout << "[OXC DEBUG] Op " << op_ctx.operation_id
                  << ": comm_type=" << commTypeToString(ctx.comm_type)
                  << ", group_type=" << groupTypeToString(ctx.group_type)
                  << ", phase=" << phaseToString(ctx.phase)
                  << ", ranks=[";
        for (size_t i = 0; i < std::min(comm_group_ranks.size(), static_cast<size_t>(4)); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << comm_group_ranks[i];
        }
        if (comm_group_ranks.size() > 4) std::cout << "...";
        std::cout << "]" << std::endl;
    }
    op_log_count++;

    if (isOxcSupported(ctx.comm_type)) {
        flows = generateAllReduceViaOxc(op_ctx, comm_group_ranks);
    } else {
        flows = generateViaNative(op_ctx, comm_group_ranks);
    }

    op_ctx.flow_count = static_cast<int>(flows.size());
    all_operations_.push_back(op_ctx);

    // 添加到全局流列表
    all_flows_.insert(all_flows_.end(), flows.begin(), flows.end());

    return flows;
}

}  // namespace OXC
