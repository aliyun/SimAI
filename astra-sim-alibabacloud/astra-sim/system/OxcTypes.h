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

#ifndef __OXC_TYPES_H__
#define __OXC_TYPES_H__

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace OXC {

// RankTable 相关结构体 - 匹配 Java API 格式

struct RankAddr {
    std::string addr_type;      // "EID"
    std::string addr;           // "000000000000002000100000df001001"
    std::vector<std::string> ports;  // ["0/0"]
    std::string plane_id;       // "plane0"
};

struct LevelInfo {
    int net_layer;              // 0
    std::string net_instance_id; // "superpod1_1"
    std::string net_type;       // "TOPO_FILE_DESC"
    std::string net_attr;       // ""
    std::vector<RankAddr> rank_addr_list;
};

struct RankInfo {
    int rank_id;
    int device_id;
    int local_id;
    std::vector<LevelInfo> level_list;
};

struct RankTable {
    std::string version = "2.0";
    std::string status = "completed";
    int rank_count;
    std::vector<RankInfo> rank_list;
};

// OXC AllReduce API 请求结构
struct OxcAllReduceRequest {
    RankTable ranktable;
    std::vector<std::vector<int>> dpCommDomain;
    double commDomainVolume;
    std::map<std::string, std::string> rankIdRackIdMap;
    std::string algName;  // "ALGO_OXC_RING", "ALGO_OXC_HD", "ALGO_OXC_NB"
};

// OXC API 响应 - 单个流条目
// 响应格式: [[src_rank, dst_rank, step, datasize], ...]
struct OxcFlowEntry {
    int src_rank;
    int dst_rank;
    int step;
    uint64_t datasize;
};

// 通信类型枚举
enum class CommType {
    NONE,
    ALL_REDUCE,
    ALL_GATHER,
    REDUCE_SCATTER,
    ALL_TO_ALL,
    ALL_REDUCE_ALL_TO_ALL
};

// 通信组类型枚举
enum class GroupType {
    TP,      // Tensor Parallelism
    DP,      // Data Parallelism
    PP,      // Pipeline Parallelism
    EP,      // Expert Parallelism
    DP_EP,   // Combined DP and EP
    NONE
};

// 训练阶段枚举
enum class TrainingPhase {
    FORWARD_PASS,
    INPUT_GRADIENT,
    WEIGHT_GRADIENT
};

// 层通信信息
struct LayerCommInfo {
    std::string layer_name;
    int layer_index;

    // 前向传播通信
    CommType fwd_comm_type;
    GroupType fwd_group_type;
    uint64_t fwd_comm_size;

    // 输入梯度通信
    CommType ig_comm_type;
    GroupType ig_group_type;
    uint64_t ig_comm_size;

    // 权重梯度通信
    CommType wg_comm_type;
    GroupType wg_group_type;
    uint64_t wg_comm_size;
};

// 工作负载配置
struct WorkloadConfig {
    std::string parallelism_policy;
    int model_parallel_npu_group = 1;  // TP size
    int ep_size = 1;                   // EP size
    int pp_size = 1;                   // PP size
    int vpp = 1;                       // Virtual PP
    int ga = 1;                        // Gradient Accumulation
    int all_gpus = 0;                  // Total GPUs
    int gpus_per_server = 8;           // GPUs per server
    int num_layers = 0;
    std::vector<LayerCommInfo> layers;
};

// 输出流结构
struct OutputFlow {
    int operation_id;
    std::string layer_name;
    TrainingPhase phase;
    CommType comm_type;
    GroupType group_type;
    int flow_id;
    int src;
    int dst;
    uint64_t flow_size;
    int step;
    std::vector<int> depends_on;  // 该流依赖的流ID列表
};

// 操作上下文
struct OperationContext {
    int operation_id;
    std::string layer_name;
    int layer_index;
    TrainingPhase phase;
    CommType comm_type;
    GroupType group_type;
    uint64_t data_size;
    int base_flow_id;
    int flow_count;
    std::vector<int> depends_on_ops;  // 依赖的操作ID列表
};

// 辅助函数：将枚举转换为字符串
inline std::string commTypeToString(CommType type) {
    switch (type) {
        case CommType::NONE: return "NONE";
        case CommType::ALL_REDUCE: return "ALLREDUCE";
        case CommType::ALL_GATHER: return "ALLGATHER";
        case CommType::REDUCE_SCATTER: return "REDUCESCATTER";
        case CommType::ALL_TO_ALL: return "ALLTOALL";
        case CommType::ALL_REDUCE_ALL_TO_ALL: return "ALLREDUCEALLTOALL";
        default: return "UNKNOWN";
    }
}

inline std::string groupTypeToString(GroupType type) {
    switch (type) {
        case GroupType::TP: return "TP";
        case GroupType::DP: return "DP";
        case GroupType::PP: return "PP";
        case GroupType::EP: return "EP";
        case GroupType::DP_EP: return "DP_EP";
        case GroupType::NONE: return "NONE";
        default: return "UNKNOWN";
    }
}

inline std::string phaseToString(TrainingPhase phase) {
    switch (phase) {
        case TrainingPhase::FORWARD_PASS: return "fwd";
        case TrainingPhase::INPUT_GRADIENT: return "ig";
        case TrainingPhase::WEIGHT_GRADIENT: return "wg";
        default: return "unknown";
    }
}

// 从字符串解析通信类型
inline CommType parseCommType(const std::string& str) {
    if (str.find("ALLREDUCE") != std::string::npos &&
        str.find("ALLTOALL") == std::string::npos) {
        return CommType::ALL_REDUCE;
    } else if (str.find("ALLGATHER") != std::string::npos) {
        return CommType::ALL_GATHER;
    } else if (str.find("REDUCESCATTER") != std::string::npos) {
        return CommType::REDUCE_SCATTER;
    } else if (str.find("ALLTOALL") != std::string::npos &&
               str.find("ALLREDUCE") == std::string::npos) {
        return CommType::ALL_TO_ALL;
    } else if (str.find("ALLREDUCEALLTOALL") != std::string::npos) {
        return CommType::ALL_REDUCE_ALL_TO_ALL;
    }
    return CommType::NONE;
}

// 从字符串解析组类型
inline GroupType parseGroupType(const std::string& str, TrainingPhase phase) {
    if (str.find("_DP_EP") != std::string::npos) {
        return GroupType::DP_EP;
    } else if (str.find("_EP") != std::string::npos) {
        return GroupType::EP;
    }
    // 默认：权重梯度用DP，其他用TP
    if (phase == TrainingPhase::WEIGHT_GRADIENT) {
        return GroupType::DP;
    }
    return GroupType::TP;
}

}  // namespace OXC

#endif  // __OXC_TYPES_H__
