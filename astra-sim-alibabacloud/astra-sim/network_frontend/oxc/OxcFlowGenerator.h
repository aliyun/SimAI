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

#ifndef __OXC_FLOW_GENERATOR_H__
#define __OXC_FLOW_GENERATOR_H__

#include <string>
#include <vector>
#include <map>
#include "astra-sim/system/OxcTypes.h"
#include "OxcHttpClient.h"

namespace OXC {

class OxcFlowGenerator {
public:
    OxcFlowGenerator(
        const std::string& oxc_server_url,
        int num_gpus,
        int gpus_per_server,
        int tp_size,
        int dp_size,
        int ep_size,
        int pp_size
    );

    ~OxcFlowGenerator();

    // 为一个集合通信操作生成流
    std::vector<OutputFlow> generateFlows(
        const OperationContext& ctx,
        const std::vector<int>& comm_group_ranks
    );

    // 检查OXC是否支持该操作类型
    bool isOxcSupported(CommType comm_type) const;

    // 获取所有生成的流
    const std::vector<OutputFlow>& getAllFlows() const;

    // 获取所有操作上下文
    const std::vector<OperationContext>& getAllOperations() const;

    // 获取下一个流ID
    int getNextFlowId();

    // 获取下一个操作ID
    int getNextOperationId();

    // 设置OXC算法名称
    void setAlgorithm(const std::string& alg_name);

    // 设置外部 RankTable（从 JSON 文件加载）
    void setRankTable(const RankTable& ranktable);

    // 设置外部 rank 到 rack 映射
    void setRankRackMap(const std::map<std::string, std::string>& rank_rack_map);

    // 检查是否已设置外部 RankTable
    bool hasExternalRankTable() const;

    // 从 MockNcclGroup 设置通信域
    // group_type -> list of comm domains, each domain is a list of ranks
    void setCommDomainsFromMockNccl(
        const std::map<GroupType, std::vector<std::vector<int>>>& domains
    );

    // 获取指定类型的通信域
    std::vector<std::vector<int>> getCommDomains(GroupType group_type) const;

    // 构建通信组的rank列表（如果没有从 MockNcclGroup 设置，则自己计算）
    std::vector<std::vector<int>> buildCommDomains(
        GroupType group_type,
        int total_gpus
    );

private:
    // 通过OXC API生成AllReduce流
    std::vector<OutputFlow> generateAllReduceViaOxc(
        const OperationContext& ctx,
        const std::vector<int>& comm_group_ranks
    );

    // 使用原生方式生成流（用于OXC不支持的操作）
    std::vector<OutputFlow> generateViaNative(
        const OperationContext& ctx,
        const std::vector<int>& comm_group_ranks
    );

    // 初始化全局通信域（在构造函数中调用一次）
    void initCommDomains();

    // 打印通信域详细信息
    void printCommDomainDetails(
        const std::string& name,
        const std::vector<std::vector<int>>& domains,
        int max_groups_to_print = 4
    );

    // 计算通信域（内部方法）
    std::vector<std::vector<int>> computeCommDomains(
        GroupType group_type,
        int total_gpus
    );

    // 将OXC响应转换为OutputFlow
    std::vector<OutputFlow> convertOxcResponse(
        const std::vector<OxcFlowEntry>& entries,
        const OperationContext& ctx
    );

    OxcHttpClient http_client_;
    std::string oxc_server_url_;
    std::string alg_name_;
    int num_gpus_;
    int gpus_per_server_;
    int tp_size_;
    int dp_size_;
    int ep_size_;
    int pp_size_;

    int global_flow_id_;
    int global_operation_id_;
    std::vector<OutputFlow> all_flows_;
    std::vector<OperationContext> all_operations_;

    // 全局 RankTable（仿真任务全局唯一）
    RankTable global_ranktable_;
    // 全局 rank 到 rack 的映射
    std::map<std::string, std::string> global_rank_rack_map_;

    // 从 MockNcclGroup 获取的通信域
    std::map<GroupType, std::vector<std::vector<int>>> comm_domains_;
    bool comm_domains_set_;

    // 是否使用外部 RankTable
    bool external_ranktable_set_;
};

}  // namespace OXC

#endif  // __OXC_FLOW_GENERATOR_H__
