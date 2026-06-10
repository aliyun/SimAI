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

#ifndef __OXC_FLOW_SCHEDULER_H__
#define __OXC_FLOW_SCHEDULER_H__

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <set>
#include "astra-sim/system/OxcTypes.h"

namespace OXC {

/**
 * OxcFlowScheduler - 流调度模拟器
 *
 * 功能：
 * 1. 将 depends_on 转换为 parent_flow_id / child_flow_id 双向链接
 * 2. 初始化 indegree_mapping
 * 3. 模拟流的调度执行（入度递减和激活）
 *
 * 不涉及实际网络仿真，只模拟依赖调度逻辑。
 */
class OxcFlowScheduler {
public:
    OxcFlowScheduler();
    ~OxcFlowScheduler();

    /**
     * 从 OutputFlow 列表构建调度流
     * - 转换为双向链接
     * - 初始化入度
     */
    void buildScheduledFlows(const std::vector<OutputFlow>& flows);

    /**
     * 运行调度模拟
     * - 每个 tick 激活入度为0的流
     * - 流完成后递减子流入度
     * - 返回调度统计结果
     */
    ScheduleStats runSimulation();

    /**
     * 获取调度后的流列表
     */
    const std::vector<ScheduledFlow>& getScheduledFlows() const { return scheduled_flows_; }

    /**
     * 获取调度统计
     */
    const ScheduleStats& getStats() const { return stats_; }

    /**
     * 写入调度结果到文件
     */
    bool writeScheduleResult(const std::string& output_prefix);

private:
    // 构建双向链接
    void buildBidirectionalLinks();

    // 初始化入度映射
    void initIndegreeMapping();

    // 激活入度为0的流
    std::vector<int> activateReadyFlows(int current_tick);

    // 完成流并更新子流入度
    void completeFlow(int flow_id, int current_tick);

    std::vector<ScheduledFlow> scheduled_flows_;
    std::map<int, int> flow_id_to_index_;  // flow_id -> 数组索引
    std::map<int, int> indegree_mapping_;  // flow_id -> 入度
    std::set<int> pending_flows_;          // 待调度的流
    std::set<int> active_flows_;           // 正在执行的流
    std::set<int> completed_flows_;        // 已完成的流
    ScheduleStats stats_;
};

}  // namespace OXC

#endif  // __OXC_FLOW_SCHEDULER_H__
