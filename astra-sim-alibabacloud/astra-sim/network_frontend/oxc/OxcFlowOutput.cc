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

#include "OxcFlowOutput.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

namespace OXC {

OxcFlowOutput::OxcFlowOutput(const std::string& output_prefix)
    : output_prefix_(output_prefix) {
}

OxcFlowOutput::~OxcFlowOutput() {
}

bool OxcFlowOutput::writeFlowMatrices(const std::vector<OutputFlow>& flows) {
    std::string filename = output_prefix_ + "_flows.csv";
    std::ofstream ofs(filename);

    if (!ofs.is_open()) {
        std::cerr << "[OXC] Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    // 写入CSV头
    ofs << "op_id,layer,phase,comm_type,group,flow_id,src,dst,size,step,depends_on" << std::endl;

    // 写入每个流
    for (const auto& flow : flows) {
        ofs << flow.operation_id << ","
            << flow.layer_name << ","
            << phaseToString(flow.phase) << ","
            << commTypeToString(flow.comm_type) << ","
            << groupTypeToString(flow.group_type) << ","
            << flow.flow_id << ","
            << flow.src << ","
            << flow.dst << ","
            << flow.flow_size << ","
            << flow.step << ",";

        // 写入依赖列表
        ofs << "\"[";
        for (size_t i = 0; i < flow.depends_on.size(); ++i) {
            if (i > 0) ofs << ",";
            ofs << flow.depends_on[i];
        }
        ofs << "]\"" << std::endl;
    }

    ofs.close();
    std::cout << "[OXC] Flow matrices written to " << filename << std::endl;
    return true;
}

bool OxcFlowOutput::writeDependencyGraph(
    const std::vector<OperationContext>& operations,
    const std::vector<OutputFlow>& flows) {

    std::string filename = output_prefix_ + "_deps.json";
    std::ofstream ofs(filename);

    if (!ofs.is_open()) {
        std::cerr << "[OXC] Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    ofs << "{" << std::endl;

    // 写入操作列表
    ofs << "  \"operations\": [" << std::endl;
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        ofs << "    {" << std::endl;
        ofs << "      \"op_id\": " << op.operation_id << "," << std::endl;
        ofs << "      \"layer\": \"" << op.layer_name << "\"," << std::endl;
        ofs << "      \"layer_index\": " << op.layer_index << "," << std::endl;
        ofs << "      \"phase\": \"" << phaseToString(op.phase) << "\"," << std::endl;
        ofs << "      \"type\": \"" << commTypeToString(op.comm_type) << "\"," << std::endl;
        ofs << "      \"group\": \"" << groupTypeToString(op.group_type) << "\"," << std::endl;
        ofs << "      \"data_size\": " << op.data_size << "," << std::endl;
        ofs << "      \"flow_count\": " << op.flow_count << "," << std::endl;
        ofs << "      \"depends_on\": [";
        for (size_t j = 0; j < op.depends_on_ops.size(); ++j) {
            if (j > 0) ofs << ", ";
            ofs << op.depends_on_ops[j];
        }
        ofs << "]" << std::endl;
        ofs << "    }";
        if (i < operations.size() - 1) ofs << ",";
        ofs << std::endl;
    }
    ofs << "  ]," << std::endl;

    // 构建操作间依赖关系
    std::map<int, std::vector<int>> op_dependencies;
    for (size_t i = 1; i < operations.size(); ++i) {
        // 简单的顺序依赖：每个操作依赖于前一个操作
        op_dependencies[operations[i].operation_id].push_back(operations[i-1].operation_id);
    }

    // 写入依赖关系
    ofs << "  \"dependencies\": {" << std::endl;
    bool first = true;
    for (const auto& pair : op_dependencies) {
        if (!first) ofs << "," << std::endl;
        first = false;
        ofs << "    \"" << pair.first << "\": [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (i > 0) ofs << ", ";
            ofs << pair.second[i];
        }
        ofs << "]";
    }
    ofs << std::endl;
    ofs << "  }" << std::endl;

    ofs << "}" << std::endl;

    ofs.close();
    std::cout << "[OXC] Dependency graph written to " << filename << std::endl;
    return true;
}

bool OxcFlowOutput::writeSummary(
    const WorkloadConfig& config,
    const std::vector<OperationContext>& operations,
    const std::vector<OutputFlow>& flows,
    const std::string& oxc_url,
    const std::string& alg_name) {

    std::string filename = output_prefix_ + "_summary.txt";
    std::ofstream ofs(filename);

    if (!ofs.is_open()) {
        std::cerr << "[OXC] Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    ofs << "SimAI-OXC Flow Generation Summary" << std::endl;
    ofs << "=================================" << std::endl;
    ofs << std::endl;

    ofs << "Workload Configuration:" << std::endl;
    ofs << "  Parallelism Policy: " << config.parallelism_policy << std::endl;
    ofs << "  Total GPUs: " << config.all_gpus << std::endl;
    ofs << "  GPUs per Server: " << config.gpus_per_server << std::endl;
    ofs << "  TP Size: " << config.model_parallel_npu_group << std::endl;
    ofs << "  EP Size: " << config.ep_size << std::endl;
    ofs << "  PP Size: " << config.pp_size << std::endl;
    ofs << "  VPP: " << config.vpp << std::endl;
    ofs << "  GA: " << config.ga << std::endl;
    ofs << "  Number of Layers: " << config.num_layers << std::endl;
    ofs << std::endl;

    ofs << "OXC Configuration:" << std::endl;
    ofs << "  Server URL: " << oxc_url << std::endl;
    ofs << "  Algorithm: " << alg_name << std::endl;
    ofs << std::endl;

    // 统计各类型操作数量
    std::map<CommType, int> op_counts;
    std::map<CommType, int> oxc_op_counts;
    for (const auto& op : operations) {
        op_counts[op.comm_type]++;
        if (op.comm_type == CommType::ALL_REDUCE) {
            oxc_op_counts[op.comm_type]++;
        }
    }

    ofs << "Operations Processed:" << std::endl;
    ofs << "  Total Operations: " << operations.size() << std::endl;
    for (const auto& pair : op_counts) {
        std::string type_str = commTypeToString(pair.first);
        bool is_oxc = (pair.first == CommType::ALL_REDUCE);
        ofs << "  - " << type_str << ": " << pair.second;
        if (is_oxc) {
            ofs << " (OXC)";
        } else {
            ofs << " (Native)";
        }
        ofs << std::endl;
    }
    ofs << std::endl;

    ofs << "Flow Statistics:" << std::endl;
    ofs << "  Total Flows Generated: " << flows.size() << std::endl;

    // 统计依赖数量
    int total_deps = 0;
    for (const auto& flow : flows) {
        total_deps += static_cast<int>(flow.depends_on.size());
    }
    ofs << "  Total Dependencies: " << total_deps << std::endl;
    ofs << std::endl;

    ofs << "Output Files:" << std::endl;
    ofs << "  Flow Matrix: " << output_prefix_ << "_flows.csv" << std::endl;
    ofs << "  Dependency Graph: " << output_prefix_ << "_deps.json" << std::endl;
    ofs << "  Summary: " << output_prefix_ << "_summary.txt" << std::endl;

    ofs.close();
    std::cout << "[OXC] Summary written to " << filename << std::endl;
    return true;
}

}  // namespace OXC
