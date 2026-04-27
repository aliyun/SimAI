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

#ifndef __OXC_FLOW_OUTPUT_H__
#define __OXC_FLOW_OUTPUT_H__

#include <string>
#include <vector>
#include "astra-sim/system/OxcTypes.h"

namespace OXC {

class OxcFlowOutput {
public:
    OxcFlowOutput(const std::string& output_prefix);
    ~OxcFlowOutput();

    // 写入流矩阵到CSV文件，返回是否成功
    bool writeFlowMatrices(const std::vector<OutputFlow>& flows);

    // 写入依赖图到JSON文件，返回是否成功
    bool writeDependencyGraph(
        const std::vector<OperationContext>& operations,
        const std::vector<OutputFlow>& flows
    );

    // 写入摘要信息，返回是否成功
    bool writeSummary(
        const WorkloadConfig& config,
        const std::vector<OperationContext>& operations,
        const std::vector<OutputFlow>& flows,
        const std::string& oxc_url,
        const std::string& alg_name
    );

private:
    std::string output_prefix_;
};

}  // namespace OXC

#endif  // __OXC_FLOW_OUTPUT_H__
