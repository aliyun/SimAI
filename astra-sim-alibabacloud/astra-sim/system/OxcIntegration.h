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

#ifndef __OXC_INTEGRATION_H__
#define __OXC_INTEGRATION_H__

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <atomic>
#include "astra-sim/system/MockNcclChannel.h"
#include "astra-sim/system/OxcTypes.h"

namespace OxcIntegration {

// 环境变量控制
// AS_OXC_ENABLE=1 启用 OXC 集成
// AS_OXC_URL=http://localhost:8080 OXC 服务器地址
// AS_OXC_ALGO=ALGO_OXC_RING OXC 算法名称
// AS_OXC_RANKTABLE=/path/to/ranktable.json RankTable 文件路径
// AS_OXC_RANK_RACK_MAP=/path/to/rank_rack_map.json Rank-Rack 映射文件路径
// AS_OXC_HTTP_TIMEOUT=30 HTTP 请求超时（秒）
// AS_OXC_CONNECT_TIMEOUT=5 HTTP 连接超时（秒）

/**
 * OXC 集成配置
 */
struct OxcConfig {
    bool enabled;                    // 是否启用 OXC
    std::string server_url;          // OXC 服务器 URL
    std::string algorithm;           // OXC 算法名称
    int gpus_per_server;             // 每服务器 NPU 数量
    int http_timeout_seconds;        // HTTP 请求超时（秒）
    int http_connect_timeout_seconds; // HTTP 连接超时（秒）
    std::string ranktable_file;      // RankTable JSON 文件路径
    std::string rank_rack_map_file;  // Rank-Rack 映射 JSON 文件路径

    OxcConfig();
    static OxcConfig fromEnvironment();
};

/**
 * OXC 集成适配器
 * 负责将 OXC 流生成结果转换为 SimAI 的 SingleFlow 格式
 */
class OxcAdapter {
public:
    OxcAdapter();
    ~OxcAdapter();

    /**
     * 初始化适配器
     * @param config OXC 配置
     * @return 是否初始化成功
     */
    bool initialize(const OxcConfig& config);

    /**
     * 检查是否应该使用 OXC 生成流
     * @param group_ranks 通信组的 rank 列表
     * @param comm_type 通信类型
     * @return 是否使用 OXC
     */
    bool shouldUseOxc(
        const std::vector<int>& group_ranks,
        MockNccl::ComType comm_type
    ) const;

    /**
     * 通过 OXC 生成 AllReduce 流
     * @param group_ranks 通信组的 rank 列表
     * @param data_size 数据大小
     * @param base_flow_id 起始流 ID
     * @param channel_id 通道 ID
     * @return SingleFlow 映射 (pair<channel_id, flow_id> -> SingleFlow)
     */
    std::map<std::pair<int,int>, MockNccl::SingleFlow> generateAllReduceFlows(
        const std::vector<int>& group_ranks,
        uint64_t data_size,
        int base_flow_id,
        int channel_id
    );

    /**
     * 设置 RankTable（从外部加载）
     */
    void setRankTable(const OXC::RankTable& ranktable);

    /**
     * 从 JSON 文件加载 RankTable
     * @param filepath ranktable.json 文件路径
     * @return 是否加载成功
     */
    bool loadRankTableFromFile(const std::string& filepath);

    /**
     * 设置 Rank 到 Rack 的映射
     */
    void setRankRackMap(const std::map<std::string, std::string>& rank_rack_map);

    /**
     * 从 JSON 文件加载 Rank-Rack 映射
     * @param filepath rank_rack_map.json 文件路径
     * @return 是否加载成功
     */
    bool loadRankRackMapFromFile(const std::string& filepath);

    /**
     * 检查通信组是否跨 Rack
     */
    bool isCrossRack(const std::vector<int>& group_ranks) const;

    /**
     * 获取最后的错误信息
     */
    std::string getLastError() const;

    /**
     * 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }

    /**
     * 检查 OXC 是否启用
     */
    bool isEnabled() const { return config_.enabled; }

private:
    /**
     * 调用 OXC API 获取流
     */
    std::vector<OXC::OxcFlowEntry> callOxcApi(
        const std::vector<int>& group_ranks,
        uint64_t data_size
    );

    /**
     * 构建 OXC 请求
     */
    OXC::OxcAllReduceRequest buildRequest(
        const std::vector<int>& group_ranks,
        uint64_t data_size
    );

    /**
     * 生成默认的 RankTable
     */
    OXC::RankTable generateDefaultRankTable(int num_ranks);

    /**
     * 生成默认的 Rank-Rack 映射
     */
    std::map<std::string, std::string> generateDefaultRankRackMap(
        const std::vector<int>& ranks
    );

    OxcConfig config_;
    bool initialized_;
    std::string last_error_;

    // RankTable 和映射
    OXC::RankTable ranktable_;
    std::map<std::string, std::string> rank_rack_map_;
    bool external_ranktable_set_;
};

/**
 * 全局 OXC 适配器单例
 */
OxcAdapter& getGlobalOxcAdapter();

/**
 * 初始化全局 OXC 适配器（在程序启动时调用）
 */
bool initializeGlobalOxcAdapter();

}  // namespace OxcIntegration

#endif  // __OXC_INTEGRATION_H__
