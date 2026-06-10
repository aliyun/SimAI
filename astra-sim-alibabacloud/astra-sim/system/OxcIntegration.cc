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

#include "OxcIntegration.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <curl/curl.h>

namespace OxcIntegration {

// ============================================================
// OxcConfig 实现
// ============================================================

OxcConfig::OxcConfig()
    : enabled(false),
      server_url("http://localhost:8080"),
      algorithm("ALGO_OXC_RING"),
      gpus_per_server(8),
      http_timeout_seconds(30),
      http_connect_timeout_seconds(5) {
}

OxcConfig OxcConfig::fromEnvironment() {
    OxcConfig config;

    // AS_OXC_ENABLE
    const char* oxc_enable = std::getenv("AS_OXC_ENABLE");
    if (oxc_enable != nullptr) {
        config.enabled = (std::string(oxc_enable) == "1" ||
                         std::string(oxc_enable) == "true");
    }

    // AS_OXC_URL
    const char* oxc_url = std::getenv("AS_OXC_URL");
    if (oxc_url != nullptr) {
        config.server_url = std::string(oxc_url);
    }

    // AS_OXC_ALGO
    const char* oxc_algo = std::getenv("AS_OXC_ALGO");
    if (oxc_algo != nullptr) {
        config.algorithm = std::string(oxc_algo);
    }

    // AS_OXC_GPUS_PER_SERVER
    const char* gpus_per_server = std::getenv("AS_OXC_GPUS_PER_SERVER");
    if (gpus_per_server != nullptr) {
        config.gpus_per_server = std::atoi(gpus_per_server);
    }

    // AS_OXC_RANKTABLE - RankTable JSON 文件路径
    const char* ranktable_file = std::getenv("AS_OXC_RANKTABLE");
    if (ranktable_file != nullptr) {
        config.ranktable_file = std::string(ranktable_file);
    }

    // AS_OXC_RANK_RACK_MAP - Rank-Rack 映射 JSON 文件路径
    const char* rank_rack_map_file = std::getenv("AS_OXC_RANK_RACK_MAP");
    if (rank_rack_map_file != nullptr) {
        config.rank_rack_map_file = std::string(rank_rack_map_file);
    }

    // AS_OXC_HTTP_TIMEOUT - HTTP 请求超时（秒）
    const char* http_timeout = std::getenv("AS_OXC_HTTP_TIMEOUT");
    if (http_timeout != nullptr) {
        config.http_timeout_seconds = std::atoi(http_timeout);
    }

    // AS_OXC_CONNECT_TIMEOUT - HTTP 连接超时（秒）
    const char* connect_timeout = std::getenv("AS_OXC_CONNECT_TIMEOUT");
    if (connect_timeout != nullptr) {
        config.http_connect_timeout_seconds = std::atoi(connect_timeout);
    }

    return config;
}

// ============================================================
// HTTP 回调函数
// ============================================================

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// ============================================================
// OxcAdapter 实现
// ============================================================

OxcAdapter::OxcAdapter()
    : initialized_(false),
      external_ranktable_set_(false) {
}

OxcAdapter::~OxcAdapter() {
}

bool OxcAdapter::initialize(const OxcConfig& config) {
    config_ = config;

    if (!config_.enabled) {
        std::cout << "[OXC Integration] OXC is disabled" << std::endl;
        initialized_ = true;
        return true;
    }

    // 初始化 libcurl（线程安全，只执行一次）
    static std::once_flag curl_init_flag;
    std::call_once(curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });

    std::cout << "[OXC Integration] Initialized with:" << std::endl;
    std::cout << "  Server URL: " << config_.server_url << std::endl;
    std::cout << "  Algorithm: " << config_.algorithm << std::endl;
    std::cout << "  NPUs per server: " << config_.gpus_per_server << std::endl;
    std::cout << "  HTTP timeout: " << config_.http_timeout_seconds << "s" << std::endl;
    std::cout << "  Connect timeout: " << config_.http_connect_timeout_seconds << "s" << std::endl;

    // 先从文件加载 Rank-Rack 映射（如果指定）
    // 这样后续加载 RankTable 时不会覆盖用户指定的映射
    if (!config_.rank_rack_map_file.empty()) {
        std::cout << "  Rank-Rack map file: " << config_.rank_rack_map_file << std::endl;
        if (!loadRankRackMapFromFile(config_.rank_rack_map_file)) {
            std::cerr << "[OXC Integration] Warning: Failed to load Rank-Rack map from "
                      << config_.rank_rack_map_file << ": " << last_error_ << std::endl;
        }
    }

    // 从文件加载 RankTable（如果指定）
    // 如果没有指定 rank_rack_map 文件，会自动从 RankTable 生成
    bool ranktable_load_failed = false;
    if (!config_.ranktable_file.empty()) {
        std::cout << "  RankTable file: " << config_.ranktable_file << std::endl;
        if (!loadRankTableFromFile(config_.ranktable_file)) {
            std::cerr << "[OXC Integration] ERROR: Failed to load RankTable from "
                      << config_.ranktable_file << ": " << last_error_ << std::endl;
            std::cerr << "[OXC Integration] OXC will be disabled due to RankTable load failure" << std::endl;
            ranktable_load_failed = true;
        }
    }

    // 如果指定了 RankTable 文件但加载失败，禁用 OXC
    if (ranktable_load_failed) {
        config_.enabled = false;
        initialized_ = true;
        return false;
    }

    initialized_ = true;
    return true;
}

bool OxcAdapter::shouldUseOxc(
    const std::vector<int>& group_ranks,
    MockNccl::ComType comm_type) const {

    if (!initialized_ || !config_.enabled) {
        return false;
    }

    // 目前只支持 AllReduce
    if (comm_type != MockNccl::ComType::All_Reduce) {
        return false;
    }

    // 检查是否跨 Rack
    bool cross_rack = isCrossRack(group_ranks);

    // 打印调试信息（只打印前几次）
    static std::atomic<int> debug_count(0);
    if (debug_count.load(std::memory_order_relaxed) < 5) {
        std::cout << "[OXC Integration] shouldUseOxc: ranks=[";
        for (size_t i = 0; i < std::min(group_ranks.size(), static_cast<size_t>(4)); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << group_ranks[i];
        }
        if (group_ranks.size() > 4) std::cout << "...";
        std::cout << "], cross_rack=" << (cross_rack ? "true" : "false") << std::endl;
        debug_count.fetch_add(1, std::memory_order_relaxed);
    }

    return cross_rack;
}

bool OxcAdapter::isCrossRack(const std::vector<int>& group_ranks) const {
    std::set<std::string> racks;
    std::map<int, std::string> rank_to_rack;  // 用于调试

    for (int rank : group_ranks) {
        std::string rank_str = std::to_string(rank);
        std::string rack_id;
        auto it = rank_rack_map_.find(rank_str);
        if (it != rank_rack_map_.end()) {
            rack_id = it->second;
        } else {
            // 默认映射：rank / gpus_per_server
            int rack_num = rank / config_.gpus_per_server;
            rack_id = "rack_" + std::to_string(rack_num);
        }
        racks.insert(rack_id);
        rank_to_rack[rank] = rack_id;
    }

    // 打印调试信息（只打印前几次）
    static std::atomic<int> cross_rack_debug_count(0);
    if (cross_rack_debug_count.load(std::memory_order_relaxed) < 3 && racks.size() > 1) {
        std::cout << "[OXC Integration] isCrossRack: found " << racks.size() << " racks: ";
        for (const auto& rack : racks) {
            std::cout << rack << " ";
        }
        std::cout << std::endl;
        cross_rack_debug_count.fetch_add(1, std::memory_order_relaxed);
    }

    return racks.size() > 1;
}

void OxcAdapter::setRankTable(const OXC::RankTable& ranktable) {
    ranktable_ = ranktable;
    external_ranktable_set_ = true;
    std::cout << "[OXC Integration] External RankTable set with "
              << ranktable.rank_count << " ranks" << std::endl;
}

void OxcAdapter::setRankRackMap(const std::map<std::string, std::string>& rank_rack_map) {
    rank_rack_map_ = rank_rack_map;
    std::cout << "[OXC Integration] RankRackMap set with "
              << rank_rack_map.size() << " entries" << std::endl;
}

// ============================================================
// JSON 文件加载实现
// ============================================================

// 辅助函数：去除字符串首尾空白
static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

// 辅助函数：去除字符串首尾引号
static std::string unquote(const std::string& str) {
    std::string s = trim(str);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// 辅助函数：查找匹配的括号位置
static size_t findMatchingBracket(const std::string& json, size_t start, char open, char close) {
    int depth = 1;
    for (size_t i = start + 1; i < json.size(); ++i) {
        if (json[i] == open) depth++;
        else if (json[i] == close) {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

// 辅助函数：提取 JSON 字符串值
static std::string extractStringValue(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

// 辅助函数：提取 JSON 整数值
static int extractIntValue(const std::string& json, const std::string& key, int default_val = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;

    // 跳过空白
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    // 读取数字
    std::string num_str;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '-')) {
        num_str += json[pos++];
    }

    if (num_str.empty()) return default_val;
    return std::atoi(num_str.c_str());
}

// 辅助函数：提取 JSON 数组
static std::vector<std::string> extractStringArray(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find('[', pos);
    if (pos == std::string::npos) return result;

    size_t end = findMatchingBracket(json, pos, '[', ']');
    if (end == std::string::npos) return result;

    std::string arr = json.substr(pos + 1, end - pos - 1);

    // 解析数组元素
    size_t i = 0;
    while (i < arr.size()) {
        size_t start = arr.find('"', i);
        if (start == std::string::npos) break;
        size_t elem_end = arr.find('"', start + 1);
        if (elem_end == std::string::npos) break;
        result.push_back(arr.substr(start + 1, elem_end - start - 1));
        i = elem_end + 1;
    }

    return result;
}

bool OxcAdapter::loadRankTableFromFile(const std::string& filepath) {
    // 读取文件
    std::ifstream file(filepath);
    if (!file.is_open()) {
        last_error_ = "Cannot open file: " + filepath;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    file.close();

    // 解析 RankTable
    OXC::RankTable ranktable;

    // 解析基本字段
    ranktable.version = extractStringValue(json, "version");
    if (ranktable.version.empty()) ranktable.version = "2.0";

    ranktable.status = extractStringValue(json, "status");
    if (ranktable.status.empty()) ranktable.status = "completed";

    ranktable.rank_count = extractIntValue(json, "rank_count", 0);

    // 解析 rank_list
    std::string search = "\"rank_list\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        last_error_ = "rank_list not found in JSON";
        return false;
    }

    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        last_error_ = "rank_list array not found";
        return false;
    }

    size_t rank_list_end = findMatchingBracket(json, pos, '[', ']');
    if (rank_list_end == std::string::npos) {
        last_error_ = "rank_list array not properly closed";
        return false;
    }

    std::string rank_list_str = json.substr(pos + 1, rank_list_end - pos - 1);

    // 解析每个 rank
    size_t rank_pos = 0;
    while (rank_pos < rank_list_str.size()) {
        size_t obj_start = rank_list_str.find('{', rank_pos);
        if (obj_start == std::string::npos) break;

        size_t obj_end = findMatchingBracket(rank_list_str, obj_start, '{', '}');
        if (obj_end == std::string::npos) break;

        std::string rank_obj = rank_list_str.substr(obj_start, obj_end - obj_start + 1);

        OXC::RankInfo rank_info;
        rank_info.rank_id = extractIntValue(rank_obj, "rank_id", -1);
        rank_info.device_id = extractIntValue(rank_obj, "device_id", 0);
        rank_info.local_id = extractIntValue(rank_obj, "local_id", 0);

        // 解析 level_list
        size_t level_pos = rank_obj.find("\"level_list\"");
        if (level_pos != std::string::npos) {
            size_t level_arr_start = rank_obj.find('[', level_pos);
            if (level_arr_start != std::string::npos) {
                size_t level_arr_end = findMatchingBracket(rank_obj, level_arr_start, '[', ']');
                if (level_arr_end != std::string::npos) {
                    std::string level_list_str = rank_obj.substr(level_arr_start + 1, level_arr_end - level_arr_start - 1);

                    // 解析每个 level
                    size_t lv_pos = 0;
                    while (lv_pos < level_list_str.size()) {
                        size_t lv_start = level_list_str.find('{', lv_pos);
                        if (lv_start == std::string::npos) break;

                        size_t lv_end = findMatchingBracket(level_list_str, lv_start, '{', '}');
                        if (lv_end == std::string::npos) break;

                        std::string level_obj = level_list_str.substr(lv_start, lv_end - lv_start + 1);

                        OXC::LevelInfo level;
                        level.net_layer = extractIntValue(level_obj, "net_layer", 0);
                        level.net_instance_id = extractStringValue(level_obj, "net_instance_id");
                        level.net_type = extractStringValue(level_obj, "net_type");
                        level.net_attr = extractStringValue(level_obj, "net_attr");

                        // 解析 rank_addr_list
                        size_t addr_pos = level_obj.find("\"rank_addr_list\"");
                        if (addr_pos != std::string::npos) {
                            size_t addr_arr_start = level_obj.find('[', addr_pos);
                            if (addr_arr_start != std::string::npos) {
                                size_t addr_arr_end = findMatchingBracket(level_obj, addr_arr_start, '[', ']');
                                if (addr_arr_end != std::string::npos) {
                                    std::string addr_list_str = level_obj.substr(addr_arr_start + 1, addr_arr_end - addr_arr_start - 1);

                                    // 解析每个 addr
                                    size_t ad_pos = 0;
                                    while (ad_pos < addr_list_str.size()) {
                                        size_t ad_start = addr_list_str.find('{', ad_pos);
                                        if (ad_start == std::string::npos) break;

                                        size_t ad_end = findMatchingBracket(addr_list_str, ad_start, '{', '}');
                                        if (ad_end == std::string::npos) break;

                                        std::string addr_obj = addr_list_str.substr(ad_start, ad_end - ad_start + 1);

                                        OXC::RankAddr addr;
                                        addr.addr_type = extractStringValue(addr_obj, "addr_type");
                                        addr.addr = extractStringValue(addr_obj, "addr");
                                        addr.plane_id = extractStringValue(addr_obj, "plane_id");
                                        addr.ports = extractStringArray(addr_obj, "ports");

                                        level.rank_addr_list.push_back(addr);
                                        ad_pos = ad_end + 1;
                                    }
                                }
                            }
                        }

                        rank_info.level_list.push_back(level);
                        lv_pos = lv_end + 1;
                    }
                }
            }
        }

        ranktable.rank_list.push_back(rank_info);
        rank_pos = obj_end + 1;
    }

    // 如果没有解析到 rank_count，使用 rank_list 大小
    if (ranktable.rank_count == 0) {
        ranktable.rank_count = static_cast<int>(ranktable.rank_list.size());
    }

    // 设置 RankTable
    setRankTable(ranktable);

    // 自动从 RankTable 生成 Rank-Rack 映射（如果尚未设置）
    if (rank_rack_map_.empty()) {
        std::map<std::string, std::string> auto_rank_rack_map;
        for (const auto& rank_info : ranktable.rank_list) {
            std::string rack_id;
            // 从 level_list 中提取 net_instance_id 作为 rack_id
            if (!rank_info.level_list.empty()) {
                rack_id = rank_info.level_list[0].net_instance_id;
            }
            if (rack_id.empty()) {
                // 如果没有 net_instance_id，使用默认映射
                rack_id = "rack_" + std::to_string(rank_info.rank_id / config_.gpus_per_server);
            }
            auto_rank_rack_map[std::to_string(rank_info.rank_id)] = rack_id;
        }
        if (!auto_rank_rack_map.empty()) {
            setRankRackMap(auto_rank_rack_map);
            std::cout << "[OXC Integration] Auto-generated Rank-Rack map from RankTable:" << std::endl;
            // 打印 Rack 分布摘要
            std::map<std::string, std::vector<int>> rack_to_ranks;
            for (const auto& kv : auto_rank_rack_map) {
                rack_to_ranks[kv.second].push_back(std::stoi(kv.first));
            }
            for (const auto& kv : rack_to_ranks) {
                std::cout << "  " << kv.first << ": ranks [";
                for (size_t i = 0; i < std::min(kv.second.size(), static_cast<size_t>(4)); ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << kv.second[i];
                }
                if (kv.second.size() > 4) std::cout << "...";
                std::cout << "] (" << kv.second.size() << " NPUs)" << std::endl;
            }
        }
    }

    std::cout << "[OXC Integration] Loaded RankTable from " << filepath
              << " with " << ranktable.rank_count << " ranks" << std::endl;

    return true;
}

bool OxcAdapter::loadRankRackMapFromFile(const std::string& filepath) {
    // 读取文件
    std::ifstream file(filepath);
    if (!file.is_open()) {
        last_error_ = "Cannot open file: " + filepath;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    file.close();

    // 解析 JSON 对象 {"rank_id": "rack_id", ...}
    std::map<std::string, std::string> rank_rack_map;

    // 查找所有 key-value 对
    size_t pos = 0;
    while (pos < json.size()) {
        // 找到 key
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;

        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        // 找到冒号
        size_t colon = json.find(':', key_end);
        if (colon == std::string::npos) break;

        // 找到 value
        size_t val_start = json.find('"', colon);
        if (val_start == std::string::npos) break;

        size_t val_end = json.find('"', val_start + 1);
        if (val_end == std::string::npos) break;

        std::string value = json.substr(val_start + 1, val_end - val_start - 1);

        // 跳过非数字 key（如 "version" 等元数据字段）
        if (!key.empty() && (isdigit(key[0]) || key[0] == '-')) {
            rank_rack_map[key] = value;
        }

        pos = val_end + 1;
    }

    if (rank_rack_map.empty()) {
        last_error_ = "No rank-rack mappings found in JSON";
        return false;
    }

    // 设置映射
    setRankRackMap(rank_rack_map);

    std::cout << "[OXC Integration] Loaded Rank-Rack map from " << filepath
              << " with " << rank_rack_map.size() << " entries" << std::endl;

    return true;
}

std::string OxcAdapter::getLastError() const {
    return last_error_;
}

std::map<std::pair<int,int>, MockNccl::SingleFlow> OxcAdapter::generateAllReduceFlows(
    const std::vector<int>& group_ranks,
    uint64_t data_size,
    int base_flow_id,
    int channel_id) {

    std::map<std::pair<int,int>, MockNccl::SingleFlow> result;

    if (!initialized_ || !config_.enabled) {
        last_error_ = "OXC adapter not initialized or disabled";
        return result;
    }

    // 调用 OXC API
    std::vector<OXC::OxcFlowEntry> entries = callOxcApi(group_ranks, data_size);

    if (entries.empty()) {
        std::cerr << "[OXC Integration] Failed to get flows from OXC: "
                  << last_error_ << std::endl;
        return result;
    }

    // 计算 chunk 数量（基于 step 数）
    int max_step = 0;
    for (const auto& entry : entries) {
        if (entry.step > max_step) {
            max_step = entry.step;
        }
    }
    int chunk_count = max_step + 1;

    // ================================================================
    // 构建辅助映射
    // ================================================================

    // step -> (dst_rank -> src_rank): 在 step N，谁发送到 dst_rank
    // 用于计算 prev（NcclTreeFlowModel 需要 prev 作为接收源 rank）
    // 注意：ring 算法保证每个 step 中每个 rank 最多作为一个 dst
    std::map<int, std::map<int, int>> step_dst_to_src;
    for (const auto& entry : entries) {
        auto& dst_map = step_dst_to_src[entry.step];
        if (dst_map.count(entry.dst_rank)) {
            std::cerr << "[OXC Integration] WARNING: duplicate dst_rank "
                      << entry.dst_rank << " at step " << entry.step
                      << " (overwriting src " << dst_map[entry.dst_rank]
                      << " with " << entry.src_rank << ")" << std::endl;
        }
        dst_map[entry.dst_rank] = entry.src_rank;
    }

    // (step, src_rank) -> flow_id: 用于依赖关系查找
    // ring 算法保证每个 step 中每个 rank 最多发送一个流
    std::map<std::pair<int, int>, int> step_src_to_flow_id;

    // ================================================================
    // 第一遍：创建所有 SingleFlow（正确的 key、prev、conn_type）
    // ================================================================

    int flow_id = base_flow_id;
    for (const auto& entry : entries) {
        MockNccl::SingleFlow sf;
        sf.flow_id = flow_id;
        sf.src = entry.src_rank;
        sf.dest = entry.dst_rank;
        sf.flow_size = entry.datasize;
        sf.channel_id = channel_id;
        sf.chunk_id = entry.step;
        sf.chunk_count = chunk_count;
        sf.conn_type = "RING";  // 必须是 RING，与 NcclTreeFlowModel 兼容

        // prev = 在当前 step 中，谁发送到 src（即 src 从谁接收数据）
        // NcclTreeFlowModel 使用 prev[0] 作为 sim_recv 的源 rank
        auto step_it = step_dst_to_src.find(entry.step);
        if (step_it != step_dst_to_src.end()) {
            auto src_it = step_it->second.find(entry.src_rank);
            if (src_it != step_it->second.end()) {
                sf.prev.push_back(src_it->second);
            }
        }
        // NcclTreeFlowModel::insert_packets 访问 prev[0]，空 prev 会崩溃
        if (sf.prev.empty()) {
            std::cerr << "[OXC Integration] ERROR: No prev for flow at step "
                      << entry.step << " src=" << entry.src_rank
                      << " dst=" << entry.dst_rank
                      << " (ring algorithm expects symmetric send/recv)" << std::endl;
        }

        auto src_key = std::make_pair(entry.step, entry.src_rank);
        if (step_src_to_flow_id.count(src_key)) {
            std::cerr << "[OXC Integration] WARNING: duplicate (step="
                      << entry.step << ", src=" << entry.src_rank
                      << ") - overwriting flow_id " << step_src_to_flow_id[src_key]
                      << " with " << flow_id << std::endl;
        }
        step_src_to_flow_id[src_key] = flow_id;

        // Key: (channel_id, flow_id) — 与 ring 路径格式一致
        result[std::make_pair(channel_id, flow_id)] = sf;
        flow_id++;
    }

    // ================================================================
    // 第二遍：设置 parent_flow_id 和 child_flow_id 依赖
    // ================================================================
    // 数据流链：step N-1 中 X->src 的流完成后，step N 中 src->dst 才能开始
    // parent_flow_id = step N-1 中发送到 sf.src 的流
    // child_flow_id = step N+1 中从 sf.dest 发送的流

    for (auto& kv : result) {
        MockNccl::SingleFlow& sf = kv.second;
        int step = sf.chunk_id;

        if (step > 0) {
            // 找到 step-1 中发送到 sf.src 的流
            auto prev_step_it = step_dst_to_src.find(step - 1);
            if (prev_step_it != step_dst_to_src.end()) {
                auto sender_it = prev_step_it->second.find(sf.src);
                if (sender_it != prev_step_it->second.end()) {
                    int prev_sender = sender_it->second;
                    auto fid_it = step_src_to_flow_id.find(
                        std::make_pair(step - 1, prev_sender));
                    if (fid_it != step_src_to_flow_id.end()) {
                        sf.parent_flow_id.push_back(fid_it->second);
                    }
                }
            }
        }

        if (step < chunk_count - 1) {
            // 找到 step+1 中从 sf.dest 发送的流
            auto fid_it = step_src_to_flow_id.find(
                std::make_pair(step + 1, sf.dest));
            if (fid_it != step_src_to_flow_id.end()) {
                sf.child_flow_id.push_back(fid_it->second);
            }
        }
    }

    std::cout << "[OXC Integration] Generated " << result.size()
              << " flows via OXC for " << group_ranks.size() << " ranks"
              << ", channel_id=" << channel_id
              << ", chunk_count=" << chunk_count << std::endl;

    return result;
}

std::vector<OXC::OxcFlowEntry> OxcAdapter::callOxcApi(
    const std::vector<int>& group_ranks,
    uint64_t data_size) {

    std::vector<OXC::OxcFlowEntry> entries;

    // 构建请求
    OXC::OxcAllReduceRequest request = buildRequest(group_ranks, data_size);

    // 构建 JSON
    std::ostringstream json;
    json << "{";

    // ranktable
    json << "\"ranktable\":{";
    json << "\"version\":\"" << request.ranktable.version << "\",";
    json << "\"status\":\"" << request.ranktable.status << "\",";
    json << "\"rank_count\":" << request.ranktable.rank_count << ",";
    json << "\"rank_list\":[";
    for (size_t i = 0; i < request.ranktable.rank_list.size(); ++i) {
        if (i > 0) json << ",";
        const auto& rank_info = request.ranktable.rank_list[i];
        json << "{\"rank_id\":" << rank_info.rank_id;
        json << ",\"device_id\":" << rank_info.device_id;
        json << ",\"local_id\":" << rank_info.local_id;
        json << ",\"level_list\":[";
        for (size_t j = 0; j < rank_info.level_list.size(); ++j) {
            if (j > 0) json << ",";
            const auto& level = rank_info.level_list[j];
            json << "{\"net_layer\":" << level.net_layer;
            json << ",\"net_instance_id\":\"" << level.net_instance_id << "\"";
            json << ",\"net_type\":\"" << level.net_type << "\"";
            json << ",\"net_attr\":\"" << level.net_attr << "\"";
            json << ",\"rank_addr_list\":[";
            for (size_t k = 0; k < level.rank_addr_list.size(); ++k) {
                if (k > 0) json << ",";
                const auto& addr = level.rank_addr_list[k];
                json << "{\"addr_type\":\"" << addr.addr_type << "\"";
                json << ",\"addr\":\"" << addr.addr << "\"";
                json << ",\"ports\":[";
                for (size_t l = 0; l < addr.ports.size(); ++l) {
                    if (l > 0) json << ",";
                    json << "\"" << addr.ports[l] << "\"";
                }
                json << "]";
                json << ",\"plane_id\":\"" << addr.plane_id << "\"}";
            }
            json << "]}";
        }
        json << "]}";
    }
    json << "]},";

    // dpCommDomain
    json << "\"dpCommDomain\":[";
    for (size_t i = 0; i < request.dpCommDomain.size(); ++i) {
        if (i > 0) json << ",";
        json << "[";
        for (size_t j = 0; j < request.dpCommDomain[i].size(); ++j) {
            if (j > 0) json << ",";
            json << request.dpCommDomain[i][j];
        }
        json << "]";
    }
    json << "],";

    // commDomainVolume
    json << "\"commDomainVolume\":" << request.commDomainVolume << ",";

    // rankIdRackIdMap
    json << "\"rankIdRackIdMap\":{";
    bool first = true;
    for (const auto& kv : request.rankIdRackIdMap) {
        if (!first) json << ",";
        first = false;
        json << "\"" << kv.first << "\":\"" << kv.second << "\"";
    }
    json << "},";

    // algName
    json << "\"algName\":\"" << request.algName << "\"";
    json << "}";

    std::string json_body = json.str();

    // Circuit breaker: avoid FD leak when OXC server is unreachable (macOS libcurl)
    static bool server_unreachable = false;
    if (server_unreachable) return {};
    CURL* curl = curl_easy_init();
    if (!curl) {
        last_error_ = "Failed to initialize CURL";
        return entries;
    }

    std::string response;
    std::string url = config_.server_url + "/api/oxc/allreduce";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config_.http_timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, static_cast<long>(config_.http_connect_timeout_seconds));

    CURLcode res = curl_easy_perform(curl);

    // 获取 HTTP 状态码
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        if (res == CURLE_OPERATION_TIMEDOUT) {
            last_error_ = "HTTP request timed out after " +
                          std::to_string(config_.http_timeout_seconds) + "s to " + url;
        } else if (res == CURLE_COULDNT_CONNECT) {
            server_unreachable = true;
            last_error_ = "Cannot connect to OXC server at " + url +
                          " (connect timeout: " + std::to_string(config_.http_connect_timeout_seconds) + "s)";
        } else if (res == CURLE_COULDNT_RESOLVE_HOST) {
            last_error_ = "Cannot resolve OXC server host: " + config_.server_url;
        } else {
            last_error_ = std::string("CURL error: ") + curl_easy_strerror(res);
        }
        std::cerr << "[OXC Integration] " << last_error_ << std::endl;
        return entries;
    }

    // 检查 HTTP 状态码
    if (http_code != 200) {
        last_error_ = "HTTP error: " + std::to_string(http_code);
        std::cerr << "[OXC Integration] " << last_error_ << ", response: " << response.substr(0, 200) << std::endl;
        return entries;
    }

    // 检查响应是否包含错误信息
    if (response.find("\"error\"") != std::string::npos ||
        response.find("\"Error\"") != std::string::npos) {
        last_error_ = "OXC API returned error: " + response.substr(0, 200);
        std::cerr << "[OXC Integration] " << last_error_ << std::endl;
        return entries;
    }

    // 解析响应 JSON
    // 响应格式: [[src, dst, step, datasize], ...]
    size_t pos = response.find("[[");
    if (pos == std::string::npos) {
        // 尝试查找空数组
        if (response.find("[]") != std::string::npos) {
            last_error_ = "OXC API returned empty flow list";
            std::cerr << "[OXC Integration] " << last_error_ << std::endl;
            return entries;
        }
        last_error_ = "Invalid response format, expected [[...]], got: " + response.substr(0, 100);
        std::cerr << "[OXC Integration] " << last_error_ << std::endl;
        return entries;
    }

    std::string data = response.substr(pos);

    // 解析数组
    size_t i = 1;  // 跳过第一个 '['
    int parsed_count = 0;
    int parse_errors = 0;
    while (i < data.size()) {
        // 找到 '['
        size_t start = data.find('[', i);
        if (start == std::string::npos) break;

        // 找到 ']'
        size_t end = data.find(']', start);
        if (end == std::string::npos) break;

        // 解析 [src, dst, step, datasize]
        std::string entry_str = data.substr(start + 1, end - start - 1);

        int src, dst, step;
        uint64_t datasize;
        if (sscanf(entry_str.c_str(), "%d,%d,%d,%llu", &src, &dst, &step, (unsigned long long*)&datasize) == 4) {
            OXC::OxcFlowEntry entry;
            entry.src_rank = src;
            entry.dst_rank = dst;
            entry.step = step;
            entry.datasize = datasize;
            entries.push_back(entry);
            parsed_count++;
        } else {
            parse_errors++;
            if (parse_errors <= 3) {
                std::cerr << "[OXC Integration] Warning: Failed to parse entry: " << entry_str << std::endl;
            }
        }

        i = end + 1;
    }

    if (parsed_count == 0 && parse_errors > 0) {
        last_error_ = "Failed to parse any flow entries from response";
        std::cerr << "[OXC Integration] " << last_error_ << std::endl;
    }

    return entries;
}

OXC::OxcAllReduceRequest OxcAdapter::buildRequest(
    const std::vector<int>& group_ranks,
    uint64_t data_size) {

    OXC::OxcAllReduceRequest request;

    // 使用外部 RankTable 或生成默认的
    if (external_ranktable_set_) {
        request.ranktable = ranktable_;
    } else {
        int max_rank = *std::max_element(group_ranks.begin(), group_ranks.end());
        request.ranktable = generateDefaultRankTable(max_rank + 1);
    }

    // 通信域
    request.dpCommDomain.push_back(std::vector<int>(group_ranks.begin(), group_ranks.end()));

    // 数据量
    request.commDomainVolume = static_cast<double>(data_size);

    // Rank-Rack 映射
    if (!rank_rack_map_.empty()) {
        request.rankIdRackIdMap = rank_rack_map_;
    } else {
        request.rankIdRackIdMap = generateDefaultRankRackMap(group_ranks);
    }

    // 算法名称
    request.algName = config_.algorithm;

    return request;
}

OXC::RankTable OxcAdapter::generateDefaultRankTable(int num_ranks) {
    OXC::RankTable ranktable;
    ranktable.version = "2.0";
    ranktable.status = "completed";
    ranktable.rank_count = num_ranks;

    for (int i = 0; i < num_ranks; ++i) {
        OXC::RankInfo rank_info;
        rank_info.rank_id = i;
        rank_info.device_id = i % config_.gpus_per_server;
        rank_info.local_id = i % config_.gpus_per_server;

        OXC::LevelInfo level;
        level.net_layer = 0;
        level.net_instance_id = "rack_" + std::to_string(i / config_.gpus_per_server);
        level.net_type = "TOPO_FILE_DESC";
        level.net_attr = "";

        OXC::RankAddr addr;
        addr.addr_type = "EID";
        std::ostringstream addr_oss;
        addr_oss << std::hex << std::setfill('0') << std::setw(32) << i;
        addr.addr = addr_oss.str();
        addr.ports.push_back("0/0");
        addr.plane_id = "plane0";

        level.rank_addr_list.push_back(addr);
        rank_info.level_list.push_back(level);
        ranktable.rank_list.push_back(rank_info);
    }

    return ranktable;
}

std::map<std::string, std::string> OxcAdapter::generateDefaultRankRackMap(
    const std::vector<int>& ranks) {

    std::map<std::string, std::string> rank_rack_map;

    for (int rank : ranks) {
        int rack_id = rank / config_.gpus_per_server;
        rank_rack_map[std::to_string(rank)] = "rack_" + std::to_string(rack_id);
    }

    return rank_rack_map;
}

// ============================================================
// 全局单例（线程安全）
// ============================================================

static OxcAdapter* g_oxc_adapter = nullptr;
static std::once_flag g_adapter_init_flag;

OxcAdapter& getGlobalOxcAdapter() {
    std::call_once(g_adapter_init_flag, []() {
        g_oxc_adapter = new OxcAdapter();
    });
    return *g_oxc_adapter;
}

bool initializeGlobalOxcAdapter() {
    OxcConfig config = OxcConfig::fromEnvironment();
    return getGlobalOxcAdapter().initialize(config);
}

}  // namespace OxcIntegration
