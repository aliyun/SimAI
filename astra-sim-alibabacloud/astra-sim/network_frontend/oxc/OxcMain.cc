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

#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <getopt.h>

#include "astra-sim/system/OxcTypes.h"
#include "OxcFlowGenerator.h"
#include "OxcFlowOutput.h"

using namespace std;
using namespace OXC;

// 命令行参数结构
struct OxcParams {
    string workload_path;
    string ranktable_path;  // 外部 RankTable JSON 文件路径
    int num_gpus;
    int gpus_per_server;
    string oxc_url;
    string oxc_algo;
    string output_prefix;

    OxcParams()
        : num_gpus(16),
          gpus_per_server(8),
          oxc_url("http://localhost:8080"),
          oxc_algo("ALGO_OXC_RING"),
          output_prefix("./results/oxc_output") {}
};

void printUsage(const char* prog_name) {
    cout << "Usage: " << prog_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -w, --workload <path>     Path to workload file (required)" << endl;
    cout << "  -ranktable <path>         Path to RankTable JSON file (required)" << endl;
    cout << "  -g, --gpus <num>          Number of GPUs (default: 16)" << endl;
    cout << "  -g_p_s, --gpus-per-server <num>  GPUs per server (default: 8)" << endl;
    cout << "  -oxc_url <url>            OXC server URL (default: http://localhost:8080)" << endl;
    cout << "  -oxc_algo <algo>          OXC algorithm (default: ALGO_OXC_RING)" << endl;
    cout << "  -o, --output <prefix>     Output file prefix (default: ./results/oxc_output)" << endl;
    cout << "  -h, --help                Show this help message" << endl;
}

int parseArgs(int argc, char* argv[], OxcParams& params) {
    // 手动解析所有参数，避免 getopt 与自定义参数冲突
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-w" || arg == "--workload") && i + 1 < argc) {
            params.workload_path = argv[++i];
        } else if ((arg == "-g" || arg == "--gpus") && i + 1 < argc) {
            params.num_gpus = atoi(argv[++i]);
        } else if ((arg == "-g_p_s" || arg == "--gpus-per-server") && i + 1 < argc) {
            params.gpus_per_server = atoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            params.output_prefix = argv[++i];
        } else if ((arg == "-ranktable" || arg == "--ranktable") && i + 1 < argc) {
            params.ranktable_path = argv[++i];
        } else if (arg == "-oxc_url" && i + 1 < argc) {
            params.oxc_url = argv[++i];
        } else if (arg == "-oxc_algo" && i + 1 < argc) {
            params.oxc_algo = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 1;
        }
    }

    if (params.workload_path.empty()) {
        cerr << "Error: Workload path is required" << endl;
        printUsage(argv[0]);
        return -1;
    }

    if (params.ranktable_path.empty()) {
        cerr << "Error: RankTable path is required" << endl;
        cerr << "  Use generate_ranktable.py to create a RankTable JSON file" << endl;
        printUsage(argv[0]);
        return -1;
    }

    return 0;
}

// ============== JSON 解析辅助函数 ==============

// 跳过空白字符
static size_t skipWhitespace(const string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\t' || s[pos] == '\r')) {
        pos++;
    }
    return pos;
}

// 解析 JSON 字符串值
static string parseJsonString(const string& s, size_t& pos) {
    pos = skipWhitespace(s, pos);
    if (pos >= s.size() || s[pos] != '"') return "";
    pos++;  // 跳过开头的 "

    string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;  // 跳过转义字符
        }
        result += s[pos++];
    }
    if (pos < s.size()) pos++;  // 跳过结尾的 "
    return result;
}

// 解析 JSON 整数值
static int64_t parseJsonInt(const string& s, size_t& pos) {
    pos = skipWhitespace(s, pos);
    bool negative = false;
    if (pos < s.size() && s[pos] == '-') {
        negative = true;
        pos++;
    }
    int64_t result = 0;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        result = result * 10 + (s[pos] - '0');
        pos++;
    }
    return negative ? -result : result;
}

// 查找 JSON 键
static size_t findJsonKey(const string& s, size_t pos, const string& key) {
    string search = "\"" + key + "\"";
    size_t found = s.find(search, pos);
    if (found != string::npos) {
        found += search.size();
        // 跳过冒号
        found = skipWhitespace(s, found);
        if (found < s.size() && s[found] == ':') {
            found++;
        }
    }
    return found;
}

// 解析 RankTable JSON 文件
bool parseRankTableJson(const string& filepath, RankTable& ranktable, map<string, string>& rank_rack_map) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Cannot open RankTable file: " << filepath << endl;
        return false;
    }

    // 读取整个文件
    stringstream buffer;
    buffer << file.rdbuf();
    string json = buffer.str();
    file.close();

    size_t pos = 0;

    // 解析 version
    pos = findJsonKey(json, 0, "version");
    if (pos != string::npos) {
        ranktable.version = parseJsonString(json, pos);
    }

    // 解析 status
    pos = findJsonKey(json, 0, "status");
    if (pos != string::npos) {
        ranktable.status = parseJsonString(json, pos);
    }

    // 解析 rank_count
    pos = findJsonKey(json, 0, "rank_count");
    if (pos != string::npos) {
        ranktable.rank_count = static_cast<int>(parseJsonInt(json, pos));
    }

    // 解析 rank_list
    pos = findJsonKey(json, 0, "rank_list");
    if (pos == string::npos) {
        cerr << "Error: rank_list not found in RankTable JSON" << endl;
        return false;
    }

    // 找到 rank_list 数组的开始
    pos = skipWhitespace(json, pos);
    if (pos >= json.size() || json[pos] != '[') {
        cerr << "Error: rank_list is not an array" << endl;
        return false;
    }
    pos++;  // 跳过 '['

    // 解析每个 rank
    while (pos < json.size()) {
        pos = skipWhitespace(json, pos);
        if (pos >= json.size() || json[pos] == ']') break;
        if (json[pos] == ',') { pos++; continue; }

        if (json[pos] != '{') break;

        // 找到这个 rank 对象的结束位置
        int brace_count = 1;
        size_t rank_start = pos;
        pos++;
        while (pos < json.size() && brace_count > 0) {
            if (json[pos] == '{') brace_count++;
            else if (json[pos] == '}') brace_count--;
            pos++;
        }
        string rank_json = json.substr(rank_start, pos - rank_start);

        RankInfo rank_info;
        size_t rpos = 0;

        // 解析 rank_id
        rpos = findJsonKey(rank_json, 0, "rank_id");
        if (rpos != string::npos) {
            rank_info.rank_id = static_cast<int>(parseJsonInt(rank_json, rpos));
        }

        // 解析 device_id
        rpos = findJsonKey(rank_json, 0, "device_id");
        if (rpos != string::npos) {
            rank_info.device_id = static_cast<int>(parseJsonInt(rank_json, rpos));
        }

        // 解析 local_id
        rpos = findJsonKey(rank_json, 0, "local_id");
        if (rpos != string::npos) {
            rank_info.local_id = static_cast<int>(parseJsonInt(rank_json, rpos));
        }

        // 解析 level_list
        rpos = findJsonKey(rank_json, 0, "level_list");
        if (rpos != string::npos) {
            rpos = skipWhitespace(rank_json, rpos);
            if (rpos < rank_json.size() && rank_json[rpos] == '[') {
                rpos++;

                while (rpos < rank_json.size()) {
                    rpos = skipWhitespace(rank_json, rpos);
                    if (rpos >= rank_json.size() || rank_json[rpos] == ']') break;
                    if (rank_json[rpos] == ',') { rpos++; continue; }

                    if (rank_json[rpos] != '{') break;

                    // 找到这个 level 对象的结束位置
                    int level_brace = 1;
                    size_t level_start = rpos;
                    rpos++;
                    while (rpos < rank_json.size() && level_brace > 0) {
                        if (rank_json[rpos] == '{') level_brace++;
                        else if (rank_json[rpos] == '}') level_brace--;
                        rpos++;
                    }
                    string level_json = rank_json.substr(level_start, rpos - level_start);

                    LevelInfo level;
                    size_t lpos = 0;

                    lpos = findJsonKey(level_json, 0, "net_layer");
                    if (lpos != string::npos) {
                        level.net_layer = static_cast<int>(parseJsonInt(level_json, lpos));
                    }

                    lpos = findJsonKey(level_json, 0, "net_instance_id");
                    if (lpos != string::npos) {
                        level.net_instance_id = parseJsonString(level_json, lpos);
                    }

                    lpos = findJsonKey(level_json, 0, "net_type");
                    if (lpos != string::npos) {
                        level.net_type = parseJsonString(level_json, lpos);
                    }

                    lpos = findJsonKey(level_json, 0, "net_attr");
                    if (lpos != string::npos) {
                        level.net_attr = parseJsonString(level_json, lpos);
                    }

                    // 解析 rank_addr_list
                    lpos = findJsonKey(level_json, 0, "rank_addr_list");
                    if (lpos != string::npos) {
                        lpos = skipWhitespace(level_json, lpos);
                        if (lpos < level_json.size() && level_json[lpos] == '[') {
                            lpos++;

                            while (lpos < level_json.size()) {
                                lpos = skipWhitespace(level_json, lpos);
                                if (lpos >= level_json.size() || level_json[lpos] == ']') break;
                                if (level_json[lpos] == ',') { lpos++; continue; }

                                if (level_json[lpos] != '{') break;

                                // 找到这个 addr 对象的结束位置
                                int addr_brace = 1;
                                size_t addr_start = lpos;
                                lpos++;
                                while (lpos < level_json.size() && addr_brace > 0) {
                                    if (level_json[lpos] == '{') addr_brace++;
                                    else if (level_json[lpos] == '}') addr_brace--;
                                    lpos++;
                                }
                                string addr_json = level_json.substr(addr_start, lpos - addr_start);

                                RankAddr addr;
                                size_t apos = 0;

                                apos = findJsonKey(addr_json, 0, "addr_type");
                                if (apos != string::npos) {
                                    addr.addr_type = parseJsonString(addr_json, apos);
                                }

                                apos = findJsonKey(addr_json, 0, "addr");
                                if (apos != string::npos) {
                                    addr.addr = parseJsonString(addr_json, apos);
                                }

                                apos = findJsonKey(addr_json, 0, "plane_id");
                                if (apos != string::npos) {
                                    addr.plane_id = parseJsonString(addr_json, apos);
                                }

                                // 解析 ports 数组
                                apos = findJsonKey(addr_json, 0, "ports");
                                if (apos != string::npos) {
                                    apos = skipWhitespace(addr_json, apos);
                                    if (apos < addr_json.size() && addr_json[apos] == '[') {
                                        apos++;
                                        while (apos < addr_json.size()) {
                                            apos = skipWhitespace(addr_json, apos);
                                            if (apos >= addr_json.size() || addr_json[apos] == ']') break;
                                            if (addr_json[apos] == ',') { apos++; continue; }
                                            if (addr_json[apos] == '"') {
                                                string port = parseJsonString(addr_json, apos);
                                                if (!port.empty()) {
                                                    addr.ports.push_back(port);
                                                }
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                }

                                level.rank_addr_list.push_back(addr);
                            }
                        }
                    }

                    rank_info.level_list.push_back(level);
                }
            }
        }

        ranktable.rank_list.push_back(rank_info);
    }

    // 自动生成 rank_rack_map（基于 level_list 中的 net_instance_id）
    for (const auto& rank : ranktable.rank_list) {
        if (!rank.level_list.empty()) {
            // 使用第一个 level 的 net_instance_id 作为 rack_id
            rank_rack_map[to_string(rank.rank_id)] = rank.level_list[0].net_instance_id;
        }
    }

    cout << "[OXC] RankTable loaded from: " << filepath << endl;
    cout << "[OXC]   Version: " << ranktable.version << endl;
    cout << "[OXC]   Rank count: " << ranktable.rank_count << endl;
    cout << "[OXC]   Parsed ranks: " << ranktable.rank_list.size() << endl;

    return true;
}

// ============== 工作负载解析 ==============

// 解析工作负载文件
WorkloadConfig parseWorkload(const string& workload_path) {
    WorkloadConfig config;
    ifstream file(workload_path);

    if (!file.is_open()) {
        cerr << "Error: Cannot open workload file: " << workload_path << endl;
        return config;
    }

    string line;

    // 读取第一行：并行策略和配置
    if (getline(file, line)) {
        istringstream iss(line);
        string token;

        // 解析并行策略
        iss >> config.parallelism_policy;

        // 解析键值对
        while (iss >> token) {
            if (token == "model_parallel_NPU_group:") {
                iss >> config.model_parallel_npu_group;
            } else if (token == "ep:") {
                iss >> config.ep_size;
            } else if (token == "pp:") {
                iss >> config.pp_size;
            } else if (token == "vpp:") {
                iss >> config.vpp;
            } else if (token == "ga:") {
                iss >> config.ga;
            } else if (token == "all_gpus:") {
                iss >> config.all_gpus;
            }
        }
    }

    // 读取第二行：层数
    if (getline(file, line)) {
        config.num_layers = stoi(line);
    }

    // 读取层定义
    while (getline(file, line)) {
        if (line.empty()) continue;

        istringstream iss(line);
        LayerCommInfo layer;

        string layer_name;
        int dependency;
        uint64_t fwd_compute, ig_compute, wg_compute, wg_update;
        string fwd_type_str, ig_type_str, wg_type_str;
        uint64_t fwd_size, ig_size, wg_size;

        // 解析层定义
        // 格式: layer_name dependency fwd_compute fwd_type fwd_size ig_compute ig_type ig_size wg_compute wg_type wg_size wg_update
        iss >> layer_name >> dependency
            >> fwd_compute >> fwd_type_str >> fwd_size
            >> ig_compute >> ig_type_str >> ig_size
            >> wg_compute >> wg_type_str >> wg_size
            >> wg_update;

        layer.layer_name = layer_name;
        layer.layer_index = static_cast<int>(config.layers.size());

        // 解析通信类型和组类型
        layer.fwd_comm_type = parseCommType(fwd_type_str);
        layer.fwd_group_type = parseGroupType(fwd_type_str, TrainingPhase::FORWARD_PASS);
        layer.fwd_comm_size = fwd_size;

        layer.ig_comm_type = parseCommType(ig_type_str);
        layer.ig_group_type = parseGroupType(ig_type_str, TrainingPhase::INPUT_GRADIENT);
        layer.ig_comm_size = ig_size;

        layer.wg_comm_type = parseCommType(wg_type_str);
        layer.wg_group_type = parseGroupType(wg_type_str, TrainingPhase::WEIGHT_GRADIENT);
        layer.wg_comm_size = wg_size;

        config.layers.push_back(layer);
    }

    file.close();
    return config;
}

int main(int argc, char* argv[]) {
    OxcParams params;

    int ret = parseArgs(argc, argv, params);
    if (ret != 0) {
        return ret;
    }

    cout << "SimAI-OXC Flow Generator" << endl;
    cout << "========================" << endl;
    cout << "Workload: " << params.workload_path << endl;
    cout << "GPUs: " << params.num_gpus << endl;
    cout << "GPUs per Server: " << params.gpus_per_server << endl;
    cout << "RankTable: " << params.ranktable_path << endl;
    cout << "OXC URL: " << params.oxc_url << endl;
    cout << "OXC Algorithm: " << params.oxc_algo << endl;
    cout << "Output Prefix: " << params.output_prefix << endl;
    cout << endl;

    // 解析工作负载
    cout << "[OXC] Parsing workload..." << endl;
    WorkloadConfig config = parseWorkload(params.workload_path);

    if (config.layers.empty()) {
        cerr << "Error: No layers found in workload" << endl;
        return -1;
    }

    // 更新配置
    config.all_gpus = params.num_gpus;
    config.gpus_per_server = params.gpus_per_server;

    cout << "[OXC] Workload parsed: " << config.num_layers << " layers" << endl;
    cout << "[OXC] Parallelism: TP=" << config.model_parallel_npu_group
         << ", EP=" << config.ep_size
         << ", PP=" << config.pp_size << endl;

    // 计算DP大小
    int dp_size = params.num_gpus / config.model_parallel_npu_group;
    if (config.ep_size > 1) {
        dp_size = dp_size / config.ep_size;
    }

    // 创建流生成器
    OxcFlowGenerator flow_gen(
        params.oxc_url,
        params.num_gpus,
        params.gpus_per_server,
        config.model_parallel_npu_group,
        dp_size,
        config.ep_size,
        config.pp_size
    );
    flow_gen.setAlgorithm(params.oxc_algo);

    // 加载外部 RankTable（必需）
    cout << "[OXC] Loading RankTable from: " << params.ranktable_path << endl;
    RankTable external_ranktable;
    map<string, string> external_rank_rack_map;

    if (!parseRankTableJson(params.ranktable_path, external_ranktable, external_rank_rack_map)) {
        cerr << "Error: Failed to load RankTable from: " << params.ranktable_path << endl;
        return -1;
    }
    flow_gen.setRankTable(external_ranktable);
    flow_gen.setRankRackMap(external_rank_rack_map);

    // 处理每一层
    cout << "[OXC] Generating flows..." << endl;
    int prev_op_id = -1;

    for (const auto& layer : config.layers) {
        // 处理前向传播通信
        if (layer.fwd_comm_type != CommType::NONE && layer.fwd_comm_size > 0) {
            OperationContext ctx;
            ctx.layer_name = layer.layer_name;
            ctx.layer_index = layer.layer_index;
            ctx.phase = TrainingPhase::FORWARD_PASS;
            ctx.comm_type = layer.fwd_comm_type;
            ctx.group_type = layer.fwd_group_type;
            ctx.data_size = layer.fwd_comm_size;
            if (prev_op_id >= 0) {
                ctx.depends_on_ops.push_back(prev_op_id);
            }

            // 构建通信组
            auto domains = flow_gen.buildCommDomains(layer.fwd_group_type, params.num_gpus);
            for (const auto& domain : domains) {
                flow_gen.generateFlows(ctx, domain);
            }
            if (!flow_gen.getAllOperations().empty()) {
                prev_op_id = flow_gen.getAllOperations().back().operation_id;
            }
        }

        // 处理输入梯度通信
        if (layer.ig_comm_type != CommType::NONE && layer.ig_comm_size > 0) {
            OperationContext ctx;
            ctx.layer_name = layer.layer_name;
            ctx.layer_index = layer.layer_index;
            ctx.phase = TrainingPhase::INPUT_GRADIENT;
            ctx.comm_type = layer.ig_comm_type;
            ctx.group_type = layer.ig_group_type;
            ctx.data_size = layer.ig_comm_size;
            if (prev_op_id >= 0) {
                ctx.depends_on_ops.push_back(prev_op_id);
            }

            auto domains = flow_gen.buildCommDomains(layer.ig_group_type, params.num_gpus);
            for (const auto& domain : domains) {
                flow_gen.generateFlows(ctx, domain);
            }
            if (!flow_gen.getAllOperations().empty()) {
                prev_op_id = flow_gen.getAllOperations().back().operation_id;
            }
        }

        // 处理权重梯度通信
        if (layer.wg_comm_type != CommType::NONE && layer.wg_comm_size > 0) {
            OperationContext ctx;
            ctx.layer_name = layer.layer_name;
            ctx.layer_index = layer.layer_index;
            ctx.phase = TrainingPhase::WEIGHT_GRADIENT;
            ctx.comm_type = layer.wg_comm_type;
            ctx.group_type = layer.wg_group_type;
            ctx.data_size = layer.wg_comm_size;
            if (prev_op_id >= 0) {
                ctx.depends_on_ops.push_back(prev_op_id);
            }

            auto domains = flow_gen.buildCommDomains(layer.wg_group_type, params.num_gpus);
            for (const auto& domain : domains) {
                flow_gen.generateFlows(ctx, domain);
            }
            if (!flow_gen.getAllOperations().empty()) {
                prev_op_id = flow_gen.getAllOperations().back().operation_id;
            }
        }
    }

    // 输出结果
    cout << "[OXC] Writing output files..." << endl;
    OxcFlowOutput output(params.output_prefix);

    const auto& all_flows = flow_gen.getAllFlows();
    const auto& all_ops = flow_gen.getAllOperations();

    output.writeFlowMatrices(all_flows);
    output.writeDependencyGraph(all_ops, all_flows);
    output.writeSummary(config, all_ops, all_flows, params.oxc_url, params.oxc_algo);

    cout << endl;
    cout << "SimAI-OXC completed successfully." << endl;
    cout << "  Total Operations: " << all_ops.size() << endl;
    cout << "  Total Flows: " << all_flows.size() << endl;

    return 0;
}
