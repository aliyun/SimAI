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

#include "OxcHttpClient.h"
#include <curl/curl.h>
#include <sstream>
#include <iostream>
#include <cstring>
#include <mutex>

namespace OXC {

// CURL 全局初始化管理器（线程安全，整个程序生命周期只初始化一次）
class CurlGlobalManager {
public:
    static CurlGlobalManager& instance() {
        static CurlGlobalManager instance;
        return instance;
    }

    bool isInitialized() const { return initialized_; }

private:
    CurlGlobalManager() : initialized_(false) {
        CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (res == CURLE_OK) {
            initialized_ = true;
        } else {
            std::cerr << "[OXC] Warning: curl_global_init failed: "
                      << curl_easy_strerror(res) << std::endl;
        }
    }

    ~CurlGlobalManager() {
        if (initialized_) {
            curl_global_cleanup();
        }
    }

    // 禁止拷贝和移动
    CurlGlobalManager(const CurlGlobalManager&) = delete;
    CurlGlobalManager& operator=(const CurlGlobalManager&) = delete;

    bool initialized_;
};

// libcurl 回调函数，用于接收响应数据
static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

OxcHttpClient::OxcHttpClient()
    : timeout_seconds_(30), initialized_(false) {
    // 确保 CURL 全局初始化（线程安全，只执行一次）
    CurlGlobalManager::instance();
}

OxcHttpClient::~OxcHttpClient() {
    // 不再调用 curl_global_cleanup()，由 CurlGlobalManager 在程序退出时处理
}

bool OxcHttpClient::initialize(const std::string& base_url) {
    base_url_ = base_url;
    // 移除末尾的斜杠
    while (!base_url_.empty() && base_url_.back() == '/') {
        base_url_.pop_back();
    }
    initialized_ = true;
    return true;
}

void OxcHttpClient::setTimeout(int timeout_seconds) {
    timeout_seconds_ = timeout_seconds;
}

std::string OxcHttpClient::getLastError() const {
    return last_error_;
}

std::string OxcHttpClient::httpPost(const std::string& url, const std::string& json_body) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        last_error_ = "Failed to initialize CURL";
        return "";
    }

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_body.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds_);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        last_error_ = std::string("CURL error: ") + curl_easy_strerror(res);
        response = "";
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return response;
}

std::string OxcHttpClient::buildRequestJson(const OxcAllReduceRequest& request) {
    std::ostringstream oss;
    oss << "{";

    // ranktable
    oss << "\"ranktable\":{";
    oss << "\"version\":\"" << request.ranktable.version << "\",";
    oss << "\"status\":\"" << request.ranktable.status << "\",";
    oss << "\"rank_count\":" << request.ranktable.rank_count << ",";
    oss << "\"rank_list\":[";

    for (size_t i = 0; i < request.ranktable.rank_list.size(); ++i) {
        const auto& rank = request.ranktable.rank_list[i];
        if (i > 0) oss << ",";
        oss << "{";
        oss << "\"rank_id\":" << rank.rank_id << ",";
        oss << "\"device_id\":" << rank.device_id << ",";
        oss << "\"local_id\":" << rank.local_id << ",";
        oss << "\"level_list\":[";

        for (size_t j = 0; j < rank.level_list.size(); ++j) {
            const auto& level = rank.level_list[j];
            if (j > 0) oss << ",";
            oss << "{";
            oss << "\"net_layer\":" << level.net_layer << ",";
            oss << "\"net_instance_id\":\"" << level.net_instance_id << "\",";
            oss << "\"net_type\":\"" << level.net_type << "\",";
            oss << "\"net_attr\":\"" << level.net_attr << "\",";
            oss << "\"rank_addr_list\":[";

            for (size_t k = 0; k < level.rank_addr_list.size(); ++k) {
                const auto& addr = level.rank_addr_list[k];
                if (k > 0) oss << ",";
                oss << "{";
                oss << "\"addr_type\":\"" << addr.addr_type << "\",";
                oss << "\"addr\":\"" << addr.addr << "\",";
                oss << "\"ports\":[";
                for (size_t p = 0; p < addr.ports.size(); ++p) {
                    if (p > 0) oss << ",";
                    oss << "\"" << addr.ports[p] << "\"";
                }
                oss << "],";
                oss << "\"plane_id\":\"" << addr.plane_id << "\"";
                oss << "}";
            }
            oss << "]";
            oss << "}";
        }
        oss << "]";
        oss << "}";
    }
    oss << "]";
    oss << "},";

    // dpCommDomain
    oss << "\"dpCommDomain\":[";
    for (size_t i = 0; i < request.dpCommDomain.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[";
        for (size_t j = 0; j < request.dpCommDomain[i].size(); ++j) {
            if (j > 0) oss << ",";
            oss << request.dpCommDomain[i][j];
        }
        oss << "]";
    }
    oss << "],";

    // commDomainVolume
    oss << "\"commDomainVolume\":" << request.commDomainVolume << ",";

    // rankIdRackIdMap
    oss << "\"rankIdRackIdMap\":{";
    bool first = true;
    for (const auto& pair : request.rankIdRackIdMap) {
        if (!first) oss << ",";
        first = false;
        oss << "\"" << pair.first << "\":\"" << pair.second << "\"";
    }
    oss << "},";

    // algName
    oss << "\"algName\":\"" << request.algName << "\"";

    oss << "}";
    return oss.str();
}

std::vector<OxcFlowEntry> OxcHttpClient::parseResponseJson(const std::string& response) {
    std::vector<OxcFlowEntry> entries;

    // 简单的JSON数组解析: [[src, dst, step, datasize], ...]
    // 跳过空白和开头的 '['
    size_t pos = 0;
    while (pos < response.size() && (response[pos] == ' ' || response[pos] == '\n' || response[pos] == '\t')) {
        pos++;
    }

    if (pos >= response.size() || response[pos] != '[') {
        last_error_ = "Invalid response format: expected '[' at start";
        return entries;
    }
    pos++;  // 跳过 '['

    while (pos < response.size()) {
        // 跳过空白
        while (pos < response.size() && (response[pos] == ' ' || response[pos] == '\n' || response[pos] == '\t' || response[pos] == ',')) {
            pos++;
        }

        if (pos >= response.size() || response[pos] == ']') {
            break;  // 数组结束
        }

        if (response[pos] != '[') {
            last_error_ = "Invalid response format: expected '[' for inner array";
            return entries;
        }
        pos++;  // 跳过内部数组的 '['

        // 解析四个数字: src, dst, step, datasize
        std::vector<int64_t> values;
        while (pos < response.size() && response[pos] != ']') {
            // 跳过空白和逗号
            while (pos < response.size() && (response[pos] == ' ' || response[pos] == ',' || response[pos] == '\n' || response[pos] == '\t')) {
                pos++;
            }

            if (response[pos] == ']') break;

            // 解析数字
            bool negative = false;
            if (response[pos] == '-') {
                negative = true;
                pos++;
            }

            int64_t num = 0;
            while (pos < response.size() && response[pos] >= '0' && response[pos] <= '9') {
                num = num * 10 + (response[pos] - '0');
                pos++;
            }
            if (negative) num = -num;
            values.push_back(num);
        }

        if (values.size() >= 4) {
            OxcFlowEntry entry;
            entry.src_rank = static_cast<int>(values[0]);
            entry.dst_rank = static_cast<int>(values[1]);
            entry.step = static_cast<int>(values[2]);
            entry.datasize = static_cast<uint64_t>(values[3]);
            entries.push_back(entry);
        }

        // 跳过 ']'
        if (pos < response.size() && response[pos] == ']') {
            pos++;
        }
    }

    return entries;
}

std::vector<OxcFlowEntry> OxcHttpClient::callAllReduceApi(const OxcAllReduceRequest& request) {
    if (!initialized_) {
        last_error_ = "HTTP client not initialized";
        return {};
    }

    std::string url = base_url_ + "/api/oxc/allreduce";
    std::string json_body = buildRequestJson(request);

    std::cout << "[OXC] Calling API: " << url << std::endl;
    // 调试：输出请求体的前500个字符
    std::cout << "[OXC] Request body (first 500 chars): "
              << json_body.substr(0, std::min(static_cast<size_t>(500), json_body.size())) << std::endl;

    std::string response = httpPost(url, json_body);

    if (response.empty()) {
        std::cerr << "[OXC] Empty response from API" << std::endl;
        return {};
    }

    // 打印响应内容
    std::cout << "[OXC] Response (first 1000 chars): "
              << response.substr(0, std::min(static_cast<size_t>(1000), response.size())) << std::endl;

    std::vector<OxcFlowEntry> entries = parseResponseJson(response);

    if (entries.empty() && !response.empty()) {
        // 解析失败，输出调试信息
        std::cerr << "[OXC] Parse failed. Response (first 200 chars): "
                  << response.substr(0, std::min(static_cast<size_t>(200), response.size())) << std::endl;
    }

    return entries;
}

}  // namespace OXC
