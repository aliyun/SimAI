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

#ifndef __OXC_HTTP_CLIENT_H__
#define __OXC_HTTP_CLIENT_H__

#include <string>
#include <vector>
#include "astra-sim/system/OxcTypes.h"

namespace OXC {

class OxcHttpClient {
public:
    OxcHttpClient();
    ~OxcHttpClient();

    // 初始化，设置服务器URL
    bool initialize(const std::string& base_url);

    // 调用 OXC AllReduce API
    // 成功返回流条目列表，失败返回空列表
    std::vector<OxcFlowEntry> callAllReduceApi(const OxcAllReduceRequest& request);

    // ============================================================
    // 预留接口：以下 API 为未来 OXC 集合通信类型预留
    // Reserved APIs for future OXC collective communication types
    // ============================================================

    // 调用 OXC AllGather API（预留）
    // TODO: 待 OXC 后端实现后完善
    // 预期 endpoint: POST /api/oxc/allgather
    std::vector<OxcFlowEntry> callAllGatherApi(const OxcAllGatherRequest& request);

    // 调用 OXC ReduceScatter API（预留）
    // TODO: 待 OXC 后端实现后完善
    // 预期 endpoint: POST /api/oxc/reducescatter
    std::vector<OxcFlowEntry> callReduceScatterApi(const OxcReduceScatterRequest& request);

    // 调用 OXC AllToAll API（预留）
    // TODO: 待 OXC 后端实现后完善
    // 预期 endpoint: POST /api/oxc/alltoall
    std::vector<OxcFlowEntry> callAllToAllApi(const OxcAllToAllRequest& request);

    // 检查 OXC 后端是否支持指定的集合通信类型
    // 可用于运行时检测后端能力
    bool isApiSupported(const std::string& api_name);

    // 获取最后的错误信息
    std::string getLastError() const;

    // 设置超时时间（秒）
    void setTimeout(int timeout_seconds);

    // 设置连接超时时间（秒）
    void setConnectTimeout(int connect_timeout_seconds);

private:
    // 构建请求JSON
    std::string buildRequestJson(const OxcAllReduceRequest& request);

    // 构建 AllGather 请求 JSON（预留）
    std::string buildAllGatherRequestJson(const OxcAllGatherRequest& request);

    // 构建 ReduceScatter 请求 JSON（预留）
    std::string buildReduceScatterRequestJson(const OxcReduceScatterRequest& request);

    // 构建 AllToAll 请求 JSON（预留）
    std::string buildAllToAllRequestJson(const OxcAllToAllRequest& request);

    // 解析响应JSON
    std::vector<OxcFlowEntry> parseResponseJson(const std::string& response);

    // 执行HTTP POST请求
    std::string httpPost(const std::string& url, const std::string& json_body);

    std::string base_url_;
    std::string last_error_;
    int timeout_seconds_;
    int connect_timeout_seconds_;
    bool initialized_;
};

}  // namespace OXC

#endif  // __OXC_HTTP_CLIENT_H__
