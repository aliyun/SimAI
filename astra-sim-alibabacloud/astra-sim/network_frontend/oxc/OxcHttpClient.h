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

    // 获取最后的错误信息
    std::string getLastError() const;

    // 设置超时时间（秒）
    void setTimeout(int timeout_seconds);

private:
    // 构建请求JSON
    std::string buildRequestJson(const OxcAllReduceRequest& request);

    // 解析响应JSON
    std::vector<OxcFlowEntry> parseResponseJson(const std::string& response);

    // 执行HTTP POST请求
    std::string httpPost(const std::string& url, const std::string& json_body);

    std::string base_url_;
    std::string last_error_;
    int timeout_seconds_;
    bool initialized_;
};

}  // namespace OXC

#endif  // __OXC_HTTP_CLIENT_H__
