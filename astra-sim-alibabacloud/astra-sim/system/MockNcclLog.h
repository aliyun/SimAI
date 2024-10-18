/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
#ifndef _MOCKNCCL_MOCKNCCLLOG_H_
#define _MOCKNCCL_MOCKNCCLLOG_H_
#include <iostream>
#include<sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <cstdarg>
#include <thread>

#define LOG_PATH  "/etc/astra-sim/"

enum class NcclLogLevel { DEBUG, INFO, WARNING,ERROR};

class MockNcclLog {
 private:
  static MockNcclLog* instance;
  static NcclLogLevel logLevel;
  static std::mutex mtx;
  static std::string LogName;
  std::ofstream logfile;
  MockNcclLog() {
    const char* logLevelEnv = std::getenv("AS_LOG_LEVEL");
    logLevel = logLevelEnv ? static_cast<NcclLogLevel>(std::atoi(logLevelEnv))
                           : NcclLogLevel::INFO;
    logfile.open(LogName, std::ios::app);
  }
  std::string getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
    localtime_r(&now_c, &timeinfo);

    std::ostringstream oss;
    char buffer[80];
    std::strftime(buffer, 80, "%Y-%m-%d %X", &timeinfo);
    oss << buffer;
    return oss.str();
  }
 public:
  static MockNcclLog* getInstance() {
    std::lock_guard<std::mutex> lock(mtx);
    if (instance == nullptr) {
      instance = new MockNcclLog();
    }
    return instance;
  }
  static void set_log_name(std::string log_name){
    LogName = LOG_PATH + log_name;
  }
  void writeLog(NcclLogLevel level, const char* format,...) {
    if (level >= logLevel) {
      std::string levelStr;
      switch (level) {
        case NcclLogLevel::DEBUG:
          levelStr = "DEBUG";
          break;
        case NcclLogLevel::INFO:
          levelStr = "INFO";
          break;
        case NcclLogLevel::WARNING:
          levelStr = "WARNING";
          break;
        case NcclLogLevel::ERROR:
          levelStr = "ERROR";
          break;
        default:
          levelStr = "UNKNOWN";
      }
      char buffer[256];
      va_list args;
      va_start(args, format);
      vsnprintf(buffer, sizeof(buffer), format, args);
      va_end(args);
      std::thread::id this_id = std::this_thread::get_id();
      std::lock_guard<std::mutex> lock(mtx);
      logfile << "[" << getCurrentTime() << "]"
              << "[" << levelStr << "] " << "["<< std::hex << this_id <<"]"<< buffer << std::endl;
    }
  }
  ~MockNcclLog() {
        logfile.close();  
    }
};
#endif 
