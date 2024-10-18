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

#include"AnalyticalNetwork.h"
#include"AnaSim.h"

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
extern map<std::pair<int, std::pair<int, int>>, int> recvHash;
extern map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
extern map<std::pair<int, int>, int64_t> nodeHash;
extern int local_rank;

AnalyticalNetWork::AnalyticalNetWork(int _local_rank)
    : AstraNetworkAPI(_local_rank) {
  this->npu_offset = 0;
}

AnalyticalNetWork::~AnalyticalNetWork() {}

AstraSim::timespec_t AnalyticalNetWork::sim_get_time() {
  AstraSim::timespec_t timeSpec;
  timeSpec.time_val = AnaSim::Now();
  return timeSpec;
}

void AnalyticalNetWork::sim_schedule(
    AstraSim::timespec_t delta,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
  AnaSim::Schedule(delta.time_val, fun_ptr, fun_arg);
  return;
}

int AnalyticalNetWork::sim_send(
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {

  return 0;
}

int AnalyticalNetWork::sim_recv(
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {

  return 0;
}