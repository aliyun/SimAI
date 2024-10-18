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

#include"astra-sim/system/MockNcclLog.h"
#include"astra-sim/system/PhyMultiThread.hh"

#include"SimAiPhyNetwork.h"
#include"SimAiEntry.h"
#include"PhySimAi.h"

extern int local_rank;

SimAiPhyNetWork::SimAiPhyNetWork(int _local_rank)
    : AstraNetworkAPI(_local_rank) {
  this->npu_offset = 0;
}

SimAiPhyNetWork::~SimAiPhyNetWork() {}

AstraSim::timespec_t SimAiPhyNetWork::sim_get_time() {
  AstraSim::timespec_t timeSpec;
  timeSpec.time_val = PhyNetSim::Now();
  return timeSpec;
}

void SimAiPhyNetWork::sim_schedule(
    AstraSim::timespec_t delta,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::DEBUG, "SimAiPhyNetWork::sim_schedule local_rank %d ", local_rank);
  PhyNetSim::Schedule(delta.time_val, fun_ptr, fun_arg);
  return;
}

int SimAiPhyNetWork::sim_send(
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  dst += npu_offset;
  send_flow(rank, dst, count, msg_handler, fun_arg, tag, request);
  return 0;
}

int SimAiPhyNetWork::sim_recv(
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