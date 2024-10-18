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

#ifndef __SIMAI_PHYNET_ENTRY_HH__
#define __SIMAI_PHYNET_ENTRY_HH__

#undef PGO_TRAINING
#include <fstream>
#include <iostream>
#include <time.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include"astra-sim/system/RecvPacketEventHadndlerData.hh"
#include"astra-sim/system/AstraNetworkAPI.hh"
#include"astra-sim/system/MockNcclQps.h"
#include"astra-sim/system/SimAiPhyCommon.hh"

using namespace std;

struct task1 {
  int src;
  int dest;
  int type;
  uint64_t count;
  void *fun_arg;
  void (*msg_handler)(void *fun_arg);
  double schTime;
};

void set_simai_network_callback();
void send_flow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void *fun_arg), void *fun_arg, int tag, AstraSim::sim_request *request);
#endif