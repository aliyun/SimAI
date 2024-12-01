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

#ifndef __NCCL_TREE_FLOW_MODEL_HH__
#define __NCCL_TREE_FLOW_MODEL_HH__

#include <assert.h>
#include <math.h>
#include<set>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>
#include<condition_variable>
#include "Algorithm.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/topology/RingTopology.hh"
#include  "astra-sim/system/MockNcclQps.h"

namespace AstraSim {
class NcclTreeFlowModel : public Algorithm {
 public:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
  MemBus::Transmition transmition;
  int id;
  int nodes_in_ring;
  std::map<int, int> _stream_count; 
  std::atomic<int> send_packets;
  std::atomic<int> recv_packets;
  int parallel_reduce;
  std::map<std::pair<int, int>, std::list<MyPacket>> packets; 
  bool toggle;
  std::map<std::pair<int,int>, int> free_packets; 
  bool processed;   
  bool send_back;
  bool NPU_to_MA;

  std::map<int, int> indegree_mapping; 
  std::map<int, int> inprocessing_indegree; 
  std::map<int, int>* zero_latency_packets;
  std::map<int, int>* non_zero_latency_packets;
  MockNccl::FlowModels _flow_models; 
  uint32_t m_channels;
  uint32_t len_channel;
  MockNccl::NcclQps* pQps;
  std::condition_variable judge_exit_cv;
  std::mutex judge_exit_mutex;
  std::mutex judge_mutex;
  std::atomic<bool> judge_exit_flag;
  NcclTreeFlowModel(){};
  ~NcclTreeFlowModel(){};

  NcclTreeFlowModel(
      ComType type,
      int id,
      int layer_num,
      RingTopology* ring_topology,
      uint64_t data_size,
      RingTopology::Direction direction,
      InjectionPolicy injection_policy,
      bool boost_mode,
      std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
      int treechannels);
  virtual void run(EventType event, CallData* data);
  void process_stream_count(int channel_id);
  void release_packets(int channel_id, int flow_id, uint64_t message_size);
  void reduce(int channel_id, int flow_id);
  bool iteratable(int channel_id);
  virtual int get_non_zero_latency_packets();
  void insert_packets(int channel_id, int flow_id);
  void init_indegree_mapping();
  bool ready(int channel_id, int flow_id);
  bool recv_ready(int channel_id, int flow_id);
  bool init_recv_ready();
  void exit();
  #ifdef PHY_MTP
  bool phy_iteratable(int channel_id);
  bool phy_ready(int channel_id,int flow_id);
  void waiting_to_exit();
  #endif
  class FlowCriticalSection
  {
  public:
    inline FlowCriticalSection ()
    {
      while (g_flow_inCriticalSection.exchange (true, std::memory_order_acquire))
        ;
    }

    inline void ExitSection() 
    {
        g_flow_inCriticalSection.store (false, std::memory_order_release);
    }

    inline ~FlowCriticalSection ()
    {
      g_flow_inCriticalSection.store (false, std::memory_order_release);
    }
  };
  static std::atomic<bool> g_flow_inCriticalSection;

};
} // namespace AstraSim
#endif
