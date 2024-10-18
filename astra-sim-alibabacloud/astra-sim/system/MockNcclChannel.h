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
#ifndef __MOCKNCCLCHANNEL_HH__
#define __MOCKNCCLCHANNEL_HH__

#include <vector>
#include <map>
#include <memory>
#include "astra-sim/system/Common.hh"
#include "MockNcclGroup.h"

namespace MockNccl {
  struct SingleFlow{
    int flow_id;
    int src;
    int dest;
    uint64_t flow_size;
    std::vector<int>prev;
    std::vector<int> parent_flow_id;
    std::vector<int> child_flow_id;
    int channel_id;
    int chunk_id;
    int chunk_count;
    std::string conn_type;
    SingleFlow(){};
    SingleFlow(
        int _flow_id,
        int _src,
        int _dest,
        uint64_t _flow_size,
        std::vector<int>_prev,
        std::vector<int> _parent_flow_id,
        std::vector<int> _child_flow_id,
        int _channel_id,
        int _chunk_id,
        int _chunk_count,
        std::string _conn_type)
        : flow_id(_flow_id),
          src(_src),
          dest(_dest),
          flow_size(_flow_size),
          prev(_prev),
          parent_flow_id(_parent_flow_id),
          child_flow_id(_child_flow_id),
          channel_id(_channel_id),
          chunk_id(_chunk_id),
          chunk_count(_chunk_count),
          conn_type(_conn_type) {}
    ~SingleFlow(){};
  };

  enum class State{
    Forward_Pass,
    Weight_Gradient,
    Input_Gradient,
  };

  enum class ComType {
    None,
    Reduce_Scatter,
    All_Gatehr,
    All_Reduce,
    All_to_All,
    All_Reduce_All_to_All
  };

  struct ncclTree {
    int depth;
    int rank;
    int up;
    std::vector<int> down;
    ncclTree(){};
    ncclTree(int _depth, int _rank, int _up, std::vector<int> _down)
        : depth(_depth), rank(_rank), up(_up), down(_down) {};
    ~ncclTree(){};
  };

  struct ncclChannelNode{
    int depth;
    int rank;
    ncclChannelNode* up;
    std::vector<ncclChannelNode*> down;
    ncclChannelNode(){};
    ncclChannelNode(int _depth,int _rank,ncclChannelNode* _up,std::vector<ncclChannelNode*>_down):depth(_depth),rank(_rank),up(_up),down(_down){};
    ~ncclChannelNode(){};
  };

  class MockNcclComm{
   public:
    MockNcclComm(int _rank,GroupType _type,MockNcclGroup* _GlobalGroup);
    ~MockNcclComm();

    MockNccl::MockNcclGroup* GlobalGroup;
    GroupType type; 
    int rank;
    std::map<int,std::map<int,std::vector<int>>> ringchannels;
    TreeChannels treechannels; 
    TreeChannels nvlschannels;
    NVLStreechannels nvlstreechannels;

    std::map<int,std::map<int,std::vector<int>>> get_rings();
    MockNccl::TreeChannels get_treechannels();
    MockNccl::TreeChannels get_nvls_channels();
    MockNccl::NVLStreechannels get_nvls_tree_channels();
    std::shared_ptr<void> get_flow_model(uint64_t data_size,AstraSim::ComType collective_type,int layer_num,State loopstate);
    struct ncclInfo* get_algo_proto_info(uint64_t data_size,AstraSim::ComType collective_type);
  };
}

#endif