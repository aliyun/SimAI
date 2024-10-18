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

#include "MockNcclChannel.h"
#include <cmath>

namespace MockNccl {
  MockNcclComm::MockNcclComm(int _rank,GroupType _type,MockNcclGroup* _GlobalGroup) :rank(_rank),type(_type),GlobalGroup(_GlobalGroup){
    this->ringchannels = this->GlobalGroup->genringchannels(rank,type);
    this->treechannels = this->GlobalGroup->gettreechannels(rank,type);
    this->nvlschannels = this->GlobalGroup->get_nvls_channels(rank,type);
    // this->nvlstreechannels = this->GlobalGroup->get_nvls_tree_channels(rank,type);
  }

  MockNcclComm::~MockNcclComm(){};

  std::map<int,std::map<int,std::vector<int>>> MockNcclComm::get_rings() {
    std::map<int,std::map<int,std::vector<int>>> result;
    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      for(auto rank_it = ring.begin();rank_it != ring.end(); rank_it++) {
        result[rank_it->first][ring_id]= rank_it->second;
      }
    }
    return result;
  }

  MockNccl::TreeChannels MockNcclComm::get_treechannels(){
    TreeChannels nvlschannel ;
    nvlschannel[0][0]=ncclTree(-1,0,8,{});
    nvlschannel[0][1]=ncclTree(-1,1,8,{});
    nvlschannel[0][2]=ncclTree(-1,2,8,{});
    nvlschannel[0][3]=ncclTree(-1,3,8,{});
    nvlschannel[0][4]=ncclTree(-1,4,8,{});
    nvlschannel[0][5]=ncclTree(-1,5,8,{});
    nvlschannel[0][6]=ncclTree(-1,6,8,{});
    nvlschannel[0][7]=ncclTree(-1,7,8,{});
    nvlschannel[0][8]=ncclTree(-1,8,-1,{0,1,2,3,4,5,6,7});
    return nvlschannel;
  }

  MockNccl::TreeChannels MockNcclComm::get_nvls_channels(){
    return this->nvlschannels;
  }

  MockNccl::NVLStreechannels MockNcclComm::get_nvls_tree_channels(){
    return this->nvlstreechannels;
  }

  std::shared_ptr<void> MockNcclComm::get_flow_model(uint64_t data_size,AstraSim::ComType collective_type,int layer_num,State loopstate) {
    return this->GlobalGroup->getFlowModels(type,rank,collective_type,data_size,layer_num,loopstate);
  }

  struct ncclInfo* MockNcclComm::get_algo_proto_info(uint64_t data_size,AstraSim::ComType collective_type){
    return this->GlobalGroup->get_algo_proto_info(type,rank,collective_type,data_size);
  }
}