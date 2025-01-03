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
#ifndef __MOCKNCCLGROUP_H__
#define __MOCKNCCLGROUP_H__

#include<stdlib.h>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include "Common.hh"
#include"MockNccl.h"
using namespace std;

namespace MockNccl {
  enum class State;
  struct SingleFlow;
  struct SingleFlow;
  enum class ComType;
  struct ncclTree;
  struct TuneInfo;
  typedef struct TuneInfo* TuneInfo_t;
  struct ncclChannelNode;
  typedef std::map<std::pair<int,int>,SingleFlow> FlowModels; 
  typedef std::map<int,std::map<int,std::vector<int>>> RingChannels; 
  typedef std::map<int,std::map<int,std::vector<ncclChannelNode*>>> NVLStreechannels;  
  typedef std::map<int,std::map<int,ncclTree>> TreeChannels;
  enum GroupType { TP, DP,DP_EP,EP,NONE };
  struct ncclInfo {
    ncclFunc_t coll;
    TuneInfo_t tuneinfo;
    int algorithm;
    int protocol;
    int nChannels;
    int nThreads;
    size_t nBytes;
    ncclInfo(){};
    ~ncclInfo(){};
  };
  struct TuneInfo{
    int nNodes;
    int nRanks;
    int nChannels;
    int collNetSupport;
    int nvlsSupport;
    int minCompCap;
    int maxCompCap;
    std::vector<ncclTopoGraph*> graphs;
    std::vector<std::vector<std::vector<float>>> latencies;
    std::vector<std::vector<std::vector<float>>> bandwidths;
    TuneInfo(){};
    ~TuneInfo(){};
    TuneInfo(
        int _nNodes,
        int _nRanks,
        int _nChannels,
        int _collNetSupport,
        int _nvlsSupport,
        int _minCompCap,
        int _maxCompCap)
        : nNodes(_nNodes),
          nRanks(_nRanks),
          nChannels(_nChannels),
          collNetSupport(_collNetSupport),
          nvlsSupport(_nvlsSupport),
          minCompCap(_minCompCap),
          maxCompCap(_maxCompCap) {
      graphs = std::vector<ncclTopoGraph*>(NCCL_NUM_ALGORITHMS, nullptr);
      latencies = std::vector<std::vector<std::vector<float>>>(
          NCCL_NUM_FUNCTIONS,
          std::vector<std::vector<float>>(
              NCCL_NUM_ALGORITHMS, std::vector<float>(NCCL_NUM_PROTOCOLS, 0)));
      bandwidths = std::vector<std::vector<std::vector<float>>>(
          NCCL_NUM_FUNCTIONS,
          std::vector<std::vector<float>>(
              NCCL_NUM_ALGORITHMS, std::vector<float>(NCCL_NUM_PROTOCOLS, 0)));
    }
  };
  class GroupInfo {
    public:
    int group_index;
    GroupType type;
    int nNodes;
    int nRanks;
    std::vector<int> Ranks;
    std::vector<int> NVSwitchs; 
    GroupInfo(){}
    GroupInfo(int _group_index, GroupType _type, int _nNodes, int _nRanks, std::vector<int> _Ranks,std::vector<int>_NVSwitchs)
        : group_index(_group_index),type(_type), nNodes(_nNodes), nRanks(_nRanks), Ranks(_Ranks),NVSwitchs(_NVSwitchs) {}
    ~GroupInfo(){}
  };
  class MockNcclGroup {
    struct DoubleBinaryTreeNode {
    int node;
    DoubleBinaryTreeNode* left;
    DoubleBinaryTreeNode* right;
    DoubleBinaryTreeNode(int _node) : node(_node), left(nullptr), right(nullptr) {}
    };
   public:
    MockNcclGroup(){}
    MockNcclGroup(std::vector<int>ngpus,int TP_size,int gpus_per_nodes,std::vector<int> _NVSwitch,GPUType _gpu_type);
    MockNcclGroup(
        std::map<int, GroupInfo> _TPGroups,
        std::map<int, GroupInfo> _DPGroups);
    ~MockNcclGroup(){};

    std::map<int, GroupInfo> TPGroups;
    std::map<int, GroupInfo> DPGroups;
    std::map<int, std::vector<int>> TPlocalrings;
    std::map<int, std::vector<int>> DPlocalrings;
    std::map<int, int> TPrank2group; 
    std::map<int, int> DPrank2group; 

    std::map<int,RingChannels> TPringchannels;
    std::map<int,RingChannels> DPringchannels;

    std::map<int,TreeChannels> TPtreechannels;
    std::map<int,TreeChannels> DPtreechannels;

    std::map<int,NVLStreechannels> TPNVLStreechannels;
    std::map<int,NVLStreechannels> DPNVLStreechannels;

    std::map<int,TreeChannels> DPNVLSchannels;
    std::map<int,TreeChannels> TPNVLSchannels;

    int g_flow_id;
    GPUType gpu_type;
    std::map<std::string,int> FlowName2nums;
    std::map<std::string ,std::map<int,std::shared_ptr<FlowModels> >> flow_models; 
    std::map<std::string ,struct ncclInfo*> nccl_infos;  
    std::shared_ptr<void> getFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size,int layer_num,State loopstate);
   private:
    bool  get_group_info(GroupType type,int rank,GroupInfo & group_info);
    std::map<int,std::shared_ptr<FlowModels>> genFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size);
    std::map<int,std::shared_ptr<FlowModels>>genReduceScatterFlowModels(GroupType type , int rank, uint64_t data_size);
    std::map<int,std::shared_ptr<FlowModels>> genAllReduceFlowModels(GroupType type , int rank,uint64_t data_size);
    std::map<int,std::shared_ptr<FlowModels>> genAllReduceRingFlowModels(GroupType type , int rank,uint64_t data_size);
    std::map<int,std::shared_ptr<FlowModels>> genAllreduceNVLSFlowModels(
        GroupType type,
        int rank,
        uint64_t data_size);
    std::shared_ptr<FlowModels>genallReduceNVLSTreeFlowModels(GroupType type,int rank,uint64_t data_size);
    FlowModels generate_flow_model_nvls_tree_allreduce_up(std::vector<ncclChannelNode*>nvlstreenodes,std::unordered_map<ncclChannelNode*, int> upinDegree,std::unordered_map<ncclChannelNode*,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result);
    FlowModels generate_flow_model_nvls_tree_allreduce_down(std::vector<ncclChannelNode*>nvlstreenodes,std::unordered_map<ncclChannelNode*, int> downinDegree,std::unordered_map<ncclChannelNode*,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result);
    std::shared_ptr<FlowModels> genAllReduceTreeFlowModels(GroupType type , int rank,uint64_t data_size);
    FlowModels generate_flow_model_tree_allreduce_up(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> upinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result);
    FlowModels generate_flow_model_tree_allreduce_down(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> downinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result);
    std::map<int,std::shared_ptr<FlowModels>> genAllGatherFlowModels(GroupType type , int rank,uint64_t data_size);
    std::vector<DoubleBinaryTreeNode*> genInterDouBinTree(GroupInfo pgroupinfo);
    DoubleBinaryTreeNode* InterDouBinTreeShift(DoubleBinaryTreeNode* root,std::vector<int>nodes);
    void ConnInterIntraTree(DoubleBinaryTreeNode*root,std::map<int,std::vector<int>>node2ranks,std::map<int,ncclTree>&TreeChannel);
   public:
    void setlocalrings(
        std::map<int, std::vector<int>> localrings,
        GroupType type);
    void generateringchannels(
        std::map<int, std::vector<int>> localrings,
        MockNccl::GroupInfo* groupInfo,
        std::map<int, std::map<int, std::vector<int>>>& ringchannels);
    std::map<int, std::vector<int>> genlocalrings(int rank, GroupType type);
    RingChannels genringchannels(
        int rank,
        GroupType type);
    TreeChannels gettreechannels(int rank, GroupType type);
    TreeChannels get_nvls_channels(int rank,GroupType type);
    NVLStreechannels get_nvls_tree_channels(int rank,GroupType type);
    ncclChannelNode* gen_nvls_tree_intra_channels(std::vector<int>intra_topo,std::map<int, vector<ncclChannelNode*>> &nvlstreechannel);
    ncclChannelNode* gen_nvls_tree_inter_channels(DoubleBinaryTreeNode* root,std::map<int,ncclChannelNode*> nodencclchannlenodes,std::map<int, vector<ncclChannelNode*>> &nvlstreechannel);
    ncclInfo* get_algo_proto_info(
        GroupType type,
        int rank,
        AstraSim::ComType op,
        uint64_t data_size);
  };
}
#endif