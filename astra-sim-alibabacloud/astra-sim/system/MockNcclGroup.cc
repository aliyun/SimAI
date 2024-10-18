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
#include "MockNcclGroup.h"
#include "MockNcclChannel.h"
#include<vector>
#include<map>
#include<set>
#include <queue>
#include<cmath>
#include "astra-sim/system/MockNcclLog.h"
using namespace std;
namespace MockNccl {
  MockNcclGroup::MockNcclGroup(std::vector<int>ngpus,int TP_size,int gpus_per_nodes,std::vector<int> _NVSwitch,GPUType _gpu_type):g_flow_id(0),gpu_type(_gpu_type){
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int nNodes = ngpus.size() > 1 ? ngpus[1] : ngpus[0] / gpus_per_nodes;
    int nlocalranks = ngpus.size() > 1 ? ngpus[0] : gpus_per_nodes;
    int nNodesPerTPGroup;
    int nNodesPerDPGroup;
    int TP_nums = (nNodes * nlocalranks) / TP_size;
    int DP_size;
    int DP_nums;
    std::vector<int> Ranks;
    nNodesPerTPGroup = TP_size / nlocalranks + (TP_size % nlocalranks > 0 ? 1 : 0);
    if(TP_size > 1) {
      int nvswitch_idx=0;

      for(int i =0 ;i < TP_nums; i++) {
        Ranks = {};
        for(int j = 0; j < TP_size; j++) {
          Ranks.push_back(i * TP_size + j);
        }
        std::vector<int>NVSwitch;
        for(int k = 0;k<nNodesPerTPGroup;k++){
          NVSwitch.push_back(_NVSwitch[nvswitch_idx]);
          nvswitch_idx++;
        }
        TPGroups[i] = GroupInfo(i,TP,nNodesPerTPGroup,TP_size,Ranks,NVSwitch);
        for(auto nv:NVSwitch){
          TPrank2group[nv] = i;
        }
      }
    }
    DP_size = TP_nums;
    DP_nums = TP_size;
    nNodesPerDPGroup = nNodes/nNodesPerTPGroup;
    if(DP_size >1) {
      int nvswitch_idx = 0;
      for(int i = 0; i < DP_nums; i++) {
        Ranks = {};
        std::set<int>DPnodes;
        for(int j = 0; j < DP_size; j++) {
          int rank = i + j * DP_nums;
          Ranks.push_back(rank);
          int node_idx = rank / gpus_per_nodes;
          DPnodes.insert(node_idx);
        }
        std::vector<int>NVSwitch;
        for(int idx:DPnodes){
          NVSwitch.push_back(_NVSwitch[idx]);
        }
        DPGroups[i] =  GroupInfo(i,DP,nNodesPerDPGroup,DP_size,Ranks,NVSwitch);
        for(auto nv:NVSwitch){
          DPrank2group[nv]=i;
        }
      }
    }
    std::map<int,GroupInfo>::iterator group_it;
    for(group_it = TPGroups.begin(); group_it != TPGroups.end(); group_it++){
      for(int i : group_it->second.Ranks){
        TPrank2group[i] = group_it->first;
        NcclLog->writeLog(NcclLogLevel::DEBUG," rank %dto TPgroup %d",i,group_it->first);
      }
    }
    for(group_it = DPGroups.begin(); group_it != DPGroups.end(); group_it++){
      for(int i : group_it->second.Ranks){
        DPrank2group[i] = group_it->first;
        NcclLog->writeLog(NcclLogLevel::DEBUG," rank %dto DPgroup %d",i,group_it->first);
      }
    }
  }
  
  MockNcclGroup::MockNcclGroup(std::map<int,GroupInfo> _TPGroups,std::map<int,GroupInfo> _DPGroups)
      :TPGroups(_TPGroups)
      ,DPGroups(_DPGroups)
      ,g_flow_id(0){
    std::map<int,GroupInfo>::iterator group_it;
    for(group_it = TPGroups.begin(); group_it != TPGroups.end(); group_it++){
      for(int i : group_it->second.Ranks)
        TPrank2group[i] = group_it->first;
    }
    for(group_it = DPGroups.begin(); group_it != DPGroups.end(); group_it++){
      for(int i : group_it->second.Ranks)
        DPrank2group[i] = group_it->first;
    }
  }

  bool MockNcclGroup::get_group_info(GroupType type,int rank,GroupInfo & group_info){
    switch (type){
      case TP:
        if(!TPrank2group.count(rank)||!TPGroups.count(TPrank2group[rank])){
          return false;
        } else{
          group_info = TPGroups[TPrank2group[rank]];
          return true;
        }
      case DP:
        if(!DPrank2group.count(rank)||!DPGroups.count(DPrank2group[rank])){
          return false;
        } else{
          group_info = DPGroups[DPrank2group[rank]];
          return true;
        }
      default:
        return false;
    }
    return false;
  }
  
  void MockNcclGroup::generateringchannels(std::map<int, std::vector<int>> localrings, MockNccl::GroupInfo* groupInfo, std::map<int, std::map<int, std::vector<int>>>& ringchannels) {
    std::map<int,std::vector<int>>::iterator ring_it;
    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes = groupInfo->nNodes;
    int nlocalRanks = groupInfo->nRanks/nNodes;
    int delta = nNodes > 1 ? groupInfo->Ranks[nlocalRanks]-groupInfo->Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];

    }
  }

  void MockNcclGroup::setlocalrings(std::map<int,std::vector<int>>localrings,GroupType type){
    switch (type) {
      case TP:
        TPlocalrings = localrings;
        break;
      case DP:
        DPlocalrings = localrings;
        break;
      default:
        break;
    }
  }

  std::map<int,std::vector<int>> MockNcclGroup::genlocalrings(int rank,GroupType type){
    GroupInfo pgroupinfo;
    int groupIndex;
    int delta;
    std::map<int,std::vector<int>>rings;
    std::map<int,std::vector<int>>result;
    switch (type) {
      case TP:
        pgroupinfo = TPGroups[TPrank2group[rank]];
        rings = TPlocalrings;
        delta = pgroupinfo.Ranks[0] - TPGroups[0].Ranks[0];
        groupIndex  = TPrank2group[rank];
        break;
      case DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        rings = DPlocalrings;
        delta = pgroupinfo.Ranks[0] - DPGroups[0].Ranks[0];
        groupIndex = DPrank2group[rank];
        break;
      default:
        break;
    }
    if(groupIndex > 0){
      std::map<int,std::vector<int>>::iterator ring_it;
      for(ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
        for(int i = 0; i < ring_it->second.size(); i++) {
          if(result.count(ring_it->first) > 0)
            result[ring_it->first].push_back(ring_it->second[i] + delta);
          else
            result[ring_it->first] = {ring_it->second[i] + delta};
        }
      }
      return result;
    } else {
      return rings;
    }
  }

  RingChannels MockNcclGroup::genringchannels(int rank, MockNccl::GroupType type) {
    std::map<int,std::map<int,std::vector<int>>>ringchannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo pgroupinfo;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    int current;
    int prev;
    int next;
    int end_rank;
    int nNodes;
    int nlocalRanks;
    int delta;
    switch (type) {
      case TP:
        pgroupinfo = TPGroups[TPrank2group[rank]];
        break;
      case DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        break;
      default:
        break;
    }
    nNodes = pgroupinfo.nNodes;
    nlocalRanks = pgroupinfo.nRanks/nNodes;
    localrings = genlocalrings(rank,type);

    delta = nNodes > 1 ? pgroupinfo.Ranks[nlocalRanks]-pgroupinfo.Ranks[0] : 0;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      prev = -1;
      next = -1;
      for(int i = 0; i < nNodes; i++) {
        int node_send;
        int node_recv;
        node_recv = ring_it->second[0] + i * delta;
        node_send = ring_it->second[nlocalRanks-1] + i * delta;
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;  
          if (j == nlocalRanks-1) {
            next = ring_it->second[0] + (i + 1) * delta;
          } else {
            next = ring_it->second[j+1] + i * delta;
          }
          ringchannels[ring_it->first][current] = {prev,next,node_recv,node_send};
          prev = current;
        }
      }
      end_rank = ring_it->second[nlocalRanks-1] + (nNodes - 1) * delta;
      ringchannels[ring_it->first][ring_it->second[0]][0] = end_rank;
      ringchannels[ring_it->first][end_rank][1] = ring_it->second[0];
    }
    switch (type)
    {
      case TP:
        TPringchannels[TPrank2group[rank]] = ringchannels;
        break;
      case DP:
        DPringchannels[DPrank2group[rank]] = ringchannels;
      default:
        break;
    }
    if (rank == 0 && type ==TP) {
      for (const auto& outer_pair : ringchannels) {
        for (const auto& inner_pair : outer_pair.second) {
          NcclLog->writeLog(NcclLogLevel::DEBUG,"channel_id :  %did:  %d",outer_pair.first,inner_pair.first);
          for (int value : inner_pair.second) {
            NcclLog->writeLog(NcclLogLevel::DEBUG," %d ",value);
          }
        }
      }
    }

    return ringchannels;
  }
  
  std::shared_ptr<void> MockNcclGroup::getFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size,int layer_num,State loopstate){
    std::string flow_model_name;
    GroupInfo pgroupinfo;
    int group_index;
    int end_rank;
    if (type == GroupType::TP) {
      flow_model_name = "TP";
      group_index = TPrank2group[rank];
      pgroupinfo = TPGroups[group_index];
    } else if (type == GroupType::DP) {
      flow_model_name = "DP";
      group_index = DPrank2group[rank];
      pgroupinfo = DPGroups[group_index];
    }
    flow_model_name = flow_model_name + "_" + std::to_string(group_index) + "_" + std::to_string(layer_num) + "_" + std::to_string(static_cast<int>(loopstate)) + "_" + std::to_string(static_cast<int>(op)) + "_" + std::to_string(data_size);
    if(flow_models.count(flow_model_name)){
      FlowName2nums[flow_model_name] ++;
      std::shared_ptr<void> presult;
      if(flow_models[flow_model_name].count(rank)!=0)
        presult = flow_models[flow_model_name][rank];
      else{
        presult = nullptr;
      }
      return presult;
    } else {
      flow_models[flow_model_name] = genFlowModels(type,rank,op,data_size);
      FlowName2nums[flow_model_name]= 1;
      return flow_models[flow_model_name][rank];
    }
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genFlowModels(GroupType type , int rank, AstraSim::ComType op,uint64_t data_size){
    switch (op) {
      case AstraSim::ComType::All_Reduce:
        return genAllReduceFlowModels(type,rank,data_size);
      case AstraSim::ComType::All_Gatehr:
        return genAllGatherFlowModels(type,rank,data_size);
      case AstraSim::ComType::Reduce_Scatter:
        return genReduceScatterFlowModels(type,rank,data_size);
      default:
        break;
    }
    return {};
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genReduceScatterFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size) {
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int, SingleFlow> task_list = {}; 
    std::map<int, SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo pgroupinfo;
    RingChannels ringchannels;
    switch (type) {
      case GroupType::TP:
        /* code */
        pgroupinfo = TPGroups[TPrank2group[rank]];
        ringchannels = TPringchannels[TPrank2group[rank]];
        break;
      case GroupType::DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        ringchannels = DPringchannels[DPrank2group[rank]];
        break;
      default:
        break;
    }
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    nranks = pgroupinfo.nRanks;
    chunkcount = nranks - 1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    for (auto it = ringchannels.begin(); it != ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if (rank_it->second[0] != -1)
              prevranks = {rank_it->second[0]};
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            if (rank_it->first != -1) {
              prevranks = {rank_it->first};
            } else {
              prevranks = {};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              if(rank_it->first!=-1){
                prevranks={rank_it->first};
              }else{
                prevranks ={};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if (prenoderecvrank != -1) {
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id .push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks= {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceFlowModels(GroupType type , int rank,uint64_t data_size){
    ncclInfo* ncc_info = get_algo_proto_info(type,rank,AstraSim::ComType::All_Reduce,data_size);
    switch (ncc_info->algorithm) {
      case NCCL_ALGO_TREE:
      case NCCL_ALGO_RING:
        return genAllReduceRingFlowModels(type, rank, data_size);
      case NCCL_ALGO_NVLS:
        return genAllreduceNVLSFlowModels(type,rank,data_size);
      case NCCL_ALGO_NVLS_TREE:
        return {};
      default:
        break;
    }
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genallReduceNVLSTreeFlowModels(
      GroupType type,
      int rank,
      uint64_t data_size) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    GroupInfo groupinfo;
    int groupindex;
    int chunk_count = 1;
    int chunk_size;
    NVLStreechannels nvlstreechannels;
    NVLStreechannels::iterator nvlstree;
    FlowModels result = {};

    switch (type) {
      case TP:
        groupindex = TPrank2group[rank];
        groupinfo = TPGroups[groupindex];
        nvlstreechannels=TPNVLStreechannels[groupindex];
        break;
      case DP:
        groupindex = DPrank2group[rank];
        groupinfo = DPGroups[groupindex];
        nvlstreechannels=DPNVLStreechannels[groupindex];
        break;
      default:
        break;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG," nvlstreechannels.size()  %d",nvlstreechannels.size());
    chunk_size = data_size / nvlstreechannels.size() / chunk_count;
    for (nvlstree = nvlstreechannels.begin();
         nvlstree != nvlstreechannels.end();
         nvlstree++) {
      std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
      if (rank == 0) {
        for (nvlstreenodes_it = nvlstree->second.begin();
             nvlstreenodes_it != nvlstree->second.end();
             nvlstreenodes_it++) {
          NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
          int i = 0;
          for (auto nvlstreenode : nvlstreenodes_it->second) {
            NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
            if(nvlstreenode->up!=nullptr){
              NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
            }
            NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
            for (auto down : nvlstreenode->down) {
              NcclLog->writeLog(NcclLogLevel::DEBUG,"%d ",down->rank);
            }
          }
        }
      }
      std::unordered_map<ncclChannelNode*, int> upinDegree;
      std::unordered_map<ncclChannelNode*, int> downinDegree;
      std::unordered_map<ncclChannelNode*, std::vector<int>> nodeprevs;
      for (int ck = 0; ck < chunk_count; ck++) {
        nodeprevs = {};
        std::vector<ncclChannelNode*> ncclchannelnodes;
        for (auto nvlstreenodes : nvlstree->second) {
          for (auto nvlstreenode : nvlstreenodes.second) {
            ncclchannelnodes.push_back(nvlstreenode);
            upinDegree[nvlstreenode] = nvlstreenode->down.size();
            if (nvlstreenode->up == nullptr)
              downinDegree[nvlstreenode] = 0;
            else
              downinDegree[nvlstreenode] = 1;
          }
        }
        generate_flow_model_nvls_tree_allreduce_up(
            ncclchannelnodes,
            upinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
        generate_flow_model_nvls_tree_allreduce_down(
            ncclchannelnodes,
            downinDegree,
            nodeprevs,
            chunk_size,
            ck,
            chunk_count,
            nvlstree->first,
            result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result =
        std::make_shared<FlowModels>(result);
    return ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_up(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> upinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
        nodeprevs[entry.first] = {};
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();
      if (current->up != nullptr) {
        upinDegree[current->up]--;
        std::vector<int> _prev;
        if (current->down.size() == 0)
          _prev = {current->up->rank};
        else {
          for (auto down : current->down) {
            _prev.push_back(down->rank);
          }
        }
        tmp_result = SingleFlow(
            g_flow_id,
            current->rank,
            current->up->rank,
            chunk_size,
            _prev,
            nodeprevs[current],
            {},
            channle_id,
            chunk_id,
            chunk_count,
            conn_tag);
        for (int parent_flow_id : nodeprevs[current]) {
          result[std::make_pair(channle_id, parent_flow_id)]
              .child_flow_id.push_back(g_flow_id);
        }
        result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current->up].push_back(tmp_result.flow_id);
        nodeprevs.erase(current);
        if (upinDegree[current->up] == 0)
          q.push(current->up);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_nvls_tree_allreduce_down(
      std::vector<ncclChannelNode*> nvlstreenodes,
      std::unordered_map<ncclChannelNode*, int> downinDegree,
      std::unordered_map<ncclChannelNode*, std::vector<int>>& nodeprevs,
      int chunk_size,
      int chunk_id,
      int chunk_count,
      int channle_id,
      FlowModels& result) {
    std::queue<ncclChannelNode*> q;
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(entry.first);
      }
    }
    std::string conn_tag = "NVLS_TREE";
    while (!q.empty()) {
      ncclChannelNode* current = q.front();
      q.pop();

      if (current->down.size() > 0) {
        for (ncclChannelNode* down : current->down) {
          downinDegree[down]--;
          std::vector<int> _prev;
          if (current->up == nullptr) {
            for (ncclChannelNode* down1 : current->down) {
              _prev.push_back(down1->rank);
            }
          } else {
            _prev = {current->up->rank};
          }
          tmp_result = SingleFlow(
              g_flow_id,
              current->rank,
              down->rank,
              chunk_size,
              _prev,
              nodeprevs[current],
              {},
              channle_id,
              chunk_id,
              chunk_count,
              conn_tag);
          for (int parent_flow_id : nodeprevs[current]) {
            result[std::make_pair(channle_id, parent_flow_id)]
                .child_flow_id.push_back(g_flow_id);
          }
          result[std::make_pair(channle_id, g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id);
          if (downinDegree[down] == 0)
            q.push(down);
        }
      }
    }
    return result;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllreduceNVLSFlowModels(GroupType type,int rank,uint64_t data_size){
    GroupInfo groupinfo;
    int groupindx;
    int chunk_count = 4;
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    switch (type) {
      case TP:
        /* code */
        groupindx = TPrank2group[rank];
        groupinfo = TPGroups[groupindx];
        break;
      case DP:
        groupindx = DPrank2group[rank];
        groupinfo = DPGroups[groupindx];
        break;
      default:
        break;
    }
    FlowModels result={};
    SingleFlow treeflow;
    if(groupinfo.nNodes == 1){  
      std::vector<int>NVswitchs = groupinfo.NVSwitchs;
      std::vector<int>ranks = groupinfo.Ranks;
      int chunk_size = data_size / chunk_count;
      for(int ck =0;ck<chunk_count;ck++){
        for(int j = 0;j<NVswitchs.size();j++){
          std::vector<int>prevs;
          std::vector<int>parents;
          for(int k = 0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,ranks[k],NVswitchs[j],chunk_size,{NVswitchs[j]},{},{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            prevs.push_back(ranks[k]);
            parents.push_back(g_flow_id);
            g_flow_id++;
          }
          for(int k =0;k<ranks.size();k++){
            treeflow = SingleFlow(g_flow_id,NVswitchs[j],ranks[k],chunk_size,prevs,parents,{},0,ck,chunk_count,"NVLS");
            result[std::make_pair(0,g_flow_id)]=treeflow;
            for(auto parent:parents){
              result[std::make_pair(0,parent)].child_flow_id.push_back(g_flow_id);
            }
            g_flow_id++;
          }
        }

      }

    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::shared_ptr<FlowModels> MockNcclGroup::genAllReduceTreeFlowModels(GroupType type , int rank,uint64_t data_size){
    int chunk_count = 64;
    int chunk_size;
    SingleFlow tmp_result;
    FlowModels result1 = {};
    FlowModels result = {};
    std::map<int,int> task_list = {}; 
    std::map<int,std::map<int,ncclTree>>::iterator tree;
    GroupInfo pgroupinfo;
    TreeChannels treechannels;

    switch (type)
    {
      case GroupType::TP:
        pgroupinfo = TPGroups[TPrank2group[rank]];
        treechannels = TPtreechannels[TPrank2group[rank]];
        break;
      case GroupType::DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        treechannels = DPtreechannels[DPrank2group[rank]];
        break;
      default:
        break;
    }
    chunk_size = data_size / treechannels.size() / chunk_count;
    for(tree = treechannels.begin(); tree !=treechannels.end(); tree++) {
      std::unordered_map<int, int> upinDegree;
      std::unordered_map<int, int> downinDegree;
      std::unordered_map<int,std::vector<int>> nodeprevs;
      for(int ck = 0; ck < chunk_count; ck++){
        nodeprevs = {};
        for(auto treenode:tree->second){
          upinDegree[treenode.first] = treenode.second.down.size();
          if(treenode.second.up == -1)
            downinDegree[treenode.first] = 0;
          else
            downinDegree[treenode.first] = 1;
        }
        generate_flow_model_tree_allreduce_up(tree->second,upinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
        generate_flow_model_tree_allreduce_down(tree->second,downinDegree,nodeprevs,chunk_size,ck,chunk_count,tree->first,result);
      }
    }
    std::shared_ptr<FlowModels> ptr_result = std::make_shared<FlowModels>(result);
    return  ptr_result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_up(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> upinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : upinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
        nodeprevs[entry.first]={};
      }
    }
    std::string conn_tag = "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.up != -1) {
        upinDegree[current.up]--; 
        std::vector<int> _prev; 
        if (current.down.size() == 0)
          _prev = {current.up};
        else
          _prev = current.down;
        tmp_result = SingleFlow(g_flow_id,current.rank,current.up,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
        for(int parent_flow_id:nodeprevs[current.rank])
          result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);  
        result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
        g_flow_id++;
        nodeprevs[current.up].push_back(tmp_result.flow_id); 
        nodeprevs.erase(current.rank); 
        if(upinDegree[current.up] == 0)
          q.push(nodes[current.up]);
      }
    }
    return result;
  }

  FlowModels MockNcclGroup::generate_flow_model_tree_allreduce_down(std::map<int,ncclTree> &nodes,std::unordered_map<int, int> downinDegree,std::unordered_map<int,std::vector<int>>& nodeprevs,int chunk_size,int chunk_id,int chunk_count,int channle_id,FlowModels& result){
    std::queue<ncclTree> q;
    std::map<int,int> task_list2={};
    SingleFlow tmp_result;
    for (auto entry : downinDegree) {
      if (entry.second == 0) {
        q.push(nodes[entry.first]);
      }
    }
    std::string conn_tag =  "TREE_INIT";
    while (!q.empty()) {
      ncclTree current = q.front();
      q.pop();
      if(current.down.size() >0 ) {
        for(int down:current.down) {
          downinDegree[down] --;
          std::vector<int> _prev;
          if (current.up == -1) {
            _prev = current.down;
          } else {
            _prev = {current.up};
          }
          tmp_result = SingleFlow(g_flow_id,current.rank,down,chunk_size,_prev,nodeprevs[current.rank],{},channle_id,chunk_id,chunk_count,conn_tag);
          for(int parent_flow_id:nodeprevs[current.rank])
            result[std::make_pair(channle_id,parent_flow_id)].child_flow_id.push_back(g_flow_id);
          result[std::make_pair(channle_id,g_flow_id)] = tmp_result;
          g_flow_id++;
          nodeprevs[down].push_back(tmp_result.flow_id); 
          if(downinDegree[down] == 0)
            q.push(nodes[down]);
        }
      }
    }
    return result;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllReduceRingFlowModels(GroupType type , int rank,uint64_t data_size){
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {}; 
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    
    GroupInfo pgroupinfo;
    RingChannels ringchannels;
    switch (type)
    {
      case GroupType::TP:
        /* code */
        pgroupinfo = TPGroups[TPrank2group[rank]];
        ringchannels = TPringchannels[TPrank2group[rank]];
        break;
      case GroupType::DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        ringchannels = DPringchannels[DPrank2group[rank]];
        break;
      default:
        break;
    }
    nranks = pgroupinfo.nRanks;
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV && strcmp(PXN_ENV, "1") == 0) {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    int chunkcout = 2*(pgroupinfo.nRanks-1);

    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      int chunk_id = 0;
      while (send_size < data_size)
      {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            prevranks = {rank_it->first};          
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if (prenoderecvrank != -1) {
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks={rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunk_id,
                chunkcout,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunk_id++;
        for(int i =0; i < nranks -1; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if (rank_it->second[0] != -1) {
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              prevranks={rank_it->first};
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
            prevranks.clear();
            if(rank_it->second[0]!=-1)
            {
              prevranks ={rank_it->second[0]};
            }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if (rank_it->first != -1) {
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "PXN_INIT");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunk_id,
                  chunkcout,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunk_id++;
        }
        send_size += real_chunksize;
      }
    }
    rank2flowmodels.clear();
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }

  std::map<int,std::shared_ptr<FlowModels>> MockNcclGroup::genAllGatherFlowModels(GroupType type , int rank,uint64_t data_size){
    FlowModels result = {};
    std::map<int,FlowModels>rank2flowmodels;
    std::map<int,std::shared_ptr<FlowModels>>rank2pflowmodels;
    std::map<int,SingleFlow> task_list = {}; 
    std::map<int,SingleFlow> task_list2 = {};
    SingleFlow tmp_result;
    uint64_t chunksize;
    uint64_t send_size;
    int nranks;
    int chunkcount;
    int chunkid;
    GroupInfo pgroupinfo;
    RingChannels ringchannels;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    switch (type)
    {
      case GroupType::TP:
        /* code */
        pgroupinfo = TPGroups[TPrank2group[rank]];
        ringchannels = TPringchannels[TPrank2group[rank]];
        break;
      case GroupType::DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        ringchannels = DPringchannels[DPrank2group[rank]];
        break;
      default:
        break;
    }

    nranks = pgroupinfo.nRanks;
    chunkcount = pgroupinfo.nRanks-1;
    chunksize = data_size / nranks / ringchannels.size();
    data_size = data_size / nranks / ringchannels.size();
    bool PXN_ENABLE = false;
    const char* PXN_ENV = std::getenv("AS_PXN_ENABLE");
    if (PXN_ENV == "1") {
      PXN_ENABLE = true;
    } else {
      PXN_ENABLE = false;
    }
    for(auto it = ringchannels.begin(); it !=ringchannels.end(); it++) {
      auto ring = it->second;
      auto ring_id = it->first;
      task_list = {};
      send_size = 0;
      chunkid = 0;
      while (send_size < data_size) {
        uint64_t real_chunksize = std::min(chunksize, data_size - send_size);
        int prenoderecvrank = ring.rbegin()->second[2];
        int prenodesendrank = ring.rbegin()->second[3];
        int curnoderecvrank = ring.begin()->second[2];
        int curnodesendrank = ring.begin()->second[3];
        std::vector<int> prevranks = {};
        for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
          int cur_rank = rank_it->first;
          if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
          if (rank_it->second[3] == cur_rank &&
              rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) { 
            prevranks.clear();
            if(rank_it->second[0]!=-1){
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[2],
                data_size,
                prevranks,
                {},
                {g_flow_id + 1},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            g_flow_id++;
            prevranks.clear();
            if(rank_it->first!=-1){
              prevranks = {rank_it->first};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->second[2],
                rank_it->second[1],
                data_size,
                prevranks,
                {g_flow_id - 1},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "PXN_INIT");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else if (
              rank_it->second[2] == cur_rank &&
              rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
              PXN_ENABLE) {
            prevranks.clear();
            if(prenoderecvrank!=-1){
              prevranks = {prenoderecvrank};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          } else { 
            prevranks.clear();
            if (rank_it->second[0] != -1) {
              prevranks = {rank_it->second[0]};
            }
            tmp_result = SingleFlow(
                g_flow_id,
                rank_it->first,
                rank_it->second[1],
                data_size,
                prevranks,
                {},
                {},
                ring_id,
                chunkid,
                chunkcount,
                "RING");
            result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
            task_list[rank_it->first] = tmp_result;
            g_flow_id++;
          }
        }
        chunkid++;
        for (int i = 0; i < nranks - 2; i++) {
          task_list2 = {};
          prenoderecvrank = ring.rbegin()->second[2];
          prenodesendrank = ring.rbegin()->second[3];
          curnoderecvrank = ring.begin()->second[2];
          curnodesendrank = ring.begin()->second[3];
          for (auto rank_it = ring.begin(); rank_it != ring.end(); rank_it++) {
            if (curnoderecvrank != rank_it->second[2] &&
              curnodesendrank != rank_it->second[3]) {
            prenoderecvrank = curnoderecvrank;
            prenodesendrank = curnodesendrank;
            curnoderecvrank = rank_it->second[2];
            curnodesendrank = rank_it->second[3];
          }
            int cur_rank = rank_it->first;
            int partner_flow_id = task_list[rank_it->second[0]].flow_id;
            if (rank_it->second[3] == cur_rank &&
                rank_it->second[2] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[2],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {g_flow_id + 1},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
              prevranks.clear();
              if(rank_it->first){
                prevranks = {rank_it->first};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->second[2],
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {g_flow_id - 1},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "PXN");
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else if (
                rank_it->second[2] == cur_rank &&
                rank_it->second[3] != cur_rank && pgroupinfo.nNodes > 1 &&
                PXN_ENABLE) {
              prevranks.clear();
              if(prenoderecvrank!=-1){
                prevranks = {prenoderecvrank};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            } else { 
              prevranks.clear();
              if(rank_it->second[0]!=-1){
                prevranks = {rank_it->second[0]};
              }
              tmp_result = SingleFlow(
                  g_flow_id,
                  rank_it->first,
                  rank_it->second[1],
                  data_size,
                  prevranks,
                  {partner_flow_id},
                  {},
                  ring_id,
                  chunkid,
                  chunkcount,
                  "RING");
              result[std::make_pair(ring_id, partner_flow_id)].child_flow_id.push_back(g_flow_id);
              task_list2[rank_it->first] = tmp_result;
              result[std::make_pair(ring_id, g_flow_id)] = tmp_result;
              g_flow_id++;
            }
          }
          task_list = task_list2;
          chunkid++;
        }
        send_size += real_chunksize;
      }
    }
    for(auto flow_models_it = result.begin();flow_models_it!=result.end();flow_models_it++){
      int src = flow_models_it->second.src;
      int dst = flow_models_it->second.dest;
      rank2flowmodels[src][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
      rank2flowmodels[dst][std::make_pair(flow_models_it->first.first,flow_models_it->first.second)]=flow_models_it->second;
    }
    for(auto it = rank2flowmodels.begin();it!=rank2flowmodels.end();it++){
      rank2pflowmodels[it->first] = std::make_shared<FlowModels>(it->second);
    }
    return rank2pflowmodels;
  }
  ncclChannelNode* MockNcclGroup::gen_nvls_tree_intra_channels(std::vector<int> intra_topo,std::map<int, vector<ncclChannelNode*>> &nvlstreechannel){
    ncclChannelNode* root = new ncclChannelNode(-1,intra_topo[0],nullptr,{});
    nvlstreechannel[root->rank].push_back(root);
    ncclChannelNode* nvswitch = new ncclChannelNode(-1,intra_topo[1],root,{});
    nvlstreechannel[nvswitch->rank].push_back(nvswitch);
    root->down.push_back(nvswitch);
    for(int i =2;i<intra_topo.size();i++){
      ncclChannelNode*leaf = new ncclChannelNode(-1,intra_topo[i],nvswitch,{});
      nvswitch->down.push_back(leaf);
      nvlstreechannel[leaf->rank].push_back(leaf);
    }
    return root;
  }

  TreeChannels MockNcclGroup::get_nvls_channels(int rank,GroupType type){
    GroupInfo pgroupinfo;
    int groupindex;
    TreeChannels nvlschannel;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    switch (type) {
      case TP:
        groupindex = TPrank2group[rank];
        pgroupinfo = TPGroups[groupindex];
        break;
      case DP:
        groupindex = DPrank2group[rank];
        pgroupinfo = DPGroups[groupindex];
      default:
        break;
    }
    if (pgroupinfo.nNodes > 1) {
      NcclLog->writeLog(NcclLogLevel::DEBUG," %d","error NVLS ALGO dont");
      return {};
    } else {
      std::vector<int> ranks = pgroupinfo.Ranks;
      int NVswitch = pgroupinfo.NVSwitchs[0];
      for (int i = 0; i < ranks.size(); i++) {
        nvlschannel[0][ranks[i]] = ncclTree(-1, ranks[i], NVswitch, {});
      }
      nvlschannel[0][ranks.size()] = ncclTree(-1, NVswitch, -1, ranks);
    }
    switch (type) {
      case TP:
        TPNVLSchannels[groupindex] = nvlschannel;
        break;
      case DP:
        DPNVLSchannels[groupindex] = nvlschannel;
      default:
        break;
    }

    return nvlschannel;
  }

  NVLStreechannels MockNcclGroup::get_nvls_tree_channels(int rank,GroupType type){
    std::map<int,std::map<int,std::vector<ncclChannelNode*>>> nvlstreechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo pgroupinfo;
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    switch (type)
    {
      case TP:
        pgroupinfo = TPGroups[TPrank2group[rank]];
        if(TPNVLStreechannels.count(TPrank2group[rank]))  return TPNVLStreechannels[TPrank2group[rank]];
        break;
      case DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        if(DPNVLStreechannels.count(DPrank2group[rank]))  return DPNVLStreechannels[DPrank2group[rank]];
        break;
      default:
        break;
    }

    std::vector<DoubleBinaryTreeNode*>roots;
    roots = genInterDouBinTree(pgroupinfo);

    nNodes = pgroupinfo.nNodes;
    nlocalRanks = pgroupinfo.nRanks/nNodes;
    localrings = genlocalrings(rank,type);
    delta = nNodes > 1 ? pgroupinfo.Ranks[nlocalRanks]-pgroupinfo.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta;
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = pgroupinfo.nRanks / nNodes;
      for (int i = 0; i < pgroupinfo.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }

    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    std::map<int, std::vector<int>> node2ranks = allnode2ranks[0];
    for (DoubleBinaryTreeNode* root : roots) {
      for (int index = 0; index < nlocalRanks; index++) {
        std::map<int, vector<ncclChannelNode*>> nvlstreechannel;
        std::map<int,ncclChannelNode*> nodencclchannlenodes;
        for (int i = 0; i < nNodes; i++) {
          std::vector<int> noderanks = node2ranks[i];
          std::vector<int> intra_topo;
          intra_topo.push_back(noderanks[index]);
          intra_topo.push_back(pgroupinfo.NVSwitchs[i]);
          intra_topo.insert(
              intra_topo.end(), noderanks.begin(), noderanks.end());
          NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d intra_topo",i);
          for(auto num:intra_topo){
            NcclLog->writeLog(NcclLogLevel::DEBUG," %d",num);
          }
          ncclChannelNode* root =
              gen_nvls_tree_intra_channels(intra_topo, nvlstreechannel);
          nodencclchannlenodes[i] = root;
        }

        std::map<int, std::vector<ncclChannelNode*>>::iterator nvlstreenodes_it;
        if (rank == 0) {
          for (nvlstreenodes_it = nvlstreechannel.begin();
               nvlstreenodes_it != nvlstreechannel.end();
               nvlstreenodes_it++) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," rank  %d nvls tree nodes ",nvlstreenodes_it->first);
            int i = 0;
            for (auto nvlstreenode : nvlstreenodes_it->second) {
              NcclLog->writeLog(NcclLogLevel::DEBUG," node  %d rank  %d",i,nvlstreenode->rank);
              if(nvlstreenode->up!=nullptr)
                NcclLog->writeLog(NcclLogLevel::DEBUG," up  %d",nvlstreenode->up->rank);
              NcclLog->writeLog(NcclLogLevel::DEBUG," down ");
              for (auto down : nvlstreenode->down) {
                NcclLog->writeLog(NcclLogLevel::DEBUG," %d ",down->rank);
              }
            }
          }
        }

        gen_nvls_tree_inter_channels(
            root, nodencclchannlenodes, nvlstreechannel);

        nvlstreechannels[channel_id] = nvlstreechannel;
        channel_id++;
      }
    }
    switch (type) {
      case TP:
        TPNVLStreechannels[TPrank2group[rank]] = nvlstreechannels;
        break;
      case DP:
        DPNVLStreechannels[DPrank2group[rank]] = nvlstreechannels;
        break;
      default:
        break;
    }
    return nvlstreechannels;
  }

  ncclChannelNode* MockNcclGroup::gen_nvls_tree_inter_channels(
      DoubleBinaryTreeNode* root,
      std::map<int, ncclChannelNode*> nodencclchannlenodes,
      std::map<int, vector<ncclChannelNode*>>& nvlstreechannel) {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if (root == nullptr)
      return nullptr;
    else {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->right:  %d",root->right);
      NcclLog->writeLog(NcclLogLevel::DEBUG,"before root->left:  %d",root->left);
      if (root->left != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->left:  %d",root->left);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* left = nodencclchannlenodes[root->left->node];
        cur->down.push_back(left);
        left->up = cur;
        gen_nvls_tree_inter_channels(root->left,nodencclchannlenodes,nvlstreechannel);
      }
      if (root->right != nullptr) {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"after root->right:  %d",root->right);
        ncclChannelNode* cur = nodencclchannlenodes[root->node];
        ncclChannelNode* right = nodencclchannlenodes[root->right->node];
        cur->down.push_back(right);
        right->up = cur;
        gen_nvls_tree_inter_channels(root->right,nodencclchannlenodes,nvlstreechannel);
      }
    }
  }

  TreeChannels MockNcclGroup::gettreechannels(int rank, GroupType type){
    TreeChannels treechannels;
    std::map<int,std::vector<int>>localrings;
    std::map<int,std::vector<int>>::iterator ring_it;
    GroupInfo pgroupinfo;
    int current;
    int nNodes;
    int nlocalRanks;
    int delta;
    switch (type) {
      case TP:
        pgroupinfo = TPGroups[TPrank2group[rank]];
        if(TPtreechannels.count(TPrank2group[rank]))  return TPtreechannels[TPrank2group[rank]];
        break;
      case DP:
        pgroupinfo = DPGroups[DPrank2group[rank]];
        if(DPtreechannels.count(DPrank2group[rank]))  return DPtreechannels[DPrank2group[rank]];
        break;
      default:
        break;
    }
  
    nNodes = pgroupinfo.nNodes;
    nlocalRanks = pgroupinfo.nRanks/nNodes;
    localrings = genlocalrings(rank,type);
    delta = nNodes > 1 ? pgroupinfo.Ranks[nlocalRanks]-pgroupinfo.Ranks[0] : 0;
    std::map<int,std::vector<int>>rings;
    for(ring_it = localrings.begin();ring_it != localrings.end();ring_it++) {
      for(int i = 0; i < nNodes; i++) {
        for(int j = 0; j < nlocalRanks; j++) {
          current = ring_it->second[j] + i * delta; 
          rings[ring_it->first].push_back(current);
        }
      }
    }
    std::vector<DoubleBinaryTreeNode*> roots;
    roots = genInterDouBinTree(pgroupinfo);
    std::map<int, std::map<int, std::vector<int>>>
        allnode2ranks; 
    for (ring_it = rings.begin(); ring_it != rings.end(); ring_it++) {
      int nrankspernode = pgroupinfo.nRanks / nNodes;
      for (int i = 0; i < pgroupinfo.nNodes; i++) {
        for (int j = 0; j < nrankspernode; j++) {
          allnode2ranks[ring_it->first][i].push_back(
              ring_it->second[i * nrankspernode + j]);
        }
      }
    }
    std::map<int, std::map<int, std::vector<int>>>::iterator allnode2ranks_it;
    int channel_id = 0;
    for (allnode2ranks_it = allnode2ranks.begin();
         allnode2ranks_it != allnode2ranks.end();
         allnode2ranks_it++) {
      std::map<int, std::vector<int>> node2ranks = allnode2ranks_it->second;
      for (DoubleBinaryTreeNode* root : roots) {
        std::map<int, ncclTree> treechannel;
        for (int rank : pgroupinfo.Ranks) {
          ncclTree cur =  ncclTree(-1, rank, -1, {});
          treechannel[rank] = cur;
        }
        ConnInterIntraTree(root, node2ranks, treechannel);
        treechannels[channel_id] = treechannel;
        channel_id++;
      }
      switch (type) {
        case TP:
          TPtreechannels[TPrank2group[rank]] = treechannels;
          break;
        case DP:
          DPtreechannels[DPrank2group[rank]] = treechannels;
          break;
        default:
          break;
      }
    }
    return treechannels;
  }

  void MockNcclGroup::ConnInterIntraTree(DoubleBinaryTreeNode*root,std::map<int,std::vector<int>>node2ranks,std::map<int,ncclTree>&treechannel) {
    if(root == nullptr) return;
    std::vector<int>ranks = node2ranks[root->node];
    for(int i=0;i<ranks.size()-1;i++) {
      ncclTree *current = &treechannel[ranks[i]];
      ncclTree *down = &treechannel[ranks[i+1]];
      current->down.push_back(ranks[i+1]);
      down->up=ranks[i];
    }

    if(root->left!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->left->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->left,node2ranks,treechannel);
    }
    if(root->right!=nullptr){
      ncclTree *current = &treechannel[ranks[0]];
      int downrank = node2ranks[root->right->node][0];
      ncclTree *down = &treechannel[downrank];
      current->down.push_back(downrank);
      down->up = ranks[0];
      ConnInterIntraTree(root->right,node2ranks,treechannel);
    }
  }

  std::vector<MockNcclGroup::DoubleBinaryTreeNode*> MockNcclGroup::genInterDouBinTree(GroupInfo pgroupinfo){
    vector<DoubleBinaryTreeNode*> q;
    vector<DoubleBinaryTreeNode*> tmp_q;
    vector<DoubleBinaryTreeNode*> result;
    int nNodes = pgroupinfo.nNodes; 
    std::vector<int> nodes;
    for(int i = 0;i < nNodes; i++)
      nodes.push_back(i);
    for(int i = 0;i < nodes.size();i++){
      q.push_back(new DoubleBinaryTreeNode(nodes[i]));
    }
    while (q.size() > 1){
      tmp_q = {};
      int i = 0;
      for(i = 0;(i + 2) < q.size();i +=4){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        DoubleBinaryTreeNode* node2 = q[i+2];
        node1->left = node0;
        node1->right = node2;
        tmp_q.push_back(node1);
        if(i+3 < q.size()) {
          DoubleBinaryTreeNode* node3 = q[i+3];
          tmp_q.push_back((node3));
        }
      }
      if(q.size() - i == 1) {
        DoubleBinaryTreeNode* node0 = q[i];
        tmp_q.push_back(node0);
      } else if(q.size() - i == 2){
        DoubleBinaryTreeNode* node0 = q[i];
        DoubleBinaryTreeNode* node1 = q[i+1];
        node1->left = node0;
        tmp_q.push_back(node1);
      }
      q = tmp_q;
    }
    DoubleBinaryTreeNode* root1 = InterDouBinTreeShift(q[0],nodes);
    int chunk_count = 1;
    for(int i =0;i<chunk_count;i++){
      result.push_back(q[0]);
      result.push_back(root1);
    }
    return result;
  }

  MockNcclGroup::DoubleBinaryTreeNode* MockNcclGroup::InterDouBinTreeShift(DoubleBinaryTreeNode* root,std::vector<int>nodes){
    std::map<int,DoubleBinaryTreeNode*>node2treenode;
    std::map<int,int>rank2index;
    std::queue<DoubleBinaryTreeNode*>q;
    for(int i =0 ;i<nodes.size();i++) {
      node2treenode[nodes[i]] = new DoubleBinaryTreeNode(nodes[i]);
      rank2index[nodes[i]] = i;
    }
    q.push(root);
    while (!q.empty())
    {
      DoubleBinaryTreeNode* current = q.front();
      q.pop();
      int node = current->node;
      int nodeshift = nodes[(rank2index[node] + 1) % nodes.size()];
      DoubleBinaryTreeNode* currentshift = node2treenode[nodeshift];
      if(current->left != nullptr) {
        int leftnode = current->left->node;
        int leftnodeshift = nodes[(rank2index[leftnode] + 1) % nodes.size()];
        currentshift->left = node2treenode[leftnodeshift];
        q.push(current->left);
      }
      if(current->right != nullptr) {
        int rightnode = current->right->node;
        int rightnodeshift = nodes[(rank2index[rightnode] + 1) % nodes.size()];
        currentshift->right = node2treenode[rightnodeshift];
        q.push(current->right);
      }
    }
    return node2treenode[(nodes[rank2index[root->node]+1]) % nodes.size()];
  }

  ncclInfo* MockNcclGroup::get_algo_proto_info(
      GroupType type,
      int rank,
      AstraSim::ComType op,
      uint64_t data_size) {
    std::string ncclInfoName ;
    GroupInfo pgroupinfo;
    switch (type)
    {
    case TP:
      ncclInfoName = "TP";
      pgroupinfo = TPGroups[TPrank2group[rank]];
      break;
    case DP:
      ncclInfoName = "DP";
      pgroupinfo = DPGroups[DPrank2group[rank]];
      break;
    default:
      break;
    }
    ncclInfoName+= "_"+std::to_string(static_cast<int>(op))+"_"+std::to_string(data_size);
    if(nccl_infos.count(ncclInfoName)){ 
      return nccl_infos[ncclInfoName];
    }else{ 
      bool NVLSenable = false;
      const char* NVLSEnv = std::getenv("AS_NVLS_ENABLE");
      if (NVLSEnv && strcmp(NVLSEnv, "1")==0) {
        NVLSenable = true;
      } else {
        NVLSenable = false;
      }
    struct ncclInfo* info = new ncclInfo();
    info->nBytes = data_size;
    info->nChannels = 0;
    info->coll = static_cast<ncclFunc_t>(op);
    switch (op) {
      case AstraSim::ComType::All_Reduce:
          if(type==TP){
            if(gpu_type==GPUType::A100||gpu_type==GPUType::A800){
              info->algorithm = NCCL_ALGO_RING;
            }else if(gpu_type==GPUType::H100||gpu_type==GPUType::H800){
              if (pgroupinfo.nRanks >= 8 && NVLSenable) {
                info->algorithm = NCCL_ALGO_NVLS;
              } else {
                info->algorithm = NCCL_ALGO_RING;
              }
            } else{
              info->algorithm = NCCL_ALGO_RING;
            }
          } else {
            info->algorithm = NCCL_ALGO_RING;
          }
          break;
      case AstraSim::ComType::All_Gatehr:
      case AstraSim::ComType::Reduce_Scatter:
      default:
          info->algorithm = NCCL_ALGO_RING;
          break;
    }
    info->protocol = NCCL_PROTO_UNDEF;
    nccl_infos[ncclInfoName] = info;
    return info;
    }
  }
}