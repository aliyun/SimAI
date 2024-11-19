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

#ifdef PHY_MTP
#include<mpi.h>
#include "astra-sim/system/PhyMultiThread.hh"
#endif
#include<chrono>

#include "NcclTreeFlowModel.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/MockNcclLog.h"
#ifdef PHY_RDMA
#include "astra-sim/system/SimAiFlowModelRdma.hh"
extern FlowPhyRdma flow_rdma; 
#endif


namespace AstraSim {
std::atomic<bool> NcclTreeFlowModel::g_flow_inCriticalSection(false);
NcclTreeFlowModel::NcclTreeFlowModel(
    ComType type,
    int id,
    int layer_num,
    RingTopology* ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode,
    std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
    int treechannels)
    : Algorithm(layer_num){
  this->start_time = std::chrono::high_resolution_clock::now();
  this->end_time = std::chrono::high_resolution_clock::now();
  this->comType = type;
  this->id = id;
  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();
  this->parallel_reduce = 1;
  this->toggle = false;
  this->name = Name::Ring;
  this->enabled = true;
  this->m_channels = treechannels;
  this->judge_exit_flag.store(false);
  this->judge_exit_mutex.unlock();
  this->judge_mutex.unlock();
  this->send_packets = 0;
  this->recv_packets = 0;
  pQps = new MockNccl::NcclQps();
  zero_latency_packets = new std::map<int, int>();
  non_zero_latency_packets = new std::map<int, int>();
  if (boost_mode) {
    this->enabled = ring_topology->is_enabled();
  }
  if (ring_topology->dimension == RingTopology::Dimension::Local) {
    transmition = MemBus::Transmition::Fast;
  } else {
    transmition = MemBus::Transmition::Usual;
  }
  if(ptr_flow_models){
    if(id == 0) 
    {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
    }
    for(auto f : *ptr_flow_models) {
      if(f.second.dest == id) {
          this->free_packets[std::make_pair(f.second.channel_id,f.second.src)]++;
          this->_flow_models[f.first] = f.second;
          recv_packets++;
        }
      if(f.second.src == id) {
        if(pQps->peer_qps.count(std::make_pair(f.second.channel_id,std::make_pair(f.second.src,f.second.dest)))==0){
          pQps->peer_qps[std::make_pair(f.second.channel_id,std::make_pair(f.second.src,f.second.dest))]=1;
        }
        NcclTreeFlowModel::FlowCriticalSection cs;
        this->_stream_count[f.second.channel_id] += 1;
        cs.ExitSection();
        assert(this->_flow_models.count(f.first) == 0);
        this->_flow_models[f.first] = f.second;
        send_packets++;
      }
    }
  }
  for(int channel_id = 0 ;channel_id<m_channels;channel_id++){
    assert(zero_latency_packets->find(channel_id) == zero_latency_packets->end());
    (*zero_latency_packets)[channel_id] = 0;
    assert(non_zero_latency_packets->find(channel_id) == non_zero_latency_packets->end());
    (*non_zero_latency_packets)[channel_id] = 0;
  }
  init_indegree_mapping();
  switch (type) {
    case ComType::All_Reduce:
      this->final_data_size = data_size;
      break;
    case ComType::All_Gather:
      this->final_data_size = data_size * nodes_in_ring;
      break;
    case ComType::Reduce_Scatter:
      this->final_data_size = data_size / nodes_in_ring;
      break;
    case ComType::All_to_All:
      this->final_data_size = data_size;
      break;
    default:;
  }
}

void NcclTreeFlowModel::init_indegree_mapping(){
  MockNccl::FlowModels::iterator tree_it;
  for(tree_it = _flow_models.begin();tree_it != _flow_models.end();tree_it++) {
    if(tree_it->second.src!=id) continue;
    indegree_mapping[tree_it->first.second] = tree_it->second.parent_flow_id.size();
  }
}

int NcclTreeFlowModel::get_non_zero_latency_packets() {
  return (nodes_in_ring - 1) * parallel_reduce * 1;
}

void NcclTreeFlowModel::run(EventType event, CallData* data) {
  BasicEventHandlerData* ehd = (BasicEventHandlerData*)data;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (event == EventType::General) {
    int channel_id = ehd->channel_id;
    int flow_id = ehd->flow_id;
    #ifndef PHY_MTP
    ready(channel_id, flow_id);
    #else
    phy_ready(channel_id, flow_id);
    #endif
  } else if (event == EventType::PacketReceived) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    RecvPacketEventHadndlerData* rcehd = (RecvPacketEventHadndlerData*)ehd;
    AstraSim::ncclFlowTag flowTag = rcehd->flowTag;
    int received_flow_id = flowTag.current_flow_id;
    int channel_id = flowTag.channel_id;
    std::vector<int> next_flow_list = flowTag.tree_flow_list;    
    #ifdef PHY_MTP
    recv_packets--;
    if(!phy_iteratable(channel_id)){
      return;
    }
    #else 
    bool flow_exist = next_flow_list.size() == 0 ? true : false;
    for(int i = 0; i < next_flow_list.size(); ++ i) {
      int next_flow_id = next_flow_list[i];
      if(next_flow_id == -1 || _flow_models.count(std::make_pair(channel_id, next_flow_id)) != 0) flow_exist = true;
      else {
        flow_exist = false;
        break;
      }
    }
    assert(flow_exist == true);
    NcclTreeFlowModel::FlowCriticalSection cs;
    free_packets[std::make_pair(channel_id, flowTag.sender_node)]--;
    bool tag = true;
    for (int i = 0; i < m_channels; i++) {
      if (_stream_count[i] != 0) {
        tag = false;
        break;
      }
    }
    cs.ExitSection();
    if(tag) { 
      ready(channel_id, -1);
      iteratable(channel_id);
      return;
    } 
    #endif
    NcclLog->writeLog(NcclLogLevel::DEBUG,"PacketReceived sender_node:  %d recevier  %d current_flow id:  %d channel_id:  %d tag_id  %d free_packets  %d next_flow_list.size %d",flowTag.sender_node,flowTag.receiver_node,flowTag.current_flow_id,flowTag.channel_id,flowTag.tag_id,free_packets[std::make_pair(channel_id,flowTag.sender_node)],next_flow_list.size());
    #ifdef PHY_MTP
    for (int next_flow_id : next_flow_list){
      if (--indegree_mapping[next_flow_id] == 0) { 
        phy_ready(channel_id, next_flow_id);
      }
    }
    #else
    flow_exist = true;
    bool flow_send = false;
    bool recv_finished_tag = true;
    for (auto it = free_packets.begin(); it != free_packets.end(); it++) {
      if (it->second != 0) {
        recv_finished_tag = false;
        break;
      }
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG,"next_flow_list.size %d",next_flow_list.size());
    for (int next_flow_id : next_flow_list) {
      NcclTreeFlowModel::FlowCriticalSection cs;
      if (indegree_mapping.count(next_flow_id) == 0) {
        flow_exist = false;
        cs.ExitSection();
        break;
      }
      if (--indegree_mapping[next_flow_id] == 0) {
        MockNccl::SingleFlow cur_flow = _flow_models[std::make_pair(channel_id, next_flow_id)];
          cs.ExitSection();
          insert_packets(channel_id, next_flow_id);
      }else{
        cs.ExitSection();
      }
    }
    assert(flow_exist = true);
    #endif
  } else if (event == EventType::StreamInit) {
    #ifdef PHY_MTP
    MPI_Barrier(MPI_COMM_WORLD);
    for(auto single_flow: _flow_models){
      if((single_flow.second.src==id||single_flow.second.dest==id)){ 
        #ifdef PHY_RDMA
        flow_rdma.ibv_create_peer_qp(id,single_flow.second.channel_id,single_flow.second.src,single_flow.second.dest,single_flow.second.chunk_count + 1 ,single_flow.second.chunk_id,single_flow.second.flow_size);
        #endif
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"streamInit time %lld",now_us);
    start_time = std::chrono::high_resolution_clock::now();
    #endif
    for (int i = 0; i < parallel_reduce; i++) {
      #ifndef PHY_MTP
      init_recv_ready();
      #endif
      for(int j = 0; j < m_channels; j ++) {
        for(const auto flow_model : _flow_models) {
          if(flow_model.second.src!=id)continue;
          std::vector<int> parent_list = flow_model.second.parent_flow_id;
          if((parent_list.size() == 0 ) && flow_model.second.channel_id == j ) {
            #ifdef PHY_MTP
            if(flow_model.second.chunk_id == 0){
              phy_ready(j, flow_model.second.flow_id);
            }
            #else
            if (flow_model.second.chunk_id == 0) {
              pQps->peer_qps[std::make_pair(
                  flow_model.second.channel_id,
                  std::make_pair(
                      flow_model.second.src, flow_model.second.dest))] = 0;
              insert_packets(j,flow_model.second.flow_id);
            } else {
              pQps->peer_wating_tasks[std::make_pair(
                      flow_model.second.channel_id,
                      std::make_pair(
                          flow_model.second.src, flow_model.second.dest))]
                  .push(flow_model.second.flow_id);
            }
            #endif
          }
        }
      }
      #ifdef PHY_MTP
      waiting_to_exit();
      NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclTreeFlowModel::waiting_to_exit end ");
      #endif
    }
  } else if(event == EventType::PacketSentFinshed){
    SendPacketEventHandlerData* rcehd = (SendPacketEventHandlerData*)ehd;
    AstraSim::ncclFlowTag flowTag = rcehd->flowTag;
    int sent_flow_id = flowTag.current_flow_id;
    int channel_id = flowTag.channel_id;
    std::vector<int> next_flow_list = flowTag.tree_flow_list;   
    NcclLog->writeLog(NcclLogLevel::DEBUG,"PacketSentFinshed src %d dst %d channel_id %d flow_id %d",flowTag.sender_node,flowTag.receiver_node,flowTag.channel_id,flowTag.current_flow_id);
    reduce(channel_id,sent_flow_id);
    bool flow_exist = next_flow_list.size() == 0 ? true : false;
    #ifndef PHY_MTP
    NcclTreeFlowModel::FlowCriticalSection cs;
    pQps->peer_qps[std::make_pair(flowTag.channel_id,std::make_pair(flowTag.sender_node,flowTag.receiver_node))]=1;
    cs.ExitSection();
    if(pQps->peer_wating_tasks[std::make_pair(flowTag.channel_id,std::make_pair(flowTag.sender_node,flowTag.receiver_node))].size()>0){
      int cur_flow_id = pQps->peer_wating_tasks[std::make_pair(flowTag.channel_id,std::make_pair(flowTag.sender_node,flowTag.receiver_node))].front();
      pQps->peer_wating_tasks[std::make_pair(flowTag.channel_id,std::make_pair(flowTag.sender_node,flowTag.receiver_node))].pop();
      pQps->peer_qps[std::make_pair(flowTag.channel_id,std::make_pair(flowTag.sender_node,flowTag.receiver_node))]=0;
      insert_packets(channel_id,cur_flow_id);
    }
    iteratable(channel_id); 
    #else
    phy_iteratable(channel_id);
    #endif
  }
}

bool NcclTreeFlowModel::init_recv_ready() {
  std::map<std::pair<int,std::vector<int>>,std::vector<int>> recv_ready_flows; 
  for(auto flow : _flow_models){
    if(flow.second.src!=id)  continue;
    if(flow.second.chunk_id!=0)continue; 
    if (flow.second.parent_flow_id.size() == 0)
      continue;
    std::pair<int, std::vector<int>> cur =
        std::make_pair(flow.second.channel_id, flow.second.prev);
    if (recv_ready_flows.count(cur) == 0) {
      recv_ready_flows[cur] = {flow.second.flow_id};
    } else { 
      std::vector<int> flow_ids = recv_ready_flows[cur];
      bool flag = true;
      for (int flow_id : flow_ids) {
        MockNccl::SingleFlow old_flow =
            _flow_models[std::make_pair(flow.second.channel_id, flow_id)];
        if (old_flow.parent_flow_id == flow.second.parent_flow_id) {
          flag = false;
          break;
        }
      }
      if (flag) {
        recv_ready_flows[cur].push_back(flow.second.flow_id);
      }
    }
  }
  std::map<std::pair<int,std::vector<int>>,std::vector<int>>::iterator recv_ready_flow_it;
    for(recv_ready_flow_it = recv_ready_flows.begin();recv_ready_flow_it!=recv_ready_flows.end();recv_ready_flow_it++){
      for(int flow_id: recv_ready_flow_it->second){
      recv_ready(recv_ready_flow_it->first.first,flow_id);
      }
    }
  return true;
}

bool NcclTreeFlowModel::recv_ready(int channel_id, int flow_id) {
  std::vector<int>recv_prevs;
  auto flow_model = _flow_models[std::make_pair(channel_id,flow_id)];
  recv_prevs = flow_model.prev;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();

  for (int recv_prev : recv_prevs) {
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;

    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        recv_prev,
        1); 
    ehd->flowTag.child_flow_id = -1;
    ehd->flowTag.current_flow_id = -1;
    ehd->flowTag.channel_id = channel_id;
    ehd->flowTag.tag_id =layer_num*flow_model.chunk_count*m_channels+ flow_model.chunk_count*flow_model.channel_id;
      stream->owner->front_end_sim_recv(
          0,
          Sys::dummy_data,
          _flow_models[std::make_pair(channel_id, flow_id)].flow_size,
          UINT8,
          recv_prev,
          channel_id,
          &rcv_req,
          &Sys::handleEvent,
          ehd);
  }
  return true;
}

void NcclTreeFlowModel::release_packets(int channel_id, int flow_id, uint64_t message_size) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }
  if (NPU_to_MA == true) {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         message_size,
         transmition,
         channel_id,
         flow_id))
        ->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         message_size,
         transmition,
         channel_id,
         flow_id))
        ->send_to_NPU();
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"id:  %d finish release_packets",id);
  locked_packets.clear();
}

void NcclTreeFlowModel::process_stream_count(int channel_id) {
  MockNcclLog*NcclLog = MockNcclLog::getInstance();
  #ifdef PHY_MTP
    send_packets--;
  #else
  NcclTreeFlowModel::FlowCriticalSection cs;
  if (_stream_count[channel_id] > 0) {
    _stream_count[channel_id]--;
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"NcclTreeFlowModel::process_stream_count channel_id %d _stream_count %d",channel_id,_stream_count[channel_id]);
  if (_stream_count[channel_id] == 0 && stream->state != StreamState::Dead) 
    stream->changeState(StreamState::Zombie);
  cs.ExitSection();
  #endif
}

void NcclTreeFlowModel::reduce(int channel_id, int flow_id) {
  process_stream_count(channel_id);
  #ifndef PHY_MTP
  if(!packets[std::make_pair(channel_id, flow_id)].empty()){
    packets[std::make_pair(channel_id, flow_id)].pop_front();
  }
  #endif
}

bool NcclTreeFlowModel::iteratable(int channel_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  bool all_channel_finished = true, all_packets_freed = true;
  NcclTreeFlowModel::FlowCriticalSection cs;
  for(int i = 0; i < m_channels; ++ i) {
    if(_stream_count.count(i) != 0 && _stream_count[i] != 0) all_channel_finished = false;
  }
  for (auto it = free_packets.begin(); it != free_packets.end(); it++) {
    if (it->second != 0) {
      all_packets_freed = false;
      break;
    }
  }
  cs.ExitSection();
  if (all_channel_finished == true &&
      all_packets_freed == true) {
    exit();
    return false;
  }
  return true;
}

void NcclTreeFlowModel::insert_packets(int channel_id, int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  assert(channel_id < m_channels);
  if (!enabled) {
    return;
  }
  assert(_flow_models.count(std::make_pair(channel_id, flow_id)) != 0);

  MockNccl::SingleFlow f = _flow_models[std::make_pair(channel_id, flow_id)];
  assert(zero_latency_packets->count(channel_id) != 0 && non_zero_latency_packets->count(channel_id) != 0);
  if ((*zero_latency_packets)[channel_id] == 0 && (*non_zero_latency_packets)[channel_id] == 0) {
    (*zero_latency_packets)[channel_id] = parallel_reduce * 1;
    (*non_zero_latency_packets)[channel_id] = get_non_zero_latency_packets();
    toggle = !toggle;
  }
  int current_receiver = f.dest;
  std::vector<int> current_sender = f.prev;
  if ((*zero_latency_packets)[channel_id] > 0) {
    NcclLog->writeLog(NcclLogLevel::DEBUG,"id:  %d (*zero_latency_packets)[channel_id] > 0",id);
    uint64_t message_size = f.flow_size;
    packets[std::make_pair(channel_id, flow_id)].push_back(MyPacket(
        stream->current_queue_id,
        current_sender[0], 
        current_receiver,
        message_size,
        channel_id,
        flow_id));
    packets[std::make_pair(channel_id, flow_id)].back().set_flow_id(flow_id);
    packets[std::make_pair(channel_id, flow_id)].back().sender = nullptr;
    locked_packets.push_back(&packets[std::make_pair(channel_id, flow_id)].back());
    processed = false;
    send_back = false;
    NPU_to_MA = true;
    release_packets(channel_id, flow_id, message_size);
    (*zero_latency_packets)[channel_id]--;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"id:  %d (*zero_latency_packets)[channel_id] : %d ",id,(*zero_latency_packets)[channel_id]);
    return;
  } else if ((*non_zero_latency_packets)[channel_id] > 0) {
    NcclLog->writeLog(NcclLogLevel::DEBUG,"id:  %d (*non_zero_latency_packets)[channel_id] > 0",id);
    uint64_t message_size = f.flow_size;
    packets[std::make_pair(channel_id, flow_id)].push_back(MyPacket(
        stream->current_queue_id,
        current_sender[0],  
        current_receiver,
        message_size,
        channel_id,
        flow_id)); 
    packets[std::make_pair(channel_id, flow_id)].back().set_flow_id(flow_id);
    packets[std::make_pair(channel_id, flow_id)].back().sender = nullptr;
    locked_packets.push_back(&packets[std::make_pair(channel_id, flow_id)].back());
    if (comType == ComType::Reduce_Scatter ||
        (comType == ComType::All_Reduce && toggle)) {
      processed = true;
    } else {
      processed = false;
    }
    if ((*non_zero_latency_packets)[channel_id] <= parallel_reduce * 1) {
      send_back = false;
    } else {
      send_back = true;
    }
    NPU_to_MA = false;
    release_packets(channel_id, flow_id, message_size);
    (*non_zero_latency_packets)[channel_id]--;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"id:  %d (*non_zero_latency_packets)[channel_id] : %d ",id,(*non_zero_latency_packets)[channel_id]);
    return;
  }
  Sys::sys_panic("should not inject nothing!");
}

bool NcclTreeFlowModel::ready(int channel_id, int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  MyPacket packet;
  {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
      stream->changeState(StreamState::Executing);
    }
    if (!enabled || packets[std::make_pair(channel_id, flow_id)].size() == 0 || _stream_count[channel_id] == 0) {
      NcclLog->writeLog(NcclLogLevel::DEBUG,"NcclTreeFlowModel not ready!");
      return false;
    }
    packet = packets[std::make_pair(channel_id, flow_id)].front();
  }
  std::vector<int>recv_prevs;
  recv_prevs = _flow_models[std::make_pair(channel_id, flow_id)].prev;
  for (int recv_prev : recv_prevs) {
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    rcv_req.reqCount = packet.msg_size;
    rcv_req.tag = channel_id;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        packet.preferred_vnet,
        packet.stream_num);
    ehd->flowTag.child_flow_id = -1;
    ehd->flowTag.current_flow_id = -1;
    auto flow_model = this->_flow_models[std::make_pair(channel_id,flow_id)];
    if(flow_model.parent_flow_id.size()==0 || flow_model.conn_type == "RING"){
      ehd->flowTag.tag_id = layer_num*flow_model.chunk_count*m_channels + flow_model.chunk_count*flow_model.channel_id+flow_model.chunk_id;
    }else{
      ehd->flowTag.tag_id = layer_num*flow_model.chunk_count*m_channels + flow_model.chunk_count*flow_model.channel_id+flow_model.chunk_id+1;
    }
    ehd->flowTag.channel_id = packet.channel_id;
    if (free_packets[std::make_pair(channel_id, recv_prev)] > 0) {
      stream->owner->front_end_sim_recv(
          0,
          Sys::dummy_data,
          rcv_req.reqCount,
          UINT8,
          recv_prev,
          rcv_req.tag,
          &rcv_req,
          &Sys::handleEvent,
          ehd);
    }
  }
  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = packet.preferred_dest;
  snd_req.tag = channel_id;
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;
  snd_req.reqCount = packet.msg_size;
  MockNccl::SingleFlow flow_model =
      this->_flow_models[std::make_pair(channel_id, flow_id)];
  snd_req.flowTag.tag_id = layer_num * flow_model.chunk_count * m_channels +
      flow_model.channel_id * flow_model.chunk_count + flow_model.chunk_id;
  snd_req.flowTag.channel_id = channel_id;
  snd_req.flowTag.flow_size = flow_model.flow_size;
  snd_req.flowTag.current_flow_id = flow_id;
  snd_req.flowTag.chunk_id = flow_model.chunk_id;
  snd_req.flowTag.child_flow_id = -1;
  snd_req.flowTag.tree_flow_list =
      this->_flow_models[std::make_pair(channel_id, flow_id)].child_flow_id;
  snd_req.flowTag.sender_node = id;
  snd_req.flowTag.receiver_node = packet.preferred_dest;
  snd_req.flowTag.pQps = this->pQps;
  if (this->comType == ComType::All_Reduce_NVLS)
    snd_req.flowTag.nvls_on = true;
  else
    snd_req.flowTag.nvls_on = false;
  SendPacketEventHandlerData* send_ehd = new SendPacketEventHandlerData(
      stream,
      id,
      packet.preferred_dest,
      channel_id,
      EventType::PacketSentFinshed);
  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      snd_req.reqCount,
      UINT8,
      packet.preferred_dest,
      snd_req.flowTag.tag_id,
      &snd_req,
      &Sys::handleEvent,
      send_ehd);
  return true;
}

void NcclTreeFlowModel::exit() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  #ifdef PHY_MTP
  auto now = std::chrono::system_clock::now();
  auto now_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          now.time_since_epoch())
          .count();
  NcclLog->writeLog(
      NcclLogLevel::DEBUG,
      "NcclTreeFlowModel exit time %lld",
      now_us);
  end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  NcclLog->writeLog(NcclLogLevel::DEBUG,"Communication Latency：%lld us",duration.count());
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);
  #else
  for(std::pair<std::pair<int, int>, std::list<MyPacket>> packet: packets) {
  if(packet.second.size() != 0)
    packet.second.clear();
  }
  if (locked_packets.size() != 0) {
    locked_packets.clear();
  }
  #endif
  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
  NcclLog->writeLog(NcclLogLevel::DEBUG,"NcclTreeFlowModel exit");
  return;
}

#ifdef PHY_RDMA
bool NcclTreeFlowModel::phy_iteratable(int channel_id){
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  bool all_send_finished = true, all_recv_finished = true;
  bool exit_flag = true;
  if(send_packets!=0||recv_packets!=0){
    exit_flag=false;
  }
  if(exit_flag){
    judge_exit_flag.store(true);
    return false;
  } else{
    return true;
  }
}

bool NcclTreeFlowModel::phy_ready(int channel_id,int flow_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (stream->state == StreamState::Created ||
      stream->state == StreamState::Ready) {
    stream->changeState(StreamState::Executing);
  }
  MockNccl::SingleFlow flow = _flow_models[std::make_pair(channel_id, flow_id)];
  std::vector<int>recv_prevs;
  recv_prevs = _flow_models[std::make_pair(channel_id, flow_id)].prev;
  for (int recv_prev : recv_prevs) {
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    rcv_req.layerNum = layer_num;
    rcv_req.reqCount = flow.flow_size;
    rcv_req.tag = channel_id;
    RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
        stream,
        stream->owner->id,
        EventType::PacketReceived,
        stream->current_queue_id,
        1);
    ehd->flowTag.child_flow_id = -1;
    ehd->flowTag.current_flow_id = -1;
    auto flow_model = this->_flow_models[std::make_pair(channel_id,flow_id)];
    if(flow_model.parent_flow_id.size()==0 || flow_model.conn_type == "RING"){
      ehd->flowTag.tag_id = layer_num*flow_model.chunk_count*m_channels + flow_model.chunk_count*flow_model.channel_id+flow_model.chunk_id;
    }else{
      ehd->flowTag.tag_id = layer_num*flow_model.chunk_count*m_channels + flow_model.chunk_count*flow_model.channel_id+flow_model.chunk_id+1;
    }
    ehd->flowTag.channel_id = flow.channel_id;
    if (free_packets[std::make_pair(channel_id, recv_prev)] > 0) {
      stream->owner->front_end_sim_recv(
          0,
          Sys::dummy_data,
          rcv_req.reqCount,
          UINT8,
          recv_prev,
          rcv_req.tag,
          &rcv_req,
          &Sys::handleEvent,
          ehd);
    }
  }
  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = flow.dest;
  snd_req.tag = channel_id;
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;
  snd_req.reqCount = flow.flow_size;
  MockNccl::SingleFlow flow_model =
      this->_flow_models[std::make_pair(channel_id, flow_id)];
  snd_req.flowTag.tag_id = layer_num * flow_model.chunk_count * m_channels +
      flow_model.channel_id * flow_model.chunk_count + flow_model.chunk_id;
  snd_req.flowTag.channel_id = channel_id;
  snd_req.flowTag.flow_size = flow_model.flow_size;
  snd_req.flowTag.current_flow_id = flow_id;
  snd_req.flowTag.chunk_id = flow_model.chunk_id;
  snd_req.flowTag.child_flow_id = -1;
  snd_req.flowTag.tree_flow_list =
      this->_flow_models[std::make_pair(channel_id, flow_id)].child_flow_id;
  snd_req.flowTag.sender_node = id;
  snd_req.flowTag.receiver_node = flow.dest;
  snd_req.flowTag.pQps = this->pQps;
  if (this->comType == ComType::All_Reduce_NVLS)
    snd_req.flowTag.nvls_on = true;
  else
    snd_req.flowTag.nvls_on = false;
  SendPacketEventHandlerData* send_ehd = new SendPacketEventHandlerData(
      stream,
      id,
      flow.dest,
      channel_id,
      EventType::PacketSentFinshed);
  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      snd_req.reqCount,
      UINT8,
      flow.dest,
      snd_req.tag,
      &snd_req,
      &Sys::handleEvent,
      send_ehd);
  return true;
}

void NcclTreeFlowModel::waiting_to_exit() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::DEBUG, "NcclTreeFlowModel::waiting_to_exit begin ");
  while (!judge_exit_flag) {
  };
  exit();
  return;
}
#endif
} // namespace AstraSim