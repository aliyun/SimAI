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
#ifdef PHY_RDMA
#include"astra-sim/system/SimAiFlowModelRdma.hh"
#endif
#include"astra-sim/system/PhyMultiThread.hh"
#include"astra-sim/system/RecvPacketEventHadndlerData.hh"
#include"astra-sim/system/Common.hh"
#include"astra-sim/system/BaseStream.hh"
#include"astra-sim/system/StreamBaseline.hh"

#include"SimAiEntry.h"
using namespace std;

extern FlowPhyRdma flow_rdma;

AstraSim::Sys* global_sys = nullptr;

static void 
notify_receiver_receive_data(int sender_node, int receiver_node,
                                  uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  AstraSim::StreamBaseline* owner = global_sys->running_list.front();
  AstraSim::RecvPacketEventHadndlerData *ehd = new AstraSim::RecvPacketEventHadndlerData(owner, AstraSim::EventType::PacketReceived, flowTag);
  owner->consume(ehd);
}

static void 
notify_sender_sending_finished(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  AstraSim::StreamBaseline* owner = global_sys->running_list.front();
  AstraSim::SendPacketEventHandlerData* send_ehd = new AstraSim::SendPacketEventHandlerData(
    owner,
    flowTag.sender_node,
    flowTag.receiver_node,
    flowTag.channel_id,
    AstraSim::EventType::PacketSentFinshed);
    send_ehd->flowTag = flowTag;
  NcclLog->writeLog(NcclLogLevel::DEBUG,"notify_sender_sending_finished_test src %d dst %d channe_id %d flow_id %d",flowTag.sender_node,flowTag.receiver_node,flowTag.channel_id,flowTag.channel_id);
  owner->sendcallback(send_ehd);                       
}

static void 
simai_recv_finish(AstraSim::ncclFlowTag flowTag) {
  uint32_t sid = flowTag.sender_node, did = flowTag.receiver_node;
  uint64_t notify_size = flowTag.flow_size;
  notify_receiver_receive_data(sid, did, notify_size, flowTag);
}

static void 
simai_send_finish(AstraSim::ncclFlowTag flowTag) {
  uint32_t sid = flowTag.sender_node, did = flowTag.receiver_node;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(
      NcclLogLevel::DEBUG,
      " 数据包出网卡队列, src %d did %d total_bytes %lu channel_id %d flow_id %d tag_id %d",
      sid,
      did,
      flowTag.flow_size,
      flowTag.channel_id,
      flowTag.current_flow_id,
      flowTag.tag_id);
  notify_sender_sending_finished(sid, did, flowTag.flow_size, flowTag);
}

void set_simai_network_callback(){  
  set_receive_finished_callback(simai_recv_finish);
  set_send_finished_callback(simai_send_finish);
}

void send_flow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void *fun_arg), void *fun_arg, int tag, AstraSim::sim_request *request) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  AstraSim::ncclFlowTag flowtag = request->flowTag;
  TransportData send_data = TransportData(
      flowtag.channel_id,
      flowtag.chunk_id,
      flowtag.current_flow_id,
      flowtag.child_flow_id,
      flowtag.sender_node,
      flowtag.receiver_node,
      flowtag.flow_size,
      flowtag.pQps,
      flowtag.tag_id,
      flowtag.nvls_on);
  send_data.child_flow_size = flowtag.tree_flow_list.size();
  for (int i = 0; i < flowtag.tree_flow_list.size(); i++) {
    send_data.child_flow_list[i] = flowtag.tree_flow_list[i];
  }
  NcclLog->writeLog(
      NcclLogLevel::DEBUG,
      "SendPackets %d SendFlow to %d channelid: %d flow_id: %d size: %lu tag_id %d",
      src,
      dst,
      tag,
      flowtag.current_flow_id,
      maxPacketCount,
      request->flowTag.tag_id);
  #ifdef PHY_RDMA
  flow_rdma.simai_ibv_post_send(
      tag,
      src,
      dst,
      &send_data,
      sizeof(struct TransportData),
      maxPacketCount,
      flowtag.chunk_id);
  #endif
}

