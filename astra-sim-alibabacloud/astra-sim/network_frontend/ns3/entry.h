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

#ifndef __ENTRY_H__
#define __ENTRY_H__

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"
#define _QPS_PER_CONNECTION_  1
#include "common.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"
#include <fstream>
#include <iostream>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <vector>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#include <map>
#include"astra-sim/system/MockNcclQps.h"
#include "astra-sim/system/MockNcclLog.h"
using namespace ns3;
using namespace std;
const bool SPLIT_DATA_ON_QPS = false;
std::unordered_map<std::string, ApplicationContainer> appCon;
std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;

std::map<std::pair<uint64_t, std::pair<int, std::pair<int, int>>>, AstraSim::ncclFlowTag> sender_src_port_map;

struct task1 {
  int src;
  int dest;
  int type;
  uint64_t count;
  void *fun_arg;
  void (*msg_handler)(void *fun_arg);
  double schTime; 
};
map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
map<std::pair<int, std::pair<int, int>>, int> recvHash;
map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
map<std::pair<int, int>, int64_t> nodeHash;
map<std::pair<int,std::pair<int,int>>,int> waiting_to_sent_callback;  
map<std::pair<int,std::pair<int,int>>,int>waiting_to_notify_receiver;
map<std::pair<int,std::pair<int,int>>,uint64_t>received_chunksize;  
map<std::pair<int,std::pair<int,int>>,uint64_t>sent_chunksize;  
bool is_sending_finished(int src,int dst,AstraSim::ncclFlowTag flowTag){
  int tag_id = flowTag.current_flow_id;
  if (waiting_to_sent_callback.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    if (--waiting_to_sent_callback[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_sent_callback.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

bool is_receive_finished(int src,int dst,AstraSim::ncclFlowTag flowTag){
  int tag_id = flowTag.current_flow_id;
  map<std::pair<int,std::pair<int,int>>,int>::iterator it;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (waiting_to_notify_receiver.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    // std:: cout << "waiting count: " << waiting_to_notify_receiver.count(
    //       std::make_pair(tag_id, std::make_pair(src, dst))) << std :: endl;
    NcclLog->writeLog(NcclLogLevel::DEBUG," is_receive_finished waiting_to_notify_receiver  tag_id  %d src  %d dst  %d count  %d",tag_id,src,dst,waiting_to_notify_receiver[std::make_pair(
                     tag_id, std::make_pair(src, dst))]);
    if (--waiting_to_notify_receiver[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_notify_receiver.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}
inline std::string getHashKey(uint32_t src, uint32_t dst, uint32_t pg, uint32_t dport){
    return std::to_string(src) + '_' + std::to_string(dst) + '_' + std::to_string(pg) + '_' + std::to_string(dport);
}

std::vector<Ptr<RdmaClient>>  getClients(uint32_t src, uint32_t dst, uint32_t pg, uint32_t dport, 
    void (*msg_handler)(void *fun_arg), void *fun_arg, int tag, int sendLat, bool nvls_on){

  std::vector<Ptr<RdmaClient>> clients;
  std::string hashKey = getHashKey(src, dst, pg, dport); 
  #ifdef NS3_MTP
  MtpInterface::explicitCriticalSection cs;
  #endif
  uint32_t port = portNumber[src][dst];
  if(appCon[hashKey].GetN()==0){
    for(int i = 0; i < _QPS_PER_CONNECTION_;i++){
      uint32_t port = portNumber[src][dst]++; // get a new port number
      RdmaClientHelper clientHelper(
      pg, 
      serverAddress[src], 
      serverAddress[dst], 
      port, 
      dport, 
      0,  // create a qp w/o message
      has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
      global_t == 1 ? maxRtt : pairRtt[src][dst], msg_handler, fun_arg, tag,
      src, dst, false);
      if(nvls_on) clientHelper.SetAttribute("NVLS_enable", UintegerValue (1));
      appCon[hashKey].Add(clientHelper.Install(n.Get(src)));
    }
    appCon[hashKey].Start(Time(sendLat));

  }
  for (int i = 0; i < _QPS_PER_CONNECTION_; i++) {
    Ptr<RdmaClient> qp = DynamicCast<RdmaClient>(appCon[hashKey].Get(i));
    clients.push_back(qp);
  }
  #ifdef NS3_MTP
    cs.ExitSection();
  #endif
  return clients;

}
void PushMessagetoClient(Ptr<RdmaClient> client, uint64_t size, uint64_t curr_flow_id) {
  client->PushMessagetoQp(size, curr_flow_id);
}
void SendFlow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void *fun_arg), void *fun_arg, int tag, AstraSim::sim_request *request) {
  MockNcclLog*NcclLog = MockNcclLog::getInstance();
  bool nvls_on = request->flowTag.nvls_on;
  int flow_id = request->flowTag.current_flow_id;
  int pg = 3, dport = 100;
  int send_lat = 6000;
  const char* send_lat_env = std::getenv("AS_SEND_LAT");
  if (send_lat_env) {
    try {
      send_lat = std::stoi(send_lat_env);
    } catch (const std::invalid_argument& e) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"send_lat set error");
      exit(-1);
    }
  }
  send_lat *= 1000;
  // UserParam* param = UserParam::getInstance();
  uint32_t  gpus_per_server = 8;
  if(src/gpus_per_server == dst/gpus_per_server){
    //in same server; not enable multi-qp
    if(maxPacketCount == 0) maxPacketCount = 1;
    uint32_t port = portNumber[src][dst]++; 
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," 发包事件  %dSendFlow to  %d channelid:  %d flow_id  %d srcip  %d dstip  %d size:  %d at the tick:  %d",src,dst,tag,flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
    NcclLog->writeLog(NcclLogLevel::DEBUG," request->flowTag 发包事件  %dSendFlow to  %d tag_id:  %d flow_id  %d srcip  %d dstip  %d size:  %d at the tick:  %d",request->flowTag.sender_node,request->flowTag.receiver_node,request->flowTag.tag_id,request->flowTag.current_flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
    RdmaClientHelper clientHelper(
      pg, serverAddress[src], serverAddress[dst], port, dport, 0,
      has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
      global_t == 1 ? maxRtt : pairRtt[src][dst], msg_handler, fun_arg, tag,
      src, dst,true);
    if(nvls_on) clientHelper.SetAttribute("NVLS_enable", UintegerValue (1));
    {
      #ifdef NS3_MTP
        MtpInterface::explicitCriticalSection cs;
      #endif
      sender_src_port_map[std::make_pair(request->flowTag.current_flow_id, std::make_pair(port, std::make_pair(src, dst)))] = request->flowTag;
      ApplicationContainer appCon1 = clientHelper.Install(n.Get(src));
      appCon1.Start(Time(send_lat));
      Ptr<RdmaClient> client = DynamicCast<RdmaClient>(appCon1.Get(0));
      Simulator::Schedule(Time(send_lat+1), PushMessagetoClient, client, maxPacketCount,request->flowTag.current_flow_id);
      waiting_to_sent_callback[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      waiting_to_notify_receiver[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      #ifdef NS3_MTP
        cs.ExitSection();
      #endif
    }
  }else{
    if(maxPacketCount == 0) maxPacketCount = 1;
    if (SPLIT_DATA_ON_QPS == false){
      int qp_index = std::rand() % _QPS_PER_CONNECTION_;
      NcclLog->writeLog(NcclLogLevel::DEBUG," 发包事件  %dSendFlow to  %d channelid:  %d flow_id  %d srcip  %d dstip  %d size:  %d at the tick:  %d",
        src,dst,tag,flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
    
      NcclLog->writeLog(NcclLogLevel::DEBUG," request->flowTag 发包事件  %dSendFlow to  %d tag_id:  %d flow_id  %d srcip  %d dstip  %d size:  %d at the tick:  %d",
        request->flowTag.sender_node,request->flowTag.receiver_node,request->flowTag.tag_id,request->flowTag.current_flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
      std::vector<Ptr<RdmaClient>> clients = getClients(src, dst, pg, dport, msg_handler, fun_arg, tag, send_lat, nvls_on);
      uint32_t port = clients[qp_index]->GetSourcePort();
      sender_src_port_map[std::make_pair(request->flowTag.current_flow_id, std::make_pair(port, std::make_pair(src, dst)))] = request->flowTag;
      Simulator::Schedule(Time(send_lat+1), PushMessagetoClient, clients[qp_index], maxPacketCount,request->flowTag.current_flow_id);
      waiting_to_sent_callback[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      waiting_to_notify_receiver[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      flow_input.idx++;
    }else{
      std::vector<Ptr<RdmaClient>> clients = getClients(src, dst, pg, dport, msg_handler, fun_arg, tag, send_lat, nvls_on);
      uint64_t base_size = maxPacketCount / _QPS_PER_CONNECTION_;
      uint64_t last_size = base_size + maxPacketCount % _QPS_PER_CONNECTION_;
      for (int i = 0; i < _QPS_PER_CONNECTION_; i++) {
        uint64_t size = (i == _QPS_PER_CONNECTION_ - 1)? last_size : base_size;
        NcclLog->writeLog(NcclLogLevel::DEBUG," 发包事件  %dSendFlow to  %d channelid:  %d flow_id  %d srcip  %d dstip  %d size:  %d at the tick:  %d",
          src,dst,tag,flow_id,serverAddress[src],serverAddress[dst],size,AstraSim::Sys::boostedTick());
        NcclLog->writeLog(NcclLogLevel::DEBUG," request->flowTag 发包事件  %d SendFlow to  %d tag_id:  %d flow_id  %d srcip  %d dstip  %d sport %d size:  %d at the tick:  %d",
          request->flowTag.sender_node,request->flowTag.receiver_node,request->flowTag.tag_id,request->flowTag.current_flow_id,serverAddress[src],serverAddress[dst],clients[i]->GetSourcePort(), size,AstraSim::Sys::boostedTick());
        sender_src_port_map[std::make_pair(request->flowTag.current_flow_id, std::make_pair(clients[i]->GetSourcePort(), std::make_pair(src, dst)))] = request->flowTag;
        Simulator::Schedule(Time(send_lat+1), PushMessagetoClient, clients[i], size,request->flowTag.current_flow_id);
        waiting_to_sent_callback[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
        waiting_to_notify_receiver[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      }
      flow_input.idx++;
    }
  }
}

void notify_receiver_receive_data(int sender_node, int receiver_node,
                                  uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;   
    #endif                         
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," %d notify recevier:  %d message size:  %d",sender_node,receiver_node,message_size);
    int tag = flowTag.tag_id;   
    if (expeRecvHash.find(make_pair(
            tag, make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
      task1 t2 =
          expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," %d notify recevier:  %d message size:  %d t2.count:  %d channle id:  %d",sender_node,receiver_node,message_size,t2.count,flowTag.channel_id);
      AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t2.fun_arg;
      if (message_size == t2.count) {
        NcclLog->writeLog(NcclLogLevel::DEBUG," message_size = t2.count expeRecvHash.erase  %d notify recevier:  %d message size:  %d channel_id  %d",sender_node,receiver_node,message_size,tag);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else if (message_size > t2.count) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] =
            message_size - t2.count;
        NcclLog->writeLog(NcclLogLevel::DEBUG,"message_size > t2.count expeRecvHash.erase %d notify recevier:  %d message size:  %d channel_id  %d",sender_node,receiver_node,message_size,tag);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else {
        t2.count -= message_size;
        expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = t2;
      }
    } else {
      receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node),tag)] = flowTag;
      if (recvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) ==
          recvHash.end()) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] =
            message_size;
      } else {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] +=
            message_size;
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  receiver_end_1st_section:
    {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs2;
    #endif  
    if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
      nodeHash[make_pair(receiver_node, 1)] = message_size;
    } else {
      nodeHash[make_pair(receiver_node, 1)] += message_size;
    }
    #ifdef NS3_MTP
    cs2.ExitSection();
    #endif
    }
  }
}

void notify_sender_sending_finished(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  { 
    MockNcclLog * NcclLog = MockNcclLog::getInstance();
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif    
    int tag = flowTag.tag_id;        
    if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
      sentHash.end()) {
      task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
      ehd->flowTag=flowTag;   
      if (t2.count == message_size) {
        sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
          nodeHash[make_pair(sender_node, 0)] = message_size;
        } else {
          nodeHash[make_pair(sender_node, 0)] += message_size;
        }
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t2.msg_handler(t2.fun_arg);
        goto sender_end_1st_section;
      }else{
        NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash msg size != sender_node %d receiver_node %d message_size %lu flow_id ",sender_node,receiver_node,message_size);
      }
    }else{
      NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash cann't find sender_node %d receiver_node %d message_size %lu",sender_node,receiver_node,message_size);
    }       
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
sender_end_1st_section:
  return;
}


void notify_sender_packet_arrivered_receiver(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  int tag = flowTag.channel_id;
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
      sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
    ehd->flowTag=flowTag;
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      t2.msg_handler(t2.fun_arg);
    }
  }
}

void Finish(){
    for(auto it:appCon){
        for(int i = 0;i < it.second.GetN(); i++){
            Ptr<RdmaClient> app = DynamicCast<RdmaClient>(it.second.Get(i));
            app->FinishQp();
        }
    }
}

void message_finish(FILE* fout, Ptr<RdmaQueuePair> q, uint64_t msgSize, uint64_t flow_id){
  MockNcclLog * NcclLog = MockNcclLog::getInstance(); 
  NcclLog->writeLog(NcclLogLevel::DEBUG,"message_finish");
  #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
  #endif
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  uint32_t pay_l = get_config_value_ns3<uint64_t>("ns3::RdmaHw::Mtu");
  uint64_t size = msgSize;
  uint32_t total_bytes = size +
      ((size - 1) / pay_l + 1) *
          (CustomHeader::GetStaticWholeHeaderSize() -
           IntHeader::GetStaticSize()); // translate to the minimum bytes
                                        // required (with header but no INT)
  uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
  fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
          q->sport, q->dport, size, q->startTime.GetTimeStep(),
          (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
  fflush(fout);

  AstraSim::ncclFlowTag flowTag;
  uint64_t notify_size;
  if (sender_src_port_map.find(make_pair(flow_id,make_pair(q->sport, make_pair(sid, did)))) ==
        sender_src_port_map.end()) {
    NcclLog->writeLog(NcclLogLevel::ERROR,"could not find the tag, there must be something wrong");
      exit(-1);
  }
  flowTag = sender_src_port_map[std::make_pair(flow_id, std::make_pair(q->sport, std::make_pair(sid, did)))];
  if(flowTag.current_flow_id != flow_id){
    NcclLog->writeLog(NcclLogLevel::DEBUG,"Exit with unequal flow_id");
    exit(-1);    
  }
  sender_src_port_map.erase(make_pair(flow_id,make_pair(q->sport, make_pair(sid, did))));
  received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=size;
  if(!is_receive_finished(sid,did,flowTag)) {
    #ifdef NS3_MTP
        cs.ExitSection();
      #endif
      return; 
    }
    notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    received_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))); 
    #ifdef NS3_MTP
      cs.ExitSection();
    #endif  
    NcclLog->writeLog(NcclLogLevel::DEBUG,"Before enter notify_receiver_data");
    notify_receiver_receive_data(sid, did, notify_size, flowTag);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"Out notify_receiver_data");

}





void qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
    uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"qp finish, src:  %d did:  %d port:  %d at the tick:  %d",sid,did,q->sport,AstraSim::Sys::boostedTick());
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
    return; 
}

void send_finish(FILE *fout, Ptr<RdmaQueuePair> q,uint64_t msgSize, uint64_t flow_id) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  AstraSim::ncclFlowTag flowTag;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"数据包出发送网卡 send finish, src:  %d did:  %d port:  %d flow_id: %d srcip  %d dstip  %d total bytes:  %d at the tick:  %d",sid,did,q->sport,flow_id,q->sip,q->dip,msgSize,AstraSim::Sys::boostedTick());
  int all_sent_chunksize;
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    flowTag = sender_src_port_map[std::make_pair(flow_id, std::make_pair(q->sport, std::make_pair(sid, did)))];
    if(flowTag.current_flow_id != flow_id){
      NcclLog->writeLog(NcclLogLevel::DEBUG,"Exit with unequal flow_id");
      exit(-1);
    }
    sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=msgSize;
    if(!is_sending_finished(sid,did,flowTag)) {
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      return;
    }
    all_sent_chunksize = sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    sent_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  notify_sender_sending_finished(sid, did, all_sent_chunksize, flowTag);
}

int main1(string network_topo,string network_conf) {
  clock_t begint, endt;
  begint = clock();

  if (!ReadConf(network_topo,network_conf))
    return -1;
  SetConfig();
  SetupNetwork(qp_finish,message_finish,send_finish);

std::cout << "Running Simulation.\n";
  fflush(stdout);
  NS_LOG_INFO("Run Simulation.");

  endt = clock();
  return 0;
}
#endif