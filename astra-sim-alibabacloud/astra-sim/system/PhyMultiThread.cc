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

#include<chrono>
#include "PhyMultiThread.hh"

extern FlowPhyRdma flow_rdma; 

std::atomic<bool> PhyMtpInterface::g_e_inCriticalSection (false);

std::map<int,std::atomic<int>> all_recv_size;
std::map<int,std::atomic<int>> all_send_size;
bool end_flag = false;

void (*send_finished_callback)(AstraSim::ncclFlowTag flowTag);
void (*receive_finished_callback)(AstraSim::ncclFlowTag flowTag);

void 
set_send_finished_callback(void (*msg_handler)(AstraSim::ncclFlowTag flowTag)){
    send_finished_callback = msg_handler;
}

void 
set_receive_finished_callback(void (*msg_handler)(AstraSim::ncclFlowTag flowTag)){
    receive_finished_callback = msg_handler;
}

static void 
insert_recv_cqe(void* buff) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    TransportData* ptrrecvdata = reinterpret_cast<TransportData*> (buff);
    AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
      ptrrecvdata->channel_id,
      ptrrecvdata->chunk_id,
      ptrrecvdata->current_flow_id,
      ptrrecvdata->child_flow_id,
      ptrrecvdata->sender_node,
      ptrrecvdata->receiver_node,
      ptrrecvdata->flow_size,
      ptrrecvdata->pQps,
      ptrrecvdata->tag_id,
      ptrrecvdata->nvls_on);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"PhyMultiThread.cc::insert_recv_cqe src_id %d dst_id %d flow_id %d channel_id %d",flowTag.sender_node,flowTag.receiver_node,flowTag.current_flow_id,flowTag.channel_id);
    flowTag.tree_flow_list.clear();
    for(int i =0;i<ptrrecvdata->child_flow_size;i++){
        flowTag.tree_flow_list.push_back(ptrrecvdata->child_flow_list[i]);
    }
    receive_finished_callback(flowTag);
}

static void 
insert_send_cqe(void* buff) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    TransportData* ptrrecvdata = reinterpret_cast<TransportData*> (buff);
    AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
      ptrrecvdata->channel_id,
      ptrrecvdata->chunk_id,
      ptrrecvdata->current_flow_id,
      ptrrecvdata->child_flow_id,
      ptrrecvdata->sender_node,
      ptrrecvdata->receiver_node,
      ptrrecvdata->flow_size,
      ptrrecvdata->pQps,
      ptrrecvdata->tag_id,
      ptrrecvdata->nvls_on);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"PhyMultiThread.cc::insert_send_cqe src_id %d dst_id %d flow_id %d channel_id %d",flowTag.sender_node,flowTag.receiver_node,flowTag.current_flow_id,flowTag.channel_id);
    flowTag.tree_flow_list.clear();
    for(int i =0;i<ptrrecvdata->child_flow_size;i++){
        flowTag.tree_flow_list.push_back(ptrrecvdata->child_flow_list[i]);
    }
    send_finished_callback(flowTag);
}

static bool 
judge_polling_all_recv_cqe(void * buff){
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  TransportData* ptrrecvdata = reinterpret_cast<TransportData*> (buff);
  AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
    ptrrecvdata->channel_id,
    ptrrecvdata->chunk_id,
    ptrrecvdata->current_flow_id,
    ptrrecvdata->child_flow_id,
    ptrrecvdata->sender_node,
    ptrrecvdata->receiver_node,
    ptrrecvdata->flow_size,
    ptrrecvdata->pQps,
    ptrrecvdata->tag_id,
    ptrrecvdata->nvls_on);
  int temp = 0;
  {
    MockNcclLog*NcclLog = MockNcclLog::getInstance();
    if (!all_recv_size.count(flowTag.current_flow_id)) {
      all_recv_size[flowTag.current_flow_id] = 1;
    } else {
      all_recv_size[flowTag.current_flow_id]++;
    }
    temp = all_recv_size[flowTag.current_flow_id];
    NcclLog->writeLog(NcclLogLevel::DEBUG,"judge_polling_all_recv_cqe flow_id %d recv_cqe_size %d",flowTag.current_flow_id,temp);
  }
  if (temp == NCCL_QPS_PER_PEER) {
    return true;
  } else {
    return false;
  }
}

static bool 
judge_polling_all_send_cqe(void * buff){
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  TransportData* ptrrecvdata = reinterpret_cast<TransportData*> (buff);
  AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
    ptrrecvdata->channel_id,
    ptrrecvdata->chunk_id,
    ptrrecvdata->current_flow_id,
    ptrrecvdata->child_flow_id,
    ptrrecvdata->sender_node,
    ptrrecvdata->receiver_node,
    ptrrecvdata->flow_size,
    ptrrecvdata->pQps,
    ptrrecvdata->tag_id,
    ptrrecvdata->nvls_on);
  int temp = 0;
  {
    MockNcclLog*NcclLog = MockNcclLog::getInstance();
    
    if (!all_send_size.count(flowTag.current_flow_id)) {
      all_send_size[flowTag.current_flow_id] = 1;
    } else {
      all_send_size[flowTag.current_flow_id]++;
    }
    temp = all_send_size[flowTag.current_flow_id];
    NcclLog->writeLog(NcclLogLevel::DEBUG,"judge_polling_all_send_cqe flow_id %d send_cqe_size %d",flowTag.current_flow_id,temp);
  }
  if (temp == NCCL_QPS_PER_PEER) {
    return true;
  } else {
    return false;
  }
}

bool 
create_polling_cqe_thread(void * cq_ptr,int lcore_id){
  #ifdef PHY_RDMA
  ibv_cq*cq = static_cast<ibv_cq*>(cq_ptr);
  struct ibv_wc wc[TEST_IO_DEPTH] = {};
  #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int ret = 0;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"PhyMultiThread.cc::create_polling_cqe_thread begin");
    while (!end_flag)
    {
      #ifdef PHY_RDMA
        memset(wc, 0, sizeof(wc));
        ret = ibv_poll_cq(cq,TEST_IO_DEPTH,wc);
        assert(ret>=0);
        if(ret >0){
        NcclLog->writeLog(NcclLogLevel::DEBUG,"PhyMultiThread.cc::create_polling_send_cqe_thread cqe num %d",ret);
          for (int i = 0; i < ret; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
              NcclLog->writeLog(
                  NcclLogLevel::ERROR,
                  " wr's status is error %d opcode %d ",
                  wc[i].status,
                  wc[i].opcode);
            }
            assert(wc[i].status == IBV_WC_SUCCESS);
            if (wc[i].opcode == IBV_WC_RECV ||
                wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              std::chrono::time_point<std::chrono::high_resolution_clock>
                  start_time = std::chrono::high_resolution_clock::now();
              auto now = std::chrono::system_clock::now();
              auto now_us =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      now.time_since_epoch())
                      .count();
              NcclLog->writeLog(
                  NcclLogLevel::DEBUG,
                  "poll_recv_cqe qpn %d wr_id %d chunk_id %d time %lld",
                  wc[i].qp_num,
                  wc[i].wr_id,
                  wc[i].imm_data,
                  now_us);
              void* recv_buff = flow_rdma.recv_wr_id_to_buff(wc[i].qp_num, wc[i].wr_id,wc[i].imm_data);
              if (judge_polling_all_recv_cqe(recv_buff)) {
                insert_recv_cqe(recv_buff);
              }
            } else if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
              auto now = std::chrono::system_clock::now();
              auto now_us =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      now.time_since_epoch())
                      .count();
              NcclLog->writeLog(
                  NcclLogLevel::DEBUG,
                  "poll_send_cqe qpn %d wr_id %d time %lld",
                  wc[i].qp_num,
                  wc[i].wr_id,
                  now_us);
              void* send_buff = flow_rdma.send_wr_id_to_buff(wc[i].qp_num, wc[i].wr_id);
              if (judge_polling_all_send_cqe(send_buff)) {
                insert_send_cqe(send_buff);
              }
            }
          }
        }
      #endif
    }
}

void 
notify_all_thread_finished(){
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  end_flag = true;
  NcclLog->writeLog(NcclLogLevel::DEBUG,"PhyMultiThread::notify_all_thread_finished end");
}