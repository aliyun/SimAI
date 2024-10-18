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
#include<thread>
#include<sys/socket.h>
#include<sys/ioctl.h>
#include <linux/if.h>
#include <mpi.h>

#include"SimAiFlowModelRdma.hh"
#include"PhyMultiThread.hh"
#include"MockNcclLog.h"
#include"BootStrapnet.hh"
#include"AstraNetworkAPI.hh"
#include"MockNcclLog.h"

#define IB_PORT 1

extern int world_size;
extern int local_rank;
extern std::map<int,std::string> rank2addr;

FlowPhyRdma flow_rdma;

FlowPhyRdma::FlowPhyRdma(int _gid_index):gid_index(_gid_index){
    g_ibv_ctx = nullptr;
}
FlowPhyRdma::~FlowPhyRdma(){
    // ibv_fini();
}


void* 
FlowPhyRdma::send_wr_id_to_buff(int qpn,int wr_id){
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    ibv_send_wr send_wr = ibv_send_wr_map[std::make_pair(qpn,wr_id)];
    void* buff = reinterpret_cast<void*>(send_wr.sg_list[0].addr);
    TransportData* ptrsendata = reinterpret_cast<TransportData*> (buff);
    AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
        ptrsendata->channel_id,
        ptrsendata->chunk_id,
        ptrsendata->current_flow_id,
        ptrsendata->child_flow_id,
        ptrsendata->sender_node,
        ptrsendata->receiver_node,
        ptrsendata->flow_size,
        ptrsendata->pQps,
        ptrsendata->tag_id,
        ptrsendata->nvls_on);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"SimAiFlowModelRdma.cc::send_wr_id_to_buff 数据包 send cqe,src_id %d dst_id %d qpn %d wr_id %d remote_addr %lld len %d  flow_id %d channel_id %d message_count: %lu",flowTag.sender_node,flowTag.receiver_node,qpn,wr_id,send_wr.wr.rdma.remote_addr,send_wr.sg_list[0].length,flowTag.current_flow_id,flowTag.channel_id,flowTag.flow_size);
    return buff;
}

void* 
FlowPhyRdma::recv_wr_id_to_buff(int qpn,int wr_id,int chunk_id){
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    insert_recv_wr(qpn);
    ibv_qp_context qp = qpn2ctx[qpn];
    uint64_t recv_addr = qp.src_info.recv_mr.addr+chunk_id * qp.chunk_size;
    void* buff = reinterpret_cast<void*>(recv_addr);
    TransportData* ptrsendata = reinterpret_cast<TransportData*> (buff);
    AstraSim::ncclFlowTag flowTag = AstraSim::ncclFlowTag(
        ptrsendata->channel_id,
        ptrsendata->chunk_id,
        ptrsendata->current_flow_id,
        ptrsendata->child_flow_id,
        ptrsendata->sender_node,
        ptrsendata->receiver_node,
        ptrsendata->flow_size,
        ptrsendata->pQps,
        ptrsendata->tag_id,
        ptrsendata->nvls_on);
    NcclLog->writeLog(NcclLogLevel::DEBUG,"SimAiFlowModelRdma.cc::recv_wr_id_to_buff 数据包 recv cqe,src_id %d dst_id %d qpn %d wr_id %d local_addr %lld flow_id %d channel_id %d message_count: %lu",flowTag.sender_node,flowTag.receiver_node,qpn,wr_id,recv_addr,flowTag.current_flow_id,flowTag.channel_id,flowTag.flow_size);
    return buff;
}

static int 
modify_qo_to_rts(struct ibv_qp_context qp_ctx){
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    struct ibv_qp_attr attr;
	int flags;
	int rc;
    /* modify qp to INIT */
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_INIT;
	attr.port_num = IB_PORT;
	attr.pkey_index = 0;
	attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
	rc = ibv_modify_qp(qp_ctx.qp, &attr, flags);
	if (rc){
        NcclLog->writeLog(NcclLogLevel::ERROR,"failed to modify QP state to INIT");
    }
    /* modify the QP to RTR */
	memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = IBV_MTU_256;
	attr.dest_qp_num = qp_ctx.dest_info.qp_num;
	attr.rq_psn = qp_ctx.dest_info.psn;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 0x12;
	attr.ah_attr.is_global = 0;
	attr.ah_attr.dlid = qp_ctx.dest_info.lid;
	attr.ah_attr.sl = 0;
	attr.ah_attr.src_path_bits = 0;
	attr.ah_attr.port_num = IB_PORT;
    /* gid */
    {
        attr.ah_attr.is_global = 1;
		attr.ah_attr.port_num = IB_PORT;
		memcpy(&attr.ah_attr.grh.dgid, &qp_ctx.dest_info.my_gid, 16);
		attr.ah_attr.grh.flow_label = 0;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.sgid_index = qp_ctx.src_info.gid_index;
		attr.ah_attr.grh.traffic_class = 0;
        
        {
        uint8_t remote_gid[16];
        memcpy(remote_gid,&attr.ah_attr.grh.dgid,16);
        uint8_t *p = remote_gid;
        NcclLog->writeLog(NcclLogLevel::DEBUG,"remote_lid 0x%x local_gidindex %d remote_qpn %d remote_psn %d remote Gid %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",attr.ah_attr.dlid, attr.ah_attr.grh.sgid_index,attr.dest_qp_num,attr.rq_psn,p[0],
                    p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);

        }

    }
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
			IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    rc = ibv_modify_qp(qp_ctx.qp, &attr, flags);
    if (rc){
        NcclLog->writeLog(NcclLogLevel::ERROR,"failed to modify QP state to RTR");
    } else {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"success to modify QP state to RTR");
    }
    /* modify the QP to RTS */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
	attr.timeout = 0x12;
	attr.retry_cnt = 6;
	attr.rnr_retry = 0;
	attr.sq_psn = 0;
	attr.max_rd_atomic = 1;
	flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
	rc = ibv_modify_qp(qp_ctx.qp, &attr, flags);
    if (rc){
        NcclLog->writeLog(NcclLogLevel::ERROR,"failed to modify QP state to RTS");
    } else {
        NcclLog->writeLog(NcclLogLevel::DEBUG,"success to modify QP state to RTS");
    }
    return rc;
}

static struct 
ibv_hand_shake ibv_qp_conn(int rank,int src_rank,int dst_rank,int tag_id,struct ibv_hand_shake send_data){
    MPI_Datatype mr_info_type;
    int blocklengths_mr[4] = {1,1,1,1};
    MPI_Datatype types_mr[4] = {MPI_UINT64_T, MPI_UINT64_T, MPI_UINT32_T, MPI_UINT32_T};
    MPI_Aint disp_mr[4];
    disp_mr[0] = offsetof(struct mr_info,addr);
    disp_mr[1] = offsetof(struct mr_info,len);
    disp_mr[2] = offsetof(struct mr_info,lkey);
    disp_mr[3] = offsetof(struct mr_info,rkey);
    MPI_Type_create_struct(4, blocklengths_mr, disp_mr, types_mr, &mr_info_type);
    MPI_Type_commit(&mr_info_type);
    MPI_Datatype mr_gid_type;
    MPI_Type_contiguous(16,MPI_UINT8_T,&mr_gid_type);
    MPI_Type_commit(&mr_gid_type);
    MPI_Datatype hand_shake_type;
    int blocklengths_hand_shake[7] = {1,1,1,1,1,1,1};
    MPI_Datatype types_hand_shake[7] ={MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED,MPI_UINT16_T, mr_gid_type, mr_info_type, mr_info_type};
    MPI_Aint disp_hande_shake[7];
    disp_hande_shake[0] = offsetof(struct ibv_hand_shake,gid_index);
    disp_hande_shake[1] = offsetof(struct ibv_hand_shake,qp_num);
    disp_hande_shake[2] = offsetof(struct ibv_hand_shake,psn);
    disp_hande_shake[3] = offsetof(struct ibv_hand_shake,lid);
    disp_hande_shake[4] = offsetof(struct ibv_hand_shake,my_gid);
    disp_hande_shake[5] = offsetof(struct ibv_hand_shake,recv_mr);
    disp_hande_shake[6] = offsetof(struct ibv_hand_shake,send_mr);
    MPI_Type_create_struct(7, blocklengths_hand_shake, disp_hande_shake, types_hand_shake, &hand_shake_type);
    MPI_Type_commit(&hand_shake_type);    
    struct ibv_hand_shake recv_data;
    if(rank == src_rank){   
        MPI_Send(&send_data, 1, hand_shake_type, dst_rank, tag_id, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, hand_shake_type, dst_rank, tag_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if(rank == dst_rank){ 
        MPI_Recv(&recv_data, 1, hand_shake_type, src_rank, tag_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&send_data, 1, hand_shake_type, src_rank, tag_id, MPI_COMM_WORLD);
    }
    return recv_data;
}

std::vector<struct ibv_qp_context> 
FlowPhyRdma::ibv_srv_alloc_ctx(
    int rank,
    int src_rank,
    int dst_rank,
    int channel_id,
    struct ibv_context* g_ibv_ctx,
    int chunk_count,
    uint64_t buffer_size,
    int qp_nums) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  int rc = 0;
  std::vector<struct ibv_qp_context> qps;
  struct ibv_port_attr port_attr;
  memset(&port_attr, 0, sizeof(ibv_port_attr));
  ibv_query_port(g_ibv_ctx, IB_PORT, &port_attr);
  union ibv_gid my_gid;
  rc = ibv_query_gid(g_ibv_ctx, IB_PORT, gid_index, &my_gid);
  if (rc) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "src_rank %d dst_rank %d get the gid failed",
        src_rank,
        dst_rank);
    rc = 1;
  }
  struct ibv_pd* pd = ibv_alloc_pd(g_ibv_ctx);
  if (!pd) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "src_rank %d dst_rank %d ibv_alloc_pd failed",
        src_rank,
        dst_rank);
    rc = 1;
  }
  ibv_cq* recv_cq = ibv_create_cq(g_ibv_ctx, 16384, NULL, NULL, 0);
  if (!recv_cq) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "src_rank %d dst_rank %d ibv_create_cq recv_cq failed",
        src_rank,
        dst_rank);
    rc = 1;
  }
  ibv_cq* send_cq = ibv_create_cq(g_ibv_ctx, 16384, NULL, NULL, 0);
  if (!send_cq) {
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "src_rank %d dst_rank %d ibv_create_cq send_cq failed",
        src_rank,
        dst_rank);
    rc = 1;
  }
  for (int i = 0; i < qp_nums; i++) {
    struct ibv_qp_context qp_ctx;
    qp_ctx.chunk_size = buffer_size;
    qp_ctx.recv_buf = malloc(chunk_count * buffer_size);
    if (!qp_ctx.recv_buf) {
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d recv_buffer alloc failed chunk_count %d buffer_size %d",
          src_rank,
          dst_rank,
          chunk_count,
          buffer_size);
      rc = 1;
    }
    qp_ctx.send_buf = malloc(chunk_count * buffer_size);
    if (!qp_ctx.send_buf) {
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d send_buffer alloc failed chunk_count %d buffer_size %d",
          src_rank,
          dst_rank,
          chunk_count,
          buffer_size);
      rc = 1;
    }
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    qp_ctx.recv_mr =
        ibv_reg_mr(pd, qp_ctx.recv_buf, chunk_count * buffer_size, mr_flags);
    if (!qp_ctx.recv_mr) {
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d recv_mr register failed",
          src_rank,
          dst_rank);
      rc = 1;
    }
    qp_ctx.send_mr =
        ibv_reg_mr(pd, qp_ctx.send_buf, chunk_count * buffer_size, mr_flags);
    if (!qp_ctx.send_mr) {
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d send_mr register failed",
          src_rank,
          dst_rank);
      rc = 1;
    }
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.cap.max_send_wr = 512;
    qp_init_attr.cap.max_recv_wr = 16384;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    struct ibv_qp* qp;
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d create qp failed",
          src_rank,
          dst_rank);
      rc = 1;
    }
    qp_ctx.qp = qp;
    qp_ctx.src_info.psn = 0;
    qp_ctx.src_info.gid_index = gid_index;
    qp_ctx.src_info.qp_num = qp->qp_num;
    qp_ctx.src_info.lid = port_attr.lid;
    memcpy(&qp_ctx.src_info.my_gid, &my_gid, 16);
    qp_ctx.src_info.recv_mr.addr = (uint64_t)qp_ctx.recv_buf;
    qp_ctx.src_info.recv_mr.len = chunk_count * buffer_size;
    qp_ctx.src_info.recv_mr.rkey = qp_ctx.recv_mr->rkey;
    qp_ctx.src_info.recv_mr.lkey = qp_ctx.recv_mr->lkey;
    qp_ctx.src_info.send_mr.addr = (uint64_t)qp_ctx.send_buf;
    qp_ctx.src_info.send_mr.len = chunk_count * buffer_size;
    qp_ctx.src_info.send_mr.lkey = qp_ctx.send_mr->lkey;
    qp_ctx.src_info.send_mr.rkey = qp_ctx.send_mr->rkey;
    {
      uint8_t local_gid[16];
      memcpy(local_gid, &qp_ctx.src_info.my_gid, 16);
      uint8_t* p = local_gid;
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d local_lid 0x%x local_gidindex %d local_qpn %d local_psn %d local Gid %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",
          src_rank,
          dst_rank,
          qp_ctx.src_info.lid,
          qp_ctx.src_info.gid_index,
          qp_ctx.src_info.qp_num,
          qp_ctx.src_info.psn,
          p[0],
          p[1],
          p[2],
          p[3],
          p[4],
          p[5],
          p[6],
          p[7],
          p[8],
          p[9],
          p[10],
          p[11],
          p[12],
          p[13],
          p[14],
          p[15]);
    }
    qp_ctx.dest_info =
        ibv_qp_conn(rank, src_rank, dst_rank, channel_id, qp_ctx.src_info);
    {
      uint8_t remote_gid[16];
      memcpy(remote_gid, &qp_ctx.dest_info.my_gid, 16);
      uint8_t* p = remote_gid;
      NcclLog->writeLog(
          NcclLogLevel::DEBUG,
          "src_rank %d dst_rank %d remote_lid 0x%x remote_gidindex %d remote_qpn %d remote_psn %d remote Gid %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",
          src_rank,
          dst_rank,
          qp_ctx.dest_info.lid,
          qp_ctx.dest_info.gid_index,
          qp_ctx.dest_info.qp_num,
          qp_ctx.dest_info.psn,
          p[0],
          p[1],
          p[2],
          p[3],
          p[4],
          p[5],
          p[6],
          p[7],
          p[8],
          p[9],
          p[10],
          p[11],
          p[12],
          p[13],
          p[14],
          p[15]);
    }
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "src_rank %d dst_rank %d create qp Success",
        src_rank,
        dst_rank);

    modify_qo_to_rts(qp_ctx);

    if (rc) {
      if (qp) {
        ibv_destroy_qp(qp);
        qp = NULL;
      }
      if (qp_ctx.recv_mr) {
        ibv_dereg_mr(qp_ctx.recv_mr);
        qp_ctx.recv_mr = NULL;
      }
      if (qp_ctx.send_mr) {
        ibv_dereg_mr(qp_ctx.send_mr);
        qp_ctx.send_mr = NULL;
      }
      if (qp_ctx.send_buf) {
        free(qp_ctx.send_buf);
        qp_ctx.send_buf = NULL;
      }
      if (qp_ctx.recv_buf) {
        free(qp_ctx.recv_buf);
        qp_ctx.recv_buf = NULL;
      }
      if (recv_cq) {
        ibv_destroy_cq(recv_cq);
        recv_cq = NULL;
      }
      if (send_cq) {
        ibv_destroy_cq(send_cq);
        send_cq = NULL;
      }
      if (pd) {
        ibv_dealloc_pd(pd);
        pd = NULL;
      }
    }
    qps.push_back(qp_ctx);
    qpn2ctx[qp_ctx.qp->qp_num] = qp_ctx;
  }
  return qps;
}

bool 
FlowPhyRdma::simai_ibv_post_send(
    int channel_id,
    int src_rank,
    int dst_rank,
    void* send_buf, 
    uint64_t len,
    uint64_t data_size,
    int chunk_id) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  int buff_size_per_qp = data_size / NCCL_QPS_PER_PEER;
  TransportData* ptrrecvdata = reinterpret_cast<TransportData*> (send_buf);
  int send_wr_nums = buff_size_per_qp / SEND_CHUNK_SIZE;
  for (int i = 0; i < NCCL_QPS_PER_PEER; i++) {
    int ret = 0;
    struct ibv_send_wr send_wr[WR_NUMS];
    struct ibv_send_wr* bad_wr = NULL;
  NcclLog->writeLog(NcclLogLevel::DEBUG,"ibv_peer_qps src_rank %d dst_rank %d channel_id %d qp_size %d",ptrrecvdata->sender_node,ptrrecvdata->receiver_node,ptrrecvdata->child_flow_id,ibv_peer_qps[std::make_pair(src_rank, dst_rank)][channel_id].size());
    struct ibv_qp_context qp =
            ibv_peer_qps[std::make_pair(src_rank, dst_rank)][channel_id][i];
    memcpy(qp.send_buf + chunk_id * buff_size_per_qp, send_buf, len);
    int send_wr_id = ibv_send_wr_id_map[qp.qp->qp_num]++;
    for(int j = 0;j<WR_NUMS;j++){
        send_wr[j].sg_list = reinterpret_cast<struct ibv_sge*>(calloc(1, sizeof(struct ibv_sge)));
        if(j !=WR_NUMS-1){
            send_wr[j].opcode = IBV_WR_RDMA_WRITE;
            send_wr[j].send_flags = 0;
        }else{
            send_wr[j].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            send_wr[j].send_flags = IBV_SEND_SIGNALED;
        }
        send_wr[j].imm_data = chunk_id;
        send_wr[j].num_sge = 1;
        if(j!=WR_NUMS-1){
            send_wr[j].next = &send_wr[j+1];
            send_wr[j].sg_list[0].addr = qp.src_info.send_mr.addr + buff_size_per_qp * chunk_id + (j+1)*(buff_size_per_qp/WR_NUMS);
        }else{
            send_wr[j].next = nullptr;
            send_wr[j].wr_id = send_wr_id;
            send_wr[j].sg_list[0].addr = qp.src_info.send_mr.addr + buff_size_per_qp * chunk_id;
        }
        send_wr[j].sg_list[0].length = buff_size_per_qp/WR_NUMS;
        send_wr[j].sg_list[0].lkey = qp.src_info.send_mr.lkey;
        send_wr[j].wr.rdma.remote_addr =
            qp.dest_info.recv_mr.addr + buff_size_per_qp * chunk_id;
        send_wr[j].wr.rdma.rkey = qp.dest_info.recv_mr.rkey;
    }
    ibv_send_wr_map[std::make_pair(qp.qp->qp_num, send_wr_id)] = send_wr[WR_NUMS-1];
    ret = ibv_post_send(qp.qp, send_wr, &bad_wr);
    if (ret != 0) {
      NcclLog->writeLog(
          NcclLogLevel::ERROR,
          "post send failed, ret: %d ,errno: %d",
          ret,
          errno);
    }
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"ibv_post_send qpn %d wr_id %d remote_addr %lld local_addr %lld channel_id %d flow_id %d time %lld",qp.qp->qp_num,send_wr_id,send_wr[WR_NUMS-1].wr.rdma.remote_addr,send_wr[WR_NUMS-1].sg_list[0].length,ptrrecvdata->channel_id,ptrrecvdata->current_flow_id,now_us);
  }

}

bool 
FlowPhyRdma::insert_recv_wr(int qpn){
    ibv_qp_context qp = qpn2ctx[qpn];
    struct ibv_recv_wr recv_wr = {};
    struct ibv_recv_wr* bad_wr = NULL;
    int ret = 0;
    recv_wr.sg_list =
        reinterpret_cast<struct ibv_sge*>(calloc(1, sizeof(struct ibv_sge)));
    assert_non_null(recv_wr.sg_list);
    int recv_wr_id = ibv_recv_wr_id_map[qp.qp->qp_num]++;
    MockNcclLog*NcclLog = MockNcclLog::getInstance();
    recv_wr.wr_id = recv_wr_id;
    recv_wr.sg_list = nullptr;
    recv_wr.num_sge = 0;
    recv_wr.next = NULL;
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "create_peer_qp,insert ibv_recv_wr_map elm, qpn %d recv_wr_id %d addr %lu len %d",
        qp.qp->qp_num,
        recv_wr_id,
        qp.src_info.recv_mr.addr,
        qp.src_info.recv_mr.len);
    ret = ibv_post_recv(qp.qp, &recv_wr, &bad_wr);
    assert(ret == 0);
}

bool 
FlowPhyRdma::init_recv_wr(ibv_qp_context qp,int nums){
    MockNcclLog* NcclLog =  MockNcclLog::getInstance();
    int ret = 0;
    for (int i = 0; i < nums; i++) {
        struct ibv_recv_wr recv_wr = {};
        struct ibv_recv_wr* bad_wr = NULL;
        recv_wr.sg_list =
            reinterpret_cast<struct ibv_sge*>(calloc(1, sizeof(struct ibv_sge)));
        assert_non_null(recv_wr.sg_list);
        int recv_wr_id = ibv_recv_wr_id_map[qp.qp->qp_num]++;
        recv_wr.wr_id = recv_wr_id;
        recv_wr.sg_list = nullptr;
        recv_wr.num_sge = 0;
        recv_wr.next = NULL;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            "create_peer_qp,insert ibv_recv_wr_map elm, qpn %d recv_wr_id %d addr %lu len %d",
            qp.qp->qp_num,
            recv_wr_id,
            qp.src_info.recv_mr.addr,
            qp.src_info.recv_mr.len);
        ret = ibv_post_recv(qp.qp, &recv_wr, &bad_wr);
        assert(ret == 0);
    }
}

bool 
FlowPhyRdma::ibv_create_peer_qp(
    int rank,
    int channel_id,
    int src_rank,
    int dst_rank,
    int chunk_count,
    int chunk_id,
    uint64_t buffer_size) {
  int ret = 0;
  int buff_size_per_qp = buffer_size / NCCL_QPS_PER_PEER;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if ((ibv_peer_qps.count(std::make_pair(src_rank, dst_rank)) == 0 ||
       ibv_peer_qps[std::make_pair(src_rank, dst_rank)].count(channel_id) ==
           0) &&
      (ibv_peer_qps.count(std::make_pair(dst_rank, src_rank)) == 0 ||
       ibv_peer_qps[std::make_pair(dst_rank, src_rank)].count(channel_id) ==
           0)) {
    ibv_peer_qps[std::make_pair(dst_rank, src_rank)][channel_id] =
        ibv_srv_alloc_ctx(
            rank,
            src_rank,
            dst_rank,
            channel_id,
            g_ibv_ctx,
            chunk_count,
            buff_size_per_qp,
            NCCL_QPS_PER_PEER);
    ibv_peer_qps[std::make_pair(src_rank, dst_rank)][channel_id] =
        ibv_peer_qps[std::make_pair(dst_rank, src_rank)][channel_id];
    std::thread poll_send_cqe_thread(
        create_polling_cqe_thread,
        ibv_peer_qps[std::make_pair(dst_rank, src_rank)][channel_id][0]
            .qp->send_cq,0);
    poll_send_cqe_thread.detach();
    std::thread poll_recv_cqe_thread(
        create_polling_cqe_thread,
        ibv_peer_qps[std::make_pair(dst_rank, src_rank)][channel_id][0]
            .qp->recv_cq,0);
    poll_recv_cqe_thread.detach();
  }
  NcclLog->writeLog(NcclLogLevel::DEBUG,"SimAiFlowModelRdma.cc create_peer_qp local_rank %d src %d dst %d channel_id %d",rank,src_rank,dst_rank,channel_id);
  if (dst_rank == rank) {
    for(int i =0;i<NCCL_QPS_PER_PEER;i++){
        ibv_qp_context qp = ibv_peer_qps[std::make_pair(src_rank, dst_rank)][channel_id][i];
        struct ibv_recv_wr recv_wr = {};
        struct ibv_recv_wr* bad_wr = NULL;
        recv_wr.sg_list = reinterpret_cast<struct ibv_sge*>(
            calloc(1, sizeof(struct ibv_sge)));
        assert_non_null(recv_wr.sg_list);
        int recv_wr_id = ibv_recv_wr_id_map[qp.qp->qp_num]++;
        recv_wr.wr_id = recv_wr_id;
        recv_wr.sg_list = nullptr;
        recv_wr.num_sge = 0;
        recv_wr.next = NULL;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            "create_peer_qp,insert ibv_recv_wr_map elm, qpn %d recv_wr_id %d addr %lu len %d",
            qp.qp->qp_num,
            recv_wr_id,
            qp.src_info.recv_mr.addr,
            qp.src_info.recv_mr.len);
        ret = ibv_post_recv(qp.qp, &recv_wr, &bad_wr);
        assert(ret == 0);
    }
  }
  return true;
}


static int
zeta_util_ifname_to_inet_addr(const char *ifname, in_addr_t *addr) {
    int ret = 0;
    int fd;

	fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd == -1) {
        return -errno;
    }

    if (addr != NULL) {
	    struct ifreq ifr = {};

        strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
        ifr.ifr_addr.sa_family = AF_INET;

        ret = ioctl(fd, SIOCGIFADDR, &ifr);
        if (ret != 0) {
            ret = -errno;
            goto out;
        }
        *addr = ((struct sockaddr_in*)&ifr.ifr_addr)->sin_addr.s_addr;
    }

out:
	close(fd);
    return ret;
}

static void 
ibdev2netdev(const char indev[64], char netdev[64]) {
  if (strncmp(indev, "mlx5_bond_", 10) == 0) {
    strcpy(netdev, "bond");
    strcat(netdev, indev + 10);
  } else {
    strcpy(netdev, indev);
  }
}

int
FlowPhyRdma::ibv_init(void) {
    int nb_dev = 0;
    struct ibv_device **dev_lst = NULL;
    struct ibv_device* ibv_dev;
    in_addr_t src_addr = inet_addr(rank2addr[local_rank].c_str());
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    dev_lst = ibv_get_device_list(&nb_dev);
    assert(nb_dev > 0);
    assert(dev_lst != NULL);

    for (int i = 0; i < nb_dev; ++i) {
        in_addr_t ia = 0;
        char netdev[64];
        ibdev2netdev(dev_lst[i]->name,netdev);
        int ret = zeta_util_ifname_to_inet_addr(netdev, &ia);
        NcclLog->writeLog(
        NcclLogLevel::DEBUG, "netdev %s ibdev %s",netdev,dev_lst[i]->name);
        if (ret == 0 && ia == src_addr) {
            ibv_dev = dev_lst[i];
            break;
        }
    }

    if (ibv_dev == NULL) {
        NcclLog->writeLog(NcclLogLevel::ERROR,"ibv Device not found");
        return -1;
    } else{
        NcclLog->writeLog(
            NcclLogLevel::DEBUG, "ibv device %s init success ", ibv_dev->dev_name);
    }
    g_ibv_ctx = ibv_open_device(ibv_dev);
    assert(g_ibv_ctx != NULL);

    return 0;
}

int
FlowPhyRdma::ibv_fini(void){
    for (auto it = ibv_peer_qps.begin(); it != ibv_peer_qps.end(); it++) {
      for (auto qp_ctxs = it->second.begin(); qp_ctxs != it->second.end(); qp_ctxs++) {
        for (int i = 0; i < qp_ctxs->second.size(); i++) {
            ibv_qp_context qp_ctx = qp_ctxs->second[i];
            if (qp_ctx.qp->recv_cq) {
                ibv_destroy_cq(qp_ctx.qp->recv_cq);
                qp_ctx.qp->recv_cq = NULL;
            }
            if (qp_ctx.qp->send_cq) {
                ibv_destroy_cq(qp_ctx.qp->send_cq);
                qp_ctx.qp->send_cq = NULL;
            }
            if (qp_ctx.qp->pd) {
                ibv_dealloc_pd(qp_ctx.qp->pd);
                qp_ctx.qp->pd = NULL;
            }
            if (qp_ctx.send_buf) {
                free(qp_ctx.send_buf);
                qp_ctx.send_buf = NULL;
            }
            if (qp_ctx.recv_buf) {
                free(qp_ctx.recv_buf);
                qp_ctx.recv_buf = NULL;
            }
            if(qp_ctx.qp){
                ibv_destroy_qp(qp_ctx.qp);
                qp_ctx.qp = NULL;
            }
        }
      }
    }
    return 0;
}