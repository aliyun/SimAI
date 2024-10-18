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
#ifndef __SIMAI_FLOWMODELIBV_HH__
#define __SIMAI_FLOWMODELIBV_HH__
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <stddef.h>
#include <setjmp.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <assert.h>
#include <getopt.h>
#include<map>
#include<vector>
#include<string>
#include <infiniband/verbs.h>

#include"SimAiPhyCommon.hh"
#include"AstraNetworkAPI.hh"

#define assert_non_null(x) assert((x) != NULL)

struct ibv_hand_shake {
    uint32_t gid_index;
    uint32_t qp_num;
    uint32_t psn;
    uint16_t lid;
    union ibv_gid my_gid;
    struct mr_info recv_mr; 
    struct mr_info send_mr;
};

struct ibv_qp_context
{
    struct ibv_qp *qp;
    int chunk_size;
    int lcore_id;
    struct ibv_mr *send_mr;
    struct ibv_mr *recv_mr;
    struct ibv_hand_shake src_info;
    struct ibv_hand_shake dest_info;
    void* send_buf;
    void* recv_buf;
};


class FlowPhyRdma{
public:
    FlowPhyRdma(){};
    FlowPhyRdma(int _gid_index);
    ~FlowPhyRdma();
    void* send_wr_id_to_buff(int qpn,int wr_id);

    void* recv_wr_id_to_buff(int qpn,int wr_id,int chunk_id);

    bool ibv_create_peer_qp(int rank,int channel,int src_rank,int dst_rank,int chunk_count,int chunk_id,uint64_t buffer_size);

    bool simai_ibv_post_send(int channel_id,int src_rank,int dst_rank, void* send_buf,uint64_t len,uint64_t data_size,int chunk_id);

    bool init_recv_wr(ibv_qp_context qp,int recv_nums);

    bool insert_recv_wr(int qpn);

    int
    ibv_init(void) ;

    int
    ibv_fini(void);
private:
    struct ibv_context *g_ibv_ctx;
    int gid_index;
    std::map<std::pair<int,int>,std::map<int,std::vector<ibv_qp_context>>>ibv_peer_qps;
    std::map<int,ibv_qp_context>qpn2ctx;
    std::map<std::pair<int,int>,struct ibv_send_wr>ibv_send_wr_map;
    std::map<int,int> ibv_recv_wr_id_map;
    std::map<int,int> ibv_send_wr_id_map;
    std::vector<struct ibv_qp_context>ibv_srv_alloc_ctx(int rank,int src_rank,int dst_rank,int channel_id,struct ibv_context *g_ibv_ctx,int chunk_count,uint64_t buffer_size,int qp_nums);
};

#endif