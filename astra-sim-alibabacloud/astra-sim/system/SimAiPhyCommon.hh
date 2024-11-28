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
#ifndef __SIMAI_PHYCOMMON_HH__
#define __SIMAI_PHYCOMMON_HH__

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

#define assert_non_null(x) assert((x) != NULL)

#define TEST_IO_DEPTH 16
#define MAX_CHILD_FLOW_SIZE 20
#define NCCL_QPS_PER_PEER 1
#define INIT_RECV_WR_NUMS 1024
#define SEND_CHUNK_SIZE 1024*1024
#define WR_NUMS 1

struct mr_info {
    uint64_t addr;
    uint64_t len;
    uint32_t lkey;
    uint32_t rkey;
};

struct TransportData{
  int channel_id;
  int chunk_id;
  int current_flow_id;    
  int child_flow_id;	
  int sender_node;
  int receiver_node;
  uint64_t flow_size;
  void* pQps;
  int tag_id;
  int child_flow_size;
  int child_flow_list[MAX_CHILD_FLOW_SIZE];
  bool nvls_on;
  TransportData(
      int _channel_id,
      int _chunk_id,
      int _current_flow_id,
      int _child_flow_id,
      int _sender_node,
      int _receiver_node,
      uint64_t _flow_size,
      void* _pQps,
      int _tag_id,
      bool _nvls_on)
      : channel_id(_channel_id),
        chunk_id(_chunk_id),
        current_flow_id(_current_flow_id),
        child_flow_id(_child_flow_id),
        sender_node(_sender_node),
        receiver_node(_receiver_node),
        flow_size(_flow_size),
        pQps(_pQps),
        tag_id(_tag_id),
        nvls_on(_nvls_on) {};
  ~TransportData(){};
};

#endif