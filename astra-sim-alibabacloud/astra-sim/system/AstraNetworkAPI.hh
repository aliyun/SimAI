/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ASTRANETWORKAPI_HH__
#define __ASTRANETWORKAPI_HH__
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <queue>
#include <map>
#include "AstraMemoryAPI.hh"
#include "AstraSimDataAPI.hh"
namespace AstraSim {
struct sim_comm {
  std::string comm_name;
};

enum time_type_e { SE, MS, US, NS, FS };

struct timespec_t {
  time_type_e time_res;
  double time_val;
};
enum req_type_e { UINT8, BFLOAT16, FP32 };

struct ncclFlowTag {
  int channel_id;
  int chunk_id;
  int current_flow_id;    
  int child_flow_id;	
  int sender_node;
  int receiver_node;
  uint64_t flow_size;
  void* pQps;
  int tag_id; 
  std::vector<int> tree_flow_list;
  bool nvls_on;
  ncclFlowTag():
    channel_id(0),
    chunk_id(0),
    current_flow_id(0),
    child_flow_id(0),
    sender_node(0),
    receiver_node(0),
    flow_size(0),
    pQps(nullptr),
    tag_id(0),
    nvls_on(false){};
  ncclFlowTag(
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
  ~ncclFlowTag() {};
};


struct sim_request {
  uint32_t srcRank;
  uint32_t dstRank;
  uint32_t tag;
  req_type_e reqType;
  uint64_t reqCount;
  uint32_t vnet;
  uint32_t layerNum;
  ncclFlowTag flowTag;
};


class MetaData {
 public:
  timespec_t timestamp;
};

class AstraNetworkAPI {
 public:
  enum class BackendType { NotSpecified, Garnet, NS3, Analytical };
  bool enabled;

  virtual BackendType get_backend_type() {
    return BackendType::NotSpecified;
  };
  virtual int sim_comm_size(sim_comm comm, int* size) = 0;
  virtual int sim_comm_get_rank() {
    return rank;
  };
  virtual int sim_comm_set_rank(int rank) {
    this->rank = rank;
    return this->rank;
  };
  virtual int sim_finish() = 0;
  virtual double sim_time_resolution() = 0;
  virtual int sim_init(AstraMemoryAPI* MEM) = 0;
  virtual timespec_t sim_get_time() = 0;
  virtual void sim_schedule(
      timespec_t delta,
      void (*fun_ptr)(void* fun_arg),
      void* fun_arg) = 0;
  virtual int sim_send(
      void* buffer,
      uint64_t count,
      int type,
      int dst,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg) = 0;
  virtual int sim_recv(
      void* buffer,
      uint64_t count,
      int type,
      int src,
      int tag,
      sim_request* request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg) = 0;
  virtual void pass_front_end_report(AstraSimDataAPI astraSimDataAPI) {
    return;
  };

  virtual double get_BW_at_dimension(int dim) {
    return -1;
  };
  AstraNetworkAPI(int rank) {
    this->rank = rank;
    enabled = true;
  };
  virtual ~AstraNetworkAPI(){}; 

  protected:
    int rank;
};
} // namespace AstraSim
#endif
