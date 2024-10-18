/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MYPACKET_HH__
#define __MYPACKET_HH__

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>
#include "Callable.hh"
#include "Common.hh"

namespace AstraSim {
class MyPacket : public Callable {
 public:
  int cycles_needed;
  // FIFOMovement *fMovement;
  // FIFO* dest;
  int fm_id;
  int stream_num;
  Callable* notifier;
  Callable* sender;
  int preferred_vnet;
  int preferred_dest;
  int preferred_src;
  uint64_t msg_size;
  Tick ready_time;
  // flow model
  int flow_id;
  int parent_flow_id;
  int child_flow_id;
  int channel_id;
  int chunk_id;
  // MyPacket(int cycles_needed, FIFOMovement *fMovement, FIFO *dest);
  MyPacket(int preferred_vnet, int preferred_src, int preferred_dest);
  MyPacket(
      int preferred_vnet,
      int preferred_src,
      int preferred_dest,
      uint64_t msg_size,
      int channel_id,
      int flow_id);
  MyPacket(){};
  MyPacket(
      uint64_t msg_size,
      int preferred_vnet,
      int preferred_src,
      int preferred_dest);
  void set_flow_id(int flow_id);
  void set_notifier(Callable* c);
  void call(EventType event, CallData* data);
  //~MyPacket()= default;
};
} // namespace AstraSim
#endif
