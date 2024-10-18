/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __PACKETBUNDLE_HH__
#define __PACKETBUNDLE_HH__

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
#include "BaseStream.hh"
#include "Callable.hh"
#include "Common.hh"
#include "MemBus.hh"
#include "MyPacket.hh"

namespace AstraSim {
class Sys;
class PacketBundle : public Callable {
 public:
  std::list<MyPacket*> locked_packets;
  bool needs_processing;
  bool send_back;
  uint64_t size;
  Sys* generator;
  BaseStream* stream;
  Tick creation_time;
  MemBus::Transmition transmition;
  int channel_id;
  int flow_id;
  PacketBundle(
      Sys* generator,
      BaseStream* stream,
      std::list<MyPacket*> locked_packets,
      bool needs_processing,
      bool send_back,
      uint64_t size,
      MemBus::Transmition transmition,
      int channel_id,
      int flow_id);
  PacketBundle(
      Sys* generator,
      BaseStream* stream,
      std::list<MyPacket*> locked_packets,
      bool needs_processing,
      bool send_back,
      uint64_t size,
      MemBus::Transmition transmition);
  PacketBundle(
      Sys* generator,
      BaseStream* stream,
      bool needs_processing,
      bool send_back,
      uint64_t size,
      MemBus::Transmition transmition);
  void send_to_MA();
  void send_to_NPU();
  Tick delay;
  void call(EventType event, CallData* data);
  //~PacketBundle()= default;
};
} // namespace AstraSim
#endif
