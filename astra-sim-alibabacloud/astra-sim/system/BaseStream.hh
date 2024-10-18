/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __BASESTREAM_HH__
#define __BASESTREAM_HH__

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
#include "CollectivePhase.hh"
#include "Common.hh"
#include "DataSet.hh"
#include "StreamStat.hh"
#include "Sys.hh"
#include "astra-sim/system/topology/LogicalTopology.hh"

namespace AstraSim {
class RecvPacketEventHadndlerData;
class SendPacketEventHandlerData;
class BaseStream : public Callable, public StreamStat {
 public:
  virtual ~BaseStream() = default;
  int stream_num;
  int total_packets_sent;
  SchedulingPolicy preferred_scheduling;
  std::list<CollectivePhase> phases_to_go;
  int current_queue_id;
  CollectivePhase my_current_phase;
  ComType current_com_type;
  Tick creation_time;
  Tick last_init;
  Sys* owner;
  DataSet* dataset;
  int steps_finished;
  int initial_data_size;
  int priority;
  StreamState state;
  bool initialized;

  Tick last_phase_change;

  int test;
  int test2;
  uint64_t phase_latencies[10];

  void changeState(StreamState state);
  virtual void consume(RecvPacketEventHadndlerData* message) = 0;
  virtual void sendcallback(SendPacketEventHandlerData* messages)=0;
  virtual void init() = 0;

  BaseStream(
      int stream_num,
      Sys* owner,
      std::list<CollectivePhase> phases_to_go);
};
} // namespace AstraSim
#endif
