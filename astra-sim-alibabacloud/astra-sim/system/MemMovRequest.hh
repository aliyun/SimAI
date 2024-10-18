/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MEMMOVREQUEST_HH__
#define __MEMMOVREQUEST_HH__

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
#include "SharedBusStat.hh"

namespace AstraSim {
class Sys;
class LogGP;
class MemMovRequest : public Callable, public SharedBusStat {
 public:
  static int id;
  int my_id;
  int size;
  int latency;
  Callable* callable;
  bool processed;
  bool send_back;
  bool mem_bus_finished;
  Sys* generator;
  EventType callEvent = EventType::General;
  LogGP* loggp;
  std::list<MemMovRequest>::iterator pointer;
  void call(EventType event, CallData* data);

  Tick total_transfer_queue_time;
  Tick total_transfer_time;
  Tick total_processing_queue_time;
  Tick total_processing_time;
  Tick start_time;
  int request_num;
  MemMovRequest(
      int request_num,
      Sys* generator,
      LogGP* loggp,
      int size,
      int latency,
      Callable* callable,
      bool processed,
      bool send_back);
  void wait_wait_for_mem_bus(std::list<MemMovRequest>::iterator pointer);
  void set_iterator(std::list<MemMovRequest>::iterator pointer) {
    this->pointer = pointer;
  }
};
} // namespace AstraSim
#endif
