/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SIMRECVCALLER_HH__
#define __SIMRECVCALLER_HH__

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
#include "CallData.hh"
#include "Callable.hh"
#include "Common.hh"

namespace AstraSim {
class Sys;
class SimRecvCaller : public Callable {
 public:
  void* buffer;
  uint64_t count;
  int type;
  int src;
  int tag;
  sim_request request;
  void (*msg_handler)(void* fun_arg);
  void* fun_arg;
  void call(EventType type, CallData* data);
  Sys* generator;
  SimRecvCaller(
      Sys* generator,
      void* buffer,
      uint64_t count,
      int type,
      int src,
      int tag,
      sim_request request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
};
} // namespace AstraSim
#endif
