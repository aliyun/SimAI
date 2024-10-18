/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RENDEZVOUSSENDDATA_HH__
#define __RENDEZVOUSSENDDATA_HH__

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
#include "BasicEventHandlerData.hh"
#include "Common.hh"
#include "SimSendCaller.hh"

namespace AstraSim {
class Sys;
class RendezvousSendData : public BasicEventHandlerData, public MetaData {
 public:
  SimSendCaller* send;
  RendezvousSendData(
      int nodeId,
      Sys* generator,
      void* buffer,
      uint64_t count,
      int type,
      int dst,
      int tag,
      sim_request request,
      void (*msg_handler)(void* fun_arg),
      void* fun_arg);
};
} // namespace AstraSim
#endif
