/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __DMAREQUEST_HH__
#define __DMAREQUEST_HH__

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
class DMA_Request {
 public:
  int id;
  int slots;
  int latency;
  bool executed;
  int bytes;
  Callable* stream_owner;
  DMA_Request(int id, int slots, int latency, int bytes);
  DMA_Request(
      int id,
      int slots,
      int latency,
      int bytes,
      Callable* stream_owner);
};
} // namespace AstraSim
#endif
