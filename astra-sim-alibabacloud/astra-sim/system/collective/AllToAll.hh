/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ALLTOALL_HH__
#define __ALLTOALL_HH__

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
#include "Ring.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {
class AllToAll : public Ring {
 public:
  AllToAll(
      ComType type,
      int window,
      int id,
      int layer_num,
      RingTopology* allToAllTopology,
      uint64_t data_size,
      RingTopology::Direction direction,
      InjectionPolicy injection_policy,
      bool boost_mode);
  void run(EventType event, CallData* data);
  void process_max_count();
  int get_non_zero_latency_packets();
  int middle_point;
};
} // namespace AstraSim
#endif
