/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __QUEUELEVELHANDLER_HH__
#define __QUEUELEVELHANDLER_HH__

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
#include "AstraNetworkAPI.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {
class QueueLevelHandler {
 public:
  std::vector<int> queues;
  int allocator;
  int first_allocator;
  int last_allocator;
  int level;
  AstraNetworkAPI::BackendType backend;
  QueueLevelHandler(
      int level,
      int start,
      int end,
      AstraNetworkAPI::BackendType backend);
  std::pair<int, RingTopology::Direction> get_next_queue_id();
  std::pair<int, RingTopology::Direction> get_next_queue_id_first();
  std::pair<int, RingTopology::Direction> get_next_queue_id_last();
};
} // namespace AstraSim
#endif
