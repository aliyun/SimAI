/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __QUEUELEVELS_HH__
#define __QUEUELEVELS_HH__

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
#include "QueueLevelHandler.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {
class QueueLevels {
 public:
  std::vector<QueueLevelHandler> levels;
  std::pair<int, RingTopology::Direction> get_next_queue_at_level(int level);
  std::pair<int, RingTopology::Direction> get_next_queue_at_level_first(
      int level);
  std::pair<int, RingTopology::Direction> get_next_queue_at_level_last(
      int level);
  QueueLevels(
      int levels,
      int queues_per_level,
      int offset,
      AstraNetworkAPI::BackendType backend);
  QueueLevels(
      std::vector<int> lv,
      int offset,
      AstraNetworkAPI::BackendType backend);
};
} // namespace AstraSim
#endif
