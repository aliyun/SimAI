/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STREAMSTAT_HH__
#define __STREAMSTAT_HH__

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
#include "Common.hh"
#include "NetworkStat.hh"
#include "SharedBusStat.hh"

namespace AstraSim {
class StreamStat : public SharedBusStat, public NetworkStat {
 public:
  std::list<double> queuing_delay;
  int stream_stat_counter;
  //~StreamStat()= default;
  StreamStat() : SharedBusStat(BusType::Shared, 0, 0, 0, 0) {
    stream_stat_counter = 0;
  }
  void update_stream_stats(StreamStat* streamStat) {
    update_bus_stats(BusType::Both, streamStat);
    update_network_stat(streamStat);
    if (queuing_delay.size() < streamStat->queuing_delay.size()) {
      int dif = streamStat->queuing_delay.size() - queuing_delay.size();
      for (int i = 0; i < dif; i++) {
        queuing_delay.push_back(0);
      }
    }
    std::list<double>::iterator it = queuing_delay.begin();
    for (auto& tick : streamStat->queuing_delay) {
      (*it) += tick;
      std::advance(it, 1);
    }
    stream_stat_counter++;
  }
  void take_stream_stats_average() {
    take_bus_stats_average();
    take_network_stat_average();
    for (auto& tick : queuing_delay) {
      tick /= stream_stat_counter;
    }
  }
};
} // namespace AstraSim
#endif
