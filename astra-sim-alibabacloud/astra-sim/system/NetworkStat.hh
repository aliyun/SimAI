/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __NETWORKSTAT_HH__
#define __NETWORKSTAT_HH__

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

namespace AstraSim {
class NetworkStat {
 public:
  std::list<double> net_message_latency;
  int net_message_counter;
  NetworkStat() {
    net_message_counter = 0;
  }
  void update_network_stat(NetworkStat* networkStat) {
    if (net_message_latency.size() < networkStat->net_message_latency.size()) {
      int dif =
          networkStat->net_message_latency.size() - net_message_latency.size();
      for (int i = 0; i < dif; i++) {
        net_message_latency.push_back(0);
      }
    }
    std::list<double>::iterator it = net_message_latency.begin();
    for (auto& ml : networkStat->net_message_latency) {
      (*it) += ml;
      std::advance(it, 1);
    }
    net_message_counter++;
  }
  void take_network_stat_average() {
    for (auto& ml : net_message_latency) {
      ml /= net_message_counter;
    }
  }
};
} // namespace AstraSim
#endif
