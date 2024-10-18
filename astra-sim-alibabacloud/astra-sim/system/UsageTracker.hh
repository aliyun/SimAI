/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __USAGETRACKER_HH__
#define __USAGETRACKER_HH__

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
#include "Usage.hh"
#include "astra-sim/workload/CSVWriter.hh"

namespace AstraSim {
class UsageTracker {
 public:
  int levels;
  int current_level;
  Tick last_tick;
  std::list<Usage> usage;
  UsageTracker(int levels);
  void increase_usage();
  void decrease_usage();
  void set_usage(int level);
  void report(CSVWriter* writer, int offset);
  std::list<std::pair<uint64_t, double>> report_percentage(uint64_t cycles);
};
} // namespace AstraSim
#endif
