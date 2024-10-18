/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COLLECTIVEPHASE_HH__
#define __COLLECTIVEPHASE_HH__

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
class Sys;
class Algorithm;
class BaseStream;
class CollectivePhase {
 public:
  Sys* generator;
  int queue_id;
  uint64_t initial_data_size;
  uint64_t final_data_size;
  bool enabled;
  ComType comm_type;
  Algorithm* algorithm;
  CollectivePhase(Sys* generator, int queue_id, Algorithm* algorithm);
  CollectivePhase();
  void init(BaseStream* stream);
};
} // namespace AstraSim
#endif
