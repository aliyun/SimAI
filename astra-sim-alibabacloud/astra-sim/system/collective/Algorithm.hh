/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ALGORITHM_HH__
#define __ALGORITHM_HH__

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
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/topology/LogicalTopology.hh"

namespace AstraSim {
class Algorithm : public Callable {
 public:
  enum class Name { Ring, DoubleBinaryTree, AllToAll, HalvingDoubling };
  Name name;
  int id;
  BaseStream* stream;
  LogicalTopology* logicalTopology;
  uint64_t data_size;
  uint64_t final_data_size;
  ComType comType;
  bool enabled;
  int layer_num;
  Algorithm(){};
  Algorithm(int layer_num);
  virtual ~Algorithm() = default;
  virtual void run(EventType event, CallData* data) = 0;
  virtual void exit();
  virtual void init(BaseStream* stream);
  virtual void call(EventType event, CallData* data);
};
} // namespace AstraSim
#endif
