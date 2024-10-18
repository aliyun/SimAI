/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __BASICLOGICALTOPOLOGY_HH__
#define __BASICLOGICALTOPOLOGY_HH__

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
#include "LogicalTopology.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class BasicLogicalTopology : public LogicalTopology {
 public:
  enum class BasicTopology { Ring, BinaryTree };
  BasicTopology basic_topology;
  BasicLogicalTopology(BasicTopology basic_topology) {
    this->basic_topology = basic_topology;
    this->complexity = LogicalTopology::Complexity::Basic;
  }
  virtual ~BasicLogicalTopology() = default;
  int get_num_of_dimensions() override {
    return 1;
  };
  virtual int get_num_of_nodes_in_dimension(int dimension) override = 0;
  BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override {
    return this;
  }
};
} // namespace AstraSim
#endif
