/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMPLEXLOGICALTOPOLOGY_HH__
#define __COMPLEXLOGICALTOPOLOGY_HH__

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
class ComplexLogicalTopology : public LogicalTopology {
 public:
  ComplexLogicalTopology() {
    this->complexity = LogicalTopology::Complexity::Complex;
  }
  virtual ~ComplexLogicalTopology() = default;
  virtual int get_num_of_nodes_in_dimension(int dimension) override = 0;
  virtual int get_num_of_dimensions() override = 0;
  virtual BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override = 0;
};
} // namespace AstraSim
#endif
