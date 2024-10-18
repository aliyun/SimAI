/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __GENERALCOMPLEXTOPOLOGY_HH__
#define __GENERALCOMPLEXTOPOLOGY_HH__

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
#include "ComplexLogicalTopology.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class GeneralComplexTopology : public ComplexLogicalTopology {
 public:
  std::vector<LogicalTopology*> dimension_topology;
  GeneralComplexTopology(
      int id,
      std::vector<int> dimension_size,
      std::vector<CollectiveImplementation*> collective_implementation);
  ~GeneralComplexTopology();
  int get_num_of_nodes_in_dimension(int dimension) override;
  BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override;
  int get_num_of_dimensions() override;
};
} // namespace AstraSim
#endif
