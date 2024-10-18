/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCALRINGNODEA2AGLOBALDBT_HH__
#define __LOCALRINGNODEA2AGLOBALDBT_HH__

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
#include "DoubleBinaryTreeTopology.hh"
#include "RingTopology.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class LocalRingNodeA2AGlobalDBT : public ComplexLogicalTopology {
 public:
  DoubleBinaryTreeTopology* global_dimension_all_reduce;
  RingTopology* global_dimension_other;
  RingTopology* local_dimension;
  RingTopology* node_dimension;
  LocalRingNodeA2AGlobalDBT(
      int id,
      int local_dim,
      int node_dim,
      int total_tree_nodes,
      int start,
      int stride);
  ~LocalRingNodeA2AGlobalDBT();
  int get_num_of_nodes_in_dimension(int dimension) override;
  BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override;
  int get_num_of_dimensions() override;
};
} // namespace AstraSim
#endif
