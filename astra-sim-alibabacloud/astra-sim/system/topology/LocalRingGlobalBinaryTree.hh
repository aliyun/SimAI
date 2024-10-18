/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCALRINGGLOBALBINARYTREE_HH__
#define __LOCALRINGGLOBALBINARYTREE_HH__

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
#include "BinaryTree.hh"
#include "ComplexLogicalTopology.hh"
#include "RingTopology.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class LocalRingGlobalBinaryTree : public ComplexLogicalTopology {
 public:
  RingTopology* local_dimension;
  RingTopology* global_dimension_other;
  BinaryTree* global_dimension_all_reduce;
  int get_num_of_nodes_in_dimension(int dimension) override;
  int get_num_of_dimensions() override;
  BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) override;
  LocalRingGlobalBinaryTree(
      int id,
      int local_dim,
      BinaryTree::TreeType tree_type,
      int total_tree_nodes,
      int start,
      int stride);
  ~LocalRingGlobalBinaryTree();
};
} // namespace AstraSim
#endif
