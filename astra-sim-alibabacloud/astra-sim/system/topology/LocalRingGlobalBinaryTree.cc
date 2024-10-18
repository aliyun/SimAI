/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "LocalRingGlobalBinaryTree.hh"
namespace AstraSim {
LocalRingGlobalBinaryTree::LocalRingGlobalBinaryTree(
    int id,
    int local_dim,
    BinaryTree::TreeType tree_type,
    int total_tree_nodes,
    int start,
    int stride) {
  this->local_dimension = new RingTopology(
      RingTopology::Dimension::Local, id, local_dim, id % local_dim, 1);
  this->global_dimension_all_reduce =
      new BinaryTree(id, tree_type, total_tree_nodes, start, stride);
  this->global_dimension_other = new RingTopology(
      RingTopology::Dimension::Horizontal,
      id,
      total_tree_nodes,
      id / local_dim,
      local_dim);
}
LocalRingGlobalBinaryTree::~LocalRingGlobalBinaryTree() {
  delete local_dimension;
  delete global_dimension_all_reduce;
  delete global_dimension_other;
}
int LocalRingGlobalBinaryTree::get_num_of_nodes_in_dimension(int dimension) {
  if (dimension == 0) {
    return local_dimension->get_num_of_nodes_in_dimension(0);
  } else if (dimension == 1) {
    return 1;
  } else if (dimension == 2) {
    return global_dimension_all_reduce->get_num_of_nodes_in_dimension(0);
  } else {
    return -1;
  }
}
int LocalRingGlobalBinaryTree::get_num_of_dimensions() {
  return 3;
}
BasicLogicalTopology* LocalRingGlobalBinaryTree::
    get_basic_topology_at_dimension(int dimension, ComType type) {
  if (dimension == 0) {
    return local_dimension;
  } else if (dimension == 1) {
    return nullptr;
  } else if (dimension == 2) {
    if (type == ComType::All_Reduce) {
      return global_dimension_all_reduce;
    }
    return global_dimension_other;
  } else {
    return nullptr;
  }
}
} // namespace AstraSim
