/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "LocalRingNodeA2AGlobalDBT.hh"
namespace AstraSim {
LocalRingNodeA2AGlobalDBT::LocalRingNodeA2AGlobalDBT(
    int id,
    int local_dim,
    int node_dim,
    int total_tree_nodes,
    int start,
    int stride) {
  this->global_dimension_all_reduce =
      new DoubleBinaryTreeTopology(id, total_tree_nodes, start, stride);
  this->global_dimension_other = new RingTopology(
      RingTopology::Dimension::Vertical,
      id,
      total_tree_nodes,
      id / (local_dim * node_dim),
      local_dim * node_dim);
  this->local_dimension = new RingTopology(
      RingTopology::Dimension::Local, id, local_dim, id % local_dim, 1);
  this->node_dimension = new RingTopology(
      RingTopology::Dimension::Horizontal,
      id,
      node_dim,
      (id % (local_dim * node_dim)) / local_dim,
      local_dim);
}
LocalRingNodeA2AGlobalDBT::~LocalRingNodeA2AGlobalDBT() {
  delete global_dimension_all_reduce;
  delete local_dimension;
  delete node_dimension;
  delete global_dimension_other;
}
BasicLogicalTopology* LocalRingNodeA2AGlobalDBT ::
    get_basic_topology_at_dimension(int dimension, ComType type) {
  if (dimension == 0) {
    return local_dimension;
  } else if (dimension == 1) {
    return node_dimension;
  } else if (dimension == 2) {
    if (type == ComType::All_Reduce) {
      return global_dimension_all_reduce->get_basic_topology_at_dimension(
          2, type);
    } else {
      return global_dimension_other;
    }
  }
  return nullptr;
}
int LocalRingNodeA2AGlobalDBT::get_num_of_nodes_in_dimension(int dimension) {
  if (dimension == 0) {
    return local_dimension->get_num_of_nodes_in_dimension(0);
  } else if (dimension == 1) {
    return node_dimension->get_num_of_nodes_in_dimension(0);
  } else if (dimension == 2) {
    return global_dimension_other->get_num_of_nodes_in_dimension(0);
  }
  return -1;
}
int LocalRingNodeA2AGlobalDBT::get_num_of_dimensions() {
  return 3;
}
} // namespace AstraSim
