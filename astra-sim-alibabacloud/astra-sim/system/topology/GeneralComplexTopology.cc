/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "GeneralComplexTopology.hh"
#include "DoubleBinaryTreeTopology.hh"
#include "RingTopology.hh"

namespace AstraSim {
BasicLogicalTopology* GeneralComplexTopology::get_basic_topology_at_dimension(
    int dimension,
    ComType type) {
  return dimension_topology[dimension]->get_basic_topology_at_dimension(
      0, type);
}
int GeneralComplexTopology::get_num_of_nodes_in_dimension(int dimension) {
  if (dimension >= dimension_topology.size()) {
    std::cout << "dim: " << dimension
              << " requested! but max dim is: " << dimension_topology.size() - 1
              << std::endl;
  }
  assert(dimension < dimension_topology.size());
  return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}
int GeneralComplexTopology::get_num_of_dimensions() {
  return dimension_topology.size();
}
GeneralComplexTopology::~GeneralComplexTopology() {
  for (int i = 0; i < dimension_topology.size(); i++) {
    delete dimension_topology[i];
  }
}
GeneralComplexTopology::GeneralComplexTopology(
    int id,
    std::vector<int> dimension_size,
    std::vector<CollectiveImplementation*> collective_implementation) {
  int offset = 1;
  int last_dim = collective_implementation.size() - 1;
  for (int dim = 0; dim < collective_implementation.size(); dim++) {
    if (collective_implementation[dim]->type ==
            CollectiveImplementationType::Ring ||
        collective_implementation[dim]->type ==
            CollectiveImplementationType::Direct ||
        collective_implementation[dim]->type ==
            CollectiveImplementationType::HalvingDoubling ||
        collective_implementation[dim]->type ==
            CollectiveImplementationType::NcclRingFlowModel || 
        collective_implementation[dim]->type ==
            CollectiveImplementationType::NcclTreeFlowModel) {
      RingTopology* ring = new RingTopology(
          RingTopology::Dimension::NA,
          id,
          dimension_size[dim],
          (id % (offset * dimension_size[dim])) / offset,
          offset);
      dimension_topology.push_back(ring);
    } else if (
        collective_implementation[dim]->type ==
            CollectiveImplementationType::OneRing ||
        collective_implementation[dim]->type ==
            CollectiveImplementationType::OneDirect ||
        collective_implementation[dim]->type ==
            CollectiveImplementationType::OneHalvingDoubling) {
      int total_npus = 1;
      for (int d : dimension_size) {
        total_npus *= d;
      }
      RingTopology* ring = new RingTopology(
          RingTopology::Dimension::NA, id, total_npus, id % total_npus, 1);
      dimension_topology.push_back(ring);
      return;
    } else if (
        collective_implementation[dim]->type ==
        CollectiveImplementationType::DoubleBinaryTree) {
      if (dim == last_dim) {
        DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
            id, dimension_size[dim], id % offset, offset);
        dimension_topology.push_back(DBT);
      } else {
        DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
            id,
            dimension_size[dim],
            (id - (id % (offset * dimension_size[dim]))) + (id % offset),
            offset);
        dimension_topology.push_back(DBT);
      }
    }
    offset *= dimension_size[dim];
  }
}
} // namespace AstraSim
