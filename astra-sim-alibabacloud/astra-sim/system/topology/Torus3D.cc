/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Torus3D.hh"
namespace AstraSim {
Torus3D::Torus3D(int id, int total_nodes, int local_dim, int vertical_dim) {
  int horizontal_dim = total_nodes / (vertical_dim * local_dim);
  local_dimension = new RingTopology(
      RingTopology::Dimension::Local, id, local_dim, id % local_dim, 1);
  vertical_dimension = new RingTopology(
      RingTopology::Dimension::Vertical,
      id,
      vertical_dim,
      id / (local_dim * horizontal_dim),
      local_dim * horizontal_dim);
  horizontal_dimension = new RingTopology(
      RingTopology::Dimension::Horizontal,
      id,
      horizontal_dim,
      (id / local_dim) % horizontal_dim,
      local_dim);
}
Torus3D::~Torus3D() {
  delete local_dimension;
  delete vertical_dimension;
  delete horizontal_dimension;
};
int Torus3D::get_num_of_dimensions() {
  return 3;
}
int Torus3D::get_num_of_nodes_in_dimension(int dimension) {
  if (dimension == 0) {
    return local_dimension->get_num_of_nodes_in_dimension(0);
  } else if (dimension == 1) {
    return vertical_dimension->get_num_of_nodes_in_dimension(0);
  } else if (dimension == 2) {
    return horizontal_dimension->get_num_of_nodes_in_dimension(0);
  }
  return -1;
}
BasicLogicalTopology* Torus3D::get_basic_topology_at_dimension(
    int dimension,
    ComType type) {
  if (dimension == 0) {
    return local_dimension;
  } else if (dimension == 1) {
    return vertical_dimension;
  } else if (dimension == 2) {
    return horizontal_dimension;
  }
  return NULL;
}
} // namespace AstraSim
