/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOGICALTOPOLOGY_HH__
#define __LOGICALTOPOLOGY_HH__
#include "astra-sim/system/Common.hh"
namespace AstraSim {
class BasicLogicalTopology;
class LogicalTopology {
 public:
  enum class Complexity { Basic, Complex };
  Complexity complexity;
  virtual LogicalTopology* get_topology();
  static int get_reminder(int number, int divisible);
  virtual ~LogicalTopology() = default;
  virtual int get_num_of_dimensions() = 0;
  virtual int get_num_of_nodes_in_dimension(int dimension) = 0;
  virtual BasicLogicalTopology* get_basic_topology_at_dimension(
      int dimension,
      ComType type) = 0;
};
} // namespace AstraSim
#endif
