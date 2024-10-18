/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "LogicalTopology.hh"
namespace AstraSim {
LogicalTopology* LogicalTopology::get_topology() {
  return this;
}
int LogicalTopology::get_reminder(int number, int divisible) {
  if (number >= 0) {
    return number % divisible;
  } else {
    return (number + divisible) % divisible;
  }
}
} // namespace AstraSim
