/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __NODE_HH__
#define __NODE_HH__

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
#include "ComputeNode.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class Node : public ComputeNode {
 public:
  int id;
  Node* parent;
  Node* left_child;
  Node* right_child;
  Node(int id, Node* parent, Node* left_child, Node* right_child);
};
} // namespace AstraSim
#endif
