/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __BINARYTREE_HH__
#define __BINARYTREE_HH__

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
#include "BasicLogicalTopology.hh"
#include "Node.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {
class BinaryTree : public BasicLogicalTopology {
 public:
  enum class TreeType { RootMax, RootMin };
  enum class Type { Leaf, Root, Intermediate };
  int total_tree_nodes;
  int start;
  TreeType tree_type;
  int stride;
  Node* tree;
  virtual ~BinaryTree();
  std::map<int, Node*> node_list;
  int get_num_of_nodes_in_dimension(int dimension) override {
    return total_tree_nodes;
  }
  BinaryTree(
      int id,
      TreeType tree_type,
      int total_tree_nodes,
      int start,
      int stride);
  Node* initialize_tree(int depth, Node* parent);
  void build_tree(Node* node);
  int get_parent_id(int id);
  int get_left_child_id(int id);
  int get_right_child_id(int id);
  Type get_node_type(int id);
  bool is_enabled(int id) {
    if (id % stride == 0) {
      return true;
    }
    return false;
  };
  void print(Node* node);
};
} // namespace AstraSim
#endif
