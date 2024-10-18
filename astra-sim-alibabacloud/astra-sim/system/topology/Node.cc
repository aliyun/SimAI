/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Node.hh"
namespace AstraSim {
Node::Node(int id, Node* parent, Node* left_child, Node* right_child) {
  this->id = id;
  this->parent = parent;
  this->left_child = left_child;
  this->right_child = right_child;
}
} // namespace AstraSim
