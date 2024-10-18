/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "BinaryTree.hh"
namespace AstraSim {
BinaryTree::~BinaryTree() {
  for (auto n : node_list) {
    delete n.second;
  }
}
BinaryTree::BinaryTree(
    int id,
    TreeType tree_type,
    int total_tree_nodes,
    int start,
    int stride)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::BinaryTree) {
  this->total_tree_nodes = total_tree_nodes;
  this->start = start;
  this->tree_type = tree_type;
  this->stride = stride;
  tree = new Node(-1, nullptr, nullptr, nullptr);
  int depth = 1;
  int tmp = total_tree_nodes;
  while (tmp > 1) {
    depth++;
    tmp /= 2;
  }
  if (tree_type == TreeType::RootMin) {
    tree->right_child = initialize_tree(depth - 1, tree);
  } else {
    tree->left_child = initialize_tree(depth - 1, tree);
  }
  build_tree(tree);
}
Node* BinaryTree::initialize_tree(int depth, Node* parent) {
  Node* tmp = new Node(-1, parent, nullptr, nullptr);
  if (depth > 1) {
    tmp->left_child = initialize_tree(depth - 1, tmp);
    tmp->right_child = initialize_tree(depth - 1, tmp);
  }
  return tmp;
}
void BinaryTree::build_tree(Node* node) {
  if (node->left_child != nullptr) {
    build_tree(node->left_child);
  }
  node->id = start;
  node_list[start] = node;
  start += stride;
  if (node->right_child != nullptr) {
    build_tree(node->right_child);
  }
  return;
}
int BinaryTree::get_parent_id(int id) {
  Node* parent = this->node_list[id]->parent;
  if (parent != nullptr) {
    return parent->id;
  }
  return -1;
}
int BinaryTree::get_right_child_id(int id) {
  Node* child = this->node_list[id]->right_child;
  if (child != nullptr) {
    return child->id;
  }
  return -1;
}
int BinaryTree::get_left_child_id(int id) {
  Node* child = this->node_list[id]->left_child;
  if (child != nullptr) {
    return child->id;
  }
  return -1;
}
BinaryTree::Type BinaryTree::get_node_type(int id) {
  Node* node = this->node_list[id];
  if (node->parent == nullptr) {
    return Type::Root;
  } else if (node->left_child == nullptr && node->right_child == nullptr) {
    return Type::Leaf;
  } else {
    return Type::Intermediate;
  }
}
void BinaryTree::print(Node* node) {
  std::cout << "I am node: " << node->id;
  if (node->left_child != nullptr) {
    std::cout << " and my left child is: " << node->left_child->id;
  }
  if (node->right_child != nullptr) {
    std::cout << " and my right child is: " << node->right_child->id;
  }
  if (node->parent != nullptr) {
    std::cout << " and my parent is: " << node->parent->id;
  }
  BinaryTree::Type typ = get_node_type(node->id);
  if (typ == BinaryTree::Type::Root) {
    std::cout << " and I am Root ";
  } else if (typ == BinaryTree::Type::Intermediate) {
    std::cout << " and I am Intermediate ";
  } else if (typ == BinaryTree::Type::Leaf) {
    std::cout << " and I am Leaf ";
  }
  std::cout << std::endl;
  if (node->left_child != nullptr) {
    print(node->left_child);
  }
  if (node->right_child != nullptr) {
    print(node->right_child);
  }
}
} // namespace AstraSim
