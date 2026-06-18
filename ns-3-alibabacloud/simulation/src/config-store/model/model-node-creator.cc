/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 * 
 * Authors: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
 
#include "model-node-creator.h"
namespace ns3 {

ModelCreator::ModelCreator ()
{
}
void

ModelCreator::Build (GtkTreeStore *treestore)
{
  m_treestore = treestore;
  m_iters.push_back (0);
  //this function will go through all the objects and call on them
  //DoStartVisitObject, DoIterate and DoEndVisitObject
  Iterate ();
  NS_ASSERT (m_iters.size () == 1);
}


void
ModelCreator::Add (ModelNode *node)
{
  GtkTreeIter *parent = m_iters.back ();
  GtkTreeIter *current = g_new (GtkTreeIter, 1);
  gtk_tree_store_append (m_treestore, current, parent);
  gtk_tree_store_set (m_treestore, current,
                      COL_NODE, node, -1);
  m_iters.push_back (current);
}
void
ModelCreator::Remove (void)
{
  GtkTreeIter *iter = m_iters.back ();
  g_free (iter);
  m_iters.pop_back ();
}

void 
ModelCreator::DoVisitAttribute (Ptr<Object> object, std::string name)
{
  ModelNode *node = new ModelNode ();
  node->type = ModelNode::NODE_ATTRIBUTE;
  node->object = object;
  node->name = name;
  Add (node);
  Remove ();
}
void 
ModelCreator::DoStartVisitObject (Ptr<Object> object)
{
  ModelNode *node = new ModelNode ();
  node->type = ModelNode::NODE_OBJECT;
  node->object = object;
  Add (node);
}
void 
ModelCreator::DoEndVisitObject (void)
{
  Remove ();
}
void 
ModelCreator::DoStartVisitPointerAttribute (Ptr<Object> object, std::string name, Ptr<Object> value)
{
  ModelNode *node = new ModelNode ();
  node->type = ModelNode::NODE_POINTER;
  node->object = object;
  node->name = name;
  Add (node);
}
void 
ModelCreator::DoEndVisitPointerAttribute (void)
{
  Remove ();
}
void 
ModelCreator::DoStartVisitArrayAttribute (Ptr<Object> object, std::string name, const ObjectPtrContainerValue &vector)
{
  ModelNode *node = new ModelNode ();
  node->type = ModelNode::NODE_VECTOR;
  node->object = object;
  node->name = name;
  Add (node);
}
void 
ModelCreator::DoEndVisitArrayAttribute (void)
{
  Remove ();
}
void 
ModelCreator::DoStartVisitArrayItem (const ObjectPtrContainerValue &vector, uint32_t index, Ptr<Object> item)
{
  GtkTreeIter *parent = m_iters.back ();
  GtkTreeIter *current = g_new (GtkTreeIter, 1);
  ModelNode *node = new ModelNode ();
  node->type = ModelNode::NODE_VECTOR_ITEM;
  node->object = item;
  node->index = index;
  gtk_tree_store_append (m_treestore, current, parent);
  gtk_tree_store_set (m_treestore, current,
                      COL_NODE, node,
                      -1);
  m_iters.push_back (current);
}
void 
ModelCreator::DoEndVisitArrayItem (void)
{
  GtkTreeIter *iter = m_iters.back ();
  g_free (iter);
  m_iters.pop_back ();
}
} //end namespace ns3
