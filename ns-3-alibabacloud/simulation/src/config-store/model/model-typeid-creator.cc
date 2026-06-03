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
 * Author: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 */

#include "model-typeid-creator.h"

namespace ns3 {

ModelTypeidCreator::ModelTypeidCreator ()
{
}
void

ModelTypeidCreator::Build (GtkTreeStore *treestore)
{
  m_treestore = treestore;
  m_iters.push_back (0);
  Iterate ();
  NS_ASSERT (m_iters.size () == 1);
}

void
ModelTypeidCreator::Add (ModelTypeid *node)
{
  GtkTreeIter *parent = m_iters.back ();
  GtkTreeIter *current = g_new (GtkTreeIter, 1);
  gtk_tree_store_append (m_treestore, current, parent);
  gtk_tree_store_set (m_treestore, current, COL_TYPEID, node, -1);
  m_iters.push_back (current);
}

void
ModelTypeidCreator::Remove (void)
{
  GtkTreeIter *iter = m_iters.back ();
  g_free (iter);
  m_iters.pop_back ();
}

void 
ModelTypeidCreator::VisitAttribute (TypeId tid, std::string name, std::string defaultValue, uint32_t index)
{
  ModelTypeid *node = new ModelTypeid ();
  node->type = ModelTypeid::NODE_ATTRIBUTE;
  node->tid = tid;
  node->name = name;
  node->defaultValue = defaultValue;
  node->index = index;
  Add (node);
  Remove ();
}

void 
ModelTypeidCreator::StartVisitTypeId (std::string name)
{
  ModelTypeid *node = new ModelTypeid ();
  node->type = ModelTypeid::NODE_TYPEID;
  node->tid = TypeId::LookupByName (name);
  Add (node);
}

void 
ModelTypeidCreator::EndVisitTypeId (void)
{
  Remove ();
}
} //end namespace ns3
