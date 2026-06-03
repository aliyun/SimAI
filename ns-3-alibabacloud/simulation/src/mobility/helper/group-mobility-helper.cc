/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2021 University of Washington: Group mobility changes
 *
 * This program is free software; you can redistribute it and/or modify
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Adapted from 'mobility-helper.cc' for group mobility by Tom Henderson
 */

#include "ns3/group-mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/position-allocator.h"
#include "ns3/hierarchical-mobility-model.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/config.h"
#include "ns3/simulator.h"
#include "ns3/names.h"
#include "ns3/string.h"
#include <iostream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GroupMobilityHelper");

GroupMobilityHelper::GroupMobilityHelper ()
{
}

GroupMobilityHelper::~GroupMobilityHelper ()
{
}

void
GroupMobilityHelper::SetReferencePositionAllocator (Ptr<PositionAllocator> allocator)
{
  m_referencePosition = allocator;
}

void
GroupMobilityHelper::SetReferencePositionAllocator (std::string type,
                                      std::string n1, const AttributeValue &v1,
                                      std::string n2, const AttributeValue &v2,
                                      std::string n3, const AttributeValue &v3,
                                      std::string n4, const AttributeValue &v4,
                                      std::string n5, const AttributeValue &v5,
                                      std::string n6, const AttributeValue &v6,
                                      std::string n7, const AttributeValue &v7,
                                      std::string n8, const AttributeValue &v8,
                                      std::string n9, const AttributeValue &v9)
{
  ObjectFactory pos;
  pos.SetTypeId (type);
  pos.Set (n1, v1);
  pos.Set (n2, v2);
  pos.Set (n3, v3);
  pos.Set (n4, v4);
  pos.Set (n5, v5);
  pos.Set (n6, v6);
  pos.Set (n7, v7);
  pos.Set (n8, v8);
  pos.Set (n9, v9);
  m_referencePosition = pos.Create ()->GetObject<PositionAllocator> ();
  NS_ABORT_MSG_IF (m_referencePosition == nullptr, "Unable to create allocator from TypeId " << type);
}

void
GroupMobilityHelper::SetMemberPositionAllocator (Ptr<PositionAllocator> allocator)
{
  m_memberPosition = allocator;
}

void
GroupMobilityHelper::SetMemberPositionAllocator (std::string type,
                                      std::string n1, const AttributeValue &v1,
                                      std::string n2, const AttributeValue &v2,
                                      std::string n3, const AttributeValue &v3,
                                      std::string n4, const AttributeValue &v4,
                                      std::string n5, const AttributeValue &v5,
                                      std::string n6, const AttributeValue &v6,
                                      std::string n7, const AttributeValue &v7,
                                      std::string n8, const AttributeValue &v8,
                                      std::string n9, const AttributeValue &v9)
{
  ObjectFactory pos;
  pos.SetTypeId (type);
  pos.Set (n1, v1);
  pos.Set (n2, v2);
  pos.Set (n3, v3);
  pos.Set (n4, v4);
  pos.Set (n5, v5);
  pos.Set (n6, v6);
  pos.Set (n7, v7);
  pos.Set (n8, v8);
  pos.Set (n9, v9);
  m_memberPosition = pos.Create ()->GetObject<PositionAllocator> ();
  NS_ABORT_MSG_IF (m_memberPosition == nullptr, "Unable to create allocator from TypeId " << type);
}

void
GroupMobilityHelper::SetReferenceMobilityModel (Ptr<MobilityModel> mobility)
{
  m_referenceMobility = mobility;
}

void 
GroupMobilityHelper::SetReferenceMobilityModel (std::string type,
                                  std::string n1, const AttributeValue &v1,
                                  std::string n2, const AttributeValue &v2,
                                  std::string n3, const AttributeValue &v3,
                                  std::string n4, const AttributeValue &v4,
                                  std::string n5, const AttributeValue &v5,
                                  std::string n6, const AttributeValue &v6,
                                  std::string n7, const AttributeValue &v7,
                                  std::string n8, const AttributeValue &v8,
                                  std::string n9, const AttributeValue &v9)
{
  NS_LOG_FUNCTION (this << type);
  ObjectFactory mob;
  mob.SetTypeId (type);
  mob.Set (n1, v1);
  mob.Set (n2, v2);
  mob.Set (n3, v3);
  mob.Set (n4, v4);
  mob.Set (n5, v5);
  mob.Set (n6, v6);
  mob.Set (n7, v7);
  mob.Set (n8, v8);
  mob.Set (n9, v9);
  m_referenceMobility = mob.Create ()->GetObject<MobilityModel> ();
  NS_ABORT_MSG_IF (m_referenceMobility == nullptr, "Unable to create mobility from TypeId " << type);
}

void 
GroupMobilityHelper::SetMemberMobilityModel (std::string type,
                                  std::string n1, const AttributeValue &v1,
                                  std::string n2, const AttributeValue &v2,
                                  std::string n3, const AttributeValue &v3,
                                  std::string n4, const AttributeValue &v4,
                                  std::string n5, const AttributeValue &v5,
                                  std::string n6, const AttributeValue &v6,
                                  std::string n7, const AttributeValue &v7,
                                  std::string n8, const AttributeValue &v8,
                                  std::string n9, const AttributeValue &v9)
{
  NS_LOG_FUNCTION (this << type);
  m_memberMobilityFactory.SetTypeId (type);
  m_memberMobilityFactory.Set (n1, v1);
  m_memberMobilityFactory.Set (n2, v2);
  m_memberMobilityFactory.Set (n3, v3);
  m_memberMobilityFactory.Set (n4, v4);
  m_memberMobilityFactory.Set (n5, v5);
  m_memberMobilityFactory.Set (n6, v6);
  m_memberMobilityFactory.Set (n7, v7);
  m_memberMobilityFactory.Set (n8, v8);
  m_memberMobilityFactory.Set (n9, v9);
}

void
GroupMobilityHelper::Install (Ptr<Node> node)
{
  NS_ABORT_MSG_IF (node->GetObject<MobilityModel> () != nullptr, "Mobility model already installed");
  NS_ABORT_MSG_IF (m_referenceMobility == nullptr, "Reference mobility model is empty");
  NS_ABORT_MSG_UNLESS (m_memberMobilityFactory.IsTypeIdSet (), "Member mobility factory is unset");
  if (m_referencePosition && !m_referencePositionSet)
    {
      Vector referencePosition = m_referencePosition->GetNext ();
      m_referenceMobility->SetPosition (referencePosition);
      m_referencePositionSet = true;
    }
  Ptr<HierarchicalMobilityModel> hierarchical = CreateObject<HierarchicalMobilityModel> ();
  hierarchical->SetParent (m_referenceMobility);
  Ptr<MobilityModel> child = m_memberMobilityFactory.Create ()->GetObject<MobilityModel> ();
  NS_ABORT_MSG_IF (child == nullptr, "Member mobility factory did not produce a MobilityModel");
  if (m_memberPosition)
    {
      Vector position = m_memberPosition->GetNext ();
      child->SetPosition (position);
    }
  hierarchical->SetChild (child);
  NS_LOG_DEBUG ("node="<<node<<", mob="<<hierarchical);
  node->AggregateObject (hierarchical);
}

void
GroupMobilityHelper::Install (std::string nodeName)
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  Install (node);
}

void 
GroupMobilityHelper::Install (NodeContainer c)
{
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Install (*i);
    }
}
int64_t
GroupMobilityHelper::AssignStreams (NodeContainer c, int64_t stream)
{
  int64_t currentStream = stream;
  Ptr<Node> node;
  bool firstNode = true;
  Ptr<HierarchicalMobilityModel> mobility;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      node = (*i);
      mobility = node->GetObject<HierarchicalMobilityModel> ();
      if (!mobility)
        {
          NS_FATAL_ERROR ("Did not find a HierarchicalMobilityModel");
        }
      if (firstNode)
        {
          // Assign streams only once for the reference node
          currentStream += mobility->GetParent ()->AssignStreams (currentStream);
          firstNode = false;
        }
      currentStream += mobility->GetChild ()->AssignStreams (currentStream);
    }
  return (currentStream - stream);
}

} // namespace ns3
