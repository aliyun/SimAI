/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 */
#include "hierarchical-mobility-model.h"
#include "ns3/pointer.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HierarchicalMobilityModel");

NS_OBJECT_ENSURE_REGISTERED (HierarchicalMobilityModel);

TypeId 
HierarchicalMobilityModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::HierarchicalMobilityModel")
    .SetParent<MobilityModel> ()
    .SetGroupName ("Mobility")
    .AddConstructor<HierarchicalMobilityModel> ()
    .AddAttribute ("Child", "The child mobility model.",
                   PointerValue (),
                   MakePointerAccessor (&HierarchicalMobilityModel::SetChild,
                                        &HierarchicalMobilityModel::GetChild),
                   MakePointerChecker<MobilityModel> ())
    .AddAttribute ("Parent", "The parent mobility model.",
                   PointerValue (),
                   MakePointerAccessor (&HierarchicalMobilityModel::SetParent,
                                        &HierarchicalMobilityModel::GetParent),
                   MakePointerChecker<MobilityModel> ())
  ;
  return tid;
}

HierarchicalMobilityModel::HierarchicalMobilityModel ()
  : m_child (0),
    m_parent (0)
{
  NS_LOG_FUNCTION (this);
}

void
HierarchicalMobilityModel::SetChild (Ptr<MobilityModel> model)
{
  NS_LOG_FUNCTION (this << model);
  Ptr<MobilityModel> oldChild = m_child;
  Vector pos;
  if (m_child)
    {
      NS_LOG_DEBUG ("Disconnecting previous child model " << m_child);
      pos = GetPosition ();
      m_child->TraceDisconnectWithoutContext ("CourseChange", MakeCallback (&HierarchicalMobilityModel::ChildChanged, this));
    }
  m_child = model;
  m_child->TraceConnectWithoutContext ("CourseChange", MakeCallback (&HierarchicalMobilityModel::ChildChanged, this));

  // if we had a child before, then we had a valid position before;
  // try to preserve the old absolute position.
  if (oldChild)
    {
      NS_LOG_DEBUG ("Restoring previous position " << pos);
      SetPosition (pos);
    }
}

void 
HierarchicalMobilityModel::SetParent (Ptr<MobilityModel> model)
{
  NS_LOG_FUNCTION (this << model);
  Vector pos;
  if (m_child)
    {
      pos = GetPosition ();
    }
  if (m_parent)
    {
      NS_LOG_DEBUG ("Disconnecting previous parent model " << m_child);
      m_parent->TraceDisconnectWithoutContext ("CourseChange", MakeCallback (&HierarchicalMobilityModel::ParentChanged, this));
    }
  m_parent = model;
  if (m_parent)
    {
      m_parent->TraceConnectWithoutContext ("CourseChange", MakeCallback (&HierarchicalMobilityModel::ParentChanged, this));
    }
  // try to preserve the old position across parent changes
  if (m_child)
    {
      NS_LOG_DEBUG ("Restoring previous position " << pos);
      SetPosition (pos);
    }
}


Ptr<MobilityModel> 
HierarchicalMobilityModel::GetChild (void) const
{
  return m_child;
}

Ptr<MobilityModel> 
HierarchicalMobilityModel::GetParent (void) const
{
  return m_parent;
}

Vector
HierarchicalMobilityModel::DoGetPosition (void) const
{
  if (!m_parent)
    {
      return m_child->GetPosition ();
    }
  Vector parentPosition = m_parent->GetPosition ();
  Vector childPosition = m_child->GetPositionWithReference (parentPosition);
  return Vector (parentPosition.x + childPosition.x,
                 parentPosition.y + childPosition.y,
                 parentPosition.z + childPosition.z);
}
void 
HierarchicalMobilityModel::DoSetPosition (const Vector &position)
{
  NS_LOG_FUNCTION (this << position);
  if (m_child == 0)
    {
      return;
    }
  // This implementation of DoSetPosition is really an arbitrary choice.
  // anything else would have been ok.
  if (m_parent)
    {
      Vector parentPosition = m_parent->GetPosition ();
      Vector childPosition (position.x - parentPosition.x,
                            position.y - parentPosition.y,
                            position.z - parentPosition.z);
      m_child->SetPosition (childPosition);
    }
  else
    {
      m_child->SetPosition (position);
    }
}
Vector
HierarchicalMobilityModel::DoGetVelocity (void) const
{
  if (m_parent)
    {
      Vector parentSpeed = m_parent->GetVelocity ();
      Vector childSpeed = m_child->GetVelocity ();
      Vector speed (parentSpeed.x + childSpeed.x,
                    parentSpeed.y + childSpeed.y,
                    parentSpeed.z + childSpeed.z);
      return speed;
    }
  else
    {
      return m_child->GetVelocity ();
    }
}

void 
HierarchicalMobilityModel::ParentChanged (Ptr<const MobilityModel> model)
{
  MobilityModel::NotifyCourseChange ();
}

void 
HierarchicalMobilityModel::ChildChanged (Ptr<const MobilityModel> model)
{
  MobilityModel::NotifyCourseChange ();
}

void
HierarchicalMobilityModel::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  if (m_parent && !m_parent->IsInitialized ())
    {
      m_parent->Initialize ();
    }
  m_child->Initialize ();
}

int64_t
HierarchicalMobilityModel::DoAssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  int64_t streamsAllocated = 0;
  streamsAllocated += m_parent->AssignStreams (stream);
  streamsAllocated += m_child->AssignStreams (stream + streamsAllocated);
  return streamsAllocated;
}

} // namespace ns3
