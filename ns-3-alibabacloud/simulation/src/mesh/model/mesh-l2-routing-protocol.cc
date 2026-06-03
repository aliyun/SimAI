/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#include "ns3/log.h"
#include "ns3/mesh-l2-routing-protocol.h"
#include "ns3/mesh-point-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MeshL2RoutingProtocol");

NS_OBJECT_ENSURE_REGISTERED (MeshL2RoutingProtocol);

TypeId
MeshL2RoutingProtocol::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MeshL2RoutingProtocol")
    .SetParent<Object> ()
    .SetGroupName ("Mesh")
  ;
  return tid;
}

MeshL2RoutingProtocol::~MeshL2RoutingProtocol ()
{
  m_mp = 0;
}

void
MeshL2RoutingProtocol::SetMeshPoint (Ptr<MeshPointDevice> mp)
{
  m_mp = mp;
}

Ptr<MeshPointDevice>
MeshL2RoutingProtocol::GetMeshPoint () const
{
  return m_mp;
}

} // namespace ns3
