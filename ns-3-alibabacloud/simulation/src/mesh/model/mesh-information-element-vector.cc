/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Pavel Boyko <boyko@iitp.ru>
 */

#include "mesh-information-element-vector.h"
#include "ns3/packet.h"
#include <algorithm>
#include "ns3/hwmp-protocol.h"
// All information elements:
#include "ns3/ie-dot11s-beacon-timing.h"
#include "ns3/ie-dot11s-configuration.h"
#include "ns3/ie-dot11s-id.h"
#include "ns3/ie-dot11s-metric-report.h"
#include "ns3/ie-dot11s-peer-management.h"
#include "ns3/ie-dot11s-peering-protocol.h"
#include "ns3/ie-dot11s-perr.h"
#include "ns3/ie-dot11s-prep.h"
#include "ns3/ie-dot11s-preq.h"
#include "ns3/ie-dot11s-rann.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (MeshInformationElementVector);

MeshInformationElementVector::MeshInformationElementVector ()
{
}

MeshInformationElementVector::~MeshInformationElementVector ()
{
}


TypeId
MeshInformationElementVector::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::MeshInformationElementVector")
    .SetParent<WifiInformationElementVector> ()
    .SetGroupName ("Mesh")
    .AddConstructor<MeshInformationElementVector> ();
  return tid;
}

TypeId
MeshInformationElementVector::GetInstanceTypeId () const
{
  return GetTypeId ();
}

uint32_t
MeshInformationElementVector::DeserializeSingleIe (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t id = i.ReadU8 ();
  uint8_t length = i.ReadU8 ();
  Ptr<WifiInformationElement> newElement;
  switch (id)
    {
    case IE_MESH_CONFIGURATION:
      newElement = Create<dot11s::IeConfiguration> ();
      break;
    case IE_MESH_ID:
      newElement = Create<dot11s::IeMeshId> ();
      break;
    case IE_MESH_LINK_METRIC_REPORT:
      newElement = Create<dot11s::IeLinkMetricReport> ();
      break;
    case IE_MESH_PEERING_MANAGEMENT:
      newElement = Create<dot11s::IePeerManagement> ();
      break;
    case IE_BEACON_TIMING:
      newElement = Create<dot11s::IeBeaconTiming> ();
      break;
    case IE_RANN:
      newElement = Create<dot11s::IeRann> ();
      break;
    case IE_PREQ:
      newElement = Create<dot11s::IePreq> ();
      break;
    case IE_PREP:
      newElement = Create<dot11s::IePrep> ();
      break;
    case IE_PERR:
      newElement = Create<dot11s::IePerr> ();
      break;
    case IE11S_MESH_PEERING_PROTOCOL_VERSION:
      newElement = Create<dot11s::IePeeringProtocol> ();
      break;
    default:
      // We peeked at the ID and length, so we need to back up the
      // pointer before deferring to our parent.
      i.Prev (2);
      return WifiInformationElementVector::DeserializeSingleIe (i);
    }
  if (GetSize () + length > m_maxSize)
    {
      NS_FATAL_ERROR ("Check max size for information element!");
    }
  newElement->DeserializeInformationField (i, length);
  i.Next (length);
  m_elements.push_back (newElement);
  return i.GetDistanceFrom (start);
}


} // namespace ns3
