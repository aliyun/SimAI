/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#include "wimax-connection.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "service-flow.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (WimaxConnection);

TypeId WimaxConnection::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WimaxConnection")

    .SetParent<Object> ()

    .SetGroupName("Wimax")

    .AddAttribute ("Type",
                   "Connection type",
                   EnumValue (Cid::INITIAL_RANGING),
                   MakeEnumAccessor (&WimaxConnection::GetType),
                   MakeEnumChecker (Cid::BROADCAST,
                                    "Broadcast",
                                    Cid::INITIAL_RANGING,
                                    "InitialRanging",
                                    Cid::BASIC,
                                    "Basic",
                                    Cid::PRIMARY,
                                    "Primary",
                                    Cid::TRANSPORT,
                                    "Transport",
                                    Cid::MULTICAST,
                                    "Multicast",
                                    Cid::PADDING,
                                    "Padding"))

    .AddAttribute ("TxQueue",
                   "Transmit queue",
                   PointerValue (),
                   MakePointerAccessor (&WimaxConnection::GetQueue),
                   MakePointerChecker<WimaxMacQueue> ());
  return tid;
}

WimaxConnection::WimaxConnection (Cid cid, enum Cid::Type type)
  : m_cid (cid),
    m_cidType (type),
    m_queue (CreateObject<WimaxMacQueue> (1024)),
    m_serviceFlow (0)
{
}

WimaxConnection::~WimaxConnection (void)
{
}

void
WimaxConnection::DoDispose (void)
{
  m_queue = 0;
  // m_serviceFlow = 0;
}

Cid
WimaxConnection::GetCid (void) const
{
  return m_cid;
}

enum Cid::Type
WimaxConnection::GetType (void) const
{
  return m_cidType;
}

Ptr<WimaxMacQueue>
WimaxConnection::GetQueue (void) const
{
  return m_queue;
}

void
WimaxConnection::SetServiceFlow (ServiceFlow *serviceFlow)
{
  m_serviceFlow = serviceFlow;
}

ServiceFlow*
WimaxConnection::GetServiceFlow (void) const
{
  return m_serviceFlow;
}

uint8_t
WimaxConnection::GetSchedulingType (void) const
{
  return m_serviceFlow->GetSchedulingType ();
}

bool
WimaxConnection::Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType, const GenericMacHeader &hdr)
{

  return m_queue->Enqueue (packet, hdrType, hdr);
}

Ptr<Packet>
WimaxConnection::Dequeue (MacHeaderType::HeaderType packetType)
{
  return m_queue->Dequeue (packetType);
}

Ptr<Packet>
WimaxConnection::Dequeue (MacHeaderType::HeaderType packetType, uint32_t availableByte)
{
  return m_queue->Dequeue (packetType, availableByte);
}

bool
WimaxConnection::HasPackets (void) const
{
  return !m_queue->IsEmpty ();
}

bool
WimaxConnection::HasPackets (MacHeaderType::HeaderType packetType) const
{
  return !m_queue->IsEmpty (packetType);
}

std::string
WimaxConnection::GetTypeStr (void) const
{
  switch (m_cidType)
    {
    case Cid::BROADCAST:
      return "Broadcast";
      break;
    case Cid::INITIAL_RANGING:
      return "Initial Ranging";
      break;
    case Cid::BASIC:
      return "Basic";
      break;
    case Cid::PRIMARY:
      return "Primary";
      break;
    case Cid::TRANSPORT:
      return "Transport";
      break;
    case Cid::MULTICAST:
      return "Multicast";
      break;
    default:
      NS_FATAL_ERROR ("Invalid connection type");
    }

  return "";
}

// Defragmentation Function
const
WimaxConnection::FragmentsQueue WimaxConnection::GetFragmentsQueue (void) const
{
  return m_fragmentsQueue;
}

void
WimaxConnection::FragmentEnqueue (Ptr<const Packet>  fragment)
{
  m_fragmentsQueue.push_back (fragment);
}

void
WimaxConnection::ClearFragmentsQueue (void)
{
  m_fragmentsQueue.clear ();
}
} // namespace ns3
