/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "rip-header.h"
#include "ns3/log.h"

namespace ns3 {

/*
 * RipRte
 */
NS_OBJECT_ENSURE_REGISTERED (RipRte);


RipRte::RipRte ()
  : m_tag (0), m_prefix ("127.0.0.1"), m_subnetMask ("0.0.0.0"), m_nextHop ("0.0.0.0"), m_metric (16)
{
}

TypeId RipRte::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RipRte")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<RipRte> ();
  return tid;
}

TypeId RipRte::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void RipRte::Print (std::ostream & os) const
{
  os << "prefix " << m_prefix << "/" << m_subnetMask.GetPrefixLength () << " Metric " << int(m_metric);
  os << " Tag " << int(m_tag) << " Next Hop " << m_nextHop;
}

uint32_t RipRte::GetSerializedSize () const
{
  return 20;
}

void RipRte::Serialize (Buffer::Iterator i) const
{
  i.WriteHtonU16 (2);
  i.WriteHtonU16 (m_tag);

  i.WriteHtonU32 (m_prefix.Get ());
  i.WriteHtonU32 (m_subnetMask.Get ());
  i.WriteHtonU32 (m_nextHop.Get ());
  i.WriteHtonU32 (m_metric);
}

uint32_t RipRte::Deserialize (Buffer::Iterator i)
{
  uint16_t tmp;

  tmp = i.ReadNtohU16 ();
  if (tmp != 2)
    {
      return 0;
    }

  m_tag = i.ReadNtohU16 ();
  m_prefix.Set (i.ReadNtohU32 ());
  m_subnetMask.Set (i.ReadNtohU32 ());
  m_nextHop.Set (i.ReadNtohU32 ());

  m_metric = i.ReadNtohU32 ();

  return GetSerializedSize ();
}

void RipRte::SetPrefix (Ipv4Address prefix)
{
  m_prefix = prefix;
}

Ipv4Address RipRte::GetPrefix () const
{
  return m_prefix;
}

void RipRte::SetSubnetMask (Ipv4Mask subnetMask)
{
  m_subnetMask = subnetMask;
}

Ipv4Mask RipRte::GetSubnetMask () const
{
  return m_subnetMask;
}

void RipRte::SetRouteTag (uint16_t routeTag)
{
  m_tag = routeTag;
}

uint16_t RipRte::GetRouteTag () const
{
  return m_tag;
}

void RipRte::SetRouteMetric (uint32_t routeMetric)
{
  m_metric = routeMetric;
}

uint32_t RipRte::GetRouteMetric () const
{
  return m_metric;
}

void RipRte::SetNextHop (Ipv4Address nextHop)
{
  m_nextHop = nextHop;
}

Ipv4Address RipRte::GetNextHop () const
{
  return m_nextHop;
}


std::ostream & operator << (std::ostream & os, const RipRte & h)
{
  h.Print (os);
  return os;
}

/*
 * RipHeader
 */
NS_LOG_COMPONENT_DEFINE ("RipHeader");
NS_OBJECT_ENSURE_REGISTERED (RipHeader);


RipHeader::RipHeader ()
  : m_command (0)
{
}

TypeId RipHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RipHeader")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<RipHeader> ();
  return tid;
}

TypeId RipHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void RipHeader::Print (std::ostream & os) const
{
  os << "command " << int(m_command);
  for (std::list<RipRte>::const_iterator iter = m_rteList.begin ();
      iter != m_rteList.end (); iter ++)
    {
      os << " | ";
      iter->Print (os);
    }
}

uint32_t RipHeader::GetSerializedSize () const
{
  RipRte rte;
  return 4 + m_rteList.size () * rte.GetSerializedSize ();
}

void RipHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (uint8_t (m_command));
  i.WriteU8 (2);
  i.WriteU16 (0);

  for (std::list<RipRte>::const_iterator iter = m_rteList.begin ();
      iter != m_rteList.end (); iter ++)
    {
      iter->Serialize (i);
      i.Next(iter->GetSerializedSize ());
    }
}

uint32_t RipHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  uint8_t temp;
  temp = i.ReadU8 ();
  if ((temp == REQUEST) || (temp == RESPONSE))
    {
      m_command = temp;
    }
  else
    {
      return 0;
    }

  if (i.ReadU8 () != 2)
    {
      NS_LOG_LOGIC ("RIP received a message with mismatch version, ignoring.");
      return 0;
    }

  if (i.ReadU16 () != 0)
    {
      NS_LOG_LOGIC ("RIP received a message with invalid filled flags, ignoring.");
      return 0;
    }

  uint8_t rteNumber = i.GetRemainingSize ()/20;
  for (uint8_t n=0; n<rteNumber; n++)
    {
      RipRte rte;
      i.Next (rte.Deserialize (i));
      m_rteList.push_back (rte);
    }

  return GetSerializedSize ();
}

void RipHeader::SetCommand (RipHeader::Command_e command)
{
  m_command = command;
}

RipHeader::Command_e RipHeader::GetCommand () const
{
  return RipHeader::Command_e (m_command);
}

void RipHeader::AddRte (RipRte rte)
{
  m_rteList.push_back (rte);
}

void RipHeader::ClearRtes ()
{
  m_rteList.clear ();
}

uint16_t RipHeader::GetRteNumber (void) const
{
  return m_rteList.size ();
}

std::list<RipRte> RipHeader::GetRteList (void) const
{
  return m_rteList;
}


std::ostream & operator << (std::ostream & os, const RipHeader & h)
{
  h.Print (os);
  return os;
}


}

