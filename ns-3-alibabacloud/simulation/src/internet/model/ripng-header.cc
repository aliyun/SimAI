/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' di Firenze, Italy
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

#include "ripng-header.h"
#include "ns3/log.h"

namespace ns3 {

/*
 * RipNgRte
 */
NS_OBJECT_ENSURE_REGISTERED (RipNgRte);

RipNgRte::RipNgRte ()
  : m_prefix ("::"), m_tag (0), m_prefixLen (0), m_metric (16)
{
}

TypeId RipNgRte::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RipNgRte")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<RipNgRte> ();
  return tid;
}

TypeId RipNgRte::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void RipNgRte::Print (std::ostream & os) const
{
  os << "prefix " << m_prefix << "/" << int(m_prefixLen) << " Metric " << int(m_metric) << " Tag " << int(m_tag);
}

uint32_t RipNgRte::GetSerializedSize () const
{
  return 20;
}

void RipNgRte::Serialize (Buffer::Iterator i) const
{
  uint8_t tmp[16];

  m_prefix.Serialize (tmp);
  i.Write (tmp, 16);

  i.WriteHtonU16 (m_tag);
  i.WriteU8 (m_prefixLen);
  i.WriteU8 (m_metric);
}

uint32_t RipNgRte::Deserialize (Buffer::Iterator i)
{
  uint8_t tmp[16];

  i.Read (tmp, 16);
  m_prefix.Set (tmp);
  m_tag = i.ReadNtohU16 ();
  m_prefixLen = i.ReadU8 ();
  m_metric = i.ReadU8 ();

  return GetSerializedSize ();
}

void RipNgRte::SetPrefix (Ipv6Address prefix)
{
  m_prefix = prefix;
}

Ipv6Address RipNgRte::GetPrefix () const
{
  return m_prefix;
}

void RipNgRte::SetPrefixLen (uint8_t prefixLen)
{
  m_prefixLen = prefixLen;
}

uint8_t RipNgRte::GetPrefixLen () const
{
  return m_prefixLen;
}

void RipNgRte::SetRouteTag (uint16_t routeTag)
{
  m_tag = routeTag;
}

uint16_t RipNgRte::GetRouteTag () const
{
  return m_tag;
}

void RipNgRte::SetRouteMetric (uint8_t routeMetric)
{
  m_metric = routeMetric;
}

uint8_t RipNgRte::GetRouteMetric () const
{
  return m_metric;
}


std::ostream & operator << (std::ostream & os, const RipNgRte & h)
{
  h.Print (os);
  return os;
}

/*
 * RipNgHeader
 */
NS_LOG_COMPONENT_DEFINE ("RipNgHeader");
NS_OBJECT_ENSURE_REGISTERED (RipNgHeader);

RipNgHeader::RipNgHeader ()
  : m_command (0)
{
}

TypeId RipNgHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RipNgHeader")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<RipNgHeader> ();
  return tid;
}

TypeId RipNgHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void RipNgHeader::Print (std::ostream & os) const
{
  os << "command " << int(m_command);
  for (std::list<RipNgRte>::const_iterator iter = m_rteList.begin ();
      iter != m_rteList.end (); iter ++)
    {
      os << " | ";
      iter->Print (os);
    }
}

uint32_t RipNgHeader::GetSerializedSize () const
{
  RipNgRte rte;
  return 4 + m_rteList.size () * rte.GetSerializedSize ();
}

void RipNgHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (uint8_t (m_command));
  i.WriteU8 (1);
  i.WriteU16 (0);

  for (std::list<RipNgRte>::const_iterator iter = m_rteList.begin ();
      iter != m_rteList.end (); iter ++)
    {
      iter->Serialize (i);
      i.Next(iter->GetSerializedSize ());
    }
}

uint32_t RipNgHeader::Deserialize (Buffer::Iterator start)
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

  if (i.ReadU8 () != 1)
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
      RipNgRte rte;
      i.Next (rte.Deserialize (i));
      m_rteList.push_back (rte);
    }

  return GetSerializedSize ();
}

void RipNgHeader::SetCommand (RipNgHeader::Command_e command)
{
  m_command = command;
}

RipNgHeader::Command_e RipNgHeader::GetCommand () const
{
  return RipNgHeader::Command_e (m_command);
}

void RipNgHeader::AddRte (RipNgRte rte)
{
  m_rteList.push_back (rte);
}

void RipNgHeader::ClearRtes ()
{
  m_rteList.clear ();
}

uint16_t RipNgHeader::GetRteNumber (void) const
{
  return m_rteList.size ();
}

std::list<RipNgRte> RipNgHeader::GetRteList (void) const
{
  return m_rteList;
}


std::ostream & operator << (std::ostream & os, const RipNgHeader & h)
{
  h.Print (os);
  return os;
}


}

