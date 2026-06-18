/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
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
 * Authors: Hajime Tazaki <tazaki@sfc.wide.ad.jp>
 */


#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "ipv4-packet-info-tag.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4PacketInfoTag");

NS_OBJECT_ENSURE_REGISTERED (Ipv4PacketInfoTag);

Ipv4PacketInfoTag::Ipv4PacketInfoTag ()
  : m_addr (Ipv4Address ()),
    m_ifindex (0),
    m_ttl (0)
{
  NS_LOG_FUNCTION (this);
}

void 
Ipv4PacketInfoTag::SetAddress (Ipv4Address addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_addr = addr;
}

Ipv4Address
Ipv4PacketInfoTag::GetAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_addr;
}

void 
Ipv4PacketInfoTag::SetRecvIf (uint32_t ifindex)
{
  NS_LOG_FUNCTION (this << ifindex);
  m_ifindex = ifindex;
}

uint32_t 
Ipv4PacketInfoTag::GetRecvIf (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ifindex;
}

void 
Ipv4PacketInfoTag::SetTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (ttl));
  m_ttl = ttl;
}

uint8_t 
Ipv4PacketInfoTag::GetTtl (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ttl;
}



TypeId
Ipv4PacketInfoTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4PacketInfoTag")
    .SetParent<Tag> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv4PacketInfoTag> ()
  ;
  return tid;
}
TypeId
Ipv4PacketInfoTag::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}

uint32_t 
Ipv4PacketInfoTag::GetSerializedSize (void) const
{ 
  NS_LOG_FUNCTION (this);
  return 4 + sizeof (uint32_t) + sizeof (uint8_t);
}
void 
Ipv4PacketInfoTag::Serialize (TagBuffer i) const
{
  NS_LOG_FUNCTION (this << &i);
  uint8_t buf[4];
  m_addr.Serialize (buf);
  i.Write (buf, 4);
  i.WriteU32 (m_ifindex);
  i.WriteU8 (m_ttl);
}
void 
Ipv4PacketInfoTag::Deserialize (TagBuffer i)
{
  NS_LOG_FUNCTION (this<< &i);
  uint8_t buf[4];
  i.Read (buf, 4);
  m_addr = Ipv4Address::Deserialize (buf);
  m_ifindex = i.ReadU32 ();
  m_ttl = i.ReadU8 ();
}
void
Ipv4PacketInfoTag::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "Ipv4 PKTINFO [DestAddr: " << m_addr;
  os << ", RecvIf:" << (uint32_t) m_ifindex;
  os << ", TTL:" << (uint32_t) m_ttl;
  os << "] ";
}
} // namespace ns3

