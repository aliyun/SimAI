/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#include "llc-snap-header.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include <string>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LlcSnalHeader");

NS_OBJECT_ENSURE_REGISTERED (LlcSnapHeader);

LlcSnapHeader::LlcSnapHeader ()
{
  NS_LOG_FUNCTION (this);
}

void
LlcSnapHeader::SetType (uint16_t type)
{
  NS_LOG_FUNCTION (this);
  m_etherType = type;
}
uint16_t
LlcSnapHeader::GetType (void)
{
  NS_LOG_FUNCTION (this);
  return m_etherType;
}

uint32_t 
LlcSnapHeader::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return LLC_SNAP_HEADER_LENGTH;
}

TypeId 
LlcSnapHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LlcSnapHeader")
    .SetParent<Header> ()
    .SetGroupName("Network")
    .AddConstructor<LlcSnapHeader> ()
  ;
  return tid;
}
TypeId 
LlcSnapHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
void 
LlcSnapHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "type 0x";
  os.setf (std::ios::hex, std::ios::basefield);
  os << m_etherType;
  os.setf (std::ios::dec, std::ios::basefield);
}

void
LlcSnapHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  uint8_t buf[] = { 0xaa, 0xaa, 0x03, 0, 0, 0};
  i.Write (buf, 6);
  i.WriteHtonU16 (m_etherType);
}
uint32_t
LlcSnapHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.Next (5+1);
  m_etherType = i.ReadNtohU16 ();
  return GetSerializedSize ();
}


} // namespace ns3
