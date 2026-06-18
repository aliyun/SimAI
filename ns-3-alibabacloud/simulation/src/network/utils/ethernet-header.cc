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
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include <iomanip>
#include <iostream>
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ethernet-header.h"
#include "address-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EthernetHeader");

NS_OBJECT_ENSURE_REGISTERED (EthernetHeader);

EthernetHeader::EthernetHeader (bool hasPreamble)
  : m_enPreambleSfd (hasPreamble),
    m_lengthType (0)
{
  NS_LOG_FUNCTION (this << hasPreamble);
}

EthernetHeader::EthernetHeader ()
  : m_enPreambleSfd (false),
    m_lengthType (0)
{
  NS_LOG_FUNCTION (this);
}

void
EthernetHeader::SetLengthType (uint16_t lengthType)
{
  NS_LOG_FUNCTION (this << lengthType);
  m_lengthType = lengthType;
}
uint16_t
EthernetHeader::GetLengthType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_lengthType;
}

void
EthernetHeader::SetPreambleSfd (uint64_t preambleSfd)
{
  NS_LOG_FUNCTION (this << preambleSfd);
  m_preambleSfd = preambleSfd;
}
uint64_t
EthernetHeader::GetPreambleSfd (void) const
{
  NS_LOG_FUNCTION (this);
  return m_preambleSfd;
}

void
EthernetHeader::SetSource (Mac48Address source)
{
  NS_LOG_FUNCTION (this << source);
  m_source = source;
}
Mac48Address
EthernetHeader::GetSource (void) const
{
  NS_LOG_FUNCTION (this);
  return m_source;
}

void 
EthernetHeader::SetDestination (Mac48Address dst)
{
  NS_LOG_FUNCTION (this << dst);
  m_destination = dst;
}
Mac48Address
EthernetHeader::GetDestination (void) const
{
  NS_LOG_FUNCTION (this);
  return m_destination;
}

ethernet_header_t 
EthernetHeader::GetPacketType (void) const
{
  NS_LOG_FUNCTION (this);
  return LENGTH;
}

uint32_t 
EthernetHeader::GetHeaderSize (void) const
{
  NS_LOG_FUNCTION (this);
  return GetSerializedSize ();
}


TypeId 
EthernetHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EthernetHeader")
    .SetParent<Header> ()
    .SetGroupName("Network")
    .AddConstructor<EthernetHeader> ()
  ;
  return tid;
}
TypeId 
EthernetHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
void 
EthernetHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  // ethernet, right ?
  if (m_enPreambleSfd)
    {
      os << "preamble/sfd=" << m_preambleSfd << ",";
    }

  os << " length/type=0x" << std::hex << m_lengthType << std::dec
     << ", source=" << m_source
     << ", destination=" << m_destination;
}
uint32_t 
EthernetHeader::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_enPreambleSfd)
    {
      return PREAMBLE_SIZE + LENGTH_SIZE + 2*MAC_ADDR_SIZE;
    } 
  else 
    {
      return LENGTH_SIZE + 2*MAC_ADDR_SIZE;
    }
}

void
EthernetHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;

  if (m_enPreambleSfd)
    {
      i.WriteU64 (m_preambleSfd);
    }
  WriteTo (i, m_destination);
  WriteTo (i, m_source);
  i.WriteHtonU16 (m_lengthType);
}
uint32_t
EthernetHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;

  if (m_enPreambleSfd)
    {
      m_enPreambleSfd = i.ReadU64 ();
    }

  ReadFrom (i, m_destination);
  ReadFrom (i, m_source);
  m_lengthType = i.ReadNtohU16 ();

  return GetSerializedSize ();
}

} // namespace ns3
