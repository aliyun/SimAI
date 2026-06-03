/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "uan-header-common.h"
#include "ns3/mac8-address.h"

static const uint16_t ARP_PROT_NUMBER = 0x0806;
static const uint16_t IPV4_PROT_NUMBER = 0x0800;
static const uint16_t IPV6_PROT_NUMBER = 0x86DD;

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (UanHeaderCommon);

UanHeaderCommon::UanHeaderCommon ()
{
}

UanHeaderCommon::UanHeaderCommon (const Mac8Address src, const Mac8Address dest, uint8_t type, uint8_t protocolNumber)
  : Header (),
  m_dest (dest),
  m_src (src)
{
  SetProtocolNumber (protocolNumber);
  m_uanProtocolBits.m_type = type;
}

TypeId
UanHeaderCommon::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanHeaderCommon")
    .SetParent<Header> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanHeaderCommon> ()
  ;
  return tid;
}

TypeId
UanHeaderCommon::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
UanHeaderCommon::~UanHeaderCommon ()
{
}


void
UanHeaderCommon::SetDest (Mac8Address dest)
{
  m_dest = dest;
}
void
UanHeaderCommon::SetSrc (Mac8Address src)
{
  m_src = src;
}

void
UanHeaderCommon::SetType (uint8_t type)
{
  m_uanProtocolBits.m_type = type;
}

void
UanHeaderCommon::SetProtocolNumber (uint16_t protocolNumber)
{
  if (protocolNumber == 0)
    m_uanProtocolBits.m_protocolNumber = 0;
  else if (protocolNumber == IPV4_PROT_NUMBER)
    m_uanProtocolBits.m_protocolNumber = 1;
  else if (protocolNumber == ARP_PROT_NUMBER)
    m_uanProtocolBits.m_protocolNumber = 2;
  else if (protocolNumber == IPV6_PROT_NUMBER)
    m_uanProtocolBits.m_protocolNumber = 3;
  else
    NS_ASSERT_MSG (false, "UanHeaderCommon::SetProtocolNumber(): Protocol not supported");
}

Mac8Address
UanHeaderCommon::GetDest (void) const
{
  return m_dest;
}
Mac8Address
UanHeaderCommon::GetSrc (void) const
{
  return m_src;
}
uint8_t
UanHeaderCommon::GetType (void) const
{
  return m_uanProtocolBits.m_type;
}

uint16_t
UanHeaderCommon::GetProtocolNumber (void) const
{
  if (m_uanProtocolBits.m_protocolNumber == 1)
    return IPV4_PROT_NUMBER;
  if (m_uanProtocolBits.m_protocolNumber == 2)
    return ARP_PROT_NUMBER;
  if (m_uanProtocolBits.m_protocolNumber == 3)
    return IPV6_PROT_NUMBER;
  return 0;
}

// Inherrited methods

uint32_t
UanHeaderCommon::GetSerializedSize (void) const
{
  return 1 + 1 + 1;
}

void
UanHeaderCommon::Serialize (Buffer::Iterator start) const
{
  uint8_t address = 0;
  m_src.CopyTo (&address);
  start.WriteU8 (address);
  m_dest.CopyTo (&address);
  start.WriteU8 (address);
  char tmp = m_uanProtocolBits.m_type;
  tmp = tmp << 4;
  tmp += m_uanProtocolBits.m_protocolNumber;
  start.WriteU8 (tmp);
}

uint32_t
UanHeaderCommon::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator rbuf = start;

  m_src = Mac8Address (rbuf.ReadU8 ());
  m_dest = Mac8Address (rbuf.ReadU8 ());
  char tmp = rbuf.ReadU8 ();
  m_uanProtocolBits.m_type = tmp >> 4;
  m_uanProtocolBits.m_protocolNumber = tmp;

  return rbuf.GetDistanceFrom (start);
}

void
UanHeaderCommon::Print (std::ostream &os) const
{
  os << "UAN src=" << m_src << " dest=" << m_dest << " type=" << (uint32_t) m_uanProtocolBits.m_type << "Protocol Number=" << (uint32_t) m_uanProtocolBits.m_protocolNumber;
}




} // namespace ns3
