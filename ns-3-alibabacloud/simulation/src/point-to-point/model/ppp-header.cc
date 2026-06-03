/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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
 */

#include <iostream>
#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ppp-header.h"

NS_LOG_COMPONENT_DEFINE ("PppHeader");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (PppHeader);

PppHeader::PppHeader ()
{
}

PppHeader::~PppHeader ()
{
}

TypeId
PppHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PppHeader")
    .SetParent<Header> ()
    .AddConstructor<PppHeader> ()
  ;
  return tid;
}

TypeId
PppHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void 
PppHeader::Print (std::ostream &os) const
{
  std::string proto;

  switch(m_protocol)
    {
    case 0x0021: /* IPv4 */
      proto = "IP (0x0021)";
      break;
    case 0x0057: /* IPv6 */
      proto = "IPv6 (0x0057)";
      break;
    default:
      NS_ASSERT_MSG (false, "PPP Protocol number not defined!");
    }
  os << "Point-to-Point Protocol: " << proto; 
}

uint32_t
PppHeader::GetSerializedSize (void) const
{
	return GetStaticSize();
}
uint32_t PppHeader::GetStaticSize (void){
	return 14;
}

void
PppHeader::Serialize (Buffer::Iterator start) const
{
  start.WriteHtonU16 (m_protocol);
  start.WriteU64(0);
  start.WriteU32(0);
}

uint32_t
PppHeader::Deserialize (Buffer::Iterator start)
{
  m_protocol = start.ReadNtohU16 ();
  start.Next(12);
  return GetSerializedSize ();
}

void
PppHeader::SetProtocol (uint16_t protocol)
{
  m_protocol=protocol;
}

uint16_t
PppHeader::GetProtocol (void)
{
  return m_protocol;
}


} // namespace ns3
