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

#include "inet-socket-address.h"
#include "ns3/assert.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("InetSocketAddress");

InetSocketAddress::InetSocketAddress (Ipv4Address ipv4, uint16_t port)
  : m_ipv4 (ipv4),
    m_port (port),
    m_tos (0)
{
  NS_LOG_FUNCTION (this << ipv4 << port);
}
InetSocketAddress::InetSocketAddress (Ipv4Address ipv4)
  : m_ipv4 (ipv4),
    m_port (0),
    m_tos (0)
{
  NS_LOG_FUNCTION (this << ipv4);
}
InetSocketAddress::InetSocketAddress (const char *ipv4, uint16_t port)
  : m_ipv4 (Ipv4Address (ipv4)),
    m_port (port),
    m_tos (0)
{
  NS_LOG_FUNCTION (this << ipv4 << port);
}
InetSocketAddress::InetSocketAddress (const char * ipv4)
  : m_ipv4 (Ipv4Address (ipv4)),
    m_port (0),
    m_tos (0)
{
  NS_LOG_FUNCTION (this << ipv4);
}
InetSocketAddress::InetSocketAddress (uint16_t port)
  : m_ipv4 (Ipv4Address::GetAny ()),
    m_port (port),
    m_tos (0)
{
  NS_LOG_FUNCTION (this << port);
}
uint16_t
InetSocketAddress::GetPort (void) const
{
  NS_LOG_FUNCTION (this);
  return m_port;
}
Ipv4Address
InetSocketAddress::GetIpv4 (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ipv4;
}
uint8_t
InetSocketAddress::GetTos (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tos;
}

void 
InetSocketAddress::SetPort (uint16_t port)
{
  NS_LOG_FUNCTION (this << port);
  m_port = port;
}
void
InetSocketAddress::SetIpv4 (Ipv4Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_ipv4 = address;
}
void
InetSocketAddress::SetTos (uint8_t tos)
{
  NS_LOG_FUNCTION (this << tos);
  m_tos = tos;
}

bool
InetSocketAddress::IsMatchingType (const Address &address)
{
  NS_LOG_FUNCTION (&address);
  return address.CheckCompatible (GetType (), 7);
}

InetSocketAddress::operator Address () const
{
  return ConvertTo ();
}

Address 
InetSocketAddress::ConvertTo (void) const
{
  NS_LOG_FUNCTION (this);
  uint8_t buf[7];
  m_ipv4.Serialize (buf);
  buf[4] = m_port & 0xff;
  buf[5] = (m_port >> 8) & 0xff;
  buf[6] = m_tos;
  return Address (GetType (), buf, 7);
}
InetSocketAddress 
InetSocketAddress::ConvertFrom (const Address &address)
{
  NS_LOG_FUNCTION (&address);
  NS_ASSERT (address.CheckCompatible (GetType (), 7));
  uint8_t buf[7];
  address.CopyTo (buf);
  Ipv4Address ipv4 = Ipv4Address::Deserialize (buf);
  uint16_t port = buf[4] | (buf[5] << 8);
  uint8_t tos = buf[6];
  InetSocketAddress inet (ipv4, port);
  inet.SetTos (tos);
  return inet;
}
uint8_t 
InetSocketAddress::GetType (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint8_t type = Address::Register ();
  return type;
}

} // namespace ns3
