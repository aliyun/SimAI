/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
#include "packet-socket-address.h"
#include "ns3/net-device.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketSocketAddress");

PacketSocketAddress::PacketSocketAddress ()
{
  NS_LOG_FUNCTION (this);
}
void
PacketSocketAddress::SetProtocol (uint16_t protocol)
{
  NS_LOG_FUNCTION (this << protocol);
  m_protocol = protocol;
}
void
PacketSocketAddress::SetAllDevices (void)
{
  NS_LOG_FUNCTION (this);
  m_isSingleDevice = false;
  m_device = 0;
}
void
PacketSocketAddress::SetSingleDevice (uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  m_isSingleDevice = true;
  m_device = index;
}
void
PacketSocketAddress::SetPhysicalAddress (const Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;
}

uint16_t
PacketSocketAddress::GetProtocol (void) const
{
  NS_LOG_FUNCTION (this);
  return m_protocol;
}
bool
PacketSocketAddress::IsSingleDevice (void) const
{
  NS_LOG_FUNCTION (this);
  return m_isSingleDevice;
}
uint32_t
PacketSocketAddress::GetSingleDevice (void) const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}
Address
PacketSocketAddress::GetPhysicalAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}

PacketSocketAddress::operator Address () const
{
  return ConvertTo ();
}

Address
PacketSocketAddress::ConvertTo (void) const
{
  NS_LOG_FUNCTION (this);
  Address address;
  uint8_t buffer[Address::MAX_SIZE];
  buffer[0] = m_protocol & 0xff;
  buffer[1] = (m_protocol >> 8) & 0xff;
  buffer[2] = (m_device >> 24) & 0xff;
  buffer[3] = (m_device >> 16) & 0xff;
  buffer[4] = (m_device >> 8) & 0xff;
  buffer[5] = (m_device >> 0) & 0xff;
  buffer[6] = m_isSingleDevice ? 1 : 0;
  uint32_t copied = m_address.CopyAllTo (buffer + 7, Address::MAX_SIZE - 7);
  return Address (GetType (), buffer, 7 + copied);
}
PacketSocketAddress
PacketSocketAddress::ConvertFrom (const Address &address)
{
  NS_LOG_FUNCTION (address);
  NS_ASSERT (IsMatchingType (address));
  uint8_t buffer[Address::MAX_SIZE];
  address.CopyTo (buffer);
  uint16_t protocol = buffer[0] | (buffer[1] << 8);
  uint32_t device = 0;
  device |= buffer[2];
  device <<= 8;
  device |= buffer[3];
  device <<= 8;
  device |= buffer[4];
  device <<= 8;
  device |= buffer[5];
  bool isSingleDevice = (buffer[6] == 1);
  Address physical;
  physical.CopyAllFrom (buffer + 7, Address::MAX_SIZE - 7);
  PacketSocketAddress ad;
  ad.SetProtocol (protocol);
  if (isSingleDevice)
    {
      ad.SetSingleDevice (device);
    }
  else
    {
      ad.SetAllDevices ();
    }
  ad.SetPhysicalAddress (physical);
  return ad;
}
bool
PacketSocketAddress::IsMatchingType (const Address &address)
{
  NS_LOG_FUNCTION (address);
  return address.IsMatchingType (GetType ());
}
uint8_t
PacketSocketAddress::GetType (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint8_t type = Address::Register ();
  return type;
}

} // namespace ns3
