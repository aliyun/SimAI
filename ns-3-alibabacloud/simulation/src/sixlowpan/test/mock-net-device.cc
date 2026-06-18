/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
#include "mock-net-device.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/boolean.h"
#include "ns3/simulator.h"
#include "ns3/channel.h"
#include "ns3/mac64-address.h"
#include "ns3/mac48-address.h"
#include "ns3/mac16-address.h"
#include "ns3/mac8-address.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MockNetDevice");
NS_OBJECT_ENSURE_REGISTERED (MockNetDevice);

TypeId
MockNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MockNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName("Network")
    .AddConstructor<MockNetDevice> ()
    .AddAttribute ("PointToPointMode",
                   "The device is configured in Point to Point mode",
                   BooleanValue (false),
                   MakeBooleanAccessor (&MockNetDevice::m_pointToPointMode),
                   MakeBooleanChecker ())
  ;
  return tid;
}

MockNetDevice::MockNetDevice ()
  : m_node (0),
    m_mtu (0xffff),
    m_ifIndex (0),
    m_linkUp (true)
{
  NS_LOG_FUNCTION (this);
}

void
MockNetDevice::Receive (Ptr<Packet> packet, uint16_t protocol,
                          Address to, Address from, NetDevice::PacketType packetType)
{
  NS_LOG_FUNCTION (this << packet << protocol << to << from);

  if (packetType != NetDevice::PACKET_OTHERHOST)
    {
      m_rxCallback (this, packet, protocol, from);
    }

  if (!m_promiscCallback.IsNull ())
    {
      m_promiscCallback (this, packet, protocol, from, to, packetType);
    }
}

void
MockNetDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  m_ifIndex = index;
}

uint32_t
MockNetDevice::GetIfIndex (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ifIndex;
}

Ptr<Channel>
MockNetDevice::GetChannel (void) const
{
  NS_LOG_FUNCTION (this);
  return 0;
}

void
MockNetDevice::SetAddress (Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;
}

Address
MockNetDevice::GetAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}

bool
MockNetDevice::SetMtu (const uint16_t mtu)
{
  NS_LOG_FUNCTION (this << mtu);
  m_mtu = mtu;
  return true;
}

uint16_t
MockNetDevice::GetMtu (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mtu;
}

bool
MockNetDevice::IsLinkUp (void) const
{
  NS_LOG_FUNCTION (this);
  return m_linkUp;
}

void
MockNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_linkChangeCallbacks.ConnectWithoutContext (callback);
}

bool
MockNetDevice::IsBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_pointToPointMode)
    {
      return false;
    }
  if (Mac64Address::IsMatchingType (m_address))
    {
      return false;
    }
  if (Mac8Address::IsMatchingType (m_address))
    {
      return false;
    }

  return true;
}

Address
MockNetDevice::GetBroadcast (void) const
{
  NS_LOG_FUNCTION (this);

  Address address;

  if (Mac48Address::IsMatchingType (m_address))
    {
      address = Mac48Address::GetBroadcast ();
    }
  else if (Mac16Address::IsMatchingType (m_address))
    {
      address = Mac16Address::GetBroadcast ();
    }

  return address;
}

bool
MockNetDevice::IsMulticast (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_pointToPointMode)
    {
      return false;
    }
  if (Mac64Address::IsMatchingType (m_address))
    {
      return false;
    }
  if (Mac8Address::IsMatchingType (m_address))
    {
      return false;
    }

  return true;
}

Address
MockNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (this << multicastGroup);

  Address address;

  if (Mac48Address::IsMatchingType (m_address))
    {
      address = Mac48Address::GetMulticast (multicastGroup);
    }
  else if (Mac16Address::IsMatchingType (m_address))
    {
      address = Mac16Address::GetMulticast (Ipv6Address::MakeIpv4MappedAddress (multicastGroup));
    }

  return address;
}

Address MockNetDevice::GetMulticast (Ipv6Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
  Address address;

  if (Mac48Address::IsMatchingType (m_address))
    {
      address = Mac48Address::GetMulticast (addr);
    }
  else if (Mac16Address::IsMatchingType (m_address))
    {
      address = Mac16Address::GetMulticast (addr);
    }

  return address;
}

bool
MockNetDevice::IsPointToPoint (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_pointToPointMode)
    {
      return true;
    }
  return false;
}

bool
MockNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

bool
MockNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << packet << dest << protocolNumber);

  return SendFrom (packet, m_address, dest, protocolNumber);
}

bool
MockNetDevice::SendFrom (Ptr<Packet> p, const Address& source, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << p << source << dest << protocolNumber);
  if (p->GetSize () > GetMtu ())
    {
      return false;
    }

  if (!m_sendCallback.IsNull ())
    {
      m_sendCallback (this, p, protocolNumber, source, dest, NetDevice::PACKET_HOST);
    }

  return true;
}


Ptr<Node>
MockNetDevice::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}
void
MockNetDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

bool
MockNetDevice::NeedsArp (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_pointToPointMode)
    {
      return false;
    }
  return true;
}

void
MockNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION (this << &cb);
  m_rxCallback = cb;
}

void
MockNetDevice::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  m_rxCallback.Nullify ();
  m_promiscCallback.Nullify ();
  m_sendCallback.Nullify ();
  NetDevice::DoDispose ();
}

void
MockNetDevice::SetPromiscReceiveCallback (PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (this << &cb);
  m_promiscCallback = cb;
}

bool
MockNetDevice::SupportsSendFrom (void) const
{
  NS_LOG_FUNCTION (this);
  return true;
}

void
MockNetDevice::SetSendCallback (PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (this << &cb);
  m_sendCallback = cb;
}


} // namespace ns3
