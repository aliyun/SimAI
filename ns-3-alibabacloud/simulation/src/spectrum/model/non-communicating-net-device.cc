/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/pointer.h"
#include "ns3/channel.h"
#include "non-communicating-net-device.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NonCommunicatingNetDevice");

NS_OBJECT_ENSURE_REGISTERED (NonCommunicatingNetDevice);

TypeId
NonCommunicatingNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NonCommunicatingNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<NonCommunicatingNetDevice> ()
    .AddAttribute ("Phy", "The PHY layer attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&NonCommunicatingNetDevice::GetPhy,
                                        &NonCommunicatingNetDevice::SetPhy),
                   MakePointerChecker<Object> ())
  ;
  return tid;
}

NonCommunicatingNetDevice::NonCommunicatingNetDevice ()
{
  NS_LOG_FUNCTION (this);
}

NonCommunicatingNetDevice::~NonCommunicatingNetDevice ()
{
  NS_LOG_FUNCTION (this);
}

void
NonCommunicatingNetDevice::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  m_channel = 0;
  m_phy = 0;
  NetDevice::DoDispose ();
}


void
NonCommunicatingNetDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION (index);
  m_ifIndex = index;
}

uint32_t
NonCommunicatingNetDevice::GetIfIndex (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ifIndex;
}

bool
NonCommunicatingNetDevice::SetMtu (uint16_t mtu)
{
  NS_LOG_FUNCTION (mtu);
  return (mtu == 0);
}

uint16_t
NonCommunicatingNetDevice::GetMtu (void) const
{
  NS_LOG_FUNCTION (this);
  return 0;
}

void
NonCommunicatingNetDevice::SetAddress (Address address)
{
  NS_LOG_FUNCTION (this);
}

Address
NonCommunicatingNetDevice::GetAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return Address ();
}

bool
NonCommunicatingNetDevice::IsBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

Address
NonCommunicatingNetDevice::GetBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  return Address ();
}

bool
NonCommunicatingNetDevice::IsMulticast (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

Address
NonCommunicatingNetDevice::GetMulticast (Ipv4Address addr) const
{
  NS_LOG_FUNCTION (addr);
  return Address ();
}

Address
NonCommunicatingNetDevice::GetMulticast (Ipv6Address addr) const
{
  NS_LOG_FUNCTION (addr);
  return Address ();
}

bool
NonCommunicatingNetDevice::IsPointToPoint (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

bool
NonCommunicatingNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}


Ptr<Node>
NonCommunicatingNetDevice::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

void
NonCommunicatingNetDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (node);

  m_node = node;
}

void
NonCommunicatingNetDevice::SetPhy (Ptr<Object> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_phy = phy;
}


Ptr<Object>
NonCommunicatingNetDevice::GetPhy () const
{
  NS_LOG_FUNCTION (this);
  return m_phy;
}


void
NonCommunicatingNetDevice::SetChannel (Ptr<Channel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}

Ptr<Channel>
NonCommunicatingNetDevice::GetChannel (void) const
{
  NS_LOG_FUNCTION (this);
  return m_channel;
}


bool
NonCommunicatingNetDevice::NeedsArp (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

bool
NonCommunicatingNetDevice::IsLinkUp (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

void
NonCommunicatingNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_FUNCTION (&callback);
}

void
NonCommunicatingNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION (&cb);
}

void
NonCommunicatingNetDevice::SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (&cb);
}

bool
NonCommunicatingNetDevice::SupportsSendFrom () const
{
  NS_LOG_FUNCTION (this);
  return false;
}


bool
NonCommunicatingNetDevice::Send (Ptr<Packet> packet,const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (packet << dest << protocolNumber);
  return false;
}

bool
NonCommunicatingNetDevice::SendFrom (Ptr<Packet> packet, const Address& src, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (packet << src << dest << protocolNumber);
  return false;
}


} // namespace ns3
