/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 *         Pavel Boyko <boyko@iitp.ru>
 */

#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/mesh-point-device.h"
#include "ns3/wifi-net-device.h"
#include "ns3/mesh-wifi-interface-mac.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MeshPointDevice");

NS_OBJECT_ENSURE_REGISTERED (MeshPointDevice);

TypeId
MeshPointDevice::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::MeshPointDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("Mesh")
    .AddConstructor<MeshPointDevice> ()
    .AddAttribute ("Mtu", "The MAC-level Maximum Transmission Unit",
                   UintegerValue (0xffff),
                   MakeUintegerAccessor (&MeshPointDevice::SetMtu,
                                         &MeshPointDevice::GetMtu),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ( "RoutingProtocol",
                    "The mesh routing protocol used by this mesh point.",
                    PointerValue (),
                    MakePointerAccessor (
                      &MeshPointDevice::GetRoutingProtocol, &MeshPointDevice::SetRoutingProtocol),
                    MakePointerChecker<
                      MeshL2RoutingProtocol> ())
    .AddAttribute ("ForwardingDelay",
                   "A random variable to account for processing time (microseconds) to forward a frame.",
                   StringValue ("ns3::UniformRandomVariable[Min=300.0|Max=400.0]"),
                   MakePointerAccessor (&MeshPointDevice::m_forwardingRandomVariable),
                   MakePointerChecker<RandomVariableStream> ());
  return tid;
}

MeshPointDevice::MeshPointDevice () :
  m_ifIndex (0)
{
  NS_LOG_FUNCTION (this);
  m_channel = CreateObject<BridgeChannel> ();
}

MeshPointDevice::~MeshPointDevice ()
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  m_channel = 0;
  m_routingProtocol = 0;
}

void
MeshPointDevice::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  for (std::vector<Ptr<NetDevice> >::iterator iter = m_ifaces.begin (); iter != m_ifaces.end (); iter++)
    {
      *iter = 0;
    }
  m_ifaces.clear ();
  m_node = 0;
  m_channel = 0;
  m_routingProtocol = 0;
  NetDevice::DoDispose ();

}

//-----------------------------------------------------------------------------
// NetDevice interface implementation
//-----------------------------------------------------------------------------

void
MeshPointDevice::ReceiveFromDevice (Ptr<NetDevice> incomingPort, Ptr<const Packet> packet, uint16_t protocol,
                                    Address const &src, Address const &dst, PacketType packetType)
{
  NS_LOG_FUNCTION (this << incomingPort << packet);
  NS_LOG_DEBUG ("UID is " << packet->GetUid ());
  const Mac48Address src48 = Mac48Address::ConvertFrom (src);
  const Mac48Address dst48 = Mac48Address::ConvertFrom (dst);
  uint16_t& realProtocol = protocol;
  NS_LOG_DEBUG ("SRC=" << src48 << ", DST = " << dst48 << ", I am: " << m_address);
  if (!m_promiscRxCallback.IsNull ())
    {
      m_promiscRxCallback (this, packet, protocol, src, dst, packetType);
    }
  if (dst48.IsGroup ())
    {
      Ptr<Packet> packet_copy = packet->Copy ();
      if (m_routingProtocol->RemoveRoutingStuff (incomingPort->GetIfIndex (), src48, dst48, packet_copy, realProtocol))
        {
          m_rxCallback (this, packet_copy, realProtocol, src);
          m_rxStats.broadcastData++;
          m_rxStats.broadcastDataBytes += packet->GetSize ();
          Time forwardingDelay = GetForwardingDelay ();
          NS_LOG_DEBUG ("Forwarding broadcast from " << src48 << " to " << dst48
            << " with delay " << forwardingDelay.As (Time::US));
          Simulator::Schedule (forwardingDelay, &MeshPointDevice::Forward, this, incomingPort, packet, protocol, src48, dst48); 
        }
      return;
    }
  if (dst48 == m_address)
    {
      Ptr<Packet> packet_copy = packet->Copy ();
      if (m_routingProtocol->RemoveRoutingStuff (incomingPort->GetIfIndex (), src48, dst48, packet_copy, realProtocol))
        {
          m_rxCallback (this, packet_copy, realProtocol, src);
          m_rxStats.unicastData++;
          m_rxStats.unicastDataBytes += packet->GetSize ();
        }
      return;
    }
  else
    {
      Time forwardingDelay = GetForwardingDelay ();
      Simulator::Schedule (forwardingDelay, &MeshPointDevice::Forward, this, incomingPort, packet->Copy (), protocol, src48, dst48); 
      NS_LOG_DEBUG ("Forwarding unicast from " << src48 << " to " << dst48 
        << " with delay " << forwardingDelay.As (Time::US));
    }
}

void
MeshPointDevice::Forward (Ptr<NetDevice> inport, Ptr<const Packet> packet, uint16_t protocol,
                          const Mac48Address src, const Mac48Address dst)
{
  NS_LOG_FUNCTION (this << inport << packet << protocol << src << dst);
  // pass through routing protocol
  NS_LOG_DEBUG ("Forwarding from " << src << " to " << dst << " at " << m_address);
  bool result = m_routingProtocol->RequestRoute (inport->GetIfIndex (), src, dst, packet, protocol, MakeCallback (
                                     &MeshPointDevice::DoSend, this));
  if (result == false)
    {
      NS_LOG_DEBUG ("Request to forward packet " << packet << " to destination " << dst << " failed; dropping packet");
    }
}

void
MeshPointDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION (this);
  m_ifIndex = index;
}

uint32_t
MeshPointDevice::GetIfIndex () const
{
  NS_LOG_FUNCTION (this);
  return m_ifIndex;
}

Ptr<Channel>
MeshPointDevice::GetChannel () const
{
  NS_LOG_FUNCTION (this);
  return m_channel;
}

Address
MeshPointDevice::GetAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}

void
MeshPointDevice::SetAddress (Address a)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Manual changing mesh point address can cause routing errors.");
  m_address = Mac48Address::ConvertFrom (a);
}

bool
MeshPointDevice::SetMtu (const uint16_t mtu)
{
  NS_LOG_FUNCTION (this);
  m_mtu = mtu;
  return true;
}

uint16_t
MeshPointDevice::GetMtu () const
{
  NS_LOG_FUNCTION (this);
  return m_mtu;
}

bool
MeshPointDevice::IsLinkUp () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

void
MeshPointDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_FUNCTION (this);
  // do nothing
  NS_LOG_WARN ("AddLinkChangeCallback does nothing");
}

bool
MeshPointDevice::IsBroadcast () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

Address
MeshPointDevice::GetBroadcast () const
{
  NS_LOG_FUNCTION (this);
  return Mac48Address ("ff:ff:ff:ff:ff:ff");
}

bool
MeshPointDevice::IsMulticast () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

Address
MeshPointDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (this << multicastGroup);
  Mac48Address multicast = Mac48Address::GetMulticast (multicastGroup);
  return multicast;
}

bool
MeshPointDevice::IsPointToPoint () const
{
  NS_LOG_FUNCTION (this);
  return false;
}

bool
MeshPointDevice::IsBridge () const
{
  NS_LOG_FUNCTION (this);
  return false;
}

bool
MeshPointDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this);
  const Mac48Address dst48 = Mac48Address::ConvertFrom (dest);
  return m_routingProtocol->RequestRoute (m_ifIndex, m_address, dst48, packet, protocolNumber, MakeCallback (
                                            &MeshPointDevice::DoSend, this));
}

bool
MeshPointDevice::SendFrom (Ptr<Packet> packet, const Address& src, const Address& dest,
                           uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this);
  const Mac48Address src48 = Mac48Address::ConvertFrom (src);
  const Mac48Address dst48 = Mac48Address::ConvertFrom (dest);
  return m_routingProtocol->RequestRoute (m_ifIndex, src48, dst48, packet, protocolNumber, MakeCallback (
                                            &MeshPointDevice::DoSend, this));
}

Ptr<Node>
MeshPointDevice::GetNode () const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

void
MeshPointDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);
  m_node = node;
}

bool
MeshPointDevice::NeedsArp () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

void
MeshPointDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION (this);
  m_rxCallback = cb;
}

void
MeshPointDevice::SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (this);
  m_promiscRxCallback = cb;
}

bool
MeshPointDevice::SupportsSendFrom () const
{
  NS_LOG_FUNCTION (this);
  return false; // don't allow to bridge mesh network with something else.
}

Address
MeshPointDevice::GetMulticast (Ipv6Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
  return Mac48Address::GetMulticast (addr);
}

//-----------------------------------------------------------------------------
// Interfaces
//-----------------------------------------------------------------------------
uint32_t
MeshPointDevice::GetNInterfaces () const
{
  NS_LOG_FUNCTION (this);
  return m_ifaces.size ();
}

Ptr<NetDevice>
MeshPointDevice::GetInterface (uint32_t n) const
{
  NS_LOG_FUNCTION (this << n);
  for (std::vector<Ptr<NetDevice> >::const_iterator i = m_ifaces.begin (); i != m_ifaces.end (); i++)
    {
      if ((*i)->GetIfIndex () == n)
        {
          return (*i);
        }
    }
  NS_FATAL_ERROR ("Mesh point interface is not found by index");
  return 0;
}
std::vector<Ptr<NetDevice> >
MeshPointDevice::GetInterfaces () const
{
  return m_ifaces;
}
void
MeshPointDevice::AddInterface (Ptr<NetDevice> iface)
{
  NS_LOG_FUNCTION (this << iface);

  NS_ASSERT (iface != this);
  if (!Mac48Address::IsMatchingType (iface->GetAddress ()))
    {
      NS_FATAL_ERROR ("Device does not support eui 48 addresses: cannot be used as a mesh point interface.");
    }
  if (!iface->SupportsSendFrom ())
    {
      NS_FATAL_ERROR ("Device does not support SendFrom: cannot be used as a mesh point interface.");
    }

  // Mesh point has MAC address of it's first interface
  if (m_ifaces.empty ())
    {
      m_address = Mac48Address::ConvertFrom (iface->GetAddress ());
    }
  Ptr<WifiNetDevice> wifiNetDev = iface->GetObject<WifiNetDevice> ();
  if (wifiNetDev == 0)
    {
      NS_FATAL_ERROR ("Device is not a WiFi NIC: cannot be used as a mesh point interface.");
    }
  Ptr<MeshWifiInterfaceMac> ifaceMac = wifiNetDev->GetMac ()->GetObject<MeshWifiInterfaceMac> ();
  if (ifaceMac == 0)
    {
      NS_FATAL_ERROR (
        "WiFi device doesn't have correct MAC installed: cannot be used as a mesh point interface.");
    }
  ifaceMac->SetMeshPointAddress (m_address);

  // Receive frames from this interface
  m_node->RegisterProtocolHandler (MakeCallback (&MeshPointDevice::ReceiveFromDevice, this), 0, iface, /*promiscuous = */
                                   true);
  m_ifaces.push_back (iface);
  m_channel->AddChannel (iface->GetChannel ());
}

//-----------------------------------------------------------------------------
// Protocols
//-----------------------------------------------------------------------------

void
MeshPointDevice::SetRoutingProtocol (Ptr<MeshL2RoutingProtocol> protocol)
{
  NS_LOG_FUNCTION (this << protocol);
  NS_ASSERT_MSG (PeekPointer (protocol->GetMeshPoint ()) == this,
                 "Routing protocol must be installed on mesh point to be useful.");
  m_routingProtocol = protocol;
}

Ptr<MeshL2RoutingProtocol>
MeshPointDevice::GetRoutingProtocol () const
{
  NS_LOG_FUNCTION (this);
  return m_routingProtocol;
}

void
MeshPointDevice::DoSend (bool success, Ptr<Packet> packet, Mac48Address src, Mac48Address dst,
                         uint16_t protocol, uint32_t outIface)
{
  NS_LOG_FUNCTION (this << success << packet << src << dst << protocol << outIface);
  if (!success)
    {
      NS_LOG_DEBUG ("Resolve failed");
      return;
    }

  // Count statistics
  Statistics * stats = ((src == m_address) ? &m_txStats : &m_fwdStats);

  if (dst.IsBroadcast ())
    {
      stats->broadcastData++;
      stats->broadcastDataBytes += packet->GetSize ();
    }
  else
    {
      stats->unicastData++;
      stats->unicastDataBytes += packet->GetSize ();
    }

  // Send
  if (outIface != 0xffffffff)
    {
      GetInterface (outIface)->SendFrom (packet, src, dst, protocol);
    }
  else
    {
      for (std::vector<Ptr<NetDevice> >::iterator i = m_ifaces.begin (); i != m_ifaces.end (); i++)
        {
          (*i)->SendFrom (packet->Copy (), src, dst, protocol);
        }
    }
}
MeshPointDevice::Statistics::Statistics () :
  unicastData (0), unicastDataBytes (0), broadcastData (0), broadcastDataBytes (0)
{
  NS_LOG_FUNCTION (this);
}

void
MeshPointDevice::Report (std::ostream & os) const
{
  NS_LOG_FUNCTION (this);
  os << "<Statistics" << std::endl <<
  "txUnicastData=\"" << m_txStats.unicastData << "\"" << std::endl <<
  "txUnicastDataBytes=\"" << m_txStats.unicastDataBytes << "\"" << std::endl <<
  "txBroadcastData=\"" << m_txStats.broadcastData << "\"" << std::endl <<
  "txBroadcastDataBytes=\"" << m_txStats.broadcastDataBytes << "\"" << std::endl <<
  "rxUnicastData=\"" << m_rxStats.unicastData << "\"" << std::endl <<
  "rxUnicastDataBytes=\"" << m_rxStats.unicastDataBytes << "\"" << std::endl <<
  "rxBroadcastData=\"" << m_rxStats.broadcastData << "\"" << std::endl <<
  "rxBroadcastDataBytes=\"" << m_rxStats.broadcastDataBytes << "\"" << std::endl <<
  "fwdUnicastData=\"" << m_fwdStats.unicastData << "\"" << std::endl <<
  "fwdUnicastDataBytes=\"" << m_fwdStats.unicastDataBytes << "\"" << std::endl <<
  "fwdBroadcastData=\"" << m_fwdStats.broadcastData << "\"" << std::endl <<
  "fwdBroadcastDataBytes=\"" << m_fwdStats.broadcastDataBytes << "\"" << std::endl <<
  "/>" << std::endl;
}

void
MeshPointDevice::ResetStats ()
{
  NS_LOG_FUNCTION (this);
  m_rxStats = Statistics ();
  m_txStats = Statistics ();
  m_fwdStats = Statistics ();
}

int64_t
MeshPointDevice::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_forwardingRandomVariable->SetStream (stream);
  return 1;
}

Time
MeshPointDevice::GetForwardingDelay () const
{
    return MicroSeconds (m_forwardingRandomVariable->GetInteger ());
}

} // namespace ns3
