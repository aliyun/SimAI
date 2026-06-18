/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: George F. Riley <riley@ece.gatech.edu>
// Author: Lalith Suresh <suresh.lalith@gmail.com>
//

#ifdef NS3_CLICK

#include "ipv4-l3-click-protocol.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/node.h"
#include "ns3/socket.h"
#include "ns3/ethernet-header.h"
#include "ns3/llc-snap-header.h"
#include "ns3/net-device.h"
#include "ns3/uinteger.h"
#include "ns3/object-vector.h"

#include "ns3/ipv4-raw-socket-impl.h"
#include "ns3/arp-l3-protocol.h"
#include "ns3/ip-l4-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/loopback-net-device.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4L3ClickProtocol");

const uint16_t Ipv4L3ClickProtocol::PROT_NUMBER = 0x0800;


NS_OBJECT_ENSURE_REGISTERED (Ipv4L3ClickProtocol);

TypeId
Ipv4L3ClickProtocol::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4L3ClickProtocol")
    .SetParent<Ipv4> ()
    .AddConstructor<Ipv4L3ClickProtocol> ()
    .SetGroupName ("Click")
    .AddAttribute ("DefaultTtl", "The TTL value set by default on all outgoing packets generated on this node.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&Ipv4L3ClickProtocol::m_defaultTtl),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("InterfaceList", "The set of Ipv4 interfaces associated to this Ipv4 stack.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&Ipv4L3ClickProtocol::m_interfaces),
                   MakeObjectVectorChecker<Ipv4Interface> ())
  ;
  return tid;
}

Ipv4L3ClickProtocol::Ipv4L3ClickProtocol ()
  : m_identification (0)
{
}

Ipv4L3ClickProtocol::~Ipv4L3ClickProtocol ()
{
}

void
Ipv4L3ClickProtocol::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  for (L4List_t::iterator i = m_protocols.begin (); i != m_protocols.end (); ++i)
    {
      i->second = 0;
    }
  m_protocols.clear ();

  for (Ipv4InterfaceList::iterator i = m_interfaces.begin (); i != m_interfaces.end (); ++i)
    {
      *i = 0;
    }
  m_interfaces.clear ();
  m_reverseInterfacesContainer.clear ();

  m_sockets.clear ();
  m_node = 0;
  m_routingProtocol = 0;
  Object::DoDispose ();
}

void
Ipv4L3ClickProtocol::NotifyNewAggregate ()
{
  if (m_node == 0)
    {
      Ptr<Node>node = this->GetObject<Node> ();
      // verify that it's a valid node and that
      // the node has not been set before
      if (node != 0)
        {
          this->SetNode (node);
        }
    }
  Ipv4::NotifyNewAggregate ();
}

void
Ipv4L3ClickProtocol::SetRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol)
{
  NS_LOG_FUNCTION (this);
  m_routingProtocol = routingProtocol;
  m_routingProtocol->SetIpv4 (this);
}


Ptr<Ipv4RoutingProtocol>
Ipv4L3ClickProtocol::GetRoutingProtocol (void) const
{
  return m_routingProtocol;
}

Ptr<Ipv4Interface>
Ipv4L3ClickProtocol::GetInterface (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  if (index < m_interfaces.size ())
    {
      return m_interfaces[index];
    }
  return 0;
}

uint32_t
Ipv4L3ClickProtocol::GetNInterfaces (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_interfaces.size ();
}

int32_t
Ipv4L3ClickProtocol::GetInterfaceForAddress (
  Ipv4Address address) const
{
  NS_LOG_FUNCTION (this << address);

  int32_t interface = 0;
  for (Ipv4InterfaceList::const_iterator i = m_interfaces.begin ();
       i != m_interfaces.end ();
       i++, interface++)
    {
      for (uint32_t j = 0; j < (*i)->GetNAddresses (); j++)
        {
          if ((*i)->GetAddress (j).GetLocal () == address)
            {
              return interface;
            }
        }
    }

  return -1;
}

int32_t
Ipv4L3ClickProtocol::GetInterfaceForPrefix (
  Ipv4Address address,
  Ipv4Mask mask) const
{
  NS_LOG_FUNCTION (this << address << mask);

  int32_t interface = 0;
  for (Ipv4InterfaceList::const_iterator i = m_interfaces.begin ();
       i != m_interfaces.end ();
       i++, interface++)
    {
      for (uint32_t j = 0; j < (*i)->GetNAddresses (); j++)
        {
          if ((*i)->GetAddress (j).GetLocal ().CombineMask (mask) == address.CombineMask (mask))
            {
              return interface;
            }
        }
    }

  return -1;
}

int32_t
Ipv4L3ClickProtocol::GetInterfaceForDevice (
  Ptr<const NetDevice> device) const
{
  NS_LOG_FUNCTION (this << device->GetIfIndex ());

  Ipv4InterfaceReverseContainer::const_iterator iter = m_reverseInterfacesContainer.find (device);
  if (iter != m_reverseInterfacesContainer.end ())
    {
      return (*iter).second;
    }

  return -1;
}

bool
Ipv4L3ClickProtocol::IsDestinationAddress (Ipv4Address address, uint32_t iif) const
{
  NS_LOG_FUNCTION (this << address << " " << iif);

  // First check the incoming interface for a unicast address match
  for (uint32_t i = 0; i < GetNAddresses (iif); i++)
    {
      Ipv4InterfaceAddress iaddr = GetAddress (iif, i);
      if (address == iaddr.GetLocal ())
        {
          NS_LOG_LOGIC ("For me (destination " << address << " match)");
          return true;
        }
      if (address == iaddr.GetBroadcast ())
        {
          NS_LOG_LOGIC ("For me (interface broadcast address)");
          return true;
        }
    }

  if (address.IsMulticast ())
    {
#ifdef NOTYET
      if (MulticastCheckGroup (iif, address ))
#endif
      if (true)
        {
          NS_LOG_LOGIC ("For me (Ipv4Addr multicast address");
          return true;
        }
    }

  if (address.IsBroadcast ())
    {
      NS_LOG_LOGIC ("For me (Ipv4Addr broadcast address)");
      return true;
    }

  if (GetWeakEsModel ())  // Check other interfaces
    {
      for (uint32_t j = 0; j < GetNInterfaces (); j++)
        {
          if (j == uint32_t (iif))
            {
              continue;
            }
          for (uint32_t i = 0; i < GetNAddresses (j); i++)
            {
              Ipv4InterfaceAddress iaddr = GetAddress (j, i);
              if (address == iaddr.GetLocal ())
                {
                  NS_LOG_LOGIC ("For me (destination " << address << " match) on another interface");
                  return true;
                }
              //  This is a small corner case:  match another interface's broadcast address
              if (address == iaddr.GetBroadcast ())
                {
                  NS_LOG_LOGIC ("For me (interface broadcast address on another interface)");
                  return true;
                }
            }
        }
    }
  return false;
}

void
Ipv4L3ClickProtocol::SetIpForward (bool forward)
{
  NS_LOG_FUNCTION (this << forward);
  m_ipForward = forward;
  for (Ipv4InterfaceList::const_iterator i = m_interfaces.begin (); i != m_interfaces.end (); i++)
    {
      (*i)->SetForwarding (forward);
    }
}

bool
Ipv4L3ClickProtocol::GetIpForward (void) const
{
  return m_ipForward;
}

void
Ipv4L3ClickProtocol::SetWeakEsModel (bool model)
{
  m_weakEsModel = model;
}

bool
Ipv4L3ClickProtocol::GetWeakEsModel (void) const
{
  return m_weakEsModel;
}

Ptr<NetDevice>
Ipv4L3ClickProtocol::GetNetDevice (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  return GetInterface (i)->GetDevice ();
}

void
Ipv4L3ClickProtocol::SetDefaultTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_defaultTtl = ttl;
}

void
Ipv4L3ClickProtocol::SetupLoopback (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  Ptr<Ipv4Interface> interface = CreateObject<Ipv4Interface> ();
  Ptr<LoopbackNetDevice> device = 0;
  // First check whether an existing LoopbackNetDevice exists on the node
  for (uint32_t i = 0; i < m_node->GetNDevices (); i++)
    {
      if ((device = DynamicCast<LoopbackNetDevice> (m_node->GetDevice (i))))
        {
          break;
        }
    }
  if (device == 0)
    {
      device = CreateObject<LoopbackNetDevice> ();
      m_node->AddDevice (device);
    }
  interface->SetDevice (device);
  interface->SetNode (m_node);
  Ipv4InterfaceAddress ifaceAddr = Ipv4InterfaceAddress (Ipv4Address::GetLoopback (), Ipv4Mask::GetLoopback ());
  interface->AddAddress (ifaceAddr);
  uint32_t index = AddIpv4Interface (interface);
  Ptr<Node> node = GetObject<Node> ();
  node->RegisterProtocolHandler (MakeCallback (&Ipv4L3ClickProtocol::Receive, this),
                                 Ipv4L3ClickProtocol::PROT_NUMBER, device);
  interface->SetUp ();
  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceUp (index);
    }
}

Ptr<Socket>
Ipv4L3ClickProtocol::CreateRawSocket (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv4RawSocketImpl> socket = CreateObject<Ipv4RawSocketImpl> ();
  socket->SetNode (m_node);
  m_sockets.push_back (socket);
  return socket;
}
void
Ipv4L3ClickProtocol::DeleteRawSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  for (SocketList::iterator i = m_sockets.begin (); i != m_sockets.end (); ++i)
    {
      if ((*i) == socket)
        {
          m_sockets.erase (i);
          return;
        }
    }
  return;
}


void
Ipv4L3ClickProtocol::SetNode (Ptr<Node> node)
{
  m_node = node;
  // Add a LoopbackNetDevice if needed, and an Ipv4Interface on top of it
  SetupLoopback ();
}

bool
Ipv4L3ClickProtocol::AddAddress (uint32_t i, Ipv4InterfaceAddress address)
{
  NS_LOG_FUNCTION (this << i << address);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  bool retVal = interface->AddAddress (address);
  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyAddAddress (i, address);
    }
  return retVal;
}

Ipv4InterfaceAddress
Ipv4L3ClickProtocol::GetAddress (uint32_t interfaceIndex, uint32_t addressIndex) const
{
  NS_LOG_FUNCTION (this << interfaceIndex << addressIndex);
  Ptr<Ipv4Interface> interface = GetInterface (interfaceIndex);
  return interface->GetAddress (addressIndex);
}

uint32_t
Ipv4L3ClickProtocol::GetNAddresses (uint32_t interface) const
{
  NS_LOG_FUNCTION (this << interface);
  Ptr<Ipv4Interface> iface = GetInterface (interface);
  return iface->GetNAddresses ();
}

bool
Ipv4L3ClickProtocol::RemoveAddress (uint32_t i, uint32_t addressIndex)
{
  NS_LOG_FUNCTION (this << i << addressIndex);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  Ipv4InterfaceAddress address = interface->RemoveAddress (addressIndex);
  if (address != Ipv4InterfaceAddress ())
    {
      if (m_routingProtocol != 0)
        {
          m_routingProtocol->NotifyRemoveAddress (i, address);
        }
      return true;
    }
  return false;
}

bool
Ipv4L3ClickProtocol::RemoveAddress (uint32_t i, Ipv4Address address)
{
  NS_LOG_FUNCTION (this << i << address);

  if (address == Ipv4Address::GetLoopback())
    {
      NS_LOG_WARN ("Cannot remove loopback address.");
      return false;
    }
  Ptr<Ipv4Interface> interface = GetInterface (i);
  Ipv4InterfaceAddress ifAddr = interface->RemoveAddress (address);
  if (ifAddr != Ipv4InterfaceAddress ())
    {
      if (m_routingProtocol != 0)
        {
          m_routingProtocol->NotifyRemoveAddress (i, ifAddr);
        }
      return true;
    }
  return false;
}

Ipv4Address
Ipv4L3ClickProtocol::SourceAddressSelection (uint32_t interfaceIdx, Ipv4Address dest)
{
  NS_LOG_FUNCTION (this << interfaceIdx << " " << dest);
  if (GetNAddresses (interfaceIdx) == 1)  // common case
    {
      return GetAddress (interfaceIdx, 0).GetLocal ();
    }
  // no way to determine the scope of the destination, so adopt the
  // following rule:  pick the first available address (index 0) unless
  // a subsequent address is on link (in which case, pick the primary
  // address if there are multiple)
  Ipv4Address candidate = GetAddress (interfaceIdx, 0).GetLocal ();
  for (uint32_t i = 0; i < GetNAddresses (interfaceIdx); i++)
    {
      Ipv4InterfaceAddress test = GetAddress (interfaceIdx, i);
      if (test.GetLocal ().CombineMask (test.GetMask ()) == dest.CombineMask (test.GetMask ()))
        {
          if (test.IsSecondary () == false)
            {
              return test.GetLocal ();
            }
        }
    }
  return candidate;
}

Ipv4Address
Ipv4L3ClickProtocol::SelectSourceAddress (Ptr<const NetDevice> device,
                                          Ipv4Address dst, Ipv4InterfaceAddress::InterfaceAddressScope_e scope)
{
  NS_LOG_FUNCTION (device << dst << scope);
  Ipv4Address addr ("0.0.0.0");
  Ipv4InterfaceAddress iaddr;
  bool found = false;

  if (device != 0)
    {
      int32_t i = GetInterfaceForDevice (device);
      NS_ASSERT_MSG (i >= 0, "No device found on node");
      for (uint32_t j = 0; j < GetNAddresses (i); j++)
        {
          iaddr = GetAddress (i, j);
          if (iaddr.IsSecondary ())
            {
              continue;
            }
          if (iaddr.GetScope () > scope)
            {
              continue;
            }
          if (dst.CombineMask (iaddr.GetMask ())  == iaddr.GetLocal ().CombineMask (iaddr.GetMask ()) )
            {
              return iaddr.GetLocal ();
            }
          if (!found)
            {
              addr = iaddr.GetLocal ();
              found = true;
            }
        }
    }
  if (found)
    {
      return addr;
    }

  // Iterate among all interfaces
  for (uint32_t i = 0; i < GetNInterfaces (); i++)
    {
      for (uint32_t j = 0; j < GetNAddresses (i); j++)
        {
          iaddr = GetAddress (i, j);
          if (iaddr.IsSecondary ())
            {
              continue;
            }
          if (iaddr.GetScope () != Ipv4InterfaceAddress::LINK
              && iaddr.GetScope () <= scope)
            {
              return iaddr.GetLocal ();
            }
        }
    }
  NS_LOG_WARN ("Could not find source address for " << dst << " and scope "
                                                    << scope << ", returning 0");
  return addr;
}

void
Ipv4L3ClickProtocol::SetMetric (uint32_t i, uint16_t metric)
{
  NS_LOG_FUNCTION (i << metric);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  interface->SetMetric (metric);
}

uint16_t
Ipv4L3ClickProtocol::GetMetric (uint32_t i) const
{
  NS_LOG_FUNCTION (i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->GetMetric ();
}

uint16_t
Ipv4L3ClickProtocol::GetMtu (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->GetDevice ()->GetMtu ();
}

bool
Ipv4L3ClickProtocol::IsUp (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->IsUp ();
}

void
Ipv4L3ClickProtocol::SetUp (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  interface->SetUp ();

  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceUp (i);
    }
}

void
Ipv4L3ClickProtocol::SetDown (uint32_t ifaceIndex)
{
  NS_LOG_FUNCTION (this << ifaceIndex);
  Ptr<Ipv4Interface> interface = GetInterface (ifaceIndex);
  interface->SetDown ();

  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceDown (ifaceIndex);
    }
}

bool
Ipv4L3ClickProtocol::IsForwarding (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  NS_LOG_LOGIC ("Forwarding state: " << interface->IsForwarding ());
  return interface->IsForwarding ();
}

void
Ipv4L3ClickProtocol::SetForwarding (uint32_t i, bool val)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  interface->SetForwarding (val);
}

void
Ipv4L3ClickProtocol::SetPromisc (uint32_t i)
{
  NS_ASSERT (i <= m_node->GetNDevices ());
  Ptr<NetDevice> netdev = GetNetDevice (i);
  NS_ASSERT (netdev);
  Ptr<Node> node = GetObject<Node> ();
  NS_ASSERT (node);
  node->RegisterProtocolHandler (MakeCallback (&Ipv4L3ClickProtocol::Receive, this),
                                 0, netdev,true);
}

uint32_t
Ipv4L3ClickProtocol::AddInterface (Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << &device);
  Ptr<Node> node = GetObject<Node> ();
  node->RegisterProtocolHandler (MakeCallback (&Ipv4L3ClickProtocol::Receive, this),
                                 Ipv4L3ClickProtocol::PROT_NUMBER, device);
  node->RegisterProtocolHandler (MakeCallback (&Ipv4L3ClickProtocol::Receive, this),
                                 ArpL3Protocol::PROT_NUMBER, device);

  Ptr<Ipv4Interface> interface = CreateObject<Ipv4Interface> ();
  interface->SetNode (m_node);
  interface->SetDevice (device);
  interface->SetForwarding (m_ipForward);
  return AddIpv4Interface (interface);
}

uint32_t
Ipv4L3ClickProtocol::AddIpv4Interface (Ptr<Ipv4Interface>interface)
{
  NS_LOG_FUNCTION (this << interface);
  uint32_t index = m_interfaces.size ();
  m_interfaces.push_back (interface);
  m_reverseInterfacesContainer[interface->GetDevice ()] = index;
  return index;
}

/// \todo when should we set ip_id?   check whether we are incrementing
/// m_identification on packets that may later be dropped in this stack
/// and whether that deviates from Linux
Ipv4Header
Ipv4L3ClickProtocol::BuildHeader (
  Ipv4Address source,
  Ipv4Address destination,
  uint8_t protocol,
  uint16_t payloadSize,
  uint8_t ttl,
  bool mayFragment)
{
  NS_LOG_FUNCTION_NOARGS ();
  Ipv4Header ipHeader;
  ipHeader.SetSource (source);
  ipHeader.SetDestination (destination);
  ipHeader.SetProtocol (protocol);
  ipHeader.SetPayloadSize (payloadSize);
  ipHeader.SetTtl (ttl);
  if (mayFragment == true)
    {
      ipHeader.SetMayFragment ();
      ipHeader.SetIdentification (m_identification);
      m_identification++;
    }
  else
    {
      ipHeader.SetDontFragment ();
      // TBD:  set to zero here; will cause traces to change
      ipHeader.SetIdentification (m_identification);
      m_identification++;
    }
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  return ipHeader;
}

void
Ipv4L3ClickProtocol::Send (Ptr<Packet> packet,
                           Ipv4Address source,
                           Ipv4Address destination,
                           uint8_t protocol,
                           Ptr<Ipv4Route> route)
{
  NS_LOG_FUNCTION (this << packet << source << destination << uint32_t (protocol) << route);

  Ipv4Header ipHeader;
  bool mayFragment = true;
  uint8_t ttl = m_defaultTtl;
  SocketIpTtlTag tag;
  bool found = packet->RemovePacketTag (tag);
  if (found)
    {
      ttl = tag.GetTtl ();
    }

  ipHeader = BuildHeader (source, destination, protocol, packet->GetSize (), ttl, mayFragment);
  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (m_routingProtocol);
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  packet->AddHeader (ipHeader);
  click->Send (packet->Copy (), source, destination);
  return;
}

void
Ipv4L3ClickProtocol::SendWithHeader (Ptr<Packet> packet,
                                     Ipv4Header ipHeader,
                                     Ptr<Ipv4Route> route)
{
  NS_LOG_FUNCTION (this << packet << ipHeader << route);

  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (m_routingProtocol);
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  packet->AddHeader (ipHeader);
  click->Send (packet->Copy (), ipHeader.GetSource (), ipHeader.GetDestination ());
}

void
Ipv4L3ClickProtocol::SendDown (Ptr<Packet> p, int ifid)
{
  // Called by Ipv4ClickRouting.

  // NetDevice::Send () attaches ethernet headers,
  // so the one that Click attaches isn't required
  // but we need the destination address and
  // protocol values from the header.

  Ptr<NetDevice> netdev = GetNetDevice (ifid);

  EthernetHeader header;
  p->RemoveHeader (header);

  uint16_t protocol;

  if (header.GetLengthType () <= 1500)
    {
      LlcSnapHeader llc;
      p->RemoveHeader (llc);
      protocol = llc.GetType ();
    }
  else
    {
      protocol = header.GetLengthType ();
    }

  // Use the destination address and protocol obtained
  // from above to send the packet.
  netdev->Send (p, header.GetDestination (), protocol);
}

void
Ipv4L3ClickProtocol::Receive ( Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from,
                               const Address &to, NetDevice::PacketType packetType)
{
  NS_LOG_FUNCTION (this << device << p << from << to);

  NS_LOG_LOGIC ("Packet from " << from << " received on node " <<
                m_node->GetId ());

  // Forward packet to raw sockets, if any
  if (protocol == Ipv4L3ClickProtocol::PROT_NUMBER && m_sockets.size () > 0)
    {
      Ptr<Packet> packetForRawSocket = p->Copy ();
      int32_t interface = GetInterfaceForDevice(device);
      NS_ASSERT_MSG (interface != -1, "Received a packet from an interface that is not known to IPv4");
      Ptr<Ipv4Interface> ipv4Interface = m_interfaces[interface];
      if (!ipv4Interface->IsUp ())
        {
          NS_LOG_LOGIC ("Dropping received packet -- interface is down");
          return;
        }

      Ipv4Header ipHeader;
      if (Node::ChecksumEnabled ())
        {
          ipHeader.EnableChecksum ();
        }
      packetForRawSocket->RemoveHeader (ipHeader);


      for (SocketList::iterator i = m_sockets.begin (); i != m_sockets.end (); ++i)
        {
          NS_LOG_LOGIC ("Forwarding to raw socket");
          Ptr<Ipv4RawSocketImpl> socket = *i;
          socket->ForwardUp (packetForRawSocket, ipHeader, ipv4Interface);
        }
    }

  Ptr<Packet> packet = p->Copy ();

  // Add an ethernet frame. This allows
  // Click to work with csma and wifi
  EthernetHeader hdr;
  hdr.SetSource (Mac48Address::ConvertFrom (from));
  hdr.SetDestination (Mac48Address::ConvertFrom (to));
  hdr.SetLengthType (protocol);
  packet->AddHeader (hdr);

  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (GetRoutingProtocol ());
  click->Receive (packet->Copy (), Mac48Address::ConvertFrom (device->GetAddress ()), Mac48Address::ConvertFrom (to));
}

void
Ipv4L3ClickProtocol::LocalDeliver (Ptr<const Packet> packet, Ipv4Header const&ip, uint32_t iif)
{
  NS_LOG_FUNCTION (this << packet << &ip);
  Ptr<Packet> p = packet->Copy (); // need to pass a non-const packet up

  m_localDeliverTrace (ip, packet, iif);

  Ptr<IpL4Protocol> protocol = GetProtocol (ip.GetProtocol ());
  if (protocol != 0)
    {
      // we need to make a copy in the unlikely event we hit the
      // RX_ENDPOINT_UNREACH codepath
      Ptr<Packet> copy = p->Copy ();
      enum IpL4Protocol::RxStatus status =
        protocol->Receive (p, ip, GetInterface (iif));
      switch (status)
        {
        case IpL4Protocol::RX_OK:
        // fall through
        case IpL4Protocol::RX_ENDPOINT_CLOSED:
        // fall through
        case IpL4Protocol::RX_CSUM_FAILED:
          break;
        case IpL4Protocol::RX_ENDPOINT_UNREACH:
          if (ip.GetDestination ().IsBroadcast () == true
              || ip.GetDestination ().IsMulticast () == true)
            {
              break; // Do not reply to broadcast or multicast
            }
          // Another case to suppress ICMP is a subnet-directed broadcast
          bool subnetDirected = false;
          for (uint32_t i = 0; i < GetNAddresses (iif); i++)
            {
              Ipv4InterfaceAddress addr = GetAddress (iif, i);
              if (addr.GetLocal ().CombineMask (addr.GetMask ()) == ip.GetDestination ().CombineMask (addr.GetMask ())
                  && ip.GetDestination ().IsSubnetDirectedBroadcast (addr.GetMask ()))
                {
                  subnetDirected = true;
                }
            }
          if (subnetDirected == false)
            {
              GetIcmp ()->SendDestUnreachPort (ip, copy);
            }
        }
    }
}

Ptr<Icmpv4L4Protocol>
Ipv4L3ClickProtocol::GetIcmp (void) const
{
  Ptr<IpL4Protocol> prot = GetProtocol (Icmpv4L4Protocol::GetStaticProtocolNumber ());
  if (prot != 0)
    {
      return prot->GetObject<Icmpv4L4Protocol> ();
    }
  else
    {
      return 0;
    }
}

void
Ipv4L3ClickProtocol::Insert (Ptr<IpL4Protocol> protocol)
{
  NS_LOG_FUNCTION (this << protocol);
  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), -1);
  if (m_protocols.find (key) != m_protocols.end ())
    {
      NS_LOG_WARN ("Overwriting default protocol " << int(protocol->GetProtocolNumber ()));
    }
  m_protocols[key] = protocol;
}

void
Ipv4L3ClickProtocol::Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
{
  NS_LOG_FUNCTION (this << protocol << interfaceIndex);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), interfaceIndex);
  if (m_protocols.find (key) != m_protocols.end ())
    {
      NS_LOG_WARN ("Overwriting protocol " << int(protocol->GetProtocolNumber ()) << " on interface " << int(interfaceIndex));
    }
  m_protocols[key] = protocol;
}

void
Ipv4L3ClickProtocol::Remove (Ptr<IpL4Protocol> protocol)
{
  NS_LOG_FUNCTION (this << protocol);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), -1);
  L4List_t::iterator iter = m_protocols.find (key);
  if (iter == m_protocols.end ())
    {
      NS_LOG_WARN ("Trying to remove an non-existent default protocol " << int(protocol->GetProtocolNumber ()));
    }
  else
    {
      m_protocols.erase (key);
    }
}

void
Ipv4L3ClickProtocol::Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
{
  NS_LOG_FUNCTION (this << protocol << interfaceIndex);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), interfaceIndex);
  L4List_t::iterator iter = m_protocols.find (key);
  if (iter == m_protocols.end ())
    {
      NS_LOG_WARN ("Trying to remove an non-existent protocol " << int(protocol->GetProtocolNumber ()) << " on interface " << int(interfaceIndex));
    }
  else
    {
      m_protocols.erase (key);
    }
}

Ptr<IpL4Protocol>
Ipv4L3ClickProtocol::GetProtocol (int protocolNumber) const
{
  NS_LOG_FUNCTION (this << protocolNumber);

  return GetProtocol (protocolNumber, -1);
}

Ptr<IpL4Protocol>
Ipv4L3ClickProtocol::GetProtocol (int protocolNumber, int32_t interfaceIndex) const
{
  NS_LOG_FUNCTION (this << protocolNumber << interfaceIndex);

  L4ListKey_t key;
  L4List_t::const_iterator i;
  if (interfaceIndex >= 0)
    {
      // try the interface-specific protocol.
      key = std::make_pair (protocolNumber, interfaceIndex);
      i = m_protocols.find (key);
      if (i != m_protocols.end ())
        {
          return i->second;
        }
    }
  // try the generic protocol.
  key = std::make_pair (protocolNumber, -1);
  i = m_protocols.find (key);
  if (i != m_protocols.end ())
    {
      return i->second;
    }

  return 0;
}

} // namespace ns3

#endif // NS3_CLICK
