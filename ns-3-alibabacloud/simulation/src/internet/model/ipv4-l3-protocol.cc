// -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*-
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
// Author: George F. Riley<riley@ece.gatech.edu>
//

#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/callback.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-route.h"
#include "ns3/node.h"
#include "ns3/socket.h"
#include "ns3/net-device.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/object-vector.h"
#include "ns3/ipv4-header.h"
#include "ns3/boolean.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/traffic-control-layer.h"

#include "loopback-net-device.h"
#include "arp-l3-protocol.h"
#include "arp-cache.h"
#include "ipv4-l3-protocol.h"
#include "icmpv4-l4-protocol.h"
#include "ipv4-interface.h"
#include "ipv4-raw-socket-impl.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4L3Protocol");

const uint16_t Ipv4L3Protocol::PROT_NUMBER = 0x0800;

NS_OBJECT_ENSURE_REGISTERED (Ipv4L3Protocol);

TypeId 
Ipv4L3Protocol::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4L3Protocol")
    .SetParent<Ipv4> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv4L3Protocol> ()
    .AddAttribute ("DefaultTtl",
                   "The TTL value set by default on "
                   "all outgoing packets generated on this node.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&Ipv4L3Protocol::m_defaultTtl),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("FragmentExpirationTimeout",
                   "When this timeout expires, the fragments "
                   "will be cleared from the buffer.",
                   TimeValue (Seconds (30)),
                   MakeTimeAccessor (&Ipv4L3Protocol::m_fragmentExpirationTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("EnableDuplicatePacketDetection",
                   "Enable multicast duplicate packet detection based on RFC 6621",
                   BooleanValue (false),
                   MakeBooleanAccessor (&Ipv4L3Protocol::m_enableDpd),
                   MakeBooleanChecker ())
    .AddAttribute ("DuplicateExpire", "Expiration delay for duplicate cache entries",
                   TimeValue (MilliSeconds (1)),
                   MakeTimeAccessor (&Ipv4L3Protocol::m_expire),
                   MakeTimeChecker ())
    .AddAttribute ("PurgeExpiredPeriod", 
                   "Time between purges of expired duplicate packet entries, "
                   "0 means never purge",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&Ipv4L3Protocol::m_purge),
                   MakeTimeChecker (Seconds (0)))
    .AddTraceSource ("Tx",
                     "Send ipv4 packet to outgoing interface.",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_txTrace),
                     "ns3::Ipv4L3Protocol::TxRxTracedCallback")
    .AddTraceSource ("Rx",
                     "Receive ipv4 packet from incoming interface.",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_rxTrace),
                     "ns3::Ipv4L3Protocol::TxRxTracedCallback")
    .AddTraceSource ("Drop",
                     "Drop ipv4 packet",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_dropTrace),
                     "ns3::Ipv4L3Protocol::DropTracedCallback")
    .AddAttribute ("InterfaceList",
                   "The set of Ipv4 interfaces associated to this Ipv4 stack.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&Ipv4L3Protocol::m_interfaces),
                   MakeObjectVectorChecker<Ipv4Interface> ())

    .AddTraceSource ("SendOutgoing",
                     "A newly-generated packet by this node is "
                     "about to be queued for transmission",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_sendOutgoingTrace),
                     "ns3::Ipv4L3Protocol::SentTracedCallback")
    .AddTraceSource ("UnicastForward",
                     "A unicast IPv4 packet was received by this node "
                     "and is being forwarded to another node",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_unicastForwardTrace),
                     "ns3::Ipv4L3Protocol::SentTracedCallback")
    .AddTraceSource ("MulticastForward",
                     "A multicast IPv4 packet was received by this node "
                     "and is being forwarded to another node",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_multicastForwardTrace),
                     "ns3::Ipv4L3Protocol::SentTracedCallback")
    .AddTraceSource ("LocalDeliver",
                     "An IPv4 packet was received by/for this node, "
                     "and it is being forward up the stack",
                     MakeTraceSourceAccessor (&Ipv4L3Protocol::m_localDeliverTrace),
                     "ns3::Ipv4L3Protocol::SentTracedCallback")

  ;
  return tid;
}

Ipv4L3Protocol::Ipv4L3Protocol()
{
  NS_LOG_FUNCTION (this);
}

Ipv4L3Protocol::~Ipv4L3Protocol ()
{
  NS_LOG_FUNCTION (this);
}

void
Ipv4L3Protocol::Insert (Ptr<IpL4Protocol> protocol)
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
Ipv4L3Protocol::Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
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
Ipv4L3Protocol::Remove (Ptr<IpL4Protocol> protocol)
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
Ipv4L3Protocol::Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
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
Ipv4L3Protocol::GetProtocol (int protocolNumber) const
{
  NS_LOG_FUNCTION (this << protocolNumber);

  return GetProtocol (protocolNumber, -1);
}

Ptr<IpL4Protocol>
Ipv4L3Protocol::GetProtocol (int protocolNumber, int32_t interfaceIndex) const
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

void
Ipv4L3Protocol::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
  // Add a LoopbackNetDevice if needed, and an Ipv4Interface on top of it
  SetupLoopback ();
}

Ptr<Socket> 
Ipv4L3Protocol::CreateRawSocket (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv4RawSocketImpl> socket = CreateObject<Ipv4RawSocketImpl> ();
  socket->SetNode (m_node);
  m_sockets.push_back (socket);
  return socket;
}
void 
Ipv4L3Protocol::DeleteRawSocket (Ptr<Socket> socket)
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
/*
 * This method is called by AggregateObject and completes the aggregation
 * by setting the node in the ipv4 stack
 */
void
Ipv4L3Protocol::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);
  if (m_node == 0)
    {
      Ptr<Node>node = this->GetObject<Node>();
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
Ipv4L3Protocol::SetRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol)
{
  NS_LOG_FUNCTION (this << routingProtocol);
  m_routingProtocol = routingProtocol;
  m_routingProtocol->SetIpv4 (this);
}


Ptr<Ipv4RoutingProtocol> 
Ipv4L3Protocol::GetRoutingProtocol (void) const
{
  NS_LOG_FUNCTION (this);
  return m_routingProtocol;
}

void 
Ipv4L3Protocol::DoDispose (void)
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

  for (MapFragments_t::iterator it = m_fragments.begin (); it != m_fragments.end (); it++)
    {
      it->second = 0;
    }

  m_fragments.clear ();
  m_timeoutEventList.clear ();
  if (m_timeoutEvent.IsRunning ())
    {
      m_timeoutEvent.Cancel ();
    }

  if (m_cleanDpd.IsRunning ())
    {
      m_cleanDpd.Cancel ();
    }
  m_dups.clear ();

  Object::DoDispose ();
}

void
Ipv4L3Protocol::SetupLoopback (void)
{
  NS_LOG_FUNCTION (this);

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
  node->RegisterProtocolHandler (MakeCallback (&Ipv4L3Protocol::Receive, this), 
                                 Ipv4L3Protocol::PROT_NUMBER, device);
  interface->SetUp ();
  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceUp (index);
    }
}

void 
Ipv4L3Protocol::SetDefaultTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (ttl));
  m_defaultTtl = ttl;
}

uint32_t 
Ipv4L3Protocol::AddInterface (Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  NS_ASSERT (m_node != 0);

  Ptr<TrafficControlLayer> tc = m_node->GetObject<TrafficControlLayer> ();

  NS_ASSERT (tc != 0);

  m_node->RegisterProtocolHandler (MakeCallback (&TrafficControlLayer::Receive, tc),
                                   Ipv4L3Protocol::PROT_NUMBER, device);
  m_node->RegisterProtocolHandler (MakeCallback (&TrafficControlLayer::Receive, tc),
                                   ArpL3Protocol::PROT_NUMBER, device);

  tc->RegisterProtocolHandler (MakeCallback (&Ipv4L3Protocol::Receive, this),
                               Ipv4L3Protocol::PROT_NUMBER, device);
  tc->RegisterProtocolHandler (MakeCallback (&ArpL3Protocol::Receive, PeekPointer (GetObject<ArpL3Protocol> ())),
                               ArpL3Protocol::PROT_NUMBER, device);

  Ptr<Ipv4Interface> interface = CreateObject<Ipv4Interface> ();
  interface->SetNode (m_node);
  interface->SetDevice (device);
  interface->SetTrafficControl (tc);
  interface->SetForwarding (m_ipForward);
  return AddIpv4Interface (interface);
}

uint32_t 
Ipv4L3Protocol::AddIpv4Interface (Ptr<Ipv4Interface>interface)
{
  NS_LOG_FUNCTION (this << interface);
  uint32_t index = m_interfaces.size ();
  m_interfaces.push_back (interface);
  m_reverseInterfacesContainer[interface->GetDevice ()] = index;
  return index;
}

Ptr<Ipv4Interface>
Ipv4L3Protocol::GetInterface (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  if (index < m_interfaces.size ())
    {
      return m_interfaces[index];
    }
  return 0;
}

uint32_t 
Ipv4L3Protocol::GetNInterfaces (void) const
{
  NS_LOG_FUNCTION (this);
  return m_interfaces.size ();
}

int32_t 
Ipv4L3Protocol::GetInterfaceForAddress (
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
Ipv4L3Protocol::GetInterfaceForPrefix (
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
Ipv4L3Protocol::GetInterfaceForDevice (
  Ptr<const NetDevice> device) const
{
  NS_LOG_FUNCTION (this << device);

  Ipv4InterfaceReverseContainer::const_iterator iter = m_reverseInterfacesContainer.find (device);
  if (iter != m_reverseInterfacesContainer.end ())
    {
      return (*iter).second;
    }

  return -1;
}

bool
Ipv4L3Protocol::IsDestinationAddress (Ipv4Address address, uint32_t iif) const
{
  NS_LOG_FUNCTION (this << address << iif);
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
          NS_LOG_LOGIC ("For me (Ipv4Addr multicast address)");
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
          if (j == uint32_t (iif)) continue;
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
Ipv4L3Protocol::Receive ( Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from,
                          const Address &to, NetDevice::PacketType packetType)
{
  NS_LOG_FUNCTION (this << device << p << protocol << from << to << packetType);

  NS_LOG_LOGIC ("Packet from " << from << " received on node " << 
                m_node->GetId ());


  int32_t interface = GetInterfaceForDevice(device);
  NS_ASSERT_MSG (interface != -1, "Received a packet from an interface that is not known to IPv4");

  Ptr<Packet> packet = p->Copy ();

  Ptr<Ipv4Interface> ipv4Interface = m_interfaces[interface];

  if (ipv4Interface->IsUp ())
    {
      m_rxTrace (packet, this, interface);
    }
  else
    {
      NS_LOG_LOGIC ("Dropping received packet -- interface is down");
      Ipv4Header ipHeader;
      packet->RemoveHeader (ipHeader);
      m_dropTrace (ipHeader, packet, DROP_INTERFACE_DOWN, this, interface);
      return;
    }

  Ipv4Header ipHeader;
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  packet->RemoveHeader (ipHeader);

  // Trim any residual frame padding from underlying devices
  if (ipHeader.GetPayloadSize () < packet->GetSize ())
    {
      packet->RemoveAtEnd (packet->GetSize () - ipHeader.GetPayloadSize ());
    }

  if (!ipHeader.IsChecksumOk ()) 
    {
      NS_LOG_LOGIC ("Dropping received packet -- checksum not ok");
      m_dropTrace (ipHeader, packet, DROP_BAD_CHECKSUM, this, interface);
      return;
    }

  // the packet is valid, we update the ARP cache entry (if present)
  Ptr<ArpCache> arpCache = ipv4Interface->GetArpCache ();
  if (arpCache)
    {
      // case one, it's a a direct routing.
      ArpCache::Entry *entry = arpCache->Lookup (ipHeader.GetSource ());
      if (entry)
        {
          if (entry->IsAlive ())
            {
              entry->UpdateSeen ();
            }
        }
      else
        {
          // It's not in the direct routing, so it's the router, and it could have multiple IP addresses.
          // In doubt, update all of them.
          // Note: it's a confirmed behavior for Linux routers.
          std::list<ArpCache::Entry *> entryList = arpCache->LookupInverse (from);
          std::list<ArpCache::Entry *>::iterator iter;
          for (iter = entryList.begin (); iter != entryList.end (); iter ++)
            {
              if ((*iter)->IsAlive ())
                {
                  (*iter)->UpdateSeen ();
                }
            }
        }
    }

  for (SocketList::iterator i = m_sockets.begin (); i != m_sockets.end (); ++i)
    {
      NS_LOG_LOGIC ("Forwarding to raw socket"); 
      Ptr<Ipv4RawSocketImpl> socket = *i;
      socket->ForwardUp (packet, ipHeader, ipv4Interface);
    }

  if (m_enableDpd && ipHeader.GetDestination ().IsMulticast () && UpdateDuplicate (packet, ipHeader))
    {
      NS_LOG_LOGIC ("Dropping received packet -- duplicate.");
      m_dropTrace (ipHeader, packet, DROP_DUPLICATE, this, interface);
      return;
    }

  NS_ASSERT_MSG (m_routingProtocol != 0, "Need a routing protocol object to process packets");
  if (!m_routingProtocol->RouteInput (packet, ipHeader, device,
                                      MakeCallback (&Ipv4L3Protocol::IpForward, this),
                                      MakeCallback (&Ipv4L3Protocol::IpMulticastForward, this),
                                      MakeCallback (&Ipv4L3Protocol::LocalDeliver, this),
                                      MakeCallback (&Ipv4L3Protocol::RouteInputError, this)
                                      ))
    {
      NS_LOG_WARN ("No route found for forwarding packet.  Drop.");
      m_dropTrace (ipHeader, packet, DROP_NO_ROUTE, this, interface);
    }
}

Ptr<Icmpv4L4Protocol> 
Ipv4L3Protocol::GetIcmp (void) const
{
  NS_LOG_FUNCTION (this);
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

bool
Ipv4L3Protocol::IsUnicast (Ipv4Address ad) const
{
  NS_LOG_FUNCTION (this << ad);

  if (ad.IsBroadcast () || ad.IsMulticast ())
    {
      return false;
    }
  else
    {
      // check for subnet-broadcast
      for (uint32_t ifaceIndex = 0; ifaceIndex < GetNInterfaces (); ifaceIndex++)
        {
          for (uint32_t j = 0; j < GetNAddresses (ifaceIndex); j++)
            {
              Ipv4InterfaceAddress ifAddr = GetAddress (ifaceIndex, j);
              NS_LOG_LOGIC ("Testing address " << ad << " with subnet-directed broadcast " << ifAddr.GetBroadcast () );
              if (ad == ifAddr.GetBroadcast () )
                {
                  return false;
                }
            }
        }
    }

  return true;
}

bool
Ipv4L3Protocol::IsUnicast (Ipv4Address ad, Ipv4Mask interfaceMask) const
{
  NS_LOG_FUNCTION (this << ad << interfaceMask);
  return !ad.IsMulticast () && !ad.IsSubnetDirectedBroadcast (interfaceMask);
}

void 
Ipv4L3Protocol::SendWithHeader (Ptr<Packet> packet, 
                                Ipv4Header ipHeader,
                                Ptr<Ipv4Route> route)
{
  NS_LOG_FUNCTION (this << packet << ipHeader << route);
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  SendRealOut (route, packet, ipHeader);
}

void
Ipv4L3Protocol::CallTxTrace (const Ipv4Header & ipHeader, Ptr<Packet> packet,
                                    Ptr<Ipv4> ipv4, uint32_t interface)
{
  if (!m_txTrace.IsEmpty ())
    {
      Ptr<Packet> packetCopy = packet->Copy ();
      packetCopy->AddHeader (ipHeader);
      m_txTrace (packetCopy, ipv4, interface);
    }
}

void 
Ipv4L3Protocol::Send (Ptr<Packet> packet, 
                      Ipv4Address source,
                      Ipv4Address destination,
                      uint8_t protocol,
                      Ptr<Ipv4Route> route)
{
  NS_LOG_FUNCTION (this << packet << source << destination << uint32_t (protocol) << route);

  bool mayFragment = true;

  // we need a copy of the packet with its tags in case we need to invoke recursion.
  Ptr<Packet> pktCopyWithTags = packet->Copy ();

  uint8_t ttl = m_defaultTtl;
  SocketIpTtlTag ipTtlTag;
  bool ipTtlTagFound = packet->RemovePacketTag (ipTtlTag);
  if (ipTtlTagFound)
    {
      ttl = ipTtlTag.GetTtl ();
    }

  uint8_t tos = 0;
  SocketIpTosTag ipTosTag;
  bool ipTosTagFound = packet->RemovePacketTag (ipTosTag);
  if (ipTosTagFound)
    {
      tos = ipTosTag.GetTos ();
    }

  // can construct the header here
  Ipv4Header ipHeader = BuildHeader (source, destination, protocol, packet->GetSize (), ttl, tos, mayFragment);

  // Handle a few cases:
  // 1) packet is passed in with a route entry
  // 1a) packet is passed in with a route entry but route->GetGateway is not set (e.g., on-demand)
  // 1b) packet is passed in with a route entry and valid gateway
  // 2) packet is passed without a route and packet is destined to limited broadcast address
  // 3) packet is passed without a route and packet is destined to a subnet-directed broadcast address
  // 4) packet is passed without a route, packet is not broadcast (e.g., a raw socket call, or ICMP)

  // 1) packet is passed in with route entry
  if (route)
    {
      // 1a) route->GetGateway is not set (e.g., on-demand)
      if (!route->GetGateway ().IsInitialized ())
        {
          // This could arise because the synchronous RouteOutput() call
          // returned to the transport protocol with a source address but
          // there was no next hop available yet (since a route may need
          // to be queried).
          NS_FATAL_ERROR ("Ipv4L3Protocol::Send case 1a: packet passed with a route but the Gateway address is uninitialized. This case not yet implemented.");
        }

      // 1b) with a valid gateway
      NS_LOG_LOGIC ("Ipv4L3Protocol::Send case 1b:  passed in with route and valid gateway");
      int32_t interface = GetInterfaceForDevice (route->GetOutputDevice ());
      m_sendOutgoingTrace (ipHeader, packet, interface);
      if (m_enableDpd && ipHeader.GetDestination ().IsMulticast ())
        {
          UpdateDuplicate (packet, ipHeader);
        }
      SendRealOut (route, packet->Copy (), ipHeader);
      return; 
    }

  // 2) packet is destined to limited broadcast address or link-local multicast address
  if (destination.IsBroadcast () || destination.IsLocalMulticast ())
    {
      NS_LOG_LOGIC ("Ipv4L3Protocol::Send case 2:  limited broadcast - no route");
      uint32_t ifaceIndex = 0;
      for (Ipv4InterfaceList::iterator ifaceIter = m_interfaces.begin ();
           ifaceIter != m_interfaces.end (); ifaceIter++, ifaceIndex++)
        {
          Ptr<Ipv4Interface> outInterface = *ifaceIter;
          // ANY source matches any interface
          bool sendIt = source.IsAny ();
          // check if some specific address on outInterface matches
          for (uint32_t index = 0; !sendIt && index < outInterface->GetNAddresses (); index++)
            {
              if (outInterface->GetAddress (index).GetLocal () == source)
                {
                  sendIt = true;
                }
            }

          if (sendIt)
            {
              // create a proxy route for this interface
              Ptr<Ipv4Route> route = Create<Ipv4Route> ();
              route->SetDestination (destination);
              route->SetGateway (Ipv4Address::GetAny ());
              route->SetSource (source);
              route->SetOutputDevice (outInterface->GetDevice ());
              DecreaseIdentification (source, destination, protocol);
              Send (pktCopyWithTags, source, destination, protocol, route);
            }
        }
      return;
    }

  // 3) check: packet is destined to a subnet-directed broadcast address
  for (Ipv4InterfaceList::iterator ifaceIter = m_interfaces.begin ();
       ifaceIter != m_interfaces.end (); ifaceIter++)
    {
      Ptr<Ipv4Interface> outInterface = *ifaceIter;
      uint32_t ifaceIndex = GetInterfaceForDevice (outInterface->GetDevice ());
      for (uint32_t j = 0; j < GetNAddresses (ifaceIndex); j++)
        {
          Ipv4InterfaceAddress ifAddr = GetAddress (ifaceIndex, j);
          NS_LOG_LOGIC ("Testing address " << ifAddr.GetLocal () << " with mask " << ifAddr.GetMask ());
          if (destination.IsSubnetDirectedBroadcast (ifAddr.GetMask ()) && 
              destination.CombineMask (ifAddr.GetMask ()) == ifAddr.GetLocal ().CombineMask (ifAddr.GetMask ())   )
            {
              NS_LOG_LOGIC ("Ipv4L3Protocol::Send case 3:  subnet directed bcast to " << ifAddr.GetLocal () << " - no route");
              // create a proxy route for this interface
              Ptr<Ipv4Route> route = Create<Ipv4Route> ();
              route->SetDestination (destination);
              route->SetGateway (Ipv4Address::GetAny ());
              route->SetSource (source);
              route->SetOutputDevice (outInterface->GetDevice ());
              DecreaseIdentification (source, destination, protocol);
              Send (pktCopyWithTags, source, destination, protocol, route);
              return;
            }
        }
    }

  // 4) packet is not broadcast, and route is NULL (e.g., a raw socket call)
  NS_LOG_LOGIC ("Ipv4L3Protocol::Send case 4:  not broadcast and passed in with no route " << destination);
  Socket::SocketErrno errno_; 
  Ptr<NetDevice> oif (0); // unused for now
  Ptr<Ipv4Route> newRoute;
  if (m_routingProtocol != 0)
    {
      newRoute = m_routingProtocol->RouteOutput (pktCopyWithTags, ipHeader, oif, errno_);
    }
  else
    {
      NS_LOG_ERROR ("Ipv4L3Protocol::Send: m_routingProtocol == 0");
    }
  if (newRoute)
    {
      DecreaseIdentification (source, destination, protocol);
      Send (pktCopyWithTags, source, destination, protocol, newRoute);
    }
  else
    {
      NS_LOG_WARN ("No route to host.  Drop.");
      m_dropTrace (ipHeader, packet, DROP_NO_ROUTE, this, 0);
      DecreaseIdentification (source, destination, protocol);
    }
}

void
Ipv4L3Protocol::DecreaseIdentification (Ipv4Address source,
                                        Ipv4Address destination,
                                        uint8_t protocol)
{
  uint64_t src = source.Get ();
  uint64_t dst = destination.Get ();
  uint64_t srcDst = dst | (src << 32);
  std::pair<uint64_t, uint8_t> key = std::make_pair (srcDst, protocol);
  m_identification[key]--;
}

Ipv4Header
Ipv4L3Protocol::BuildHeader (
  Ipv4Address source,
  Ipv4Address destination,
  uint8_t protocol,
  uint16_t payloadSize,
  uint8_t ttl,
  uint8_t tos,
  bool mayFragment)
{
  NS_LOG_FUNCTION (this << source << destination << (uint16_t)protocol << payloadSize << (uint16_t)ttl << (uint16_t)tos << mayFragment);
  Ipv4Header ipHeader;
  ipHeader.SetSource (source);
  ipHeader.SetDestination (destination);
  ipHeader.SetProtocol (protocol);
  ipHeader.SetPayloadSize (payloadSize);
  ipHeader.SetTtl (ttl);
  ipHeader.SetTos (tos);

  uint64_t src = source.Get ();
  uint64_t dst = destination.Get ();
  uint64_t srcDst = dst | (src << 32);
  std::pair<uint64_t, uint8_t> key = std::make_pair (srcDst, protocol);

  if (mayFragment == true)
    {
      ipHeader.SetMayFragment ();
      ipHeader.SetIdentification (m_identification[key]);
      m_identification[key]++;
    }
  else
    {
      ipHeader.SetDontFragment ();
      // RFC 6864 does not state anything about atomic datagrams
      // identification requirement:
      // >> Originating sources MAY set the IPv4 ID field of atomic datagrams
      //    to any value.
      ipHeader.SetIdentification (m_identification[key]);
      m_identification[key]++;
    }
  if (Node::ChecksumEnabled ())
    {
      ipHeader.EnableChecksum ();
    }
  return ipHeader;
}

void
Ipv4L3Protocol::SendRealOut (Ptr<Ipv4Route> route,
                             Ptr<Packet> packet,
                             Ipv4Header const &ipHeader)
{
  NS_LOG_FUNCTION (this << route << packet << &ipHeader);
  if (route == 0)
    {
      NS_LOG_WARN ("No route to host.  Drop.");
      m_dropTrace (ipHeader, packet, DROP_NO_ROUTE, this, 0);
      return;
    }
  Ptr<NetDevice> outDev = route->GetOutputDevice ();
  int32_t interface = GetInterfaceForDevice (outDev);
  NS_ASSERT (interface >= 0);
  Ptr<Ipv4Interface> outInterface = GetInterface (interface);
  NS_LOG_LOGIC ("Send via NetDevice ifIndex " << outDev->GetIfIndex () << " ipv4InterfaceIndex " << interface);

  Ipv4Address target;
  std::string targetLabel;
  if (route->GetGateway ().IsAny ())
    {
      target = ipHeader.GetDestination ();
      targetLabel = "destination";
    }
  else
    {
      target = route->GetGateway ();
      targetLabel = "gateway";
    }

  if (outInterface->IsUp ())
    {
      NS_LOG_LOGIC ("Send to " << targetLabel << " " << target);
      if ( packet->GetSize () + ipHeader.GetSerializedSize () > outInterface->GetDevice ()->GetMtu () )
        {
          std::list<Ipv4PayloadHeaderPair> listFragments;
          DoFragmentation (packet, ipHeader, outInterface->GetDevice ()->GetMtu (), listFragments);
          for ( std::list<Ipv4PayloadHeaderPair>::iterator it = listFragments.begin (); it != listFragments.end (); it++ )
            {
              NS_LOG_LOGIC ("Sending fragment " << *(it->first) );
              CallTxTrace (it->second, it->first, this, interface);
              outInterface->Send (it->first, it->second, target);
            }
        }
      else
        {
          CallTxTrace (ipHeader, packet, this, interface);
          outInterface->Send (packet, ipHeader, target);
        }
    }
}

// This function analogous to Linux ip_mr_forward()
void
Ipv4L3Protocol::IpMulticastForward (Ptr<Ipv4MulticastRoute> mrtentry, Ptr<const Packet> p, const Ipv4Header &header)
{
  NS_LOG_FUNCTION (this << mrtentry << p << header);
  NS_LOG_LOGIC ("Multicast forwarding logic for node: " << m_node->GetId ());

  std::map<uint32_t, uint32_t> ttlMap = mrtentry->GetOutputTtlMap ();
  std::map<uint32_t, uint32_t>::iterator mapIter;

  for (mapIter = ttlMap.begin (); mapIter != ttlMap.end (); mapIter++)
    {
      uint32_t interface = mapIter->first;
      //uint32_t outputTtl = mapIter->second;  // Unused for now

      Ptr<Packet> packet = p->Copy ();
      Ipv4Header ipHeader = header;
      ipHeader.SetTtl (header.GetTtl () - 1);
      if (ipHeader.GetTtl () == 0)
        {
          NS_LOG_WARN ("TTL exceeded.  Drop.");
          m_dropTrace (header, packet, DROP_TTL_EXPIRED, this, interface);
          return;
        }
      NS_LOG_LOGIC ("Forward multicast via interface " << interface);
      Ptr<Ipv4Route> rtentry = Create<Ipv4Route> ();
      rtentry->SetSource (ipHeader.GetSource ());
      rtentry->SetDestination (ipHeader.GetDestination ());
      rtentry->SetGateway (Ipv4Address::GetAny ());
      rtentry->SetOutputDevice (GetNetDevice (interface));
      
      m_multicastForwardTrace (ipHeader, packet, interface);
      SendRealOut (rtentry, packet, ipHeader);
      continue;
    }
}

// This function analogous to Linux ip_forward()
void
Ipv4L3Protocol::IpForward (Ptr<Ipv4Route> rtentry, Ptr<const Packet> p, const Ipv4Header &header)
{
  NS_LOG_FUNCTION (this << rtentry << p << header);
  NS_LOG_LOGIC ("Forwarding logic for node: " << m_node->GetId ());
  // Forwarding
  Ipv4Header ipHeader = header;
  Ptr<Packet> packet = p->Copy ();
  int32_t interface = GetInterfaceForDevice (rtentry->GetOutputDevice ());
  ipHeader.SetTtl (ipHeader.GetTtl () - 1);
  if (ipHeader.GetTtl () == 0)
    {
      // Do not reply to multicast/broadcast IP address
      if (ipHeader.GetDestination ().IsBroadcast () == false &&
          ipHeader.GetDestination ().IsMulticast () == false)
        {
          Ptr<Icmpv4L4Protocol> icmp = GetIcmp ();
          icmp->SendTimeExceededTtl (ipHeader, packet, false);
        }
      NS_LOG_WARN ("TTL exceeded.  Drop.");
      m_dropTrace (header, packet, DROP_TTL_EXPIRED, this, interface);
      return;
    }
  // in case the packet still has a priority tag attached, remove it
  SocketPriorityTag priorityTag;
  packet->RemovePacketTag (priorityTag);
  uint8_t priority = Socket::IpTos2Priority (ipHeader.GetTos ());
  // add a priority tag if the priority is not null
  if (priority)
    {
      priorityTag.SetPriority (priority);
      packet->AddPacketTag (priorityTag);
    }

  m_unicastForwardTrace (ipHeader, packet, interface);
  SendRealOut (rtentry, packet, ipHeader);
}

void
Ipv4L3Protocol::LocalDeliver (Ptr<const Packet> packet, Ipv4Header const&ip, uint32_t iif)
{
  NS_LOG_FUNCTION (this << packet << &ip << iif);
  Ptr<Packet> p = packet->Copy (); // need to pass a non-const packet up
  Ipv4Header ipHeader = ip;

  if ( !ipHeader.IsLastFragment () || ipHeader.GetFragmentOffset () != 0 )
    {
      NS_LOG_LOGIC ("Received a fragment, processing " << *p );
      bool isPacketComplete;
      isPacketComplete = ProcessFragment (p, ipHeader, iif);
      if ( isPacketComplete == false)
        {
          return;
        }
      NS_LOG_LOGIC ("Got last fragment, Packet is complete " << *p );
      ipHeader.SetFragmentOffset (0);
      ipHeader.SetPayloadSize (p->GetSize ());
    }

  m_localDeliverTrace (ipHeader, p, iif);

  Ptr<IpL4Protocol> protocol = GetProtocol (ipHeader.GetProtocol (), iif);
  if (protocol != 0)
    {
      // we need to make a copy in the unlikely event we hit the
      // RX_ENDPOINT_UNREACH codepath
      Ptr<Packet> copy = p->Copy ();
      enum IpL4Protocol::RxStatus status = 
        protocol->Receive (p, ipHeader, GetInterface (iif));
      switch (status) {
        case IpL4Protocol::RX_OK:
        // fall through
        case IpL4Protocol::RX_ENDPOINT_CLOSED:
        // fall through
        case IpL4Protocol::RX_CSUM_FAILED:
          break;
        case IpL4Protocol::RX_ENDPOINT_UNREACH:
          if (ipHeader.GetDestination ().IsBroadcast () == true ||
              ipHeader.GetDestination ().IsMulticast () == true)
            {
              break; // Do not reply to broadcast or multicast
            }
          // Another case to suppress ICMP is a subnet-directed broadcast
          bool subnetDirected = false;
          for (uint32_t i = 0; i < GetNAddresses (iif); i++)
            {
              Ipv4InterfaceAddress addr = GetAddress (iif, i);
              if (addr.GetLocal ().CombineMask (addr.GetMask ()) == ipHeader.GetDestination ().CombineMask (addr.GetMask ()) &&
                  ipHeader.GetDestination ().IsSubnetDirectedBroadcast (addr.GetMask ()))
                {
                  subnetDirected = true;
                }
            }
          if (subnetDirected == false)
            {
              GetIcmp ()->SendDestUnreachPort (ipHeader, copy);
            }
        }
    }
}

bool
Ipv4L3Protocol::AddAddress (uint32_t i, Ipv4InterfaceAddress address)
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
Ipv4L3Protocol::GetAddress (uint32_t interfaceIndex, uint32_t addressIndex) const
{
  NS_LOG_FUNCTION (this << interfaceIndex << addressIndex);
  Ptr<Ipv4Interface> interface = GetInterface (interfaceIndex);
  return interface->GetAddress (addressIndex);
}

uint32_t 
Ipv4L3Protocol::GetNAddresses (uint32_t interface) const
{
  NS_LOG_FUNCTION (this << interface);
  Ptr<Ipv4Interface> iface = GetInterface (interface);
  return iface->GetNAddresses ();
}

bool
Ipv4L3Protocol::RemoveAddress (uint32_t i, uint32_t addressIndex)
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
Ipv4L3Protocol::RemoveAddress (uint32_t i, Ipv4Address address)
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
Ipv4L3Protocol::SourceAddressSelection (uint32_t interfaceIdx, Ipv4Address dest)
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
Ipv4L3Protocol::SelectSourceAddress (Ptr<const NetDevice> device,
                                     Ipv4Address dst, Ipv4InterfaceAddress::InterfaceAddressScope_e scope)
{
  NS_LOG_FUNCTION (this << device << dst << scope);
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
          if (iaddr.IsSecondary ()) continue;
          if (iaddr.GetScope () > scope) continue; 
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
          if (iaddr.IsSecondary ()) continue;
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
Ipv4L3Protocol::SetMetric (uint32_t i, uint16_t metric)
{
  NS_LOG_FUNCTION (this << i << metric);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  interface->SetMetric (metric);
}

uint16_t
Ipv4L3Protocol::GetMetric (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->GetMetric ();
}

uint16_t 
Ipv4L3Protocol::GetMtu (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->GetDevice ()->GetMtu ();
}

bool 
Ipv4L3Protocol::IsUp (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  return interface->IsUp ();
}

void 
Ipv4L3Protocol::SetUp (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);

  // RFC 791, pg.25:
  //  Every internet module must be able to forward a datagram of 68
  //  octets without further fragmentation.  This is because an internet
  //  header may be up to 60 octets, and the minimum fragment is 8 octets.
  if (interface->GetDevice ()->GetMtu () >= 68)
    {
      interface->SetUp ();

      if (m_routingProtocol != 0)
        {
          m_routingProtocol->NotifyInterfaceUp (i);
        }
    }
  else
    {
      NS_LOG_LOGIC ("Interface " << int(i) << " is set to be down for IPv4. Reason: not respecting minimum IPv4 MTU (68 octects)");
    }
}

void 
Ipv4L3Protocol::SetDown (uint32_t ifaceIndex)
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
Ipv4L3Protocol::IsForwarding (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  NS_LOG_LOGIC ("Forwarding state: " << interface->IsForwarding ());
  return interface->IsForwarding ();
}

void 
Ipv4L3Protocol::SetForwarding (uint32_t i, bool val)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv4Interface> interface = GetInterface (i);
  interface->SetForwarding (val);
}

Ptr<NetDevice>
Ipv4L3Protocol::GetNetDevice (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  return GetInterface (i)->GetDevice ();
}

void 
Ipv4L3Protocol::SetIpForward (bool forward) 
{
  NS_LOG_FUNCTION (this << forward);
  m_ipForward = forward;
  for (Ipv4InterfaceList::const_iterator i = m_interfaces.begin (); i != m_interfaces.end (); i++)
    {
      (*i)->SetForwarding (forward);
    }
}

bool 
Ipv4L3Protocol::GetIpForward (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ipForward;
}

void 
Ipv4L3Protocol::SetWeakEsModel (bool model)
{
  NS_LOG_FUNCTION (this << model);
  m_weakEsModel = model;
}

bool 
Ipv4L3Protocol::GetWeakEsModel (void) const
{
  NS_LOG_FUNCTION (this);
  return m_weakEsModel;
}

void
Ipv4L3Protocol::RouteInputError (Ptr<const Packet> p, const Ipv4Header & ipHeader, Socket::SocketErrno sockErrno)
{
  NS_LOG_FUNCTION (this << p << ipHeader << sockErrno);
  NS_LOG_LOGIC ("Route input failure-- dropping packet to " << ipHeader << " with errno " << sockErrno); 
  m_dropTrace (ipHeader, p, DROP_ROUTE_ERROR, this, 0);

  // \todo Send an ICMP no route.
}

void
Ipv4L3Protocol::DoFragmentation (Ptr<Packet> packet, const Ipv4Header & ipv4Header, uint32_t outIfaceMtu, std::list<Ipv4PayloadHeaderPair>& listFragments)
{
  // BEWARE: here we do assume that the header options are not present.
  // a much more complex handling is necessary in case there are options.
  // If (when) IPv4 option headers will be implemented, the following code shall be changed.
  // Of course also the reassemby code shall be changed as well.

  NS_LOG_FUNCTION (this << *packet << outIfaceMtu << &listFragments);

  Ptr<Packet> p = packet->Copy ();

  NS_ASSERT_MSG( (ipv4Header.GetSerializedSize() == 5*4),
                 "IPv4 fragmentation implementation only works without option headers." );

  uint16_t offset = 0;
  bool moreFragment = true;
  uint16_t originalOffset = ipv4Header.GetFragmentOffset();
  bool isLastFragment = ipv4Header.IsLastFragment();
  uint32_t currentFragmentablePartSize = 0;

  // IPv4 fragments are all 8 bytes aligned but the last.
  // The IP payload size is:
  // floor( ( outIfaceMtu - ipv4Header.GetSerializedSize() ) /8 ) *8
  uint32_t fragmentSize = (outIfaceMtu - ipv4Header.GetSerializedSize () ) & ~uint32_t (0x7);

  NS_LOG_LOGIC ("Fragmenting - Target Size: " << fragmentSize );

  do
    {
      Ipv4Header fragmentHeader = ipv4Header;

      if (p->GetSize () > offset + fragmentSize )
        {
          moreFragment = true;
          currentFragmentablePartSize = fragmentSize;
          fragmentHeader.SetMoreFragments ();
        }
      else
        {
          moreFragment = false;
          currentFragmentablePartSize = p->GetSize () - offset;
          if (!isLastFragment)
            {
              fragmentHeader.SetMoreFragments ();
            }
          else
            {
              fragmentHeader.SetLastFragment ();
            }
        }

      NS_LOG_LOGIC ("Fragment creation - " << offset << ", " << currentFragmentablePartSize  );
      Ptr<Packet> fragment = p->CreateFragment (offset, currentFragmentablePartSize);
      NS_LOG_LOGIC ("Fragment created - " << offset << ", " << fragment->GetSize ()  );

      fragmentHeader.SetFragmentOffset (offset+originalOffset);
      fragmentHeader.SetPayloadSize (currentFragmentablePartSize);

      if (Node::ChecksumEnabled ())
        {
          fragmentHeader.EnableChecksum ();
        }

      NS_LOG_LOGIC ("Fragment check - " << fragmentHeader.GetFragmentOffset ()  );

      NS_LOG_LOGIC ("New fragment Header " << fragmentHeader);

      std::ostringstream oss;
      oss << fragmentHeader;
      fragment->Print (oss);

      NS_LOG_LOGIC ("New fragment " << *fragment);

      listFragments.emplace_back (fragment, fragmentHeader);

      offset += currentFragmentablePartSize;

    }
  while (moreFragment);

  return;
}

bool
Ipv4L3Protocol::ProcessFragment (Ptr<Packet>& packet, Ipv4Header& ipHeader, uint32_t iif)
{
  NS_LOG_FUNCTION (this << packet << ipHeader << iif);

  uint64_t addressCombination = uint64_t (ipHeader.GetSource ().Get ()) << 32 | uint64_t (ipHeader.GetDestination ().Get ());
  uint32_t idProto = uint32_t (ipHeader.GetIdentification ()) << 16 | uint32_t (ipHeader.GetProtocol ());
  FragmentKey_t key;
  bool ret = false;
  Ptr<Packet> p = packet->Copy ();

  key.first = addressCombination;
  key.second = idProto;

  Ptr<Fragments> fragments;

  MapFragments_t::iterator it = m_fragments.find (key);
  if (it == m_fragments.end ())
    {
      fragments = Create<Fragments> ();
      m_fragments.insert (std::make_pair (key, fragments));

      FragmentsTimeoutsListI_t iter = SetTimeout (key, ipHeader, iif);
      fragments->SetTimeoutIter (iter);
    }
  else
    {
      fragments = it->second;
    }

  NS_LOG_LOGIC ("Adding fragment - Size: " << packet->GetSize ( ) << " - Offset: " << (ipHeader.GetFragmentOffset ()) );

  fragments->AddFragment (p, ipHeader.GetFragmentOffset (), !ipHeader.IsLastFragment () );

  if ( fragments->IsEntire () )
    {
      packet = fragments->GetPacket ();
      m_timeoutEventList.erase (fragments->GetTimeoutIter ());
      fragments = 0;
      m_fragments.erase (key);
      ret = true;
    }

  return ret;
}

Ipv4L3Protocol::Fragments::Fragments ()
  : m_moreFragment (0)
{
  NS_LOG_FUNCTION (this);
}

Ipv4L3Protocol::Fragments::~Fragments ()
{
  NS_LOG_FUNCTION (this);
}

void
Ipv4L3Protocol::Fragments::AddFragment (Ptr<Packet> fragment, uint16_t fragmentOffset, bool moreFragment)
{
  NS_LOG_FUNCTION (this << fragment << fragmentOffset << moreFragment);

  std::list<std::pair<Ptr<Packet>, uint16_t> >::iterator it;

  for (it = m_fragments.begin (); it != m_fragments.end (); it++)
    {
      if (it->second > fragmentOffset)
        {
          break;
        }
    }

  if (it == m_fragments.end ())
    {
      m_moreFragment = moreFragment;
    }

  m_fragments.insert (it, std::pair<Ptr<Packet>, uint16_t> (fragment, fragmentOffset));
}

bool
Ipv4L3Protocol::Fragments::IsEntire () const
{
  NS_LOG_FUNCTION (this);

  bool ret = !m_moreFragment && m_fragments.size () > 0;

  if (ret)
    {
      uint16_t lastEndOffset = 0;

      for (std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator it = m_fragments.begin (); it != m_fragments.end (); it++)
        {
          // overlapping fragments do exist
          NS_LOG_LOGIC ("Checking overlaps " << lastEndOffset << " - " << it->second );

          if (lastEndOffset < it->second)
            {
              ret = false;
              break;
            }
          // fragments might overlap in strange ways
          uint16_t fragmentEnd = it->first->GetSize () + it->second;
          lastEndOffset = std::max ( lastEndOffset, fragmentEnd );
        }
    }

  return ret;
}

Ptr<Packet>
Ipv4L3Protocol::Fragments::GetPacket () const
{
  NS_LOG_FUNCTION (this);

  std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator it = m_fragments.begin ();

  Ptr<Packet> p = it->first->Copy ();
  uint16_t lastEndOffset = p->GetSize ();
  it++;

  for ( ; it != m_fragments.end (); it++)
    {
      if ( lastEndOffset > it->second )
        {
          // The fragments are overlapping.
          // We do not overwrite the "old" with the "new" because we do not know when each arrived.
          // This is different from what Linux does.
          // It is not possible to emulate a fragmentation attack.
          uint32_t newStart = lastEndOffset - it->second;
          if ( it->first->GetSize () > newStart )
            {
              uint32_t newSize = it->first->GetSize () - newStart;
              Ptr<Packet> tempFragment = it->first->CreateFragment (newStart, newSize);
              p->AddAtEnd (tempFragment);
            }
        }
      else
        {
          NS_LOG_LOGIC ("Adding: " << *(it->first) );
          p->AddAtEnd (it->first);
        }
      lastEndOffset = p->GetSize ();
    }

  return p;
}

Ptr<Packet>
Ipv4L3Protocol::Fragments::GetPartialPacket () const
{
  NS_LOG_FUNCTION (this);
  
  std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator it = m_fragments.begin ();

  Ptr<Packet> p = Create<Packet> ();
  uint16_t lastEndOffset = 0;

  if ( m_fragments.begin ()->second > 0 )
    {
      return p;
    }

  for ( it = m_fragments.begin (); it != m_fragments.end (); it++)
    {
      if ( lastEndOffset > it->second )
        {
          uint32_t newStart = lastEndOffset - it->second;
          uint32_t newSize = it->first->GetSize () - newStart;
          Ptr<Packet> tempFragment = it->first->CreateFragment (newStart, newSize);
          p->AddAtEnd (tempFragment);
        }
      else if ( lastEndOffset == it->second )
        {
          NS_LOG_LOGIC ("Adding: " << *(it->first) );
          p->AddAtEnd (it->first);
        }
      lastEndOffset = p->GetSize ();
    }

  return p;
}

void
Ipv4L3Protocol::Fragments::SetTimeoutIter (FragmentsTimeoutsListI_t iter)
{
  m_timeoutIter = iter;
  return;
}

Ipv4L3Protocol::FragmentsTimeoutsListI_t
Ipv4L3Protocol::Fragments::GetTimeoutIter ()
{
  return m_timeoutIter;
}


void
Ipv4L3Protocol::HandleFragmentsTimeout (FragmentKey_t key, Ipv4Header & ipHeader, uint32_t iif)
{
  NS_LOG_FUNCTION (this << &key << &ipHeader << iif);

  MapFragments_t::iterator it = m_fragments.find (key);
  Ptr<Packet> packet = it->second->GetPartialPacket ();

  // if we have at least 8 bytes, we can send an ICMP.
  if ( packet->GetSize () > 8 )
    {
      Ptr<Icmpv4L4Protocol> icmp = GetIcmp ();
      icmp->SendTimeExceededTtl (ipHeader, packet, true);
    }
  m_dropTrace (ipHeader, packet, DROP_FRAGMENT_TIMEOUT, this, iif);

  // clear the buffers
  it->second = 0;

  m_fragments.erase (key);
}

bool
Ipv4L3Protocol::UpdateDuplicate (Ptr<const Packet> p, const Ipv4Header &header)
{
  NS_LOG_FUNCTION (this << p << header);

  // \todo RFC 6621 mandates SHA-1 hash.  For now ns3 hash should be fine.
  uint8_t proto = header.GetProtocol ();
  Ipv4Address src = header.GetSource ();
  Ipv4Address dst = header.GetDestination ();
  uint64_t id = header.GetIdentification ();

  // concat hash value onto id
  uint64_t hash = id << 32;
  if (header.GetFragmentOffset () || !header.IsLastFragment ())
    {
      // use I-DPD (RFC 6621, Sec 6.2.1)
      hash |= header.GetFragmentOffset ();
    }
  else
    {
      // use H-DPD (RFC 6621, Sec 6.2.2)

      // serialize packet
      Ptr<Packet> pkt = p->Copy ();
      pkt->AddHeader (header);

      std::ostringstream oss (std::ios_base::binary);
      pkt->CopyData (&oss, pkt->GetSize ());
      std::string bytes = oss.str ();

      NS_ASSERT_MSG (bytes.size () >= 20, "Degenerate header serialization");

      // zero out mutable fields
      bytes[1] = 0;               // DSCP / ECN
      bytes[6] = bytes[7] = 0;    // Flags / Fragment offset
      bytes[8] = 0;               // TTL
      bytes[10] = bytes[11] = 0;  // Header checksum
      if (header.GetSerializedSize () > 20)     // assume options should be 0'd
        {
          std::fill_n (bytes.begin () + 20, header.GetSerializedSize () - 20, 0);
        }
        
      // concat hash onto ID
      hash |= (uint64_t)Hash32 (bytes);
    }

  // set cleanup job for new duplicate entries
  if (!m_cleanDpd.IsRunning () && m_purge.IsStrictlyPositive ())
    {
      m_cleanDpd = Simulator::Schedule (m_expire, &Ipv4L3Protocol::RemoveDuplicates, this);
    }

  // assume this is a new entry
  DupTuple_t key {hash, proto, src, dst};
  NS_LOG_DEBUG ("Packet " << p->GetUid () << " key = (" <<
                std::hex << std::get<0> (key) << ", " <<
                std::dec << +std::get<1> (key) << ", " <<
                std::get<2> (key) << ", " <<
                std::get<3> (key) << ")");

  // place a new entry, on collision the existing entry iterator is returned
  DupMap_t::iterator iter;
  bool inserted, isDup;
  std::tie (iter, inserted) = m_dups.emplace (key, Seconds (0));
  isDup = !inserted && iter->second > Simulator::Now ();

  // set the expiration event
  iter->second = Simulator::Now () + m_expire;
  return isDup;
}

void
Ipv4L3Protocol::RemoveDuplicates (void)
{
  NS_LOG_FUNCTION (this);

  DupMap_t::size_type n = 0;
  Time expire = Simulator::Now ();
  auto iter = m_dups.cbegin ();
  while (iter != m_dups.cend ())
    {
      if (iter->second < expire)
        {
          NS_LOG_LOGIC ("Remove key = (" <<
                        std::hex << std::get<0> (iter->first) << ", " <<
                        std::dec << +std::get<1> (iter->first) << ", " <<
                        std::get<2> (iter->first) << ", " <<
                        std::get<3> (iter->first) << ")");
          iter = m_dups.erase (iter);
          ++n;
        }
      else
        {
          ++iter;
        }
    }
  
  NS_LOG_DEBUG ("Purged " << n << " expired duplicate entries out of " << (n + m_dups.size ()));
  
  // keep cleaning up if necessary
  if (!m_dups.empty () && m_purge.IsStrictlyPositive ())
    {
      m_cleanDpd = Simulator::Schedule (m_purge, &Ipv4L3Protocol::RemoveDuplicates, this);
    }
}

Ipv4L3Protocol::FragmentsTimeoutsListI_t
Ipv4L3Protocol::SetTimeout (FragmentKey_t key, Ipv4Header ipHeader, uint32_t iif)
{
  Time now = Simulator::Now () + m_fragmentExpirationTimeout;

  if (m_timeoutEventList.empty ())
    {
      m_timeoutEvent = Simulator::Schedule (m_fragmentExpirationTimeout, &Ipv4L3Protocol::HandleTimeout, this);
    }
  m_timeoutEventList.emplace_back (now, key, ipHeader, iif);

  Ipv4L3Protocol::FragmentsTimeoutsListI_t iter = --m_timeoutEventList.end();

  return (iter);
}

void
Ipv4L3Protocol::HandleTimeout (void)
{
  Time now = Simulator::Now ();

  while (!m_timeoutEventList.empty () && std::get<0> (*m_timeoutEventList.begin ()) == now)
    {
      HandleFragmentsTimeout (std::get<1> (*m_timeoutEventList.begin ()),
                              std::get<2> (*m_timeoutEventList.begin ()),
                              std::get<3> (*m_timeoutEventList.begin ()));
      m_timeoutEventList.pop_front ();
    }

  if (m_timeoutEventList.empty ())
    {
      return;
    }

  Time difference = std::get<0> (*m_timeoutEventList.begin ()) - now;
  m_timeoutEvent = Simulator::Schedule (difference, &Ipv4L3Protocol::HandleTimeout, this);

  return;
}

} // namespace ns3
