/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' di Firenze, Italy
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

#include <iomanip>
#include "ripng.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ipv6-route.h"
#include "ns3/node.h"
#include "ns3/names.h"
#include "ns3/ripng-header.h"
#include "ns3/udp-header.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "ns3/ipv6-packet-info-tag.h"
#include "ns3/simulator.h"

#define RIPNG_ALL_NODE "ff02::9"
#define RIPNG_PORT 521

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RipNg");

NS_OBJECT_ENSURE_REGISTERED (RipNg);

RipNg::RipNg ()
  : m_ipv6 (0), m_splitHorizonStrategy (RipNg::POISON_REVERSE), m_initialized (false)
{
  m_rng = CreateObject<UniformRandomVariable> ();
}

RipNg::~RipNg ()
{
}

TypeId
RipNg::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RipNg")
    .SetParent<Ipv6RoutingProtocol> ()
    .SetGroupName ("Internet")
    .AddConstructor<RipNg> ()
    .AddAttribute ("UnsolicitedRoutingUpdate", "The time between two Unsolicited Routing Updates.",
                   TimeValue (Seconds(30)),
                   MakeTimeAccessor (&RipNg::m_unsolicitedUpdate),
                   MakeTimeChecker ())
    .AddAttribute ("StartupDelay", "Maximum random delay for protocol startup (send route requests).",
                   TimeValue (Seconds(1)),
                   MakeTimeAccessor (&RipNg::m_startupDelay),
                   MakeTimeChecker ())
    .AddAttribute ("TimeoutDelay", "The delay to invalidate a route.",
                   TimeValue (Seconds(180)),
                   MakeTimeAccessor (&RipNg::m_timeoutDelay),
                   MakeTimeChecker ())
    .AddAttribute ("GarbageCollectionDelay", "The delay to delete an expired route.",
                   TimeValue (Seconds(120)),
                   MakeTimeAccessor (&RipNg::m_garbageCollectionDelay),
                   MakeTimeChecker ())
    .AddAttribute ("MinTriggeredCooldown", "Min cooldown delay after a Triggered Update.",
                   TimeValue (Seconds(1)),
                   MakeTimeAccessor (&RipNg::m_minTriggeredUpdateDelay),
                   MakeTimeChecker ())
    .AddAttribute ("MaxTriggeredCooldown", "Max cooldown delay after a Triggered Update.",
                   TimeValue (Seconds(5)),
                   MakeTimeAccessor (&RipNg::m_maxTriggeredUpdateDelay),
                   MakeTimeChecker ())
    .AddAttribute ("SplitHorizon", "Split Horizon strategy.",
                   EnumValue (RipNg::POISON_REVERSE),
                   MakeEnumAccessor (&RipNg::m_splitHorizonStrategy),
                   MakeEnumChecker (RipNg::NO_SPLIT_HORIZON, "NoSplitHorizon",
                                    RipNg::SPLIT_HORIZON, "SplitHorizon",
                                    RipNg::POISON_REVERSE, "PoisonReverse"))
    .AddAttribute ("LinkDownValue", "Value for link down in count to infinity.",
                   UintegerValue (16),
                   MakeUintegerAccessor (&RipNg::m_linkDown),
                   MakeUintegerChecker<uint8_t> ())
    ;
  return tid;
}

int64_t RipNg::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);

  m_rng->SetStream (stream);
  return 1;
}

void RipNg::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  bool addedGlobal = false;

  m_initialized = true;

  Time delay = m_unsolicitedUpdate + Seconds (m_rng->GetValue (0, 0.5*m_unsolicitedUpdate.GetSeconds ()) );
  m_nextUnsolicitedUpdate = Simulator::Schedule (delay, &RipNg::SendUnsolicitedRouteUpdate, this);


  for (uint32_t i = 0 ; i < m_ipv6->GetNInterfaces (); i++)
    {
      bool activeInterface = false;
      if (m_interfaceExclusions.find (i) == m_interfaceExclusions.end ())
        {
          activeInterface = true;
          m_ipv6->SetForwarding (i, true);
        }

      for (uint32_t j = 0; j < m_ipv6->GetNAddresses (i); j++)
        {
          Ipv6InterfaceAddress address = m_ipv6->GetAddress (i, j);
          if (address.GetScope() == Ipv6InterfaceAddress::LINKLOCAL && activeInterface == true)
            {
              NS_LOG_LOGIC ("RIPng: adding socket to " << address.GetAddress ());
              TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
              Ptr<Node> theNode = GetObject<Node> ();
              Ptr<Socket> socket = Socket::CreateSocket (theNode, tid);
              Inet6SocketAddress local = Inet6SocketAddress (address.GetAddress (), RIPNG_PORT);
              socket->BindToNetDevice (m_ipv6->GetNetDevice (i));
              int ret = socket->Bind (local);
              NS_ASSERT_MSG (ret == 0, "Bind unsuccessful");
              socket->SetRecvCallback (MakeCallback (&RipNg::Receive, this));
              socket->SetIpv6RecvHopLimit (true);
              socket->SetRecvPktInfo (true);
              m_unicastSocketList[socket] = i;
            }
          else if (m_ipv6->GetAddress (i, j).GetScope() == Ipv6InterfaceAddress::GLOBAL)
            {
              addedGlobal = true;
            }
        }
    }

  if (!m_multicastRecvSocket)
    {
      NS_LOG_LOGIC ("RIPng: adding receiving socket");
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      Ptr<Node> theNode = GetObject<Node> ();
      m_multicastRecvSocket = Socket::CreateSocket (theNode, tid);
      Inet6SocketAddress local = Inet6SocketAddress (RIPNG_ALL_NODE, RIPNG_PORT);
      m_multicastRecvSocket->Bind (local);
      m_multicastRecvSocket->SetRecvCallback (MakeCallback (&RipNg::Receive, this));
      m_multicastRecvSocket->SetIpv6RecvHopLimit (true);
      m_multicastRecvSocket->SetRecvPktInfo (true);
    }


  if (addedGlobal)
    {
      Time delay = Seconds (m_rng->GetValue (m_minTriggeredUpdateDelay.GetSeconds (), m_maxTriggeredUpdateDelay.GetSeconds ()));
      m_nextTriggeredUpdate = Simulator::Schedule (delay, &RipNg::DoSendRouteUpdate, this, false);
    }

  delay = Seconds (m_rng->GetValue (0.01, m_startupDelay.GetSeconds ()));
  m_nextTriggeredUpdate = Simulator::Schedule (delay, &RipNg::SendRouteRequest, this);

  Ipv6RoutingProtocol::DoInitialize ();
}

Ptr<Ipv6Route> RipNg::RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)
{
  NS_LOG_FUNCTION (this << header << oif);

  Ipv6Address destination = header.GetDestination ();
  Ptr<Ipv6Route> rtentry = 0;

  if (destination.IsMulticast ())
    {
      // Note:  Multicast routes for outbound packets are stored in the
      // normal unicast table.  An implication of this is that it is not
      // possible to source multicast datagrams on multiple interfaces.
      // This is a well-known property of sockets implementation on
      // many Unix variants.
      // So, we just log it and fall through to LookupStatic ()
      NS_LOG_LOGIC ("RouteOutput (): Multicast destination");
    }

  rtentry = Lookup (destination, true, oif);
  if (rtentry)
    {
      sockerr = Socket::ERROR_NOTERROR;
    }
  else
    {
      sockerr = Socket::ERROR_NOROUTETOHOST;
    }
  return rtentry;
}

bool RipNg::RouteInput (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev,
                        UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                        LocalDeliverCallback lcb, ErrorCallback ecb)
{
  NS_LOG_FUNCTION (this << p << header << header.GetSource () << header.GetDestination () << idev);

  NS_ASSERT (m_ipv6 != 0);
  // Check if input device supports IP
  NS_ASSERT (m_ipv6->GetInterfaceForDevice (idev) >= 0);
  uint32_t iif = m_ipv6->GetInterfaceForDevice (idev);
  Ipv6Address dst = header.GetDestination ();

  if (dst.IsMulticast ())
    {
      NS_LOG_LOGIC ("Multicast route not supported by RIPng");
      return false; // Let other routing protocols try to handle this
    }

  if (header.GetDestination ().IsLinkLocal () ||
      header.GetSource ().IsLinkLocal ())
    {
      NS_LOG_LOGIC ("Dropping packet not for me and with src or dst LinkLocal");
      if (!ecb.IsNull ())
        {
          ecb (p, header, Socket::ERROR_NOROUTETOHOST);
        }
      return false;
    }

  // Check if input device supports IP forwarding
  if (m_ipv6->IsForwarding (iif) == false)
    {
      NS_LOG_LOGIC ("Forwarding disabled for this interface");
      if (!ecb.IsNull ())
        {
          ecb (p, header, Socket::ERROR_NOROUTETOHOST);
        }
      return true;
    }
  // Next, try to find a route
  NS_LOG_LOGIC ("Unicast destination");
  Ptr<Ipv6Route> rtentry = Lookup (header.GetDestination (), false);

  if (rtentry != 0)
    {
      NS_LOG_LOGIC ("Found unicast destination - calling unicast callback");
      ucb (idev, rtentry, p, header);  // unicast forwarding callback
      return true;
    }
  else
    {
      NS_LOG_LOGIC ("Did not find unicast destination - returning false");
      return false; // Let other routing protocols try to handle this
    }
}

void RipNg::NotifyInterfaceUp (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);

  for (uint32_t j = 0; j < m_ipv6->GetNAddresses (i); j++)
    {
      Ipv6InterfaceAddress address = m_ipv6->GetAddress (i, j);
      Ipv6Prefix networkMask = address.GetPrefix ();
      Ipv6Address networkAddress = address.GetAddress ().CombinePrefix (networkMask);

      if (address.GetScope () == Ipv6InterfaceAddress::GLOBAL)
        {
          AddNetworkRouteTo (networkAddress, networkMask, i);
        }
    }

  if (!m_initialized)
    {
      return;
    }


  bool sendSocketFound = false;
  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      if (iter->second == i)
        {
          sendSocketFound = true;
          break;
        }
    }

  bool activeInterface = false;
  if (m_interfaceExclusions.find (i) == m_interfaceExclusions.end ())
    {
      activeInterface = true;
      m_ipv6->SetForwarding (i, true);
    }

  for (uint32_t j = 0; j < m_ipv6->GetNAddresses (i); j++)
    {
      Ipv6InterfaceAddress address = m_ipv6->GetAddress (i, j);

      if (address.GetScope() == Ipv6InterfaceAddress::LINKLOCAL && sendSocketFound == false && activeInterface == true)
        {
          NS_LOG_LOGIC ("RIPng: adding sending socket to " << address.GetAddress ());
          TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
          Ptr<Node> theNode = GetObject<Node> ();
          Ptr<Socket> socket = Socket::CreateSocket (theNode, tid);
          Inet6SocketAddress local = Inet6SocketAddress (address.GetAddress (), RIPNG_PORT);
          socket->BindToNetDevice (m_ipv6->GetNetDevice (i));
          socket->Bind (local);
          socket->SetRecvCallback (MakeCallback (&RipNg::Receive, this));
          socket->SetIpv6RecvHopLimit (true);
          socket->SetRecvPktInfo (true);
          m_unicastSocketList[socket] = i;
        }
      else if (address.GetScope() == Ipv6InterfaceAddress::GLOBAL)
        {
          SendTriggeredRouteUpdate ();
        }
    }

  if (!m_multicastRecvSocket)
    {
      NS_LOG_LOGIC ("RIPng: adding receiving socket");
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      Ptr<Node> theNode = GetObject<Node> ();
      m_multicastRecvSocket = Socket::CreateSocket (theNode, tid);
      Inet6SocketAddress local = Inet6SocketAddress (RIPNG_ALL_NODE, RIPNG_PORT);
      m_multicastRecvSocket->Bind (local);
      m_multicastRecvSocket->SetRecvCallback (MakeCallback (&RipNg::Receive, this));
      m_multicastRecvSocket->SetIpv6RecvHopLimit (true);
      m_multicastRecvSocket->SetRecvPktInfo (true);
    }
}

void RipNg::NotifyInterfaceDown (uint32_t interface)
{
  NS_LOG_FUNCTION (this << interface);

  /* remove all routes that are going through this interface */
  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first->GetInterface () == interface)
        {
          InvalidateRoute (it->first);
        }
    }

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      NS_LOG_INFO ("Checking socket for interface " << interface);
      if (iter->second == interface)
        {
          NS_LOG_INFO ("Removed socket for interface " << interface);
          iter->first->Close ();
          m_unicastSocketList.erase (iter);
          break;
        }
    }

  if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
    {
      SendTriggeredRouteUpdate ();
    }
}

void RipNg::NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address)
{
  NS_LOG_FUNCTION (this << interface << address);

  if (!m_ipv6->IsUp (interface))
    {
      return;
    }

  if (m_interfaceExclusions.find (interface) != m_interfaceExclusions.end ())
    {
      return;
    }

  Ipv6Address networkAddress = address.GetAddress ().CombinePrefix (address.GetPrefix ());
  Ipv6Prefix networkMask = address.GetPrefix ();

  if (address.GetScope () == Ipv6InterfaceAddress::GLOBAL)
    {
      AddNetworkRouteTo (networkAddress, networkMask, interface);
    }

  SendTriggeredRouteUpdate ();
}

void RipNg::NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address)
{
  NS_LOG_FUNCTION (this << interface << address);

  if (!m_ipv6->IsUp (interface))
    {
      return;
    }

  if (address.GetScope() != Ipv6InterfaceAddress::GLOBAL)
    {
      return;
    }

  Ipv6Address networkAddress = address.GetAddress ().CombinePrefix (address.GetPrefix ());
  Ipv6Prefix networkMask = address.GetPrefix ();

  // Remove all routes that are going through this interface
  // which reference this network
  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first->GetInterface () == interface
          && it->first->IsNetwork ()
          && it->first->GetDestNetwork () == networkAddress
          && it->first->GetDestNetworkPrefix () == networkMask)
        {
          InvalidateRoute (it->first);
        }
    }

  if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
    {
      SendTriggeredRouteUpdate ();
    }

}

void RipNg::NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  NS_LOG_INFO (this << dst << mask << nextHop << interface << prefixToUse);
  // \todo this can be used to add delegate routes
}

void RipNg::NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  NS_LOG_FUNCTION (this << dst << mask << nextHop << interface);
  // \todo this can be used to delete delegate routes
}

void RipNg::SetIpv6 (Ptr<Ipv6> ipv6)
{
  NS_LOG_FUNCTION (this << ipv6);

  NS_ASSERT (m_ipv6 == 0 && ipv6 != 0);
  uint32_t i = 0;
  m_ipv6 = ipv6;

  for (i = 0; i < m_ipv6->GetNInterfaces (); i++)
    {
      if (m_ipv6->IsUp (i))
        {
          NotifyInterfaceUp (i);
        }
      else
        {
          NotifyInterfaceDown (i);
        }
    }
}

void RipNg::PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION (this << stream);

  std::ostream* os = stream->GetStream ();
  *os << std::resetiosflags (std::ios::adjustfield) << std::setiosflags (std::ios::left);

  *os << "Node: " << m_ipv6->GetObject<Node> ()->GetId ()
      << ", Time: " << Now().As (unit)
      << ", Local time: " << m_ipv6->GetObject<Node> ()->GetLocalTime ().As (unit)
      << ", IPv6 RIPng table" << std::endl;

  if (!m_routes.empty ())
    {
      *os << "Destination                    Next Hop                   Flag Met Ref Use If" << std::endl;
      for (RoutesCI it = m_routes.begin (); it != m_routes.end (); it++)
        {
          RipNgRoutingTableEntry* route = it->first;
          RipNgRoutingTableEntry::Status_e status = route->GetRouteStatus();

          if (status == RipNgRoutingTableEntry::RIPNG_VALID)
            {
              std::ostringstream dest, gw, mask, flags;

              dest << route->GetDest () << "/" << int(route->GetDestNetworkPrefix ().GetPrefixLength ());
              *os << std::setw (31) << dest.str ();
              gw << route->GetGateway ();
              *os << std::setw (27) << gw.str ();
              flags << "U";
              if (route->IsHost ())
                {
                  flags << "H";
                }
              else if (route->IsGateway ())
                {
                  flags << "G";
                }
              *os << std::setw (5) << flags.str ();
              *os << std::setw (4) << int(route->GetRouteMetric ());
              // Ref ct not implemented
              *os << "-" << "   ";
              // Use not implemented
              *os << "-" << "   ";
              if (Names::FindName (m_ipv6->GetNetDevice (route->GetInterface ())) != "")
                {
                  *os << Names::FindName (m_ipv6->GetNetDevice (route->GetInterface ()));
                }
              else
                {
                  *os << route->GetInterface ();
                }
              *os << std::endl;
            }
        }
    }
  *os << std::endl;
}

void RipNg::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  for (RoutesI j = m_routes.begin ();  j != m_routes.end (); j = m_routes.erase (j))
    {
      delete j->first;
    }
  m_routes.clear ();

  m_nextTriggeredUpdate.Cancel ();
  m_nextUnsolicitedUpdate.Cancel ();
  m_nextTriggeredUpdate = EventId ();
  m_nextUnsolicitedUpdate = EventId ();

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      iter->first->Close ();
    }
  m_unicastSocketList.clear ();

  m_multicastRecvSocket->Close ();
  m_multicastRecvSocket = 0;

  m_ipv6 = 0;

  Ipv6RoutingProtocol::DoDispose ();
}

Ptr<Ipv6Route> RipNg::Lookup (Ipv6Address dst, bool setSource, Ptr<NetDevice> interface)
{
  NS_LOG_FUNCTION (this << dst << interface);

  Ptr<Ipv6Route> rtentry = 0;
  uint16_t longestMask = 0;

  /* when sending on link-local multicast, there have to be interface specified */
  if (dst.IsLinkLocalMulticast ())
    {
      NS_ASSERT_MSG (interface, "Try to send on link-local multicast address, and no interface index is given!");
      rtentry = Create<Ipv6Route> ();
      rtentry->SetSource (m_ipv6->SourceAddressSelection (m_ipv6->GetInterfaceForDevice (interface), dst));
      rtentry->SetDestination (dst);
      rtentry->SetGateway (Ipv6Address::GetZero ());
      rtentry->SetOutputDevice (interface);
      return rtentry;
    }

  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      RipNgRoutingTableEntry* j = it->first;

      if (j->GetRouteStatus () == RipNgRoutingTableEntry::RIPNG_VALID)
        {
          Ipv6Prefix mask = j->GetDestNetworkPrefix ();
          uint16_t maskLen = mask.GetPrefixLength ();
          Ipv6Address entry = j->GetDestNetwork ();

          NS_LOG_LOGIC ("Searching for route to " << dst << ", mask length " << maskLen);

          if (mask.IsMatch (dst, entry))
            {
              NS_LOG_LOGIC ("Found global network route " << j << ", mask length " << maskLen);

              /* if interface is given, check the route will output on this interface */
              if (!interface || interface == m_ipv6->GetNetDevice (j->GetInterface ()))
                {
                  if (maskLen < longestMask)
                    {
                      NS_LOG_LOGIC ("Previous match longer, skipping");
                      continue;
                    }

                  longestMask = maskLen;

                  Ipv6RoutingTableEntry* route = j;
                  uint32_t interfaceIdx = route->GetInterface ();
                  rtentry = Create<Ipv6Route> ();

                  if (setSource)
                    {
                      if (route->GetGateway ().IsAny ())
                        {
                          rtentry->SetSource (m_ipv6->SourceAddressSelection (interfaceIdx, route->GetDest ()));
                        }
                      else if (route->GetDest ().IsAny ()) /* default route */
                        {
                          rtentry->SetSource (m_ipv6->SourceAddressSelection (interfaceIdx, route->GetPrefixToUse ().IsAny () ? dst : route->GetPrefixToUse ()));
                        }
                      else
                        {
                          rtentry->SetSource (m_ipv6->SourceAddressSelection (interfaceIdx, route->GetDest ()));
                        }
                    }

                  rtentry->SetDestination (route->GetDest ());
                  rtentry->SetGateway (route->GetGateway ());
                  rtentry->SetOutputDevice (m_ipv6->GetNetDevice (interfaceIdx));
                }
            }
        }
    }

  if (rtentry)
    {
      NS_LOG_LOGIC ("Matching route via " << rtentry->GetDestination () << " (through " << rtentry->GetGateway () << ") at the end");
    }
  return rtentry;
}

void RipNg::AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << nextHop << interface << prefixToUse);

  if (nextHop.IsLinkLocal())
    {
      NS_LOG_WARN ("Ripng::AddNetworkRouteTo - Next hop should be link-local");
    }

  RipNgRoutingTableEntry* route = new RipNgRoutingTableEntry (network, networkPrefix, nextHop, interface, prefixToUse);
  route->SetRouteMetric (1);
  route->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
  route->SetRouteChanged (true);

  m_routes.push_back (std::make_pair (route, EventId ()));
}

void RipNg::AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << interface);

  RipNgRoutingTableEntry* route = new RipNgRoutingTableEntry (network, networkPrefix, interface);
  route->SetRouteMetric (1);
  route->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
  route->SetRouteChanged (true);

  m_routes.push_back (std::make_pair (route, EventId ()));
}

void RipNg::InvalidateRoute (RipNgRoutingTableEntry *route)
{
  NS_LOG_FUNCTION (this << *route);

  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first == route)
        {
          route->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_INVALID);
          route->SetRouteMetric (m_linkDown);
          route->SetRouteChanged (true);
          if (it->second.IsRunning ())
            {
              it->second.Cancel ();
            }
          it->second = Simulator::Schedule (m_garbageCollectionDelay, &RipNg::DeleteRoute, this, route);
          return;
        }
    }
  NS_ABORT_MSG ("Ripng::InvalidateRoute - cannot find the route to update");
}

void RipNg::DeleteRoute (RipNgRoutingTableEntry *route)
{
  NS_LOG_FUNCTION (this << *route);

  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first == route)
        {
          delete route;
          m_routes.erase (it);
          return;
        }
    }
  NS_ABORT_MSG ("Ripng::DeleteRoute - cannot find the route to delete");
}


void RipNg::Receive (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  Address sender;
  Ptr<Packet> packet = socket->RecvFrom (sender);
  Inet6SocketAddress senderAddr = Inet6SocketAddress::ConvertFrom (sender);
  NS_LOG_INFO ("Received " << *packet << " from " << senderAddr);

  Ipv6Address senderAddress = senderAddr.GetIpv6 ();
  uint16_t senderPort = senderAddr.GetPort ();

  Ipv6PacketInfoTag interfaceInfo;
  if (!packet->RemovePacketTag (interfaceInfo))
    {
      NS_ABORT_MSG ("No incoming interface on RIPng message, aborting.");
    }
  uint32_t incomingIf = interfaceInfo.GetRecvIf ();
  Ptr<Node> node = this->GetObject<Node> ();
  Ptr<NetDevice> dev = node->GetDevice (incomingIf);
  uint32_t ipInterfaceIndex = m_ipv6->GetInterfaceForDevice (dev);

  SocketIpv6HopLimitTag hoplimitTag;
  if (!packet->RemovePacketTag (hoplimitTag))
    {
      NS_ABORT_MSG ("No incoming Hop Count on RIPng message, aborting.");
    }
  uint8_t hopLimit = hoplimitTag.GetHopLimit ();

  int32_t interfaceForAddress = m_ipv6->GetInterfaceForAddress (senderAddress);
  if (interfaceForAddress != -1)
    {
      NS_LOG_LOGIC ("Ignoring a packet sent by myself.");
      return;
    }

  RipNgHeader hdr;
  packet->RemoveHeader (hdr);

  if (hdr.GetCommand () == RipNgHeader::RESPONSE)
    {
      HandleResponses (hdr, senderAddress, ipInterfaceIndex, hopLimit);
    }
  else if (hdr.GetCommand () == RipNgHeader::REQUEST)
    {
      HandleRequests (hdr, senderAddress, senderPort, ipInterfaceIndex, hopLimit);
    }
  else
    {
      NS_LOG_LOGIC ("Ignoring message with unknown command: " << int (hdr.GetCommand ()));
    }
  return;
}

void RipNg::HandleRequests (RipNgHeader requestHdr, Ipv6Address senderAddress, uint16_t senderPort, uint32_t incomingInterface, uint8_t hopLimit)
{
  NS_LOG_FUNCTION (this << senderAddress << int (senderPort) << incomingInterface << int (hopLimit) << requestHdr);

  std::list<RipNgRte> rtes = requestHdr.GetRteList ();

  if (rtes.empty ())
    {
      return;
    }

  // check if it's a request for the full table from a neighbor
  if (rtes.size () == 1 && senderAddress.IsLinkLocal ())
    {
      if (rtes.begin ()->GetPrefix () == Ipv6Address::GetAny () &&
          rtes.begin ()->GetPrefixLen () == 0 &&
          rtes.begin ()->GetRouteMetric () == m_linkDown)
        {
          // Output whole thing. Use Split Horizon
          if (m_interfaceExclusions.find (incomingInterface) == m_interfaceExclusions.end ())
            {
              // we use one of the sending sockets, as they're bound to the right interface
              // and the local address might be used on different interfaces.
              Ptr<Socket> sendingSocket;
              for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
                {
                  if (iter->second == incomingInterface)
                    {
                      sendingSocket = iter->first;
                    }
                }
              NS_ASSERT_MSG (sendingSocket, "HandleRequest - Impossible to find a socket to send the reply");

              uint16_t mtu = m_ipv6->GetMtu (incomingInterface);
              uint16_t maxRte = (mtu - Ipv6Header ().GetSerializedSize () - UdpHeader ().GetSerializedSize () - RipNgHeader ().GetSerializedSize ()) / RipNgRte ().GetSerializedSize ();

              Ptr<Packet> p = Create<Packet> ();
              SocketIpv6HopLimitTag tag;
              p->RemovePacketTag (tag);
              tag.SetHopLimit (255);
              p->AddPacketTag (tag);

              RipNgHeader hdr;
              hdr.SetCommand (RipNgHeader::RESPONSE);

              for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
                {
                  bool splitHorizoning = (rtIter->first->GetInterface () == incomingInterface);

                  Ipv6InterfaceAddress rtDestAddr = Ipv6InterfaceAddress (rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkPrefix ());

                  bool isGlobal = (rtDestAddr.GetScope () == Ipv6InterfaceAddress::GLOBAL);
                  bool isDefaultRoute = ((rtIter->first->GetDestNetwork () == Ipv6Address::GetAny ()) &&
                      (rtIter->first->GetDestNetworkPrefix () == Ipv6Prefix::GetZero ()) &&
                      (rtIter->first->GetInterface () != incomingInterface));

                  if ((isGlobal || isDefaultRoute) &&
                      (rtIter->first->GetRouteStatus () == RipNgRoutingTableEntry::RIPNG_VALID) )
                    {
                      RipNgRte rte;
                      rte.SetPrefix (rtIter->first->GetDestNetwork ());
                      rte.SetPrefixLen (rtIter->first->GetDestNetworkPrefix ().GetPrefixLength ());
                      if (m_splitHorizonStrategy == POISON_REVERSE && splitHorizoning)
                        {
                          rte.SetRouteMetric (m_linkDown);
                        }
                      else
                        {
                          rte.SetRouteMetric (rtIter->first->GetRouteMetric ());
                        }
                      rte.SetRouteTag (rtIter->first->GetRouteTag ());
                      if ((m_splitHorizonStrategy != SPLIT_HORIZON) ||
                          (m_splitHorizonStrategy == SPLIT_HORIZON && !splitHorizoning))
                        {
                          hdr.AddRte (rte);
                        }
                    }
                  if (hdr.GetRteNumber () == maxRte)
                    {
                      p->AddHeader (hdr);
                      NS_LOG_DEBUG ("SendTo: " << *p);
                      sendingSocket->SendTo (p, 0, Inet6SocketAddress (senderAddress, RIPNG_PORT));
                      p->RemoveHeader (hdr);
                      hdr.ClearRtes ();
                    }
                }
              if (hdr.GetRteNumber () > 0)
                {
                  p->AddHeader (hdr);
                  NS_LOG_DEBUG ("SendTo: " << *p);
                  sendingSocket->SendTo (p, 0, Inet6SocketAddress (senderAddress, RIPNG_PORT));
                }
            }
        }
    }
  else
    {
      // note: we got the request as a single packet, so no check is necessary for MTU limit

      // we use one of the sending sockets, as they're bound to the right interface
      // and the local address might be used on different interfaces.
      Ptr<Socket> sendingSocket;
      if (senderAddress.IsLinkLocal ())
        {
          for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
            {
              if (iter->second == incomingInterface)
                {
                  sendingSocket = iter->first;
                }
            }
        }
      else
        {
          sendingSocket = m_multicastRecvSocket;
        }

      Ptr<Packet> p = Create<Packet> ();
      SocketIpv6HopLimitTag tag;
      p->RemovePacketTag (tag);
      tag.SetHopLimit (255);
      p->AddPacketTag (tag);

      RipNgHeader hdr;
      hdr.SetCommand (RipNgHeader::RESPONSE);

      for (std::list<RipNgRte>::iterator iter = rtes.begin ();
          iter != rtes.end (); iter++)
        {
          bool found = false;
          for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
            {
              Ipv6InterfaceAddress rtDestAddr = Ipv6InterfaceAddress(rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkPrefix ());
              if ((rtDestAddr.GetScope () == Ipv6InterfaceAddress::GLOBAL) &&
                  (rtIter->first->GetRouteStatus () == RipNgRoutingTableEntry::RIPNG_VALID))
                {
                  Ipv6Address requestedAddress = iter->GetPrefix ();
                  requestedAddress.CombinePrefix (Ipv6Prefix (iter->GetPrefixLen ()));
                  Ipv6Address rtAddress = rtIter->first->GetDestNetwork ();
                  rtAddress.CombinePrefix (rtIter->first->GetDestNetworkPrefix ());

                  if (requestedAddress == rtAddress)
                    {
                      iter->SetRouteMetric (rtIter->first->GetRouteMetric ());
                      iter->SetRouteTag (rtIter->first->GetRouteTag ());
                      hdr.AddRte (*iter);
                      found = true;
                      break;
                    }
                }
            }
          if (!found)
            {
              iter->SetRouteMetric (m_linkDown);
              iter->SetRouteTag (0);
              hdr.AddRte (*iter);
            }
        }
      p->AddHeader (hdr);
      NS_LOG_DEBUG ("SendTo: " << *p);
      sendingSocket->SendTo (p, 0, Inet6SocketAddress (senderAddress, senderPort));
    }

}

void RipNg::HandleResponses (RipNgHeader hdr, Ipv6Address senderAddress, uint32_t incomingInterface, uint8_t hopLimit)
{
  NS_LOG_FUNCTION (this << senderAddress << incomingInterface << int (hopLimit) << hdr);

  if (m_interfaceExclusions.find (incomingInterface) != m_interfaceExclusions.end ())
    {
      NS_LOG_LOGIC ("Ignoring an update message from an excluded interface: " << incomingInterface);
      return;
    }

  if (!senderAddress.IsLinkLocal ())
    {
      NS_LOG_LOGIC ("Ignoring an update message from a non-link-local source: " << senderAddress);
      return;
    }

  if (hopLimit != 255)
    {
      NS_LOG_LOGIC ("Ignoring an update message with suspicious hop count: " << int (hopLimit));
      return;
    }

  std::list<RipNgRte> rtes = hdr.GetRteList ();

  // validate the RTEs before processing
  for (std::list<RipNgRte>::iterator iter = rtes.begin ();
      iter != rtes.end (); iter++)
    {
      if (iter->GetRouteMetric () == 0 || iter->GetRouteMetric () > m_linkDown)
        {
          NS_LOG_LOGIC ("Ignoring an update message with malformed metric: " << int (iter->GetRouteMetric ()));
          return;
        }
      if (iter->GetPrefixLen () > 128)
        {
          NS_LOG_LOGIC ("Ignoring an update message with malformed prefix length: " << int (iter->GetPrefixLen ()));
          return;
        }
      if (iter->GetPrefix ().IsLocalhost () ||
          iter->GetPrefix ().IsLinkLocal () ||
          iter->GetPrefix ().IsMulticast ())
        {
          NS_LOG_LOGIC ("Ignoring an update message with wrong prefixes: " << iter->GetPrefix ());
          return;
        }
    }

  bool changed = false;

  for (std::list<RipNgRte>::iterator iter = rtes.begin ();
      iter != rtes.end (); iter++)
    {
      Ipv6Prefix rtePrefix = Ipv6Prefix (iter->GetPrefixLen ());
      Ipv6Address rteAddr = iter->GetPrefix ().CombinePrefix (rtePrefix);

      NS_LOG_LOGIC ("Processing RTE " << *iter);

      uint8_t interfaceMetric = 1;
      if (m_interfaceMetrics.find (incomingInterface) != m_interfaceMetrics.end ())
        {
          interfaceMetric = m_interfaceMetrics[incomingInterface];
        }
      uint16_t rteMetric = iter->GetRouteMetric () + interfaceMetric;
      if (rteMetric > m_linkDown)
        {
          rteMetric = m_linkDown;
        }
      RoutesI it;
      bool found = false;
      for (it = m_routes.begin (); it != m_routes.end (); it++)
        {
          if (it->first->GetDestNetwork () == rteAddr &&
              it->first->GetDestNetworkPrefix () == rtePrefix)
            {
              found = true;
              if (rteMetric < it->first->GetRouteMetric ())
                {
                  if (senderAddress != it->first->GetGateway ())
                    {
                      RipNgRoutingTableEntry* route = new RipNgRoutingTableEntry (rteAddr, rtePrefix, senderAddress, incomingInterface, Ipv6Address::GetAny ());
                      delete it->first;
                      it->first = route;
                    }
                  it->first->SetRouteMetric (rteMetric);
                  it->first->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
                  it->first->SetRouteTag (iter->GetRouteTag ());
                  it->first->SetRouteChanged (true);
                  it->second.Cancel ();
                  it->second = Simulator::Schedule (m_timeoutDelay, &RipNg::InvalidateRoute, this, it->first);
                  changed = true;
                }
              else if (rteMetric == it->first->GetRouteMetric ())
                {
                  if (senderAddress == it->first->GetGateway ())
                    {
                      it->second.Cancel ();
                      it->second = Simulator::Schedule (m_timeoutDelay, &RipNg::InvalidateRoute, this, it->first);
                    }
                  else
                    {
                      if (Simulator::GetDelayLeft (it->second) < m_timeoutDelay/2)
                        {
                          RipNgRoutingTableEntry* route = new RipNgRoutingTableEntry (rteAddr, rtePrefix, senderAddress, incomingInterface, Ipv6Address::GetAny ());
                          route->SetRouteMetric (rteMetric);
                          route->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
                          route->SetRouteTag (iter->GetRouteTag ());
                          route->SetRouteChanged (true);
                          delete it->first;
                          it->first = route;
                          it->second.Cancel ();
                          it->second = Simulator::Schedule (m_timeoutDelay, &RipNg::InvalidateRoute, this, route);
                          changed = true;
                        }
                    }
                }
              else if (rteMetric > it->first->GetRouteMetric () && senderAddress == it->first->GetGateway ())
                {
                  it->second.Cancel ();
                  if (rteMetric < m_linkDown)
                    {
                      it->first->SetRouteMetric (rteMetric);
                      it->first->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
                      it->first->SetRouteTag (iter->GetRouteTag ());
                      it->first->SetRouteChanged (true);
                      it->second.Cancel ();
                      it->second = Simulator::Schedule (m_timeoutDelay, &RipNg::InvalidateRoute, this, it->first);
                    }
                  else
                    {
                      InvalidateRoute (it->first);
                    }
                  changed = true;
                }
            }
        }
      if (!found && rteMetric != m_linkDown)
        {
          NS_LOG_LOGIC ("Received a RTE with new route, adding.");

          RipNgRoutingTableEntry* route = new RipNgRoutingTableEntry (rteAddr, rtePrefix, senderAddress, incomingInterface, Ipv6Address::GetAny ());
          route->SetRouteMetric (rteMetric);
          route->SetRouteStatus (RipNgRoutingTableEntry::RIPNG_VALID);
          route->SetRouteChanged (true);
          m_routes.push_front (std::make_pair (route, EventId ()));
          EventId invalidateEvent = Simulator::Schedule (m_timeoutDelay, &RipNg::InvalidateRoute, this, route);
          (m_routes.begin ())->second = invalidateEvent;
          changed = true;
        }
    }

  if (changed)
    {
      SendTriggeredRouteUpdate ();
    }
}

void RipNg::DoSendRouteUpdate (bool periodic)
{
  NS_LOG_FUNCTION (this << (periodic ? " periodic" : " triggered"));

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      uint32_t interface = iter->second;

      if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
        {
          uint16_t mtu = m_ipv6->GetMtu (interface);
          uint16_t maxRte = (mtu - Ipv6Header ().GetSerializedSize () - UdpHeader ().GetSerializedSize () - RipNgHeader ().GetSerializedSize ()) / RipNgRte ().GetSerializedSize ();

          Ptr<Packet> p = Create<Packet> ();
          SocketIpv6HopLimitTag tag;
          tag.SetHopLimit (255);
          p->AddPacketTag (tag);

          RipNgHeader hdr;
          hdr.SetCommand (RipNgHeader::RESPONSE);

          for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
            {
              bool splitHorizoning = (rtIter->first->GetInterface () == interface);
              Ipv6InterfaceAddress rtDestAddr = Ipv6InterfaceAddress(rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkPrefix ());

              NS_LOG_DEBUG ("Processing RT " << rtDestAddr << " " << int(rtIter->first->IsRouteChanged ()));

              bool isGlobal = (rtDestAddr.GetScope () == Ipv6InterfaceAddress::GLOBAL);
              bool isDefaultRoute = ((rtIter->first->GetDestNetwork () == Ipv6Address::GetAny ()) &&
                  (rtIter->first->GetDestNetworkPrefix () == Ipv6Prefix::GetZero ()) &&
                  (rtIter->first->GetInterface () != interface));

              if ((isGlobal || isDefaultRoute) &&
                  (periodic || rtIter->first->IsRouteChanged ()))
                {
                  RipNgRte rte;
                  rte.SetPrefix (rtIter->first->GetDestNetwork ());
                  rte.SetPrefixLen (rtIter->first->GetDestNetworkPrefix ().GetPrefixLength ());
                  if (m_splitHorizonStrategy == POISON_REVERSE && splitHorizoning)
                    {
                      rte.SetRouteMetric (m_linkDown);
                    }
                  else
                    {
                      rte.SetRouteMetric (rtIter->first->GetRouteMetric ());
                    }
                  rte.SetRouteTag (rtIter->first->GetRouteTag ());
                  if (m_splitHorizonStrategy == SPLIT_HORIZON && !splitHorizoning)
                    {
                      hdr.AddRte (rte);
                    }
                  else if (m_splitHorizonStrategy != SPLIT_HORIZON)
                    {
                      hdr.AddRte (rte);
                    }
                }
              if (hdr.GetRteNumber () == maxRte)
                {
                  p->AddHeader (hdr);
                  NS_LOG_DEBUG ("SendTo: " << *p);
                  iter->first->SendTo (p, 0, Inet6SocketAddress (RIPNG_ALL_NODE, RIPNG_PORT));
                  p->RemoveHeader (hdr);
                  hdr.ClearRtes ();
                }
            }
          if (hdr.GetRteNumber () > 0)
            {
              p->AddHeader (hdr);
              NS_LOG_DEBUG ("SendTo: " << *p);
              iter->first->SendTo (p, 0, Inet6SocketAddress (RIPNG_ALL_NODE, RIPNG_PORT));
            }
        }
    }
  for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
    {
      rtIter->first->SetRouteChanged (false);
    }
}

void RipNg::SendTriggeredRouteUpdate ()
{
  NS_LOG_FUNCTION (this);

  if (m_nextTriggeredUpdate.IsRunning())
    {
      NS_LOG_LOGIC ("Skipping Triggered Update due to cooldown");
      return;
    }

  // DoSendRouteUpdate (false);

  // note: The RFC states:
  //     After a triggered
  //     update is sent, a timer should be set for a random interval between 1
  //     and 5 seconds.  If other changes that would trigger updates occur
  //     before the timer expires, a single update is triggered when the timer
  //     expires.  The timer is then reset to another random value between 1
  //     and 5 seconds.  Triggered updates may be suppressed if a regular
  //     update is due by the time the triggered update would be sent.
  // Here we rely on this:
  // When an update occurs (either Triggered or Periodic) the "IsChanged ()"
  // route field will be cleared.
  // Hence, the following Triggered Update will be fired, but will not send
  // any route update.

  Time delay = Seconds (m_rng->GetValue (m_minTriggeredUpdateDelay.GetSeconds (), m_maxTriggeredUpdateDelay.GetSeconds ()));
  m_nextTriggeredUpdate = Simulator::Schedule (delay, &RipNg::DoSendRouteUpdate, this, false);
}

void RipNg::SendUnsolicitedRouteUpdate ()
{
  NS_LOG_FUNCTION (this);

  if (m_nextTriggeredUpdate.IsRunning())
    {
      m_nextTriggeredUpdate.Cancel ();
    }

  DoSendRouteUpdate (true);

  Time delay = m_unsolicitedUpdate + Seconds (m_rng->GetValue (0, 0.5*m_unsolicitedUpdate.GetSeconds ()) );
  m_nextUnsolicitedUpdate = Simulator::Schedule (delay, &RipNg::SendUnsolicitedRouteUpdate, this);
}

std::set<uint32_t> RipNg::GetInterfaceExclusions () const
{
  return m_interfaceExclusions;
}

void RipNg::SetInterfaceExclusions (std::set<uint32_t> exceptions)
{
  NS_LOG_FUNCTION (this);

  m_interfaceExclusions = exceptions;
}

uint8_t RipNg::GetInterfaceMetric (uint32_t interface) const
{
  NS_LOG_FUNCTION (this << interface);

  std::map<uint32_t, uint8_t>::const_iterator iter = m_interfaceMetrics.find (interface);
  if (iter != m_interfaceMetrics.end ())
    {
      return iter->second;
    }
  return 1;
}

void RipNg::SetInterfaceMetric (uint32_t interface, uint8_t metric)
{
  NS_LOG_FUNCTION (this << interface << int (metric));

  if (metric < m_linkDown)
    {
      m_interfaceMetrics[interface] = metric;
    }
}

void RipNg::SendRouteRequest ()
{
  NS_LOG_FUNCTION (this);

  Ptr<Packet> p = Create<Packet> ();
  SocketIpv6HopLimitTag tag;
  p->RemovePacketTag (tag);
  tag.SetHopLimit (255);
  p->AddPacketTag (tag);

  RipNgHeader hdr;
  hdr.SetCommand (RipNgHeader::REQUEST);

  RipNgRte rte;
  rte.SetPrefix (Ipv6Address::GetAny ());
  rte.SetPrefixLen (0);
  rte.SetRouteMetric (m_linkDown);

  hdr.AddRte (rte);
  p->AddHeader (hdr);

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      uint32_t interface = iter->second;

      if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
        {
          NS_LOG_DEBUG ("SendTo: " << *p);
          iter->first->SendTo (p, 0, Inet6SocketAddress (RIPNG_ALL_NODE, RIPNG_PORT));
        }
    }
}

void RipNg::AddDefaultRouteTo (Ipv6Address nextHop, uint32_t interface)
{
  NS_LOG_FUNCTION (this << interface);

  AddNetworkRouteTo (Ipv6Address ("::"), Ipv6Prefix::GetZero (), nextHop, interface, Ipv6Address ("::"));
}


/*
 * RipNgRoutingTableEntry
 */

RipNgRoutingTableEntry::RipNgRoutingTableEntry ()
  : m_tag (0), m_metric (0), m_status (RIPNG_INVALID), m_changed (false)
{
}

RipNgRoutingTableEntry::RipNgRoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
  : Ipv6RoutingTableEntry ( RipNgRoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, nextHop, interface, prefixToUse) ),
    m_tag (0), m_metric (0), m_status (RIPNG_INVALID), m_changed (false)
{
}

RipNgRoutingTableEntry::RipNgRoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface)
  : Ipv6RoutingTableEntry ( Ipv6RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, interface) ),
    m_tag (0), m_metric (0), m_status (RIPNG_INVALID), m_changed (false)
{
}

RipNgRoutingTableEntry::~RipNgRoutingTableEntry ()
{
}


void RipNgRoutingTableEntry::SetRouteTag (uint16_t routeTag)
{
  if (m_tag != routeTag)
    {
      m_tag = routeTag;
      m_changed = true;
    }
}

uint16_t RipNgRoutingTableEntry::GetRouteTag () const
{
  return m_tag;
}

void RipNgRoutingTableEntry::SetRouteMetric (uint8_t routeMetric)
{
  if (m_metric != routeMetric)
    {
      m_metric = routeMetric;
      m_changed = true;
    }
}

uint8_t RipNgRoutingTableEntry::GetRouteMetric () const
{
  return m_metric;
}

void RipNgRoutingTableEntry::SetRouteStatus (Status_e status)
{
  if (m_status != status)
    {
      m_status = status;
      m_changed = true;
    }
}

RipNgRoutingTableEntry::Status_e RipNgRoutingTableEntry::GetRouteStatus (void) const
{
  return m_status;
}

void RipNgRoutingTableEntry::SetRouteChanged (bool changed)
{
  m_changed = changed;
}

bool RipNgRoutingTableEntry::IsRouteChanged (void) const
{
  return m_changed;
}


std::ostream & operator << (std::ostream& os, const RipNgRoutingTableEntry& rte)
{
  os << static_cast<const Ipv6RoutingTableEntry &>(rte);
  os << ", metric: " << int (rte.GetRouteMetric ()) << ", tag: " << int (rte.GetRouteTag ());

  return os;
}


}

