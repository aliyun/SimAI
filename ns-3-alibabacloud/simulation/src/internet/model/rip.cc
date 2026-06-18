/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' di Firenze, Italy
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
#include "rip.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/assert.h"
#include "ns3/random-variable-stream.h"
#include "ns3/ipv4-route.h"
#include "ns3/node.h"
#include "ns3/names.h"
#include "ns3/rip-header.h"
#include "ns3/udp-header.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "ns3/ipv4-packet-info-tag.h"
#include "ns3/loopback-net-device.h"

#define RIP_ALL_NODE "224.0.0.9"
#define RIP_PORT 520

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Rip");

NS_OBJECT_ENSURE_REGISTERED (Rip);

Rip::Rip ()
  : m_ipv4 (0), m_splitHorizonStrategy (Rip::POISON_REVERSE), m_initialized (false)
{
  m_rng = CreateObject<UniformRandomVariable> ();
}

Rip::~Rip ()
{
}

TypeId
Rip::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Rip")
    .SetParent<Ipv4RoutingProtocol> ()
    .SetGroupName ("Internet")
    .AddConstructor<Rip> ()
    .AddAttribute ("UnsolicitedRoutingUpdate", "The time between two Unsolicited Routing Updates.",
                   TimeValue (Seconds(30)),
                   MakeTimeAccessor (&Rip::m_unsolicitedUpdate),
                   MakeTimeChecker ())
    .AddAttribute ("StartupDelay", "Maximum random delay for protocol startup (send route requests).",
                   TimeValue (Seconds(1)),
                   MakeTimeAccessor (&Rip::m_startupDelay),
                   MakeTimeChecker ())
    .AddAttribute ("TimeoutDelay", "The delay to invalidate a route.",
                   TimeValue (Seconds(180)),
                   MakeTimeAccessor (&Rip::m_timeoutDelay),
                   MakeTimeChecker ())
    .AddAttribute ("GarbageCollectionDelay", "The delay to delete an expired route.",
                   TimeValue (Seconds(120)),
                   MakeTimeAccessor (&Rip::m_garbageCollectionDelay),
                   MakeTimeChecker ())
    .AddAttribute ("MinTriggeredCooldown", "Min cooldown delay after a Triggered Update.",
                   TimeValue (Seconds(1)),
                   MakeTimeAccessor (&Rip::m_minTriggeredUpdateDelay),
                   MakeTimeChecker ())
    .AddAttribute ("MaxTriggeredCooldown", "Max cooldown delay after a Triggered Update.",
                   TimeValue (Seconds(5)),
                   MakeTimeAccessor (&Rip::m_maxTriggeredUpdateDelay),
                   MakeTimeChecker ())
    .AddAttribute ("SplitHorizon", "Split Horizon strategy.",
                   EnumValue (Rip::POISON_REVERSE),
                   MakeEnumAccessor (&Rip::m_splitHorizonStrategy),
                   MakeEnumChecker (Rip::NO_SPLIT_HORIZON, "NoSplitHorizon",
                                    Rip::SPLIT_HORIZON, "SplitHorizon",
                                    Rip::POISON_REVERSE, "PoisonReverse"))
    .AddAttribute ("LinkDownValue", "Value for link down in count to infinity.",
                   UintegerValue (16),
                   MakeUintegerAccessor (&Rip::m_linkDown),
                   MakeUintegerChecker<uint32_t> ())
    ;
  return tid;
}

int64_t Rip::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);

  m_rng->SetStream (stream);
  return 1;
}

void Rip::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  bool addedGlobal = false;

  m_initialized = true;

  Time delay = m_unsolicitedUpdate + Seconds (m_rng->GetValue (0, 0.5*m_unsolicitedUpdate.GetSeconds ()) );
  m_nextUnsolicitedUpdate = Simulator::Schedule (delay, &Rip::SendUnsolicitedRouteUpdate, this);


  for (uint32_t i = 0 ; i < m_ipv4->GetNInterfaces (); i++)
    {
      Ptr<LoopbackNetDevice> check = DynamicCast<LoopbackNetDevice> (m_ipv4->GetNetDevice (i));
      if (check)
        {
          continue;
        }

      bool activeInterface = false;
      if (m_interfaceExclusions.find (i) == m_interfaceExclusions.end ())
        {
          activeInterface = true;
          m_ipv4->SetForwarding (i, true);
        }

      for (uint32_t j = 0; j < m_ipv4->GetNAddresses (i); j++)
        {
          Ipv4InterfaceAddress address = m_ipv4->GetAddress (i, j);
          if (address.GetScope() != Ipv4InterfaceAddress::HOST && activeInterface == true)
            {
              NS_LOG_LOGIC ("RIP: adding socket to " << address.GetLocal ());
              TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
              Ptr<Node> theNode = GetObject<Node> ();
              Ptr<Socket> socket = Socket::CreateSocket (theNode, tid);
              InetSocketAddress local = InetSocketAddress (address.GetLocal (), RIP_PORT);
              socket->BindToNetDevice (m_ipv4->GetNetDevice (i));
              int ret = socket->Bind (local);
              NS_ASSERT_MSG (ret == 0, "Bind unsuccessful");

              socket->SetRecvCallback (MakeCallback (&Rip::Receive, this));
              socket->SetIpRecvTtl (true);
              socket->SetRecvPktInfo (true);

              m_unicastSocketList[socket] = i;
            }
          else if (m_ipv4->GetAddress (i, j).GetScope() == Ipv4InterfaceAddress::GLOBAL)
            {
              addedGlobal = true;
            }
        }
    }

  if (!m_multicastRecvSocket)
    {
      NS_LOG_LOGIC ("RIP: adding receiving socket");
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      Ptr<Node> theNode = GetObject<Node> ();
      m_multicastRecvSocket = Socket::CreateSocket (theNode, tid);
      InetSocketAddress local = InetSocketAddress (RIP_ALL_NODE, RIP_PORT);
      m_multicastRecvSocket->Bind (local);
      m_multicastRecvSocket->SetRecvCallback (MakeCallback (&Rip::Receive, this));
      m_multicastRecvSocket->SetIpRecvTtl (true);
      m_multicastRecvSocket->SetRecvPktInfo (true);
    }


  if (addedGlobal)
    {
      Time delay = Seconds (m_rng->GetValue (m_minTriggeredUpdateDelay.GetSeconds (), m_maxTriggeredUpdateDelay.GetSeconds ()));
      m_nextTriggeredUpdate = Simulator::Schedule (delay, &Rip::DoSendRouteUpdate, this, false);
    }

  delay = Seconds (m_rng->GetValue (0.01, m_startupDelay.GetSeconds ()));
  m_nextTriggeredUpdate = Simulator::Schedule (delay, &Rip::SendRouteRequest, this);

  Ipv4RoutingProtocol::DoInitialize ();
}

Ptr<Ipv4Route> Rip::RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)
{
  NS_LOG_FUNCTION (this << header << oif);

  Ipv4Address destination = header.GetDestination ();
  Ptr<Ipv4Route> rtentry = 0;

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

bool Rip::RouteInput (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
                        UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                        LocalDeliverCallback lcb, ErrorCallback ecb)
{
  NS_LOG_FUNCTION (this << p << header << header.GetSource () << header.GetDestination () << idev);

  NS_ASSERT (m_ipv4 != 0);
  // Check if input device supports IP
  NS_ASSERT (m_ipv4->GetInterfaceForDevice (idev) >= 0);
  uint32_t iif = m_ipv4->GetInterfaceForDevice (idev);
  Ipv4Address dst = header.GetDestination ();

    if (m_ipv4->IsDestinationAddress (header.GetDestination (), iif))
    {
      if (!lcb.IsNull ())
        {
          NS_LOG_LOGIC ("Local delivery to " << header.GetDestination ());
          lcb (p, header, iif);
          return true;
        }
      else
        {
          // The local delivery callback is null.  This may be a multicast
          // or broadcast packet, so return false so that another
          // multicast routing protocol can handle it.  It should be possible
          // to extend this to explicitly check whether it is a unicast
          // packet, and invoke the error callback if so
          return false;
        }
    }

    if (dst.IsMulticast ())
      {
        NS_LOG_LOGIC ("Multicast route not supported by RIP");
        return false; // Let other routing protocols try to handle this
      }

  if (header.GetDestination ().IsBroadcast ())
    {
      NS_LOG_LOGIC ("Dropping packet not for me and with dst Broadcast");
      if (!ecb.IsNull ())
        {
          ecb (p, header, Socket::ERROR_NOROUTETOHOST);
        }
      return false;
    }

  // Check if input device supports IP forwarding
  if (m_ipv4->IsForwarding (iif) == false)
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
  Ptr<Ipv4Route> rtentry = Lookup (header.GetDestination (), false);

  if (rtentry != 0)
    {
      NS_LOG_LOGIC ("Found unicast destination - calling unicast callback");
      ucb (rtentry, p, header);  // unicast forwarding callback
      return true;
    }
  else
    {
      NS_LOG_LOGIC ("Did not find unicast destination - returning false");
      return false; // Let other routing protocols try to handle this
    }
}

void Rip::NotifyInterfaceUp (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);

  Ptr<LoopbackNetDevice> check = DynamicCast<LoopbackNetDevice> (m_ipv4->GetNetDevice (i));
  if (check)
    {
      return;
    }

  for (uint32_t j = 0; j < m_ipv4->GetNAddresses (i); j++)
    {
      Ipv4InterfaceAddress address = m_ipv4->GetAddress (i, j);
      Ipv4Mask networkMask = address.GetMask ();
      Ipv4Address networkAddress = address.GetLocal ().CombineMask (networkMask);

      if (address.GetScope () == Ipv4InterfaceAddress::GLOBAL)
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
      m_ipv4->SetForwarding (i, true);
    }

  for (uint32_t j = 0; j < m_ipv4->GetNAddresses (i); j++)
    {
      Ipv4InterfaceAddress address = m_ipv4->GetAddress (i, j);

      if (address.GetScope() != Ipv4InterfaceAddress::HOST && sendSocketFound == false && activeInterface == true)
        {
          NS_LOG_LOGIC ("RIP: adding sending socket to " << address.GetLocal ());
          TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
          Ptr<Node> theNode = GetObject<Node> ();
          Ptr<Socket> socket = Socket::CreateSocket (theNode, tid);
          InetSocketAddress local = InetSocketAddress (address.GetLocal (), RIP_PORT);
          socket->BindToNetDevice (m_ipv4->GetNetDevice (i));
          socket->Bind (local);
          socket->SetRecvCallback (MakeCallback (&Rip::Receive, this));
          socket->SetIpRecvTtl (true);
          socket->SetRecvPktInfo (true);
          m_unicastSocketList[socket] = i;
       }
      if (address.GetScope () == Ipv4InterfaceAddress::GLOBAL)
        {
          SendTriggeredRouteUpdate ();
        }
    }

  if (!m_multicastRecvSocket)
    {
      NS_LOG_LOGIC ("RIP: adding receiving socket");
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      Ptr<Node> theNode = GetObject<Node> ();
      m_multicastRecvSocket = Socket::CreateSocket (theNode, tid);
      InetSocketAddress local = InetSocketAddress (RIP_ALL_NODE, RIP_PORT);
      m_multicastRecvSocket->Bind (local);
      m_multicastRecvSocket->SetRecvCallback (MakeCallback (&Rip::Receive, this));
      m_multicastRecvSocket->SetIpRecvTtl (true);
      m_multicastRecvSocket->SetRecvPktInfo (true);
    }
}

void Rip::NotifyInterfaceDown (uint32_t interface)
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

void Rip::NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address)
{
  NS_LOG_FUNCTION (this << interface << address);

  if (!m_ipv4->IsUp (interface))
    {
      return;
    }

  if (m_interfaceExclusions.find (interface) != m_interfaceExclusions.end ())
    {
      return;
    }

  Ipv4Address networkAddress = address.GetLocal ().CombineMask (address.GetMask ());
  Ipv4Mask networkMask = address.GetMask ();

  if (address.GetScope () == Ipv4InterfaceAddress::GLOBAL)
    {
      AddNetworkRouteTo (networkAddress, networkMask, interface);
    }

  SendTriggeredRouteUpdate ();
}

void Rip::NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address)
{
  NS_LOG_FUNCTION (this << interface << address);

  if (!m_ipv4->IsUp (interface))
    {
      return;
    }

  if (address.GetScope() != Ipv4InterfaceAddress::GLOBAL)
    {
      return;
    }

  Ipv4Address networkAddress = address.GetLocal ().CombineMask (address.GetMask ());
  Ipv4Mask networkMask = address.GetMask ();

  // Remove all routes that are going through this interface
  // which reference this network
  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first->GetInterface () == interface
          && it->first->IsNetwork ()
          && it->first->GetDestNetwork () == networkAddress
          && it->first->GetDestNetworkMask () == networkMask)
        {
          InvalidateRoute (it->first);
        }
    }

  if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
    {
      SendTriggeredRouteUpdate ();
    }

}

void Rip::SetIpv4 (Ptr<Ipv4> ipv4)
{
  NS_LOG_FUNCTION (this << ipv4);

  NS_ASSERT (m_ipv4 == 0 && ipv4 != 0);
  uint32_t i = 0;
  m_ipv4 = ipv4;

  for (i = 0; i < m_ipv4->GetNInterfaces (); i++)
    {
      if (m_ipv4->IsUp (i))
        {
          NotifyInterfaceUp (i);
        }
      else
        {
          NotifyInterfaceDown (i);
        }
    }
}

void Rip::PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION (this << stream);

  std::ostream* os = stream->GetStream ();
  // Copy the current ostream state
  std::ios oldState (nullptr);
  oldState.copyfmt (*os);

  *os << std::resetiosflags (std::ios::adjustfield) << std::setiosflags (std::ios::left);

  *os << "Node: " << m_ipv4->GetObject<Node> ()->GetId ()
      << ", Time: " << Now().As (unit)
      << ", Local time: " << m_ipv4->GetObject<Node> ()->GetLocalTime ().As (unit)
      << ", IPv4 RIP table" << std::endl;

  if (!m_routes.empty ())
    {
      *os << "Destination     Gateway         Genmask         Flags Metric Ref    Use Iface" << std::endl;
      for (RoutesCI it = m_routes.begin (); it != m_routes.end (); it++)
        {
          RipRoutingTableEntry* route = it->first;
          RipRoutingTableEntry::Status_e status = route->GetRouteStatus();

          if (status == RipRoutingTableEntry::RIP_VALID)
            {
              std::ostringstream dest, gw, mask, flags;
              dest << route->GetDest ();
              *os << std::setw (16) << dest.str ();
              gw << route->GetGateway ();
              *os << std::setw (16) << gw.str ();
              mask << route->GetDestNetworkMask ();
              *os << std::setw (16) << mask.str ();
              flags << "U";
              if (route->IsHost ())
                {
                  flags << "HS";
                }
              else if (route->IsGateway ())
                {
                  flags << "GS";
                }
              *os << std::setw (6) << flags.str ();
              *os << std::setw (7) << int(route->GetRouteMetric ());
              // Ref ct not implemented
              *os << "-" << "      ";
              // Use not implemented
              *os << "-" << "   ";
              if (Names::FindName (m_ipv4->GetNetDevice (route->GetInterface ())) != "")
                {
                  *os << Names::FindName (m_ipv4->GetNetDevice (route->GetInterface ()));
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
  // Restore the previous ostream state
  (*os).copyfmt (oldState);
}

void Rip::DoDispose ()
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

  m_ipv4 = 0;

  Ipv4RoutingProtocol::DoDispose ();
}

Ptr<Ipv4Route> Rip::Lookup (Ipv4Address dst, bool setSource, Ptr<NetDevice> interface)
{
  NS_LOG_FUNCTION (this << dst << interface);

  Ptr<Ipv4Route> rtentry = 0;
  uint16_t longestMask = 0;

  /* when sending on local multicast, there have to be interface specified */
  if (dst.IsLocalMulticast ())
    {
      NS_ASSERT_MSG (interface, "Try to send on local multicast address, and no interface index is given!");
      rtentry = Create<Ipv4Route> ();
      rtentry->SetSource (m_ipv4->SourceAddressSelection (m_ipv4->GetInterfaceForDevice (interface), dst));
      rtentry->SetDestination (dst);
      rtentry->SetGateway (Ipv4Address::GetZero ());
      rtentry->SetOutputDevice (interface);
      return rtentry;
    }

  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      RipRoutingTableEntry* j = it->first;

      if (j->GetRouteStatus () == RipRoutingTableEntry::RIP_VALID)
        {
          Ipv4Mask mask = j->GetDestNetworkMask ();
          uint16_t maskLen = mask.GetPrefixLength ();
          Ipv4Address entry = j->GetDestNetwork ();

          NS_LOG_LOGIC ("Searching for route to " << dst << ", mask length " << maskLen);

          if (mask.IsMatch (dst, entry))
            {
              NS_LOG_LOGIC ("Found global network route " << j << ", mask length " << maskLen);

              /* if interface is given, check the route will output on this interface */
              if (!interface || interface == m_ipv4->GetNetDevice (j->GetInterface ()))
                {
                  if (maskLen < longestMask)
                    {
                      NS_LOG_LOGIC ("Previous match longer, skipping");
                      continue;
                    }

                  longestMask = maskLen;

                  Ipv4RoutingTableEntry* route = j;
                  uint32_t interfaceIdx = route->GetInterface ();
                  rtentry = Create<Ipv4Route> ();

                  if (setSource)
                    {
                      if (route->GetDest ().IsAny ()) /* default route */
                        {
                          rtentry->SetSource (m_ipv4->SourceAddressSelection (interfaceIdx, route->GetGateway ()));
                        }
                      else
                        {
                          rtentry->SetSource (m_ipv4->SourceAddressSelection (interfaceIdx, route->GetDest ()));
                        }
                    }

                  rtentry->SetDestination (route->GetDest ());
                  rtentry->SetGateway (route->GetGateway ());
                  rtentry->SetOutputDevice (m_ipv4->GetNetDevice (interfaceIdx));
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

void Rip::AddNetworkRouteTo (Ipv4Address network, Ipv4Mask networkPrefix, Ipv4Address nextHop, uint32_t interface)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << nextHop << interface);

  RipRoutingTableEntry* route = new RipRoutingTableEntry (network, networkPrefix, nextHop, interface);
  route->SetRouteMetric (1);
  route->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
  route->SetRouteChanged (true);

  m_routes.push_back (std::make_pair (route, EventId ()));
}

void Rip::AddNetworkRouteTo (Ipv4Address network, Ipv4Mask networkPrefix, uint32_t interface)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << interface);

  RipRoutingTableEntry* route = new RipRoutingTableEntry (network, networkPrefix, interface);
  route->SetRouteMetric (1);
  route->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
  route->SetRouteChanged (true);

  m_routes.push_back (std::make_pair (route, EventId ()));
}

void Rip::InvalidateRoute (RipRoutingTableEntry *route)
{
  NS_LOG_FUNCTION (this << *route);

  for (RoutesI it = m_routes.begin (); it != m_routes.end (); it++)
    {
      if (it->first == route)
        {
          route->SetRouteStatus (RipRoutingTableEntry::RIP_INVALID);
          route->SetRouteMetric (m_linkDown);
          route->SetRouteChanged (true);
          if (it->second.IsRunning ())
            {
              it->second.Cancel ();
            }
          it->second = Simulator::Schedule (m_garbageCollectionDelay, &Rip::DeleteRoute, this, route);
          return;
        }
    }
  NS_ABORT_MSG ("RIP::InvalidateRoute - cannot find the route to update");
}

void Rip::DeleteRoute (RipRoutingTableEntry *route)
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
  NS_ABORT_MSG ("RIP::DeleteRoute - cannot find the route to delete");
}


void Rip::Receive (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  Address sender;
  Ptr<Packet> packet = socket->RecvFrom (sender);
  InetSocketAddress senderAddr = InetSocketAddress::ConvertFrom (sender);
  NS_LOG_INFO ("Received " << *packet << " from " << senderAddr.GetIpv4 () << ":" << senderAddr.GetPort ());

  Ipv4Address senderAddress = senderAddr.GetIpv4 ();
  uint16_t senderPort = senderAddr.GetPort ();

  if (socket == m_multicastRecvSocket)
    {
      NS_LOG_LOGIC ("Received a packet from the multicast socket");
    }
  else
    {
      NS_LOG_LOGIC ("Received a packet from one of the unicast sockets");
    }

  Ipv4PacketInfoTag interfaceInfo;
  if (!packet->RemovePacketTag (interfaceInfo))
    {
      NS_ABORT_MSG ("No incoming interface on RIP message, aborting.");
    }
  uint32_t incomingIf = interfaceInfo.GetRecvIf ();
  Ptr<Node> node = this->GetObject<Node> ();
  Ptr<NetDevice> dev = node->GetDevice (incomingIf);
  uint32_t ipInterfaceIndex = m_ipv4->GetInterfaceForDevice (dev);

  SocketIpTtlTag hoplimitTag;
  if (!packet->RemovePacketTag (hoplimitTag))
    {
      NS_ABORT_MSG ("No incoming Hop Count on RIP message, aborting.");
    }
  uint8_t hopLimit = hoplimitTag.GetTtl ();

  int32_t interfaceForAddress = m_ipv4->GetInterfaceForAddress (senderAddress);
  if (interfaceForAddress != -1)
    {
      NS_LOG_LOGIC ("Ignoring a packet sent by myself.");
      return;
    }

  RipHeader hdr;
  packet->RemoveHeader (hdr);

  if (hdr.GetCommand () == RipHeader::RESPONSE)
    {
      NS_LOG_LOGIC ("The message is a Response from " << senderAddr.GetIpv4 () << ":" << senderAddr.GetPort ());
      HandleResponses (hdr, senderAddress, ipInterfaceIndex, hopLimit);
    }
  else if (hdr.GetCommand () == RipHeader::REQUEST)
    {
      NS_LOG_LOGIC ("The message is a Request from " << senderAddr.GetIpv4 () << ":" << senderAddr.GetPort ());
      HandleRequests (hdr, senderAddress, senderPort, ipInterfaceIndex, hopLimit);
    }
  else
    {
      NS_LOG_LOGIC ("Ignoring message with unknown command: " << int (hdr.GetCommand ()));
    }
  return;
}

void Rip::HandleRequests (RipHeader requestHdr, Ipv4Address senderAddress, uint16_t senderPort, uint32_t incomingInterface, uint8_t hopLimit)
{
  NS_LOG_FUNCTION (this << senderAddress << int (senderPort) << incomingInterface << int (hopLimit) << requestHdr);

  std::list<RipRte> rtes = requestHdr.GetRteList ();

  if (rtes.empty ())
    {
      return;
    }

  // check if it's a request for the full table from a neighbor
  if (rtes.size () == 1)
    {
      if (rtes.begin ()->GetPrefix () == Ipv4Address::GetAny () &&
          rtes.begin ()->GetSubnetMask ().GetPrefixLength () == 0 &&
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

              uint16_t mtu = m_ipv4->GetMtu (incomingInterface);
              uint16_t maxRte = (mtu - Ipv4Header ().GetSerializedSize () - UdpHeader ().GetSerializedSize () - RipHeader ().GetSerializedSize ()) / RipRte ().GetSerializedSize ();

              Ptr<Packet> p = Create<Packet> ();
              SocketIpTtlTag tag;
              p->RemovePacketTag (tag);
              if (senderAddress == Ipv4Address(RIP_ALL_NODE))
                {
                  tag.SetTtl (1);
                }
              else
                {
                  tag.SetTtl (255);
                }
              p->AddPacketTag (tag);

              RipHeader hdr;
              hdr.SetCommand (RipHeader::RESPONSE);

              for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
                {
                  bool splitHorizoning = (rtIter->first->GetInterface () == incomingInterface);

                  Ipv4InterfaceAddress rtDestAddr = Ipv4InterfaceAddress(rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkMask ());

                  bool isGlobal = (rtDestAddr.GetScope () == Ipv4InterfaceAddress::GLOBAL);
                  bool isDefaultRoute = ((rtIter->first->GetDestNetwork () == Ipv4Address::GetAny ()) &&
                      (rtIter->first->GetDestNetworkMask () == Ipv4Mask::GetZero ()) &&
                      (rtIter->first->GetInterface () != incomingInterface));

                  if ((isGlobal || isDefaultRoute) &&
                      (rtIter->first->GetRouteStatus () == RipRoutingTableEntry::RIP_VALID) )
                    {
                      RipRte rte;
                      rte.SetPrefix (rtIter->first->GetDestNetwork ());
                      rte.SetSubnetMask (rtIter->first->GetDestNetworkMask ());
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
                      sendingSocket->SendTo (p, 0, InetSocketAddress (senderAddress, RIP_PORT));
                      p->RemoveHeader (hdr);
                      hdr.ClearRtes ();
                    }
                }
              if (hdr.GetRteNumber () > 0)
                {
                  p->AddHeader (hdr);
                  NS_LOG_DEBUG ("SendTo: " << *p);
                  sendingSocket->SendTo (p, 0, InetSocketAddress (senderAddress, RIP_PORT));
                }
            }
        }
    }
  else
    {
      // note: we got the request as a single packet, so no check is necessary for MTU limit

      Ptr<Packet> p = Create<Packet> ();
      SocketIpTtlTag tag;
      p->RemovePacketTag (tag);
      if (senderAddress == Ipv4Address(RIP_ALL_NODE))
        {
          tag.SetTtl (1);
        }
      else
        {
          tag.SetTtl (255);
        }
      p->AddPacketTag (tag);

      RipHeader hdr;
      hdr.SetCommand (RipHeader::RESPONSE);

      for (std::list<RipRte>::iterator iter = rtes.begin ();
          iter != rtes.end (); iter++)
        {
          bool found = false;
          for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
            {

              Ipv4InterfaceAddress rtDestAddr = Ipv4InterfaceAddress (rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkMask ());
              if ((rtDestAddr.GetScope () == Ipv4InterfaceAddress::GLOBAL) &&
                  (rtIter->first->GetRouteStatus () == RipRoutingTableEntry::RIP_VALID))
                {
                  Ipv4Address requestedAddress = iter->GetPrefix ();
                  requestedAddress.CombineMask (iter->GetSubnetMask ());
                  Ipv4Address rtAddress = rtIter->first->GetDestNetwork ();
                  rtAddress.CombineMask (rtIter->first->GetDestNetworkMask ());

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
      m_multicastRecvSocket->SendTo (p, 0, InetSocketAddress (senderAddress, senderPort));
    }

}

void Rip::HandleResponses (RipHeader hdr, Ipv4Address senderAddress, uint32_t incomingInterface, uint8_t hopLimit)
{
  NS_LOG_FUNCTION (this << senderAddress << incomingInterface << int (hopLimit) << hdr);

  if (m_interfaceExclusions.find (incomingInterface) != m_interfaceExclusions.end ())
    {
      NS_LOG_LOGIC ("Ignoring an update message from an excluded interface: " << incomingInterface);
      return;
    }

  std::list<RipRte> rtes = hdr.GetRteList ();

  // validate the RTEs before processing
  for (std::list<RipRte>::iterator iter = rtes.begin ();
      iter != rtes.end (); iter++)
    {
      if (iter->GetRouteMetric () == 0 || iter->GetRouteMetric () > m_linkDown)
        {
          NS_LOG_LOGIC ("Ignoring an update message with malformed metric: " << int (iter->GetRouteMetric ()));
          return;
        }
      if (iter->GetPrefix ().IsLocalhost () ||
          iter->GetPrefix ().IsBroadcast () ||
          iter->GetPrefix ().IsMulticast ())
        {
          NS_LOG_LOGIC ("Ignoring an update message with wrong prefixes: " << iter->GetPrefix ());
          return;
        }
    }

  bool changed = false;

  for (std::list<RipRte>::iterator iter = rtes.begin ();
      iter != rtes.end (); iter++)
    {
      Ipv4Mask rtePrefixMask = iter->GetSubnetMask ();
      Ipv4Address rteAddr = iter->GetPrefix ().CombineMask (rtePrefixMask);

      NS_LOG_LOGIC ("Processing RTE " << *iter);

      uint32_t interfaceMetric = 1;
      if (m_interfaceMetrics.find (incomingInterface) != m_interfaceMetrics.end ())
        {
          interfaceMetric = m_interfaceMetrics[incomingInterface];
        }
      uint64_t rteMetric = iter->GetRouteMetric () + interfaceMetric;
      if (rteMetric > m_linkDown)
        {
          rteMetric = m_linkDown;
        }

      RoutesI it;
      bool found = false;
      for (it = m_routes.begin (); it != m_routes.end (); it++)
        {
          if (it->first->GetDestNetwork () == rteAddr &&
              it->first->GetDestNetworkMask () == rtePrefixMask)
            {
              found = true;
              if (rteMetric < it->first->GetRouteMetric ())
                {
                  if (senderAddress != it->first->GetGateway ())
                    {
                      RipRoutingTableEntry* route = new RipRoutingTableEntry (rteAddr, rtePrefixMask, senderAddress, incomingInterface);
                      delete it->first;
                      it->first = route;
                    }
                  it->first->SetRouteMetric (rteMetric);
                  it->first->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
                  it->first->SetRouteTag (iter->GetRouteTag ());
                  it->first->SetRouteChanged (true);
                  it->second.Cancel ();
                  it->second = Simulator::Schedule (m_timeoutDelay, &Rip::InvalidateRoute, this, it->first);
                  changed = true;
                }
              else if (rteMetric == it->first->GetRouteMetric ())
                {
                  if (senderAddress == it->first->GetGateway ())
                    {
                      it->second.Cancel ();
                      it->second = Simulator::Schedule (m_timeoutDelay, &Rip::InvalidateRoute, this, it->first);
                    }
                  else
                    {
                      if (Simulator::GetDelayLeft (it->second) < m_timeoutDelay/2)
                        {
                          RipRoutingTableEntry* route = new RipRoutingTableEntry (rteAddr, rtePrefixMask, senderAddress, incomingInterface);
                          route->SetRouteMetric (rteMetric);
                          route->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
                          route->SetRouteTag (iter->GetRouteTag ());
                          route->SetRouteChanged (true);
                          delete it->first;
                          it->first = route;
                          it->second.Cancel ();
                          it->second = Simulator::Schedule (m_timeoutDelay, &Rip::InvalidateRoute, this, route);
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
                      it->first->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
                      it->first->SetRouteTag (iter->GetRouteTag ());
                      it->first->SetRouteChanged (true);
                      it->second.Cancel ();
                      it->second = Simulator::Schedule (m_timeoutDelay, &Rip::InvalidateRoute, this, it->first);
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

          RipRoutingTableEntry* route = new RipRoutingTableEntry (rteAddr, rtePrefixMask, senderAddress, incomingInterface);
          route->SetRouteMetric (rteMetric);
          route->SetRouteStatus (RipRoutingTableEntry::RIP_VALID);
          route->SetRouteChanged (true);
          m_routes.push_front (std::make_pair (route, EventId ()));
          EventId invalidateEvent = Simulator::Schedule (m_timeoutDelay, &Rip::InvalidateRoute, this, route);
          (m_routes.begin ())->second = invalidateEvent;
          changed = true;
        }
    }

  if (changed)
    {
      SendTriggeredRouteUpdate ();
    }
}

void Rip::DoSendRouteUpdate (bool periodic)
{
  NS_LOG_FUNCTION (this << (periodic ? " periodic" : " triggered"));

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      uint32_t interface = iter->second;

      if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
        {
          uint16_t mtu = m_ipv4->GetMtu (interface);
          uint16_t maxRte = (mtu - Ipv4Header ().GetSerializedSize () - UdpHeader ().GetSerializedSize () - RipHeader ().GetSerializedSize ()) / RipRte ().GetSerializedSize ();

          Ptr<Packet> p = Create<Packet> ();
          SocketIpTtlTag tag;
          tag.SetTtl (1);
          p->AddPacketTag (tag);

          RipHeader hdr;
          hdr.SetCommand (RipHeader::RESPONSE);

          for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
            {
              bool splitHorizoning = (rtIter->first->GetInterface () == interface);
              Ipv4InterfaceAddress rtDestAddr = Ipv4InterfaceAddress(rtIter->first->GetDestNetwork (), rtIter->first->GetDestNetworkMask ());

              NS_LOG_DEBUG ("Processing RT " << rtDestAddr << " " << int(rtIter->first->IsRouteChanged ()));

              bool isGlobal = (rtDestAddr.GetScope () == Ipv4InterfaceAddress::GLOBAL);
              bool isDefaultRoute = ((rtIter->first->GetDestNetwork () == Ipv4Address::GetAny ()) &&
                  (rtIter->first->GetDestNetworkMask () == Ipv4Mask::GetZero ()) &&
                  (rtIter->first->GetInterface () != interface));

              bool sameNetwork = false;
              for (uint32_t index = 0; index < m_ipv4->GetNAddresses (interface); index++)
                {
                  Ipv4InterfaceAddress addr = m_ipv4->GetAddress (interface, index);
                  if (addr.GetLocal ().CombineMask (addr.GetMask ()) == rtIter->first->GetDestNetwork ())
                    {
                      sameNetwork = true;
                    }
                }

              if ((isGlobal || isDefaultRoute) &&
                  (periodic || rtIter->first->IsRouteChanged ()) &&
                  !sameNetwork)
                {
                  RipRte rte;
                  rte.SetPrefix (rtIter->first->GetDestNetwork ());
                  rte.SetSubnetMask (rtIter->first->GetDestNetworkMask ());
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
                  iter->first->SendTo (p, 0, InetSocketAddress (RIP_ALL_NODE, RIP_PORT));
                  p->RemoveHeader (hdr);
                  hdr.ClearRtes ();
                }
            }
          if (hdr.GetRteNumber () > 0)
            {
              p->AddHeader (hdr);
              NS_LOG_DEBUG ("SendTo: " << *p);
              iter->first->SendTo (p, 0, InetSocketAddress (RIP_ALL_NODE, RIP_PORT));
            }
        }
    }
  for (RoutesI rtIter = m_routes.begin (); rtIter != m_routes.end (); rtIter++)
    {
      rtIter->first->SetRouteChanged (false);
    }
}

void Rip::SendTriggeredRouteUpdate ()
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
  m_nextTriggeredUpdate = Simulator::Schedule (delay, &Rip::DoSendRouteUpdate, this, false);
}

void Rip::SendUnsolicitedRouteUpdate ()
{
  NS_LOG_FUNCTION (this);

  if (m_nextTriggeredUpdate.IsRunning())
    {
      m_nextTriggeredUpdate.Cancel ();
    }

  DoSendRouteUpdate (true);

  Time delay = m_unsolicitedUpdate + Seconds (m_rng->GetValue (0, 0.5*m_unsolicitedUpdate.GetSeconds ()) );
  m_nextUnsolicitedUpdate = Simulator::Schedule (delay, &Rip::SendUnsolicitedRouteUpdate, this);
}

std::set<uint32_t> Rip::GetInterfaceExclusions () const
{
  return m_interfaceExclusions;
}

void Rip::SetInterfaceExclusions (std::set<uint32_t> exceptions)
{
  NS_LOG_FUNCTION (this);

  m_interfaceExclusions = exceptions;
}

uint8_t Rip::GetInterfaceMetric (uint32_t interface) const
{
  NS_LOG_FUNCTION (this << interface);

  std::map<uint32_t, uint8_t>::const_iterator iter = m_interfaceMetrics.find (interface);
  if (iter != m_interfaceMetrics.end ())
    {
      return iter->second;
    }
  return 1;
}

void Rip::SetInterfaceMetric (uint32_t interface, uint8_t metric)
{
  NS_LOG_FUNCTION (this << interface << int (metric));

  if (metric < m_linkDown)
    {
      m_interfaceMetrics[interface] = metric;
    }
}

void Rip::SendRouteRequest ()
{
  NS_LOG_FUNCTION (this);

  Ptr<Packet> p = Create<Packet> ();
  SocketIpTtlTag tag;
  p->RemovePacketTag (tag);
  tag.SetTtl (1);
  p->AddPacketTag (tag);

  RipHeader hdr;
  hdr.SetCommand (RipHeader::REQUEST);

  RipRte rte;
  rte.SetPrefix (Ipv4Address::GetAny ());
  rte.SetSubnetMask (Ipv4Mask::GetZero ());
  rte.SetRouteMetric (m_linkDown);

  hdr.AddRte (rte);
  p->AddHeader (hdr);

  for (SocketListI iter = m_unicastSocketList.begin (); iter != m_unicastSocketList.end (); iter++ )
    {
      uint32_t interface = iter->second;

      if (m_interfaceExclusions.find (interface) == m_interfaceExclusions.end ())
        {
          NS_LOG_DEBUG ("SendTo: " << *p);
          iter->first->SendTo (p, 0, InetSocketAddress (RIP_ALL_NODE, RIP_PORT));
        }
    }
}

void Rip::AddDefaultRouteTo (Ipv4Address nextHop, uint32_t interface)
{
  NS_LOG_FUNCTION (this << interface);

  AddNetworkRouteTo (Ipv4Address ("0.0.0.0"), Ipv4Mask::GetZero (), nextHop, interface);
}


/*
 * RipRoutingTableEntry
 */

RipRoutingTableEntry::RipRoutingTableEntry ()
  : m_tag (0), m_metric (0), m_status (RIP_INVALID), m_changed (false)
{
}

RipRoutingTableEntry::RipRoutingTableEntry (Ipv4Address network, Ipv4Mask networkPrefix, Ipv4Address nextHop, uint32_t interface)
  : Ipv4RoutingTableEntry ( Ipv4RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, nextHop, interface) ),
    m_tag (0), m_metric (0), m_status (RIP_INVALID), m_changed (false)
{
}

RipRoutingTableEntry::RipRoutingTableEntry (Ipv4Address network, Ipv4Mask networkPrefix, uint32_t interface)
  : Ipv4RoutingTableEntry ( Ipv4RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, interface) ),
    m_tag (0), m_metric (0), m_status (RIP_INVALID), m_changed (false)
{
}

RipRoutingTableEntry::~RipRoutingTableEntry ()
{
}


void RipRoutingTableEntry::SetRouteTag (uint16_t routeTag)
{
  if (m_tag != routeTag)
    {
      m_tag = routeTag;
      m_changed = true;
    }
}

uint16_t RipRoutingTableEntry::GetRouteTag () const
{
  return m_tag;
}

void RipRoutingTableEntry::SetRouteMetric (uint8_t routeMetric)
{
  if (m_metric != routeMetric)
    {
      m_metric = routeMetric;
      m_changed = true;
    }
}

uint8_t RipRoutingTableEntry::GetRouteMetric () const
{
  return m_metric;
}

void RipRoutingTableEntry::SetRouteStatus (Status_e status)
{
  if (m_status != status)
    {
      m_status = status;
      m_changed = true;
    }
}

RipRoutingTableEntry::Status_e RipRoutingTableEntry::GetRouteStatus (void) const
{
  return m_status;
}

void RipRoutingTableEntry::SetRouteChanged (bool changed)
{
  m_changed = changed;
}

bool RipRoutingTableEntry::IsRouteChanged (void) const
{
  return m_changed;
}


std::ostream & operator << (std::ostream& os, const RipRoutingTableEntry& rte)
{
  os << static_cast<const Ipv4RoutingTableEntry &>(rte);
  os << ", metric: " << int (rte.GetRouteMetric ()) << ", tag: " << int (rte.GetRouteTag ());

  return os;
}


}

