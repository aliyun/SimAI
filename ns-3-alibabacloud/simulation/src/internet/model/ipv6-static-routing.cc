/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include <iomanip>
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/ipv6-route.h"
#include "ns3/net-device.h"
#include "ns3/names.h"

#include "ipv6-static-routing.h"
#include "ipv6-routing-table-entry.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6StaticRouting");

NS_OBJECT_ENSURE_REGISTERED (Ipv6StaticRouting);

TypeId Ipv6StaticRouting::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6StaticRouting")
    .SetParent<Ipv6RoutingProtocol> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv6StaticRouting> ()
  ;
  return tid;
}

Ipv6StaticRouting::Ipv6StaticRouting ()
  : m_ipv6 (0)
{
  NS_LOG_FUNCTION (this);
}

Ipv6StaticRouting::~Ipv6StaticRouting ()
{
  NS_LOG_FUNCTION (this);
}

void Ipv6StaticRouting::SetIpv6 (Ptr<Ipv6> ipv6)
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

// Formatted like output of "route -n" command
void
Ipv6StaticRouting::PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
  NS_LOG_FUNCTION (this << stream);
  std::ostream* os = stream->GetStream ();
  // Copy the current ostream state
  std::ios oldState (nullptr);
  oldState.copyfmt (*os);

  *os << std::resetiosflags (std::ios::adjustfield) << std::setiosflags (std::ios::left);

  *os << "Node: " << m_ipv6->GetObject<Node> ()->GetId ()
      << ", Time: " << Now().As (unit)
      << ", Local time: " << m_ipv6->GetObject<Node> ()->GetLocalTime ().As (unit)
      << ", Ipv6StaticRouting table" << std::endl;

  if (GetNRoutes () > 0)
    {
      *os << "Destination                    Next Hop                   Flag Met Ref Use If" << std::endl;
      for (uint32_t j = 0; j < GetNRoutes (); j++)
        {
          std::ostringstream dest, gw, mask, flags;
          Ipv6RoutingTableEntry route = GetRoute (j);
          dest << route.GetDest () << "/" << int(route.GetDestNetworkPrefix ().GetPrefixLength ());
          *os << std::setw (31) << dest.str ();
          gw << route.GetGateway ();
          *os << std::setw (27) << gw.str ();
          flags << "U";
          if (route.IsHost ())
            {
              flags << "H";
            }
          else if (route.IsGateway ())
            {
              flags << "G";
            }
          *os << std::setw (5) << flags.str ();
          *os << std::setw (4) << GetMetric (j);
          // Ref ct not implemented
          *os << "-" << "   ";
          // Use not implemented
          *os << "-" << "   ";
          if (Names::FindName (m_ipv6->GetNetDevice (route.GetInterface ())) != "")
            {
              *os << Names::FindName (m_ipv6->GetNetDevice (route.GetInterface ()));
            }
          else
            {
              *os << route.GetInterface ();
            }
          *os << std::endl;
        }
    }
  *os << std::endl;
  // Restore the previous ostream state
  (*os).copyfmt (oldState);
}

void Ipv6StaticRouting::AddHostRouteTo (Ipv6Address dst, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse, uint32_t metric)
{
  NS_LOG_FUNCTION (this << dst << nextHop << interface << prefixToUse << metric);
  if (nextHop.IsLinkLocal())
    {
      NS_LOG_WARN ("Ipv6StaticRouting::AddHostRouteTo - Next hop should be link-local");
    }

  AddNetworkRouteTo (dst, Ipv6Prefix::GetOnes (), nextHop, interface, prefixToUse, metric);
}

void Ipv6StaticRouting::AddHostRouteTo (Ipv6Address dst, uint32_t interface, uint32_t metric)
{
  NS_LOG_FUNCTION (this << dst << interface << metric);
  AddNetworkRouteTo (dst, Ipv6Prefix::GetOnes (), interface, metric);
}

void Ipv6StaticRouting::AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, uint32_t metric)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << nextHop << interface << metric);

  Ipv6RoutingTableEntry route = Ipv6RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, nextHop, interface);

  if (!LookupRoute (route, metric))
    {
      Ipv6RoutingTableEntry* routePtr = new Ipv6RoutingTableEntry (route);
      m_networkRoutes.push_back (std::make_pair (routePtr, metric));
    }
}

void Ipv6StaticRouting::AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse, uint32_t metric)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << nextHop << interface << prefixToUse << metric);
  if (nextHop.IsLinkLocal())
    {
      NS_LOG_WARN ("Ipv6StaticRouting::AddNetworkRouteTo - Next hop should be link-local");
    }

  Ipv6RoutingTableEntry route = Ipv6RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, nextHop, interface, prefixToUse);
  if (!LookupRoute (route, metric))
    {
      Ipv6RoutingTableEntry* routePtr = new Ipv6RoutingTableEntry (route);
      m_networkRoutes.push_back (std::make_pair (routePtr, metric));
    }
}

void Ipv6StaticRouting::AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface, uint32_t metric)
{
  NS_LOG_FUNCTION (this << network << networkPrefix << interface);

  Ipv6RoutingTableEntry route = Ipv6RoutingTableEntry::CreateNetworkRouteTo (network, networkPrefix, interface);
  if (!LookupRoute (route, metric))
    {
      Ipv6RoutingTableEntry* routePtr = new Ipv6RoutingTableEntry (route);
      m_networkRoutes.push_back (std::make_pair (routePtr, metric));
    }
}

void Ipv6StaticRouting::SetDefaultRoute (Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse, uint32_t metric)
{
  NS_LOG_FUNCTION (this << nextHop << interface << prefixToUse);
  AddNetworkRouteTo (Ipv6Address ("::"), Ipv6Prefix::GetZero (), nextHop, interface, prefixToUse, metric);
}

void Ipv6StaticRouting::AddMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces)
{
  NS_LOG_FUNCTION (this << origin << group << inputInterface);
  Ipv6MulticastRoutingTableEntry* route = new Ipv6MulticastRoutingTableEntry ();
  *route = Ipv6MulticastRoutingTableEntry::CreateMulticastRoute (origin, group, inputInterface, outputInterfaces);
  m_multicastRoutes.push_back (route);
}

void Ipv6StaticRouting::SetDefaultMulticastRoute (uint32_t outputInterface)
{
  NS_LOG_FUNCTION (this << outputInterface);
  Ipv6RoutingTableEntry *route = new Ipv6RoutingTableEntry ();
  Ipv6Address network = Ipv6Address ("ff00::"); /* RFC 3513 */
  Ipv6Prefix networkMask = Ipv6Prefix (8);
  *route = Ipv6RoutingTableEntry::CreateNetworkRouteTo (network, networkMask, outputInterface);
  m_networkRoutes.push_back (std::make_pair (route, 0));
}

uint32_t Ipv6StaticRouting::GetNMulticastRoutes () const
{
  NS_LOG_FUNCTION (this);
  return m_multicastRoutes.size ();
}

Ipv6MulticastRoutingTableEntry Ipv6StaticRouting::GetMulticastRoute (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  NS_ASSERT_MSG (index < m_multicastRoutes.size (), "Ipv6StaticRouting::GetMulticastRoute () : Index out of range");

  if (index < m_multicastRoutes.size ())
    {
      uint32_t tmp = 0;
      for (MulticastRoutesCI i = m_multicastRoutes.begin (); i != m_multicastRoutes.end (); i++)
        {
          if (tmp  == index)
            {
              return *i;
            }
          tmp++;
        }
    }
  return 0;
}

bool Ipv6StaticRouting::RemoveMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface)
{
  NS_LOG_FUNCTION (this << origin << group << inputInterface);
  for (MulticastRoutesI i = m_multicastRoutes.begin (); i != m_multicastRoutes.end (); i++)
    {
      Ipv6MulticastRoutingTableEntry *route = *i;
      if (origin == route->GetOrigin ()
          && group == route->GetGroup ()
          && inputInterface == route->GetInputInterface ())
        {
          delete *i;
          m_multicastRoutes.erase (i);
          return true;
        }
    }
  return false;
}

void Ipv6StaticRouting::RemoveMulticastRoute (uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  uint32_t tmp = 0;

  for (MulticastRoutesI i = m_multicastRoutes.begin (); i != m_multicastRoutes.end (); i++)
    {
      if (tmp == index)
        {
          delete *i;
          m_multicastRoutes.erase (i);
          return;
        }
      tmp++;
    }
}

bool Ipv6StaticRouting::HasNetworkDest (Ipv6Address network, uint32_t interfaceIndex)
{
  NS_LOG_FUNCTION (this << network << interfaceIndex);

  /* in the network table */
  for (NetworkRoutesI j = m_networkRoutes.begin (); j != m_networkRoutes.end (); j++)
    {
      Ipv6RoutingTableEntry* rtentry = j->first;
      Ipv6Prefix prefix = rtentry->GetDestNetworkPrefix ();
      Ipv6Address entry = rtentry->GetDestNetwork ();

      if (prefix.IsMatch (network, entry) && rtentry->GetInterface () == interfaceIndex)
        {
          return true;
        }
    }

  /* beuh!!! not route at all */
  return false;
}

bool Ipv6StaticRouting::LookupRoute (const Ipv6RoutingTableEntry &route, uint32_t metric)
{
  for (NetworkRoutesI j = m_networkRoutes.begin (); j != m_networkRoutes.end (); j++)
    {
      Ipv6RoutingTableEntry* rtentry = j->first;

      if (rtentry->GetDest () == route.GetDest () &&
          rtentry->GetDestNetworkPrefix () == route.GetDestNetworkPrefix () &&
          rtentry->GetGateway () == route.GetGateway () &&
          rtentry->GetInterface () == route.GetInterface () &&
          rtentry->GetPrefixToUse () == route.GetPrefixToUse () &&
          j->second == metric)
        {
          return true;
        }
    }
  return false;
}

Ptr<Ipv6Route> Ipv6StaticRouting::LookupStatic (Ipv6Address dst, Ptr<NetDevice> interface)
{
  NS_LOG_FUNCTION (this << dst << interface);
  Ptr<Ipv6Route> rtentry = 0;
  uint16_t longestMask = 0;
  uint32_t shortestMetric = 0xffffffff;

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

  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      Ipv6RoutingTableEntry* j = it->first;
      uint32_t metric = it->second;
      Ipv6Prefix mask = j->GetDestNetworkPrefix ();
      uint16_t maskLen = mask.GetPrefixLength ();
      Ipv6Address entry = j->GetDestNetwork ();

      NS_LOG_LOGIC ("Searching for route to " << dst << ", mask length " << maskLen << ", metric " << metric);

      if (mask.IsMatch (dst, entry))
        {
          NS_LOG_LOGIC ("Found global network route " << *j << ", mask length " << maskLen << ", metric " << metric);

          /* if interface is given, check the route will output on this interface */
          if (!interface || interface == m_ipv6->GetNetDevice (j->GetInterface ()))
            {
              if (maskLen < longestMask)
                {
                  NS_LOG_LOGIC ("Previous match longer, skipping");
                  continue;
                }

              if (maskLen > longestMask)
                {
                  shortestMetric = 0xffffffff;
                }

              longestMask = maskLen;
              if (metric > shortestMetric)
                {
                  NS_LOG_LOGIC ("Equal mask length, but previous metric shorter, skipping");
                  continue;
                }

              shortestMetric = metric;
              Ipv6RoutingTableEntry* route = j;
              uint32_t interfaceIdx = route->GetInterface ();
              rtentry = Create<Ipv6Route> ();

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
                  rtentry->SetSource (m_ipv6->SourceAddressSelection (interfaceIdx, route->GetGateway ()));
                }

              rtentry->SetDestination (route->GetDest ());
              rtentry->SetGateway (route->GetGateway ());
              rtentry->SetOutputDevice (m_ipv6->GetNetDevice (interfaceIdx));
              if (maskLen == 128)
                {
                  break;
                }
            }
        }
    }

  if (rtentry)
    {
      NS_LOG_LOGIC ("Matching route via " << rtentry->GetDestination () << " (Through " << rtentry->GetGateway () << ") at the end");
    }
  return rtentry;
}

void Ipv6StaticRouting::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  for (NetworkRoutesI j = m_networkRoutes.begin ();  j != m_networkRoutes.end (); j = m_networkRoutes.erase (j))
    {
      delete j->first;
    }
  m_networkRoutes.clear ();

  for (MulticastRoutesI i = m_multicastRoutes.begin (); i != m_multicastRoutes.end (); i = m_multicastRoutes.erase (i))
    {
      delete (*i);
    }
  m_multicastRoutes.clear ();

  m_ipv6 = 0;
  Ipv6RoutingProtocol::DoDispose ();
}

Ptr<Ipv6MulticastRoute> Ipv6StaticRouting::LookupStatic (Ipv6Address origin, Ipv6Address group, uint32_t interface)
{
  NS_LOG_FUNCTION (this << origin << group << interface);
  Ptr<Ipv6MulticastRoute> mrtentry = 0;

  for (MulticastRoutesI i = m_multicastRoutes.begin (); i != m_multicastRoutes.end (); i++)
    {
      Ipv6MulticastRoutingTableEntry* route = *i;

      /*
         We've been passed an origin address, a multicast group address and an
         interface index.  We have to decide if the current route in the list is
         a match.

         The first case is the restrictive case where the origin, group and index
         matches.  This picks up exact routes during forwarded and exact routes from
         the local node (in which case the ifIndex is a wildcard).
         */

      if (origin == route->GetOrigin () && group == route->GetGroup ())
        {
          /* skipping SSM case */
          NS_LOG_LOGIC ("Find source specific multicast route" << *i);
        }

      if (group == route->GetGroup ())
        {
          if (interface == Ipv6::IF_ANY || interface == route->GetInputInterface ())
            {
              NS_LOG_LOGIC ("Found multicast route" << *i);
              mrtentry = Create<Ipv6MulticastRoute> ();
              mrtentry->SetGroup (route->GetGroup ());
              mrtentry->SetOrigin (route->GetOrigin ());
              mrtentry->SetParent (route->GetInputInterface ());
              for (uint32_t j = 0; j < route->GetNOutputInterfaces (); j++)
                {
                  if (route->GetOutputInterface (j))
                    {
                      NS_LOG_LOGIC ("Setting output interface index " << route->GetOutputInterface (j));
                      mrtentry->SetOutputTtl (route->GetOutputInterface (j), Ipv6MulticastRoute::MAX_TTL - 1);
                    }
                }
              return mrtentry;
            }
        }
    }
  return mrtentry;
}

uint32_t Ipv6StaticRouting::GetNRoutes () const
{
  return m_networkRoutes.size ();
}

Ipv6RoutingTableEntry Ipv6StaticRouting::GetDefaultRoute ()
{
  NS_LOG_FUNCTION (this);
  Ipv6Address dst ("::");
  uint32_t shortestMetric = 0xffffffff;
  Ipv6RoutingTableEntry* result = 0;

  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      Ipv6RoutingTableEntry* j = it->first;
      uint32_t metric = it->second;
      Ipv6Prefix mask = j->GetDestNetworkPrefix ();
      uint16_t maskLen = mask.GetPrefixLength ();
      Ipv6Address entry = j->GetDestNetwork ();

      if (maskLen)
        {
          continue;
        }

      if (metric > shortestMetric)
        {
          continue;
        }
      shortestMetric = metric;
      result = j;
    }

  if (result)
    {
      return result;
    }
  else
    {
      return Ipv6RoutingTableEntry ();
    }
}

Ipv6RoutingTableEntry Ipv6StaticRouting::GetRoute (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  uint32_t tmp = 0;

  for (NetworkRoutesCI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      if (tmp == index)
        {
          return it->first;
        }
      tmp++;
    }
  NS_ASSERT (false);
  // quiet compiler.
  return 0;
}

uint32_t Ipv6StaticRouting::GetMetric (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);
  uint32_t tmp = 0;

  for (NetworkRoutesCI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      if (tmp == index)
        {
          return it->second;
        }
      tmp++;
    }
  NS_ASSERT (false);
  // quiet compiler.
  return 0;
}

void Ipv6StaticRouting::RemoveRoute (uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  uint32_t tmp = 0;

  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      if (tmp == index)
        {
          delete it->first;
          m_networkRoutes.erase (it);
          return;
        }
      tmp++;
    }
  NS_ASSERT (false);
}

void Ipv6StaticRouting::RemoveRoute (Ipv6Address network, Ipv6Prefix prefix, uint32_t ifIndex, Ipv6Address prefixToUse)
{
  NS_LOG_FUNCTION (this << network << prefix << ifIndex);

  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); it++)
    {
      Ipv6RoutingTableEntry* rtentry = it->first;
      if (network == rtentry->GetDest () && rtentry->GetInterface () == ifIndex
          && rtentry->GetPrefixToUse () == prefixToUse)
        {
          delete it->first;
          m_networkRoutes.erase (it);
          return;
        }
    }
}

Ptr<Ipv6Route> Ipv6StaticRouting::RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)
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
      NS_LOG_LOGIC ("RouteOutput ()::Multicast destination");
    }

  rtentry = LookupStatic (destination, oif);
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

bool Ipv6StaticRouting::RouteInput (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev,
                                    UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                                    LocalDeliverCallback lcb, ErrorCallback ecb)
{
  NS_LOG_FUNCTION (this << p << header << header.GetSource () << header.GetDestination () << idev);
  NS_ASSERT (m_ipv6 != 0);
  // Check if input device supports IP
  NS_ASSERT (m_ipv6->GetInterfaceForDevice (idev) >= 0);
  uint32_t iif = m_ipv6->GetInterfaceForDevice (idev);
  Ipv6Address dst = header.GetDestination ();

  // Multicast recognition; handle local delivery here
  if (dst.IsMulticast ())
    {
      NS_LOG_LOGIC ("Multicast destination");
      Ptr<Ipv6MulticastRoute> mrtentry = LookupStatic (header.GetSource (),
                                                       header.GetDestination (), m_ipv6->GetInterfaceForDevice (idev));

      // \todo check if we want to forward up the packet
      if (mrtentry)
        {
          NS_LOG_LOGIC ("Multicast route found");
          mcb (idev, mrtentry, p, header); // multicast forwarding callback
          return true;
        }
      else
        {
          NS_LOG_LOGIC ("Multicast route not found");
          return false; // Let other routing protocols try to handle this
        }
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
  Ptr<Ipv6Route> rtentry = LookupStatic (header.GetDestination ());

  if (rtentry != 0)
    {
      NS_LOG_LOGIC ("Found unicast destination- calling unicast callback");
      ucb (idev, rtentry, p, header);  // unicast forwarding callback
      return true;
    }
  else
    {
      NS_LOG_LOGIC ("Did not find unicast destination- returning false");
      return false; // Let other routing protocols try to handle this
    }
}

void Ipv6StaticRouting::NotifyInterfaceUp (uint32_t i)
{
  for (uint32_t j = 0; j < m_ipv6->GetNAddresses (i); j++)
    {
      Ipv6InterfaceAddress addr = m_ipv6->GetAddress (i, j);

      if (addr.GetAddress () != Ipv6Address ()
          && addr.GetPrefix () != Ipv6Prefix ())
        {
          if (addr.GetPrefix () == Ipv6Prefix (128))
            {
              /* host route */
              AddHostRouteTo (addr.GetAddress (), i);
            }
          else
            {
              if (addr.GetOnLink ())
                {
                  AddNetworkRouteTo (addr.GetAddress ().CombinePrefix (addr.GetPrefix ()),
                                     addr.GetPrefix (), i);
                }
            }
        }
    }
}

void Ipv6StaticRouting::NotifyInterfaceDown (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);

  /* remove all static routes that are going through this interface */
  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); )
    {
      if (it->first->GetInterface () == i)
        {
          delete it->first;
          it = m_networkRoutes.erase (it);
        }
      else
        {
          it++;
        }
    }
}

void Ipv6StaticRouting::NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address)
{
  if (!m_ipv6->IsUp (interface))
    {
      return;
    }
}

void Ipv6StaticRouting::NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address)
{
  if (!m_ipv6->IsUp (interface))
    {
      return;
    }

  Ipv6Address networkAddress = address.GetAddress ().CombinePrefix (address.GetPrefix ());
  Ipv6Prefix networkMask = address.GetPrefix ();

  // Remove all static routes that are going through this interface
  // which reference this network
  for (NetworkRoutesI it = m_networkRoutes.begin (); it != m_networkRoutes.end (); )
    {
      if (it->first->GetInterface () == interface
          && it->first->IsNetwork ()
          && it->first->GetDestNetwork () == networkAddress
          && it->first->GetDestNetworkPrefix () == networkMask)
        {
          delete it->first;
          it = m_networkRoutes.erase (it);
        }
      else
        {
          it++;
        }
    }
}

void Ipv6StaticRouting::NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  NS_LOG_INFO (this << dst << mask << nextHop << interface << prefixToUse);
  if (nextHop == Ipv6Address::GetZero ())
    {
      AddNetworkRouteTo (dst, mask, interface);
    }
  else if (dst != Ipv6Address::GetZero ())
    {
      AddNetworkRouteTo (dst, mask, nextHop, interface);
    }
  else /* default route */
    {
      /* this case is mainly used by configuring default route following RA processing,
       * in case of multiple prefix in RA, the first will configured default route
       */

      /* for the moment, all default route has the same metric
       * so according to the longest prefix algorithm,
       * the default route chosen will be the last added
       */
      SetDefaultRoute (nextHop, interface, prefixToUse);
    }
}

void Ipv6StaticRouting::NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  NS_LOG_FUNCTION (this << dst << mask << nextHop << interface);
  if (dst != Ipv6Address::GetZero ())
    {
      for (NetworkRoutesI j = m_networkRoutes.begin (); j != m_networkRoutes.end ();)
        {
          Ipv6RoutingTableEntry* rtentry = j->first;
          Ipv6Prefix prefix = rtentry->GetDestNetworkPrefix ();
          Ipv6Address entry = rtentry->GetDestNetwork ();

          if (dst == entry && prefix == mask && rtentry->GetInterface () == interface)
            {
              delete j->first;
              j = m_networkRoutes.erase (j);
            }
          else
            {
              ++j;
            }
        }
    }
  else
    {
      /* default route case */
      RemoveRoute (dst, mask, interface, prefixToUse);
    }
}

} /* namespace ns3 */

