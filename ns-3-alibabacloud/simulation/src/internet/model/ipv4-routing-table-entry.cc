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

#include "ipv4-routing-table-entry.h"
#include "ns3/assert.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4RoutingTableEntry");

/*****************************************************
 *     Network Ipv4RoutingTableEntry
 *****************************************************/

Ipv4RoutingTableEntry::Ipv4RoutingTableEntry ()
{
  NS_LOG_FUNCTION (this);
}

Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4RoutingTableEntry const &route)
  : m_dest (route.m_dest),
    m_destNetworkMask (route.m_destNetworkMask),
    m_gateway (route.m_gateway),
    m_interface (route.m_interface)
{
  NS_LOG_FUNCTION (this << route);
}

Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4RoutingTableEntry const *route)
  : m_dest (route->m_dest),
    m_destNetworkMask (route->m_destNetworkMask),
    m_gateway (route->m_gateway),
    m_interface (route->m_interface)
{
  NS_LOG_FUNCTION (this << route);
}

Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4Address dest,
                                              Ipv4Address gateway,
                                              uint32_t interface)
  : m_dest (dest),
    m_destNetworkMask (Ipv4Mask::GetOnes ()),
    m_gateway (gateway),
    m_interface (interface)
{
}
Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4Address dest,
                                              uint32_t interface)
  : m_dest (dest),
    m_destNetworkMask (Ipv4Mask::GetOnes ()),
    m_gateway (Ipv4Address::GetZero ()),
    m_interface (interface)
{
}
Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4Address network,
                                              Ipv4Mask networkMask,
                                              Ipv4Address gateway,
                                              uint32_t interface)
  : m_dest (network),
    m_destNetworkMask (networkMask),
    m_gateway (gateway),
    m_interface (interface)
{
  NS_LOG_FUNCTION (this << network << networkMask << gateway << interface);
}
Ipv4RoutingTableEntry::Ipv4RoutingTableEntry (Ipv4Address network,
                                              Ipv4Mask networkMask,
                                              uint32_t interface)
  : m_dest (network),
    m_destNetworkMask (networkMask),
    m_gateway (Ipv4Address::GetZero ()),
    m_interface (interface)
{
  NS_LOG_FUNCTION (this << network << networkMask << interface);
}

bool
Ipv4RoutingTableEntry::IsHost (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_destNetworkMask == Ipv4Mask::GetOnes ())
    {
      return true;
    }
  else
    {
      return false;
    }
}
Ipv4Address
Ipv4RoutingTableEntry::GetDest (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dest;
}
bool
Ipv4RoutingTableEntry::IsNetwork (void) const
{
  NS_LOG_FUNCTION (this);
  return !IsHost ();
}
bool
Ipv4RoutingTableEntry::IsDefault (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_dest == Ipv4Address::GetZero ())
    {
      return true;
    }
  else
    {
      return false;
    }
}
Ipv4Address
Ipv4RoutingTableEntry::GetDestNetwork (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dest;
}
Ipv4Mask
Ipv4RoutingTableEntry::GetDestNetworkMask (void) const
{
  NS_LOG_FUNCTION (this);
  return m_destNetworkMask;
}
bool
Ipv4RoutingTableEntry::IsGateway (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_gateway == Ipv4Address::GetZero ())
    {
      return false;
    }
  else
    {
      return true;
    }
}
Ipv4Address
Ipv4RoutingTableEntry::GetGateway (void) const
{
  NS_LOG_FUNCTION (this);
  return m_gateway;
}
uint32_t
Ipv4RoutingTableEntry::GetInterface (void) const
{
  NS_LOG_FUNCTION (this);
  return m_interface;
}

Ipv4RoutingTableEntry 
Ipv4RoutingTableEntry::CreateHostRouteTo (Ipv4Address dest, 
                                          Ipv4Address nextHop,
                                          uint32_t interface)
{
  NS_LOG_FUNCTION (dest << nextHop << interface);
  return Ipv4RoutingTableEntry (dest, nextHop, interface);
}
Ipv4RoutingTableEntry 
Ipv4RoutingTableEntry::CreateHostRouteTo (Ipv4Address dest,
                                          uint32_t interface)
{
  NS_LOG_FUNCTION (dest << interface);
  return Ipv4RoutingTableEntry (dest, interface);
}
Ipv4RoutingTableEntry 
Ipv4RoutingTableEntry::CreateNetworkRouteTo (Ipv4Address network, 
                                             Ipv4Mask networkMask,
                                             Ipv4Address nextHop,
                                             uint32_t interface)
{
  NS_LOG_FUNCTION (network << networkMask << nextHop << interface);
  return Ipv4RoutingTableEntry (network, networkMask, 
                                nextHop, interface);
}
Ipv4RoutingTableEntry 
Ipv4RoutingTableEntry::CreateNetworkRouteTo (Ipv4Address network, 
                                             Ipv4Mask networkMask,
                                             uint32_t interface)
{
  NS_LOG_FUNCTION (network << networkMask << interface);
  return Ipv4RoutingTableEntry (network, networkMask, 
                                interface);
}
Ipv4RoutingTableEntry 
Ipv4RoutingTableEntry::CreateDefaultRoute (Ipv4Address nextHop, 
                                           uint32_t interface)
{
  NS_LOG_FUNCTION (nextHop << interface);
  return Ipv4RoutingTableEntry (Ipv4Address::GetZero (), Ipv4Mask::GetZero (), nextHop, interface);
}


std::ostream& operator<< (std::ostream& os, Ipv4RoutingTableEntry const& route)
{
  if (route.IsDefault ())
    {
      NS_ASSERT (route.IsGateway ());
      os << "default out=" << route.GetInterface () << ", next hop=" << route.GetGateway ();
    }
  else if (route.IsHost ())
    {
      if (route.IsGateway ())
        {
          os << "host="<< route.GetDest () << 
          ", out=" << route.GetInterface () <<
          ", next hop=" << route.GetGateway ();
        }
      else
        {
          os << "host="<< route.GetDest () << 
          ", out=" << route.GetInterface ();
        }
    }
  else if (route.IsNetwork ()) 
    {
      if (route.IsGateway ())
        {
          os << "network=" << route.GetDestNetwork () <<
          ", mask=" << route.GetDestNetworkMask () <<
          ",out=" << route.GetInterface () <<
          ", next hop=" << route.GetGateway ();
        }
      else
        {
          os << "network=" << route.GetDestNetwork () <<
          ", mask=" << route.GetDestNetworkMask () <<
          ",out=" << route.GetInterface ();
        }
    }
  else
    {
      NS_ASSERT (false);
    }
  return os;
}

bool operator== (const Ipv4RoutingTableEntry a, const Ipv4RoutingTableEntry b)
{
  return (a.GetDest () == b.GetDest () && 
          a.GetDestNetworkMask () == b.GetDestNetworkMask () &&
          a.GetGateway () == b.GetGateway () &&
          a.GetInterface () == b.GetInterface ());
}

/*****************************************************
 *     Ipv4MulticastRoutingTableEntry
 *****************************************************/

Ipv4MulticastRoutingTableEntry::Ipv4MulticastRoutingTableEntry ()
{
  NS_LOG_FUNCTION (this);
}

Ipv4MulticastRoutingTableEntry::Ipv4MulticastRoutingTableEntry (Ipv4MulticastRoutingTableEntry const &route)
  :
    m_origin (route.m_origin),
    m_group (route.m_group),
    m_inputInterface (route.m_inputInterface),
    m_outputInterfaces (route.m_outputInterfaces)
{
  NS_LOG_FUNCTION (this << route);
}

Ipv4MulticastRoutingTableEntry::Ipv4MulticastRoutingTableEntry (Ipv4MulticastRoutingTableEntry const *route)
  :
    m_origin (route->m_origin),
    m_group (route->m_group),
    m_inputInterface (route->m_inputInterface),
    m_outputInterfaces (route->m_outputInterfaces)
{
  NS_LOG_FUNCTION (this << route);
}

Ipv4MulticastRoutingTableEntry::Ipv4MulticastRoutingTableEntry (
  Ipv4Address origin, 
  Ipv4Address group, 
  uint32_t inputInterface, 
  std::vector<uint32_t> outputInterfaces)
{
  NS_LOG_FUNCTION (this << origin << group << inputInterface << &outputInterfaces);
  m_origin = origin;
  m_group = group;
  m_inputInterface = inputInterface;
  m_outputInterfaces = outputInterfaces;
}

Ipv4Address 
Ipv4MulticastRoutingTableEntry::GetOrigin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_origin;
}

Ipv4Address 
Ipv4MulticastRoutingTableEntry::GetGroup (void) const
{
  NS_LOG_FUNCTION (this);
  return m_group;
}

uint32_t 
Ipv4MulticastRoutingTableEntry::GetInputInterface (void) const
{
  NS_LOG_FUNCTION (this);
  return m_inputInterface;
}

uint32_t
Ipv4MulticastRoutingTableEntry::GetNOutputInterfaces (void) const
{
  NS_LOG_FUNCTION (this);
  return m_outputInterfaces.size ();
}

uint32_t
Ipv4MulticastRoutingTableEntry::GetOutputInterface (uint32_t n) const
{
  NS_LOG_FUNCTION (this << n);
  NS_ASSERT_MSG (n < m_outputInterfaces.size (),
                 "Ipv4MulticastRoutingTableEntry::GetOutputInterface (): index out of bounds");

  return m_outputInterfaces[n];
}

std::vector<uint32_t>
Ipv4MulticastRoutingTableEntry::GetOutputInterfaces (void) const
{
  NS_LOG_FUNCTION (this);
  return m_outputInterfaces;
}

Ipv4MulticastRoutingTableEntry 
Ipv4MulticastRoutingTableEntry::CreateMulticastRoute (Ipv4Address origin,
                                                      Ipv4Address group,
                                                      uint32_t inputInterface,
                                                      std::vector<uint32_t> outputInterfaces)
{
  NS_LOG_FUNCTION (origin << group << inputInterface << outputInterfaces);
  return Ipv4MulticastRoutingTableEntry (origin, group, inputInterface, outputInterfaces);
}

std::ostream& 
operator<< (std::ostream& os, Ipv4MulticastRoutingTableEntry const& route)
{
  os << "origin=" << route.GetOrigin () << 
  ", group=" << route.GetGroup () <<
  ", input interface=" << route.GetInputInterface () <<
  ", output interfaces=";

  for (uint32_t i = 0; i < route.GetNOutputInterfaces (); ++i)
    {
      os << route.GetOutputInterface (i) << " ";

    }

  return os;
}

bool operator== (const Ipv4MulticastRoutingTableEntry a, const Ipv4MulticastRoutingTableEntry b)
{
  return (a.GetOrigin () == b.GetOrigin () && 
          a.GetGroup () == b.GetGroup () &&
          a.GetInputInterface () == b.GetInputInterface () &&
          a.GetOutputInterfaces () == b.GetOutputInterfaces ());
}

} // namespace ns3
