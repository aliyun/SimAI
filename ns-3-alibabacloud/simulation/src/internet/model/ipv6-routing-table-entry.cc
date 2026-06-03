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

#include "ns3/assert.h"

#include "ipv6-routing-table-entry.h"

namespace ns3
{

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry ()
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6RoutingTableEntry const& route)
  : m_dest (route.m_dest),
    m_destNetworkPrefix (route.m_destNetworkPrefix),
    m_gateway (route.m_gateway),
    m_interface (route.m_interface),
    m_prefixToUse (route.m_prefixToUse)
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6RoutingTableEntry const* route) 
  : m_dest (route->m_dest),
    m_destNetworkPrefix (route->m_destNetworkPrefix),
    m_gateway (route->m_gateway),
    m_interface (route->m_interface),
    m_prefixToUse (route->m_prefixToUse)
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address dest, Ipv6Address gateway, uint32_t interface)
  : m_dest (dest),
    m_destNetworkPrefix (Ipv6Prefix::GetZero ()),
    m_gateway (gateway),
    m_interface (interface),
    m_prefixToUse (Ipv6Address ("::"))
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address dest, uint32_t interface)
  : m_dest (dest),
    m_destNetworkPrefix (Ipv6Prefix::GetOnes ()),
    m_gateway (Ipv6Address::GetZero ()),
    m_interface (interface),
    m_prefixToUse (Ipv6Address ("::"))
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address gateway, uint32_t interface, Ipv6Address prefixToUse)
  : m_dest (network),
    m_destNetworkPrefix (networkPrefix),
    m_gateway (gateway),
    m_interface (interface),
    m_prefixToUse (prefixToUse)
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address gateway, uint32_t interface)
  : m_dest (network),
    m_destNetworkPrefix (networkPrefix),
    m_gateway (gateway),
    m_interface (interface),
    m_prefixToUse (Ipv6Address::GetZero ())
{
}


Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface, Ipv6Address prefixToUse)
  : m_dest (network),
    m_destNetworkPrefix (networkPrefix),
    m_gateway (Ipv6Address::GetZero ()),
    m_interface (interface),
    m_prefixToUse (prefixToUse)
{
}

Ipv6RoutingTableEntry::Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface)
  : m_dest (network),
    m_destNetworkPrefix (networkPrefix),
    m_gateway (Ipv6Address::GetZero ()),
    m_interface (interface),
    m_prefixToUse (Ipv6Address ("::"))
{
}

Ipv6RoutingTableEntry::~Ipv6RoutingTableEntry ()
{
}

bool Ipv6RoutingTableEntry::IsHost () const
{
  if (m_destNetworkPrefix == Ipv6Prefix::GetOnes ())
    {
      return true;
    }
  return false;
}

Ipv6Address Ipv6RoutingTableEntry::GetDest () const
{
  return m_dest;
}

Ipv6Address Ipv6RoutingTableEntry::GetPrefixToUse () const
{
  return m_prefixToUse;
}

void Ipv6RoutingTableEntry::SetPrefixToUse (Ipv6Address prefix)
{
  m_prefixToUse = prefix;
}

bool Ipv6RoutingTableEntry::IsNetwork () const
{
  return !IsHost ();
}

bool Ipv6RoutingTableEntry::IsDefault () const
{
  if (m_dest == Ipv6Address::GetZero ())
    {
      return true;
    }
  return false;
}

Ipv6Address Ipv6RoutingTableEntry::GetDestNetwork () const
{
  return m_dest;
}

Ipv6Prefix Ipv6RoutingTableEntry::GetDestNetworkPrefix () const
{
  return m_destNetworkPrefix;
}

bool Ipv6RoutingTableEntry::IsGateway () const
{
  if (m_gateway == Ipv6Address::GetZero ())
    {
      return false;
    }
  return true;
}

Ipv6Address Ipv6RoutingTableEntry::GetGateway () const
{
  return m_gateway;
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateHostRouteTo (Ipv6Address dest, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  return Ipv6RoutingTableEntry (dest, Ipv6Prefix::GetOnes (), nextHop, interface, prefixToUse);
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateHostRouteTo (Ipv6Address dest, uint32_t interface)
{
  return Ipv6RoutingTableEntry (dest, interface);
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface)
{
  return Ipv6RoutingTableEntry (network, networkPrefix, nextHop, interface);
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse)
{
  return Ipv6RoutingTableEntry (network, networkPrefix, nextHop, interface, prefixToUse);
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface)
{
  return Ipv6RoutingTableEntry (network, networkPrefix, interface, network);
}

Ipv6RoutingTableEntry Ipv6RoutingTableEntry::CreateDefaultRoute (Ipv6Address nextHop, uint32_t interface)
{
  return Ipv6RoutingTableEntry (Ipv6Address::GetZero (), nextHop, interface);
}

uint32_t Ipv6RoutingTableEntry::GetInterface () const
{
  return m_interface;
}

std::ostream& operator<< (std::ostream& os, Ipv6RoutingTableEntry const& route)
{
  if (route.IsDefault ())
    {
      NS_ASSERT (route.IsGateway ());
      os << "default out: " << route.GetInterface () << ", next hop: " << route.GetGateway ();
    }
  else if (route.IsHost ())
    {
      if (route.IsGateway ())
        {
          os << "host: "<< route.GetDest () <<
          ", out: " << route.GetInterface () <<
          ", next hop: " << route.GetGateway ();
        }
      else
        {
          os << "host: " << route.GetDest () <<
          ", out: " << route.GetInterface ();
        }
    }
  else if (route.IsNetwork ())
    {
      if (route.IsGateway ())
        {
          os << "network: " << route.GetDestNetwork () <<
          "/ " << int(route.GetDestNetworkPrefix ().GetPrefixLength ()) <<
          ", out: " << route.GetInterface () <<
          ", next hop: " << route.GetGateway ();
        }
      else
        {
          os << "network: " << route.GetDestNetwork () <<
          "/" << int(route.GetDestNetworkPrefix ().GetPrefixLength ()) <<
          ", out: " << route.GetInterface ();
        }
    }
  else
    {
      NS_ASSERT (false);
    }
  return os;
}

Ipv6MulticastRoutingTableEntry::Ipv6MulticastRoutingTableEntry ()
{
}

Ipv6MulticastRoutingTableEntry::Ipv6MulticastRoutingTableEntry (Ipv6MulticastRoutingTableEntry const & route) 
  : m_origin (route.m_origin),
    m_group (route.m_group),
    m_inputInterface (route.m_inputInterface),
    m_outputInterfaces (route.m_outputInterfaces)
{
}

Ipv6MulticastRoutingTableEntry::Ipv6MulticastRoutingTableEntry (Ipv6MulticastRoutingTableEntry const* route)
  : m_origin (route->m_origin),
    m_group (route->m_group),
    m_inputInterface (route->m_inputInterface),
    m_outputInterfaces (route->m_outputInterfaces)
{
}

Ipv6MulticastRoutingTableEntry::Ipv6MulticastRoutingTableEntry (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces)
  : m_origin (origin),
    m_group (group),
    m_inputInterface (inputInterface),
    m_outputInterfaces (outputInterfaces)
{
}

Ipv6Address Ipv6MulticastRoutingTableEntry::GetOrigin () const
{
  return m_origin;
}

Ipv6Address Ipv6MulticastRoutingTableEntry::GetGroup () const
{
  return m_group;
}

uint32_t Ipv6MulticastRoutingTableEntry::GetInputInterface () const
{
  return m_inputInterface;
}

uint32_t Ipv6MulticastRoutingTableEntry::GetNOutputInterfaces () const
{
  return m_outputInterfaces.size ();
}

uint32_t Ipv6MulticastRoutingTableEntry::GetOutputInterface (uint32_t n) const
{
  NS_ASSERT_MSG (n < m_outputInterfaces.size (), "Ipv6MulticastRoutingTableEntry::GetOutputInterface () : index out of bounds");

  return m_outputInterfaces[n];
}

std::vector <uint32_t> Ipv6MulticastRoutingTableEntry::GetOutputInterfaces () const
{
  return m_outputInterfaces;
}

Ipv6MulticastRoutingTableEntry Ipv6MulticastRoutingTableEntry::CreateMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces)
{
  return Ipv6MulticastRoutingTableEntry (origin, group, inputInterface, outputInterfaces);
}

std::ostream& operator<< (std::ostream& os, Ipv6MulticastRoutingTableEntry const& route)
{
  os << "origin: " << route.GetOrigin () <<
  ", group: " << route.GetGroup () <<
  ", input interface: " << route.GetInputInterface () <<
  ", output interfaces: ";

  for (uint32_t i = 0; i < route.GetNOutputInterfaces (); ++i)
    {
      os << route.GetOutputInterface (i) << " ";
    }

  return os;
}

} /* namespace ns3 */

