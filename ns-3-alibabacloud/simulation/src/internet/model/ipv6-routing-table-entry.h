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

#ifndef IPV6_ROUTING_TABLE_ENTRY_H
#define IPV6_ROUTING_TABLE_ENTRY_H

#include <list>
#include <vector>
#include <ostream>

#include "ns3/ipv6-address.h"

namespace ns3
{

/**
 * \ingroup ipv6Routing
 *
 * \brief A record of an IPv6 route.
 */
class Ipv6RoutingTableEntry 
{
public:
  /**
   * \brief Constructor.
   */
  Ipv6RoutingTableEntry ();

  /**
   * \brief Copy constructor.
   * \param route the route to copy
   */
  Ipv6RoutingTableEntry (Ipv6RoutingTableEntry const & route);

  /**
   * \brief Copy constructor.
   * \param route the route to copy
   */
  Ipv6RoutingTableEntry (Ipv6RoutingTableEntry const* route);

  /**
   * \brief Destructor
   */
  virtual ~Ipv6RoutingTableEntry ();

  /**
   * \brief Is the route entry correspond to a host ?
   * \return true if the route is a host, false otherwise
   */
  bool IsHost () const;

  /**
   * \brief Get the destination.
   * \return the IPv6 address of the destination of this route
   */
  Ipv6Address GetDest () const;

  /**
   * \brief Get the prefix to use (for multihomed link).
   * \return prefix address to use
   */
  Ipv6Address GetPrefixToUse () const;

  /**
   * \brief Set the prefix to use.
   * \param prefix prefix to use
   */
  void SetPrefixToUse (Ipv6Address prefix);

  /**
   * \brief Is the route entry correspond to a network ? 
   * \return true if the route is a network, false otherwise
   */
  bool IsNetwork () const;

  /**
   * \brief Get the destination network.
   * \return the destination network
   */
  Ipv6Address GetDestNetwork () const;

  /**
   * \brief Get the destination prefix.
   * \return the destination prefix
   */
  Ipv6Prefix GetDestNetworkPrefix () const;

  /**
   * \brief Is it the default route ?
   * \return true if this route is a default route, false otherwise
   */
  bool IsDefault () const;

  /**
   * \brief Is it the gateway ? 
   * \return true if this route is a gateway, false otherwise
   */
  bool IsGateway () const;

  /**
   * \brief Get the gateway.
   * \return the IPv6 address of the gateway
   */
  Ipv6Address GetGateway () const;

  /**
   * \brief Get the interface index.
   * \return the index of the interface
   */
  uint32_t GetInterface () const;

  /**
   * \brief Create a route to a host.
   * \param dest destination address
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \param prefixToUse prefix that should be used for source address for this destination
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateHostRouteTo (Ipv6Address dest, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse=Ipv6Address ());

  /**
   * \brief Create a route to a host.
   * \param dest destination address
   * \param interface interface index
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateHostRouteTo (Ipv6Address dest, uint32_t interface);

  /**
   * \brief Create a route to a network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface);

  /**
   * \brief Create a route to a network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \param prefixToUse prefix that should be used for source address for this destination
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse);

  /**
   * \brief Create a route to a network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param interface interface index
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface);

  /**
   * \brief Create a default route.
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \return IPv6Route object
   */
  static Ipv6RoutingTableEntry CreateDefaultRoute (Ipv6Address nextHop, uint32_t interface);

private:
  /**
   * \brief Constructor.
   * \param network network address
   * \param prefix network prefix
   * \param gateway the gateway
   * \param interface the interface index
   */
  Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix prefix, Ipv6Address gateway, uint32_t interface);

  /**
   * \brief Constructor.
   * \param network network address
   * \param prefix network prefix
   * \param interface the interface index
   * \param prefixToUse prefix to use
   */
  Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix prefix, uint32_t interface, Ipv6Address prefixToUse);

  /**
   * \brief Constructor.
   * \param network network address
   * \param prefix network prefix
   * \param gateway the gateway
   * \param interface the interface index
   * \param prefixToUse prefix to use
   */
  Ipv6RoutingTableEntry (Ipv6Address network, Ipv6Prefix prefix, Ipv6Address gateway, uint32_t interface, Ipv6Address prefixToUse);

  /**
   * \brief Constructor.
   * \param dest destination address
   * \param prefix destination prefix
   * \param interface interface index
   */
  Ipv6RoutingTableEntry (Ipv6Address dest, Ipv6Prefix prefix, uint32_t interface);

  /**
   * \brief Constructor.
   * \param dest destination address
   * \param gateway the gateway
   * \param interface interface index
   */
  Ipv6RoutingTableEntry (Ipv6Address dest, Ipv6Address gateway, uint32_t interface);

  /**
   * \brief Constructor.
   * \param dest destination address
   * \param interface interface index
   */
  Ipv6RoutingTableEntry (Ipv6Address dest, uint32_t interface);

  /**
   * \brief IPv6 address of the destination.
   */
  Ipv6Address m_dest;

  /**
   * \brief IPv6 prefix of the destination
   */
  Ipv6Prefix m_destNetworkPrefix;

  /**
   * \brief IPv6 address of the gateway.
   */
  Ipv6Address m_gateway;

  /**
   * \brief The interface index.
   */
  uint32_t m_interface;

  /**
   * \brief Prefix to use.
   */
  Ipv6Address m_prefixToUse;

};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param route the Ipv6 routing table entry
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, Ipv6RoutingTableEntry const& route);

/**
 * \ingroup ipv6Routing
 *
 * \brief A record of an IPv6 multicast route.
 */
class Ipv6MulticastRoutingTableEntry
{
public:
  /**
   * \brief Constructor.
   */
  Ipv6MulticastRoutingTableEntry ();

  /**
   * \brief Copy constructor.
   * \param route the route to copy
   */
  Ipv6MulticastRoutingTableEntry (Ipv6MulticastRoutingTableEntry const & route);

  /**
   * \brief Copy constructor.
   * \param route the route to copy
   */
  Ipv6MulticastRoutingTableEntry (Ipv6MulticastRoutingTableEntry const* route);

  /**
   * \brief Get the source of this route
   * \return IPv6 address of the source of this route
   */
  Ipv6Address GetOrigin () const;

  /**
   * \brief Get the group.
   * \return IPv6 address of the multicast group of this route
   */
  Ipv6Address GetGroup () const;

  /**
   * \brief Get the input interface address.
   * \return input interface index
   */
  uint32_t GetInputInterface () const;

  /**
   * \brief Get the number of output interfaces of this route.
   * \return number of output interfaces of this route.
   */
  uint32_t GetNOutputInterfaces () const;

  /**
   * \brief Get a specified output interface.
   * \param n index
   * \return a specified output interface
   */
  uint32_t GetOutputInterface (uint32_t n) const;

  /**
   * \brief Get all of the output interfaces of this route.
   * \return a vector of all output interfaces of this route
   */
  std::vector<uint32_t> GetOutputInterfaces () const;

  /**
   * \brief Create a multicast route.
   * \param origin IPv6 address of the origin source
   * \param group Ipv6Address of the group
   * \param inputInterface interface number
   * \param outputInterfaces list of output interface number
   * \return a multicast route
   */
  static Ipv6MulticastRoutingTableEntry CreateMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces);

private:
  /**
   * \brief Constructor.
   * \param origin IPv6 address of the source
   * \param group IPv6 address of the group
   * \param inputInterface interface number
   * \param outputInterfaces list of output interface number
   */
  Ipv6MulticastRoutingTableEntry (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces);

  /**
   * \brief The IPv6 address of the source.
   */
  Ipv6Address m_origin;

  /**
   * \brief The IPv6 address of the group.
   */
  Ipv6Address m_group;

  /**
   * \brief The input interface.
   */
  uint32_t m_inputInterface;

  /**
   * \brief The output interfaces.
   */
  std::vector<uint32_t> m_outputInterfaces;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param route the Ipv6 multicast routing table entry
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, Ipv6MulticastRoutingTableEntry const& route);

} /* namespace ns3 */

#endif /* IPV6_ROUTING_TABLE_ENTRY_H */

