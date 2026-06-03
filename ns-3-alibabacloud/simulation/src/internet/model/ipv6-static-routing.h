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

#ifndef IPV6_STATIC_ROUTING_H
#define IPV6_STATIC_ROUTING_H

#include <stdint.h>

#include <list>

#include "ns3/ptr.h"
#include "ns3/ipv6-address.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-header.h"
#include "ns3/ipv6-routing-protocol.h"

namespace ns3 {

class Packet;
class NetDevice;
class Ipv6Interface;
class Ipv6Route;
class Node;
class Ipv6RoutingTableEntry;
class Ipv6MulticastRoutingTableEntry;

/**
 * \ingroup ipv6Routing
 *
 * \brief Static routing protocol for IP version 6 stacks.
 *
 * This class provides a basic set of methods for inserting static
 * unicast and multicast routes into the Ipv6 routing system.
 * This particular protocol is designed to be inserted into an
 * Ipv6ListRouting protocol but can be used also as a standalone
 * protocol.
 *
 * The Ipv6StaticRouting class inherits from the abstract base class
 * Ipv6RoutingProtocol that defines the interface methods that a routing
 * protocol must support.
 *
 * \see Ipv6RoutingProtocol
 * \see Ipv6ListRouting
 * \see Ipv6ListRouting::AddRoutingProtocol
 */
class Ipv6StaticRouting : public Ipv6RoutingProtocol
{
public:
  /**
   * \brief The interface Id associated with this class.
   * \return type identifier
   */
  static TypeId GetTypeId ();

  Ipv6StaticRouting ();
  virtual ~Ipv6StaticRouting ();

  /**
   * \brief Add route to host.
   * \param dest destination address
   * \param nextHop next hop address to route the packet.
   * \param interface interface index
   * \param prefixToUse prefix that should be used for source address for this destination
   * \param metric metric of route in case of multiple routes to same destination
   */
  void AddHostRouteTo (Ipv6Address dest, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address ("::"), uint32_t metric = 0);

  /**
   * \brief Add route to host.
   * \param dest destination address.
   * \param interface interface index
   * \param metric metric of route in case of multiple routes to same destination
   */
  void AddHostRouteTo (Ipv6Address dest, uint32_t interface, uint32_t metric = 0);

  /**
   * \brief Add route to network.
   * \param network network address
   * \param networkPrefix network prefix*
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \param metric metric of route in case of multiple routes to same destination
   */
  void AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, uint32_t metric = 0);

  /**
   * \brief Add route to network.
   * \param network network address
   * \param networkPrefix network prefix*
   * \param nextHop next hop address to route the packet.
   * \param interface interface index
   * \param prefixToUse prefix that should be used for source address for this destination
   * \param metric metric of route in case of multiple routes to same destination
   */
  void AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse, uint32_t metric = 0);

  /**
   * \brief Add route to network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param interface interface index
   * \param metric metric of route in case of multiple routes to same destination
   */
  void AddNetworkRouteTo (Ipv6Address network, Ipv6Prefix networkPrefix, uint32_t interface, uint32_t metric = 0);

  /**
   * \brief Set the default route.
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   * \param prefixToUse prefix to use (i.e for multihoming)
   * \param metric metric of route in case of multiple routes to same destination
   */
  void SetDefaultRoute (Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address ("::"), uint32_t metric = 0);

  /**
   * \brief Get the number or entries in the routing table.
   * \return number of entries
   */
  uint32_t GetNRoutes () const;

  /**
   * \brief Get the default route.
   *
   * If multiple default routes exist, the one with lowest metric is returned.
   * \return default Ipv6Route
   */
  Ipv6RoutingTableEntry GetDefaultRoute ();

  /**
   * \brief Get a specified route.
   * \param i index
   * \return the route whose index is i
   */
  Ipv6RoutingTableEntry GetRoute (uint32_t i) const;

  /**
   * \brief Get a metric for route from the static unicast routing table.
   * \param index The index (into the routing table) of the route to retrieve.
   * \return If route is set, the metric is returned. If not, an infinity metric (0xffffffff) is returned
   */
  uint32_t GetMetric (uint32_t index) const;

  /**
   * \brief Remove a route from the routing table.
   * \param i index
   */
  void RemoveRoute (uint32_t i);

  /**
   * \brief Remove a route from the routing table.
   * \param network IPv6 network
   * \param prefix IPv6 prefix
   * \param ifIndex interface index
   * \param prefixToUse IPv6 prefix to use with this route (multihoming)
   */
  void RemoveRoute (Ipv6Address network, Ipv6Prefix prefix, uint32_t ifIndex, Ipv6Address prefixToUse);

  /**
   * \brief Add a multicast route for a given multicast source and group.
   * \param origin IPv6 address of the source
   * \param group the multicast group address.
   * \param inputInterface the interface index
   * \param outputInterfaces the list of output interface indices over which the packet
   * should be sent (excluding the inputInterface).
   */
  void AddMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface, std::vector<uint32_t> outputInterfaces);

  /**
   * \brief Set the default multicast route.
   * \param outputInterface default output interface
   */
  void SetDefaultMulticastRoute (uint32_t outputInterface);

  /**
   * \brief Get the number of entries in the multicast routing table.
   * \return number of entries
   */
  uint32_t GetNMulticastRoutes () const;

  /**
   * \brief Get the specified multicast route.
   * \param i index
   * \return the route whose index is i
   */
  Ipv6MulticastRoutingTableEntry GetMulticastRoute (uint32_t i) const;

  /**
   * \brief Remove a static multicast route.
   * \param origin IPv6 address of the source
   * \param group the multicast group address.
   * \param inputInterface the input interface index
   * \return true on success
   */
  bool RemoveMulticastRoute (Ipv6Address origin, Ipv6Address group, uint32_t inputInterface);

  /**
   * \brief Remove a multicast route.
   * \param i index of route to remove
   */
  void RemoveMulticastRoute (uint32_t i);

  /**
   * \brief If the destination is already present in network destination list.
   * \param dest destination address
   * \param interfaceIndex interface index
   * \return true if dest is already in list, false otherwise
   */
  bool HasNetworkDest (Ipv6Address dest, uint32_t interfaceIndex);

  virtual Ptr<Ipv6Route> RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr);

  virtual bool RouteInput  (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev,
                            UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                            LocalDeliverCallback lcb, ErrorCallback ecb);

  virtual void NotifyInterfaceUp (uint32_t interface);
  virtual void NotifyInterfaceDown (uint32_t interface);
  virtual void NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address);
  virtual void NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address);
  virtual void NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::GetZero ());
  virtual void NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::GetZero ());
  virtual void SetIpv6 (Ptr<Ipv6> ipv6);
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const;

protected:
  /**
   * \brief Dispose this object.
   */
  virtual void DoDispose ();

private:
  /// Container for the network routes
  typedef std::list<std::pair <Ipv6RoutingTableEntry *, uint32_t> > NetworkRoutes;

  /// Const Iterator for container for the network routes
  typedef std::list<std::pair <Ipv6RoutingTableEntry *, uint32_t> >::const_iterator NetworkRoutesCI;

  /// Iterator for container for the network routes
  typedef std::list<std::pair <Ipv6RoutingTableEntry *, uint32_t> >::iterator NetworkRoutesI;

  /// Container for the multicast routes
  typedef std::list<Ipv6MulticastRoutingTableEntry *> MulticastRoutes;

  /// Const Iterator for container for the multicast routes
  typedef std::list<Ipv6MulticastRoutingTableEntry *>::const_iterator MulticastRoutesCI;

  /// Iterator for container for the multicast routes
  typedef std::list<Ipv6MulticastRoutingTableEntry *>::iterator MulticastRoutesI;

  /**
   * \brief Checks if a route is already present in the forwarding table.
   * \param route route
   * \param metric metric of route
   * \return true if the route/metric is already in the forwarding table
   */
  bool LookupRoute (const Ipv6RoutingTableEntry &route, uint32_t metric);

  /**
   * \brief Lookup in the forwarding table for destination.
   * \param dest destination address
   * \param interface output interface if any (put 0 otherwise)
   * \return Ipv6Route to route the packet to reach dest address
   */
  Ptr<Ipv6Route> LookupStatic (Ipv6Address dest, Ptr<NetDevice> = 0);

  /**
   * \brief Lookup in the multicast forwarding table for destination.
   * \param origin source address
   * \param group group multicast address
   * \param ifIndex interface index
   * \return Ipv6MulticastRoute to route the packet to reach dest address
   */
  Ptr<Ipv6MulticastRoute> LookupStatic (Ipv6Address origin, Ipv6Address group, uint32_t ifIndex);

  /**
   * \brief the forwarding table for network.
   */
  NetworkRoutes m_networkRoutes;

  /**
   * \brief the forwarding table for multicast.
   */
  MulticastRoutes m_multicastRoutes;

  /**
   * \brief Ipv6 reference.
   */
  Ptr<Ipv6> m_ipv6;
};

} /* namespace ns3 */

#endif /* IPV6_STATIC_ROUTING_H */

