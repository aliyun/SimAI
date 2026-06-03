/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
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
 * Author: George F. Riley<riley@ece.gatech.edu>
 *         Gustavo Carneiro <gjc@inescporto.pt>
 */

#ifndef IPV4_STATIC_ROUTING_H
#define IPV4_STATIC_ROUTING_H

#include <list>
#include <utility>
#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/socket.h"
#include "ns3/ptr.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-routing-protocol.h"

namespace ns3 {

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;
class Ipv4MulticastRoutingTableEntry;
class Node;

/**
 * \ingroup ipv4Routing
 * 
 * \brief Static routing protocol for IP version 4 stacks.
 *
 * This class provides a basic set of methods for inserting static
 * unicast and multicast routes into the Ipv4 routing system.
 * This particular protocol is designed to be inserted into an 
 * Ipv4ListRouting protocol but can be used also as a standalone
 * protocol.
 * 
 * The Ipv4StaticRouting class inherits from the abstract base class 
 * Ipv4RoutingProtocol that defines the interface methods that a routing 
 * protocol must support.
 *
 * \see Ipv4RoutingProtocol
 * \see Ipv4ListRouting
 * \see Ipv4ListRouting::AddRoutingProtocol
 */
class Ipv4StaticRouting : public Ipv4RoutingProtocol
{
public:
  /**
   * \brief The interface Id associated with this class.
   * \return type identifier
   */
  static TypeId GetTypeId (void);

  Ipv4StaticRouting ();
  virtual ~Ipv4StaticRouting ();

  virtual Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr);

  virtual bool RouteInput  (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
                            UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                            LocalDeliverCallback lcb, ErrorCallback ecb);

  virtual void NotifyInterfaceUp (uint32_t interface);
  virtual void NotifyInterfaceDown (uint32_t interface);
  virtual void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address);
  virtual void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address);
  virtual void SetIpv4 (Ptr<Ipv4> ipv4);
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const;

/**
 * \brief Add a network route to the static routing table.
 *
 * \param network The Ipv4Address network for this route.
 * \param networkMask The Ipv4Mask to extract the network.
 * \param nextHop The next hop in the route to the destination network.
 * \param interface The network interface index used to send packets to the
 * destination.
 * \param metric Metric of route in case of multiple routes to same destination
 *
 * \see Ipv4Address
 */
  void AddNetworkRouteTo (Ipv4Address network, 
                          Ipv4Mask networkMask, 
                          Ipv4Address nextHop, 
                          uint32_t interface,
                          uint32_t metric = 0);

/**
 * \brief Add a network route to the static routing table.
 *
 * \param network The Ipv4Address network for this route.
 * \param networkMask The Ipv4Mask to extract the network.
 * \param interface The network interface index used to send packets to the
 * destination.
 * \param metric Metric of route in case of multiple routes to same destination
 *
 * \see Ipv4Address
 */
  void AddNetworkRouteTo (Ipv4Address network, 
                          Ipv4Mask networkMask, 
                          uint32_t interface,
                          uint32_t metric = 0);

/**
 * \brief Add a host route to the static routing table.
 *
 * \param dest The Ipv4Address destination for this route.
 * \param nextHop The Ipv4Address of the next hop in the route.
 * \param interface The network interface index used to send packets to the
 * destination.
 * \param metric Metric of route in case of multiple routes to same destination
 *
 * \see Ipv4Address
 */
  void AddHostRouteTo (Ipv4Address dest, 
                       Ipv4Address nextHop, 
                       uint32_t interface,
                       uint32_t metric = 0);
/**
 * \brief Add a host route to the static routing table.
 *
 * \param dest The Ipv4Address destination for this route.
 * \param interface The network interface index used to send packets to the
 * destination.
 * \param metric Metric of route in case of multiple routes to same destination
 *
 * \see Ipv4Address
 */
  void AddHostRouteTo (Ipv4Address dest, 
                       uint32_t interface,
                       uint32_t metric = 0);
/**
 * \brief Add a default route to the static routing table.
 *
 * This method tells the routing system what to do in the case where a specific
 * route to a destination is not found.  The system forwards packets to the
 * specified node in the hope that it knows better how to route the packet.
 * 
 * If the default route is set, it is returned as the selected route from 
 * LookupStatic irrespective of destination address if no specific route is
 * found.
 *
 * \param nextHop The Ipv4Address to send packets to in the hope that they
 * will be forwarded correctly.
 * \param interface The network interface index used to send packets.
 * \param metric Metric of route in case of multiple routes to same destination
 *
 * \see Ipv4Address
 * \see Ipv4StaticRouting::Lookup
 */
  void SetDefaultRoute (Ipv4Address nextHop, 
                        uint32_t interface,
                        uint32_t metric = 0);

/**
 * \brief Get the number of individual unicast routes that have been added
 * to the routing table.
 *
 * \warning The default route counts as one of the routes.
 * \return number of entries
 */
  uint32_t GetNRoutes (void) const;

/**
 * \brief Get the default route with lowest metric from the static routing table.
 *
 * \return If the default route is set, a pointer to that Ipv4RoutingTableEntry is
 * returned, otherwise an empty routing table entry is returned. 
*  If multiple default routes exist, the one with lowest metric is returned.
 *
 * \see Ipv4RoutingTableEntry
 */
  Ipv4RoutingTableEntry GetDefaultRoute (void);

/**
 * \brief Get a route from the static unicast routing table.
 *
 * Externally, the unicast static routing table appears simply as a table with
 * n entries.
 *
 * \param i The index (into the routing table) of the route to retrieve.
 * \return If route is set, a pointer to that Ipv4RoutingTableEntry is returned, otherwise
 * a zero pointer is returned.
 *
 * \see Ipv4RoutingTableEntry
 * \see Ipv4StaticRouting::RemoveRoute
 */
  Ipv4RoutingTableEntry GetRoute (uint32_t i) const;

/**
 * \brief Get a metric for route from the static unicast routing table.
 *
 * \param index The index (into the routing table) of the route to retrieve.
 * \return If route is set, the metric is returned. If not, an infinity metric (0xffffffff) is returned
 *
 */
  uint32_t GetMetric (uint32_t index) const;

/**
 * \brief Remove a route from the static unicast routing table.
 *
 * Externally, the unicast static routing table appears simply as a table with
 * n entries.
 *
 * \param i The index (into the routing table) of the route to remove.
 *
 * \see Ipv4RoutingTableEntry
 * \see Ipv4StaticRouting::GetRoute
 * \see Ipv4StaticRouting::AddRoute
 */
  void RemoveRoute (uint32_t i);

/**
 * \brief Add a multicast route to the static routing table.
 *
 * A multicast route must specify an origin IP address, a multicast group and
 * an input network interface index as conditions and provide a vector of
 * output network interface indices over which packets matching the conditions
 * are sent.
 *
 * Typically there are two main types of multicast routes:  routes used during
 * forwarding, and routes used in the originator node.
 * For forwarding, all of the conditions must be explicitly provided.
 * For originator nodes, the route is equivalent to a unicast route, and
 * must be added through `AddHostRouteTo`.
 * 
 * \param origin The Ipv4Address of the origin of packets for this route.  May
 * be Ipv4Address:GetAny for open groups.
 * \param group The Ipv4Address of the multicast group or this route.
 * \param inputInterface The input network interface index over which to 
 * expect packets destined for this route.
 * \param outputInterfaces A vector of network interface indexes used to specify
 * how to send packets to the destination(s).
 *
 * \see Ipv4Address
 */
  void AddMulticastRoute (Ipv4Address origin,
                          Ipv4Address group,
                          uint32_t inputInterface,
                          std::vector<uint32_t> outputInterfaces);

/**
 * \brief Add a default multicast route to the static routing table.
 *
 * This is the multicast equivalent of the unicast version SetDefaultRoute.
 * We tell the routing system what to do in the case where a specific route
 * to a destination multicast group is not found.  The system forwards 
 * packets out the specified interface in the hope that "something out there"
 * knows better how to route the packet.  This method is only used in 
 * initially sending packets off of a host.  The default multicast route is
 * not consulted during forwarding -- exact routes must be specified using
 * AddMulticastRoute for that case.
 *
 * Since we're basically sending packets to some entity we think may know
 * better what to do, we don't pay attention to "subtleties" like origin
 * address, nor do we worry about forwarding out multiple  interfaces.  If the
 * default multicast route is set, it is returned as the selected route from 
 * LookupStatic irrespective of origin or multicast group if another specific
 * route is not found.
 *
 * \param outputInterface The network interface index used to specify where
 * to send packets in the case of unknown routes.
 *
 * \see Ipv4Address
 */
  void SetDefaultMulticastRoute (uint32_t outputInterface);

/**
 * \brief Get the number of individual multicast routes that have been added
 * to the routing table.
 *
 * \warning The default multicast route counts as one of the routes.
 * \return number of entries
 */
  uint32_t GetNMulticastRoutes (void) const;

/**
 * \brief Get a route from the static multicast routing table.
 *
 * Externally, the multicast static routing table appears simply as a table 
 * with n entries.
 * 
 * \param i The index (into the routing table) of the multicast route to
 * retrieve.
 * \return If route \e i is set, a pointer to that Ipv4MulticastRoutingTableEntry is
 * returned, otherwise a zero pointer is returned.
 *
 * \see Ipv4MulticastRoutingTableEntry
 * \see Ipv4StaticRouting::RemoveRoute
 */
  Ipv4MulticastRoutingTableEntry GetMulticastRoute (uint32_t i) const;

/**
 * \brief Remove a route from the static multicast routing table.
 *
 * Externally, the multicast static routing table appears simply as a table 
 * with n entries.
 * This method causes the multicast routing table to be searched for the first
 * route that matches the parameters and removes it.
 *
 * Wildcards may be provided to this function, but the wildcards are used to
 * exactly match wildcards in the routes (see AddMulticastRoute).  That is,
 * calling RemoveMulticastRoute with the origin set to "0.0.0.0" will not
 * remove routes with any address in the origin, but will only remove routes
 * with "0.0.0.0" set as the the origin.
 *
 * \param origin The IP address specified as the origin of packets for the
 * route.
 * \param group The IP address specified as the multicast group address of
 * the route.
 * \param inputInterface The network interface index specified as the expected
 * input interface for the route.
 * \returns true if a route was found and removed, false otherwise.
 *
 * \see Ipv4MulticastRoutingTableEntry
 * \see Ipv4StaticRouting::AddMulticastRoute
 */
  bool RemoveMulticastRoute (Ipv4Address origin,
                             Ipv4Address group,
                             uint32_t inputInterface);

/**
 * \brief Remove a route from the static multicast routing table.
 *
 * Externally, the multicast static routing table appears simply as a table 
 * with n entries.
 *
 * \param index The index (into the multicast routing table) of the route to
 * remove.
 *
 * \see Ipv4RoutingTableEntry
 * \see Ipv4StaticRouting::GetRoute
 * \see Ipv4StaticRouting::AddRoute
 */
  void RemoveMulticastRoute (uint32_t index);

protected:
  virtual void DoDispose (void);

private:
  /// Container for the network routes
  typedef std::list<std::pair <Ipv4RoutingTableEntry *, uint32_t> > NetworkRoutes;

  /// Const Iterator for container for the network routes
  typedef std::list<std::pair <Ipv4RoutingTableEntry *, uint32_t> >::const_iterator NetworkRoutesCI;

  /// Iterator for container for the network routes
  typedef std::list<std::pair <Ipv4RoutingTableEntry *, uint32_t> >::iterator NetworkRoutesI;

  /// Container for the multicast routes
  typedef std::list<Ipv4MulticastRoutingTableEntry *> MulticastRoutes;

  /// Const Iterator for container for the multicast routes
  typedef std::list<Ipv4MulticastRoutingTableEntry *>::const_iterator MulticastRoutesCI;

  /// Iterator for container for the multicast routes
  typedef std::list<Ipv4MulticastRoutingTableEntry *>::iterator MulticastRoutesI;

  /**
   * \brief Checks if a route is already present in the forwarding table.
   * \param route route
   * \param metric metric of route
   * \return true if the route/metric is already in the forwarding table
   */
  bool LookupRoute (const Ipv4RoutingTableEntry &route, uint32_t metric);

  /**
   * \brief Lookup in the forwarding table for destination.
   * \param dest destination address
   * \param oif output interface if any (put 0 otherwise)
   * \return Ipv4Route to route the packet to reach dest address
   */
  Ptr<Ipv4Route> LookupStatic (Ipv4Address dest, Ptr<NetDevice> oif = 0);

  /**
   * \brief Lookup in the multicast forwarding table for destination.
   * \param origin source address
   * \param group group multicast address
   * \param interface interface index
   * \return Ipv4MulticastRoute to route the packet to reach dest address
   */
  Ptr<Ipv4MulticastRoute> LookupStatic (Ipv4Address origin, Ipv4Address group,
                                        uint32_t interface);

  /**
   * \brief the forwarding table for network.
   */
  NetworkRoutes m_networkRoutes;

  /**
   * \brief the forwarding table for multicast.
   */
  MulticastRoutes m_multicastRoutes;

  /**
   * \brief Ipv4 reference.
   */
  Ptr<Ipv4> m_ipv4;
};

} // Namespace ns3

#endif /* IPV4_STATIC_ROUTING_H */
