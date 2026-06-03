/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

/* taken from src/node/ipv4-routing-protocol.h and adapted to IPv6 */

#ifndef IPV6_ROUTING_PROTOCOL_H
#define IPV6_ROUTING_PROTOCOL_H

#include "ns3/packet.h"
#include "ns3/callback.h"
#include "ns3/object.h"
#include "ns3/socket.h"

#include "ipv6-header.h"
#include "ipv6-interface-address.h"
#include "ipv6.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/nstime.h"

namespace ns3 {

class Ipv6MulticastRoute;
class Ipv6Route;
class NetDevice;

/**
 * \ingroup internet 
 * \defgroup ipv6Routing IPv6 Routing Protocols.
 *
 * The classes in this group implement different routing protocols
 * for IPv6. Other modules could implement further protocols.
 */

/**
 * \ingroup ipv6Routing
 * \brief Abstract base class for IPv6 routing protocols.
 * 
 * Defines two virtual functions for packet routing and forwarding.  The first, 
 * RouteOutput (), is used for locally originated packets, and the second,
 * RouteInput (), is used for forwarding and/or delivering received packets. 
 * Also defines the signatures of four callbacks used in RouteInput ().
 */

class Ipv6RoutingProtocol : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /// Callback for unicast packets to be forwarded
  typedef Callback<void, Ptr<const NetDevice>, Ptr<Ipv6Route>, Ptr<const Packet>, const Ipv6Header &> UnicastForwardCallback;

  /// Callback for multicast packets to be forwarded
  typedef Callback<void, Ptr<const NetDevice>, Ptr<Ipv6MulticastRoute>, Ptr<const Packet>, const Ipv6Header &> MulticastForwardCallback;

  /// Callback for packets to be locally delivered
  typedef Callback<void, Ptr<const Packet>, const Ipv6Header &, uint32_t > LocalDeliverCallback;

  /// Callback for routing errors (e.g., no route found)
  typedef Callback<void, Ptr<const Packet>, const Ipv6Header &, Socket::SocketErrno > ErrorCallback;

  /**
   * \brief Query routing cache for an existing route, for an outbound packet
   *
   * This lookup is used by transport protocols.  It does not cause any
   * packet to be forwarded, and is synchronous.  Can be used for
   * multicast or unicast.  The Linux equivalent is ip_route_output ()
   *
   * \param p packet to be routed.  Note that this method may modify the packet.
   *          Callers may also pass in a null pointer. 
   * \param header input parameter (used to form key to search for the route)
   * \param oif Output interface device.  May be zero, or may be bound via
   *            socket options to a particular output interface.
   * \param sockerr Output parameter; socket errno 
   *
   * \returns a code that indicates what happened in the lookup
   */
  virtual Ptr<Ipv6Route> RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr) = 0;

  /**
   * \brief Route an input packet (to be forwarded or locally delivered)
   *
   * This lookup is used in the forwarding process.  The packet is
   * handed over to the Ipv6RoutingProtocol, and will get forwarded onward
   * by one of the callbacks.  The Linux equivalent is ip_route_input ().
   * There are four valid outcomes, and a matching callbacks to handle each.
   *
   * \param p received packet
   * \param header input parameter used to form a search key for a route
   * \param idev Pointer to ingress network device
   * \param ucb Callback for the case in which the packet is to be forwarded
   *            as unicast
   * \param mcb Callback for the case in which the packet is to be forwarded
   *            as multicast
   * \param lcb Callback for the case in which the packet is to be locally
   *            delivered
   * \param ecb Callback to call if there is an error in forwarding
   * \returns true if the Ipv6RoutingProtocol takes responsibility for 
   *          forwarding or delivering the packet, false otherwise
   */ 
  virtual bool RouteInput  (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev, 
                            UnicastForwardCallback ucb, MulticastForwardCallback mcb, 
                            LocalDeliverCallback lcb, ErrorCallback ecb) = 0;

  /**
   * \brief Notify when specified interface goes UP.
   *
   * Protocols are expected to implement this method to be notified of the state change of
   * an interface in a node.
   * \param interface the index of the interface we are being notified about
   */
  virtual void NotifyInterfaceUp (uint32_t interface) = 0;

  /**
   * \brief Notify when specified interface goes DOWN.
   *
   * Protocols are expected to implement this method to be notified of the state change of
   * an interface in a node.
   * \param interface the index of the interface we are being notified about
   */
  virtual void NotifyInterfaceDown (uint32_t interface) = 0;

  /**
   * \brief Notify when specified interface add an address.
   *
   * Protocols are expected to implement this method to be notified whenever
   * a new address is added to an interface. Typically used to add a 'network route' on an
   * interface. Can be invoked on an up or down interface.
   * \param interface the index of the interface we are being notified about
   * \param address a new address being added to an interface
   */
  virtual void NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address) = 0;

  /**
   * \brief Notify when specified interface add an address.
   *
   * Protocols are expected to implement this method to be notified whenever
   * a new address is removed from an interface. Typically used to remove the 'network route' of an
   * interface. Can be invoked on an up or down interface.
   * \param interface the index of the interface we are being notified about
   * \param address a new address being added to an interface
   */
  virtual void NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address) = 0;

  /**
   * \brief Notify a new route.
   *
   * Typically this is used to add another route from IPv6 stack (i.e. ICMPv6
   * redirect case, ...).
   * \param dst destination address
   * \param mask destination mask
   * \param nextHop nextHop for this destination
   * \param interface output interface
   * \param prefixToUse prefix to use as source with this route
   */
  virtual void NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::GetZero ()) = 0;

  /**
   * \brief Notify route removing.
   * \param dst destination address
   * \param mask destination mask
   * \param nextHop nextHop for this destination
   * \param interface output interface
   * \param prefixToUse prefix to use as source with this route
   */
  virtual void NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::GetZero ()) = 0;

  /**
   * \brief Typically, invoked directly or indirectly from ns3::Ipv6::SetRoutingProtocol
   * \param ipv6 the ipv6 object this routing protocol is being associated with
   */
  virtual void SetIpv6 (Ptr<Ipv6> ipv6) = 0;

  /**
   * \brief Print the Routing Table entries
   *
   * \param stream The ostream the Routing table is printed to
   * \param unit The time unit to be used in the report
   */
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const = 0;

};

} // namespace ns3

#endif /* IPV6_ROUTING_PROTOCOL_H */

