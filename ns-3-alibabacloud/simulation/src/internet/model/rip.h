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

#ifndef RIP_H
#define RIP_H

#include <list>

#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4-interface.h"
#include "ns3/inet-socket-address.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rip-header.h"

namespace ns3 {

/**
 * \ingroup ipv4Routing
 * \defgroup rip RIP
 *
 * The RIP protocol (\RFC{2453}) is a unicast-only IPv4 IGP (Interior Gateway Protocol).
 * Its convergence time is rather long. As a consequence, it is suggested to
 * carefully check the network topology and the route status before sending
 * data flows.
 *
 * RIP implements the Bellman-Ford algorithm (although the RFC does not state it).
 * Bellman-Ford algorithm convergence time is O(|V|*|E|) where |V| and |E| are the
 * number of vertices (routers) and edges (links) respectively. Since unsolicited
 * updates are exchanged every 30 seconds, the convergence might require a long time.
 *
 * For the RIP protocol, the exact convergence time is shorter, thanks to the
 * use of triggered updates, which are sent when a route changes.
 * Even with triggered updates, the convergence is in the order of magnitude of
 * O(|V|*|E|) * 5 seconds, which is still quite long for complex topologies.
 *
 * \todo: Add routing table compression (CIDR). The most evident result: without
 * it a router will announce to be the default router *and* more RTEs, which is silly.
 */

/**
 * \ingroup rip
 * \brief Rip Routing Table Entry
 */
class RipRoutingTableEntry : public Ipv4RoutingTableEntry
{
public:

  /**
   * Route status
   */
  enum Status_e {
    RIP_VALID,
    RIP_INVALID,
  };

  RipRoutingTableEntry (void);

  /**
   * \brief Constructor
   * \param network network address
   * \param networkPrefix network prefix
   * \param nextHop next hop address to route the packet
   * \param interface interface index
   */
  RipRoutingTableEntry (Ipv4Address network, Ipv4Mask networkPrefix, Ipv4Address nextHop, uint32_t interface);

  /**
   * \brief Constructor
   * \param network network address
   * \param networkPrefix network prefix
   * \param interface interface index
   */
  RipRoutingTableEntry (Ipv4Address network, Ipv4Mask networkPrefix, uint32_t interface);

  virtual ~RipRoutingTableEntry ();

  /**
   * \brief Set the route tag
   * \param routeTag the route tag
   */
  void SetRouteTag (uint16_t routeTag);

  /**
   * \brief Get the route tag
   * \returns the route tag
   */
  uint16_t GetRouteTag (void) const;

  /**
   * \brief Set the route metric
   * \param routeMetric the route metric
   */
  void SetRouteMetric (uint8_t routeMetric);

  /**
   * \brief Get the route metric
   * \returns the route metric
   */
  uint8_t GetRouteMetric (void) const;

  /**
   * \brief Set the route status
   * \param status the route status
   */
  void SetRouteStatus (Status_e status);

  /**
   * \brief Get the route status
   * \returns the route status
   */
  Status_e GetRouteStatus (void) const;

  /**
   * \brief Set the route as changed
   *
   * The changed routes are scheduled for a Triggered Update.
   * After a Triggered Update, all the changed flags are cleared
   * from the routing table.
   *
   * \param changed true if route is changed
   */
  void SetRouteChanged (bool changed);

  /**
   * \brief Get the route changed status
   *
   * \returns true if route is changed
   */
  bool IsRouteChanged (void) const;

private:
  uint16_t m_tag; //!< route tag
  uint8_t m_metric; //!< route metric
  Status_e m_status; //!< route status
  bool m_changed; //!< route has been updated
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param route the Ipv4 routing table entry
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, RipRoutingTableEntry const& route);



/**
 * \ingroup rip
 *
 * \brief RIP Routing Protocol, defined in \RFC{2453}.
 */
class Rip : public Ipv4RoutingProtocol
{
public:
  // /< C-tor
  Rip ();
  virtual ~Rip ();

  /**
   * \brief Get the type ID
   * \return type ID
   */
  static TypeId GetTypeId (void);

  // From Ipv4RoutingProtocol
  Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif,
                              Socket::SocketErrno &sockerr);
  bool RouteInput (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
                   UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                   LocalDeliverCallback lcb, ErrorCallback ecb);
  virtual void NotifyInterfaceUp (uint32_t interface);
  virtual void NotifyInterfaceDown (uint32_t interface);
  virtual void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address);
  virtual void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address);
  virtual void SetIpv4 (Ptr<Ipv4> ipv4);
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const;

  /**
   * Split Horizon strategy type. See \RFC{2453}.
   */
  enum SplitHorizonType_e {
    NO_SPLIT_HORIZON,//!< No Split Horizon
    SPLIT_HORIZON,   //!< Split Horizon
    POISON_REVERSE,  //!< Poison Reverse Split Horizon
  };

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

  /**
   * \brief Get the set of interface excluded from the protocol
   * \return the set of excluded interfaces
   */
  std::set<uint32_t> GetInterfaceExclusions () const;

  /**
   * \brief Set the set of interface excluded from the protocol
   * \param exceptions the set of excluded interfaces
   */
  void SetInterfaceExclusions (std::set<uint32_t> exceptions);

  /**
   * \brief Get the metric for an interface
   * \param interface the interface
   * \returns the interface metric
   */
  uint8_t GetInterfaceMetric (uint32_t interface) const;

  /**
   * \brief Set the metric for an interface
   * \param interface the interface
   * \param metric the interface metric
   */
  void SetInterfaceMetric (uint32_t interface, uint8_t metric);

  /**
   * \brief Add a default route to the router through the nextHop located on interface.
   *
   * The default route is usually installed manually, or it is the result of
   * some "other" routing protocol (e.g., BGP).
   *
   * \param nextHop the next hop
   * \param interface the interface
   */
  void AddDefaultRouteTo (Ipv4Address nextHop, uint32_t interface);

protected:
  /**
   * \brief Dispose this object.
   */
  virtual void DoDispose ();

  /**
   * Start protocol operation
   */
  void DoInitialize ();

private:
  /// Container for the network routes - pair RipRoutingTableEntry *, EventId (update event)
  typedef std::list<std::pair <RipRoutingTableEntry *, EventId> > Routes;

  /// Const Iterator for container for the network routes
  typedef std::list<std::pair <RipRoutingTableEntry *, EventId> >::const_iterator RoutesCI;

  /// Iterator for container for the network routes
  typedef std::list<std::pair <RipRoutingTableEntry *, EventId> >::iterator RoutesI;


  /**
   * \brief Receive RIP packets.
   *
   * \param socket the socket the packet was received to.
   */
  void Receive (Ptr<Socket> socket);

  /**
   * \brief Handle RIP requests.
   *
   * \param hdr message header (including RTEs)
   * \param senderAddress sender address
   * \param senderPort sender port
   * \param incomingInterface incoming interface
   * \param hopLimit packet's hop limit
   */
  void HandleRequests (RipHeader hdr, Ipv4Address senderAddress, uint16_t senderPort, uint32_t incomingInterface, uint8_t hopLimit);

  /**
   * \brief Handle RIP responses.
   *
   * \param hdr message header (including RTEs)
   * \param senderAddress sender address
   * \param incomingInterface incoming interface
   * \param hopLimit packet's hop limit
   */
  void HandleResponses (RipHeader hdr, Ipv4Address senderAddress, uint32_t incomingInterface, uint8_t hopLimit);

  /**
   * \brief Lookup in the forwarding table for destination.
   * \param dest destination address
   * \param setSource set source address in the route
   * \param interface output interface if any (put 0 otherwise)
   * \return Ipv4Route to route the packet to reach dest address
   */
  Ptr<Ipv4Route> Lookup (Ipv4Address dest, bool setSource, Ptr<NetDevice> = 0);

  /**
   * Receive and process unicast packet
   * \param socket socket where packet is arrived
   */
  void RecvUnicastRip (Ptr<Socket> socket);
  /**
   * Receive and process multicast packet
   * \param socket socket where packet is arrived
   */
  void RecvMulticastRip (Ptr<Socket> socket);

  /**
   * \brief Add route to network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param nextHop next hop address to route the packet.
   * \param interface interface index
   */
  void AddNetworkRouteTo (Ipv4Address network, Ipv4Mask networkPrefix, Ipv4Address nextHop, uint32_t interface);

  /**
   * \brief Add route to network.
   * \param network network address
   * \param networkPrefix network prefix
   * \param interface interface index
   */
  void AddNetworkRouteTo (Ipv4Address network, Ipv4Mask networkPrefix, uint32_t interface);

  /**
   * \brief Send Routing Updates on all interfaces.
   * \param periodic true for periodic update, else triggered.
   */
  void DoSendRouteUpdate (bool periodic);

  /**
   * \brief Send Routing Request on all interfaces.
   */
  void SendRouteRequest ();

  /**
   * \brief Send Triggered Routing Updates on all interfaces.
   */
  void SendTriggeredRouteUpdate ();

  /**
   * \brief Send Unsolicited Routing Updates on all interfaces.
   */
  void SendUnsolicitedRouteUpdate (void);

  /**
   * \brief Invalidate a route.
   * \param route the route to be removed
   */
  void InvalidateRoute (RipRoutingTableEntry *route);

  /**
   * \brief Delete a route.
   * \param route the route to be removed
   */
  void DeleteRoute (RipRoutingTableEntry *route);

  Routes m_routes; //!<  the forwarding table for network.
  Ptr<Ipv4> m_ipv4; //!< IPv4 reference
  Time m_startupDelay; //!< Random delay before protocol startup.
  Time m_minTriggeredUpdateDelay; //!< Min cooldown delay after a Triggered Update.
  Time m_maxTriggeredUpdateDelay; //!< Max cooldown delay after a Triggered Update.
  Time m_unsolicitedUpdate; //!< time between two Unsolicited Routing Updates
  Time m_timeoutDelay; //!< Delay before invalidating a route
  Time m_garbageCollectionDelay; //!< Delay before deleting an INVALID route

  // note: we can not trust the result of socket->GetBoundNetDevice ()->GetIfIndex ();
  // it is dependent on the interface initialization (i.e., if the loopback is already up).
  /// Socket list type
  typedef std::map< Ptr<Socket>, uint32_t> SocketList;
  /// Socket list type iterator
  typedef std::map<Ptr<Socket>, uint32_t>::iterator SocketListI;
  /// Socket list type const iterator
  typedef std::map<Ptr<Socket>, uint32_t>::const_iterator SocketListCI;

  SocketList m_unicastSocketList; //!< list of sockets for unicast messages (socket, interface index)
  Ptr<Socket> m_multicastRecvSocket; //!< multicast receive socket

  EventId m_nextUnsolicitedUpdate; //!< Next Unsolicited Update event
  EventId m_nextTriggeredUpdate; //!< Next Triggered Update event

  Ptr<UniformRandomVariable> m_rng; //!< Rng stream.

  std::set<uint32_t> m_interfaceExclusions; //!< Set of excluded interfaces
  std::map<uint32_t, uint8_t> m_interfaceMetrics; //!< Map of interface metrics

  SplitHorizonType_e m_splitHorizonStrategy; //!< Split Horizon strategy

  bool m_initialized; //!< flag to allow socket's late-creation.
  uint32_t m_linkDown; //!< Link down value.
};

} // namespace ns3
#endif /* RIP_H */

