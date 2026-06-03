/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Georgia Institute of Technology
 * Copyright (c) 2021 NITK Surathkal
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
 * This file is adapted from the old ipv4-nix-vector-routing.h.
 *
 * Authors: Josh Pelkey <jpelkey@gatech.edu>
 * 
 * Modified by: Ameya Deshpande <ameyanrd@outlook.com>
 */

#ifndef NIX_VECTOR_ROUTING_H
#define NIX_VECTOR_ROUTING_H

#include "ns3/channel.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/net-device-container.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/nix-vector.h"
#include "ns3/bridge-net-device.h"
#include "ns3/nstime.h"
#include "ns3/ipv4-interface.h"
#include "ns3/ipv6-interface.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6-l3-protocol.h"

#include <atomic>
#include <map>
#include <unordered_map>

namespace ns3 {

/**
 * \defgroup nix-vector-routing Nix-Vector Routing
 *
 * Nix-vector routing is a simulation specific routing protocol and is
 * intended for large network topologies.
 */

/**
 * \ingroup nix-vector-routing
 * Nix-vector routing protocol
 *
 * \internal
 * Since this class is meant to be specialized only by Ipv4RoutingProtocol or 
 * Ipv6RoutingProtocol the implementation of this class doesn't need to be
 * exposed here; it is in nix-vector-routing.cc.
 */
template <typename T>
class NixVectorRouting : public std::enable_if<std::is_same<Ipv4RoutingProtocol, T>::value || std::is_same<Ipv6RoutingProtocol, T>::value, T>::type
{
  /// Alias for determining whether the parent is Ipv4RoutingProtocol or Ipv6RoutingProtocol
  using IsIpv4 = std::is_same <Ipv4RoutingProtocol, T>;

  /// Alias for Ipv4 and Ipv6 classes
  using Ip = typename std::conditional <IsIpv4::value, Ipv4, Ipv6>::type;

  /// Alias for Ipv4Address and Ipv6Address classes
  using IpAddress = typename std::conditional<IsIpv4::value, Ipv4Address, Ipv6Address>::type;

  /// Alias for Ipv4Route and Ipv6Route classes
  using IpRoute = typename std::conditional<IsIpv4::value, Ipv4Route, Ipv6Route>::type;

  /// Alias for Ipv4AddressHash and Ipv6AddressHash classes
  using IpAddressHash = typename std::conditional<IsIpv4::value, Ipv4AddressHash, Ipv6AddressHash>::type;

  /// Alias for Ipv4Header and Ipv6Header classes
  using IpHeader = typename std::conditional<IsIpv4::value, Ipv4Header, Ipv6Header>::type;

  /// Alias for Ipv4InterfaceAddress and Ipv6InterfaceAddress classes
  using IpInterfaceAddress = typename std::conditional<IsIpv4::value, Ipv4InterfaceAddress, Ipv6InterfaceAddress>::type;

  /// Alias for Ipv4Interface and Ipv6Interface classes
  using IpInterface = typename std::conditional<IsIpv4::value, Ipv4Interface, Ipv6Interface>::type;

  /// Alias for Ipv4L3Protocol and Ipv4L3Protocol classes
  using IpL3Protocol = typename std::conditional<IsIpv4::value, Ipv4L3Protocol, Ipv6L3Protocol>::type;

public:
  NixVectorRouting ();
  ~NixVectorRouting ();
  /**
   * @brief The Interface ID of the Global Router interface.
   * @return The Interface ID
   * @see Object::GetObject ()
   */
  static TypeId GetTypeId (void);
  /**
   * @brief Set the Node pointer of the node for which this
   * routing protocol is to be placed
   *
   * @param node Node pointer
   */
  void SetNode (Ptr<Node> node);

  /**
   * @brief Called when run-time link topology change occurs
   * which iterates through the node list and flushes any
   * nix vector caches
   *
   * \internal
   * \c const is used here due to need to potentially flush the cache
   * in const methods such as PrintRoutingTable.  Caches are stored in
   * mutable variables and flushed in const methods.
   */
  void FlushGlobalNixRoutingCache (void) const;

  /**
   * @brief Print the Routing Path according to Nix Routing
   * \param source Source node
   * \param dest Destination node address
   * \param stream The ostream the Routing path is printed to
   * \param unit the time unit to be used in the report
   * 
   * \note IpAddress is alias for either Ipv4Address or Ipv6Address
   *       depending on on whether the network is IPv4 or IPv6 respectively.
   */
  void PrintRoutingPath (Ptr<Node> source, IpAddress dest, Ptr<OutputStreamWrapper> stream, Time::Unit unit) const;


private:

  /**
   * Flushes the cache which stores nix-vector based on
   * destination IP
   */
  void FlushNixCache (void) const;

  /**
   * Flushes the cache which stores the Ip route
   * based on the destination IP
   */
  void FlushIpRouteCache (void) const;

  /**
   * Upon a run-time topology change caches are
   * flushed and the total number of neighbors is
   * reset to zero
   */
  void ResetTotalNeighbors (void);

  /**
   * Takes in the source node and dest IP and calls GetNodeByIp,
   * BFS, accounting for any output interface specified, and finally
   * BuildNixVector to return the built nix-vector
   *
   * \param source Source node
   * \param dest Destination node address
   * \param oif Preferred output interface
   * \returns The NixVector to be used in routing.
   */
  Ptr<NixVector> GetNixVector (Ptr<Node> source, IpAddress dest, Ptr<NetDevice> oif) const;

  /**
   * Checks the cache based on dest IP for the nix-vector
   * \param address Address to check
   * \param foundInCache Address found in cache
   * \returns The NixVector to be used in routing.
   */
  Ptr<NixVector> GetNixVectorInCache (const IpAddress &address,  bool &foundInCache) const;

  /**
   * Checks the cache based on dest IP for the IpRoute
   * \param address Address to check
   * \returns The cached route.
   */
  Ptr<IpRoute> GetIpRouteInCache (IpAddress address);

  /**
   * Given a net-device returns all the adjacent net-devices,
   * essentially getting the neighbors on that channel
   * \param [in] netDevice the NetDevice attached to the channel.
   * \param [in] channel the channel to check
   * \param [out] netDeviceContainer the NetDeviceContainer of the NetDevices in the channel.
   */
  void GetAdjacentNetDevices (Ptr<NetDevice> netDevice, Ptr<Channel> channel, NetDeviceContainer & netDeviceContainer) const;

  /**
   * Iterates through the node list and finds the one
   * corresponding to the given IpAddress
   * \param dest destination node IP
   * \return The node with the specified IP.
   */
  Ptr<Node> GetNodeByIp (IpAddress dest) const;

  /**
   * Iterates through the node list and finds the one
   * corresponding to the given IpAddress
   * \param netDevice NetDevice pointer
   * \return The node with the specified IP.
   */
  Ptr<IpInterface> GetInterfaceByNetDevice (Ptr<NetDevice> netDevice) const;

  /**
   * Recurses the T vector, created by BFS and actually builds the nixvector
   * \param [in] parentVector Parent vector for retracing routes
   * \param [in] source Source Node index
   * \param [in] dest Destination Node index
   * \param [out] nixVector the NixVector to be used for routing
   * \returns true on success, false otherwise.
   */
  bool BuildNixVector (const std::vector< Ptr<Node> > & parentVector, uint32_t source, uint32_t dest, Ptr<NixVector> nixVector) const;

  /**
   * Simply iterates through the nodes net-devices and determines
   * how many neighbors the node has.
   * \param [in] node node pointer
   * \returns the number of neighbors of m_node.
   */
  uint32_t FindTotalNeighbors (Ptr<Node> node) const;


  /**
   * Determine if the NetDevice is bridged
   * \param nd the NetDevice to check
   * \returns the bridging NetDevice (or null if the NetDevice is not bridged)
   */
  Ptr<BridgeNetDevice> NetDeviceIsBridged (Ptr<NetDevice> nd) const;


  /**
   * Nix index is with respect to the neighbors.  The net-device index must be
   * derived from this
   * \param [in] node the current node under consideration
   * \param [in] nodeIndex Nix Node index
   * \param [out] gatewayIp IP address of the gateway
   * \returns the index of the NetDevice in the node.
   */
  uint32_t FindNetDeviceForNixIndex (Ptr<Node> node, uint32_t nodeIndex, IpAddress & gatewayIp) const;

  /**
   * \brief Breadth first search algorithm.
   * \param [in] numberOfNodes total number of nodes
   * \param [in] source Source Node
   * \param [in] dest Destination Node
   * \param [out] parentVector Parent vector for retracing routes
   * \param [in] oif specific output interface to use from source node, if not null
   * \returns false if dest not found, true o.w.
   */
  bool BFS (uint32_t numberOfNodes,
            Ptr<Node> source,
            Ptr<Node> dest,
            std::vector< Ptr<Node> > & parentVector,
            Ptr<NetDevice> oif) const;

  /**
   * \sa Ipv4RoutingProtocol::DoInitialize
   * \sa Ipv6RoutingProtocol::DoInitialize
   */
  void DoInitialize ();

  /**
   * \sa Ipv4RoutingProtocol::DoDispose
   * \sa Ipv6RoutingProtocol::DoDispose
   */
  void DoDispose (void);

  /// Map of IpAddress to NixVector
  typedef std::map<IpAddress, Ptr<NixVector> > NixMap_t;
  /// Map of IpAddress to IpRoute
  typedef std::map<IpAddress, Ptr<IpRoute> > IpRouteMap_t;

  /// Callback for IPv4 unicast packets to be forwarded
  typedef Callback<void, Ptr<IpRoute>, Ptr<const Packet>, const IpHeader &> UnicastForwardCallbackv4;

  /// Callback for IPv6 unicast packets to be forwarded
  typedef Callback<void, Ptr<const NetDevice>, Ptr<IpRoute>, Ptr<const Packet>, const IpHeader &> UnicastForwardCallbackv6;

  /// Callback for unicast packets to be forwarded
  typedef typename std::conditional<IsIpv4::value, UnicastForwardCallbackv4, UnicastForwardCallbackv6>::type UnicastForwardCallback;

  /// Callback for IPv4 multicast packets to be forwarded
  typedef Callback<void, Ptr<Ipv4MulticastRoute>, Ptr<const Packet>, const IpHeader &> MulticastForwardCallbackv4;

  /// Callback for IPv6 multicast packets to be forwarded
  typedef Callback<void, Ptr<const NetDevice>, Ptr<Ipv6MulticastRoute>, Ptr<const Packet>, const IpHeader &> MulticastForwardCallbackv6;

  /// Callback for multicast packets to be forwarded
  typedef typename std::conditional<IsIpv4::value, MulticastForwardCallbackv4, MulticastForwardCallbackv6>::type MulticastForwardCallback;

  /// Callback for packets to be locally delivered
  typedef Callback<void, Ptr<const Packet>, const IpHeader &, uint32_t > LocalDeliverCallback;

  /// Callback for routing errors (e.g., no route found)
  typedef Callback<void, Ptr<const Packet>, const IpHeader &, Socket::SocketErrno > ErrorCallback;

  /* From Ipv4RoutingProtocol and Ipv6RoutingProtocol */
  /**
   * \brief Query routing cache for an existing route, for an outbound packet
   * \param p packet to be routed.  Note that this method may modify the packet.
   *          Callers may also pass in a null pointer.
   * \param header input parameter (used to form key to search for the route)
   * \param oif Output interface Netdevice.  May be zero, or may be bound via
   *            socket options to a particular output interface.
   * \param sockerr Output parameter; socket errno
   *
   * \returns a code that indicates what happened in the lookup
   *
   * \sa Ipv4RoutingProtocol::RouteOutput
   * \sa Ipv6RoutingProtocol::RouteOutput
   */
  virtual Ptr<IpRoute> RouteOutput (Ptr<Packet> p, const IpHeader &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr);

  /**
   * \brief Route an input packet (to be forwarded or locally delivered)
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
   *
   * \returns true if NixVectorRouting class takes responsibility for
   *          forwarding or delivering the packet, false otherwise
   *
   * \sa Ipv4RoutingProtocol::RouteInput
   * \sa Ipv6RoutingProtocol::RouteInput
   */
  virtual bool RouteInput (Ptr<const Packet> p, const IpHeader &header, Ptr<const NetDevice> idev,
                           UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                           LocalDeliverCallback lcb, ErrorCallback ecb);
  
  /**
   * \param interface the index of the interface we are being notified about
   *
   * \sa Ipv4RoutingProtocol::NotifyInterfaceUp
   * \sa Ipv6RoutingProtocol::NotifyInterfaceUp
   */
  virtual void NotifyInterfaceUp (uint32_t interface);

  /**
   * \param interface the index of the interface we are being notified about
   *
   * \sa Ipv4RoutingProtocol::NotifyInterfaceDown
   * \sa Ipv6RoutingProtocol::NotifyInterfaceDown
   */
  virtual void NotifyInterfaceDown (uint32_t interface);

  /**
   * \param interface the index of the interface we are being notified about
   * \param address a new address being added to an interface
   *
   * \sa Ipv4RoutingProtocol::NotifyAddAddress
   * \sa Ipv6RoutingProtocol::NotifyAddAddress
   */
  virtual void NotifyAddAddress (uint32_t interface, IpInterfaceAddress address);

  /**
   * \param interface the index of the interface we are being notified about
   * \param address a new address being added to an interface
   *
   * \sa Ipv4RoutingProtocol::NotifyRemoveAddress
   * \sa Ipv6RoutingProtocol::NotifyRemoveAddress
   */
  virtual void NotifyRemoveAddress (uint32_t interface, IpInterfaceAddress address);

  /**
   * \brief Print the Routing Table entries
   *
   * \param stream The ostream the Routing table is printed to
   * \param unit The time unit to be used in the report
   *
   * \sa Ipv4RoutingProtocol::PrintRoutingTable
   * \sa Ipv6RoutingProtocol::PrintRoutingTable
   */
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit = Time::S) const;

  /* From IPv4RoutingProtocol */
  /**
   * \brief Typically, invoked directly or indirectly from ns3::Ipv4::SetRoutingProtocol
   *
   * \param ipv4 the ipv4 object this routing protocol is being associated with
   *
   * \sa Ipv4RoutingProtocol::SetIpv4
   */
  virtual void SetIpv4 (Ptr<Ip> ipv4);

  /* From IPv6RoutingProtocol */
  /**
   * \brief Typically, invoked directly or indirectly from ns3::Ipv6::SetRoutingProtocol
   *
   * \param ipv6 the ipv6 object this routing protocol is being associated with
   *
   * \sa Ipv6RoutingProtocol::SetIpv6
   */
  virtual void SetIpv6 (Ptr<Ip> ipv6);

  /**
   * \brief Notify a new route.
   *
   * \param dst destination address
   * \param mask destination mask
   * \param nextHop nextHop for this destination
   * \param interface output interface
   * \param prefixToUse prefix to use as source with this route
   *
   * \sa Ipv6RoutingProtocol::NotifyAddRoute
   */
  virtual void NotifyAddRoute (IpAddress dst, Ipv6Prefix mask, IpAddress nextHop, uint32_t interface, IpAddress prefixToUse = IpAddress::GetZero ());

  /**
   * \brief Notify route removing.
   *
   * \param dst destination address
   * \param mask destination mask
   * \param nextHop nextHop for this destination
   * \param interface output interface
   * \param prefixToUse prefix to use as source with this route
   *
   * \sa Ipv6RoutingProtocol::NotifyRemoveRoute
   */
  virtual void NotifyRemoveRoute (IpAddress dst, Ipv6Prefix mask, IpAddress nextHop, uint32_t interface, IpAddress prefixToUse = IpAddress::GetZero ());

  /**
   * Flushes routing caches if required.
   */
  void CheckCacheStateAndFlush (void) const;

  /**
   * Build map from IP Address to Node for faster lookup.
   */
  void BuildIpAddressToNodeMap (void) const;

  /**
   * Flag to mark when caches are dirty and need to be flushed.  
   * Used for lazy cleanup of caches when there are many topology changes.
   */
#ifdef NS3_MTP
  static std::atomic<bool> g_isCacheDirty;
  static std::atomic<bool> g_cacheFlushing;
  static std::atomic<bool> g_isMapBuilt;
  static std::atomic<bool> g_mapBuilding;
#else
  static bool g_isCacheDirty;
#endif

  /**
   * Nix Epoch, incremented each time a flush is perfomed.
   */
  static uint32_t g_epoch;

  /** Cache stores nix-vectors based on destination ip */
  mutable NixMap_t m_nixCache;

  /** Cache stores IpRoutes based on destination ip */
  mutable IpRouteMap_t m_ipRouteCache;

  Ptr<Ip> m_ip; //!< IP object
  Ptr<Node> m_node; //!< Node object

  /** Total neighbors used for nix-vector to determine number of bits */
  uint32_t m_totalNeighbors;


  /**
   * Mapping of IP address to ns-3 node.
   *
   * Used to avoid linear searching of nodes/devices to find a node in
   * GetNodeByIp() method.  NIX vector routing assumes IP addresses
   * are unique so mapping can be done without duplication.
   **/
  typedef std::unordered_map<IpAddress, ns3::Ptr<ns3::Node>, IpAddressHash > IpAddressToNodeMap;
  static IpAddressToNodeMap g_ipAddressToNodeMap; //!< Address to node map.

  /// Mapping of Ptr<NetDevice> to Ptr<IpInterface>.
  typedef std::unordered_map<Ptr<NetDevice>, Ptr<IpInterface>> NetDeviceToIpInterfaceMap;
  static NetDeviceToIpInterfaceMap g_netdeviceToIpInterfaceMap; //!< NetDevice pointer to IpInterface pointer map
};


/**
 * \ingroup nix-vector-routing
 * Create the typedef Ipv4NixVectorRouting with T as Ipv4RoutingProtocol
 *
 * Note: This typedef enables also backwards compatibility with original Ipv4NixVectorRouting.
 */
typedef NixVectorRouting<Ipv4RoutingProtocol> Ipv4NixVectorRouting;

/**
 * \ingroup nix-vector-routing
 * Create the typedef Ipv6NixVectorRouting with T as Ipv6RoutingProtocol
 */
typedef NixVectorRouting<Ipv6RoutingProtocol> Ipv6NixVectorRouting;
} // namespace ns3

#endif /* NIX_VECTOR_ROUTING_H */
