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

#ifndef IPV6_L3_PROTOCOL_H
#define IPV6_L3_PROTOCOL_H

#include <list>

#include "ns3/traced-callback.h"
#include "ns3/net-device.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-address.h"
#include "ns3/ipv6-header.h"
#include "ns3/ipv6-pmtu-cache.h"

class Ipv6L3ProtocolTestCase;

namespace ns3
{

class Node;
class Ipv6Interface;
class IpL4Protocol;
class Ipv6Route;
class Ipv6MulticastRoute;
class Ipv6RawSocketImpl;
class Icmpv6L4Protocol;
class Ipv6AutoconfiguredPrefix;

/**
 * \ingroup ipv6
 *
 * \brief IPv6 layer implementation.
 *
 * This class contains two distinct groups of trace sources.  The
 * trace sources 'Rx' and 'Tx' are called, respectively, immediately
 * after receiving from the NetDevice and immediately before sending
 * to a NetDevice for transmitting a packet.  These are low level
 * trace sources that include the Ipv6Header already serialized into
 * the packet.  In contrast, the Drop, SendOutgoing, UnicastForward,
 * and LocalDeliver trace sources are slightly higher-level and pass
 * around the Ipv6Header as an explicit parameter and not as part of
 * the packet.
 */
class Ipv6L3Protocol : public Ipv6
{
public:
  /** 
   * \brief Get the type ID of this class.
   * \return type ID
   */
  static TypeId GetTypeId ();

  /**
   * \brief The protocol number for IPv6 (0x86DD).
   */
  static const uint16_t PROT_NUMBER;

  /**
   * \enum DropReason
   * \brief Reason why a packet has been dropped.
   */
  enum DropReason 
  {
    DROP_TTL_EXPIRED = 1, /**< Packet TTL has expired */
    DROP_NO_ROUTE, /**< No route to host */
    DROP_INTERFACE_DOWN, /**< Interface is down so can not send packet */
    DROP_ROUTE_ERROR, /**< Route error */
    DROP_UNKNOWN_PROTOCOL, /**< Unknown L4 protocol */
    DROP_UNKNOWN_OPTION, /**< Unknown option */
    DROP_MALFORMED_HEADER, /**< Malformed header */
    DROP_FRAGMENT_TIMEOUT, /**< Fragment timeout */
  };

  /**
   * \brief Constructor.
   */
  Ipv6L3Protocol ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6L3Protocol ();

  // Delete copy constructor and assignment operator to avoid misuse
  Ipv6L3Protocol (const Ipv6L3Protocol &) = delete;
  Ipv6L3Protocol & operator = (const Ipv6L3Protocol &) = delete;

  /**
   * \brief Set node associated with this stack.
   * \param node node to set
   */
  void SetNode (Ptr<Node> node);

  virtual void Insert (Ptr<IpL4Protocol> protocol);
  virtual void Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual void Remove (Ptr<IpL4Protocol> protocol);
  virtual void Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber) const;
  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber, int32_t interfaceIndex) const;

  /**
   * \brief Create raw IPv6 socket.
   * \return newly raw socket
   */
  Ptr<Socket> CreateRawSocket ();

  /**
   * \brief Remove raw IPv6 socket.
   * \param socket socket to remove
   */
  void DeleteRawSocket (Ptr<Socket> socket);

  /**
   * \brief Set the default TTL.
   * \param ttl TTL to set
   */
  void SetDefaultTtl (uint8_t ttl);

  /**
   * \brief Set the default TCLASS.
   * \param tclass TCLASS to set
   */
  void SetDefaultTclass (uint8_t tclass);

  /**
   * \brief Receive method when a packet arrive in the stack.
   * This method removes IPv6 header and forward up to L4 protocol.
   *
   * \param device network device
   * \param p the packet
   * \param protocol next header value
   * \param from address of the correspondent
   * \param to address of the destination
   * \param packetType type of the packet
   */
  void Receive (Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from, const Address &to, NetDevice::PacketType packetType);

  virtual void Send (Ptr<Packet> packet, Ipv6Address source, Ipv6Address destination, uint8_t protocol, Ptr<Ipv6Route> route);

  /**
   * \brief Set routing protocol for this stack.
   * \param routingProtocol IPv6 routing protocol to set
   */
  void SetRoutingProtocol (Ptr<Ipv6RoutingProtocol> routingProtocol);

  /**
   * \brief Get current routing protocol used.
   * \return routing protocol
   */
  Ptr<Ipv6RoutingProtocol> GetRoutingProtocol () const;

  /**
   * \brief Add IPv6 interface for a device.
   * \param device net device
   * \return interface index
   */
  uint32_t AddInterface (Ptr<NetDevice> device);

  /**
   * \brief Get an interface.
   * \param i interface index
   * \return IPv6 interface pointer
   */
  Ptr<Ipv6Interface> GetInterface (uint32_t i) const;

  /**
   * \brief Get current number of interface on this stack.
   * \return number of interface registered
   */
  uint32_t GetNInterfaces () const;

  /**
   * \brief Get interface index which has specified IPv6 address
   * \param addr IPv6 address
   * \return interface index or -1 if not found
   */
  int32_t GetInterfaceForAddress (Ipv6Address addr) const;

  /**
   * \brief Get interface index which match specified address/prefix.
   * \param addr IPv6 address
   * \param mask IPv6 prefix (mask)
   * \return interface index or -1 if not found
   */
  int32_t GetInterfaceForPrefix (Ipv6Address addr, Ipv6Prefix mask) const;

  /**
   * \brief Get interface index which is on a specified net device.
   * \param device net device
   * \returns the interface index
   */
  int32_t GetInterfaceForDevice (Ptr<const NetDevice> device) const;

  /**
   * \brief Add an address on interface.
   * \param i interface index
   * \param address address to add
   * \param addOnLinkRoute add on-link route to the network (default true)
   * \returns true if the operation succeeded
   */
  bool AddAddress (uint32_t i, Ipv6InterfaceAddress address, bool addOnLinkRoute = true);

  /**
   * \brief Get an address.
   * \param interfaceIndex interface index
   * \param addressIndex address index on the interface
   * \return Ipv6InterfaceAddress or assert if not found
   */
  Ipv6InterfaceAddress GetAddress (uint32_t interfaceIndex, uint32_t addressIndex) const;

  /**
   * \brief Get number of address for an interface.
   * \param interface interface index
   * \return number of address
   */
  uint32_t GetNAddresses (uint32_t interface) const;

  /**
   * \brief Remove an address from an interface.
   * \param interfaceIndex interface index
   * \param addressIndex address index on the interface
   * \returns true if the operation succeeded
   */
  bool RemoveAddress (uint32_t interfaceIndex, uint32_t addressIndex);

  /**
   * \brief Remove a specified Ipv6 address from an interface.
   * \param interfaceIndex interface index
   * \param address Ipv6Address to be removed from the interface
   * \returns true if the operation succeeded
   */
  bool RemoveAddress (uint32_t interfaceIndex, Ipv6Address address);

  /**
   * \brief Set metric for an interface.
   * \param i index
   * \param metric
   */
  void SetMetric (uint32_t i, uint16_t metric);

  /**
   * \brief Get metric for an interface.
   * \param i index
   * \return metric
   */
  uint16_t GetMetric (uint32_t i) const;

  /**
   * \brief Get MTU for an interface.
   * \param i index
   * \return MTU
   */
  uint16_t GetMtu (uint32_t i) const;

  /**
   * \brief Set the Path MTU for the specified IPv6 destination address.
   * \param dst Ipv6 destination address
   * \param pmtu the Path MTU
   */
  virtual void SetPmtu (Ipv6Address dst, uint32_t pmtu);

  /**
   * \brief Is specified interface up ?
   * \param i interface index
   * \returns true if the interface is up
   */
  bool IsUp (uint32_t i) const;

  /**
   * \brief Set an interface up.
   * \param i interface index
   */
  void SetUp (uint32_t i);

  /**
   * \brief set an interface down.
   * \param i interface index
   */
  void SetDown (uint32_t i);

  /**
   * \brief Is interface allows forwarding ?
   * \param i interface index
   * \returns true if the interface is forwarding
   */
  bool IsForwarding (uint32_t i) const;

  /**
   * \brief Enable or disable forwarding on interface
   * \param i interface index
   * \param val true = enable forwarding, false = disable
   */
  void SetForwarding (uint32_t i, bool val);

  Ipv6Address SourceAddressSelection (uint32_t interface, Ipv6Address dest);

  /**
   * \brief Get device by index.
   * \param i device index on this stack
   * \return NetDevice pointer
   */
  Ptr<NetDevice> GetNetDevice (uint32_t i);

  /**
   * \brief Get ICMPv6 protocol.
   * \return Icmpv6L4Protocol pointer
   */
  Ptr<Icmpv6L4Protocol> GetIcmpv6 () const;

  /**
   * \brief Add an autoconfigured address with RA information.
   * \param interface interface index
   * \param network network prefix
   * \param mask network mask
   * \param flags flags of the prefix information option (home agent, ...)
   * \param validTime valid time of the prefix
   * \param preferredTime preferred time of the prefix
   * \param defaultRouter default router address
   */
  void AddAutoconfiguredAddress (uint32_t interface, Ipv6Address network, Ipv6Prefix mask, uint8_t flags, uint32_t validTime, uint32_t preferredTime, Ipv6Address defaultRouter = Ipv6Address::GetZero ());

  /**
   * \brief Remove an autoconfigured address.
   *
   * Typically it is used when an autoconfigured address expires.
   * \param interface interface index
   * \param network network prefix
   * \param mask network mask
   * \param defaultRouter gateway
   */
  void RemoveAutoconfiguredAddress (uint32_t interface, Ipv6Address network, Ipv6Prefix mask, Ipv6Address defaultRouter);

  /**
   * \brief Register the IPv6 Extensions.
   */
  virtual void RegisterExtensions ();

  /**
   * \brief Register the IPv6 Options.
   */
  virtual void RegisterOptions ();

  /**
   * \brief Report a packet drop
   *
   * This function is used by Fragment Timeout handling to signal a fragment drop.
   *
   * \param ipHeader the IPv6 header of dropped packet
   * \param p the packet (if available)
   * \param dropReason the drop reason
   *
   */
  virtual void ReportDrop (Ipv6Header ipHeader, Ptr<Packet> p, DropReason dropReason);

  /**
   * TracedCallback signature for packet sent, forwarded or
   * local-delivered events.
   *
   * \param [in] header the Ipv6Header.
   * \param [in] packet the packet.
   * \param [in] interface the IP-level interface index
   */
  typedef void (* SentTracedCallback)
    (const Ipv6Header & header, Ptr<const Packet> packet, uint32_t interface);

  /**
   * TracedCallback signature for packet transmission or reception events.
   *
   * \param [in] packet the packet.
   * \param [in] ipv6 the Ipv6 protocol
   * \param [in] interface the IP-level interface index
   * \deprecated The non-const \c Ptr<Ipv6> argument is deprecated
   * and will be changed to \c Ptr<const Ipv6> in a future release.
   */
  typedef void (* TxRxTracedCallback)
    (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface);

  /**
   * TracedCallback signature for packet drop events.
   *
   * \param [in] header the Ipv6Header.
   * \param [in] packet the packet.
   * \param [in] reason the reason the packet was dropped.
   * \param [in] ipv6 the Ipv6 protocol
   * \param [in] interface the IP-level interface index
   * \deprecated The non-const \c Ptr<Ipv6> argument is deprecated
   * and will be changed to \c Ptr<const Ipv6> in a future release.
   */
  typedef void (* DropTracedCallback)
    (const Ipv6Header & header, Ptr<const Packet> packet,
     DropReason reason, Ptr<Ipv6> ipv6,
     uint32_t interface);

  /**
   * Adds a multicast address to the list of addresses to pass to local deliver.
   * \param address the address.
   */
  void AddMulticastAddress (Ipv6Address address);

  /**
   * Adds a multicast address to the list of addresses to pass to local deliver.
   * \param address the address.
   * \param interface the incoming interface.
   */
  void AddMulticastAddress (Ipv6Address address, uint32_t interface);

  /**
   * Removes a multicast address from the list of addresses to pass to local deliver.
   * \param address the address.
   */
  void RemoveMulticastAddress (Ipv6Address address);

  /**
   * Removes a multicast address from the list of addresses to pass to local deliver.
   * \param address the address.
   * \param interface the incoming interface.
   */
  void RemoveMulticastAddress (Ipv6Address address, uint32_t interface);

  /**
   * Checks if the address has been registered.
   * \param address the address.
   * \return true if the address is registered.
   */
  bool IsRegisteredMulticastAddress (Ipv6Address address) const;

  /**
   * Checks if the address has been registered for a specific interface.
   * \param address the address.
   * \param interface the incoming interface.
   * \return true if the address is registered.
   */
  bool IsRegisteredMulticastAddress (Ipv6Address address, uint32_t interface) const;

  /**
   * Provides reachability hint for Neighbor Cache Entries from L4-L7 protocols.
   * 
   * This function shall be called by L4-L7 protocols when an address is confirmed
   * to be reachable (i.e., at least a packet send and a reply received).
   * The net effect is to extend the NCE reachability time if the NCE is in
   * REACHABLE state, and to mark the NCE as REACHABLE if it is in STALE, PROBE, or
   * DELAY states. NCEs in INCOMPLETE state are not changed.
   * 
   * Note that the IP interface index might not be the same as the NetDevice index.
   * The correct way to check the IP interface index is by using 
   * Ipv6::GetInterfaceForDevice ().
   * 
   * \param ipInterfaceIndex IP interface index
   * \param address reachable address
   * \return true if the NCE has been successfully updated.
   */
  bool ReachabilityHint (uint32_t ipInterfaceIndex, Ipv6Address address);

protected:
  /**
   * \brief Dispose object.
   */
  virtual void DoDispose ();

  /**
   * \brief Notify other components connected to the node that a new stack member is now connected.
   *
   * This will be used to notify Layer 3 protocol of layer 4 protocol stack to connect them together.
   */
  virtual void NotifyNewAggregate ();

private:
  /**
   * \brief Ipv6L3ProtocolTestCase test case.
   * \relates Ipv6L3ProtocolTestCase
   */
  friend class ::Ipv6L3ProtocolTestCase;
  /**
   * \brief Ipv6ExtensionLooseRouting.
   * \relates Ipv6ExtensionLooseRouting
   */
  friend class Ipv6ExtensionLooseRouting;

  /**
   * \brief Container of the IPv6 Interfaces.
   */
  typedef std::vector<Ptr<Ipv6Interface> > Ipv6InterfaceList;

  /**
   * \brief Container of NetDevices registered to IPv6 and their interface indexes.
   */
  typedef std::map<Ptr<const NetDevice>, uint32_t > Ipv6InterfaceReverseContainer;

  /**
   * \brief Container of the IPv6 Raw Sockets.
   */
  typedef std::list<Ptr<Ipv6RawSocketImpl> > SocketList;

  /**
   * \brief Container of the IPv6 L4 keys: protocol number, interface index
   */
  typedef std::pair<int, int32_t> L4ListKey_t;

  /**
   * \brief Container of the IPv6 L4 instances.
   */
  typedef std::map<L4ListKey_t, Ptr<IpL4Protocol> > L4List_t;

  /**
   * \brief Container of the IPv6 Autoconfigured addresses.
   */
  typedef std::list< Ptr<Ipv6AutoconfiguredPrefix> > Ipv6AutoconfiguredPrefixList;

  /**
   * \brief Iterator of the container of the IPv6 Autoconfigured addresses.
   */
  typedef std::list< Ptr<Ipv6AutoconfiguredPrefix> >::iterator Ipv6AutoconfiguredPrefixListI;

  /**
   * \brief Make a copy of the packet, add the header and invoke the TX trace callback
   * \param ipHeader the IP header that will be added to the packet
   * \param packet the packet
   * \param ipv6 the Ipv6 protocol
   * \param interface the IP-level interface index
   *
   * Note: If the TracedCallback API ever is extended, we could consider
   * to check for connected functions before adding the header
   */
  void CallTxTrace (const Ipv6Header & ipHeader, Ptr<Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface);

  /**
   * \brief Callback to trace TX (transmission) packets.
   * \deprecated The non-const \c Ptr<Ipv6> argument is deprecated
   * and will be changed to \c Ptr<const Ipv6> in a future release.
   */ 
  TracedCallback<Ptr<const Packet>, Ptr<Ipv6>, uint32_t> m_txTrace;

  /**
   * \brief Callback to trace RX (reception) packets.
   * \deprecated The non-const \c Ptr<Ipv6> argument is deprecated
   * and will be changed to \c Ptr<const Ipv6> in a future release.
   */ 
  TracedCallback<Ptr<const Packet>, Ptr<Ipv6>, uint32_t> m_rxTrace;

  /**
   * \brief Callback to trace drop packets.
   * \deprecated The non-const \c Ptr<Ipv6> argument is deprecated
   * and will be changed to \c Ptr<const Ipv6> in a future release.
   */ 
  TracedCallback<const Ipv6Header &, Ptr<const Packet>, DropReason, Ptr<Ipv6>, uint32_t> m_dropTrace;

  /// Trace of sent packets
  TracedCallback<const Ipv6Header &, Ptr<const Packet>, uint32_t> m_sendOutgoingTrace;
  /// Trace of unicast forwarded packets
  TracedCallback<const Ipv6Header &, Ptr<const Packet>, uint32_t> m_unicastForwardTrace;
  /// Trace of locally delivered packets
  TracedCallback<const Ipv6Header &, Ptr<const Packet>, uint32_t> m_localDeliverTrace;

  /**
   * \brief Construct an IPv6 header.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param protocol L4 protocol
   * \param payloadSize payload size
   * \param hopLimit Hop limit
   * \param tclass Tclass
   * \return newly created IPv6 header
   */
  Ipv6Header BuildHeader (Ipv6Address src, Ipv6Address dst, uint8_t protocol,
                          uint16_t payloadSize, uint8_t hopLimit, uint8_t tclass);

  /**
   * \brief Send packet with route.
   * \param route route 
   * \param packet packet to send
   * \param ipHeader IPv6 header to add to the packet
   */
  void SendRealOut (Ptr<Ipv6Route> route, Ptr<Packet> packet, Ipv6Header const& ipHeader);

  /**
   * \brief Forward a packet.
   * \param idev Pointer to ingress network device
   * \param rtentry route 
   * \param p packet to forward
   * \param header IPv6 header to add to the packet
   */
  void IpForward (Ptr<const NetDevice> idev, Ptr<Ipv6Route> rtentry, Ptr<const Packet> p, const Ipv6Header& header);

  /**
   * \brief Forward a multicast packet.
   * \param idev Pointer to ingress network device
   * \param mrtentry route 
   * \param p packet to forward
   * \param header IPv6 header to add to the packet
   */
  void IpMulticastForward (Ptr<const NetDevice> idev, Ptr<Ipv6MulticastRoute> mrtentry, Ptr<const Packet> p, const Ipv6Header& header);

  /**
   * \brief Deliver a packet.
   * \param p packet delivered
   * \param ip IPv6 header
   * \param iif input interface packet was received
   */
  void LocalDeliver (Ptr<const Packet> p, Ipv6Header const& ip, uint32_t iif);

  /**
   * \brief Fallback when no route is found.
   * \param p packet
   * \param ipHeader IPv6 header
   * \param sockErrno error number
   */
  void RouteInputError (Ptr<const Packet> p, const Ipv6Header& ipHeader, Socket::SocketErrno sockErrno);

  /**
   * \brief Add an IPv6 interface to the stack.
   * \param interface interface to add
   * \return index of newly added interface
   */
  uint32_t AddIpv6Interface (Ptr<Ipv6Interface> interface);

  /**
   * \brief Setup loopback interface.
   */
  void SetupLoopback ();

  /**
   * \brief Set IPv6 forwarding state.
   * \param forward IPv6 forwarding enabled or not
   */
  virtual void SetIpForward (bool forward);

  /**
   * \brief Get IPv6 forwarding state.
   * \return forwarding state (enabled or not)
   */
  virtual bool GetIpForward () const;

  /**
   * \brief Set IPv6 MTU discover state.
   * \param mtuDiscover IPv6 MTU discover enabled or not
   */
  virtual void SetMtuDiscover (bool mtuDiscover);

  /**
   * \brief Get IPv6 MTU discover state.
   * \return MTU discover state (enabled or not)
   */
  virtual bool GetMtuDiscover (void) const;

  /**
   * \brief Set the ICMPv6 Redirect sending state.
   * \param sendIcmpv6Redirect ICMPv6 Redirect sending enabled or not
   */
  virtual void SetSendIcmpv6Redirect (bool sendIcmpv6Redirect);

  /**
   * \brief Get the ICMPv6 Redirect sending state.
   * \return ICMPv6 Redirect sending state (enabled or not)
   */
  virtual bool GetSendIcmpv6Redirect () const;

  /**
   * \brief Node attached to stack.
   */
  Ptr<Node> m_node;

  /**
   * \brief Forwarding packets (i.e. router mode) state.
   */
  bool m_ipForward;

  /**
   * \brief MTU Discover (i.e. Path MTU) state.
   */
  bool m_mtuDiscover;

  /**
   * \brief Path MTU Cache.
   */
  Ptr<Ipv6PmtuCache> m_pmtuCache;

  /**
   * \brief List of transport protocol.
   */
  L4List_t m_protocols;

  /**
   * \brief List of IPv6 interfaces.
   */
  Ipv6InterfaceList m_interfaces;

  /**
   * Container of NetDevice / Interface index associations.
   */
  Ipv6InterfaceReverseContainer m_reverseInterfacesContainer;

  /**
   * \brief Number of IPv6 interfaces managed by the stack.
   */
  uint32_t m_nInterfaces;

  /**
   * \brief Default TTL for outgoing packets.
   */
  uint8_t m_defaultTtl;

  /**
   * \brief Default TCLASS for outgoing packets.
   */
  uint8_t m_defaultTclass;

  /**
   * \brief Rejects packets directed to an interface with wrong address (\RFC{1222}).
   */
  bool m_strongEndSystemModel;

  /**
   * \brief Routing protocol.
   */
  Ptr<Ipv6RoutingProtocol> m_routingProtocol;

  /**
   * \brief List of IPv6 raw sockets.
   */
  SocketList m_sockets;

  /**
   * \brief List of IPv6 prefix received from RA.
   */
  Ipv6AutoconfiguredPrefixList m_prefixes;

  /**
   * \brief Allow ICMPv6 Redirect sending state
   */
  bool m_sendIcmpv6Redirect;

  /**
   * \brief IPv6 multicast addresses / interface key.
   */
  typedef std::pair<Ipv6Address, uint64_t> Ipv6RegisteredMulticastAddressKey_t;

  /**
   * \brief Container of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6RegisteredMulticastAddressKey_t, uint32_t> Ipv6RegisteredMulticastAddress_t;

  /**
   * \brief Container Iterator of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6RegisteredMulticastAddressKey_t, uint32_t>::iterator Ipv6RegisteredMulticastAddressIter_t;

  /**
   * \brief Container Const Iterator of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6RegisteredMulticastAddressKey_t, uint32_t>::const_iterator Ipv6RegisteredMulticastAddressCIter_t;

  /**
   * \brief Container of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6Address, uint32_t> Ipv6RegisteredMulticastAddressNoInterface_t;

  /**
   * \brief Container Iterator of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6Address, uint32_t>::iterator Ipv6RegisteredMulticastAddressNoInterfaceIter_t;

  /**
   * \brief Container Const Iterator of the IPv6 multicast addresses.
   */
  typedef std::map<Ipv6Address, uint32_t>::const_iterator Ipv6RegisteredMulticastAddressNoInterfaceCIter_t;

  /**
   * \brief List of multicast IP addresses of interest, divided per interface.
   */
  Ipv6RegisteredMulticastAddress_t m_multicastAddresses;

  /**
   * \brief List of multicast IP addresses of interest for all the interfaces.
   */
  Ipv6RegisteredMulticastAddressNoInterface_t m_multicastAddressesNoInterface;
};

} /* namespace ns3 */

#endif /* IPV6_L3_PROTOCOL_H */

