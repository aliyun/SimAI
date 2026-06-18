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
 *         David Gross <gdavid.devel@gmail.com>
 *         Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 */

#ifndef ICMPV6_L4_PROTOCOL_H
#define ICMPV6_L4_PROTOCOL_H

#include <list>

#include "ns3/ipv6-address.h"
#include "ns3/random-variable-stream.h"

#include "icmpv6-header.h"
#include "ip-l4-protocol.h"
#include "ndisc-cache.h"

namespace ns3 {

class NetDevice;
class Node;
class Packet;
class TraceContext;

/**
 * \ingroup ipv6
 * \defgroup icmpv6 ICMPv6 protocol and associated headers.
 */

/**
 * \ingroup icmpv6
 *
 * \brief An implementation of the ICMPv6 protocol.
 */
class Icmpv6L4Protocol : public IpL4Protocol
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief ICMPv6 protocol number (58).
   */
  static const uint8_t PROT_NUMBER;

  /**
   * \brief Neighbor Discovery node constants: max multicast solicitations.
   * \returns The max multicast solicitations number.
   */
  uint8_t GetMaxMulticastSolicit () const;

  /**
   * \brief Neighbor Discovery node constants: max unicast solicitations.
   * \returns The max unicast solicitations number.
   */
  uint8_t GetMaxUnicastSolicit () const;

  /**
   * \brief Neighbor Discovery node constants: reachable time.
   * \returns The Reachable time for an Neighbor cache entry.
   */
  Time GetReachableTime () const;

  /**
   * \brief Neighbor Discovery node constants: retransmission timer.
   * \returns The Retransmission time for an Neighbor cache entry probe.
   */
  Time GetRetransmissionTime () const;

  /**
   * \brief Neighbor Discovery node constants : delay for the first probe.
   * \returns The time before a first probe for an Neighbor cache entry.
   */
  Time GetDelayFirstProbe () const;

  /**
   * \brief Get ICMPv6 protocol number.
   * \return protocol number
   */
  static uint16_t GetStaticProtocolNumber ();

  /**
   * \brief Constructor.
   */
  Icmpv6L4Protocol ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6L4Protocol ();

  /**
   * \brief Set the node.
   * \param node the node to set
   */
  void SetNode (Ptr<Node> node);

  /**
   * \brief Get the node.
   * \return node
   */
  Ptr<Node> GetNode ();

  /**
   * \brief This method is called by AggregateObject and completes the aggregation
   * by setting the node in the ICMPv6 stack and adding ICMPv6 factory to
   * IPv6 stack connected to the node.
   */
  void NotifyNewAggregate ();

  /**
   * \brief Get the protocol number.
   * \return protocol number
   */
  virtual int GetProtocolNumber () const;

  /**
   * \brief Get the version of the protocol.
   * \return version
   */
  virtual int GetVersion () const;

  /**
   * \brief Send a packet via ICMPv6, note that packet already contains ICMPv6 header.
   * \param packet the packet to send which contains ICMPv6 header
   * \param src source address
   * \param dst destination address
   * \param ttl next hop limit
   */
  void SendMessage (Ptr<Packet> packet, Ipv6Address src, Ipv6Address dst, uint8_t ttl);

  /**
   * \brief Helper function used during delayed solicitation. Calls SendMessage internally
   * \param packet the packet to send which contains ICMPv6 header
   * \param src source address
   * \param dst destination address
   * \param ttl next hop limit
   */
  void DelayedSendMessage (Ptr<Packet> packet, Ipv6Address src, Ipv6Address dst, uint8_t ttl);

  /**
   * \brief Send a packet via ICMPv6.
   * \param packet the packet to send
   * \param dst destination address
   * \param icmpv6Hdr ICMPv6 header (needed to calculate checksum
   * after source address is determined by routing stuff
   * \param ttl next hop limit
   */
  void SendMessage (Ptr<Packet> packet, Ipv6Address dst, Icmpv6Header& icmpv6Hdr, uint8_t ttl);

  /**
   * \brief Do the Duplication Address Detection (DAD).
   * It consists in sending a NS with our IPv6 as target. If
   * we received a NA with matched target address, we could not use
   * the address, else the address pass from TENTATIVE to PERMANENT.
   *
   * \param target target address
   * \param interface interface
   */
  void DoDAD (Ipv6Address target, Ptr<Ipv6Interface> interface);

  /**
   * \brief Send a Neighbor Adverstisement.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param hardwareAddress our MAC address
   * \param flags to set (4 = flag R, 2 = flag S, 3 = flag O)
   */
  void SendNA (Ipv6Address src, Ipv6Address dst, Address* hardwareAddress, uint8_t flags);

  /**
   * \brief Send a Echo Reply.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param id id of the packet
   * \param seq sequence number
   * \param data auxiliary data
   */
  void SendEchoReply (Ipv6Address src, Ipv6Address dst, uint16_t id, uint16_t seq, Ptr<Packet> data);

  /**
   * \brief Send a Neighbor Solicitation.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param target target IPv6 address
   * \param hardwareAddress our mac address
   */
  virtual void SendNS (Ipv6Address src, Ipv6Address dst, Ipv6Address target, Address hardwareAddress);

  /**
   * \brief Send an error Destination Unreachable.
   * \param malformedPacket the malformed packet
   * \param dst destination IPv6 address
   * \param code code of the error
   */
  void SendErrorDestinationUnreachable (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code);

  /**
   * \brief Send an error Too Big.
   * \param malformedPacket the malformed packet
   * \param dst destination IPv6 address
   * \param mtu the mtu
   */
  void SendErrorTooBig (Ptr<Packet> malformedPacket, Ipv6Address dst, uint32_t mtu);

  /**
   * \brief Send an error Time Exceeded.
   * \param malformedPacket the malformed packet
   * \param dst destination IPv6 address
   * \param code code of the error
   */
  void SendErrorTimeExceeded (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code);

  /**
   * \brief Send an error Parameter Error.
   * \param malformedPacket the malformed packet
   * \param dst destination IPv6 address
   * \param code code of the error
   * \param ptr byte of p where the error is located
   */
  void SendErrorParameterError (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code, uint32_t ptr);

  /**
   * \brief Send an ICMPv6 Redirection.
   * \param redirectedPacket the redirected packet
   * \param src IPv6 address to send the redirect from
   * \param dst IPv6 address to send the redirect to
   * \param redirTarget IPv6 target address for Icmpv6Redirection
   * \param redirDestination IPv6 destination address for Icmpv6Redirection
   * \param redirHardwareTarget L2 target address for Icmpv6OptionRdirected
   */
  void SendRedirection (Ptr<Packet> redirectedPacket, Ipv6Address src, Ipv6Address dst, Ipv6Address redirTarget, Ipv6Address redirDestination, Address redirHardwareTarget);

  /**
   * \brief Forge a Neighbor Solicitation.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param target target IPv6 address
   * \param hardwareAddress our mac address
   * \return NS packet (with IPv6 header)
   */
  NdiscCache::Ipv6PayloadHeaderPair ForgeNS (Ipv6Address src, Ipv6Address dst, Ipv6Address target, Address hardwareAddress);

  /**
   * \brief Forge a Neighbor Advertisement.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param hardwareAddress our mac address
   * \param flags flags (bitfield => R (4), S (2), O (1))
   * \return NA packet (with IPv6 header)
   */
  NdiscCache::Ipv6PayloadHeaderPair ForgeNA (Ipv6Address src, Ipv6Address dst, Address* hardwareAddress, uint8_t flags);

  /**
   * \brief Forge a Router Solicitation.
   * \param src source IPv6 address
   * \param dst destination IPv6 address
   * \param hardwareAddress our mac address
   * \return RS packet (with IPv6 header)
   */
  NdiscCache::Ipv6PayloadHeaderPair ForgeRS (Ipv6Address src, Ipv6Address dst, Address hardwareAddress);

  /**
   * \brief Forge an Echo Request.
   * \param src source address
   * \param dst destination address
   * \param id ID of the packet
   * \param seq sequence number
   * \param data the data
   * \return Echo Request packet (with IPv6 header)
   */
  NdiscCache::Ipv6PayloadHeaderPair ForgeEchoRequest (Ipv6Address src, Ipv6Address dst, uint16_t id, uint16_t seq, Ptr<Packet> data);

  /**
   * \brief Receive method.
   * \param p the packet
   * \param header the IPv4 header
   * \param interface the interface from which the packet is coming
   * \returns the receive status
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv4Header const &header,
                                               Ptr<Ipv4Interface> interface);

  /**
   * \brief Receive method.
   * \param p the packet
   * \param header the IPv6 header
   * \param interface the interface from which the packet is coming
   * \returns the receive status
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv6Header const &header,
                                               Ptr<Ipv6Interface> interface);

  /**
   * \brief Function called when DAD timeout.
   * \param interface the interface
   * \param addr the IPv6 address
   */
  void FunctionDadTimeout (Ipv6Interface* interface, Ipv6Address addr);

  /**
   * \brief Lookup in the ND cache for the IPv6 address
   *
   * \note Unlike other Lookup method, it does not send NS request!
   *
   * \param dst destination address
   * \param device device
   * \param cache the neighbor cache
   * \param hardwareDestination hardware address
   * \return true if the address is in the ND cache, the hardwareDestination is updated.
   */
  bool Lookup (Ipv6Address dst, Ptr<NetDevice> device, Ptr<NdiscCache> cache, Address* hardwareDestination);

  /**
   * \brief Lookup in the ND cache for the IPv6 address (similar as ARP protocol).
   *
   * It also send NS request to target and store the waiting packet.
   * \param p the packet
   * \param ipHeader IPv6 header
   * \param dst destination address
   * \param device device
   * \param cache the neighbor cache
   * \param hardwareDestination hardware address
   * \return true if the address is in the ND cache, the hardwareDestination is updated.
   */
  bool Lookup (Ptr<Packet> p, const Ipv6Header & ipHeader, Ipv6Address dst, Ptr<NetDevice> device, Ptr<NdiscCache> cache, Address* hardwareDestination);

  /**
   * \brief Send a Router Solicitation.
   * \param src link-local source address
   * \param dst destination address (usually ff02::2 i.e all-routers)
   * \param hardwareAddress link-layer address (SHOULD be included if src is not ::)
   */
  void SendRS (Ipv6Address src, Ipv6Address dst,  Address hardwareAddress);

  /**
   * \brief Create a neighbor cache.
   * \param device thet NetDevice
   * \param interface the IPv6 interface
   * \return a smart pointer of NdCache or 0 if problem
   */
  virtual Ptr<NdiscCache> CreateCache (Ptr<NetDevice> device, Ptr<Ipv6Interface> interface);

  /**
   * \brief Is the node must do DAD.
   * \return true if node has to do DAD.
   */
  bool IsAlwaysDad () const;

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

protected:
  /**
   * \brief Dispose this object.
   */
  virtual void DoDispose ();

  typedef std::list<Ptr<NdiscCache> > CacheList; //!< container of NdiscCaches

  /**
   * \brief Notify an ICMPv6 reception to upper layers (if requested).
   * \param source the ICMP source
   * \param icmp the ICMP header
   * \param info information about the ICMP
   * \param ipHeader the IP header carried by the ICMP
   * \param payload the data carried by the ICMP
   */
  void Forward (Ipv6Address source, Icmpv6Header icmp,
                uint32_t info, Ipv6Header ipHeader,
                const uint8_t payload[8]);

  /**
   * \brief Receive Neighbor Solicitation method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleNS (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Router Solicitation method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleRS (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Router Advertisement method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleRA (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Echo Request method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleEchoRequest (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Neighbor Advertisement method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleNA (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Redirection method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleRedirection (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Destination Unreachable method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleDestinationUnreachable (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Time Exceeded method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleTimeExceeded (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Packet Too Big method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandlePacketTooBig (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Receive Parameter Error method.
   * \param p the packet
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void HandleParameterError (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Link layer address option processing.
   * \param lla LLA option
   * \param src source address
   * \param dst destination address
   * \param interface the interface from which the packet is coming
   */
  void ReceiveLLA (Icmpv6OptionLinkLayerAddress lla, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface);

  /**
   * \brief Get the cache corresponding to the device.
   * \param device the device
   * \returns the NdiscCache associated with the device
   */
  Ptr<NdiscCache> FindCache (Ptr<NetDevice> device);

  // From IpL4Protocol
  virtual void SetDownTarget (IpL4Protocol::DownTargetCallback cb);
  virtual void SetDownTarget6 (IpL4Protocol::DownTargetCallback6 cb);
  // From IpL4Protocol
  virtual IpL4Protocol::DownTargetCallback GetDownTarget (void) const;
  virtual IpL4Protocol::DownTargetCallback6 GetDownTarget6 (void) const;

  /**
   * \brief Always do DAD ?
   */
  bool m_alwaysDad;

  /**
   * \brief A list of cache by device.
   */
  CacheList m_cacheList;

private:
  /**
   * \brief Neighbor Discovery node constants: max multicast solicitations.
   */
  uint8_t m_maxMulticastSolicit;

  /**
   * \brief Neighbor Discovery node constants: max unicast solicitations.
   */
  uint8_t m_maxUnicastSolicit;

  /**
   * \brief Neighbor Discovery node constants: reachable time.
   */
  Time m_reachableTime;

  /**
   * \brief Neighbor Discovery node constants: retransmission timer.
   */
  Time m_retransmissionTime;

  /**
   * \brief Neighbor Discovery node constants: delay for the first probe.
   */
  Time m_delayFirstProbe;

  /**
   * \brief The node.
   */
  Ptr<Node> m_node;

  /**
   * \brief Random jitter before sending solicitations
   */
  Ptr<RandomVariableStream> m_solicitationJitter;

  IpL4Protocol::DownTargetCallback6 m_downTarget; //!< callback to Ipv6::Send

};

} /* namespace ns3 */

#endif /* ICMPV6_L4_PROTOCOL_H */

