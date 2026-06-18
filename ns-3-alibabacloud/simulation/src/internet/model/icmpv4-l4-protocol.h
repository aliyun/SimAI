/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#ifndef ICMPV4_L4_PROTOCOL_H
#define ICMPV4_L4_PROTOCOL_H

#include "ip-l4-protocol.h"
#include "icmpv4.h"
#include "ns3/ipv4-address.h"

namespace ns3 {

class Node;
class Ipv4Interface;
class Ipv4Route;

/**
 * \ingroup ipv4
 * \defgroup icmp ICMP protocol and associated headers.
 */

/**
 * \ingroup icmp
 *
 * \brief This is the implementation of the ICMP protocol as
 * described in \RFC{792}.
 */
class Icmpv4L4Protocol : public IpL4Protocol
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  static const uint8_t PROT_NUMBER; //!< ICMP protocol number (0x1)

  Icmpv4L4Protocol ();
  virtual ~Icmpv4L4Protocol ();

  /**
   * \brief Set the node the protocol is associated with.
   * \param node the node
   */
  void SetNode (Ptr<Node> node);

  /**
   * Get the protocol number
   * \returns the protocol number
   */
  static uint16_t GetStaticProtocolNumber (void);

  /**
   * Get the protocol number
   * \returns the protocol number
   */
  virtual int GetProtocolNumber (void) const;

  /**
   * \brief Receive method.
   * \param p the packet
   * \param header the IPv4 header
   * \param incomingInterface the interface from which the packet is coming
   * \returns the receive status
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv4Header const &header,
                                               Ptr<Ipv4Interface> incomingInterface);

  /**
   * \brief Receive method.
   * \param p the packet
   * \param header the IPv6 header
   * \param incomingInterface the interface from which the packet is coming
   * \returns the receive status
   */
  virtual enum IpL4Protocol::RxStatus Receive (Ptr<Packet> p,
                                               Ipv6Header const &header,
                                               Ptr<Ipv6Interface> incomingInterface);

  /**
   * \brief Send a Destination Unreachable - Fragmentation needed ICMP error
   * \param header the original IP header
   * \param orgData the original packet
   * \param nextHopMtu the next hop MTU
   */
  void SendDestUnreachFragNeeded (Ipv4Header header, Ptr<const Packet> orgData, uint16_t nextHopMtu);

  /**
   * \brief Send a Time Exceeded ICMP error
   * \param header the original IP header
   * \param orgData the original packet
   * \param isFragment true if the opcode must be FRAGMENT_REASSEMBLY
   */
  void SendTimeExceededTtl (Ipv4Header header, Ptr<const Packet> orgData, bool isFragment);

  /**
   * \brief Send a Time Exceeded ICMP error
   * \param header the original IP header
   * \param orgData the original packet
   */
  void SendDestUnreachPort (Ipv4Header header, Ptr<const Packet> orgData);

  // From IpL4Protocol
  virtual void SetDownTarget (IpL4Protocol::DownTargetCallback cb);
  virtual void SetDownTarget6 (IpL4Protocol::DownTargetCallback6 cb);
  // From IpL4Protocol
  virtual IpL4Protocol::DownTargetCallback GetDownTarget (void) const;
  virtual IpL4Protocol::DownTargetCallback6 GetDownTarget6 (void) const;
protected:
  /*
   * This function will notify other components connected to the node that a new stack member is now connected
   * This will be used to notify Layer 3 protocol of layer 4 protocol stack to connect them together.
   */
  virtual void NotifyNewAggregate ();
private:
  /**
   * \brief Handles an incoming ICMP Echo packet
   * \param p the packet
   * \param header the IP header
   * \param source the source address
   * \param destination the destination address
   */
  void HandleEcho (Ptr<Packet> p,
                   Icmpv4Header header,
                   Ipv4Address source,
                   Ipv4Address destination);
  /**
   * \brief Handles an incoming ICMP Destination Unreachable packet
   * \param p the packet
   * \param header the IP header
   * \param source the source address
   * \param destination the destination address
   */
  void HandleDestUnreach (Ptr<Packet> p,
                          Icmpv4Header header,
                          Ipv4Address source,
                          Ipv4Address destination);
  /**
   * \brief Handles an incoming ICMP Time Exceeded packet
   * \param p the packet
   * \param icmp the ICMP header
   * \param source the source address
   * \param destination the destination address
   */
  void HandleTimeExceeded (Ptr<Packet> p,
                           Icmpv4Header icmp,
                           Ipv4Address source,
                           Ipv4Address destination);
  /**
   * \brief Send an ICMP Destination Unreachable packet
   *
   * \param header the original IP header
   * \param orgData the original packet
   * \param code the ICMP code
   * \param nextHopMtu the next hop MTU
   */
  void SendDestUnreach (Ipv4Header header, Ptr<const Packet> orgData, 
                        uint8_t code, uint16_t nextHopMtu);
  /**
   * \brief Send a generic ICMP packet
   *
   * \param packet the packet
   * \param dest the destination
   * \param type the ICMP type
   * \param code the ICMP code
   */
  void SendMessage (Ptr<Packet> packet, Ipv4Address dest, uint8_t type, uint8_t code);
  /**
   * \brief Send a generic ICMP packet
   *
   * \param packet the packet
   * \param source the source
   * \param dest the destination
   * \param type the ICMP type
   * \param code the ICMP code
   * \param route the route to be used
   */
  void SendMessage (Ptr<Packet> packet, Ipv4Address source, Ipv4Address dest, uint8_t type, uint8_t code, Ptr<Ipv4Route> route);
  /**
   * \brief Forward the message to an L4 protocol
   *
   * \param source the source
   * \param icmp the ICMP header
   * \param info info data (e.g., the target MTU)
   * \param ipHeader the IP header carried by ICMP
   * \param payload payload chunk carried by ICMP
   */
  void Forward (Ipv4Address source, Icmpv4Header icmp,
                uint32_t info, Ipv4Header ipHeader,
                const uint8_t payload[8]);

  virtual void DoDispose (void);

  Ptr<Node> m_node; //!< the node this protocol is associated with
  IpL4Protocol::DownTargetCallback m_downTarget; //!< callback to Ipv4::Send
};

} // namespace ns3

#endif /* ICMPV4_L4_PROTOCOL_H */
