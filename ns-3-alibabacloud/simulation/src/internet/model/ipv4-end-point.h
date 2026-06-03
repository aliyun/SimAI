/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef IPV4_END_POINT_H
#define IPV4_END_POINT_H

#include <stdint.h>
#include "ns3/ipv4-address.h"
#include "ns3/callback.h"
#include "ns3/net-device.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-interface.h"

namespace ns3 {

class Header;
class Packet;

/**
 * \ingroup ipv4
 *
 * \brief A representation of an internet endpoint/connection
 *
 * This class provides an internet four-tuple (source and destination ports 
 * and addresses).  These are used in the ns3::Ipv4EndPointDemux as targets
 * of lookups.  The class also has a callback for notification to higher
 * layers that a packet from a lower layer was received.  In the ns3
 * internet-stack, these notifications are automatically registered to be
 * received by the corresponding socket.
 */

class Ipv4EndPoint {
public:
  /**
   * \brief Constructor.
   * \param address the IPv4 address
   * \param port the port
   */
  Ipv4EndPoint (Ipv4Address address, uint16_t port);
  ~Ipv4EndPoint ();

  /**
   * \brief Get the local address.
   * \return the local address
   */
  Ipv4Address GetLocalAddress (void);

  /**
   * \brief Set the local address.
   * \param address the address to set
   */
  void SetLocalAddress (Ipv4Address address);

  /**
   * \brief Get the local port.
   * \return the local port
   */
  uint16_t GetLocalPort (void);

  /**
   * \brief Get the peer address.
   * \return the peer address
   */
  Ipv4Address GetPeerAddress (void);

  /**
   * \brief Get the peer port.
   * \return the peer port
   */
  uint16_t GetPeerPort (void);

  /**
   * \brief Set the peer information (address and port).
   * \param address peer address
   * \param port peer port
   */
  void SetPeer (Ipv4Address address, uint16_t port);

  /**
   * \brief Bind a socket to specific device.
   *
   * This method corresponds to using setsockopt() SO_BINDTODEVICE
   * of real network or BSD sockets.   If set on a socket, this option will
   * force packets to leave the bound device regardless of the device that
   * IP routing would naturally choose.  In the receive direction, only
   * packets received from the bound interface will be delivered.
   *
   * This option has no particular relationship to binding sockets to
   * an address via Socket::Bind ().  It is possible to bind sockets to a
   * specific IP address on the bound interface by calling both
   * Socket::Bind (address) and Socket::BindToNetDevice (device), but it
   * is also possible to bind to mismatching device and address, even if
   * the socket can not receive any packets as a result.
   *
   * \param netdevice Pointer to Netdevice of desired interface
   */
  void BindToNetDevice (Ptr<NetDevice> netdevice);

  /**
   * \brief Returns socket's bound netdevice, if any.
   *
   * This method corresponds to using getsockopt() SO_BINDTODEVICE
   * of real network or BSD sockets.
   *
   *
   * \returns Pointer to interface.
   */
  Ptr<NetDevice> GetBoundNetDevice (void);

  // Called from socket implementations to get notified about important events.
  /**
   * \brief Set the reception callback.
   * \param callback callback function
   */
  void SetRxCallback (Callback<void,Ptr<Packet>, Ipv4Header, uint16_t, Ptr<Ipv4Interface> > callback);
  /**
   * \brief Set the ICMP callback.
   * \param callback callback function
   */
  void SetIcmpCallback (Callback<void,Ipv4Address,uint8_t,uint8_t,uint8_t,uint32_t> callback);
  /**
   * \brief Set the default destroy callback.
   * \param callback callback function
   */
  void SetDestroyCallback (Callback<void> callback);

  /**
   * \brief Forward the packet to the upper level.
   *
   * Called from an L4Protocol implementation to notify an endpoint of a
   * packet reception.
   * \param p the packet
   * \param header the packet header
   * \param sport source port
   * \param incomingInterface incoming interface
   */
  void ForwardUp (Ptr<Packet> p, const Ipv4Header& header, uint16_t sport, 
                  Ptr<Ipv4Interface> incomingInterface);

  /**
   * \brief Forward the ICMP packet to the upper level.
   *
   * Called from an L4Protocol implementation to notify an endpoint of
   * an icmp message reception.
   *
   * \param icmpSource source IP address
   * \param icmpTtl time-to-live
   * \param icmpType ICMP type
   * \param icmpCode ICMP code
   * \param icmpInfo ICMP info
   */
  void ForwardIcmp (Ipv4Address icmpSource, uint8_t icmpTtl, 
                    uint8_t icmpType, uint8_t icmpCode,
                    uint32_t icmpInfo);

  /**
   * \brief Enable or Disable the endpoint Rx capability.
   * \param enabled true if Rx is enabled
   */
  void SetRxEnabled (bool enabled);

  /**
   * \brief Checks if the endpoint can receive packets.
   * \returns true if the endpoint can receive packets.
   */
  bool IsRxEnabled (void);

private:
  /**
   * \brief The local address.
   */
  Ipv4Address m_localAddr;

  /**
   * \brief The local port.
   */
  uint16_t m_localPort;

  /**
   * \brief The peer address.
   */
  Ipv4Address m_peerAddr;

  /**
   * \brief The peer port.
   */
  uint16_t m_peerPort;

  /**
   * \brief The NetDevice the EndPoint is bound to (if any).
   */
  Ptr<NetDevice> m_boundnetdevice;

  /**
   * \brief The RX callback.
   */
  Callback<void,Ptr<Packet>, Ipv4Header, uint16_t, Ptr<Ipv4Interface> > m_rxCallback;

  /**
   * \brief The ICMPv6 callback.
   */
  Callback<void,Ipv4Address,uint8_t,uint8_t,uint8_t,uint32_t> m_icmpCallback;

  /**
   * \brief The destroy callback.
   */
  Callback<void> m_destroyCallback;

  /**
   * \brief true if the endpoint can receive packets.
   */
  bool m_rxEnabled;
};

} // namespace ns3


#endif /* IPV4_END_POINT_H */
