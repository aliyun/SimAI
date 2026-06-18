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

#ifndef IPV6_END_POINT_H
#define IPV6_END_POINT_H

#include <stdint.h>

#include "ns3/ipv6-address.h"
#include "ns3/callback.h"
#include "ns3/ipv6-header.h"
#include "ns3/net-device.h"
#include "ns3/ipv6-interface.h"

namespace ns3
{

class Header;
class Packet;

/**
 * \ingroup ipv6
 *
 * \brief A representation of an IPv6 endpoint/connection
 *
 * This class provides an Internet four-tuple (source and destination ports
 * and addresses).  These are used in the ns3::Ipv6EndPointDemux as targets
 * of lookups.  The class also has a callback for notification to higher
 * layers that a packet from a lower layer was received.  In the ns3
 * internet-stack, these notifications are automatically registered to be
 * received by the corresponding socket.
 */
class Ipv6EndPoint
{
public:
  /**
   * \brief Constructor.
   * \param addr the IPv6 address
   * \param port the port
   */
  Ipv6EndPoint (Ipv6Address addr, uint16_t port);

  ~Ipv6EndPoint ();

  /**
   * \brief Get the local address.
   * \return the local address
   */
  Ipv6Address GetLocalAddress ();

  /**
   * \brief Set the local address.
   * \param addr the address to set
   */
  void SetLocalAddress (Ipv6Address addr);

  /**
   * \brief Get the local port.
   * \return the local port
   */
  uint16_t GetLocalPort ();

  /**
   * \brief Set the local port.
   * \param port the port to set
   */
  void SetLocalPort (uint16_t port);

  /**
   * \brief Get the peer address.
   * \return the peer address
   */
  Ipv6Address GetPeerAddress ();

  /**
   * \brief Get the peer port.
   * \return the peer port
   */
  uint16_t GetPeerPort ();

  /**
   * \brief Set the peer information (address and port).
   * \param addr peer address
   * \param port peer port
   */
  void SetPeer (Ipv6Address addr, uint16_t port);

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

  /**
   * \brief Set the reception callback.
   * \param callback callback function
   */
  void SetRxCallback (Callback<void, Ptr<Packet>, Ipv6Header, uint16_t, Ptr<Ipv6Interface> > callback);

  /**
   * \brief Set the ICMP callback.
   * \param callback callback function
   */
  void SetIcmpCallback (Callback<void, Ipv6Address, uint8_t, uint8_t, uint8_t, uint32_t> callback);

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
   *
   * \param p the packet
   * \param header the packet header
   * \param port source port
   * \param incomingInterface incoming interface
   */
  void ForwardUp (Ptr<Packet> p, Ipv6Header header, uint16_t port, Ptr<Ipv6Interface> incomingInterface);

  /**
   * \brief Forward the ICMP packet to the upper level.
   *
   * Called from an L4Protocol implementation to notify an endpoint of
   * an icmp message reception.
   *
   * \param src source IPv6 address
   * \param ttl time-to-live
   * \param type ICMPv6 type
   * \param code ICMPv6 code
   * \param info ICMPv6 info
   */
  void ForwardIcmp (Ipv6Address src, uint8_t ttl, uint8_t type,
                    uint8_t code, uint32_t info);

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
  Ipv6Address m_localAddr;

  /**
   * \brief The local port.
   */
  uint16_t m_localPort;

  /**
   * \brief The peer address.
   */
  Ipv6Address m_peerAddr;

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
  Callback<void, Ptr<Packet>, Ipv6Header, uint16_t, Ptr<Ipv6Interface> > m_rxCallback;

  /**
   * \brief The ICMPv6 callback.
   */
  Callback<void, Ipv6Address, uint8_t, uint8_t, uint8_t, uint32_t> m_icmpCallback;

  /**
   * \brief The destroy callback.
   */
  Callback<void> m_destroyCallback;

  /**
   * \brief true if the endpoint can receive packets.
   */
  bool m_rxEnabled;
};

} /* namespace ns3 */

#endif /* IPV6_END_POINT_H */

