/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
 *               2007 INRIA
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
 * Authors: George F. Riley<riley@ece.gatech.edu>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef UDP_SOCKET_H
#define UDP_SOCKET_H

#include "ns3/socket.h"
#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/ptr.h"
#include "ns3/object.h"

namespace ns3 {

class Node;
class Packet;

/**
 * \ingroup socket
 * \ingroup udp
 *
 * \brief (abstract) base class of all UdpSockets
 *
 * This class exists solely for hosting UdpSocket attributes that can
 * be reused across different implementations, and for declaring
 * UDP-specific multicast API.
 */
class UdpSocket : public Socket
{
public:
  /**
   * Get the type ID.
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
 
  UdpSocket (void);
  virtual ~UdpSocket (void);

  /**
   * \brief Corresponds to socket option MCAST_JOIN_GROUP
   *
   * \param interface interface number, or 0
   * \param groupAddress multicast group address
   * \returns on success, zero is returned.  On error, -1 is returned,
   *          and errno is set appropriately
   *
   * Enable reception of multicast datagrams for this socket on the
   * interface number specified.  If zero is specified as
   * the interface, then a single local interface is chosen by
   * system.  In the future, this function will generate trigger IGMP
   * joins as necessary when IGMP is implemented, but for now, this
   * just enables multicast datagram reception in the system if not already
   * enabled for this interface/groupAddress combination.
   *
   * \attention IGMP is not yet implemented in ns-3
   *
   * This function may be called repeatedly on a given socket but each
   * join must be for a different multicast address, or for the same
   * multicast address but on a different interface from previous joins.
   * This enables host multihoming, and the ability to join the same 
   * group on different interfaces.
   */
  virtual int MulticastJoinGroup (uint32_t interface, const Address &groupAddress) = 0;

  /**
   * \brief Corresponds to socket option MCAST_LEAVE_GROUP
   *
   * \param interface interface number, or 0
   * \param groupAddress multicast group address
   * \returns on success, zero is returned.  On error, -1 is returned,
   *          and errno is set appropriately
   *
   * Disable reception of multicast datagrams for this socket on the
   * interface number specified.  If zero is specified as
   * the interfaceIndex, then a single local interface is chosen by
   * system.  In the future, this function will generate trigger IGMP
   * leaves as necessary when IGMP is implemented, but for now, this
   * just disables multicast datagram reception in the system if this
   * socket is the last for this interface/groupAddress combination.
   *
   * \attention IGMP is not yet implemented in ns-3
   */
  virtual int MulticastLeaveGroup (uint32_t interface, const Address &groupAddress) = 0;

private:
  // Indirect the attribute setting and getting through private virtual methods
  /**
   * \brief Set the receiving buffer size
   * \param size the buffer size
   */
  virtual void SetRcvBufSize (uint32_t size) = 0;
  /**
   * \brief Get the receiving buffer size
   * \returns the buffer size
   */
  virtual uint32_t GetRcvBufSize (void) const = 0;
  /**
   * \brief Set the IP multicast TTL
   * \param ipTtl the IP multicast TTL
   */
  virtual void SetIpMulticastTtl (uint8_t ipTtl) = 0;
  /**
   * \brief Get the IP multicast TTL
   * \returns the IP multicast TTL
   */
  virtual uint8_t GetIpMulticastTtl (void) const = 0;
  /**
   * \brief Set the IP multicast interface
   * \param ipIf the IP multicast interface
   */
  virtual void SetIpMulticastIf (int32_t ipIf) = 0;
  /**
   * \brief Get the IP multicast interface
   * \returns the IP multicast interface
   */
  virtual int32_t GetIpMulticastIf (void) const = 0;
  /**
   * \brief Set the IP multicast loop capability
   *
   * This means that the socket will receive the packets
   * sent by itself on a multicast address.
   * Equivalent to setsockopt  IP_MULTICAST_LOOP
   *
   * \param loop the IP multicast loop capability
   */
  virtual void SetIpMulticastLoop (bool loop) = 0;
  /**
   * \brief Get the IP multicast loop capability
   *
   * This means that the socket will receive the packets
   * sent by itself on a multicast address.
   * Equivalent to setsockopt  IP_MULTICAST_LOOP
   *
   * \returns the IP multicast loop capability
   */
  virtual bool GetIpMulticastLoop (void) const = 0;
  /**
   * \brief Set the MTU discover capability
   *
   * \param discover the MTU discover capability
   */
  virtual void SetMtuDiscover (bool discover) = 0;
  /**
   * \brief Get the MTU discover capability
   *
   * \returns the MTU discover capability
   */
  virtual bool GetMtuDiscover (void) const = 0;
};

} // namespace ns3

#endif /* UDP_SOCKET_H */


