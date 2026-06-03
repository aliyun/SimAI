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

#ifndef PING6_H
#define PING6_H

#include "ns3/application.h"
#include "ns3/ipv6-address.h"
#include "ns3/ipv6-l3-protocol.h"

namespace ns3
{

class Packet;
class Socket;

/**
 * \ingroup internet-apps
 * \defgroup ping6 Ping6
 */

/**
 * \ingroup ping6
 * \brief A ping6 application.
 */
class Ping6 : public Application
{
public:
  /**
   * \brief Get the type ID.
   * \return type ID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Constructor.
   */
  Ping6 ();

  /**
   * \brief Destructor.
   */
  virtual ~Ping6 ();

  /**
   * \brief Set the local address.
   * \param ipv6 the local IPv6 address
   */
  void SetLocal (Ipv6Address ipv6);

  /**
   * \brief Set the remote peer.
   * \param ipv6 IPv6 address of the peer
   */
  void SetRemote (Ipv6Address ipv6);

  /**
   * \brief Set the out interface index.
   * This is to send to link-local (unicast or multicast) address
   * when a node has multiple interfaces.
   * \param ifIndex interface index
   */
  void SetIfIndex (uint32_t ifIndex);

  /**
   * \brief Set routers for routing type 0 (loose routing).
   * \param routers routers addresses
   */
  void SetRouters (std::vector<Ipv6Address> routers);

protected:
  /**
   * \brief Dispose this object;
   */
  virtual void DoDispose ();

private:
  /**
   * \brief Start the application.
   */
  virtual void StartApplication ();

  /**
   * \brief Stop the application.
   */
  virtual void StopApplication ();

  /**
   * \brief Schedule sending a packet.
   * \param dt interval between packet
   */
  void ScheduleTransmit (Time dt);

  /**
   * \brief Send a packet.
   */
  void Send ();

  /**
   * \brief Receive method.
   * \param socket socket that receive a packet
   */
  void HandleRead (Ptr<Socket> socket);

  /**
   * \brief Peer IPv6 address.
   */
  Ipv6Address m_address;

  /**
   * \brief Number of "Echo request" packets that will be sent.
   */
  uint32_t m_count;

  /**
   * \brief Number of packets sent.
   */
  uint32_t m_sent;

  /**
   * \brief Size of the packet.
   */
  uint32_t m_size;

  /**
   * \brief Interval between packets sent.
   */
  Time m_interval;

  /**
   * \brief Local address.
   */
  Ipv6Address m_localAddress;

  /**
   * \brief IP interface index relative to the local address.
   */
  uint32_t m_ipInterfaceIndex;

  /**
   * \brief IP interface index relative to the local address.
   */
  Ptr<Ipv6L3Protocol> m_ipv6Protocol;

  /**
   * \brief Peer address.
   */
  Ipv6Address m_peerAddress;

  /**
   * \brief Local socket.
   */
  Ptr<Socket> m_socket;

  /**
   * \brief Sequence number.
   */
  uint16_t m_seq;

  /**
   * \brief Event ID.
   */
  EventId m_sendEvent;

  /**
   * \brief Out interface (i.e. for link-local communication).
   */
  uint32_t m_ifIndex;

  /**
   * \brief Routers addresses for routing type 0.
   */
  std::vector<Ipv6Address> m_routers;
};

} /* namespace ns3 */

#endif /* PING6_H */

