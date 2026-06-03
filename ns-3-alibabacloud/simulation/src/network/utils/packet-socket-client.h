/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' di Firenze
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

#ifndef PACKET_SOCKET_CLIENT_H
#define PACKET_SOCKET_CLIENT_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/packet-socket-address.h"

namespace ns3 {

class Socket;
class Packet;

/**
 * \ingroup socket
 *
 * \brief A simple client.
 *
 * Sends packets using PacketSocket. It does not require (or use) IP.
 *
 * Packets are sent as soon as PacketSocketClient::Start is called.
 * The application has the same requirements as the PacketSocket for
 * what concerns the underlying NetDevice and the Address scheme.
 * It is meant to be used in ns-3 tests.
 *
 * The application will send `MaxPackets' packets, one every `Interval'
 * time. Packet size (`PacketSize') can be configured.
 * Provides a "Tx" Traced Callback (transmitted packets, source address).
 *
 * Note: packets larger than the NetDevice MTU will not be sent.
 */
class PacketSocketClient : public Application
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  PacketSocketClient ();

  virtual ~PacketSocketClient ();

  /**
   * \brief set the remote address and protocol to be used
   * \param addr remote address
   */
  void SetRemote (PacketSocketAddress addr);

  /**
   * \brief Query the priority value of this socket
   * \return The priority value
   */
  uint8_t GetPriority (void) const;

protected:
  virtual void DoDispose (void);

private:

  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /**
   * \brief Manually set the socket priority
   * \param priority The socket priority (in the range 0..6)
   */
  void SetPriority (uint8_t priority);

  /**
   * \brief Send a packet
   */
  void Send (void);

  uint32_t m_maxPackets; //!< Maximum number of packets the application will send
  Time m_interval;       //!< Packet inter-send time
  uint32_t m_size;       //!< Size of the sent packet
  uint8_t m_priority;    //!< Priority of the sent packets

  uint32_t m_sent;       //!< Counter for sent packets
  Ptr<Socket> m_socket;  //!< Socket
  PacketSocketAddress m_peerAddress; //!< Remote peer address
  bool m_peerAddressSet; //!< Sanity check
  EventId m_sendEvent;   //!< Event to send the next packet

  /// Traced Callback: sent packets, source address.
  TracedCallback<Ptr<const Packet>, const Address &> m_txTrace;
};

} // namespace ns3

#endif /* PACKET_SOCKET_CLIENT_H */
