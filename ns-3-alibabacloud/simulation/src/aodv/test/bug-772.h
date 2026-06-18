/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 IITP RAS
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
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

#ifndef BUG_772_H
#define BUG_772_H

#include "ns3/test.h"
#include "ns3/nstime.h"
#include "ns3/node-container.h"
#include "ns3/socket.h"

using namespace ns3;

/**
 * \ingroup aodv
 *
 * \brief AODV deferred route lookup test case (see \bugid{772})
 *
 * UDP packet transfers are delayed while a route is found and then while
 * ARP completes.  Eight packets should be sent, queued until the path
 * becomes functional, and then delivered.
 */
class Bug772ChainTest : public TestCase
{
public:
  /**
   * Create test case
   *
   * \param prefix              Unique file names prefix
   * \param proto               ns3::UdpSocketFactory or ns3::TcpSocketFactory
   * \param size                Number of nodes in the chain
   * \param time                Simulation time
   */
  Bug772ChainTest (const char * const prefix, const char * const proto, Time time, uint32_t size);
  ~Bug772ChainTest ();

private:
  /// \internal It is important to have pointers here
  NodeContainer * m_nodes;

  /// PCAP file names prefix
  const std::string m_prefix;
  /// Socket factory TID
  const std::string m_proto;
  /// Total simulation time
  const Time m_time;
  /// Chain size
  const uint32_t m_size;
  /// Chain step, meters
  const double m_step;
  /// port number
  const uint16_t m_port;

  /// Create test topology
  void CreateNodes ();
  /// Create devices, install TCP/IP stack and applications
  void CreateDevices ();
  /// Compare traces with reference ones
  void CheckResults ();
  /// Go
  void DoRun ();
  /**
   * Receive data function
   * \param socket the socket to receive from
   */
  void HandleRead (Ptr<Socket> socket);

  /// Receiving socket
  Ptr<Socket> m_recvSocket;
  /// Transmitting socket
  Ptr<Socket> m_sendSocket;

  /// Received packet count
  uint32_t m_receivedPackets;

  /**
   * Send data
   * \param socket the sending socket
   */
  void SendData (Ptr<Socket> socket);
};

#endif /* BUG_772_H */
