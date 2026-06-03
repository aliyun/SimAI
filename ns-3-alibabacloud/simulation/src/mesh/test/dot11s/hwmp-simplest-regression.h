/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev  <andreev@iitp.ru>
 */

#include "ns3/test.h"
#include "ns3/node-container.h"
#include "ns3/nstime.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/pcap-file.h"

using namespace ns3;

/**
 * \ingroup dot11s-test
 * \ingroup tests
 *
 * \brief Peering Management & HWM Protocol regression test
 * Initiate scenario with 2 stations. Procedure of opening peer link
 * is the following:
 * \verbatim
 *           server       client
 * <-----------|----------->   Broadcast frame
 *             |----------->|  Unicast frame
 *
 *                                        !!! PMP routines:
 * <-----------|----------->|             Beacon
 *             |----------->|             Peer Link Open frame
 *             |<-----------|             Peer Link Confirm frame
 *             |<-----------|             Peer Link Open frame 
 *             |----------->|             Peer Link Confirm frame
 *             |............|             !!! Data started:
 *             |<-----------|-----------> ARP Request (time 2)
 * <-----------|----------->|             PREQ
 *             |<-----------|             PREP
 *             |----------->|             ARP reply
 * <-----------|----------->|             ARP Request (reflooded after delay)
 *             |<-----------|             Data (first UDP datagram)
 * <-----------|----------->|             ARP Request
 *             |<-----------|             ARP reply
 *             |----------->|             Data
 *             |<-----------|-----------> ARP Request (reflooded after delay)
 *             |............|             Some other beacons
 *             |<-----------|             Data
 *             |----------->|             Data
 *             |............|             !!! Route expiration routines:
 *             |............|             !!! (after time 7)
 *             |<-----------|-----------> PREQ (route expired)
 *             |----------->|             PREP
 *             |<-----------|             Data
 *             |----------->|             Data
 *             |............|
 * \endverbatim
 * At 10 seconds stations become unreachable, so UDP client tries to
 * close peer link due to TX-fail, and UDP server tries to close peer link
 * due to beacon loss
 */
class HwmpSimplestRegressionTest : public TestCase
{
public:
  HwmpSimplestRegressionTest ();
  virtual ~HwmpSimplestRegressionTest();

  virtual void DoRun ();
  /// Check results function
  void CheckResults ();

private:
  /// \internal It is important to have pointers here
  NodeContainer * m_nodes;
  /// Simulation time
  Time m_time;
  Ipv4InterfaceContainer m_interfaces; ///< interfaces

  /// Create nodes function
  void CreateNodes ();
  /// Create devices function
  void CreateDevices ();
  /// Install application function
  void InstallApplications ();
  /// Reset position
  void ResetPosition ();

  /// Server-side socket
  Ptr<Socket> m_serverSocket;
  /// Client-side socket
  Ptr<Socket> m_clientSocket;

  /// sent packets counter
  uint32_t m_sentPktsCounter;

  /**
   * Send data
   * \param socket the sending socket
   */
  void SendData (Ptr<Socket> socket);

  /**
   * \brief Handle a packet reception.
   *
   * This function is called by lower layers.
   *
   * \param socket the socket the packet was received to.
   */
  void HandleReadServer (Ptr<Socket> socket);

  /**
   * \brief Handle a packet reception.
   *
   * This function is called by lower layers.
   *
   * \param socket the socket the packet was received to.
   */
  void HandleReadClient (Ptr<Socket> socket);
};

