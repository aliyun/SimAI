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
 * \brief This is a test for intermediate reply and saving routing
 * information about neighbour. 4 stations and 3 UDP ping streams are initiated.
 * \verbatim
 *  <-----------|----------->   Broadcast frame
 *              |----------->|  Unicast frame
 * (Node ID)   0           1             2            3
 * (MAC addr) 10           11           12           13
 *             |            |<-----------|----------->|             ARP request (12 asks who has 10)
 *             |            |            |<-----------|-----------> ARP request
 *             |<-----------|----------->|            |             ARP request
 * <-----------|----------->|            |            |             PREQ
 *             |<-----------|----------->|            |             PREQ
 *             |            |<-----------|            |             PREP
 *             |<-----------|            |            |             PREP
 *             |----------->|            |            |             ARP reply
 *             |            |----------->|            |             ARP REPLY
 *             |            |<-----------|            |             Data
 *             |............|............|............|
 *             |<-----------|----------->|            |             ARP request (11 asks who has 10)
 *             |............|............|............|
 *             |----------->|            |            |             ARP reply
 *                  ^ Note, that this arp reply goes without route
 *                  discovery procedure, because route is known from
 *                  previous PREQ/PREP exchange
 *             |<-----------|            |            |             DATA
 *             |............|............|............|
 * <-----------|----------->|            |            |             ARP request (10 asks who has 13)
 *             |............|............|............|
 *             |            |            |<-----------|-----------> PREQ (13 asks about 10) DO=0 RF=1
 *             |            |            |----------->|             PREP (intermediate reply - 12 knows about 10)
 *             |            |<-----------|----------->|             PREQ DO=1 RF=0
 *             |............|............|............|
 *             |----------->|            |            |             PREP
 *             |            |----------->|            |             PREP
 *             |            |            |----------->|             PREP
 * \endverbatim
 */
class HwmpDoRfRegressionTest : public TestCase
{
public:
  HwmpDoRfRegressionTest ();
  virtual ~HwmpDoRfRegressionTest();

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
  /// Reset position function
  void ResetPosition ();

  /// Server-side socket
  Ptr<Socket> m_serverSocketA;
  /// Server-side socket
  Ptr<Socket> m_serverSocketB;
  /// Client-side socket
  Ptr<Socket> m_clientSocketA;
  /// Client-side socket
  Ptr<Socket> m_clientSocketB;
  /// Client-side socket
  Ptr<Socket> m_clientSocketC;

  /// sent packets counter A
  uint32_t m_sentPktsCounterA;
  /// sent packets counter B
  uint32_t m_sentPktsCounterB;
  /// sent packets counter C
  uint32_t m_sentPktsCounterC;

  /**
   * Send data A
   * \param socket the sending socket
   */
  void SendDataA (Ptr<Socket> socket);

  /**
   * Send data B
   * \param socket the sending socket
   */
  void SendDataB (Ptr<Socket> socket);

  /**
   * Send data C
   * \param socket the sending socket
   */
  void SendDataC (Ptr<Socket> socket);

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

