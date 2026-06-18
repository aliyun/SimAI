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
 * \brief There are 5 stations set into a row, the center station is root.
 * Regression test indicates that traffic goes from the first to the
 * last stations without reactive route discovery procedure
 * \verbatim
 *          Src                Root                 Dst
 * (node ID) 0         1         2         3         4
 * (MAC)   00:0b     00:0c     00:0d     00:0e     00:0f
 *           |         |<--------|-------->|         |          Proactive PREQ
 *           |         |-------->|         |         |          PREP
 *           |         |         |<--------|         |          PREP
 *           |<--------|-------->|<--------|-------->|          Proactive PREQ
 *           |-------->|         |         |<--------|          PREP
 *           |         |-------->|         |         |          PREP
 *           |         |         |<--------|         |          PREP
 *  <--------|-------->|         |         |<--------|--------> Proactive PREQ
 *  Note, that at this moment all routes are known, and no reactive
 *  path discovery procedure shall be initiated
 *           |         |         |<--------|-------->|          ARP request
 *           |.........|.........|.........|.........|
 *  <--------|-------->|         |         |         |          ARP request
 *           |-------->|         |         |         |          ARP reply
 *           |.........|.........|.........|.........|
 *           |         |         |         |-------->|          ARP reply
 *           |         |         |         |<--------|          DATA
 *                                             ^
 *               Further data is forwarded by proactive path
 * \endverbatim
 *
 */
class HwmpProactiveRegressionTest : public TestCase
{
public:
  HwmpProactiveRegressionTest ();
  virtual ~HwmpProactiveRegressionTest();

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

