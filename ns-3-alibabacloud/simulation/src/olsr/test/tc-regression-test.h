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
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

#ifndef TC_REGRESSION_TEST_H
#define TC_REGRESSION_TEST_H

#include "ns3/test.h"
#include "ns3/nstime.h"
#include "ns3/socket.h"
#include "ns3/ipv4-raw-socket-impl.h"
#include "ns3/node-container.h"

namespace ns3 {
namespace olsr {
/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * \brief Less trivial test of OLSR Topology Control message generation
 *
 * This test simulates 3 Wi-Fi stations with chain topology and runs OLSR without any extra traffic.
 * It is expected that only second station will send TC messages.
 *
 * Expected trace (20 seconds, note random b-cast jitter):
   \verbatim
         1       2       3
         |<------|------>|              HELLO (empty) src = 10.1.1.2
         |       |<------|------>       HELLO (empty) src = 10.1.1.3
  <------|------>|       |              HELLO (empty) src = 10.1.1.1
  <------|------>|       |              HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<------|------>|              HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.3; Link Type: Asymmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
         |<------|------>|              HELLO (Link Type: Asymmetric, Neighbor: 10.1.1.3; Link Type: Asymmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
  <------|------>|       |              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: Symmetric, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<======|======>|              TC (10.1.1.3; 10.1.1.1) + HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
         |<======|======>|              TC (10.1.1.3; 10.1.1.1) src = 10.1.1.2
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
  <------|------>|       |              HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.1
         |       |<------|------>       HELLO (Link Type: MPR Link, Neighbor: 10.1.1.2) src = 10.1.1.3
         |<------|------>|              HELLO (Link Type: Symmetric, Neighbor: 10.1.1.3; Link Type: Symmetric, Neighbor: 10.1.1.1) src = 10.1.1.2
   \endverbatim
 */
class TcRegressionTest : public TestCase
{
public:
  TcRegressionTest ();
  ~TcRegressionTest ();
private:
  /// Total simulation time
  const Time m_time;
  /// Create & configure test network
  void CreateNodes ();
  void DoRun ();

  /**
   * Receive raw data on node A
   * \param socket receiving socket
   */
  void ReceivePktProbeA (Ptr<Socket> socket);
  /// Packet counter on node A
  uint8_t m_countA;
  /// Receiving socket on node A
  Ptr<Ipv4RawSocketImpl> m_rxSocketA;

  /**
   * Receive raw data on node B
   * \param socket receiving socket
   */
  void ReceivePktProbeB (Ptr<Socket> socket);
  /// Packet counter on node B
  uint8_t m_countB;
  /// Receiving socket on node B
  Ptr<Ipv4RawSocketImpl> m_rxSocketB;

  /**
   * Receive raw data on node C
   * \param socket receiving socket
   */
  void ReceivePktProbeC (Ptr<Socket> socket);
  /// Packet counter on node C
  uint8_t m_countC;
  /// Receiving socket on node C
  Ptr<Ipv4RawSocketImpl> m_rxSocketC;

};

}
}

#endif /* TC_REGRESSION_TEST_H */
