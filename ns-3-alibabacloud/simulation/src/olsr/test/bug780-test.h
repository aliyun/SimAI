/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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

#ifndef BUG780_TEST_H
#define BUG780_TEST_H

#include "ns3/test.h"
#include "ns3/nstime.h"
#include "ns3/node-container.h"

namespace ns3 {
namespace olsr {

/**
 * \ingroup olsr-test
 * \ingroup tests
 *
 * See \bugid{780}
 */
class Bug780Test : public TestCase
{
public:
  Bug780Test ();
  ~Bug780Test ();
private:
  /// Total simulation time
  const Time m_time;
  /// Create & configure test network
  void CreateNodes ();
  void DoRun ();
  /// Send one ping
  void SendPing ();
  /**
   * Receive echo reply
   * \param socket the socket
   */
  void Receive (Ptr<Socket> socket);
  /// Socket
  Ptr<Socket> m_socket;
  /// Sequence number
  uint16_t m_seq;
  /// Received ECHO Reply counter
  uint16_t m_recvCount;
};

}
}

#endif /* BUG780_TEST_H */
