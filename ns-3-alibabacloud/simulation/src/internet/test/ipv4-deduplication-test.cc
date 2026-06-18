/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
 * Copyright (c) 2019 Caliola Engineering, LLC : RFC 6621 multicast packet de-duplication
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
 * Modified (2019): Jared Dulmage <jared.dulmage@caliola.com>
 *   Tests dissemination of multicast packets across a mesh
 *   network to all nodes over multiple hops.  Tests check
 *   the number of received packets and dropped packets
 *   with RFC 6621 de-duplication enabled or disabled.
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/socket.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/config.h"

#include "ns3/names.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/random-variable-stream.h"

#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/udp-socket.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/simple-net-device-helper.h"

#include "ns3/traffic-control-layer.h"

#include <string>
#include <limits>
#include <functional>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Deduplication Test
 *
 * Tests topology:
 *
 *        /---- B ----\
 *  A ----      |       ---- D ---- E
 *        \---- C ----/
 *
 * This test case counts the number of packets received
 * and dropped at each node across the topology.  Every
 * node is configured to forward the multicast packet
 * which originates at node A.
 *
 * With RFC 6621 de-duplication enabled, one 1 packet
 * is received  while some number of duplicate relayed
 * packets are dropped by RFC 6621 at each node.
 * 
 * When RFC6621 is disabled, the original packet has TTL = 4.
 * Multiple packets are received at each node and several packets
 * are dropped due to TTL expiry at each node.
 */
class Ipv4DeduplicationTest : public TestCase
{
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void DoSendData (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param packet The packet to send.
   * \param to Destination address.
   */
  void DoSendPacket (Ptr<Socket> socket, Ptr<Packet> packet, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void SendData (Ptr<Socket> socket, std::string to);

  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param packet The packet to send.
   * \param to Destination address.
   */
  void SendPacket (Ptr<Socket> socket, Ptr<Packet> packet, std::string to);

  /**
   * \brief Check packet receptions
   * \param name Node name
   */
  void CheckPackets (const std::string &name);

  /**
   * \brief Check packet drops
   * \param name Node name
   */
  void CheckDrops (const std::string &name);

  static const Time DELAY; //!< Channel delay

  /**
   * Creates the test name according to the parameters
   * \param enable deduplication enabled
   * \param expire expiration delay for duplicate cache entries
   * \returns A string describing the test
   */
  static std::string MakeName (bool enable, Time expire);

  /**
   * Enum of test types
   */
  enum MODE {ENABLED = 0,
             DISABLED,
             DEGENERATE};  // enabled, but expiration time too low
  MODE m_mode; //!< Test type
  Time m_expire; //!< Expiration delay for duplicate cache entries
  std::map<std::string, uint32_t> m_packetCountMap; //!< map of received packets (node name, counter)
  std::map<std::string, uint32_t> m_dropCountMap; //!< map of received packets (node name, counter)

public:
  virtual void DoRun (void);

  /**
   * Constructor
   * \param enable deduplication enabled
   * \param expire expiration delay for duplicate cache entries
   */
  Ipv4DeduplicationTest (bool enable, Time expire = Seconds (1));

  /**
   * \brief Receive data.
   * \param [in] socket The receive socket.
   */
  void ReceivePkt (Ptr<Socket> socket);

  /**
   * \brief Register dropped packet.
   * \param [in] ipHeader IP header
   * \param [in] packet Packet that was dropped
   * \param [in] reason Reason for dropping packet
   * \param [in] ipv4 Ipv4 instance
   * \param [in] interface Interface number
   */
  void DropPkt (const Ipv4Header &ipHeader,
                Ptr<const Packet> packet, Ipv4L3Protocol::DropReason reason,
                Ptr<Ipv4> ipv4, uint32_t interface);
};

const Time 
Ipv4DeduplicationTest::DELAY = MilliSeconds (1);

Ipv4DeduplicationTest::Ipv4DeduplicationTest (bool enable, Time expire)
  : TestCase (MakeName (enable, expire))
  , m_mode (ENABLED)
  , m_expire (expire)
{
  if (!enable) 
    {
      m_mode = DISABLED;
    }
  else if (m_expire < DELAY)
    {
      m_mode = DEGENERATE;
    }
}

void Ipv4DeduplicationTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  Ptr<Packet> packet = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, packet->GetSize (), "Received packet size is not equal to the Rx buffer size");

  auto node = socket->GetNode ();
  std::string name = Names::FindName (node);
  m_packetCountMap.insert ({name, 0});  // only inserts when not there
  ++m_packetCountMap[name];
}

void
Ipv4DeduplicationTest::DropPkt (const Ipv4Header &ipHeader,
                                Ptr<const Packet> packet, Ipv4L3Protocol::DropReason reason,
                                Ptr<Ipv4> ipv4, uint32_t interface)
{
  switch (m_mode)
    {
      case ENABLED:
        NS_TEST_EXPECT_MSG_EQ (reason, Ipv4L3Protocol::DROP_DUPLICATE, "Wrong reason for drop");
        break;
      case DISABLED:
        NS_TEST_EXPECT_MSG_EQ (reason, Ipv4L3Protocol::DROP_TTL_EXPIRED, "Wrong reason for drop");
        break;
      case DEGENERATE:
        // reason can be either
        break;
    };
  auto node = ipv4->GetNetDevice (interface)->GetNode ();
  std::string name = Names::FindName (node);
  m_dropCountMap.insert ({name, 0});
  ++m_dropCountMap[name];
}

void
Ipv4DeduplicationTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  SendPacket (socket, Create<Packet> (123), to);
}

void
Ipv4DeduplicationTest::DoSendPacket (Ptr<Socket> socket, Ptr<Packet> packet, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (packet, 0, realTo), 123, "100");
}

void
Ipv4DeduplicationTest::SendData (Ptr<Socket> socket, std::string to)
{
  DoSendData(socket, to);
}

void
Ipv4DeduplicationTest::SendPacket (Ptr<Socket> socket, Ptr<Packet> packet, std::string to)
{
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), MilliSeconds (50),
                                  &Ipv4DeduplicationTest::DoSendPacket, this, socket, packet, to);
}

std::string
Ipv4DeduplicationTest::MakeName (bool enabled, Time expire)
{
  std::ostringstream oss;
  oss << "IP v4 RFC 6621 De-duplication: ";
  if (!enabled)
    {
      oss << "disabled";
    }
  else if (expire > DELAY)
    {
      oss << "enabled";
    }
  else
    {
      oss << "degenerate";
    }
  oss << ", expire = " << expire.ToDouble (Time::MS) << "ms";

  return oss.str ();
}

void
Ipv4DeduplicationTest::DoRun (void)
{
  // multicast target
  const std::string targetAddr = "239.192.100.1";
  Config::SetDefault ("ns3::Ipv4L3Protocol::EnableDuplicatePacketDetection", BooleanValue (m_mode != DISABLED));
  Config::SetDefault ("ns3::Ipv4L3Protocol::DuplicateExpire", TimeValue (m_expire));

  // Create topology

  // Create nodes
  auto nodes = NodeContainer ();
  nodes.Create (5);

  // Name nodes
  Names::Add ("A", nodes.Get (0));
  Names::Add ("B", nodes.Get (1));
  Names::Add ("C", nodes.Get (2));
  Names::Add ("D", nodes.Get (3));
  Names::Add ("E", nodes.Get (4));

  SimpleNetDeviceHelper simplenet;
  auto devices = simplenet.Install (nodes);
  // name devices
  Names::Add ("A/dev", devices.Get (0));
  Names::Add ("B/dev", devices.Get (1));
  Names::Add ("C/dev", devices.Get (2));
  Names::Add ("D/dev", devices.Get (3));
  Names::Add ("E/dev", devices.Get (4));

  Ipv4ListRoutingHelper listRouting;
  Ipv4StaticRoutingHelper staticRouting;
  listRouting.Add (staticRouting, 0);

  InternetStackHelper internet;
  internet.SetIpv6StackInstall (false);
  internet.SetIpv4ArpJitter (true);
  internet.SetRoutingHelper (listRouting);
  internet.Install (nodes);

  Ipv4AddressHelper ipv4address;
  ipv4address.SetBase ("10.0.0.0", "255.255.255.0");
  ipv4address.Assign (devices);

  // add static routes for each node / device
  auto diter = devices.Begin ();
  for (auto end = nodes.End (),
           iter = nodes.Begin (); iter != end; ++iter)
    {
      // route for forwarding
      staticRouting.AddMulticastRoute (*iter, Ipv4Address::GetAny (), targetAddr.c_str (), *diter, NetDeviceContainer (*diter));
      
      // route for host
      // Use host routing entry according to note in Ipv4StaticRouting::RouteOutput:
      //// Note:  Multicast routes for outbound packets are stored in the
      //// normal unicast table.  An implication of this is that it is not
      //// possible to source multicast datagrams on multiple interfaces.
      //// This is a well-known property of sockets implementation on 
      //// many Unix variants.
      //// So, we just log it and fall through to LookupStatic ()
      auto ipv4 = (*iter)->GetObject <Ipv4> ();
      NS_TEST_ASSERT_MSG_EQ ((bool) ipv4, true, "Node " << Names::FindName (*iter) << " does not have Ipv4 aggregate");
      auto routing = staticRouting.GetStaticRouting (ipv4);
      routing->AddHostRouteTo (targetAddr.c_str (), ipv4->GetInterfaceForDevice (*diter), 0);

      ++diter;
    }

  // set the topology, by default fully-connected
  auto channel = devices.Get (0)->GetChannel ();
  auto simplechannel = channel->GetObject <SimpleChannel> ();
  // ensure some time progress between re-transmissions
  simplechannel->SetAttribute ("Delay", TimeValue (DELAY));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("A/dev"), Names::Find <SimpleNetDevice> ("D/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("D/dev"), Names::Find <SimpleNetDevice> ("A/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("A/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("A/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("B/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("B/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("C/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("C/dev"));

  // Create the UDP sockets
  std::list<Ptr<Socket> > sockets;
  for (auto end = nodes.End (),
           iter = nodes.Begin (); iter != end; ++iter)
    {
      auto SocketFactory = (*iter)->GetObject<UdpSocketFactory> ();
      auto socket = SocketFactory->CreateSocket ();
      socket->SetAllowBroadcast (true);
      NS_TEST_EXPECT_MSG_EQ (socket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 1234)), 0,
                             "Could not bind socket for node " << Names::FindName (*iter));

      auto udpSocket = socket->GetObject<UdpSocket> ();
      udpSocket->MulticastJoinGroup (0, Ipv4Address (targetAddr.c_str ())); // future proof?
      udpSocket->SetAttribute ("IpMulticastTtl", StringValue ("4"));

      socket->SetRecvCallback (MakeCallback (&Ipv4DeduplicationTest::ReceivePkt, this));
      sockets.push_back (socket);
    }

  // connect up drop traces
  Config::ConnectWithoutContext ("/NodeList/*/$ns3::Ipv4L3Protocol/Drop",
                                 MakeCallback (&Ipv4DeduplicationTest::DropPkt, this));

  // start TX from A
  auto txSocket = sockets.front ();

  // ------ Now the tests ------------

  // Broadcast 1 packet
  SendData (txSocket, targetAddr);
  Simulator::Run ();
  Simulator::Destroy ();

  for (auto end = nodes.End (),
           iter = nodes.Begin (); iter != end; ++iter)
    {
      std::string name = Names::FindName (*iter);
      CheckPackets (name);
      CheckDrops (name);
    }

  m_packetCountMap.clear ();
  sockets.clear ();
  Names::Clear ();
}

// NOTE:
// The de-duplicate disabled received packets and drops can be
// computed by forming the connectivity matrix C with 1's in
// coordinates (row, column) where row and column nodes are connected.
// Reception of packets with TTL n are v_(n-1) = v_n * C where
// v_TTL = [1 0 0 0 0] (corresponding to nodes [A B C D E]).  
// The number of drops for each node is v_0 and the number of received
// packets at each node is sum (v_TTL-1, ..., v_0).
void
Ipv4DeduplicationTest::CheckPackets (const std::string &name)
{
    // a priori determined packet receptions based on initial TTL of 4, disabled de-dup
    std::map <std::string, uint32_t> packets = {
      {"A", 14}, {"B", 16}, {"C", 16}, {"D", 16}, {"E", 4}
    };

    // a priori determined packet receptions based on
    std:: map <std::string, uint32_t> packetsDuped = {
      {"A", 0}, {"B", 1}, {"C", 1}, {"D", 1}, {"E", 1}
    };
    // a priori determined packet receptions based on initial TTL of 4, degenerate de-dup
    // There are TTL (4) rounds of packets.  Each round a node will register a
    // received packet if another connected node transmits.  A misses the 1st round
    // since it is the only one transmitting.  D is not connected to A in 1st round
    // either.  E only hears a packet in the 3rd and 4th rounds.
    std::map <std::string, uint32_t> degenerates = {
      {"A", 3}, {"B", 4}, {"C", 4}, {"D", 3}, {"E", 2}
    };

    switch (m_mode)
      {
        case ENABLED:
          NS_TEST_ASSERT_MSG_EQ (m_packetCountMap[name], packetsDuped[name], "Wrong number of packets received for node " << name);
          break;
        case DISABLED:
          NS_TEST_EXPECT_MSG_EQ (m_packetCountMap[name],  packets[name], "Wrong number of packets received for node " << name);
          break;
        case DEGENERATE:
          NS_TEST_EXPECT_MSG_EQ (m_packetCountMap[name],  degenerates[name], "Wrong number of packets received for node " << name);
          break;
      };
}

void
Ipv4DeduplicationTest::CheckDrops (const std::string &name)
{
    std::map <std::string, uint32_t> drops;
    switch (m_mode)
      {
        case ENABLED:
          // a priori determined packet drops based on initial TTL of 4, enabled de-dup;
          // A hears from B & C -- > 2 drops
          // D hears from B, C, AND E
          // B (C) hears from A, C (B), D,
          drops = {{"A", 2}, {"B", 2}, {"C", 2}, {"D", 2}, {"E", 0}};
          break;
        case DISABLED:
          // a priori determined packet drops based on initial TTL of 4, disabled de-dup
          drops = {{"A", 10}, {"B", 9}, {"C", 9}, {"D", 12}, {"E", 2}};
          break;
        case DEGENERATE:
          // a priori determined packet drops based on initial TTL of 4, degenerate de-dup
          // There are TTL (4) rounds of transmissions.  Since all transmitters are
          // synchronized, multiple packets are received each round when there are
          // multiple transmitters.  Only 1 packet per round is delivered, others are
          // dropped.  So this can be computed via "disabled" procedure described
          // in check packets, but with only a 1 for each node in each round when packets
          // are received.  Drops are the sum of receptions using these indicator receive vectors
          // subtracting 1 for each node (for the delivered packet) and adding 1
          // at all nodes for TTL expiry.
          drops = {{"A", 4}, {"B", 5}, {"C", 5}, {"D", 5}, {"E", 1}};
          break;
      }

    if (drops[name])
      {
        NS_TEST_ASSERT_MSG_NE ((m_dropCountMap.find (name) == m_dropCountMap.end ()), true, "No drops for node " << name);
        NS_TEST_EXPECT_MSG_EQ (m_dropCountMap[name], drops[name], "Wrong number of drops for node " << name);
      }
    else
      {
        NS_TEST_EXPECT_MSG_EQ ((m_dropCountMap.find (name) == m_dropCountMap.end ()), true, "Non-0 drops for node " << name);
      }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Deduplication TestSuite
 */
class Ipv4DeduplicationTestSuite : public TestSuite
{
public:
  Ipv4DeduplicationTestSuite ();
private:
};

Ipv4DeduplicationTestSuite::Ipv4DeduplicationTestSuite ()
  : TestSuite ("ipv4-deduplication", UNIT)
{
  
  AddTestCase (new Ipv4DeduplicationTest (true), TestCase::QUICK);
  AddTestCase (new Ipv4DeduplicationTest (false), TestCase::QUICK);
  // degenerate case is enabled RFC but with too short an expiry
  AddTestCase (new Ipv4DeduplicationTest (true, MicroSeconds (50)), TestCase::QUICK);
}

static Ipv4DeduplicationTestSuite g_ipv4DeduplicationTestSuite; //!< Static variable for test initialization



/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Deduplication Performance Test
 *
 * This test case sets up a fully connected network of
 * 10 nodes.  Each node transmits 2 packets / second
 * for about 20 seconds.  Packets are relayed from
 * every receiver.  The test outputs the number of
 * events that have been processed during the course
 * of the simulation.  Test runtime is also a metric.
 * 
 * The de-duplication cache entry expiration algorithm 
 * has evolved from an event-per-expiry (EPE) algorithm to
 * a periodic event, batch purge (PBP) algorithm.  The
 * current metrics are taken from tests on the development
 * box.  Periodic batch purge period defaults to 1s.
 * 
 *        Events        Runtime
 *  EVE   656140          29s
 *  PBP   337420          29s
 * 
 */
class Ipv4DeduplicationPerformanceTest : public TestCase
{
  public:
    Ipv4DeduplicationPerformanceTest (void);
    virtual void DoRun (void);
  private:
    std::vector <Ptr<Socket> > m_sockets; //!< sockets in use
    std::vector <uint8_t> m_txPackets;  //!< transmitted packets for each socket
    uint8_t m_target; //!< number of packets to transmit on each socket

    /**
     * Send data
     * \param socket output socket
     * \param to destination address
     * \param socketIndex index of the socket
     */
    void DoSendData (Ptr<Socket> socket, Address to, uint8_t socketIndex);
};

Ipv4DeduplicationPerformanceTest::Ipv4DeduplicationPerformanceTest ()
  : TestCase ("Ipv4Deduplication performance test")
{
  m_target = 40;
}

void
Ipv4DeduplicationPerformanceTest::DoRun (void)
{
  // multicast target
  const std::string targetAddr = "239.192.100.1";
  Config::SetDefault ("ns3::Ipv4L3Protocol::EnableDuplicatePacketDetection", BooleanValue (true));
  Config::SetDefault ("ns3::Ipv4L3Protocol::DuplicateExpire", TimeValue (Time ("10s")));

  // Create nodes
  auto nodes = NodeContainer ();
  nodes.Create (20);

  SimpleNetDeviceHelper simplenet;
  auto devices = simplenet.Install (nodes);

  Ipv4ListRoutingHelper listRouting;
  Ipv4StaticRoutingHelper staticRouting;
  listRouting.Add (staticRouting, 0);

  InternetStackHelper internet;
  internet.SetIpv6StackInstall (false);
  internet.SetIpv4ArpJitter (true);
  internet.SetRoutingHelper (listRouting);
  internet.Install (nodes);

  Ipv4AddressHelper ipv4address;
  ipv4address.SetBase ("10.0.0.0", "255.255.255.0");
  ipv4address.Assign (devices);

  // add static routes for each node / device
  auto diter = devices.Begin ();
  for (auto iter = nodes.Begin (); iter != nodes.End (); ++iter)
    {
      // route for forwarding
      staticRouting.AddMulticastRoute (*iter, Ipv4Address::GetAny (), targetAddr.c_str (), *diter, NetDeviceContainer (*diter));
      
      // route for host
      // Use host routing entry according to note in Ipv4StaticRouting::RouteOutput:
      //// Note:  Multicast routes for outbound packets are stored in the
      //// normal unicast table.  An implication of this is that it is not
      //// possible to source multicast datagrams on multiple interfaces.
      //// This is a well-known property of sockets implementation on 
      //// many Unix variants.
      //// So, we just log it and fall through to LookupStatic ()
      auto ipv4 = (*iter)->GetObject <Ipv4> ();
      NS_TEST_ASSERT_MSG_EQ ((bool) ipv4, true, "Node " << (*iter)->GetId () << " does not have Ipv4 aggregate");
      auto routing = staticRouting.GetStaticRouting (ipv4);
      routing->AddHostRouteTo (targetAddr.c_str (), ipv4->GetInterfaceForDevice (*diter), 0);

      ++diter;
    }
  

  // Create the UDP sockets
  Ptr<UniformRandomVariable> jitter = CreateObjectWithAttributes <UniformRandomVariable> ("Max", DoubleValue (4));
  Address to = InetSocketAddress (Ipv4Address (targetAddr.c_str ()), 1234);
  for (auto iter = nodes.Begin (); iter != nodes.End (); ++iter)
    {
      Ptr<SocketFactory> udpSocketFactory = (*iter)->GetObject<UdpSocketFactory> ();
      m_sockets.push_back (udpSocketFactory->CreateSocket ());
      m_txPackets.push_back (0);
    }

  for (uint8_t i = 0; i<nodes.GetN (); i++)
    {
      Simulator::ScheduleWithContext (m_sockets[i]->GetNode ()->GetId (),
                                      Seconds (4+jitter->GetValue ()),
                                      &Ipv4DeduplicationPerformanceTest::DoSendData, this, m_sockets[i], to, i);
    }

  Simulator::Run ();
  NS_LOG_UNCOND ("Executed " << Simulator::GetEventCount () << " events");

  for (auto iter = m_sockets.begin (); iter != m_sockets.end (); iter++)
    {
      (*iter)->Close ();
    }

  Simulator::Destroy ();
}

void
Ipv4DeduplicationPerformanceTest::DoSendData (Ptr<Socket> socket, Address to, uint8_t socketIndex)
{
  socket->SendTo (Create<Packet> (512), 0, to);
  if (m_txPackets[socketIndex] < m_target)
    {
      m_txPackets[socketIndex] += 1;
      Simulator::ScheduleWithContext (m_sockets[socketIndex]->GetNode ()->GetId (), Seconds (.5),
                                      &Ipv4DeduplicationPerformanceTest::DoSendData, this, m_sockets[socketIndex], to, socketIndex);
    }
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Deduplication Performance TestSuite
 */
class Ipv4DeduplicationPerformanceTestSuite : public TestSuite
{
public:
  Ipv4DeduplicationPerformanceTestSuite ();
};

Ipv4DeduplicationPerformanceTestSuite::Ipv4DeduplicationPerformanceTestSuite ()
  : TestSuite ("ipv4-deduplication-performance", PERFORMANCE)
{
  AddTestCase (new Ipv4DeduplicationPerformanceTest, TestCase::EXTENSIVE);
}

static Ipv4DeduplicationPerformanceTestSuite g_ipv4DeduplicationPerformanceTestSuite; //!< Static variable for test initialization
