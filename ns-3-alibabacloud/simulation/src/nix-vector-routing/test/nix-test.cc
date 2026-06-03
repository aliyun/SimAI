/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 NITK Surathkal
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
 * Author: Ameya Deshpande <ameyanrd@outlook.com>
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/nix-vector-helper.h"

using namespace ns3;
/**
 * \defgroup nix-vector-routing-test Nix-Vector Routing Tests
 */

/**
 * \ingroup nix-vector-routing-test
 * \ingroup tests
 *
 * The topology is of the form:
 * \verbatim
              __________
             /          \
    nSrc -- nA -- nB -- nC -- nDst
   \endverbatim
 *
 * Following are the tests in this test case:
 * - Test the routing from nSrc to nDst.
 * - Test if the path taken is the shortest path.
 * (Set down the interface of nA on nA-nC channel.)
 * - Test if the NixCache and Ipv4RouteCache are empty.
 * - Test the routing from nSrc to nDst again.
 * - Test if the new shortest path is taken.
 * (Set down the interface of nC on nB-nC channel.)
 * - Test that routing is not possible from nSrc to nDst.
 *
 * \brief IPv4 Nix-Vector Routing Test
 */
class NixVectorRoutingTest : public TestCase
{
  Ptr<Packet> m_receivedPacket; //!< Received packet

  /**
   * \brief Send data immediately after being called.
   * \param socket The sending socket.
   * \param to IPv4 Destination address.
   */
  void DoSendDatav4 (Ptr<Socket> socket, Ipv4Address to);

  /**
   * \brief Send data immediately after being called.
   * \param socket The sending socket.
   * \param to IPv6 Destination address.
   */
  void DoSendDatav6 (Ptr<Socket> socket, Ipv6Address to);

  /**
   * \brief Schedules the DoSendData () function to send the data.
   * \param delay The scheduled time to send data.
   * \param socket The sending socket.
   * \param to IPv4 Destination address.
   */
  void SendData (Time delay, Ptr<Socket> socket, Ipv4Address to);

  /**
   * \brief Schedules the DoSendData () function to send the data.
   * \param delay The scheduled time to send data.
   * \param socket The sending socket.
   * \param to IPv6 Destination address.
   */
  void SendData (Time delay, Ptr<Socket> socket, Ipv6Address to);

public:
  virtual void DoRun (void);
  NixVectorRoutingTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);

  std::vector<uint32_t> m_receivedPacketSizes; //!< Received packet sizes
};

NixVectorRoutingTest::NixVectorRoutingTest ()
  : TestCase ("three router, two path test")
{
}

void NixVectorRoutingTest::ReceivePkt (Ptr<Socket> socket)
{
  [[maybe_unused]] uint32_t availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (),
                         "availableData should be equal to the size of packet received.");
  m_receivedPacketSizes.push_back (m_receivedPacket->GetSize ());
}

void
NixVectorRoutingTest::DoSendDatav4 (Ptr<Socket> socket, Ipv4Address to)
{
  Address realTo = InetSocketAddress (to, 1234);
  socket->SendTo (Create<Packet> (123), 0, realTo);
}

void
NixVectorRoutingTest::DoSendDatav6 (Ptr<Socket> socket, Ipv6Address to)
{
  Address realTo = Inet6SocketAddress (to, 1234);
  socket->SendTo (Create<Packet> (123), 0, realTo);
}

void
NixVectorRoutingTest::SendData (Time delay, Ptr<Socket> socket, Ipv4Address to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), delay,
                                  &NixVectorRoutingTest::DoSendDatav4, this, socket, to);
}

void
NixVectorRoutingTest::SendData (Time delay, Ptr<Socket> socket, Ipv6Address to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), delay,
                                  &NixVectorRoutingTest::DoSendDatav6, this, socket, to);
}

void
NixVectorRoutingTest::DoRun (void)
{
  // Create topology
  NodeContainer nSrcnA;
  NodeContainer nAnB;
  NodeContainer nBnC;
  NodeContainer nCnDst;
  NodeContainer nAnC;

  nSrcnA.Create (2);

  nAnB.Add (nSrcnA.Get (1));
  nAnB.Create (1);

  nBnC.Add (nAnB.Get (1));
  nBnC.Create (1);

  nCnDst.Add (nBnC.Get (1));
  nCnDst.Create (1);

  nAnC.Add (nAnB.Get (0));
  nAnC.Add (nCnDst.Get (0));

  SimpleNetDeviceHelper devHelper;
  devHelper.SetNetDevicePointToPointMode (true);

  NodeContainer allNodes = NodeContainer (nSrcnA, nBnC, nCnDst.Get (1));

  std::ostringstream stringStream1v4;
  Ptr<OutputStreamWrapper> routingStream1v4 = Create<OutputStreamWrapper> (&stringStream1v4);
  std::ostringstream stringStream1v6;
  Ptr<OutputStreamWrapper> routingStream1v6 = Create<OutputStreamWrapper> (&stringStream1v6);
  std::ostringstream stringStream2v4;
  Ptr<OutputStreamWrapper> cacheStreamv4 = Create<OutputStreamWrapper> (&stringStream2v4);
  std::ostringstream stringStream2v6;
  Ptr<OutputStreamWrapper> cacheStreamv6 = Create<OutputStreamWrapper> (&stringStream2v6);
  std::ostringstream stringStream3v4;
  Ptr<OutputStreamWrapper> routingStream3v4 = Create<OutputStreamWrapper> (&stringStream3v4);
  std::ostringstream stringStream3v6;
  Ptr<OutputStreamWrapper> routingStream3v6 = Create<OutputStreamWrapper> (&stringStream3v6);

  // NixHelper to install nix-vector routing on all nodes
  Ipv4NixVectorHelper ipv4NixRouting;
  Ipv6NixVectorHelper ipv6NixRouting;
  InternetStackHelper stack;
  stack.SetRoutingHelper (ipv4NixRouting); // has effect on the next Install ()
  stack.SetRoutingHelper (ipv6NixRouting); // has effect on the next Install ()
  stack.Install (allNodes);

  NetDeviceContainer dSrcdA;
  NetDeviceContainer dAdB;
  NetDeviceContainer dBdC;
  NetDeviceContainer dCdDst;
  NetDeviceContainer dAdC;
  dSrcdA = devHelper.Install (nSrcnA);
  dAdB = devHelper.Install (nAnB);
  dBdC = devHelper.Install (nBnC);
  dCdDst = devHelper.Install (nCnDst);
  dAdC = devHelper.Install (nAnC);

  Ipv4AddressHelper aSrcaAv4;
  aSrcaAv4.SetBase ("10.1.0.0", "255.255.255.0");
  Ipv4AddressHelper aAaBv4;
  aAaBv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4AddressHelper aBaCv4;
  aBaCv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4AddressHelper aCaDstv4;
  aCaDstv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4AddressHelper aAaCv4;
  aAaCv4.SetBase ("10.1.4.0", "255.255.255.0");

  Ipv6AddressHelper aSrcaAv6;
  aSrcaAv6.SetBase (Ipv6Address ("2001:0::"), Ipv6Prefix (64));
  Ipv6AddressHelper aAaBv6;
  aAaBv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6AddressHelper aBaCv6;
  aBaCv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6AddressHelper aCaDstv6;
  aCaDstv6.SetBase (Ipv6Address ("2001:3::"), Ipv6Prefix (64));
  Ipv6AddressHelper aAaCv6;
  aAaCv6.SetBase (Ipv6Address ("2001:4::"), Ipv6Prefix (64));

  aSrcaAv4.Assign (dSrcdA);
  aAaBv4.Assign (dAdB);
  aBaCv4.Assign (dBdC);
  Ipv4InterfaceContainer iCiDstv4 = aCaDstv4.Assign (dCdDst);
  Ipv4InterfaceContainer iAiCv4 = aAaCv4.Assign (dAdC);

  aSrcaAv6.Assign (dSrcdA);
  aAaBv6.Assign (dAdB);
  aBaCv6.Assign (dBdC);
  Ipv6InterfaceContainer iCiDstv6 = aCaDstv6.Assign (dCdDst);
  Ipv6InterfaceContainer iAiCv6 = aAaCv6.Assign (dAdC);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = nCnDst.Get (1)->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocketv4 = rxSocketFactory->CreateSocket ();
  Ptr<Socket> rxSocketv6 = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocketv4->Bind (InetSocketAddress (iCiDstv4.GetAddress (1), 1234)), 0, "trivial");
  NS_TEST_EXPECT_MSG_EQ (rxSocketv6->Bind (Inet6SocketAddress (iCiDstv6.GetAddress (1, 1), 1234)), 0, "trivial");
  rxSocketv4->SetRecvCallback (MakeCallback (&NixVectorRoutingTest::ReceivePkt, this));
  rxSocketv6->SetRecvCallback (MakeCallback (&NixVectorRoutingTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = nSrcnA.Get (0)->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  SendData (Seconds (2), txSocket, Ipv4Address ("10.1.3.2"));
  SendData (Seconds (2), txSocket, Ipv6Address ("2001:3::200:ff:fe00:8"));

  ipv4NixRouting.PrintRoutingPathAt (Seconds (3), nSrcnA.Get (0), iCiDstv4.GetAddress (1), routingStream1v4);
  ipv6NixRouting.PrintRoutingPathAt (Seconds (3), nSrcnA.Get (0), iCiDstv6.GetAddress (1, 1), routingStream1v6);

  // Set the IPv4 nA interface on nA - nC channel down.
  Ptr<Ipv4> ipv4 = nAnC.Get (0)->GetObject<Ipv4> ();
  int32_t ifIndex = ipv4->GetInterfaceForDevice (dAdC.Get (0));
  Simulator::Schedule (Seconds (5), &Ipv4::SetDown, ipv4, ifIndex);

  // Set the IPv6 nA interface on nA - nC channel down.
  Ptr<Ipv6> ipv6 = nAnC.Get (0)->GetObject<Ipv6> ();
  ifIndex = ipv6->GetInterfaceForDevice (dAdC.Get (0));
  Simulator::Schedule (Seconds (5), &Ipv6::SetDown, ipv6, ifIndex);

  ipv4NixRouting.PrintRoutingTableAllAt (Seconds (7), cacheStreamv4);
  ipv6NixRouting.PrintRoutingTableAllAt (Seconds (7), cacheStreamv6);

  SendData (Seconds (8), txSocket, Ipv4Address ("10.1.3.2"));
  SendData (Seconds (8), txSocket, Ipv6Address ("2001:3::200:ff:fe00:8"));

  ipv4NixRouting.PrintRoutingPathAt (Seconds (9), nSrcnA.Get (0), iCiDstv4.GetAddress (1), routingStream3v4);
  ipv6NixRouting.PrintRoutingPathAt (Seconds (9), nSrcnA.Get (0), iCiDstv6.GetAddress (1, 1), routingStream3v6);

  // Set the IPv4 nC interface on nB - nC channel down.
  ipv4 = nBnC.Get (1)->GetObject<Ipv4> ();
  ifIndex = ipv4->GetInterfaceForDevice (dBdC.Get (1));
  Simulator::Schedule (Seconds (10), &Ipv4::SetDown, ipv4, ifIndex);

  // Set the IPv6 nC interface on nB - nC channel down.
  ipv6 = nBnC.Get (1)->GetObject<Ipv6> ();
  ifIndex = ipv6->GetInterfaceForDevice (dBdC.Get (1));
  Simulator::Schedule (Seconds (10), &Ipv6::SetDown, ipv6, ifIndex);

  SendData (Seconds (11), txSocket, Ipv4Address ("10.1.3.2"));
  SendData (Seconds (11), txSocket, Ipv6Address ("2001:3::200:ff:fe00:8"));

  Simulator::Stop (Seconds (66));
  Simulator::Run ();

  // ------ Now the tests ------------

  // Test the Routing
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacketSizes[0], 123, "IPv4 Nix-Vector Routing should work.");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacketSizes[1], 123, "IPv6 Nix-Vector Routing should work.");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacketSizes.size (), 4, "IPv4 and IPv6 Nix-Vector Routing should have received only 1 packet.");

  // Test the Path
  const std::string p_nSrcnAnCnDstv4 = "Time: +3s, Nix Routing\n"
                                              "Route path from Node 0 to Node 4, Nix Vector: 01001 (5 bits left)\n"
                                              "10.1.0.1                 (Node 0)  ---->   10.1.0.2                 (Node 1)\n"
                                              "10.1.4.1                 (Node 1)  ---->   10.1.4.2                 (Node 3)\n"
                                              "10.1.3.1                 (Node 3)  ---->   10.1.3.2                 (Node 4)\n\n";
  NS_TEST_EXPECT_MSG_EQ (stringStream1v4.str (), p_nSrcnAnCnDstv4, "Routing Path is incorrect.");

  const std::string p_nSrcnAnCnDstv6 = "Time: +3s, Nix Routing\n"
                                              "Route path from Node 0 to Node 4, Nix Vector: 01001 (5 bits left)\n"
                                              "2001::200:ff:fe00:1      (Node 0)  ---->   fe80::200:ff:fe00:2      (Node 1)\n"
                                              "fe80::200:ff:fe00:9      (Node 1)  ---->   fe80::200:ff:fe00:a      (Node 3)\n"
                                              "fe80::200:ff:fe00:7      (Node 3)  ---->   2001:3::200:ff:fe00:8    (Node 4)\n\n";
  NS_TEST_EXPECT_MSG_EQ (stringStream1v6.str (), p_nSrcnAnCnDstv6, "Routing Path is incorrect.");

  const std::string p_nSrcnAnBnCnDstv4 = "Time: +9s, Nix Routing\n"
                                                 "Route path from Node 0 to Node 4, Nix Vector: 0111 (4 bits left)\n"
                                                 "10.1.0.1                 (Node 0)  ---->   10.1.0.2                 (Node 1)\n"
                                                 "10.1.1.1                 (Node 1)  ---->   10.1.1.2                 (Node 2)\n"
                                                 "10.1.2.1                 (Node 2)  ---->   10.1.2.2                 (Node 3)\n"
                                                 "10.1.3.1                 (Node 3)  ---->   10.1.3.2                 (Node 4)\n\n";
  NS_TEST_EXPECT_MSG_EQ (stringStream3v4.str (), p_nSrcnAnBnCnDstv4, "Routing Path is incorrect.");

  const std::string p_nSrcnAnBnCnDstv6 = "Time: +9s, Nix Routing\n"
                                                 "Route path from Node 0 to Node 4, Nix Vector: 0111 (4 bits left)\n"
                                                 "2001::200:ff:fe00:1      (Node 0)  ---->   fe80::200:ff:fe00:2      (Node 1)\n"
                                                 "fe80::200:ff:fe00:3      (Node 1)  ---->   fe80::200:ff:fe00:4      (Node 2)\n"
                                                 "fe80::200:ff:fe00:5      (Node 2)  ---->   fe80::200:ff:fe00:6      (Node 3)\n"
                                                 "fe80::200:ff:fe00:7      (Node 3)  ---->   2001:3::200:ff:fe00:8    (Node 4)\n\n";
  NS_TEST_EXPECT_MSG_EQ (stringStream3v6.str (), p_nSrcnAnBnCnDstv6, "Routing Path is incorrect.");

  const std::string emptyCaches = "Node: 0, Time: +7s, Local time: +7s, Nix Routing\n"
                                  "NixCache:\n"
                                  "IpRouteCache:\n\n"
                                  "Node: 1, Time: +7s, Local time: +7s, Nix Routing\n"
                                  "NixCache:\n"
                                  "IpRouteCache:\n\n"
                                  "Node: 2, Time: +7s, Local time: +7s, Nix Routing\n"
                                  "NixCache:\n"
                                  "IpRouteCache:\n\n"
                                  "Node: 3, Time: +7s, Local time: +7s, Nix Routing\n"
                                  "NixCache:\n"
                                  "IpRouteCache:\n\n"
                                  "Node: 4, Time: +7s, Local time: +7s, Nix Routing\n"
                                  "NixCache:\n"
                                  "IpRouteCache:\n\n";
  NS_TEST_EXPECT_MSG_EQ (stringStream2v4.str (), emptyCaches, "The caches should have been empty.");
  NS_TEST_EXPECT_MSG_EQ (stringStream2v6.str (), emptyCaches, "The caches should have been empty.");
  
  Simulator::Destroy ();
}

/**
 * \ingroup nix-vector-routing-test
 * \ingroup tests
 *
 * \brief IPv4 Nix-Vector Routing TestSuite
 */
class NixVectorRoutingTestSuite : public TestSuite
{
public:
  NixVectorRoutingTestSuite () : TestSuite ("nix-vector-routing", UNIT)
  {
    AddTestCase (new NixVectorRoutingTest (), TestCase::QUICK);
  }
};

/// Static variable for test initialization
static NixVectorRoutingTestSuite g_nixVectorRoutingTestSuite;
