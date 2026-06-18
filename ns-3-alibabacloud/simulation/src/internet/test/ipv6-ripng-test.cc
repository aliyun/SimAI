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

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/socket.h"
#include "ns3/boolean.h"
#include "ns3/enum.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet6-socket-address.h"

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/ripng.h"
#include "ns3/ripng-helper.h"
#include "ns3/node-container.h"

#include <string>
#include <limits>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RIPng Test
 */
class Ipv6RipngTest : public TestCase
{
  Ptr<Packet> m_receivedPacket; //!< Received packet

  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void DoSendData (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void SendData (Ptr<Socket> socket, std::string to);

public:
  virtual void DoRun (void);
  Ipv6RipngTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv6RipngTest::Ipv6RipngTest ()
  : TestCase ("RIPng")
{
}

void Ipv6RipngTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv6RipngTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = Inet6SocketAddress (Ipv6Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv6RipngTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (60),
                                  &Ipv6RipngTest::DoSendData, this, socket, to);
  Simulator::Stop (Seconds (66));
  Simulator::Run ();
}

void
Ipv6RipngTest::DoRun (void)
{
  // Create topology

  Ptr<Node> txNode = CreateObject<Node> ();
  Ptr<Node> rxNode = CreateObject<Node> ();
  Ptr<Node> routerA = CreateObject<Node> ();
  Ptr<Node> routerB = CreateObject<Node> ();
  Ptr<Node> routerC = CreateObject<Node> ();

  NodeContainer nodes (txNode, rxNode);
  NodeContainer routers (routerA, routerB, routerC);
  NodeContainer all (nodes, routers);

  RipNgHelper ripNgRouting;
  InternetStackHelper internetv6routers;
  internetv6routers.SetRoutingHelper (ripNgRouting);
  internetv6routers.Install (routers);

  InternetStackHelper internetv6nodes;
  internetv6nodes.Install (nodes);

  NetDeviceContainer net1;
  NetDeviceContainer net2;
  NetDeviceContainer net3;
  NetDeviceContainer net4;

  // Sender Node
  Ptr<SimpleNetDevice> txDev;
  {
    txDev = CreateObject<SimpleNetDevice> ();
    txDev->SetAddress (Mac48Address ("00:00:00:00:00:01"));
    txNode->AddDevice (txDev);
  }
  net1.Add (txDev);

  // Router A
  Ptr<SimpleNetDevice> fwDev1routerA, fwDev2routerA;
  { // first interface
    fwDev1routerA = CreateObject<SimpleNetDevice> ();
    fwDev1routerA->SetAddress (Mac48Address ("00:00:00:00:00:02"));
    routerA->AddDevice (fwDev1routerA);
  }
  net1.Add (fwDev1routerA);

  { // second interface
    fwDev2routerA = CreateObject<SimpleNetDevice> ();
    fwDev2routerA->SetAddress (Mac48Address ("00:00:00:00:00:03"));
    routerA->AddDevice (fwDev2routerA);
  }
  net2.Add (fwDev2routerA);

  // Router B
  Ptr<SimpleNetDevice> fwDev1routerB, fwDev2routerB;
  { // first interface
    fwDev1routerB = CreateObject<SimpleNetDevice> ();
    fwDev1routerB->SetAddress (Mac48Address ("00:00:00:00:00:04"));
    routerB->AddDevice (fwDev1routerB);
  }
  net2.Add (fwDev1routerB);

  { // second interface
    fwDev2routerB = CreateObject<SimpleNetDevice> ();
    fwDev2routerB->SetAddress (Mac48Address ("00:00:00:00:00:05"));
    routerB->AddDevice (fwDev2routerB);
  }
  net3.Add (fwDev2routerB);

  // Router C
  Ptr<SimpleNetDevice> fwDev1routerC, fwDev2routerC;
  { // first interface
    fwDev1routerC = CreateObject<SimpleNetDevice> ();
    fwDev1routerC->SetAddress (Mac48Address ("00:00:00:00:00:06"));
    routerC->AddDevice (fwDev1routerC);
  }
  net3.Add (fwDev1routerC);

  { // second interface
    fwDev2routerC = CreateObject<SimpleNetDevice> ();
    fwDev2routerC->SetAddress (Mac48Address ("00:00:00:00:00:07"));
    routerC->AddDevice (fwDev2routerC);
  }
  net4.Add (fwDev2routerC);

  // Rx node
  Ptr<SimpleNetDevice> rxDev;
  { // first interface
    rxDev = CreateObject<SimpleNetDevice> ();
    rxDev->SetAddress (Mac48Address ("00:00:00:00:00:08"));
    rxNode->AddDevice (rxDev);
  }
  net4.Add (rxDev);

  // link the channels
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  txDev->SetChannel (channel1);
  fwDev1routerA->SetChannel (channel1);

  Ptr<SimpleChannel> channel2 = CreateObject<SimpleChannel> ();
  fwDev2routerA->SetChannel (channel2);
  fwDev1routerB->SetChannel (channel2);

  Ptr<SimpleChannel> channel3 = CreateObject<SimpleChannel> ();
  fwDev2routerB->SetChannel (channel3);
  fwDev1routerC->SetChannel (channel3);

  Ptr<SimpleChannel> channel4 = CreateObject<SimpleChannel> ();
  fwDev2routerC->SetChannel (channel4);
  rxDev->SetChannel (channel4);

  // Setup IPv6 addresses and forwarding
  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer iic1 = ipv6.AssignWithoutOnLink (net1);
  iic1.SetForwarding (1, true);
  iic1.SetDefaultRouteInAllNodes (1);

  Ipv6InterfaceContainer iic2 = ipv6.AssignWithoutAddress (net2);
  iic2.SetForwarding (0, true);
  iic2.SetForwarding (1, true);

  Ipv6InterfaceContainer iic3 = ipv6.AssignWithoutAddress (net3);
  iic3.SetForwarding (0, true);
  iic3.SetForwarding (1, true);

  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer iic4 = ipv6.AssignWithoutOnLink (net4);
  iic4.SetForwarding (0, true);
  iic4.SetDefaultRouteInAllNodes (0);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address ("2001:2::200:ff:fe00:8"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv6RipngTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "2001:2::200:ff:fe00:8");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "IPv6 RIPng should work.");

  m_receivedPacket->RemoveAllByteTags ();

  Simulator::Destroy ();
}

// Ipv6RipngCountToInfinityTest

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RIPng count to infinity Test
 */
class Ipv6RipngCountToInfinityTest : public TestCase
{
  Ptr<Packet> m_receivedPacket; //!< Received packet

  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void DoSendData (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void SendData (Ptr<Socket> socket, std::string to);

public:
  virtual void DoRun (void);
  Ipv6RipngCountToInfinityTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv6RipngCountToInfinityTest::Ipv6RipngCountToInfinityTest ()
  : TestCase ("RIPng counting to infinity")
{
}

void Ipv6RipngCountToInfinityTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv6RipngCountToInfinityTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = Inet6SocketAddress (Ipv6Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv6RipngCountToInfinityTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (60),
                                  &Ipv6RipngCountToInfinityTest::DoSendData, this, socket, to);
  Simulator::Stop (Seconds (66));
  Simulator::Run ();
}

void
Ipv6RipngCountToInfinityTest::DoRun (void)
{
  // Create topology

  Ptr<Node> txNode = CreateObject<Node> ();
  Ptr<Node> rxNode = CreateObject<Node> ();
  Ptr<Node> routerA = CreateObject<Node> ();
  Ptr<Node> routerB = CreateObject<Node> ();
  Ptr<Node> routerC = CreateObject<Node> ();

  NodeContainer nodes (txNode, rxNode);
  NodeContainer routers (routerA, routerB, routerC);
  NodeContainer all (nodes, routers);

  RipNgHelper ripNgRouting;
  // Change the router's interface metric to 10, must not send packets (count to infinity)
  // note: Interface 0 is the loopback.
  ripNgRouting.SetInterfaceMetric (routerA, 2, 10);
  ripNgRouting.SetInterfaceMetric (routerB, 1, 10);
  ripNgRouting.SetInterfaceMetric (routerB, 2, 10);
  ripNgRouting.SetInterfaceMetric (routerC, 1, 10);

  InternetStackHelper internetv6routers;
  internetv6routers.SetRoutingHelper (ripNgRouting);
  internetv6routers.Install (routers);

  InternetStackHelper internetv6nodes;
  internetv6nodes.Install (nodes);

  NetDeviceContainer net1;
  NetDeviceContainer net2;
  NetDeviceContainer net3;
  NetDeviceContainer net4;

  // Sender Node
  Ptr<SimpleNetDevice> txDev;
  {
    txDev = CreateObject<SimpleNetDevice> ();
    txDev->SetAddress (Mac48Address ("00:00:00:00:00:01"));
    txNode->AddDevice (txDev);
  }
  net1.Add (txDev);

  // Router A
  Ptr<SimpleNetDevice> fwDev1routerA, fwDev2routerA;
  { // first interface
    fwDev1routerA = CreateObject<SimpleNetDevice> ();
    fwDev1routerA->SetAddress (Mac48Address ("00:00:00:00:00:02"));
    routerA->AddDevice (fwDev1routerA);
  }
  net1.Add (fwDev1routerA);

  { // second interface
    fwDev2routerA = CreateObject<SimpleNetDevice> ();
    fwDev2routerA->SetAddress (Mac48Address ("00:00:00:00:00:03"));
    routerA->AddDevice (fwDev2routerA);
  }
  net2.Add (fwDev2routerA);

  // Router B
  Ptr<SimpleNetDevice> fwDev1routerB, fwDev2routerB;
  { // first interface
    fwDev1routerB = CreateObject<SimpleNetDevice> ();
    fwDev1routerB->SetAddress (Mac48Address ("00:00:00:00:00:04"));
    routerB->AddDevice (fwDev1routerB);
  }
  net2.Add (fwDev1routerB);

  { // second interface
    fwDev2routerB = CreateObject<SimpleNetDevice> ();
    fwDev2routerB->SetAddress (Mac48Address ("00:00:00:00:00:05"));
    routerB->AddDevice (fwDev2routerB);
  }
  net3.Add (fwDev2routerB);

  // Router C
  Ptr<SimpleNetDevice> fwDev1routerC, fwDev2routerC;
  { // first interface
    fwDev1routerC = CreateObject<SimpleNetDevice> ();
    fwDev1routerC->SetAddress (Mac48Address ("00:00:00:00:00:06"));
    routerC->AddDevice (fwDev1routerC);
  }
  net3.Add (fwDev1routerC);

  { // second interface
    fwDev2routerC = CreateObject<SimpleNetDevice> ();
    fwDev2routerC->SetAddress (Mac48Address ("00:00:00:00:00:07"));
    routerC->AddDevice (fwDev2routerC);
  }
  net4.Add (fwDev2routerC);

  // Rx node
  Ptr<SimpleNetDevice> rxDev;
  { // first interface
    rxDev = CreateObject<SimpleNetDevice> ();
    rxDev->SetAddress (Mac48Address ("00:00:00:00:00:08"));
    rxNode->AddDevice (rxDev);
  }
  net4.Add (rxDev);

  // link the channels
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  txDev->SetChannel (channel1);
  fwDev1routerA->SetChannel (channel1);

  Ptr<SimpleChannel> channel2 = CreateObject<SimpleChannel> ();
  fwDev2routerA->SetChannel (channel2);
  fwDev1routerB->SetChannel (channel2);

  Ptr<SimpleChannel> channel3 = CreateObject<SimpleChannel> ();
  fwDev2routerB->SetChannel (channel3);
  fwDev1routerC->SetChannel (channel3);

  Ptr<SimpleChannel> channel4 = CreateObject<SimpleChannel> ();
  fwDev2routerC->SetChannel (channel4);
  rxDev->SetChannel (channel4);

  // Setup IPv6 addresses and forwarding
  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer iic1 = ipv6.AssignWithoutOnLink (net1);
  iic1.SetForwarding (1, true);
  iic1.SetDefaultRouteInAllNodes (1);

  Ipv6InterfaceContainer iic2 = ipv6.AssignWithoutAddress (net2);
  iic2.SetForwarding (0, true);
  iic2.SetForwarding (1, true);

  Ipv6InterfaceContainer iic3 = ipv6.AssignWithoutAddress (net3);
  iic3.SetForwarding (0, true);
  iic3.SetForwarding (1, true);

  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer iic4 = ipv6.AssignWithoutOnLink (net4);
  iic4.SetForwarding (0, true);
  iic4.SetDefaultRouteInAllNodes (0);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address ("2001:2::200:ff:fe00:8"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv6RipngCountToInfinityTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  SendData (txSocket, "2001:2::200:ff:fe00:8");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "RIPng counting to infinity.");

  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RIPng SplitHorizon strategy Test
 */
class Ipv6RipngSplitHorizonStrategyTest : public TestCase
{
  RipNg::SplitHorizonType_e m_setStrategy;      //!< Strategy set.
  RipNg::SplitHorizonType_e m_detectedStrategy; //!< Strategy detected.

public:
  virtual void DoRun (void);
  /**
   * \brief Constructor.
   * \param strategy The SplitHorizon strategy.
   */
  Ipv6RipngSplitHorizonStrategyTest (RipNg::SplitHorizonType_e strategy);

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePktProbe (Ptr<Socket> socket);
};

Ipv6RipngSplitHorizonStrategyTest::Ipv6RipngSplitHorizonStrategyTest (RipNg::SplitHorizonType_e strategy)
  : TestCase ("RIPng Split Horizon strategy")
{
  m_setStrategy = strategy;
}

void Ipv6RipngSplitHorizonStrategyTest::ReceivePktProbe (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  Address srcAddr;
  Ptr<Packet> receivedPacketProbe = socket->RecvFrom (std::numeric_limits<uint32_t>::max (), 0, srcAddr);
  NS_TEST_ASSERT_MSG_EQ (availableData, receivedPacketProbe->GetSize (), "ReceivedPacketProbe size is not equal to the Rx buffer size");
  Ipv6Address senderAddress = Inet6SocketAddress::ConvertFrom (srcAddr).GetIpv6 ();

  if (senderAddress == "fe80::200:ff:fe00:4")
    {
      RipNgHeader hdr;
      receivedPacketProbe->RemoveHeader (hdr);
      std::list<RipNgRte> rtes = hdr.GetRteList ();

      // validate the RTEs before processing
      for (std::list<RipNgRte>::iterator iter = rtes.begin ();
          iter != rtes.end (); iter++)
        {
          if (iter->GetPrefix () == "2001:1::")
            {
              bool correct = false;
              if (iter->GetRouteMetric () == 16)
                {
                  correct = true;
                  m_detectedStrategy = RipNg::POISON_REVERSE;
                }
              else if (iter->GetRouteMetric () == 2)
                {
                  correct = true;
                  m_detectedStrategy = RipNg::NO_SPLIT_HORIZON;
                }
              NS_TEST_EXPECT_MSG_EQ (correct, true, "RIPng: unexpected metric value: " << iter->GetRouteMetric ());
            }
        }
    }

  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv6RipngSplitHorizonStrategyTest::DoRun (void)
{
  // Create topology

  Ptr<Node> fakeNode = CreateObject<Node> ();
  Ptr<Node> listener = CreateObject<Node> ();

  Ptr<Node> routerA = CreateObject<Node> ();
  Ptr<Node> routerB = CreateObject<Node> ();

  NodeContainer listeners (listener, fakeNode);
  NodeContainer routers (routerA, routerB);
  NodeContainer all (routers, listeners);

  RipNgHelper ripNgRouting;
  ripNgRouting.Set ("SplitHorizon", EnumValue (m_setStrategy));

  InternetStackHelper internetv6routers;
  internetv6routers.SetRoutingHelper (ripNgRouting);
  internetv6routers.Install (routers);

  InternetStackHelper internetv6nodes;
  internetv6nodes.Install (listeners);

  NetDeviceContainer net0;
  NetDeviceContainer net1;

  // Fake Node
  Ptr<SimpleNetDevice> silentDev;
  {
    silentDev = CreateObject<SimpleNetDevice> ();
    silentDev->SetAddress (Mac48Address ("00:00:00:00:00:01"));
    fakeNode->AddDevice (silentDev);
  }
  net0.Add (silentDev);

  // Router A
  Ptr<SimpleNetDevice> silentDevRouterA, fwDevRouterA;
  { // silent interface
    silentDevRouterA = CreateObject<SimpleNetDevice> ();
    silentDevRouterA->SetAddress (Mac48Address ("00:00:00:00:00:02"));
    routerA->AddDevice (silentDevRouterA);
  }
  net0.Add (silentDevRouterA);

  { // first interface
    fwDevRouterA = CreateObject<SimpleNetDevice> ();
    fwDevRouterA->SetAddress (Mac48Address ("00:00:00:00:00:03"));
    routerA->AddDevice (fwDevRouterA);
  }
  net1.Add (fwDevRouterA);

  // Router B
  Ptr<SimpleNetDevice> fwDevRouterB;
  { // first interface
    fwDevRouterB = CreateObject<SimpleNetDevice> ();
    fwDevRouterB->SetAddress (Mac48Address ("00:00:00:00:00:04"));
    routerB->AddDevice (fwDevRouterB);
  }
  net1.Add (fwDevRouterB);

  // listener A
  Ptr<SimpleNetDevice> listenerDev;
  {
    listenerDev = CreateObject<SimpleNetDevice> ();
    listenerDev->SetAddress (Mac48Address ("00:00:00:00:00:05"));
    listener->AddDevice (listenerDev);
  }
  net1.Add (listenerDev);

  // link the channels
  Ptr<SimpleChannel> channel0 = CreateObject<SimpleChannel> ();
  silentDev->SetChannel (channel0);
  silentDevRouterA->SetChannel (channel0);

  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  fwDevRouterA->SetChannel (channel1);
  fwDevRouterB->SetChannel (channel1);
  listenerDev->SetChannel (channel1);

  // Setup IPv6 addresses and forwarding
  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer iic0 = ipv6.AssignWithoutOnLink (net0);

  Ipv6InterfaceContainer iic1 = ipv6.AssignWithoutAddress (net1);
  iic1.SetForwarding (0, true);
  iic1.SetForwarding (1, true);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = listener->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  rxSocket->BindToNetDevice (listenerDev);
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address ("ff02::9"), 521)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv6RipngSplitHorizonStrategyTest::ReceivePktProbe, this));

  // ------ Now the tests ------------

  // If the strategy is Split Horizon, then no packet will be received.
  m_detectedStrategy = RipNg::SPLIT_HORIZON;

  Simulator::Stop (Seconds (66));
  Simulator::Run ();
  NS_TEST_EXPECT_MSG_EQ (m_detectedStrategy, m_setStrategy, "RIPng counting to infinity.");

  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RIPng TestSuite
 */
class Ipv6RipngTestSuite : public TestSuite
{
public:
  Ipv6RipngTestSuite () : TestSuite ("ipv6-ripng", UNIT)
  {
    AddTestCase (new Ipv6RipngTest, TestCase::QUICK);
    AddTestCase (new Ipv6RipngCountToInfinityTest, TestCase::QUICK);
    AddTestCase (new Ipv6RipngSplitHorizonStrategyTest (RipNg::POISON_REVERSE), TestCase::QUICK);
    AddTestCase (new Ipv6RipngSplitHorizonStrategyTest (RipNg::SPLIT_HORIZON), TestCase::QUICK);
    AddTestCase (new Ipv6RipngSplitHorizonStrategyTest (RipNg::NO_SPLIT_HORIZON), TestCase::QUICK);
  }
};

static Ipv6RipngTestSuite g_ipv6ripngTestSuite; //!< Static variable for test initialization
