/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' di Firenze
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
#include "ns3/inet-socket-address.h"

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/rip.h"
#include "ns3/rip-helper.h"
#include "ns3/node-container.h"
#include "ns3/ipv4-static-routing.h"

#include <string>
#include <limits>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 RIP Test
 */
class Ipv4RipTest : public TestCase
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
  Ipv4RipTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv4RipTest::Ipv4RipTest ()
  : TestCase ("RIP")
{
}

void Ipv4RipTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received Packet size is not equal to the Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv4RipTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv4RipTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (60),
                                  &Ipv4RipTest::DoSendData, this, socket, to);
  Simulator::Stop (Seconds (66));
  Simulator::Run ();
}

void
Ipv4RipTest::DoRun (void)
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

  RipHelper ripRouting;
  InternetStackHelper internetRouters;
  internetRouters.SetRoutingHelper (ripRouting);
  internetRouters.Install (routers);

  InternetStackHelper internetNodes;
  internetNodes.Install (nodes);

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

  // Setup IPv4 addresses and forwarding
  Ipv4AddressHelper ipv4;

  ipv4.SetBase (Ipv4Address ("10.0.1.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic1 = ipv4.Assign (net1);

  ipv4.SetBase (Ipv4Address ("192.168.0.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic2 = ipv4.Assign (net2);

  ipv4.SetBase (Ipv4Address ("192.168.1.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic3 = ipv4.Assign (net3);

  ipv4.SetBase (Ipv4Address ("10.0.2.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic4 = ipv4.Assign (net4);

  Ptr<Ipv4StaticRouting> staticRouting;
  staticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (txNode->GetObject<Ipv4> ()->GetRoutingProtocol ());
  staticRouting->SetDefaultRoute ("10.0.1.2", 1 );
  staticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (rxNode->GetObject<Ipv4> ()->GetRoutingProtocol ());
  staticRouting->SetDefaultRoute ("10.0.2.1", 1 );

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("10.0.2.2"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4RipTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "10.0.2.2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "IPv4 RIP should work.");

  m_receivedPacket->RemoveAllByteTags ();

  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 RIP count to infinity Test
 */
class Ipv4RipCountToInfinityTest : public TestCase
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
  Ipv4RipCountToInfinityTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv4RipCountToInfinityTest::Ipv4RipCountToInfinityTest ()
  : TestCase ("RIP counting to infinity")
{
}

void Ipv4RipCountToInfinityTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received Packet size is not equal to the Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv4RipCountToInfinityTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv4RipCountToInfinityTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (60),
                                  &Ipv4RipCountToInfinityTest::DoSendData, this, socket, to);
  Simulator::Stop (Seconds (66));
  Simulator::Run ();
}

void
Ipv4RipCountToInfinityTest::DoRun (void)
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

  RipHelper ripNgRouting;
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

  // Setup IPv4 addresses and forwarding
  Ipv4AddressHelper ipv4;

  ipv4.SetBase (Ipv4Address ("10.0.1.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic1 = ipv4.Assign (net1);

  ipv4.SetBase (Ipv4Address ("192.168.0.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic2 = ipv4.Assign (net2);

  ipv4.SetBase (Ipv4Address ("192.168.1.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic3 = ipv4.Assign (net3);

  ipv4.SetBase (Ipv4Address ("10.0.2.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic4 = ipv4.Assign (net4);

  Ptr<Ipv4StaticRouting> staticRouting;
  staticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (txNode->GetObject<Ipv4> ()->GetRoutingProtocol ());
  staticRouting->SetDefaultRoute ("10.0.1.2", 1 );
  staticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (rxNode->GetObject<Ipv4> ()->GetRoutingProtocol ());
  staticRouting->SetDefaultRoute ("10.0.2.1", 1 );


  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("10.0.2.2"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4RipCountToInfinityTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  SendData (txSocket, "10.0.2.2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "RIP counting to infinity.");

  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 RIP SplitHorizon strategy Test
 */
class Ipv4RipSplitHorizonStrategyTest : public TestCase
{
  Rip::SplitHorizonType_e m_setStrategy;      //!< Strategy set.
  Rip::SplitHorizonType_e m_detectedStrategy; //!< Strategy detected.

public:
  virtual void DoRun (void);

  /**
   * \brief Constructor.
   * \param strategy The SplitHorizon strategy.
   */
  Ipv4RipSplitHorizonStrategyTest (Rip::SplitHorizonType_e strategy);

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePktProbe (Ptr<Socket> socket);
};

Ipv4RipSplitHorizonStrategyTest::Ipv4RipSplitHorizonStrategyTest (Rip::SplitHorizonType_e strategy)
  : TestCase ("RIP Split Horizon strategy")
{
  m_setStrategy = strategy;
}

void Ipv4RipSplitHorizonStrategyTest::ReceivePktProbe (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  Address srcAddr;
  Ptr<Packet> receivedPacketProbe = socket->RecvFrom (std::numeric_limits<uint32_t>::max (), 0, srcAddr);
  NS_TEST_ASSERT_MSG_EQ (availableData, receivedPacketProbe->GetSize (), "Received Packet size is not equal to the Rx buffer size");
  Ipv4Address senderAddress = InetSocketAddress::ConvertFrom (srcAddr).GetIpv4 ();

  if (senderAddress == "192.168.0.2")
    {
      RipHeader hdr;
      receivedPacketProbe->RemoveHeader (hdr);

      std::list<RipRte> rtes = hdr.GetRteList ();

      // validate the RTEs before processing
      for (std::list<RipRte>::iterator iter = rtes.begin ();
          iter != rtes.end (); iter++)
        {
          if (iter->GetPrefix () == "10.0.1.0")
            {
              bool correct = false;
              if (iter->GetRouteMetric () == 16)
                {
                  correct = true;
                  m_detectedStrategy = Rip::POISON_REVERSE;
                }
              else if (iter->GetRouteMetric () == 2)
                {
                  correct = true;
                  m_detectedStrategy = Rip::NO_SPLIT_HORIZON;
                }
              NS_TEST_EXPECT_MSG_EQ (correct, true, "RIP: unexpected metric value: " << iter->GetRouteMetric ());
            }
        }
    }

  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Ipv4RipSplitHorizonStrategyTest::DoRun (void)
{
  // Create topology

  Ptr<Node> fakeNode = CreateObject<Node> ();
  Ptr<Node> listener = CreateObject<Node> ();

  Ptr<Node> routerA = CreateObject<Node> ();
  Ptr<Node> routerB = CreateObject<Node> ();

  NodeContainer listeners (listener, fakeNode);
  NodeContainer routers (routerA, routerB);
  NodeContainer all (routers, listeners);

  RipHelper ripNgRouting;
  ripNgRouting.Set ("SplitHorizon", EnumValue (m_setStrategy));

  InternetStackHelper internetRouters;
  internetRouters.SetRoutingHelper (ripNgRouting);
  internetRouters.Install (routers);

  InternetStackHelper internetNodes;
  internetNodes.Install (listeners);

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
  Ipv4AddressHelper ipv4;

  ipv4.SetBase (Ipv4Address ("10.0.1.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic0 = ipv4.Assign (net0);

  ipv4.SetBase (Ipv4Address ("192.168.0.0"), Ipv4Mask ("255.255.255.0"));
  Ipv4InterfaceContainer iic1 = ipv4.Assign (net1);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = listener->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  rxSocket->BindToNetDevice (listenerDev);
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("224.0.0.9"), 520)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4RipSplitHorizonStrategyTest::ReceivePktProbe, this));

  // ------ Now the tests ------------

  // If the strategy is Split Horizon, then no packet will be received.
  m_detectedStrategy = Rip::SPLIT_HORIZON;

  Simulator::Stop (Seconds (66));
  Simulator::Run ();
  NS_TEST_EXPECT_MSG_EQ (m_detectedStrategy, m_setStrategy, "RIP counting to infinity.");

  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 RIP TestSuite
 */
class Ipv4RipTestSuite : public TestSuite
{
public:
  Ipv4RipTestSuite () : TestSuite ("ipv4-rip", UNIT)
  {
    AddTestCase (new Ipv4RipTest, TestCase::QUICK);
    AddTestCase (new Ipv4RipCountToInfinityTest, TestCase::QUICK);
    AddTestCase (new Ipv4RipSplitHorizonStrategyTest (Rip::POISON_REVERSE), TestCase::QUICK);
    AddTestCase (new Ipv4RipSplitHorizonStrategyTest (Rip::SPLIT_HORIZON), TestCase::QUICK);
    AddTestCase (new Ipv4RipSplitHorizonStrategyTest (Rip::NO_SPLIT_HORIZON), TestCase::QUICK);
  }
};

static Ipv4RipTestSuite g_ipv4ripTestSuite; //!< Static variable for test initialization
