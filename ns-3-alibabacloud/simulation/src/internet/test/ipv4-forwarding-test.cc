/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
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

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"

#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-routing-helper.h"

#include "ns3/traffic-control-layer.h"

#include <string>
#include <limits>

using namespace ns3;


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Forwarding Test
 */
class Ipv4ForwardingTest : public TestCase
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
  Ipv4ForwardingTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv4ForwardingTest::Ipv4ForwardingTest ()
  : TestCase ("UDP socket implementation")
{
}

void Ipv4ForwardingTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
}

void
Ipv4ForwardingTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv4ForwardingTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv4ForwardingTest::DoSendData, this, socket, to);
  Simulator::Run ();
}

void
Ipv4ForwardingTest::DoRun (void)
{
  // Create topology

  // Receiver Node
  Ptr<Node> rxNode = CreateObject<Node> ();

  InternetStackHelper internet;
  internet.SetIpv6StackInstall (false);

  internet.Install (rxNode);
  Ptr<SimpleNetDevice> rxDev;
  { // first interface
    rxDev = CreateObject<SimpleNetDevice> ();
    rxDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    rxNode->AddDevice (rxDev);
    Ptr<Ipv4> ipv4 = rxNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (rxDev);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.2"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
  }

  // Forwarding Node
  Ptr<Node> fwNode = CreateObject<Node> ();

  internet.Install (fwNode);
  Ptr<SimpleNetDevice> fwDev1, fwDev2;
  { // first interface
    fwDev1 = CreateObject<SimpleNetDevice> ();
    fwDev1->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    fwNode->AddDevice (fwDev1);
    Ptr<Ipv4> ipv4 = fwNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (fwDev1);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.1"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
  }

  { // second interface
    fwDev2 = CreateObject<SimpleNetDevice> ();
    fwDev2->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    fwNode->AddDevice (fwDev2);
    Ptr<Ipv4> ipv4 = fwNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (fwDev2);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.1.0.1"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
  }

  // Sender Node
  Ptr<Node> txNode = CreateObject<Node> ();

  internet.Install (txNode);
  Ptr<SimpleNetDevice> txDev;
  {
    txDev = CreateObject<SimpleNetDevice> ();
    txDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    txNode->AddDevice (txDev);
    Ptr<Ipv4> ipv4 = txNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (txDev);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.1.0.2"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
    Ptr<Ipv4StaticRouting> ipv4StaticRouting = Ipv4RoutingHelper::GetRouting <Ipv4StaticRouting> (txNode->GetObject<Ipv4> ()->GetRoutingProtocol ());
    ipv4StaticRouting->SetDefaultRoute(Ipv4Address("10.1.0.1"), netdev_idx);
  }

  // link the two nodes
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  rxDev->SetChannel (channel1);
  fwDev1->SetChannel (channel1);

  Ptr<SimpleChannel> channel2 = CreateObject<SimpleChannel> ();
  fwDev2->SetChannel (channel2);
  txDev->SetChannel (channel2);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("10.0.0.2"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4ForwardingTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "10.0.0.2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "IPv4 Forwarding on");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket = 0;

  Ptr<Ipv4> ipv4 = fwNode->GetObject<Ipv4> ();
  ipv4->SetAttribute("IpForward", BooleanValue (false));
  SendData (txSocket, "10.0.0.2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "IPv4 Forwarding off");

  Simulator::Destroy ();

}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Forwarding TestSuite
 */
class Ipv4ForwardingTestSuite : public TestSuite
{
public:
  Ipv4ForwardingTestSuite ();
private:
};

Ipv4ForwardingTestSuite::Ipv4ForwardingTestSuite ()
  : TestSuite ("ipv4-forwarding", UNIT)
{
  AddTestCase (new Ipv4ForwardingTest, TestCase::QUICK);
}

static Ipv4ForwardingTestSuite g_ipv4forwardingTestSuite; //!< Static variable for test initialization

