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
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"
#include "ns3/boolean.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet6-socket-address.h"

#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/ipv6-static-routing.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/ipv6-routing-helper.h"

#include <string>
#include <limits>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 Forwarding Test
 */
class Ipv6ForwardingTest : public TestCase
{
  Ptr<Packet> m_receivedPacket;   //!< Received packet.

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
  Ipv6ForwardingTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};

Ipv6ForwardingTest::Ipv6ForwardingTest ()
  : TestCase ("IPv6 forwarding")
{
}

void Ipv6ForwardingTest::ReceivePkt (Ptr<Socket> socket)
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
Ipv6ForwardingTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = Inet6SocketAddress (Ipv6Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
Ipv6ForwardingTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv6ForwardingTest::DoSendData, this, socket, to);
  Simulator::Run ();
}

void
Ipv6ForwardingTest::DoRun (void)
{
  // Create topology

  // Receiver Node
  Ptr<Node> rxNode = CreateObject<Node> ();
  // Forwarding Node
  Ptr<Node> fwNode = CreateObject<Node> ();
  // Sender Node
  Ptr<Node> txNode = CreateObject<Node> ();

  NodeContainer net1nodes (rxNode, fwNode);
  NodeContainer net2nodes (fwNode, txNode);
  NodeContainer nodes (rxNode, fwNode, txNode);

  SimpleNetDeviceHelper helperChannel1;
  helperChannel1.SetNetDevicePointToPointMode (true);
  NetDeviceContainer net1 = helperChannel1.Install (net1nodes);

  SimpleNetDeviceHelper helperChannel2;
  helperChannel2.SetNetDevicePointToPointMode (true);
  NetDeviceContainer net2 = helperChannel2.Install (net2nodes);

  InternetStackHelper internetv6;
  internetv6.Install (nodes);

  txNode->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("DAD", BooleanValue (false));
  fwNode->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("DAD", BooleanValue (false));
  rxNode->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("DAD", BooleanValue (false));

  Ipv6AddressHelper ipv6helper;
  Ipv6InterfaceContainer iic1 = ipv6helper.AssignWithoutAddress (net1);
  Ipv6InterfaceContainer iic2 = ipv6helper.AssignWithoutAddress (net2);

  Ptr<NetDevice> device;
  Ptr<Ipv6> ipv6;
  int32_t ifIndex;
  Ipv6InterfaceAddress ipv6Addr;

  ipv6 = rxNode->GetObject<Ipv6> ();
  device = net1.Get (0);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:1::2"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  ipv6 = fwNode->GetObject<Ipv6> ();
  device = net1.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:1::1"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  device = net2.Get (0);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:2::1"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  ipv6 = txNode->GetObject<Ipv6> ();
  device = net2.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:2::2"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  // Setup at least a route from the sender.
  Ptr<Ipv6StaticRouting> ipv6StaticRouting = Ipv6RoutingHelper::GetRouting <Ipv6StaticRouting> (txNode->GetObject<Ipv6> ()->GetRoutingProtocol ());
  ipv6StaticRouting->SetDefaultRoute (Ipv6Address ("2001:2::1"), ifIndex);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address ("2001:1::2"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv6ForwardingTest::ReceivePkt, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "2001:1::2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "IPv6 Forwarding off");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket = 0;

  ipv6 = fwNode->GetObject<Ipv6> ();
  ipv6->SetAttribute("IpForward", BooleanValue (true));
  SendData (txSocket, "2001:1::2");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "IPv6 Forwarding on");

  m_receivedPacket->RemoveAllByteTags ();

  Simulator::Destroy ();

}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 Forwarding TestSuite
 */
class Ipv6ForwardingTestSuite : public TestSuite
{
public:
  Ipv6ForwardingTestSuite () : TestSuite ("ipv6-forwarding", UNIT)
  {
    AddTestCase (new Ipv6ForwardingTest, TestCase::QUICK);
  }
};

static Ipv6ForwardingTestSuite g_ipv6forwardingTestSuite; //!< Static variable for test initialization
