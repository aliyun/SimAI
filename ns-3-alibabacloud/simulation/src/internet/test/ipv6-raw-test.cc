/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Hajime Tazaki
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
 * Author: Hajime Tazaki <tazaki@sfc.wide.ad.jp>
 */
/**
 * This is the test code for ipv6-raw-socket-impl.cc.
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/ipv6-raw-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/node-container.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/ipv6-static-routing.h"
#include "ns3/ipv6-address-helper.h"

#include <string>
#include <limits>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RAW Socket Test
 */
class Ipv6RawSocketImplTest : public TestCase
{
  Ptr<Packet> m_receivedPacket;   //!< Received packet (1).
  Ptr<Packet> m_receivedPacket2;  //!< Received packet (2).

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
  Ipv6RawSocketImplTest ();

  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   * \param packet The received packet.
   * \param from The sender.
   */
  void ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from);
  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   * \param packet The received packet.
   * \param from The sender.
   */
  void ReceivePacket2 (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from);
  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
  /**
   * \brief Receive data.
   * \param socket The receiving socket.
   */
  void ReceivePkt2 (Ptr<Socket> socket);
};


Ipv6RawSocketImplTest::Ipv6RawSocketImplTest ()
  : TestCase ("Ipv6 Raw socket implementation") 
{
}

void Ipv6RawSocketImplTest::ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket = packet;
}

void Ipv6RawSocketImplTest::ReceivePacket2 (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket2 = packet;
}

void Ipv6RawSocketImplTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (2, MSG_PEEK);
  NS_TEST_ASSERT_MSG_EQ (m_receivedPacket->GetSize (), 2, "ReceivedPacket size is not equal to 2");
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
}

void Ipv6RawSocketImplTest::ReceivePkt2 (Ptr<Socket> socket)
{
  uint32_t availableData;
  Address addr;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket2 = socket->Recv (2, MSG_PEEK);
  NS_TEST_ASSERT_MSG_EQ (m_receivedPacket2->GetSize (), 2, "ReceivedPacket size is not equal to 2");
  m_receivedPacket2 = socket->RecvFrom (std::numeric_limits<uint32_t>::max (), 0, addr);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket2->GetSize (), "Received packet size is not equal to Rx buffer size");
  Inet6SocketAddress v6addr = Inet6SocketAddress::ConvertFrom (addr);
  NS_TEST_EXPECT_MSG_EQ (v6addr.GetIpv6 (), Ipv6Address ("2001:db8::2"), "recvfrom");
}

void
Ipv6RawSocketImplTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = Inet6SocketAddress (Ipv6Address (to.c_str ()), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, to);
}

void
Ipv6RawSocketImplTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  m_receivedPacket2 = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv6RawSocketImplTest::DoSendData, this, socket, to);
  Simulator::Run ();
}

void
Ipv6RawSocketImplTest::DoRun (void)
{
  // Create topology

  // Receiver Node
  Ptr<Node> rxNode = CreateObject<Node> ();
  // Sender Node
  Ptr<Node> txNode = CreateObject<Node> ();

  NodeContainer nodes (rxNode, txNode);

  SimpleNetDeviceHelper helperChannel1;
  helperChannel1.SetNetDevicePointToPointMode (true);
  NetDeviceContainer net1 = helperChannel1.Install (nodes);

  SimpleNetDeviceHelper helperChannel2;
  helperChannel2.SetNetDevicePointToPointMode (true);
  NetDeviceContainer net2 = helperChannel2.Install (nodes);

  InternetStackHelper internetv6;
  internetv6.Install (nodes);

  txNode->GetObject<Icmpv6L4Protocol> ()->SetAttribute ("DAD", BooleanValue (false));
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
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:db8::1"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  device = net2.Get (0);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:db8:1::3"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);

  ipv6 = txNode->GetObject<Ipv6> ();
  device = net1.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:db8::2"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetForwarding (ifIndex, true);

  device = net2.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:db8:1::4"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetForwarding (ifIndex, true);

  // Create the Ipv6 Raw sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<Ipv6RawSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address::GetAny (), 0)), 0, "trivial");
  rxSocket->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
  rxSocket->SetRecvCallback (MakeCallback (&Ipv6RawSocketImplTest::ReceivePkt, this));

  Ptr<Socket> rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Ipv6RawSocketImplTest::ReceivePkt2, this));
  rxSocket2->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (Inet6SocketAddress (Ipv6Address ("2001:db8:1::3"), 0)), 0, "trivial");

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<Ipv6RawSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "2001:db8::1");

  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 163, "recv: 2001:db8::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second interface should not receive it");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Simple Link-local multicast test
  txSocket->Bind (Inet6SocketAddress (Ipv6Address ("2001:db8::2"), 0));
  SendData (txSocket, "ff02::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 163, "recv: ff02::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second socket should not receive it (it is bound specifically to the second interface's address");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Broadcast test with multiple receiving sockets

  // When receiving broadcast packets, all sockets sockets bound to
  // the address/port should receive a copy of the same packet -- if
  // the socket address matches.
  rxSocket2->Dispose ();
  rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Ipv6RawSocketImplTest::ReceivePkt2, this));
  rxSocket2->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (Inet6SocketAddress (Ipv6Address::GetAny (), 0)), 0, "trivial");

  SendData (txSocket, "ff02::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 163, "recv: ff02::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 163, "recv: ff02::1");

  m_receivedPacket = 0;
  m_receivedPacket2 = 0;

  // Simple getpeername tests

  Address peerAddress;
  int err = txSocket->GetPeerName (peerAddress);
  NS_TEST_EXPECT_MSG_EQ (err, -1, "socket GetPeerName() should fail when socket is not connected");
  NS_TEST_EXPECT_MSG_EQ (txSocket->GetErrno (), Socket::ERROR_NOTCONN, "socket error code should be ERROR_NOTCONN");

  Inet6SocketAddress peer ("2001:db8::1", 1234);
  err = txSocket->Connect (peer);
  NS_TEST_EXPECT_MSG_EQ (err, 0, "socket Connect() should succeed");

  err = txSocket->GetPeerName (peerAddress);
  NS_TEST_EXPECT_MSG_EQ (err, 0, "socket GetPeerName() should succeed when socket is connected");
  peer.SetPort (0);
  NS_TEST_EXPECT_MSG_EQ (peerAddress, peer, "address from socket GetPeerName() should equal the connected address");

  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 RAW Socket TestSuite
 */
class Ipv6RawTestSuite : public TestSuite
{
public:
  Ipv6RawTestSuite () : TestSuite ("ipv6-raw", UNIT)
  {
    AddTestCase (new Ipv6RawSocketImplTest, TestCase::QUICK);
  }
};

static Ipv6RawTestSuite g_ipv6rawTestSuite; //!< Static variable for test initialization

