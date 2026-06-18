/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
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
 * This is the test code for ipv4-raw-socket-impl.cc.
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/boolean.h"

#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/internet-stack-helper.h"

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
 * \brief IPv4 RAW Socket Test
 */
class Ipv4RawSocketImplTest : public TestCase
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
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void DoSendData_IpHdr (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to Destination address.
   */
  void SendData_IpHdr (Ptr<Socket> socket, std::string to);

public:
  virtual void DoRun (void);
  Ipv4RawSocketImplTest ();

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


Ipv4RawSocketImplTest::Ipv4RawSocketImplTest ()
  : TestCase ("IPv4 Raw socket implementation") 
{
}

void Ipv4RawSocketImplTest::ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket = packet;
}

void Ipv4RawSocketImplTest::ReceivePacket2 (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket2 = packet;
}

void Ipv4RawSocketImplTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (2, MSG_PEEK);
  NS_TEST_ASSERT_MSG_EQ (m_receivedPacket->GetSize (), 2, "ReceivedPacket size is not equal to 2");
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
}

void Ipv4RawSocketImplTest::ReceivePkt2 (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket2 = socket->Recv (2, MSG_PEEK);
  NS_TEST_ASSERT_MSG_EQ (m_receivedPacket2->GetSize (), 2, "ReceivedPacket size is not equal to 2");
  m_receivedPacket2 = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket2->GetSize (), "Received packet size is not equal to Rx buffer size");
}

void
Ipv4RawSocketImplTest::DoSendData (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, to);
}

void
Ipv4RawSocketImplTest::SendData (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  m_receivedPacket2 = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv4RawSocketImplTest::DoSendData, this, socket, to);
  Simulator::Run ();
}

void
Ipv4RawSocketImplTest::DoSendData_IpHdr (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 0);
  socket->SetAttribute ("IpHeaderInclude", BooleanValue (true));
  Ptr<Packet> p = Create<Packet> (123);
  Ipv4Header ipHeader;
  ipHeader.SetSource (Ipv4Address ("10.0.0.2"));
  ipHeader.SetDestination (Ipv4Address (to.c_str ()));
  ipHeader.SetProtocol (0);
  ipHeader.SetPayloadSize (p->GetSize ());
  ipHeader.SetTtl (255);
  p->AddHeader (ipHeader);

  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (p, 0, realTo),
                         143, to);
  socket->SetAttribute ("IpHeaderInclude", BooleanValue (false));
}

void
Ipv4RawSocketImplTest::SendData_IpHdr (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  m_receivedPacket2 = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv4RawSocketImplTest::DoSendData_IpHdr, this, socket, to);
  Simulator::Run ();
}

void
Ipv4RawSocketImplTest::DoRun (void)
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

  InternetStackHelper internet;
  internet.Install (nodes);

  Ptr<Ipv4> ipv4;
  uint32_t netdev_idx;
  Ipv4InterfaceAddress ipv4Addr;

  // Receiver Node
  ipv4 = rxNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net1.Get (0));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.1"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  netdev_idx = ipv4->AddInterface (net2.Get (0));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.1.1"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  // Sender Node
  ipv4 = txNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net1.Get (1));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.2"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  netdev_idx = ipv4->AddInterface (net2.Get (1));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.1.2"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  // Create the IPv4 Raw sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<Ipv4RawSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("0.0.0.0"), 0)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4RawSocketImplTest::ReceivePkt, this));

  Ptr<Socket> rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Ipv4RawSocketImplTest::ReceivePkt2, this));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (InetSocketAddress (Ipv4Address ("10.0.1.1"), 0)), 0, "trivial");

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<Ipv4RawSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();

  // ------ Now the tests ------------

  // Unicast test
  SendData (txSocket, "10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv: 10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second interface should not receive it");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Unicast w/ header test
  SendData_IpHdr (txSocket, "10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv(hdrincl): 10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second interface should not receive it");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

#if 0
  // Simple broadcast test

  SendData (txSocket, "255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv: 255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second socket should not receive it (it is bound specifically to the second interface's address");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();
#endif

  // Simple Link-local multicast test

  txSocket->Bind (InetSocketAddress (Ipv4Address ("10.0.0.2"), 0));
  SendData (txSocket, "224.0.0.9");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv: 224.0.0.9");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second socket should not receive it (it is bound specifically to the second interface's address");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

#if 0
  // Broadcast test with multiple receiving sockets

  // When receiving broadcast packets, all sockets sockets bound to
  // the address/port should receive a copy of the same packet -- if
  // the socket address matches.
  rxSocket2->Dispose ();
  rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Ipv4RawSocketImplTest::ReceivePkt2, this));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (InetSocketAddress (Ipv4Address ("0.0.0.0"), 0)), 0, "trivial");

  SendData (txSocket, "255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv: 255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 143, "recv: 255.255.255.255");
#endif

  m_receivedPacket = 0;
  m_receivedPacket2 = 0;

  // Simple getpeername tests

  Address peerAddress;
  int err = txSocket->GetPeerName (peerAddress);
  NS_TEST_EXPECT_MSG_EQ (err, -1, "socket GetPeerName() should fail when socket is not connected");
  NS_TEST_EXPECT_MSG_EQ (txSocket->GetErrno (), Socket::ERROR_NOTCONN, "socket error code should be ERROR_NOTCONN");

  InetSocketAddress peer ("10.0.0.1", 1234);
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
 * \brief IPv4 RAW Socket TestSuite
 */
class Ipv4RawTestSuite : public TestSuite
{
public:
  Ipv4RawTestSuite () : TestSuite ("ipv4-raw", UNIT)
  {
    AddTestCase (new Ipv4RawSocketImplTest, TestCase::QUICK);
  }
};

static Ipv4RawTestSuite g_ipv4rawTestSuite; //!< Static variable for test initialization
