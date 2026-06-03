/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
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
 * Author: Raj Bhattacharjea <raj.b@gatech.edu>
 */
/**
 * This is the test code for udp-socket-impl.cc, it was moved out of udp-socket-impl.cc to 
 * be in an independent file for clarity purposes.
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"
#include "ns3/traffic-control-helper.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"

#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-queue-disc-item.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/ipv6-static-routing.h"
#include "ns3/ipv6-address-helper.h"

#include <string>
#include <limits>

using namespace ns3;


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief UDP Socket Loopback over IPv4 Test
 */
class UdpSocketLoopbackTest : public TestCase
{
public:
  UdpSocketLoopbackTest ();
  virtual void DoRun (void);

  /**
   * \brief Receive a packet.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
  Ptr<Packet> m_receivedPacket; //!< Received packet
};

UdpSocketLoopbackTest::UdpSocketLoopbackTest ()
  : TestCase ("UDP loopback test") 
{
}

void UdpSocketLoopbackTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
}

void
UdpSocketLoopbackTest::DoRun ()
{
  Ptr<Node> rxNode = CreateObject<Node> ();
  InternetStackHelper internet;
  internet.Install (rxNode);

  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  rxSocket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 80));
  rxSocket->SetRecvCallback (MakeCallback (&UdpSocketLoopbackTest::ReceivePkt, this));

  Ptr<Socket> txSocket = rxSocketFactory->CreateSocket ();
  txSocket->SendTo (Create<Packet> (246), 0, InetSocketAddress ("127.0.0.1", 80));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 246, "first socket should not receive it (it is bound specifically to the second interface's address");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief UDP Socket Loopback over IPv6 Test
 */
class Udp6SocketLoopbackTest : public TestCase
{
public:
  Udp6SocketLoopbackTest ();
  virtual void DoRun (void);

  /**
   * \brief Receive a packet.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
  Ptr<Packet> m_receivedPacket; //!< Received packet
};

Udp6SocketLoopbackTest::Udp6SocketLoopbackTest ()
  : TestCase ("UDP6 loopback test") 
{
}

void Udp6SocketLoopbackTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Udp6SocketLoopbackTest::DoRun ()
{
  Ptr<Node> rxNode = CreateObject<Node> ();
  InternetStackHelper internet;
  internet.Install (rxNode);

  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  rxSocket->Bind (Inet6SocketAddress (Ipv6Address::GetAny (), 80));
  rxSocket->SetRecvCallback (MakeCallback (&Udp6SocketLoopbackTest::ReceivePkt, this));

  Ptr<Socket> txSocket = rxSocketFactory->CreateSocket ();
  txSocket->SendTo (Create<Packet> (246), 0, Inet6SocketAddress ("::1", 80));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 246, "first socket should not receive it (it is bound specifically to the second interface's address");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief UDP Socket over IPv4 Test
 */
class UdpSocketImplTest : public TestCase
{
  Ptr<Packet> m_receivedPacket;   //!< Received packet (1).
  Ptr<Packet> m_receivedPacket2;  //!< Received packet (2).
  Ptr<Ipv4QueueDiscItem> m_sentPacket;  //!< Sent packet.

  /**
   * \brief Get the TOS of the received packet.
   * \returns The TOS.
   */
  uint32_t GetTos (void);

  /**
   * \brief Get the priority of the received packet.
   * \returns The priority.
   */
  uint32_t GetPriority (void);

  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to The destination address.
   */
  void DoSendDataTo (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to The destination address.
   */
  void SendDataTo (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   */
  void DoSendData (Ptr<Socket> socket);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   */
  void SendData (Ptr<Socket> socket);

public:
  virtual void DoRun (void);
  UdpSocketImplTest ();

  /**
   * \brief Receive packets (1).
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
  /**
   * \brief Receive packets (2).
   * \param socket The receiving socket.
   */
  void ReceivePkt2 (Ptr<Socket> socket);

  /**
   * \brief Adds a packet to the list of sent packets.
   * \param item The sent packet.
   */
  void SentPkt (Ptr<const QueueDiscItem> item);
};

UdpSocketImplTest::UdpSocketImplTest ()
  : TestCase ("UDP socket implementation") 
{
}

void UdpSocketImplTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
}

void UdpSocketImplTest::ReceivePkt2 (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket2 = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket2->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
}

void UdpSocketImplTest::SentPkt (Ptr<const QueueDiscItem> item)
{
  Ptr<const Ipv4QueueDiscItem> ipv4Item = DynamicCast<const Ipv4QueueDiscItem> (item);
  NS_TEST_EXPECT_MSG_NE (ipv4Item, 0, "no IPv4 packet");
  Address addr;
  m_sentPacket = Create<Ipv4QueueDiscItem> (ipv4Item->GetPacket ()->Copy (), addr, 0, ipv4Item->GetHeader ());
}

uint32_t UdpSocketImplTest::GetTos (void)
{
  return static_cast<uint32_t> (m_sentPacket->GetHeader ().GetTos ());
}

uint32_t UdpSocketImplTest::GetPriority (void)
{
  SocketPriorityTag priorityTag;
  bool found = m_sentPacket->GetPacket ()->PeekPacketTag (priorityTag);
  NS_TEST_EXPECT_MSG_EQ (found, true, "the packet should carry a SocketPriorityTag");
  return static_cast<uint32_t> (priorityTag.GetPriority ());
}

void
UdpSocketImplTest::DoSendDataTo (Ptr<Socket> socket, std::string to)
{
  Address realTo = InetSocketAddress (Ipv4Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "100");
}

void
UdpSocketImplTest::SendDataTo (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  m_receivedPacket2 = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &UdpSocketImplTest::DoSendDataTo, this, socket, to);
  Simulator::Run ();
}

void
UdpSocketImplTest::DoSendData (Ptr<Socket> socket)
{
  NS_TEST_EXPECT_MSG_EQ (socket->Send (Create<Packet> (123), 0), 123, "100");
}

void
UdpSocketImplTest::SendData (Ptr<Socket> socket)
{
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &UdpSocketImplTest::DoSendData, this, socket);
  Simulator::Run ();
}

void
UdpSocketImplTest::DoRun (void)
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

  TrafficControlHelper tch = TrafficControlHelper::Default ();
  QueueDiscContainer qdiscs = tch.Install (net1.Get (1));

  Ptr<Ipv4> ipv4;
  uint32_t netdev_idx;
  Ipv4InterfaceAddress ipv4Addr;

  // Receiver Node
  ipv4 = rxNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net1.Get (0));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.1"), Ipv4Mask ("/24"));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  netdev_idx = ipv4->AddInterface (net2.Get (0));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.1.1"), Ipv4Mask ("/24"));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  // Sender Node
  ipv4 = txNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net1.Get (1));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.2"), Ipv4Mask ("/24"));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  netdev_idx = ipv4->AddInterface (net2.Get (1));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.1.2"), Ipv4Mask ("/24"));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();

  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("10.0.0.1"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&UdpSocketImplTest::ReceivePkt, this));

  Ptr<Socket> rxSocket2 = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (InetSocketAddress (Ipv4Address ("10.0.1.1"), 1234)), 0, "trivial");
  rxSocket2->SetRecvCallback (MakeCallback (&UdpSocketImplTest::ReceivePkt2, this));

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);

  // ------ Now the tests ------------

  // Unicast test
  SendDataTo (txSocket, "10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "trivial");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second interface should not receive it");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Simple broadcast test

  SendDataTo (txSocket, "255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "first socket should not receive it (it is bound specifically to the first interface's address");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second socket should not receive it (it is bound specifically to the second interface's address");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Broadcast test with multiple receiving sockets

  // When receiving broadcast packets, all sockets sockets bound to
  // the address/port should receive a copy of the same packet -- if
  // the socket address matches.
  rxSocket2->Dispose ();
  rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&UdpSocketImplTest::ReceivePkt2, this));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (InetSocketAddress (Ipv4Address ("0.0.0.0"), 1234)), 0, "trivial");

  SendDataTo (txSocket, "255.255.255.255");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "first socket should not receive it (it is bound specifically to the first interface's address");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 123, "trivial");

  m_receivedPacket = 0;
  m_receivedPacket2 = 0;

  // Simple Link-local multicast test

  txSocket->BindToNetDevice (net1.Get (1));
  SendDataTo (txSocket, "224.0.0.9");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "first socket should not receive it (it is bound specifically to the first interface's address");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 123, "recv2: 224.0.0.9");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

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
  NS_TEST_EXPECT_MSG_EQ (peerAddress, peer, "address from socket GetPeerName() should equal the connected address");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // TOS and priority tests

  // Intercept the packets dequeued by the queue disc on the sender node
  qdiscs.Get (0)->TraceConnectWithoutContext ("Dequeue", MakeCallback (&UdpSocketImplTest::SentPkt, this));

  // The socket is not connected.
  txSocket->SetIpTos (0x28);  // AF11
  txSocket->SetPriority (6);  // Interactive
  // Send a packet to a specified destination:
  // - for not connected sockets, the tos specified in the destination address (0) is used
  // - since the tos is zero, the priority set for the socket is used
  SendDataTo (txSocket, "10.0.0.1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "trivial");

  NS_TEST_EXPECT_MSG_EQ (GetTos (), 0, "the TOS should be set to 0");
  NS_TEST_EXPECT_MSG_EQ (GetPriority (), 6, "Interactive (6)");

  m_receivedPacket->RemoveAllByteTags ();

  InetSocketAddress dest ("10.0.0.1", 1234);
  dest.SetTos (0xb8);  // EF
  // the connect operation sets the tos (and priority) for the socket
  NS_TEST_EXPECT_MSG_EQ (txSocket->Connect (dest), 0, "the connect operation failed");

  SendData (txSocket);
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "trivial");

  NS_TEST_EXPECT_MSG_EQ (GetTos (), 0xb8, "the TOS should be set to 0xb8");
  NS_TEST_EXPECT_MSG_EQ (GetPriority (), 4, "Interactive bulk (4)");

  m_receivedPacket->RemoveAllByteTags ();

  Simulator::Destroy ();

}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief UDP Socket over IPv6 Test
 */
class Udp6SocketImplTest : public TestCase
{
  Ptr<Packet> m_receivedPacket;   //!< Received packet (1).
  Ptr<Packet> m_receivedPacket2;  //!< Received packet (2).

  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to The destination address.
   */
  void DoSendDataTo (Ptr<Socket> socket, std::string to);
  /**
   * \brief Send data.
   * \param socket The sending socket.
   * \param to The destination address.
   */
  void SendDataTo (Ptr<Socket> socket, std::string to);

public:
  virtual void DoRun (void);
  Udp6SocketImplTest ();

  /**
   * \brief Receive packets (1).
   * \param socket The receiving socket.
   * \param packet The received packet.
   * \param from The source address.
   */
  void ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from);
  /**
   * \brief Receive packets (2).
   * \param socket The receiving socket.
   * \param packet The received packet.
   * \param from The source address.
   */
  void ReceivePacket2 (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from);
  /**
   * \brief Receive packets (1).
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
  /**
   * \brief Receive packets (2).
   * \param socket The receiving socket.
   */
  void ReceivePkt2 (Ptr<Socket> socket);
};

Udp6SocketImplTest::Udp6SocketImplTest ()
  : TestCase ("UDP6 socket implementation")
{
}

void Udp6SocketImplTest::ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket = packet;
}

void Udp6SocketImplTest::ReceivePacket2 (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket2 = packet;
}

void Udp6SocketImplTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void Udp6SocketImplTest::ReceivePkt2 (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket2 = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket2->GetSize (), "ReceivedPacket size is not equal to the Rx buffer size");
  //cast availableData to void, to suppress 'availableData' set but not used
  //compiler warning
  (void) availableData;
}

void
Udp6SocketImplTest::DoSendDataTo (Ptr<Socket> socket, std::string to)
{
  Address realTo = Inet6SocketAddress (Ipv6Address (to.c_str ()), 1234);
  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (Create<Packet> (123), 0, realTo),
                         123, "200");
}

void
Udp6SocketImplTest::SendDataTo (Ptr<Socket> socket, std::string to)
{
  m_receivedPacket = Create<Packet> ();
  m_receivedPacket2 = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Udp6SocketImplTest::DoSendDataTo, this, socket, to);
  Simulator::Run ();
}

void
Udp6SocketImplTest::DoRun (void)
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
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100::1"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetUp (ifIndex);

  device = net2.Get (0);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100:1::1"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetUp (ifIndex);

  ipv6 = txNode->GetObject<Ipv6> ();
  device = net1.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100::2"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetUp (ifIndex);

  device = net2.Get (1);
  ifIndex = ipv6->GetInterfaceForDevice (device);
  ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100:1::2"), Ipv6Prefix (64));
  ipv6->AddAddress (ifIndex, ipv6Addr);
  ipv6->SetUp (ifIndex);

  // Create the UDP sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (Inet6SocketAddress (Ipv6Address ("2001:0100::1"), 1234)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Udp6SocketImplTest::ReceivePkt, this));

  Ptr<Socket> rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Udp6SocketImplTest::ReceivePkt2, this));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (Inet6SocketAddress (Ipv6Address ("2001:0100:1::1"), 1234)), 0, "trivial");

  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<UdpSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();
  txSocket->SetAllowBroadcast (true);
  // ------ Now the tests ------------

  // Unicast test
  SendDataTo (txSocket, "2001:0100::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 123, "trivial");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 0, "second interface should receive it");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Simple Link-local multicast test

  // When receiving broadcast packets, all sockets sockets bound to
  // the address/port should receive a copy of the same packet -- if
  // the socket address matches.
  rxSocket2->Dispose ();
  rxSocket2 = rxSocketFactory->CreateSocket ();
  rxSocket2->SetRecvCallback (MakeCallback (&Udp6SocketImplTest::ReceivePkt2, this));
  NS_TEST_EXPECT_MSG_EQ (rxSocket2->Bind (Inet6SocketAddress (Ipv6Address ("::"), 1234)), 0, "trivial");

  txSocket->BindToNetDevice (net1.Get (1));
  SendDataTo (txSocket, "ff02::1");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 0, "first socket should not receive it (it is bound specifically to the second interface's address");
  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket2->GetSize (), 123, "recv2: ff02::1");

  m_receivedPacket->RemoveAllByteTags ();
  m_receivedPacket2->RemoveAllByteTags ();

  // Simple getpeername tests
  Address peerAddress;
  int err = txSocket->GetPeerName (peerAddress);
  NS_TEST_EXPECT_MSG_EQ (err, -1, "socket GetPeerName() should fail when socket is not connected");
  NS_TEST_EXPECT_MSG_EQ (txSocket->GetErrno (), Socket::ERROR_NOTCONN, "socket error code should be ERROR_NOTCONN");

  Inet6SocketAddress peer ("2001:0100::1", 1234);
  err = txSocket->Connect (peer);
  NS_TEST_EXPECT_MSG_EQ (err, 0, "socket Connect() should succeed");

  err = txSocket->GetPeerName (peerAddress);
  NS_TEST_EXPECT_MSG_EQ (err, 0, "socket GetPeerName() should succeed when socket is connected");
  NS_TEST_EXPECT_MSG_EQ (peerAddress, peer, "address from socket GetPeerName() should equal the connected address");

  Simulator::Destroy ();

}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief UDP TestSuite
 */
class UdpTestSuite : public TestSuite
{
public:
  UdpTestSuite () : TestSuite ("udp", UNIT)
  {
    AddTestCase (new UdpSocketImplTest, TestCase::QUICK);
    AddTestCase (new UdpSocketLoopbackTest, TestCase::QUICK);
    AddTestCase (new Udp6SocketImplTest, TestCase::QUICK);
    AddTestCase (new Udp6SocketLoopbackTest, TestCase::QUICK);
  }
};

static UdpTestSuite g_udpTestSuite; //!< Static variable for test initialization

