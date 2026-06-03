/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
 * Copyright (c) 2009 INRIA
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
 * Authors: Ken Renard <kenneth.d.renard.ctr@mail.mil>
 *
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/config.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv6-static-routing.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"

#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/traffic-control-layer.h"

#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ipv6DualStackTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 dual stack Test
 */
class DualStackTestCase : public TestCase
{
public:
  DualStackTestCase ();
private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /**
   * \brief Setup the test.
   */
  void SetUpSim ();

  Ptr<Node> node0;  //!< Node 0.
  Ptr<Node> node1;  //!< Node 1.

  /**
   * Handle connection created (1).
   * \param s The socket.
   * \param addr The peer address.
   */
  void ServerHandleConnectionCreated1 (Ptr<Socket> s, const Address & addr);
  /**
   * Handle connection created (2).
   * \param s The socket.
   * \param addr The peer address.
   */
  void ServerHandleConnectionCreated2 (Ptr<Socket> s, const Address & addr);
  /**
   * Handle connection created (3).
   * \param s The socket.
   * \param addr The peer address.
   */
  void ServerHandleConnectionCreated3 (Ptr<Socket> s, const Address & addr);
  /**
   * Handle connection created (4).
   * \param s The socket.
   * \param addr The peer address.
   */
  void ServerHandleConnectionCreated4 (Ptr<Socket> s, const Address & addr);

  Ptr<Socket> server1;  //!< Server socket (1).
  Ptr<Socket> server2;  //!< Server socket (2).
  Ptr<Socket> server3;  //!< Server socket (3).
  Ptr<Socket> server4;  //!< Server socket (4).

  Ptr<Socket> source1;  //!< Sending socket (1).
  Ptr<Socket> source2;  //!< Sending socket (2).
  Ptr<Socket> source3;  //!< Sending socket (3).
  Ptr<Socket> source4;  //!< Sending socket (4).

  Address receivedAddr1;  //!< Received address (1).
  Address receivedAddr2;  //!< Received address (2).
  Address receivedAddr3;  //!< Received address (3).
  Address receivedAddr4;  //!< Received address (4).
};

Ptr<Node>
CreateDualStackNode ()
{
  Ptr<Node> node = CreateObject<Node> ();

  // Traffic Control
  Ptr<TrafficControlLayer> tc = CreateObject<TrafficControlLayer> ();
  node->AggregateObject (tc);

  //ARP
  Ptr<ArpL3Protocol> arp = CreateObject<ArpL3Protocol> ();
  node->AggregateObject (arp);
  arp->SetTrafficControl (tc);

  //IPV4
  Ptr<Ipv4L3Protocol> ipv4 = CreateObject<Ipv4L3Protocol> ();
  //Routing for Ipv4
  Ptr<Ipv4ListRouting> ipv4Routing = CreateObject<Ipv4ListRouting> ();
  ipv4->SetRoutingProtocol (ipv4Routing);
  Ptr<Ipv4StaticRouting> ipv4staticRouting = CreateObject<Ipv4StaticRouting> ();
  ipv4Routing->AddRoutingProtocol (ipv4staticRouting, 0);
  node->AggregateObject (ipv4);

  //ICMPv4
  Ptr<Icmpv4L4Protocol> icmp = CreateObject<Icmpv4L4Protocol> ();
  node->AggregateObject (icmp);

  //UDP
  Ptr<UdpL4Protocol> udp = CreateObject<UdpL4Protocol> ();
  node->AggregateObject (udp);

  //TCP
  Ptr<TcpL4Protocol> tcp = CreateObject<TcpL4Protocol> ();
  node->AggregateObject (tcp);

  //IPV6
  Ptr<Ipv6L3Protocol> ipv6 = CreateObject<Ipv6L3Protocol> ();

  //Routing for Ipv6
  Ptr<Ipv6ListRouting> ipv6Routing = CreateObject<Ipv6ListRouting> ();
  ipv6->SetRoutingProtocol (ipv6Routing);
  Ptr<Ipv6StaticRouting> ipv6staticRouting = CreateObject<Ipv6StaticRouting> ();
  ipv6Routing->AddRoutingProtocol (ipv6staticRouting, 0);
  node->AggregateObject (ipv6);

  //ICMPv6
  Ptr<Icmpv6L4Protocol> icmp6 = CreateObject<Icmpv6L4Protocol> ();
  node->AggregateObject (icmp6);

  //Ipv6 Extensions
  ipv6->RegisterExtensions ();
  ipv6->RegisterOptions ();

  return node;
}

Ptr<SimpleNetDevice>
AddSimpleNetDevice (Ptr<Node> node, Ipv4Address v4Addr, Ipv4Mask v4Mask,
                    Ipv6Address v6Addr, Ipv6Prefix v6Prefix)
{
  Ptr<SimpleNetDevice> dev = CreateObject<SimpleNetDevice> ();
  dev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
  node->AddDevice (dev);

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  uint32_t ndid = ipv4->AddInterface (dev);
  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (v4Addr, v4Mask);
  ipv4->AddAddress (ndid, ipv4Addr);
  ipv4->SetUp (ndid);

  Ptr<Ipv6> ipv6 = node->GetObject<Ipv6> ();
  ndid = ipv6->AddInterface (dev);
  Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (v6Addr, v6Prefix);
  ipv6->AddAddress (ndid, ipv6Addr);
  ipv6->SetUp (ndid);

  return dev;
}

void
DualStackTestCase::SetUpSim ()
{
  node0 = CreateDualStackNode ();
  node1 = CreateDualStackNode ();

  Ptr<SimpleNetDevice> dev0 = AddSimpleNetDevice(node0, Ipv4Address("10.0.0.1"),
            Ipv4Mask(0xffffff00), Ipv6Address("2001::1"), Ipv6Prefix(64));
  Ptr<SimpleNetDevice> dev1 = AddSimpleNetDevice(node1, Ipv4Address("10.0.0.2"),
            Ipv4Mask(0xffffff00), Ipv6Address("2001::2"), Ipv6Prefix(64));

  Ptr<SimpleChannel> channel = CreateObject<SimpleChannel> ();
  dev0->SetChannel (channel);
  dev1->SetChannel (channel);

  Ptr<SocketFactory> sockFactory0 = node0->GetObject<TcpSocketFactory> ();
  Ptr<SocketFactory> sockFactory1 = node1->GetObject<TcpSocketFactory> ();

  server1 = sockFactory0->CreateSocket ();
  server2 = sockFactory0->CreateSocket ();
  server3 = sockFactory0->CreateSocket ();
  server4 = sockFactory0->CreateSocket ();

  source1 = sockFactory1->CreateSocket ();
  source2 = sockFactory1->CreateSocket ();
  source3 = sockFactory1->CreateSocket ();
  source4 = sockFactory1->CreateSocket ();

  return;
}

void
DualStackTestCase::ServerHandleConnectionCreated1 (Ptr<Socket> s, const Address & addr)
{
  receivedAddr1 = addr;
  return;
}

void
DualStackTestCase::ServerHandleConnectionCreated2 (Ptr<Socket> s, const Address & addr)
{
  receivedAddr2 = addr;
  return;
}

void
DualStackTestCase::ServerHandleConnectionCreated3 (Ptr<Socket> s, const Address & addr)
{
  receivedAddr3 = addr;
  return;
}

void
DualStackTestCase::ServerHandleConnectionCreated4 (Ptr<Socket> s, const Address & addr)
{
  receivedAddr4 = addr;
  return;
}


DualStackTestCase::DualStackTestCase ()
  : TestCase ("DualStackTestCase")
{
  receivedAddr1 = Address ();
  receivedAddr2 = Address ();
  receivedAddr3 = Address ();
  receivedAddr4 = Address ();
}

void
DualStackTestCase::DoRun (void)
{
  SetUpSim ();

  uint16_t port1 = 5000;
  uint16_t port2 = 5001;
  uint16_t port3 = 5002;
  uint16_t port4 = 5003;

  /* Server 1: listen on 0.0.0.0 for IPv4 connection */
  server1->Bind (InetSocketAddress(Ipv4Address::GetAny(), port1));
  server1->Listen ();
  server1->SetAcceptCallback (MakeNullCallback<bool, Ptr< Socket >, const Address &> (),
                 MakeCallback (&DualStackTestCase::ServerHandleConnectionCreated1, this));

  /* Server 2: listen on 0.0.0.0 for IPv4 connection - should reject IPv6 */
  server2->Bind (InetSocketAddress(Ipv4Address::GetAny(), port2));
  server2->Listen ();
  server2->SetAcceptCallback (MakeNullCallback<bool, Ptr< Socket >, const Address &> (),
                 MakeCallback (&DualStackTestCase::ServerHandleConnectionCreated2, this));

  /* Server 3: listen on :: for IPv4 (mapped into IPv6 address) connection */
  server3->Bind (Inet6SocketAddress(Ipv6Address::GetAny(), port3));
  server3->Listen ();
  server3->SetAcceptCallback (MakeNullCallback<bool, Ptr< Socket >, const Address &> (),
                 MakeCallback (&DualStackTestCase::ServerHandleConnectionCreated3, this));

  /* Server 4: listen on :: for IPv6 connection */
  server4->Bind (Inet6SocketAddress(Ipv6Address::GetAny(), port4));
  server4->Listen ();
  server4->SetAcceptCallback (MakeNullCallback<bool, Ptr< Socket >, const Address &> (),
                 MakeCallback (&DualStackTestCase::ServerHandleConnectionCreated4, this));


  /* Source 1: connect to server 1 via IPv4 */
  source1->Connect(InetSocketAddress(Ipv4Address("10.0.0.1"), port1));

  /* Source 2: connect to server 2 via IPv6 */
  source2->Connect(Inet6SocketAddress(Ipv6Address("2001::1"), port2));

  /* Source 3: connect to server 3 via IPv4 */
  source3->Connect(InetSocketAddress(Ipv4Address("10.0.0.1"), port3));

  /* Source 4: connect to server 4 via IPv6 */
  source4->Connect(Inet6SocketAddress(Ipv6Address("2001::1"), port4));

  Simulator::Run();

  /* Server 1: should have connection from Ipv4 address of Node 1 */
  NS_TEST_EXPECT_MSG_EQ (InetSocketAddress::IsMatchingType(receivedAddr1), true, "Accepted address is of proper type");
  NS_TEST_EXPECT_MSG_EQ(InetSocketAddress::ConvertFrom(receivedAddr1).GetIpv4 (), Ipv4Address("10.0.0.2"), "Accepted address is correct");

  /* Server 2: should have no connection */
  NS_TEST_EXPECT_MSG_EQ(receivedAddr2.IsInvalid(), true, "IPv4 socket correctly ignored IPv6 connection");

  /* Server 3: should have connection from Ipv4-mapped IPv6 address of Node 1 */
  NS_TEST_EXPECT_MSG_EQ (Inet6SocketAddress::IsMatchingType(receivedAddr3), true, "Accepted address is of proper type");
  NS_TEST_EXPECT_MSG_EQ(Inet6SocketAddress::ConvertFrom(receivedAddr3).GetIpv6 (), Ipv6Address("::ffff:0a00:0002"), "Accepted address is correct");

  /* Server 4: should have connection from IPv6 address of Node 1 */
  NS_TEST_EXPECT_MSG_EQ (Inet6SocketAddress::IsMatchingType(receivedAddr4), true, "Accepted address is of proper type");
  NS_TEST_EXPECT_MSG_EQ(Inet6SocketAddress::ConvertFrom(receivedAddr4).GetIpv6 (), Ipv6Address("2001::2"), "Accepted address is correct");
}

void
DualStackTestCase::DoTeardown (void)
{
  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 dual stack TestSuite
 */
class Ipv6DualStackTestSuite : public TestSuite
{
public:
  Ipv6DualStackTestSuite ()
    : TestSuite ("ipv6-dual-stack", UNIT)
  {
    AddTestCase (new DualStackTestCase(), TestCase::QUICK);
  }
};

static Ipv6DualStackTestSuite g_ipv6DualStackTestSuite; //!< Static variable for test initialization
