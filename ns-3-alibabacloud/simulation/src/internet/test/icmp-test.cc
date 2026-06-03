/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan.
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
 * Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 */

// Test program for the Internet Control Message Protocol (ICMP) responses.
//
// IcmpEchoReplyTestCase scenario:
//
//               n0 <------------------> n1
//              i(0,0)                 i(1,0)
//
//        Test that sends a single ICMP echo packet with TTL = 1 from n0 to n1,
//        n1 receives the packet and send an ICMP echo reply.
//
//
// IcmpTimeExceedTestCase scenario:
//
//                           channel1            channel2
//               n0 <------------------> n1 <---------------------> n2
//              i(0,0)                 i(1,0)                     i2(1,0)
//                                     i2(0,0)
//
//         Test that sends a single ICMP echo packet with TTL = 1 from n0 to n4,
//         however, the TTL is not enough and n1 reply to n0 with an ICMP time exceed.
//
//
// IcmpV6EchoReplyTestCase scenario:
//
//               n0 <-------------------> n1
//              i(0,1)                  i(1,1)
//
//         Test that sends a single ICMPV6 ECHO request with hopLimit = 1 from n0 to n1,
//         n1 receives the packet and send an ICMPV6 echo reply.
//
// IcmpV6TimeExceedTestCase scenario:
//
//                        channel1                channel2
//               n0 <------------------> n1 <---------------------> n2
//              i(0,0)                  i(1,0)                    i2(1,0)
//                                      i2(0,0)
//
//         Test that sends a single ICMPV6 echo packet with hopLimit = 1 from n0 to n4,
//         however, the hopLimit is not enough and n1 reply to n0 with an ICMPV6 time exceed error.


#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simulator.h"
#include "ns3/icmpv6-header.h"
#include "ns3/icmpv4.h"
#include "ns3/socket.h"
#include "ns3/socket-factory.h"
#include "ns3/uinteger.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/ipv6-routing-helper.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/internet-stack-helper.h"

#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup internet-apps
 * \defgroup icmp-test ICMP protocol tests
 */


/**
 * \ingroup icmp-test
 * \ingroup tests
 *
 * \brief ICMP  Echo Reply Test
 */
class IcmpEchoReplyTestCase : public TestCase
{
public:
  IcmpEchoReplyTestCase ();
  virtual ~IcmpEchoReplyTestCase ();

  /**
   * Send data
   * \param socket output socket
   * \param dst destination address
   */
  void SendData (Ptr<Socket> socket, Ipv4Address dst);
  /**
   * Receive data
   * \param socket input socket
   */
  void ReceivePkt (Ptr<Socket> socket);

private:
  virtual void DoRun (void);
  Ptr<Packet> m_receivedPacket; //!< received packet

};


IcmpEchoReplyTestCase::IcmpEchoReplyTestCase ()
  : TestCase ("ICMP:EchoReply test case")
{

}


IcmpEchoReplyTestCase::~IcmpEchoReplyTestCase ()
{

}


void
IcmpEchoReplyTestCase::SendData (Ptr<Socket> socket, Ipv4Address dst)
{
  Ptr<Packet> p = Create<Packet> ();
  Icmpv4Echo echo;
  echo.SetSequenceNumber (1);
  echo.SetIdentifier (0);
  p->AddHeader (echo);

  Icmpv4Header header;
  header.SetType (Icmpv4Header::ICMPV4_ECHO);
  header.SetCode (0);
  p->AddHeader (header);

  Address realTo = InetSocketAddress (dst, 1234);

  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (p, 0, realTo),
                         (int) p->GetSize (), " Unable to send ICMP Echo Packet");

}


void
IcmpEchoReplyTestCase::ReceivePkt (Ptr <Socket> socket)
{
  Address from;
  Ptr<Packet> p = socket->RecvFrom (0xffffffff, 0, from);
  m_receivedPacket = p->Copy ();

  Ipv4Header ipv4;
  p->RemoveHeader (ipv4);
  NS_TEST_EXPECT_MSG_EQ (ipv4.GetProtocol (), 1," The received Packet is not an ICMP packet");

  Icmpv4Header icmp;
  p->RemoveHeader (icmp);

  NS_TEST_EXPECT_MSG_EQ (icmp.GetType (), Icmpv4Header::ICMPV4_ECHO_REPLY,
                         " The received Packet is not a ICMPV4_ECHO_REPLY");
}


void
IcmpEchoReplyTestCase::DoRun ()
{
  NodeContainer n;
  n.Create (2);

  InternetStackHelper internet;
  internet.Install (n);

  // link the two nodes
  Ptr<SimpleNetDevice> txDev = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> rxDev = CreateObject<SimpleNetDevice> ();
  n.Get (0)->AddDevice (txDev);
  n.Get (1)->AddDevice (rxDev);
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  rxDev->SetChannel (channel1);
  txDev->SetChannel (channel1);
  NetDeviceContainer d;
  d.Add (txDev);
  d.Add (rxDev);

  Ipv4AddressHelper ipv4;

  ipv4.SetBase ("10.0.0.0", "255.255.255.252");
  Ipv4InterfaceContainer i = ipv4.Assign (d);

  Ptr<Socket> socket;
  socket = Socket::CreateSocket (n.Get (0), TypeId::LookupByName ("ns3::Ipv4RawSocketFactory"));
  socket->SetAttribute ("Protocol", UintegerValue (1)); // ICMP protocol
  socket->SetRecvCallback (MakeCallback (&IcmpEchoReplyTestCase::ReceivePkt, this));

  InetSocketAddress src = InetSocketAddress (Ipv4Address::GetAny (), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->Bind (src),0," Socket Binding failed");

  // Set a TTL big enough
  socket->SetIpTtl (1);
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &IcmpEchoReplyTestCase::SendData, this, socket, i.GetAddress (1,0));
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 28, " Unexpected ICMPV4_ECHO_REPLY packet size");


  Simulator::Destroy ();
}


/**
 * \ingroup icmp-test
 * \ingroup tests
 *
 * \brief ICMP Time Exceed Reply Test
 */
class IcmpTimeExceedTestCase : public TestCase
{
public:
  IcmpTimeExceedTestCase ();
  virtual ~IcmpTimeExceedTestCase ();

  /**
   * Send data
   * \param socket output socket
   * \param dst destination address
   */
  void SendData (Ptr<Socket> socket, Ipv4Address dst);
  /**
   * Receive data
   * \param socket input socket
   */
  void ReceivePkt (Ptr<Socket> socket);

private:
  virtual void DoRun (void);
  Ptr<Packet> m_receivedPacket; //!< received packet

};


IcmpTimeExceedTestCase::IcmpTimeExceedTestCase ()
  : TestCase ("ICMP:TimeExceedReply test case")
{

}


IcmpTimeExceedTestCase::~IcmpTimeExceedTestCase ()
{

}


void
IcmpTimeExceedTestCase::SendData (Ptr<Socket> socket, Ipv4Address dst)
{
  Ptr<Packet> p = Create<Packet> ();
  Icmpv4Echo echo;
  echo.SetSequenceNumber (1);
  echo.SetIdentifier (0);
  p->AddHeader (echo);

  Icmpv4Header header;
  header.SetType (Icmpv4Header::ICMPV4_ECHO);
  header.SetCode (0);
  p->AddHeader (header);

  Address realTo = InetSocketAddress (dst, 1234);

  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (p, 0, realTo),
                         (int) p->GetSize (), " Unable to send ICMP Echo Packet");
}


void
IcmpTimeExceedTestCase::ReceivePkt (Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> p = socket->RecvFrom (0xffffffff, 0, from);
  m_receivedPacket = p->Copy ();

  Ipv4Header ipv4;
  p->RemoveHeader (ipv4);
  NS_TEST_EXPECT_MSG_EQ (ipv4.GetProtocol (), 1,"The received packet is not an ICMP packet");

  NS_TEST_EXPECT_MSG_EQ (ipv4.GetSource (),Ipv4Address ("10.0.0.2"),
                         "ICMP Time Exceed Response should come from 10.0.0.2");

  Icmpv4Header icmp;
  p->RemoveHeader (icmp);

  NS_TEST_EXPECT_MSG_EQ (icmp.GetType (), Icmpv4Header::ICMPV4_TIME_EXCEEDED,
                         "The received packet is not a ICMPV4_TIME_EXCEEDED packet ");
}


void
IcmpTimeExceedTestCase::DoRun ()
{
  NodeContainer n, n0n1,n1n2;
  n.Create (3);
  n0n1.Add (n.Get (0));
  n0n1.Add (n.Get (1));
  n1n2.Add (n.Get (1));
  n1n2.Add (n.Get (2));

  Ptr<SimpleChannel> channel = CreateObject <SimpleChannel> ();
  Ptr<SimpleChannel> channel2 = CreateObject <SimpleChannel> ();

  SimpleNetDeviceHelper simpleHelper;
  simpleHelper.SetNetDevicePointToPointMode (true);

  SimpleNetDeviceHelper simpleHelper2;
  simpleHelper2.SetNetDevicePointToPointMode (true);

  NetDeviceContainer devices;
  devices = simpleHelper.Install (n0n1,channel);
  NetDeviceContainer devices2;
  devices2 = simpleHelper2.Install (n1n2,channel2);

  InternetStackHelper internet;
  internet.Install (n);

  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0","255.255.255.255");
  Ipv4InterfaceContainer i = address.Assign (devices);

  address.SetBase ("10.0.1.0","255.255.255.255");
  Ipv4InterfaceContainer i2 = address.Assign (devices2);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Ptr<Socket> socket;
  socket = Socket::CreateSocket (n.Get (0), TypeId::LookupByName ("ns3::Ipv4RawSocketFactory"));
  socket->SetAttribute ("Protocol", UintegerValue (1)); // ICMP protocol
  socket->SetRecvCallback (MakeCallback (&IcmpTimeExceedTestCase::ReceivePkt, this));

  InetSocketAddress src = InetSocketAddress (Ipv4Address::GetAny (), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->Bind (src),0," Socket Binding failed");


  // The ttl is not big enough , causing an ICMP Time Exceeded response
  socket->SetIpTtl (1);
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &IcmpTimeExceedTestCase::SendData, this, socket, i2.GetAddress (1,0));
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 56, " Unexpected ICMP Time Exceed Response packet size");

  Simulator::Destroy ();
}


/**
 * \ingroup icmp-test
 * \ingroup tests
 *
 * \brief ICMPV6  Echo Reply Test
 */
class IcmpV6EchoReplyTestCase : public TestCase
{
public:
  IcmpV6EchoReplyTestCase ();
  virtual ~IcmpV6EchoReplyTestCase ();

  /**
   * Send data
   * \param socket output socket
   * \param dst destination address
   */
  void SendData (Ptr<Socket> socket, Ipv6Address dst);
  /**
   * Receive data
   * \param socket input socket
   */
  void ReceivePkt (Ptr<Socket> socket);

private:
  virtual void DoRun (void);
  Ptr<Packet> m_receivedPacket; //!< received packet

};


IcmpV6EchoReplyTestCase::IcmpV6EchoReplyTestCase ()
  : TestCase ("ICMPV6:EchoReply test case")
{

}


IcmpV6EchoReplyTestCase::~IcmpV6EchoReplyTestCase ()
{

}


void
IcmpV6EchoReplyTestCase::SendData (Ptr<Socket> socket, Ipv6Address dst)
{
  Ptr<Packet> p = Create<Packet> ();
  Icmpv6Echo echo (1);
  echo.SetSeq (1);
  echo.SetId (0XB1ED);
  p->AddHeader (echo);

  Icmpv6Header header;
  header.SetType (Icmpv6Header::ICMPV6_ECHO_REQUEST);
  header.SetCode (0);
  p->AddHeader (header);

  Address realTo = Inet6SocketAddress (dst, 1234);

  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (p, 0, realTo),
                         (int) p->GetSize (), " Unable to send ICMP Echo Packet");

}


void
IcmpV6EchoReplyTestCase::ReceivePkt (Ptr <Socket> socket)
{
  Address from;
  Ptr<Packet> p = socket->RecvFrom (from);
  Ptr<Packet> pkt = p->Copy ();

  if (Inet6SocketAddress::IsMatchingType (from))
    {
      Ipv6Header ipv6;
      p->RemoveHeader (ipv6);

      NS_TEST_EXPECT_MSG_EQ (ipv6.GetNextHeader (),
                             Ipv6Header::IPV6_ICMPV6,
                             "The received Packet is not an ICMPV6 packet");
      Icmpv6Header icmpv6;
      p->RemoveHeader (icmpv6);

      if (icmpv6.GetType () == Icmpv6Header::ICMPV6_ECHO_REPLY)
        {
          m_receivedPacket = pkt->Copy ();
        }
    }
}


void
IcmpV6EchoReplyTestCase::DoRun ()
{
  NodeContainer n;
  n.Create (2);

  InternetStackHelper internet;
  internet.Install (n);

  // link the two nodes
  Ptr<SimpleNetDevice> txDev = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> rxDev = CreateObject<SimpleNetDevice> ();
  txDev->SetAddress (Mac48Address ("00:00:00:00:00:01"));
  rxDev->SetAddress (Mac48Address ("00:00:00:00:00:02"));
  n.Get (0)->AddDevice (txDev);
  n.Get (1)->AddDevice (rxDev);
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  rxDev->SetChannel (channel1);
  txDev->SetChannel (channel1);
  NetDeviceContainer d;
  d.Add (txDev);
  d.Add (rxDev);

  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i = ipv6.Assign (d);

  Ptr<Socket> socket;
  socket = Socket::CreateSocket (n.Get (0), TypeId::LookupByName ("ns3::Ipv6RawSocketFactory"));
  socket->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
  socket->SetRecvCallback (MakeCallback (&IcmpV6EchoReplyTestCase::ReceivePkt, this));

  Inet6SocketAddress src = Inet6SocketAddress (Ipv6Address::GetAny (), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->Bind (src),0," SocketV6 Binding failed");

  // Set a TTL big enough
  socket->SetIpTtl (1);

  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &IcmpV6EchoReplyTestCase::SendData, this, socket, i.GetAddress (1,1));
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 52, " Unexpected ICMPV6_ECHO_REPLY packet size");

  Simulator::Destroy ();
}


/**
 * \ingroup icmp-test
 * \ingroup tests
 *
 * \brief ICMPV6  Time Exceed response test
 */
class IcmpV6TimeExceedTestCase : public TestCase
{
public:
  IcmpV6TimeExceedTestCase ();
  virtual ~IcmpV6TimeExceedTestCase ();

  /**
   * Send data
   * \param socket output socket
   * \param dst destination address
   */
  void SendData (Ptr<Socket> socket, Ipv6Address dst);
  /**
   * Receive data
   * \param socket input socket
   */
  void ReceivePkt (Ptr<Socket> socket);

private:
  virtual void DoRun (void);
  Ptr<Packet> m_receivedPacket; //!< received packet

};


IcmpV6TimeExceedTestCase::IcmpV6TimeExceedTestCase ()
  : TestCase ("ICMPV6:TimeExceed test case")
{

}


IcmpV6TimeExceedTestCase::~IcmpV6TimeExceedTestCase ()
{

}


void
IcmpV6TimeExceedTestCase::SendData (Ptr<Socket> socket, Ipv6Address dst)
{
  Ptr<Packet> p = Create<Packet> ();
  Icmpv6Echo echo (1);
  echo.SetSeq (1);
  echo.SetId (0XB1ED);
  p->AddHeader (echo);

  Icmpv6Header header;
  header.SetType (Icmpv6Header::ICMPV6_ECHO_REQUEST);
  header.SetCode (0);
  p->AddHeader (header);

  Address realTo = Inet6SocketAddress (dst, 1234);


  socket->SendTo (p, 0, realTo);

}


void
IcmpV6TimeExceedTestCase::ReceivePkt (Ptr <Socket> socket)
{
  Address from;
  Ptr<Packet> p = socket->RecvFrom (from);
  Ptr<Packet> pkt = p->Copy ();

  if (Inet6SocketAddress::IsMatchingType (from))
    {
      Ipv6Header ipv6;
      p->RemoveHeader (ipv6);


      NS_TEST_EXPECT_MSG_EQ (ipv6.GetNextHeader (),
                             Ipv6Header::IPV6_ICMPV6,
                             "The received Packet is not an ICMPV6 packet");

      Icmpv6Header icmpv6;
      p->RemoveHeader (icmpv6);

      // Ignore any packet except ICMPV6_ERROR_TIME_EXCEEDED
      if (icmpv6.GetType () == Icmpv6Header::ICMPV6_ERROR_TIME_EXCEEDED)
        {
          m_receivedPacket = pkt->Copy ();
        }
    }
}


void
IcmpV6TimeExceedTestCase::DoRun ()
{
  NodeContainer n, n0n1,n1n2;
  n.Create (3);
  n0n1.Add (n.Get (0));
  n0n1.Add (n.Get (1));
  n1n2.Add (n.Get (1));
  n1n2.Add (n.Get (2));

  Ptr<SimpleChannel> channel = CreateObject <SimpleChannel> ();
  Ptr<SimpleChannel> channel2 = CreateObject <SimpleChannel> ();

  SimpleNetDeviceHelper simpleHelper;
  simpleHelper.SetNetDevicePointToPointMode (true);

  SimpleNetDeviceHelper simpleHelper2;
  simpleHelper2.SetNetDevicePointToPointMode (true);

  NetDeviceContainer devices;
  devices = simpleHelper.Install (n0n1,channel);

  NetDeviceContainer devices2;
  devices2 = simpleHelper2.Install (n1n2,channel2);

  InternetStackHelper internet;
  internet.Install (n);

  Ipv6AddressHelper address;

  address.NewNetwork ();
  address.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));

  Ipv6InterfaceContainer interfaces = address.Assign (devices);
  interfaces.SetForwarding (1,true);
  interfaces.SetDefaultRouteInAllNodes (1);
  address.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer interfaces2 = address.Assign (devices2);

  interfaces2.SetForwarding (0,true);
  interfaces2.SetDefaultRouteInAllNodes (0);

  Ptr<Socket> socket;
  socket = Socket::CreateSocket (n.Get (0), TypeId::LookupByName ("ns3::Ipv6RawSocketFactory"));
  socket->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
  socket->SetRecvCallback (MakeCallback (&IcmpV6TimeExceedTestCase::ReceivePkt, this));

  Inet6SocketAddress src = Inet6SocketAddress (Ipv6Address::GetAny (), 0);
  NS_TEST_EXPECT_MSG_EQ (socket->Bind (src),0," SocketV6 Binding failed");

  // In Ipv6 TTL is renamed hop limit in IPV6.
  // The hop limit is not big enough , causing an ICMPV6 Time Exceeded error
  socket->SetIpv6HopLimit (1);

  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &IcmpV6TimeExceedTestCase::SendData, this, socket, interfaces2.GetAddress (1,1));
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 100, " Unexpected ICMPV6_ECHO_REPLY packet size");

  Simulator::Destroy ();
}


/**
 * \ingroup icmp-test
 * \ingroup tests
 *
 * \brief ICMP TestSuite
 */

class IcmpTestSuite : public TestSuite
{
public:
  IcmpTestSuite ();
};

IcmpTestSuite::IcmpTestSuite ()
  : TestSuite ("icmp", UNIT)
{
  AddTestCase (new IcmpEchoReplyTestCase, TestCase::QUICK);
  AddTestCase (new IcmpTimeExceedTestCase, TestCase::QUICK);
  AddTestCase (new IcmpV6EchoReplyTestCase, TestCase::QUICK);
  AddTestCase (new IcmpV6TimeExceedTestCase, TestCase::QUICK);
}

static IcmpTestSuite icmpTestSuite; //!< Static variable for test initialization

