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
 * Author: John Abraham <john.abraham@gatech.edu>
 * Adapted from: ipv4-raw-test.cc
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
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
#include "ns3/traffic-control-layer.h"
#include "ns3/internet-stack-helper.h"

#include <string>
#include <sstream>
#include <limits>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

using namespace ns3;


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Header Test
 */
class Ipv4HeaderTest : public TestCase
{
  Ptr<Packet> m_receivedPacket; //!< Received packet.
  Ipv4Header m_receivedHeader;  //!< Received header.

  /**
   * \brief Send a packet with speciic DSCP and ECN fields.
   * \param socket The source socket.
   * \param to The destination address.
   * \param dscp The DSCP field.
   * \param ecn The ECN field.
   */
  void DoSendData_IpHdr_Dscp (Ptr<Socket> socket, std::string to, Ipv4Header::DscpType dscp, Ipv4Header::EcnType ecn);

  /**
   * \brief Send a packet with speciic DSCP and ECN fields.
   * \param socket The source socket.
   * \param to The destination address.
   * \param dscp The DSCP field.
   * \param ecn The ECN field.
   */
  void SendData_IpHdr_Dscp (Ptr<Socket> socket, std::string to, Ipv4Header::DscpType dscp, Ipv4Header::EcnType ecn);

public:
  virtual void DoRun (void);
  Ipv4HeaderTest ();

  /**
   * \brief Receives a packet.
   * \param socket The receiving socket.
   * \param packet The packet.
   * \param from The source address.
   */
  void ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from);
  /**
   * \brief Receives a packet.
   * \param socket The receiving socket.
   */
  void ReceivePkt (Ptr<Socket> socket);
};


Ipv4HeaderTest::Ipv4HeaderTest ()
  : TestCase ("IPv4 Header Test")
{
}

void Ipv4HeaderTest::ReceivePacket (Ptr<Socket> socket, Ptr<Packet> packet, const Address &from)
{
  m_receivedPacket = packet;
}

void Ipv4HeaderTest::ReceivePkt (Ptr<Socket> socket)
{
  uint32_t availableData;
  availableData = socket->GetRxAvailable ();
  m_receivedPacket = socket->Recv (2, MSG_PEEK);
  NS_TEST_ASSERT_MSG_EQ (m_receivedPacket->GetSize (),2, "Received packet size is  not equal to 2");
  m_receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);
  NS_TEST_ASSERT_MSG_EQ (availableData, m_receivedPacket->GetSize (), "Received packet size is not equal to Rx buffer size");
  m_receivedPacket->PeekHeader (m_receivedHeader);
}

void
Ipv4HeaderTest::DoSendData_IpHdr_Dscp (Ptr<Socket> socket, std::string to, Ipv4Header::DscpType dscp, Ipv4Header::EcnType ecn)
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
  ipHeader.SetDscp (dscp);
  ipHeader.SetEcn (ecn);
  p->AddHeader (ipHeader);

  NS_TEST_EXPECT_MSG_EQ (socket->SendTo (p, 0, realTo),
                         143, to);
  socket->SetAttribute ("IpHeaderInclude", BooleanValue (false));
}

void
Ipv4HeaderTest::SendData_IpHdr_Dscp (Ptr<Socket> socket, std::string to, Ipv4Header::DscpType dscp, Ipv4Header::EcnType ecn)
{
  m_receivedPacket = Create<Packet> ();
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (0),
                                  &Ipv4HeaderTest::DoSendData_IpHdr_Dscp, this, socket, to, dscp, ecn);
  Simulator::Run ();
}

void
Ipv4HeaderTest::DoRun (void)
{
  // Create topology

  InternetStackHelper internet;
  internet.SetIpv6StackInstall (false);

  // Receiver Node
  Ptr<Node> rxNode = CreateObject<Node> ();
  internet.Install (rxNode);

  Ptr<SimpleNetDevice> rxDev1, rxDev2;
  { // first interface
    rxDev1 = CreateObject<SimpleNetDevice> ();
    rxDev1->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    rxNode->AddDevice (rxDev1);
    Ptr<Ipv4> ipv4 = rxNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (rxDev1);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.1"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
  }

  // Sender Node
  Ptr<Node> txNode = CreateObject<Node> ();
  internet.Install (txNode);
  Ptr<SimpleNetDevice> txDev1;
  {
    txDev1 = CreateObject<SimpleNetDevice> ();
    txDev1->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    txNode->AddDevice (txDev1);
    Ptr<Ipv4> ipv4 = txNode->GetObject<Ipv4> ();
    uint32_t netdev_idx = ipv4->AddInterface (txDev1);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.2"), Ipv4Mask (0xffff0000U));
    ipv4->AddAddress (netdev_idx, ipv4Addr);
    ipv4->SetUp (netdev_idx);
  }

  // link the two nodes
  Ptr<SimpleChannel> channel1 = CreateObject<SimpleChannel> ();
  rxDev1->SetChannel (channel1);
  txDev1->SetChannel (channel1);


  // Create the IPv4 Raw sockets
  Ptr<SocketFactory> rxSocketFactory = rxNode->GetObject<Ipv4RawSocketFactory> ();
  Ptr<Socket> rxSocket = rxSocketFactory->CreateSocket ();
  NS_TEST_EXPECT_MSG_EQ (rxSocket->Bind (InetSocketAddress (Ipv4Address ("0.0.0.0"), 0)), 0, "trivial");
  rxSocket->SetRecvCallback (MakeCallback (&Ipv4HeaderTest::ReceivePkt, this));


  Ptr<SocketFactory> txSocketFactory = txNode->GetObject<Ipv4RawSocketFactory> ();
  Ptr<Socket> txSocket = txSocketFactory->CreateSocket ();

  // ------ Now the tests ------------

  // Dscp Tests
  std::cout << "Dscp Test\n";

  std::vector <Ipv4Header::DscpType> vDscpTypes;
  vDscpTypes.push_back (Ipv4Header::DscpDefault);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS1);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF11);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF12);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF13);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS2);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF21);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF22);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF23);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS3);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF31);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF32);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF33);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS4);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF41);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF42);
  vDscpTypes.push_back (Ipv4Header::DSCP_AF43);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS5);
  vDscpTypes.push_back (Ipv4Header::DSCP_EF);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS6);
  vDscpTypes.push_back (Ipv4Header::DSCP_CS7);

  for (uint32_t i = 0; i < vDscpTypes.size (); i++)
    {
      SendData_IpHdr_Dscp (txSocket, "10.0.0.1", vDscpTypes [i], Ipv4Header::ECN_ECT1);
      NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv(hdrincl): 10.0.0.1");
      NS_TEST_EXPECT_MSG_EQ (m_receivedHeader.GetDscp (), vDscpTypes [i], "recv(hdrincl): 10.0.0.1");
      m_receivedHeader.Print (std::cout);
      std::cout << std::endl;
      m_receivedPacket->RemoveAllByteTags ();
      m_receivedPacket = 0;
    }

  // Ecn tests
  std::cout << "Ecn Test\n";
  std::vector <Ipv4Header::EcnType> vEcnTypes;
  vEcnTypes.push_back (Ipv4Header::ECN_NotECT);
  vEcnTypes.push_back (Ipv4Header::ECN_ECT1);
  vEcnTypes.push_back (Ipv4Header::ECN_ECT0);
  vEcnTypes.push_back (Ipv4Header::ECN_CE);

  for (uint32_t i = 0; i < vEcnTypes.size (); i++)
    {
      SendData_IpHdr_Dscp (txSocket, "10.0.0.1", Ipv4Header::DscpDefault, vEcnTypes [i]);
      NS_TEST_EXPECT_MSG_EQ (m_receivedPacket->GetSize (), 143, "recv(hdrincl): 10.0.0.1");
      NS_TEST_EXPECT_MSG_EQ (m_receivedHeader.GetEcn (), vEcnTypes [i], "recv(hdrincl): 10.0.0.1");
      m_receivedHeader.Print (std::cout);
      std::cout << std::endl;
      m_receivedPacket->RemoveAllByteTags ();
      m_receivedPacket = 0;
    }

  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 Header TestSuite
 */
class Ipv4HeaderTestSuite : public TestSuite
{
public:
  Ipv4HeaderTestSuite () : TestSuite ("ipv4-header", UNIT)
  {
    AddTestCase (new Ipv4HeaderTest, TestCase::QUICK);
  }
};

static Ipv4HeaderTestSuite g_ipv4HeaderTestSuite; //!< Static variable for test initialization
