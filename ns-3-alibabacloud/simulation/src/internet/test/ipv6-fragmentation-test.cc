/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Universita' di Firenze, Italy
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
/**
 * \file
 *
 * This is the test code for ipv6-l3protocol.cc (only the fragmentation and reassembly part).
 */

#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/socket-factory.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/ipv6-raw-socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-net-device.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/boolean.h"

#include "ns3/ipv6-static-routing.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/inet6-socket-address.h"
#
#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/udp-l4-protocol.h"

#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/traffic-control-layer.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/error-channel.h"

#include <string>
#include <limits>
#include <netinet/in.h>

using namespace ns3;

class UdpSocketImpl;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Tag used in IPv6 Fragmentation Test
 */
class IPv6TestTag : public Tag {
private:
  uint64_t token; //!< Token carried by the tag.
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId () {
    static TypeId tid = TypeId ("ns3::IPv6TestTag").SetParent<Tag> ().AddConstructor<IPv6TestTag> ();
    return tid;
  }
  virtual TypeId GetInstanceTypeId () const { return GetTypeId (); }
  virtual uint32_t GetSerializedSize () const { return sizeof (token); }
  virtual void Serialize (TagBuffer buffer) const { buffer.WriteU64 (token); }
  virtual void Deserialize (TagBuffer buffer) { token = buffer.ReadU64 (); }
  virtual void Print (std::ostream &os) const { os << "token=" << token; }
  /**
   * \brief Set the token.
   * \param token The token.
   */
  void SetToken (uint64_t token) { this->token = token; }
  /**
   * \brief Get the token.
   * \returns The token.
   */
  uint64_t GetToken () { return token; }
};


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 Fragmentation Test
 */
class Ipv6FragmentationTest : public TestCase
{
  Ptr<Packet> m_sentPacketClient;      //!< Packet sent by client.
  Ptr<Packet> m_receivedPacketClient;  //!< Packet received by client.
  Ptr<Packet> m_receivedPacketServer;  //!< Packet received by server.


  Ptr<Socket> m_socketServer;   //!< Server socket.
  Ptr<Socket> m_socketClient;   //!< Client socket.
  uint32_t m_dataSize;    //!< Data size.
  uint8_t *m_data;        //!< Data.
  uint32_t m_size;        //!< packet size.
  uint8_t m_icmpType;     //!< ICMP type.
  uint8_t m_icmpCode;     //!< ICMP code.

public:
  virtual void DoRun (void);
  Ipv6FragmentationTest ();
  ~Ipv6FragmentationTest ();

  // server part

  /**
   * \brief Start the server.
   * \param ServerNode The server.
   */
  void StartServer (Ptr<Node> ServerNode);
  /**
   * \brief Handle incoming packets.
   * \param socket The receiving socket.
   */
  void HandleReadServer (Ptr<Socket> socket);

  // client part

  /**
   * \brief Start the client.
   * \param ClientNode The client.
   */
  void StartClient (Ptr<Node> ClientNode);
  /**
   * \brief Handle incoming packets.
   * \param socket The receiving socket.
   */
  void HandleReadClient (Ptr<Socket> socket);
  /**
   * \brief Handle incoming ICMP packets.
   * \param icmpSource The ICMP sender.
   * \param icmpTtl The ICMP TTL.
   * \param icmpType The ICMP Type.
   * \param icmpCode The ICMP Code.
   * \param icmpInfo The ICMP Info.
   */
  void HandleReadIcmpClient (Ipv6Address icmpSource, uint8_t icmpTtl, uint8_t icmpType,
                             uint8_t icmpCode, uint32_t icmpInfo);

  /**
   * \brief Set the packet fill.
   * \param fill The fill.
   * \param fillSize The fill size.
   * \param dataSize The packet size.
   */
  void SetFill (uint8_t *fill, uint32_t fillSize, uint32_t dataSize);

  /**
   * \brief Send a packet.
   * \returns The sent packet.
   */
  Ptr<Packet> SendClient (void);

  /**
   * \brief Handle Server's incoming packets.
   * Ensure no packet greater than MTU is received
   *
   * \param packet the packet.
   * \param ipv6 the Ipv6 protocol.
   * \param interface the IP-level interface index.
   */
  void HandleServerRx (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface);

  /**
   * \brief Handle Client's transmitting packets.
   * Ensure no packet greater than MTU is transmitted
   *
   * \param packet the packet.
   * \param ipv6 the Ipv6 protocol.
   * \param interface the IP-level interface index.
   */
  void HandleClientTx (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface);
};


Ipv6FragmentationTest::Ipv6FragmentationTest ()
  : TestCase ("Verify the IPv6 layer 3 protocol fragmentation and reassembly")
{
  m_socketServer = 0;
  m_data = 0;
  m_dataSize = 0;
  m_size = 0;
  m_icmpType = 0;
  m_icmpCode = 0;
}

Ipv6FragmentationTest::~Ipv6FragmentationTest ()
{
  if ( m_data )
    {
      delete[] m_data;
    }
  m_data = 0;
  m_dataSize = 0;
}


void
Ipv6FragmentationTest::StartServer (Ptr<Node> ServerNode)
{

  if (m_socketServer == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketServer = Socket::CreateSocket (ServerNode, tid);
      Inet6SocketAddress local = Inet6SocketAddress (Ipv6Address ("2001::1"), 9);
      m_socketServer->Bind (local);
      Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket> (m_socketServer);
    }

  m_socketServer->SetRecvCallback (MakeCallback (&Ipv6FragmentationTest::HandleReadServer, this));
}

void
Ipv6FragmentationTest::HandleReadServer (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      if (Inet6SocketAddress::IsMatchingType (from))
        {
          m_receivedPacketServer = packet->Copy ();
        }
    }
}

void
Ipv6FragmentationTest::StartClient (Ptr<Node> ClientNode)
{

  if (m_socketClient == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketClient = Socket::CreateSocket (ClientNode, tid);
      m_socketClient->Bind (Inet6SocketAddress (Ipv6Address::GetAny (), 9));
      m_socketClient->Connect (Inet6SocketAddress (Ipv6Address ("2001::1"), 9));
      CallbackValue cbValue = MakeCallback (&Ipv6FragmentationTest::HandleReadIcmpClient, this);
      m_socketClient->SetAttribute ("IcmpCallback6", cbValue);
    }

  m_socketClient->SetRecvCallback (MakeCallback (&Ipv6FragmentationTest::HandleReadClient, this));
}

void
Ipv6FragmentationTest::HandleReadClient (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      if (Inet6SocketAddress::IsMatchingType (from))
        {
          m_receivedPacketClient = packet->Copy ();
        }
    }
}

void
Ipv6FragmentationTest::HandleReadIcmpClient (Ipv6Address icmpSource,
                                             uint8_t icmpTtl, uint8_t icmpType,
                                             uint8_t icmpCode, uint32_t icmpInfo)
{
  m_icmpType = icmpType;
  m_icmpCode = icmpCode;
}

void
Ipv6FragmentationTest::SetFill (uint8_t *fill, uint32_t fillSize, uint32_t dataSize)
{
  if (dataSize != m_dataSize)
    {
      delete [] m_data;
      m_data = new uint8_t [dataSize];
      m_dataSize = dataSize;
    }

  if (fillSize >= dataSize)
    {
      memcpy (m_data, fill, dataSize);
      return;
    }

  uint32_t filled = 0;
  while (filled + fillSize < dataSize)
    {
      memcpy (&m_data[filled], fill, fillSize);
      filled += fillSize;
    }

  memcpy (&m_data[filled], fill, dataSize - filled);

  m_size = dataSize;
}

Ptr<Packet> Ipv6FragmentationTest::SendClient (void)
{
  Ptr<Packet> p;
  if (m_dataSize)
    {
      p = Create<Packet> (m_data, m_dataSize);
    }
  else
    {
      p = Create<Packet> (m_size);
    }
  IPv6TestTag tag;
  tag.SetToken (42);
  p->AddPacketTag (tag);
  p->AddByteTag (tag);

  m_socketClient->Send (p);

  return p;
}

void Ipv6FragmentationTest::HandleServerRx (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface)
{
  NS_TEST_EXPECT_MSG_LT_OR_EQ (packet->GetSize (), ipv6->GetMtu (interface), "Received packet size > MTU: packetSizes: " << packet->GetSize ());
}

void Ipv6FragmentationTest::HandleClientTx (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface)
{
  NS_TEST_EXPECT_MSG_LT_OR_EQ (packet->GetSize (), ipv6->GetMtu (interface), "Transmitted packet size > MTU: packetSizes: " << packet->GetSize ());
}

void
Ipv6FragmentationTest::DoRun (void)
{
  // Create topology

  InternetStackHelper internet;
  internet.SetIpv4StackInstall (false);

  // Receiver Node
  Ptr<Node> serverNode = CreateObject<Node> ();
  internet.Install (serverNode);
  Ptr<SimpleNetDevice> serverDev;
  Ptr<BinaryErrorModel> serverDevErrorModel = CreateObject<BinaryErrorModel> ();
  {
    serverDev = CreateObject<SimpleNetDevice> ();
    serverDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    serverDev->SetMtu (1500);
    serverDev->SetReceiveErrorModel (serverDevErrorModel);
    serverDevErrorModel->Disable ();
    serverNode->AddDevice (serverDev);
    Ptr<Ipv6> ipv6 = serverNode->GetObject<Ipv6> ();
    uint32_t netdev_idx = ipv6->AddInterface (serverDev);
    Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001::1"), Ipv6Prefix (32));
    ipv6->AddAddress (netdev_idx, ipv6Addr);
    ipv6->SetUp (netdev_idx);
    ipv6->TraceConnectWithoutContext ("Rx", MakeCallback (&Ipv6FragmentationTest::HandleServerRx, this));
  }
  StartServer (serverNode);

  // Sender Node
  Ptr<Node> clientNode = CreateObject<Node> ();
  internet.Install (clientNode);
  Ptr<SimpleNetDevice> clientDev;
  Ptr<BinaryErrorModel> clientDevErrorModel = CreateObject<BinaryErrorModel> ();
  {
    clientDev = CreateObject<SimpleNetDevice> ();
    clientDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    clientDev->SetMtu (1500);
    clientDev->SetReceiveErrorModel (clientDevErrorModel);
    clientDevErrorModel->Disable ();
    clientNode->AddDevice (clientDev);
    Ptr<Ipv6> ipv6 = clientNode->GetObject<Ipv6> ();
    uint32_t netdev_idx = ipv6->AddInterface (clientDev);
    Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001::2"), Ipv6Prefix (32));
    ipv6->AddAddress (netdev_idx, ipv6Addr);
    ipv6->SetUp (netdev_idx);
    ipv6->TraceConnectWithoutContext ("Tx", MakeCallback (&Ipv6FragmentationTest::HandleClientTx, this));
  }
  StartClient (clientNode);

  // link the two nodes
  Ptr<ErrorChannel> channel = CreateObject<ErrorChannel> ();
  serverDev->SetChannel (channel);
  clientDev->SetChannel (channel);
  channel->SetJumpingTime (Seconds (0.5));


  // some small packets, some rather big ones
  uint32_t packetSizes[5] = {1500, 2000, 5000, 10000, 65000};

  // using the alphabet
  uint8_t fillData[78];
  for ( uint32_t k = 48; k <= 125; k++ )
    {
      fillData[k - 48] = k;
    }

  // First test: normal channel, no errors, no delays
  for ( int i = 0; i < 5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      m_receivedPacketServer = Create<Packet> ();
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv6FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint8_t recvBuffer[65000];

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ (recvSize, packetSizes[i],
                             "Packet size not correct: recvSize: " << recvSize << " packetSizes[" << i << "]: " << packetSizes[i] );

      m_receivedPacketServer->CopyData (recvBuffer, 65000);
      NS_TEST_EXPECT_MSG_EQ (memcmp (m_data, recvBuffer, m_receivedPacketServer->GetSize ()),
                             0, "Packet content differs");
    }

  // Second test: normal channel, no errors, delays each 2 packets.
  // Each other fragment will arrive out-of-order.
  // The packets should be received correctly since reassembly will reorder the fragments.
  channel->SetJumpingMode (true);
  for ( int i = 0; i < 5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      m_receivedPacketServer = Create<Packet> ();
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv6FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint8_t recvBuffer[65000];

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ (recvSize, packetSizes[i],
                             "Packet size not correct: recvSize: " << recvSize << " packetSizes[" << i << "]: " << packetSizes[i] );

      m_receivedPacketServer->CopyData (recvBuffer, 65000);
      NS_TEST_EXPECT_MSG_EQ (memcmp (m_data, recvBuffer, m_receivedPacketServer->GetSize ()),
                             0, "Packet content differs");
    }
  channel->SetJumpingMode (false);

  // Third test: normal channel, some errors, no delays.
  // The reassembly procedure should fire a timeout after 30 seconds (as specified in the RFCs).
  // Upon the timeout, the fragments received so far are discarded and an ICMP should be sent back
  // to the sender (if the first fragment has been received).
  // In this test case the first fragment is received, so we do expect an ICMP.
  // Client -> Server : errors enabled
  // Server -> Client : errors disabled (we want to have back the ICMP)
  clientDevErrorModel->Disable ();
  serverDevErrorModel->Enable ();
  for ( int i = 1; i < 5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      // reset the model, we want to receive the very first fragment.
      serverDevErrorModel->Reset ();

      m_receivedPacketServer = Create<Packet> ();
      m_icmpType = 0;
      m_icmpCode = 0;
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv6FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ ((recvSize == 0), true, "Server got a packet, something wrong");
      NS_TEST_EXPECT_MSG_EQ ((m_icmpType == Icmpv6Header::ICMPV6_ERROR_TIME_EXCEEDED)
                             && (m_icmpCode == Icmpv6Header::ICMPV6_FRAGTIME),
                             true, "Client did not receive ICMPv6::TIME_EXCEEDED " << int(m_icmpType) << int(m_icmpCode) );
    }

  // Fourth test: normal channel, no errors, no delays.
  // We check tags
  clientDevErrorModel->Disable ();
  serverDevErrorModel->Disable ();
  for (int i= 0; i<5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      m_receivedPacketServer = Create<Packet> ();
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv6FragmentationTest::SendClient, this);
      Simulator::Run ();

      IPv6TestTag packetTag;
      bool found = m_receivedPacketServer->PeekPacketTag (packetTag);

      NS_TEST_EXPECT_MSG_EQ (found, true, "PacketTag not found");
      NS_TEST_EXPECT_MSG_EQ (packetTag.GetToken (), 42, "PacketTag value not correct");

      ByteTagIterator iter = m_receivedPacketServer->GetByteTagIterator ();

      uint32_t end = 0;
      uint32_t tagStart = 0;
      uint32_t tagEnd = 0;
      while (iter.HasNext ())
        {
          ByteTagIterator::Item item = iter.Next ();
          NS_TEST_EXPECT_MSG_EQ (item.GetTypeId ().GetName (), "ns3::IPv6TestTag", "ByteTag name not correct");
          tagStart = item.GetStart ();
          tagEnd = item.GetEnd ();
          if (end == 0)
            {
              NS_TEST_EXPECT_MSG_EQ (tagStart, 0, "First ByteTag Start not correct");
            }
          if (end != 0)
            {
              NS_TEST_EXPECT_MSG_EQ (tagStart, end, "ByteTag End not correct");
            }
          end = tagEnd;
          IPv6TestTag *byteTag = dynamic_cast<IPv6TestTag *> (item.GetTypeId ().GetConstructor () ());
          NS_TEST_EXPECT_MSG_NE (byteTag, 0, "ByteTag not found");
          item.GetTag (*byteTag);
          NS_TEST_EXPECT_MSG_EQ (byteTag->GetToken (), 42, "ByteTag value not correct");
          delete byteTag;
        }
      NS_TEST_EXPECT_MSG_EQ (end, m_receivedPacketServer->GetSize (), "trivial");
    }

  Simulator::Destroy ();
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 Fragmentation TestSuite
 */
class Ipv6FragmentationTestSuite : public TestSuite
{
public:
  Ipv6FragmentationTestSuite () : TestSuite ("ipv6-fragmentation", UNIT)
  {
    AddTestCase (new Ipv6FragmentationTest, TestCase::QUICK);
  }
};

static Ipv6FragmentationTestSuite g_ipv6fragmentationTestSuite; //!< Static variable for test initialization

