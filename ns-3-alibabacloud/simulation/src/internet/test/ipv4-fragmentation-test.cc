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
 * This is the test code for ipv4-l3protocol.cc (only the fragmentation and reassembly part).
 */

#include "ns3/test.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/socket-factory.h"
#include "ns3/ipv4-raw-socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/boolean.h"

#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/udp-l4-protocol.h"
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
 * \brief Tag used in IPv4 Fragmentation Test
 */
class IPv4TestTag : public Tag {
private:
  uint64_t token; //!< Token carried by the tag.
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId () {
    static TypeId tid = TypeId ("ns3::IPv4TestTag").SetParent<Tag> ().AddConstructor<IPv4TestTag> ();
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
 * \brief IPv4 Fragmentation Test
 */
class Ipv4FragmentationTest: public TestCase
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
  bool m_broadcast;       //!< broadcast packets

public:
  virtual void DoRun (void);
  /**
   * Constructor
   * \param broadcast send broadcast packets (true) or unicast packets (false)
   */
  Ipv4FragmentationTest (bool broadcast);
  ~Ipv4FragmentationTest ();

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
  void HandleReadIcmpClient (Ipv4Address icmpSource, uint8_t icmpTtl, uint8_t icmpType,
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

};

Ipv4FragmentationTest::Ipv4FragmentationTest (bool broadcast)
  : TestCase (std::string ("Verify the IPv4 layer 3 protocol fragmentation and reassembly: ") +
              (broadcast? "broadcast": "unicast"))
{
  m_socketServer = 0;
  m_data = 0;
  m_dataSize = 0;
  m_size = 0;
  m_icmpType = 0;
  m_broadcast = broadcast;
}

Ipv4FragmentationTest::~Ipv4FragmentationTest ()
{
  if ( m_data )
    {
      delete[] m_data;
    }
  m_data = 0;
  m_dataSize = 0;
}


void
Ipv4FragmentationTest::StartServer (Ptr<Node> ServerNode)
{

  if (m_socketServer == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketServer = Socket::CreateSocket (ServerNode, tid);
      InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 9);
      m_socketServer->Bind (local);
      Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket> (m_socketServer);
      if (udpSocket)
        {
          // equivalent to setsockopt (MCAST_JOIN_GROUP)
          udpSocket->MulticastJoinGroup (0, Ipv4Address ("10.0.0.1"));
        }
   }

  m_socketServer->SetRecvCallback(MakeCallback(&Ipv4FragmentationTest::HandleReadServer, this));
}

void
Ipv4FragmentationTest::HandleReadServer (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      if (InetSocketAddress::IsMatchingType (from))
        {
          m_receivedPacketServer = packet->Copy();
        }
    }
}

void
Ipv4FragmentationTest::StartClient (Ptr<Node> ClientNode)
{

  if (m_socketClient == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketClient = Socket::CreateSocket (ClientNode, tid);
      m_socketClient->Bind ();
      m_socketClient->Connect (InetSocketAddress (Ipv4Address ("10.0.0.1"), 9));
      CallbackValue cbValue = MakeCallback(&Ipv4FragmentationTest::HandleReadIcmpClient, this);
      m_socketClient->SetAttribute ("IcmpCallback", cbValue);
      m_socketClient->SetAllowBroadcast (m_broadcast);
    }

  m_socketClient->SetRecvCallback(MakeCallback(&Ipv4FragmentationTest::HandleReadClient, this));
}

void
Ipv4FragmentationTest::HandleReadClient (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      if (InetSocketAddress::IsMatchingType (from))
        {
          m_receivedPacketClient = packet->Copy();
        }
    }
}

void
Ipv4FragmentationTest::HandleReadIcmpClient (Ipv4Address icmpSource,
                                             uint8_t icmpTtl, uint8_t icmpType,
                                             uint8_t icmpCode, uint32_t icmpInfo)
{
  m_icmpType = icmpType;
}

void
Ipv4FragmentationTest::SetFill (uint8_t *fill, uint32_t fillSize, uint32_t dataSize)
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

  memcpy(&m_data[filled], fill, dataSize - filled);

  m_size = dataSize;
}

Ptr<Packet> Ipv4FragmentationTest::SendClient (void)
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
  IPv4TestTag tag;
  tag.SetToken (42);
  p->AddPacketTag (tag);
  p->AddByteTag (tag);

  if (m_broadcast)
    {
      Address address;
      m_socketClient->GetPeerName (address);
      InetSocketAddress saddress = InetSocketAddress::ConvertFrom (address);
      saddress.SetIpv4 (Ipv4Address::GetBroadcast ());
      m_socketClient->SendTo (p, 0, saddress);
    }
  else
    {
      m_socketClient->Send (p);
    }

  return p;
}

void
Ipv4FragmentationTest::DoRun (void)
{
  // set the arp cache to something quite high
  // we shouldn't need because the NetDevice used doesn't need arp, but still
  Config::SetDefault ("ns3::ArpCache::PendingQueueSize", UintegerValue (100));

  // Create topology
  
   // Receiver Node
  Ptr<Node> serverNode = CreateObject<Node> ();
  // Sender Node
  Ptr<Node> clientNode = CreateObject<Node> ();

  NodeContainer nodes (serverNode, clientNode);

  Ptr<ErrorChannel> channel = CreateObject<ErrorChannel> ();
  channel->SetJumpingTime (Seconds (0.5));

  SimpleNetDeviceHelper helperChannel;
  helperChannel.SetNetDevicePointToPointMode (true);
  NetDeviceContainer net = helperChannel.Install (nodes, channel);

  InternetStackHelper internet;
  internet.Install (nodes);

  Ptr<Ipv4> ipv4;
  uint32_t netdev_idx;
  Ipv4InterfaceAddress ipv4Addr;

  // Receiver Node
  ipv4 = serverNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net.Get (0));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.1"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);
  Ptr<BinaryErrorModel> serverDevErrorModel = CreateObject<BinaryErrorModel> ();
  Ptr<SimpleNetDevice> serverDev = DynamicCast<SimpleNetDevice> (net.Get (0));
  serverDevErrorModel->Disable ();
  serverDev->SetMtu(1500);
  serverDev->SetReceiveErrorModel (serverDevErrorModel);
  StartServer (serverNode);

  // Sender Node
  ipv4 = clientNode->GetObject<Ipv4> ();
  netdev_idx = ipv4->AddInterface (net.Get (1));
  ipv4Addr = Ipv4InterfaceAddress (Ipv4Address ("10.0.0.2"), Ipv4Mask (0xffff0000U));
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);
  Ptr<BinaryErrorModel> clientDevErrorModel = CreateObject<BinaryErrorModel> ();
  Ptr<SimpleNetDevice> clientDev = DynamicCast<SimpleNetDevice> (net.Get (1));
  clientDevErrorModel->Disable ();
  clientDev->SetMtu(1500);
  clientDev->SetReceiveErrorModel (clientDevErrorModel);
  StartClient (clientNode);

  // some small packets, some rather big ones
  uint32_t packetSizes[5] = {1000, 2000, 5000, 10000, 65000};

  // using the alphabet
  uint8_t fillData[78];
  for ( uint32_t k=48; k<=125; k++ )
    {
      fillData[k-48] = k;
    }

  // First test: normal channel, no errors, no delays
  for( int i= 0; i<5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      m_receivedPacketServer = Create<Packet> ();
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv4FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint8_t recvBuffer[65000];

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ (recvSize, packetSizes[i], "Packet size not correct");

      m_receivedPacketServer->CopyData(recvBuffer, 65000);
      NS_TEST_EXPECT_MSG_EQ (memcmp(m_data, recvBuffer, m_receivedPacketServer->GetSize ()),
                             0, "Packet content differs");
    }

  // Second test: normal channel, no errors, delays each 2 packets.
  // Each other fragment will arrive out-of-order.
  // The packets should be received correctly since reassembly will reorder the fragments.
  channel->SetJumpingMode(true);
  for( int i= 0; i<5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      m_receivedPacketServer = Create<Packet> ();
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv4FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint8_t recvBuffer[65000];

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ (recvSize, packetSizes[i], "Packet size not correct");

      m_receivedPacketServer->CopyData(recvBuffer, 65000);
      NS_TEST_EXPECT_MSG_EQ (memcmp(m_data, recvBuffer, m_receivedPacketServer->GetSize ()),
                             0, "Packet content differs");
    }
  channel->SetJumpingMode(false);

  // Third test: normal channel, some errors, no delays.
  // The reassembly procedure should fire a timeout after 30 seconds (as specified in the RFCs).
  // Upon the timeout, the fragments received so far are discarded and an ICMP should be sent back
  // to the sender (if the first fragment has been received).
  // In this test case the first fragment is received, so we do expect an ICMP.
  // Client -> Server : errors enabled
  // Server -> Client : errors disabled (we want to have back the ICMP)
  clientDevErrorModel->Disable();
  serverDevErrorModel->Enable();
  for( int i= 1; i<5; i++)
    {
      uint32_t packetSize = packetSizes[i];

      SetFill (fillData, 78, packetSize);

      // reset the model, we want to receive the very first fragment.
      serverDevErrorModel->Reset();

      m_receivedPacketServer = Create<Packet> ();
      m_icmpType = 0;
      Simulator::ScheduleWithContext (m_socketClient->GetNode ()->GetId (), Seconds (0),
                                      &Ipv4FragmentationTest::SendClient, this);
      Simulator::Run ();

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ ((recvSize == 0), true, "Server got a packet, something wrong");
      NS_TEST_EXPECT_MSG_EQ ((m_icmpType == 11), true, "Client did not receive ICMP::TIME_EXCEEDED");
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
                                      &Ipv4FragmentationTest::SendClient, this);
      Simulator::Run ();

      IPv4TestTag packetTag;
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
          NS_TEST_EXPECT_MSG_EQ (item.GetTypeId ().GetName (), "ns3::IPv4TestTag", "ByteTag name not correct");
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
          IPv4TestTag *byteTag = dynamic_cast<IPv4TestTag *> (item.GetTypeId ().GetConstructor () ());
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
 * \brief IPv4 Fragmentation TestSuite
 */
class Ipv4FragmentationTestSuite : public TestSuite
{
public:
  Ipv4FragmentationTestSuite ();
};

Ipv4FragmentationTestSuite::Ipv4FragmentationTestSuite ()
  : TestSuite ("ipv4-fragmentation", UNIT)
{
  AddTestCase (new Ipv4FragmentationTest(false), TestCase::QUICK);
  AddTestCase (new Ipv4FragmentationTest(true), TestCase::QUICK);
}

static Ipv4FragmentationTestSuite g_ipv4fragmentationTestSuite; //!< Static variable for test initialization
