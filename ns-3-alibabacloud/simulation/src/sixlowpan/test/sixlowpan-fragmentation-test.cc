/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze, Italy
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
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/socket-factory.h"
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

#include "ns3/sixlowpan-net-device.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/error-channel.h"

#include <string>
#include <limits>
#include <netinet/in.h>

using namespace ns3;

/**
 * \ingroup sixlowpan-tests
 *
 * \brief 6LoWPAN Fragmentation Test
 */
class SixlowpanFragmentationTest : public TestCase
{
  Ptr<Packet> m_sentPacketClient;     //!< Packet sent by client.
  Ptr<Packet> m_receivedPacketClient; //!< Packet received by the client.
  Ptr<Packet> m_receivedPacketServer; //!< packet received by the server.

  Ptr<Socket> m_socketServer; //!< Socket on the server.
  Ptr<Socket> m_socketClient; //!< Socket on the client.
  uint32_t m_dataSize;  //!< Size of the data (if any).
  uint8_t *m_data;      //!< Data to be carried in the packet
  uint32_t m_size;      //!< Size of the packet if no data has been provided.
  uint8_t m_icmpType;   //!< ICMP type.
  uint8_t m_icmpCode;   //!< ICMP code.

public:
  virtual void DoRun (void);
  SixlowpanFragmentationTest ();
  ~SixlowpanFragmentationTest ();

  // server part

  /**
   * Start the server node.
   * \param serverNode The server node.
   */
  void StartServer (Ptr<Node> serverNode);
  /**
   * Handles incoming packets in the server.
   * \param socket The receiving socket.
   */
  void HandleReadServer (Ptr<Socket> socket);

  // client part

  /**
   * Start the client node.
   * \param clientNode The client node.
   */
  void StartClient (Ptr<Node> clientNode);
  /**
   * Handles incoming packets in the client.
   * \param socket The receiving socket.
   */
  void HandleReadClient (Ptr<Socket> socket);
  /**
   * Handles incoming ICMP packets in the client.
   * \param icmpSource ICMP sender address.
   * \param icmpTtl ICMP TTL.
   * \param icmpType ICMP type.
   * \param icmpCode ICMP code.
   * \param icmpInfo ICMP info.
   */
  void HandleReadIcmpClient (Ipv6Address icmpSource, uint8_t icmpTtl, uint8_t icmpType,
                             uint8_t icmpCode, uint32_t icmpInfo);
  /**
   * Set the packet optional content.
   * \param fill Pointer to an array of data.
   * \param fillSize Size of the array of data.
   * \param dataSize Size of the packet - if fillSize is less than dataSize, the data is repeated.
   */
  void SetFill (uint8_t *fill, uint32_t fillSize, uint32_t dataSize);
  /**
   * Send a packet to the server.
   * \returns The packet sent.
   */
  Ptr<Packet> SendClient (void);

};


SixlowpanFragmentationTest::SixlowpanFragmentationTest ()
  : TestCase ("Verify the 6LoWPAN protocol fragmentation and reassembly")
{
  m_socketServer = 0;
  m_data = 0;
  m_dataSize = 0;
  m_size = 0;
  m_icmpType = 0;
  m_icmpCode = 0;
}

SixlowpanFragmentationTest::~SixlowpanFragmentationTest ()
{
  if ( m_data )
    {
      delete[] m_data;
    }
  m_data = 0;
  m_dataSize = 0;
}


void
SixlowpanFragmentationTest::StartServer (Ptr<Node> serverNode)
{

  if (m_socketServer == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketServer = Socket::CreateSocket (serverNode, tid);
      Inet6SocketAddress local = Inet6SocketAddress (Ipv6Address ("2001:0100::1"), 9);
      m_socketServer->Bind (local);
      Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket> (m_socketServer);
    }

  m_socketServer->SetRecvCallback (MakeCallback (&SixlowpanFragmentationTest::HandleReadServer, this));
}

void
SixlowpanFragmentationTest::HandleReadServer (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      if (Inet6SocketAddress::IsMatchingType (from))
        {
          packet->RemoveAllPacketTags ();
          packet->RemoveAllByteTags ();

          m_receivedPacketServer = packet->Copy ();
        }
    }
}

void
SixlowpanFragmentationTest::StartClient (Ptr<Node> clientNode)
{

  if (m_socketClient == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socketClient = Socket::CreateSocket (clientNode, tid);
      m_socketClient->Bind (Inet6SocketAddress (Ipv6Address::GetAny (), 9));
      m_socketClient->Connect (Inet6SocketAddress (Ipv6Address ("2001:0100::1"), 9));
      CallbackValue cbValue = MakeCallback (&SixlowpanFragmentationTest::HandleReadIcmpClient, this);
      m_socketClient->SetAttribute ("IcmpCallback6", cbValue);
    }

  m_socketClient->SetRecvCallback (MakeCallback (&SixlowpanFragmentationTest::HandleReadClient, this));
}

void
SixlowpanFragmentationTest::HandleReadClient (Ptr<Socket> socket)
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
SixlowpanFragmentationTest::HandleReadIcmpClient (Ipv6Address icmpSource,
                                                  uint8_t icmpTtl, uint8_t icmpType,
                                                  uint8_t icmpCode, uint32_t icmpInfo)
{
  m_icmpType = icmpType;
  m_icmpCode = icmpCode;
}

void
SixlowpanFragmentationTest::SetFill (uint8_t *fill, uint32_t fillSize, uint32_t dataSize)
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

Ptr<Packet> SixlowpanFragmentationTest::SendClient (void)
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
  m_socketClient->Send (p);

  return p;
}

void
SixlowpanFragmentationTest::DoRun (void)
{
  // Create topology
  InternetStackHelper internet;
  internet.SetIpv4StackInstall (false);
  Packet::EnablePrinting ();

  // Receiver Node
  Ptr<Node> serverNode = CreateObject<Node> ();
  internet.Install (serverNode);
  Ptr<SimpleNetDevice> serverDev;
  Ptr<BinaryErrorModel> serverDevErrorModel = CreateObject<BinaryErrorModel> ();
  {
    Ptr<Icmpv6L4Protocol> icmpv6l4 = serverNode->GetObject<Icmpv6L4Protocol> ();
    icmpv6l4->SetAttribute ("DAD", BooleanValue (false));

    serverDev = CreateObject<SimpleNetDevice> ();
    serverDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    serverDev->SetMtu (1500);
    serverDev->SetReceiveErrorModel (serverDevErrorModel);
    serverDevErrorModel->Disable ();
    serverNode->AddDevice (serverDev);

    Ptr<SixLowPanNetDevice> serverSix = CreateObject<SixLowPanNetDevice> ();
    serverSix->SetAttribute ("ForceEtherType", BooleanValue (true) );
    serverNode->AddDevice (serverSix);
    serverSix->SetNetDevice (serverDev);

    Ptr<Ipv6> ipv6 = serverNode->GetObject<Ipv6> ();
    ipv6->AddInterface (serverDev);
    uint32_t netdev_idx = ipv6->AddInterface (serverSix);
    Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100::1"), Ipv6Prefix (64));
    ipv6->AddAddress (netdev_idx, ipv6Addr);
    ipv6->SetUp (netdev_idx);

  }
  StartServer (serverNode);

  // Sender Node
  Ptr<Node> clientNode = CreateObject<Node> ();
  internet.Install (clientNode);
  Ptr<SimpleNetDevice> clientDev;
  Ptr<BinaryErrorModel> clientDevErrorModel = CreateObject<BinaryErrorModel> ();
  {
    Ptr<Icmpv6L4Protocol> icmpv6l4 = clientNode->GetObject<Icmpv6L4Protocol> ();
    icmpv6l4->SetAttribute ("DAD", BooleanValue (false));

    clientDev = CreateObject<SimpleNetDevice> ();
    clientDev->SetAddress (Mac48Address::ConvertFrom (Mac48Address::Allocate ()));
    clientDev->SetMtu (150);
    clientDev->SetReceiveErrorModel (clientDevErrorModel);
    clientDevErrorModel->Disable ();
    clientNode->AddDevice (clientDev);

    Ptr<SixLowPanNetDevice> clientSix = CreateObject<SixLowPanNetDevice> ();
    clientSix->SetAttribute ("ForceEtherType", BooleanValue (true) );
    clientNode->AddDevice (clientSix);
    clientSix->SetNetDevice (clientDev);

    Ptr<Ipv6> ipv6 = clientNode->GetObject<Ipv6> ();
    ipv6->AddInterface (clientDev);
    uint32_t netdev_idx = ipv6->AddInterface (clientSix);
    Ipv6InterfaceAddress ipv6Addr = Ipv6InterfaceAddress (Ipv6Address ("2001:0100::2"), Ipv6Prefix (64));
    ipv6->AddAddress (netdev_idx, ipv6Addr);
    ipv6->SetUp (netdev_idx);
  }
  StartClient (clientNode);

  // link the two nodes
  Ptr<ErrorChannel> channel = CreateObject<ErrorChannel> ();
  serverDev->SetChannel (channel);
  clientDev->SetChannel (channel);


  // some small packets, some rather big ones
  uint32_t packetSizes[5] = {200, 300, 400, 500, 600};

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
                                      &SixlowpanFragmentationTest::SendClient, this);
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
                                      &SixlowpanFragmentationTest::SendClient, this);
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


  // Third test: normal channel, some packets are duplicate.
  // The duplicate fragments should be discarded, so no error should be fired.
  channel->SetDuplicateMode (true);
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
                                      &SixlowpanFragmentationTest::SendClient, this);
      Simulator::Run ();

      uint8_t recvBuffer[65000];

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ (recvSize, packetSizes[i],
                             "Packet size not correct: recvSize: " << recvSize << " packetSizes[" << i << "]: " << packetSizes[i] );

      m_receivedPacketServer->CopyData (recvBuffer, 65000);
      NS_TEST_EXPECT_MSG_EQ (memcmp (m_data, recvBuffer, m_receivedPacketServer->GetSize ()),
                             0, "Packet content differs");
    }
  channel->SetDuplicateMode (false);

  // Fourth test: normal channel, some errors, no delays.
  // The reassembly procedure does NOT fire any ICMP, so we do not expect any reply from the server.
  // Client -> Server : errors enabled
  // Server -> Client : errors disabled
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
                                      &SixlowpanFragmentationTest::SendClient, this);
      Simulator::Run ();

      uint16_t recvSize = m_receivedPacketServer->GetSize ();

      NS_TEST_EXPECT_MSG_EQ ((recvSize == 0), true, "Server got a packet, something wrong");
      // Note that a 6LoWPAN fragment timeout does NOT send any ICMPv6.
    }

  Simulator::Destroy ();
}


/**
 * \ingroup sixlowpan-tests
 *
 * \brief 6LoWPAN Fragmentation TestSuite
 */
class SixlowpanFragmentationTestSuite : public TestSuite
{
public:
  SixlowpanFragmentationTestSuite ();
private:
};

SixlowpanFragmentationTestSuite::SixlowpanFragmentationTestSuite ()
  : TestSuite ("sixlowpan-fragmentation", UNIT)
{
  AddTestCase (new SixlowpanFragmentationTest (), TestCase::QUICK);
}

static SixlowpanFragmentationTestSuite g_sixlowpanFragmentationTestSuite; //!< Static variable for test initialization
