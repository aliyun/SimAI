/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 IITP RAS
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
 * Authors: Pavel Boyko <boyko@iitp.ru>
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/mobility-helper.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/abort.h"
#include "ns3/udp-echo-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/pcap-file.h"
#include "ns3/aodv-helper.h"
#include "ns3/v4ping.h"
#include "ns3/config.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/names.h"
#include <sstream>

namespace ns3 {
namespace aodv {

/**
 * \ingroup aodv
 *
 * \brief AODV loopback UDP echo test case
 */
class LoopbackTestCase : public TestCase
{
  uint32_t m_count; //!< number of packet received;
  Ptr<Socket> m_txSocket; //!< transmit socket;
  Ptr<Socket> m_echoSocket; //!< echo socket;
  Ptr<Socket> m_rxSocket; //!< receive socket;
  uint16_t m_echoSendPort; //!< echo send port;
  uint16_t m_echoReplyPort; //!< echo reply port;

  /**
   * Send data function
   * \param socket The socket to send data
   */
  void SendData (Ptr<Socket> socket);
  /**
   * Receive packet function
   * \param socket The socket to receive data
   */
  void ReceivePkt (Ptr<Socket> socket);
  /**
   * Echo data function
   * \param socket The socket to echo data
   */
  void EchoData (Ptr<Socket> socket);

public:
  LoopbackTestCase ();
  void DoRun ();
};

LoopbackTestCase::LoopbackTestCase ()
  : TestCase ("UDP Echo 127.0.0.1 test"),
    m_count (0)
{
  m_echoSendPort = 1233;
  m_echoReplyPort = 1234;
}

void LoopbackTestCase::ReceivePkt (Ptr<Socket> socket)
{
  Ptr<Packet> receivedPacket = socket->Recv (std::numeric_limits<uint32_t>::max (), 0);

  m_count++;
}

void
LoopbackTestCase::EchoData (Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> receivedPacket = socket->RecvFrom (std::numeric_limits<uint32_t>::max (), 0, from);

  Ipv4Address src = InetSocketAddress::ConvertFrom (from).GetIpv4 ();
  Address to = InetSocketAddress (src, m_echoReplyPort);

  receivedPacket->RemoveAllPacketTags ();
  receivedPacket->RemoveAllByteTags ();

  socket->SendTo (receivedPacket, 0, to);
}

void
LoopbackTestCase::SendData (Ptr<Socket> socket)
{
  Address realTo = InetSocketAddress (Ipv4Address::GetLoopback (), m_echoSendPort);
  socket->SendTo (Create<Packet> (123), 0, realTo);

  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (1.0),
                                  &LoopbackTestCase::SendData, this, socket);
}

void
LoopbackTestCase::DoRun ()
{
  NodeContainer nodes;
  nodes.Create (1);
  Ptr<MobilityModel> m = CreateObject<ConstantPositionMobilityModel> ();
  m->SetPosition (Vector (0, 0, 0));
  nodes.Get (0)->AggregateObject (m);
  // Setup WiFi
  WifiMacHelper wifiMac;
  wifiMac.SetType ("ns3::AdhocWifiMac");
  YansWifiPhyHelper wifiPhy;
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  wifiPhy.SetChannel (wifiChannel.Create ());
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode", StringValue ("OfdmRate6Mbps"), "RtsCtsThreshold", StringValue ("2200"));
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, nodes);

  // Setup TCP/IP & AODV
  AodvHelper aodv; // Use default parameters here
  InternetStackHelper internetStack;
  internetStack.SetRoutingHelper (aodv);
  internetStack.Install (nodes);
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  // Setup echos
  Ptr<SocketFactory> socketFactory = nodes.Get (0)->GetObject<UdpSocketFactory> ();
  m_rxSocket = socketFactory->CreateSocket ();
  m_rxSocket->Bind (InetSocketAddress (Ipv4Address::GetLoopback (), m_echoReplyPort));
  m_rxSocket->SetRecvCallback (MakeCallback (&LoopbackTestCase::ReceivePkt, this));

  m_echoSocket = socketFactory->CreateSocket ();
  m_echoSocket->Bind (InetSocketAddress (Ipv4Address::GetLoopback (), m_echoSendPort));
  m_echoSocket->SetRecvCallback (MakeCallback (&LoopbackTestCase::EchoData, this));

  m_txSocket = socketFactory->CreateSocket ();

  Simulator::ScheduleWithContext (m_txSocket->GetNode ()->GetId (), Seconds (1.0),
                                  &LoopbackTestCase::SendData, this, m_txSocket);

  // Run
  Simulator::Stop (Seconds (5));
  Simulator::Run ();

  m_txSocket->Close ();
  m_echoSocket->Close ();
  m_rxSocket->Close ();

  Simulator::Destroy ();

  // Check that 4 packets delivered
  NS_TEST_ASSERT_MSG_EQ (m_count, 4, "Exactly 4 echo replies must be delivered.");
}

/**
 * \ingroup aodv-test
 * \ingroup tests
 *
 * \brief AODV Loopback test suite
 */
class AodvLoopbackTestSuite : public TestSuite
{
public:
  AodvLoopbackTestSuite () : TestSuite ("routing-aodv-loopback", SYSTEM)
  {
    SetDataDir (NS_TEST_SOURCEDIR);
    // UDP Echo loopback test case
    AddTestCase (new LoopbackTestCase (), TestCase::QUICK);
  }
} g_aodvLoopbackTestSuite; ///< the test suite


}  // namespace aodv
}  // namespace ns3
