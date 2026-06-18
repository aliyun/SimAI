/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev  <andreev@iitp.ru>
 */

#include "ns3/mesh-helper.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/mobility-helper.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/abort.h"
#include "ns3/udp-echo-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/pcap-test.h"
#include <sstream>
#include "hwmp-target-flags-regression.h"

using namespace ns3;

/// Unique PCAP file name prefix
const char * const PREFIX = "hwmp-target-flags-regression-test";


HwmpDoRfRegressionTest::HwmpDoRfRegressionTest () : TestCase ("HWMP target flags regression test"),
                                                    m_nodes (0),
                                                    m_time (Seconds (5)),
                                                    m_sentPktsCounterA (0),
                                                    m_sentPktsCounterB (0),
                                                    m_sentPktsCounterC (0)
{
}

HwmpDoRfRegressionTest::~HwmpDoRfRegressionTest ()
{
  delete m_nodes;
}

void
HwmpDoRfRegressionTest::DoRun ()
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  CreateNodes ();
  CreateDevices ();
  InstallApplications ();

  Simulator::Stop (m_time);
  Simulator::Run ();
  Simulator::Destroy ();

  CheckResults ();

  delete m_nodes, m_nodes = 0;
}

void
HwmpDoRfRegressionTest::CreateNodes ()
{
  m_nodes = new NodeContainer;
  m_nodes->Create (4);
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (100),
                                 "DeltaY", DoubleValue (0),
                                 "GridWidth", UintegerValue (4),
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (*m_nodes);
}

void
HwmpDoRfRegressionTest::InstallApplications ()
{
  // server socket A
  m_serverSocketA = Socket::CreateSocket (m_nodes->Get (0), TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_serverSocketA->Bind (InetSocketAddress (Ipv4Address::GetAny (), 9));
  m_serverSocketA->SetRecvCallback (MakeCallback (&HwmpDoRfRegressionTest::HandleReadServer, this));

  // server socket B
  m_serverSocketB = Socket::CreateSocket (m_nodes->Get (3), TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_serverSocketB->Bind (InetSocketAddress (Ipv4Address::GetAny (), 10));
  m_serverSocketB->SetRecvCallback (MakeCallback (&HwmpDoRfRegressionTest::HandleReadServer, this));

  // client socket A
  m_clientSocketA = Socket::CreateSocket (m_nodes->Get (1), TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_clientSocketA->Bind ();
  m_clientSocketA->Connect (InetSocketAddress (m_interfaces.GetAddress (0), 9));
  m_clientSocketA->SetRecvCallback (MakeCallback (&HwmpDoRfRegressionTest::HandleReadClient, this));
  Simulator::ScheduleWithContext (m_clientSocketA->GetNode ()->GetId (), Seconds (2.2),
                                  &HwmpDoRfRegressionTest::SendDataA, this, m_clientSocketA);

  // client socket B
  m_clientSocketB = Socket::CreateSocket (m_nodes->Get (2), TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_clientSocketB->Bind ();
  m_clientSocketB->Connect (InetSocketAddress (m_interfaces.GetAddress (0), 9));
  m_clientSocketB->SetRecvCallback (MakeCallback (&HwmpDoRfRegressionTest::HandleReadClient, this));
  Simulator::ScheduleWithContext (m_clientSocketB->GetNode ()->GetId (), Seconds (2.0),
                                  &HwmpDoRfRegressionTest::SendDataB, this, m_clientSocketB);

  // client socket C
  m_clientSocketB = Socket::CreateSocket (m_nodes->Get (0), TypeId::LookupByName ("ns3::UdpSocketFactory"));
  m_clientSocketB->Bind ();
  m_clientSocketB->Connect (InetSocketAddress (m_interfaces.GetAddress (3), 10));
  m_clientSocketB->SetRecvCallback (MakeCallback (&HwmpDoRfRegressionTest::HandleReadClient, this));
  Simulator::ScheduleWithContext (m_clientSocketB->GetNode ()->GetId (), Seconds (2.4),
                                  &HwmpDoRfRegressionTest::SendDataC, this, m_clientSocketB);
}

void
HwmpDoRfRegressionTest::CreateDevices ()
{
  int64_t streamsUsed = 0;
  // 1. setup WiFi
  YansWifiPhyHelper wifiPhy;
  // This test suite output was originally based on YansErrorRateModel
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  Ptr<YansWifiChannel> chan = wifiChannel.Create ();
  wifiPhy.SetChannel (chan);
  // This scenario setup predates the preamble detection model
  wifiPhy.DisablePreambleDetectionModel ();

  // 2. setup mesh
  MeshHelper mesh = MeshHelper::Default ();
  mesh.SetStackInstaller ("ns3::Dot11sStack");
  mesh.SetMacType ("RandomStart", TimeValue (Seconds (0.1)));
  mesh.SetNumberOfInterfaces (1);
  NetDeviceContainer meshDevices = mesh.Install (wifiPhy, *m_nodes);
  // Four devices, ten streams per mac
  streamsUsed += mesh.AssignStreams (meshDevices, streamsUsed);
  NS_TEST_ASSERT_MSG_EQ (streamsUsed, (meshDevices.GetN () * 10), "Stream assignment mismatch");
  streamsUsed += wifiChannel.AssignStreams (chan, streamsUsed);
  NS_TEST_ASSERT_MSG_EQ (streamsUsed, (meshDevices.GetN () * 10), "Stream assignment mismatch");

  // 3. setup TCP/IP
  InternetStackHelper internetStack;
  internetStack.Install (*m_nodes);
  streamsUsed += internetStack.AssignStreams (*m_nodes, streamsUsed);
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  m_interfaces = address.Assign (meshDevices);
  // 4. write PCAP if needed
  wifiPhy.EnablePcapAll (CreateTempDirFilename (PREFIX));
}

void
HwmpDoRfRegressionTest::CheckResults ()
{
  for (int i = 0; i < 4; ++i)
    {
      NS_PCAP_TEST_EXPECT_EQ (PREFIX << "-" << i << "-1.pcap");
    }
}

void
HwmpDoRfRegressionTest::SendDataA (Ptr<Socket> socket)
{
  if ((Simulator::Now () < m_time) && (m_sentPktsCounterA < 300))
    {
      socket->Send (Create<Packet> (100));
      m_sentPktsCounterA ++;
      Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (1.0),
                                      &HwmpDoRfRegressionTest::SendDataA, this, socket);
    }
}

void
HwmpDoRfRegressionTest::SendDataB (Ptr<Socket> socket)
{
  if ((Simulator::Now () < m_time) && (m_sentPktsCounterA < 300))
    {
      socket->Send (Create<Packet> (100));
      m_sentPktsCounterB ++;
      Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (1.0),
                                      &HwmpDoRfRegressionTest::SendDataB, this, socket);
    }
}

void
HwmpDoRfRegressionTest::SendDataC (Ptr<Socket> socket)
{
  if ((Simulator::Now () < m_time) && (m_sentPktsCounterA < 300))
    {
      socket->Send (Create<Packet> (100));
      m_sentPktsCounterC ++;
      Simulator::ScheduleWithContext (socket->GetNode ()->GetId (), Seconds (1.0),
                                      &HwmpDoRfRegressionTest::SendDataC, this, socket);
    }
}

void
HwmpDoRfRegressionTest::HandleReadServer (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      packet->RemoveAllPacketTags ();
      packet->RemoveAllByteTags ();

      socket->SendTo (packet, 0, from);
    }
}

void
HwmpDoRfRegressionTest::HandleReadClient (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
    }
}
