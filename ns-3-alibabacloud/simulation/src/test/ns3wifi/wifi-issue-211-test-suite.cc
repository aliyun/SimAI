/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020
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
 * Authors: Stefano Avallone <stavallo@unina.it>
 *          Rémy Grünblatt <remy@grunblatt.org>
 */

#include "ns3/test.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/qos-utils.h"
#include "ns3/packet.h"
#include "ns3/wifi-net-device.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/config.h"
#include "ns3/queue-size.h"

using namespace ns3;


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test for issue 211 (https://gitlab.com/nsnam/ns-3-dev/-/issues/211)
 *
 * This test aims to check that the transmission of data frames (under a
 * Block Ack agreement) resumes after a period in which the connectivity
 * between originator and recipient is interrupted (e.g., the station
 * moves away and comes back). Issue 211 revealed that MSDUs with expired
 * lifetime were not removed from the wifi MAC queue if the station had
 * to transmit a Block Ack Request. If the connectivity was lost for enough
 * time, the wifi MAC queue could become full of MSDUs with expired lifetime,
 * thus preventing the traffic control layer to forward down new packets.
 * At this point, the station gave up transmitting the Block Ack Request
 * and did not request channel access anymore.
 */
class Issue211Test : public TestCase
{
public:
  /**
   * \brief Constructor
   */
  Issue211Test ();
  virtual ~Issue211Test ();

  virtual void DoRun (void);

private:
  /**
   * Compute the average throughput since the last check-point
   * \param server the UDP server
   */
  void CalcThroughput (Ptr<UdpServer> server);

  std::vector<double> m_tputValues;   ///< throughput in sub-intervals
  uint64_t m_lastRxBytes;             ///< RX bytes at last check-point
  Time m_lastCheckPointTime;          ///< time of last check-point
  uint32_t m_payloadSize;             ///< payload size in bytes
};

Issue211Test::Issue211Test ()
  : TestCase ("Test case for resuming data transmission when the recipient moves back"),
    m_lastRxBytes (0),
    m_lastCheckPointTime (Seconds (0)),
    m_payloadSize (2000)
{
}

Issue211Test::~Issue211Test ()
{
}

void
Issue211Test::CalcThroughput (Ptr<UdpServer> server)
{
  uint64_t rxBytes = m_payloadSize * server->GetReceived ();
  double tput = (rxBytes - m_lastRxBytes) * 8. / (Simulator::Now () - m_lastCheckPointTime).ToDouble (Time::US);  // Mb/s
  m_tputValues.push_back (tput);
  m_lastRxBytes = rxBytes;
  m_lastCheckPointTime = Simulator::Now ();
}

void
Issue211Test::DoRun (void)
{
  Time simulationTime (Seconds (6.0));
  Time moveAwayTime (Seconds (2.0));
  Time moveBackTime (Seconds (4.0));

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (40);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  SpectrumWifiPhyHelper phy;
  phy.SetChannel (spectrumChannel);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("HtMcs0"),
                                "ControlMode", StringValue ("HtMcs0"));

  Config::SetDefault ("ns3::WifiMacQueue::MaxSize", QueueSizeValue (QueueSize ("50p")));

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (Ssid ("issue211-test")));

  NetDeviceContainer staDevices = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (Ssid ("issue211-test")),
               "EnableBeaconJitter", BooleanValue (false));

  NetDeviceContainer apDevices = wifi.Install (phy, mac, wifiApNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (apDevices, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  /* Internet stack*/
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNode);

  Ipv4AddressHelper address;
  address.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterface;
  Ipv4InterfaceContainer apNodeInterface;

  staNodeInterface = address.Assign (staDevices.Get (0));
  apNodeInterface = address.Assign (apDevices.Get (0));

  ApplicationContainer serverApp;
  Time warmup (Seconds (1.0));  // to account for association

  uint16_t port = 9;
  UdpServerHelper server (port);
  serverApp = server.Install (wifiStaNode.Get (0));
  serverApp.Start (Seconds (0.0));
  serverApp.Stop (warmup + simulationTime);

  UdpClientHelper client (staNodeInterface.GetAddress (0), port);
  client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
  client.SetAttribute ("Interval", TimeValue (MilliSeconds (1)));
  client.SetAttribute ("PacketSize", UintegerValue (m_payloadSize));  // 16 Mb/s
  ApplicationContainer clientApp = client.Install (wifiApNode.Get (0));
  clientApp.Start (warmup);
  clientApp.Stop (warmup + simulationTime);

  Ptr<MobilityModel> staMobility = wifiStaNode.Get (0)->GetObject<MobilityModel> ();

  // First check-point: station moves away
  Simulator::Schedule (warmup + moveAwayTime, &MobilityModel::SetPosition, staMobility,
                       Vector (10000.0, 0.0, 0.0));
  Simulator::Schedule (warmup + moveAwayTime + MilliSeconds (10), &Issue211Test::CalcThroughput, this,
                       DynamicCast<UdpServer> (serverApp.Get (0)));

  // Second check-point: station moves back
  Simulator::Schedule (warmup + moveBackTime, &MobilityModel::SetPosition, staMobility,
                       Vector (5.0, 0.0, 0.0));
  Simulator::Schedule (warmup + moveBackTime, &Issue211Test::CalcThroughput, this,
                       DynamicCast<UdpServer> (serverApp.Get (0)));

  // Last check-point: simulation finish time
  Simulator::Schedule (warmup + simulationTime, &Issue211Test::CalcThroughput, this,
                       DynamicCast<UdpServer> (serverApp.Get (0)));

  Simulator::Stop (warmup + simulationTime);
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_tputValues.size (), 3, "Unexpected number of throughput values");
  NS_TEST_EXPECT_MSG_GT (m_tputValues[0], 0, "Throughput must be non null before station moves away");
  NS_TEST_EXPECT_MSG_EQ (m_tputValues[1], 0,"Throughput must be null while the station is away");
  NS_TEST_EXPECT_MSG_GT (m_tputValues[2], 0, "Throughput must be non null when the station is back");

  // Print throughput values when the test is run through test-runner
  for (const auto& t : m_tputValues)
    {
      std::cout << "Throughput = " << t << " Mb/s" << std::endl;
    }

  Simulator::Destroy ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Block Ack Test Suite
 */
class Issue211TestSuite : public TestSuite
{
public:
  Issue211TestSuite ();
};

Issue211TestSuite::Issue211TestSuite ()
  : TestSuite ("wifi-issue-211", UNIT)
{
  AddTestCase (new Issue211Test, TestCase::QUICK);
}

static Issue211TestSuite g_issue211TestSuite; ///< the test suite
