/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/test.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/qos-utils.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-net-device.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/config.h"
#include <array>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiChannelSwitchingTest");

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * This test verifies that communication between an AP and a STA resumes
 * after that both switch channel and PHY band. The channel switch is
 * scheduled to happen during the transmission of a frame sent by the AP
 * to the STA. STA discards the frame, associates with the AP again and
 * finally receives the frame successfully.
 */
class WifiChannelSwitchingTest : public TestCase
{
public:
  /**
   * \brief Constructor
   */
  WifiChannelSwitchingTest ();
  virtual ~WifiChannelSwitchingTest ();

  virtual void DoRun (void);

  /**
   * Callback invoked when a station associates with an AP. Tracks the number of
   * times the association procedure is performed.
   *
   * \param bssid the BSSID
   */
  void Associated (Mac48Address bssid);
  /**
   * Callback invoked when PHY receives a PSDU to transmit from the MAC. Tracks the
   * number of times a QoS data frame is transmitted by the AP.
   *
   * \param psduMap the PSDU map
   * \param txVector the TX vector
   * \param txPowerW the tx power in Watts
   */
  void Transmit (WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);
  /**
   * Function to trace packets received by the server application
   *
   * \param p the packet
   * \param addr the address
   */
  void L7Receive (Ptr<const Packet> p, const Address &addr);
  /**
   * Send a packet from the AP to the STA through a packet socket
   */
  void SendPacket (void);
  /**
   * Request channel switch on both AP and STA
   */
  void ChannelSwitch (void);
  /**
   * Callback invoked when the PHY on the given node changes state.
   *
   * \param nodeId the given node ID
   * \param start the time state changes
   * \param duration the time the PHY will stay in the new state
   * \param state the new PHY state
   */
  void StateChange (uint32_t nodeId, Time start, Time duration, WifiPhyState state);

private:
  NodeContainer m_apNode;             ///< AP node container
  NodeContainer m_staNode;            ///< STA node container
  NetDeviceContainer m_apDevice;      ///< AP device container
  NetDeviceContainer m_staDevice;     ///< STA device container
  uint8_t m_assocCount;               ///< count of completed Assoc Request procedures
  uint8_t m_txCount;                  ///< count of transmissions
  uint64_t m_rxBytes;                 ///< RX bytes
  uint32_t m_payloadSize;             ///< payload size in bytes
  std::array<uint8_t, 2> m_channelSwitchCount {0, 0}; ///< Per-node number of channel switch events
};

WifiChannelSwitchingTest::WifiChannelSwitchingTest ()
  : TestCase ("Test case for resuming data transmission when the recipient moves back"),
    m_assocCount (0),
    m_txCount (0),
    m_rxBytes (0),
    m_payloadSize (2000)
{
}

WifiChannelSwitchingTest::~WifiChannelSwitchingTest ()
{
}

void
WifiChannelSwitchingTest::Associated (Mac48Address bssid)
{
  m_assocCount++;
}

void
WifiChannelSwitchingTest::Transmit (WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
  for (const auto& psduPair : psduMap)
    {
      std::stringstream ss;
      ss << " " << psduPair.second->GetHeader (0).GetTypeString ()
         << " seq " << psduPair.second->GetHeader (0).GetSequenceNumber ()
         << " from " << psduPair.second->GetAddr2 ()
         << " to " << psduPair.second->GetAddr1 ();
      NS_LOG_INFO (ss.str ());
    }
  NS_LOG_INFO (" TXVECTOR " << txVector << "\n");

  if (psduMap.begin ()->second->GetHeader (0).IsQosData ())
    {
      m_txCount++;

      if (!psduMap.begin ()->second->GetHeader (0).IsRetry ())
        {
          // packet transmitted after first association. Switch channel during its
          // transmission
          Time txDuration = WifiPhy::CalculateTxDuration (psduMap, txVector, WIFI_PHY_BAND_5GHZ);
          Simulator::Schedule (txDuration / 2, &WifiChannelSwitchingTest::ChannelSwitch, this);
        }
    }
}

void
WifiChannelSwitchingTest::L7Receive (Ptr<const Packet> p, const Address &addr)
{
  if (p->GetSize () == m_payloadSize)
    {
      m_rxBytes += m_payloadSize;
    }
}

void
WifiChannelSwitchingTest::SendPacket (void)
{
  PacketSocketAddress socket;
  socket.SetSingleDevice (m_apDevice.Get (0)->GetIfIndex ());
  socket.SetPhysicalAddress (m_staDevice.Get (0)->GetAddress ());
  socket.SetProtocol (1);

  // give packet socket powers to nodes.
  PacketSocketHelper packetSocket;
  packetSocket.Install (m_staNode);
  packetSocket.Install (m_apNode);

  Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
  client->SetAttribute ("PacketSize", UintegerValue (m_payloadSize));
  client->SetAttribute ("MaxPackets", UintegerValue (1));
  client->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
  client->SetRemote (socket);
  m_apNode.Get (0)->AddApplication (client);
  client->SetStartTime (Seconds (0.5));
  client->SetStopTime (Seconds (1.0));

  Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
  server->SetLocal (socket);
  m_staNode.Get (0)->AddApplication (server);
  server->SetStartTime (Seconds (0.0));
  server->SetStopTime (Seconds (1.0));
}

void
WifiChannelSwitchingTest::ChannelSwitch (void)
{
  NS_LOG_INFO ("CHANNEL SWITCH\n");
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/ChannelSettings",
               StringValue ("{1, 20, BAND_2_4GHZ, 0}"));
}

void
WifiChannelSwitchingTest::StateChange (uint32_t nodeId, ns3::Time start, ns3::Time duration,
                                       WifiPhyState state)
{
  if (state == WifiPhyState::SWITCHING)
    {
      m_channelSwitchCount[nodeId]++;
    }
}

void
WifiChannelSwitchingTest::DoRun (void)
{
  Time simulationTime (Seconds (6.0));

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (40);
  int64_t streamNumber = 100;

  m_apNode.Create (1);
  m_staNode.Create (1);

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  SpectrumWifiPhyHelper phy;
  phy.SetChannel (spectrumChannel);
  phy.Set ("ChannelSettings", StringValue ("{36, 20, BAND_5GHZ, 0}"));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (Ssid ("channel-switching-test")));

  m_staDevice = wifi.Install (phy, mac, m_staNode);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (Ssid ("channel-switching-test")),
               "EnableBeaconJitter", BooleanValue (false));

  m_apDevice = wifi.Install (phy, mac, m_apNode);

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (m_apDevice, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (m_apNode);
  mobility.Install (m_staNode);

  SendPacket ();

  Config::ConnectWithoutContext ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::StaWifiMac/Assoc",
                                 MakeCallback (&WifiChannelSwitchingTest::Associated, this));
  Config::ConnectWithoutContext ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                                 MakeCallback (&WifiChannelSwitchingTest::Transmit, this));
  Config::ConnectWithoutContext ("/NodeList/*/ApplicationList/0/$ns3::PacketSocketServer/Rx",
                                 MakeCallback (&WifiChannelSwitchingTest::L7Receive, this));
  Config::ConnectWithoutContext ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/State/State",
                                 MakeCallback (&WifiChannelSwitchingTest::StateChange, this).Bind (0));
  Config::ConnectWithoutContext ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/State/State",
                                 MakeCallback (&WifiChannelSwitchingTest::StateChange, this).Bind (1));

  Simulator::Stop (Seconds (2));
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (+m_assocCount, 2, "STA did not associate twice");
  NS_TEST_EXPECT_MSG_EQ (+m_txCount, 2, "The QoS Data frame should have been transmitted twice by the AP");
  NS_TEST_EXPECT_MSG_EQ (m_rxBytes, m_payloadSize, "The QoS Data frame should have been received once by the STA");
  NS_TEST_EXPECT_MSG_EQ (+m_channelSwitchCount[0], 1, "AP had to perform one channel switch");
  NS_TEST_EXPECT_MSG_EQ (+m_channelSwitchCount[1], 1, "STA had to perform one channel switch");

  Simulator::Destroy ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Block Ack Test Suite
 */
class WifiChannelSwitchingTestSuite : public TestSuite
{
public:
  WifiChannelSwitchingTestSuite ();
};

WifiChannelSwitchingTestSuite::WifiChannelSwitchingTestSuite ()
  : TestSuite ("wifi-channel-switching", UNIT)
{
  AddTestCase (new WifiChannelSwitchingTest, TestCase::QUICK);
}

static WifiChannelSwitchingTestSuite g_issue211TestSuite; ///< the test suite
