 /* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
#include "ns3/packet.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/single-model-spectrum-channel.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/config.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-ppdu.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/qos-txop.h"

using namespace ns3;


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test TXOP rules
 *
 *
 */
class WifiTxopTest : public TestCase
{
public:
  /**
   * Constructor
   * \param pifsRecovery whether PIFS recovery is used after failure of a non-initial frame
   */
  WifiTxopTest (bool pifsRecovery);
  virtual ~WifiTxopTest ();

  /**
   * Function to trace packets received by the server application
   * \param context the context
   * \param p the packet
   * \param addr the address
   */
  void L7Receive (std::string context, Ptr<const Packet> p, const Address &addr);
  /**
   * Callback invoked when PHY receives a PSDU to transmit
   * \param context the context
   * \param psduMap the PSDU map
   * \param txVector the TX vector
   * \param txPowerW the tx power in Watts
   */
  void Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);
  /**
   * Check correctness of transmitted frames
   */
  void CheckResults (void);

private:
  void DoRun (void) override;

  /// Information about transmitted frames
  struct FrameInfo
  {
    Time txStart;                         ///< Frame start TX time
    Time txDuration;                      ///< Frame TX duration
    WifiMacHeader header;                 ///< Frame MAC header
    WifiTxVector txVector;                ///< TX vector used to transmit the frame
  };

  uint16_t m_nStations;                   ///< number of stations
  NetDeviceContainer m_staDevices;        ///< container for stations' NetDevices
  NetDeviceContainer m_apDevices;         ///< container for AP's NetDevice
  std::vector<FrameInfo> m_txPsdus;       ///< transmitted PSDUs
  Time m_txopLimit;                       ///< TXOP limit
  uint8_t m_aifsn;                        ///< AIFSN for BE
  uint32_t m_cwMin;                       ///< CWmin for BE
  uint16_t m_received;                    ///< number of packets received by the stations
  bool m_pifsRecovery;                    ///< whether to use PIFS recovery
};

WifiTxopTest::WifiTxopTest (bool pifsRecovery)
  : TestCase ("Check correct operation within TXOPs"),
    m_nStations (3),
    m_txopLimit (MicroSeconds (4768)),
    m_received (0),
    m_pifsRecovery (pifsRecovery)
{
}

WifiTxopTest::~WifiTxopTest ()
{
}

void
WifiTxopTest::L7Receive (std::string context, Ptr<const Packet> p, const Address &addr)
{
  if (p->GetSize () >= 500)
    {
      m_received++;
    }
}

void
WifiTxopTest::Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
  // Log all transmitted frames that are not beacon frames and have been transmitted
  // after 400ms (so as to skip association requests/responses)
  if (!psduMap.begin ()->second->GetHeader (0).IsBeacon ()
      && Simulator::Now () > MilliSeconds (400))
    {
      m_txPsdus.push_back ({Simulator::Now (),
                            WifiPhy::CalculateTxDuration (psduMap, txVector, WIFI_PHY_BAND_5GHZ),
                            psduMap[SU_STA_ID]->GetHeader (0), txVector});
    }

  // Print all the transmitted frames if the test is executed through test-runner
  std::cout << Simulator::Now () << " " << psduMap.begin ()->second->GetHeader (0).GetTypeString ()
            << " seq " << psduMap.begin ()->second->GetHeader (0).GetSequenceNumber ()
            << " to " << psduMap.begin ()->second->GetAddr1 ()
            << " TX duration " << WifiPhy::CalculateTxDuration (psduMap, txVector, WIFI_PHY_BAND_5GHZ)
            << " duration/ID " << psduMap.begin ()->second->GetHeader (0).GetDuration () << std::endl;
}

void
WifiTxopTest::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (40);
  int64_t streamNumber = 100;

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (m_nStations);

  Ptr<SingleModelSpectrumChannel> spectrumChannel = CreateObject<SingleModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  SpectrumWifiPhyHelper phy;
  phy.SetChannel (spectrumChannel);

  Config::SetDefault ("ns3::QosFrameExchangeManager::PifsRecovery", BooleanValue (m_pifsRecovery));
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", UintegerValue (1900));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("OfdmRate12Mbps"),
                                "ControlMode", StringValue ("OfdmRate6Mbps"));

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac",
               "QosSupported", BooleanValue (true),
               "Ssid", SsidValue (Ssid ("non-existent-ssid")));

  m_staDevices = wifi.Install (phy, mac, wifiStaNodes);

  mac.SetType ("ns3::ApWifiMac",
               "QosSupported", BooleanValue (true),
               "Ssid", SsidValue (Ssid ("wifi-txop-ssid")),
               "BeaconInterval", TimeValue (MicroSeconds (102400)),
               "EnableBeaconJitter", BooleanValue (false));

  m_apDevices = wifi.Install (phy, mac, wifiApNode);

  // schedule association requests at different times
  Time init = MilliSeconds (100);
  Ptr<WifiNetDevice> dev;

  for (uint16_t i = 0; i < m_nStations; i++)
    {
      dev = DynamicCast<WifiNetDevice> (m_staDevices.Get (i));
      Simulator::Schedule (init + i * MicroSeconds (102400), &WifiMac::SetSsid,
                           dev->GetMac (), Ssid ("wifi-txop-ssid"));
    }

  // Assign fixed streams to random variables in use
  wifi.AssignStreams (m_apDevices, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  positionAlloc->Add (Vector (0.0, 1.0, 0.0));
  positionAlloc->Add (Vector (-1.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNodes);

  // set the TXOP limit on BE AC
  dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (0));
  PointerValue ptr;
  dev->GetMac ()->GetAttribute ("BE_Txop", ptr);
  ptr.Get<QosTxop> ()->SetTxopLimit (m_txopLimit);
  m_aifsn = ptr.Get<QosTxop> ()->GetAifsn ();
  m_cwMin = ptr.Get<QosTxop> ()->GetMinCw ();

  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiApNode);
  packetSocket.Install (wifiStaNodes);

  // DL frames
  for (uint16_t i = 0; i < m_nStations; i++)
    {
      PacketSocketAddress socket;
      socket.SetSingleDevice (m_apDevices.Get (0)->GetIfIndex ());
      socket.SetPhysicalAddress (m_staDevices.Get (i)->GetAddress ());
      socket.SetProtocol (1);

      // Send one QoS data frame (not protected by RTS/CTS) to each station
      Ptr<PacketSocketClient> client1 = CreateObject<PacketSocketClient> ();
      client1->SetAttribute ("PacketSize", UintegerValue (500));
      client1->SetAttribute ("MaxPackets", UintegerValue (1));
      client1->SetAttribute ("Interval", TimeValue (MicroSeconds (1)));
      client1->SetRemote (socket);
      wifiApNode.Get (0)->AddApplication (client1);
      client1->SetStartTime (MilliSeconds (410));
      client1->SetStopTime (Seconds (1.0));

      // Send one QoS data frame (protected by RTS/CTS) to each station
      Ptr<PacketSocketClient> client2 = CreateObject<PacketSocketClient> ();
      client2->SetAttribute ("PacketSize", UintegerValue (2000));
      client2->SetAttribute ("MaxPackets", UintegerValue (1));
      client2->SetAttribute ("Interval", TimeValue (MicroSeconds (1)));
      client2->SetRemote (socket);
      wifiApNode.Get (0)->AddApplication (client2);
      client2->SetStartTime (MilliSeconds (520));
      client2->SetStopTime (Seconds (1.0));

      Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
      server->SetLocal (socket);
      wifiStaNodes.Get (i)->AddApplication (server);
      server->SetStartTime (Seconds (0.0));
      server->SetStopTime (Seconds (1.0));
    }

  // The AP does not correctly receive the Ack sent in response to the QoS
  // data frame sent to the first station
  Ptr<ReceiveListErrorModel> apPem = CreateObject<ReceiveListErrorModel> ();
  apPem->SetList ({9});
  dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (0));
  dev->GetMac ()->GetWifiPhy ()->SetPostReceptionErrorModel (apPem);

  // The second station does not correctly receive the first QoS
  // data frame sent by the AP
  Ptr<ReceiveListErrorModel> sta2Pem = CreateObject<ReceiveListErrorModel> ();
  sta2Pem->SetList ({24});
  dev = DynamicCast<WifiNetDevice> (m_staDevices.Get (1));
  dev->GetMac ()->GetWifiPhy ()->SetPostReceptionErrorModel (sta2Pem);

  // UL Traffic (the first station sends one frame to the AP)
    {
      PacketSocketAddress socket;
      socket.SetSingleDevice (m_staDevices.Get (0)->GetIfIndex ());
      socket.SetPhysicalAddress (m_apDevices.Get (0)->GetAddress ());
      socket.SetProtocol (1);

      Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient> ();
      client->SetAttribute ("PacketSize", UintegerValue (1500));
      client->SetAttribute ("MaxPackets", UintegerValue (1));
      client->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
      client->SetRemote (socket);
      wifiStaNodes.Get (0)->AddApplication (client);
      client->SetStartTime (MilliSeconds (412));
      client->SetStopTime (Seconds (1.0));

      Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
      server->SetLocal (socket);
      wifiApNode.Get (0)->AddApplication (server);
      server->SetStartTime (Seconds (0.0));
      server->SetStopTime (Seconds (1.0));
    }

  Config::Connect ("/NodeList/*/ApplicationList/*/$ns3::PacketSocketServer/Rx",
                   MakeCallback (&WifiTxopTest::L7Receive, this));
  // Trace PSDUs passed to the PHY on all devices
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                   MakeCallback (&WifiTxopTest::Transmit, this));

  Simulator::Stop (Seconds (1));
  Simulator::Run ();

  CheckResults ();

  Simulator::Destroy ();
}

void
WifiTxopTest::CheckResults (void)
{
  Time tEnd,                           // TX end for a frame
       tStart,                         // TX start fot the next frame
       txopStart,                      // TXOP start time
       tolerance = NanoSeconds (50),   // due to propagation delay
       sifs = DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetPhy ()->GetSifs (),
       slot = DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetPhy ()->GetSlot (),
       navEnd;

  // lambda to round Duration/ID (in microseconds) up to the next higher integer
  auto RoundDurationId = [] (Time t)
    {
      return MicroSeconds (ceil (static_cast<double> (t.GetNanoSeconds ()) / 1000));
    };

  /*
   * Verify the different behavior followed when an initial/non-initial frame of a TXOP
   * fails. Also, verify that a CF-end frame is sent if enough time remains in the TXOP.
   * The destination of failed frames is put in square brackets below.
   *
   *          |---NAV-----till end TXOP--------->|
   *          |      |----NAV--till end TXOP---->|
   *          |      |              |---------------------------NAV---------------------------------->|
   *          |      |              |      |--------------------------NAV---------------------------->|
   *          |      |              |      |      |------------------------NAV----------------------->|
   *          |      |              |      |      |                  |-------------NAV--------------->|
   *     Start|      |         Start|      |      |                  |      |----------NAV----------->|
   *     TXOP |      |         TXOP |      |      |   Ack            |      |      |-------NAV------->|
   *      |   |      |          |   |      |      | Timeout          |      |      |      |---NAV---->|
   *      |---|  |---|-backoff->|---|  |---|  |---|   |-PIFS or->|---|  |---|  |---|  |---|  |-----|
   *      |QoS|  |Ack|          |QoS|  |Ack|  |QoS|   |-backoff->|QoS|  |Ack|  |QoS|  |Ack|  |CFend|
   * ----------------------------------------------------------------------------------------------------
   * From:  AP    STA1            AP    STA1    AP                 AP    STA2    AP    STA3    AP
   *   To: STA1   [AP]           STA1    AP   [STA2]              STA2    AP    STA3    AP     all
   */

  NS_TEST_ASSERT_MSG_EQ (m_txPsdus.size (), 25, "Expected 25 transmitted frames");

  // the first frame sent after 400ms is a QoS data frame sent by the AP to STA1
  txopStart = m_txPsdus[0].txStart;

  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[0].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[0].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the first station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[0].header.GetDuration (),
                         RoundDurationId (m_txopLimit - m_txPsdus[0].txDuration),
                         "Duration/ID of the first frame must cover the whole TXOP");

  // a Normal Ack is sent by STA1
  tEnd = m_txPsdus[0].txStart + m_txPsdus[0].txDuration;
  tStart = m_txPsdus[1].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the first frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the first frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[1].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[1].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[1].header.GetDuration (),
                         RoundDurationId (m_txPsdus[0].header.GetDuration () - sifs - m_txPsdus[1].txDuration),
                         "Duration/ID of the Ack must be derived from that of the first frame");

  // the AP receives a corrupted Ack in response to the frame it sent, which is the initial
  // frame of a TXOP. Hence, the TXOP is terminated and the AP retransmits the frame after
  // invoking the backoff
  txopStart = m_txPsdus[2].txStart;

  tEnd = m_txPsdus[1].txStart + m_txPsdus[1].txDuration;
  tStart = m_txPsdus[2].txStart;

  NS_TEST_ASSERT_MSG_GT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot,
                               "Less than AIFS elapsed between AckTimeout and the next TXOP start");
  NS_TEST_ASSERT_MSG_LT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot + (2 * (m_cwMin + 1) - 1) * slot,
                               "More than AIFS+BO elapsed between AckTimeout and the next TXOP start");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[2].header.IsQosData (), true,
                         "Expected to retransmit a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[2].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected to retransmit a frame to the first station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[2].header.GetDuration (),
                         RoundDurationId (m_txopLimit - m_txPsdus[2].txDuration),
                         "Duration/ID of the retransmitted frame must cover the whole TXOP");

  // a Normal Ack is then sent by STA1
  tEnd = m_txPsdus[2].txStart + m_txPsdus[2].txDuration;
  tStart = m_txPsdus[3].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the first frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the first frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[3].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[3].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[3].header.GetDuration (),
                         RoundDurationId (m_txPsdus[2].header.GetDuration () - sifs - m_txPsdus[3].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // the AP sends a frame to STA2
  tEnd = m_txPsdus[3].txStart + m_txPsdus[3].txDuration;
  tStart = m_txPsdus[4].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Second frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Second frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[4].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[4].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (1))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the second station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[4].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[4].txStart - txopStart) - m_txPsdus[4].txDuration),
                         "Duration/ID of the second frame does not cover the remaining TXOP");

  // STA2 receives a corrupted frame and hence it does not send the Ack. When the AckTimeout
  // expires, the AP performs PIFS recovery or invoke backoff, without terminating the TXOP,
  // because a non-initial frame of the TXOP failed
  tEnd = m_txPsdus[4].txStart + m_txPsdus[4].txDuration
         + sifs + slot + WifiPhy::CalculatePhyPreambleAndHeaderDuration (m_txPsdus[4].txVector); // AckTimeout
  tStart = m_txPsdus[5].txStart;

  if (m_pifsRecovery)
    {
      NS_TEST_ASSERT_MSG_EQ (tEnd + sifs + slot, tStart, "Second frame must have been sent after a PIFS");
    }
  else
    {
      NS_TEST_ASSERT_MSG_GT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot,
                                   "Less than AIFS elapsed between AckTimeout and the next transmission");
      NS_TEST_ASSERT_MSG_LT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot + (2 * (m_cwMin + 1) - 1) * slot,
                                   "More than AIFS+BO elapsed between AckTimeout and the next TXOP start");
    }
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[5].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[5].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (1))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the second station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[5].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[5].txStart - txopStart) - m_txPsdus[5].txDuration),
                         "Duration/ID of the second frame does not cover the remaining TXOP");

  // a Normal Ack is then sent by STA2
  tEnd = m_txPsdus[5].txStart + m_txPsdus[5].txDuration;
  tStart = m_txPsdus[6].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the second frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the second frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[6].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[6].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[6].header.GetDuration (),
                         RoundDurationId (m_txPsdus[5].header.GetDuration () - sifs - m_txPsdus[6].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // the AP sends a frame to STA3
  tEnd = m_txPsdus[6].txStart + m_txPsdus[6].txDuration;
  tStart = m_txPsdus[7].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Third frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Third frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[7].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[7].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (2))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the third station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[7].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[7].txStart - txopStart) - m_txPsdus[7].txDuration),
                         "Duration/ID of the third frame does not cover the remaining TXOP");

  // a Normal Ack is then sent by STA3
  tEnd = m_txPsdus[7].txStart + m_txPsdus[7].txDuration;
  tStart = m_txPsdus[8].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the third frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the third frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[8].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[8].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[8].header.GetDuration (),
                         RoundDurationId (m_txPsdus[7].header.GetDuration () - sifs - m_txPsdus[8].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // the TXOP limit is such that enough time for sending a CF-End frame remains
  tEnd = m_txPsdus[8].txStart + m_txPsdus[8].txDuration;
  tStart = m_txPsdus[9].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "CF-End sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "CF-End sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[9].header.IsCfEnd (), true, "Expected a CF-End frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[9].header.GetDuration (), Seconds (0),
                        "Duration/ID must be set to 0 for CF-End frames");

  // the CF-End frame resets the NAV on STA1, which can now transmit
  tEnd = m_txPsdus[9].txStart + m_txPsdus[9].txDuration;
  tStart = m_txPsdus[10].txStart;

  NS_TEST_ASSERT_MSG_GT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot,
                               "Less than AIFS elapsed between two TXOPs");
  NS_TEST_ASSERT_MSG_LT_OR_EQ (tStart - tEnd, sifs + m_aifsn * slot + m_cwMin * slot + tolerance,
                               "More than AIFS+BO elapsed between two TXOPs");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[10].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[10].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the first station to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[10].header.GetDuration (),
                         RoundDurationId (m_txopLimit - m_txPsdus[10].txDuration),
                         "Duration/ID of the frame sent by the first station does not cover the remaining TXOP");

  // a Normal Ack is then sent by the AP
  tEnd = m_txPsdus[10].txStart + m_txPsdus[10].txDuration;
  tStart = m_txPsdus[11].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[11].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[11].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the first station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[11].header.GetDuration (),
                         RoundDurationId (m_txPsdus[10].header.GetDuration () - sifs - m_txPsdus[11].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // the TXOP limit is such that enough time for sending a CF-End frame remains
  tEnd = m_txPsdus[11].txStart + m_txPsdus[11].txDuration;
  tStart = m_txPsdus[12].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "CF-End sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "CF-End sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[12].header.IsCfEnd (), true, "Expected a CF-End frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[12].header.GetDuration (), Seconds (0),
                        "Duration/ID must be set to 0 for CF-End frames");


  /*
   * Verify that the Duration/ID of RTS/CTS frames is set correctly, that the TXOP holder is
   * kept and allows stations to ignore NAV properly and that the CF-End Frame is not sent if
   * not enough time remains
   *
   *          |---------------------------------------------NAV---------------------------------->|
   *          |      |-----------------------------------------NAV------------------------------->|
   *          |      |      |-------------------------------------NAV---------------------------->|
   *          |      |      |      |---------------------------------NAV------------------------->|
   *          |      |      |      |      |-----------------------------NAV---------------------->|
   *          |      |      |      |      |      |-------------------------NAV------------------->|
   *          |      |      |      |      |      |      |---------------------NAV---------------->|
   *          |      |      |      |      |      |      |      |-----------------NAV------------->|
   *          |      |      |      |      |      |      |      |      |-------------NAV---------->|
   *          |      |      |      |      |      |      |      |      |      |---------NAV------->|
   *          |      |      |      |      |      |      |      |      |      |      |-----NAV---->|
   *          |      |      |      |      |      |      |      |      |      |      |      |-NAV->|
   *      |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|
   *      |RTS|  |CTS|  |QoS|  |Ack|  |RTS|  |CTS|  |QoS|  |Ack|  |RTS|  |CTS|  |QoS|  |Ack|
   * ----------------------------------------------------------------------------------------------------
   * From:  AP    STA1    AP    STA1    AP    STA2    AP    STA2    AP    STA3    AP    STA3
   *   To: STA1    AP    STA1    AP    STA2    AP    STA2    AP    STA3    AP    STA3    AP
   */

  // the first frame is an RTS frame sent by the AP to STA1
  txopStart = m_txPsdus[13].txStart;

  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[13].header.IsRts (), true,
                         "Expected an RTS frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[13].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected an RTS frame sent by the AP to the first station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[13].header.GetDuration (),
                         RoundDurationId (m_txopLimit - m_txPsdus[13].txDuration),
                         "Duration/ID of the first RTS frame must cover the whole TXOP");

  // a CTS is sent by STA1
  tEnd = m_txPsdus[13].txStart + m_txPsdus[13].txDuration;
  tStart = m_txPsdus[14].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "CTS in response to the first RTS frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "CTS in response to the first RTS frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[14].header.IsCts (), true, "Expected a CTS");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[14].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a CTS frame sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[14].header.GetDuration (),
                         RoundDurationId (m_txPsdus[13].header.GetDuration () - sifs - m_txPsdus[14].txDuration),
                         "Duration/ID of the CTS frame must be derived from that of the RTS frame");

  // the AP sends a frame to STA1
  tEnd = m_txPsdus[14].txStart + m_txPsdus[14].txDuration;
  tStart = m_txPsdus[15].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "First QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "First QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[15].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[15].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the first station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[15].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[15].txStart - txopStart) - m_txPsdus[15].txDuration),
                         "Duration/ID of the first QoS data frame does not cover the remaining TXOP");

  // a Normal Ack is then sent by STA1
  tEnd = m_txPsdus[15].txStart + m_txPsdus[15].txDuration;
  tStart = m_txPsdus[16].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the first QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the first QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[16].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[16].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[16].header.GetDuration (),
                         RoundDurationId (m_txPsdus[15].header.GetDuration () - sifs - m_txPsdus[16].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // An RTS frame is sent by the AP to STA2
  tEnd = m_txPsdus[16].txStart + m_txPsdus[16].txDuration;
  tStart = m_txPsdus[17].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Second RTS frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Second RTS frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[17].header.IsRts (), true,
                         "Expected an RTS frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[17].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (1))->GetMac ()->GetAddress (),
                         "Expected an RTS frame sent by the AP to the second station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[17].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[17].txStart - txopStart) - m_txPsdus[17].txDuration),
                         "Duration/ID of the second RTS frame must cover the whole TXOP");

  // a CTS is sent by STA2 (which ignores the NAV)
  tEnd = m_txPsdus[17].txStart + m_txPsdus[17].txDuration;
  tStart = m_txPsdus[18].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "CTS in response to the second RTS frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "CTS in response to the second RTS frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[18].header.IsCts (), true, "Expected a CTS");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[18].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a CTS frame sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[18].header.GetDuration (),
                         RoundDurationId (m_txPsdus[17].header.GetDuration () - sifs - m_txPsdus[18].txDuration),
                         "Duration/ID of the CTS frame must be derived from that of the RTS frame");

  // the AP sends a frame to STA2
  tEnd = m_txPsdus[18].txStart + m_txPsdus[18].txDuration;
  tStart = m_txPsdus[19].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Second QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Second QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[19].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[19].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (1))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the second station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[19].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[19].txStart - txopStart) - m_txPsdus[19].txDuration),
                         "Duration/ID of the second QoS data frame does not cover the remaining TXOP");

  // a Normal Ack is then sent by STA2
  tEnd = m_txPsdus[19].txStart + m_txPsdus[19].txDuration;
  tStart = m_txPsdus[20].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the second QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the second QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[20].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[20].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[20].header.GetDuration (),
                         RoundDurationId (m_txPsdus[19].header.GetDuration () - sifs - m_txPsdus[20].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // An RTS frame is sent by the AP to STA3
  tEnd = m_txPsdus[20].txStart + m_txPsdus[20].txDuration;
  tStart = m_txPsdus[21].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Third RTS frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Third RTS frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[21].header.IsRts (), true,
                         "Expected an RTS frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[21].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (2))->GetMac ()->GetAddress (),
                         "Expected an RTS frame sent by the AP to the third station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[21].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[21].txStart - txopStart) - m_txPsdus[21].txDuration),
                         "Duration/ID of the third RTS frame must cover the whole TXOP");

  // a CTS is sent by STA3 (which ignores the NAV)
  tEnd = m_txPsdus[21].txStart + m_txPsdus[21].txDuration;
  tStart = m_txPsdus[22].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "CTS in response to the third RTS frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "CTS in response to the third RTS frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[22].header.IsCts (), true, "Expected a CTS");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[22].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a CTS frame sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[22].header.GetDuration (),
                         RoundDurationId (m_txPsdus[21].header.GetDuration () - sifs - m_txPsdus[22].txDuration),
                         "Duration/ID of the CTS frame must be derived from that of the RTS frame");

  // the AP sends a frame to STA3
  tEnd = m_txPsdus[22].txStart + m_txPsdus[22].txDuration;
  tStart = m_txPsdus[23].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Third QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Third QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[23].header.IsQosData (), true,
                         "Expected a QoS data frame");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[23].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_staDevices.Get (2))->GetMac ()->GetAddress (),
                         "Expected a frame sent by the AP to the third station");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[23].header.GetDuration (),
                         RoundDurationId (m_txopLimit - (m_txPsdus[23].txStart - txopStart) - m_txPsdus[23].txDuration),
                         "Duration/ID of the third QoS data frame does not cover the remaining TXOP");

  // a Normal Ack is then sent by STA3
  tEnd = m_txPsdus[23].txStart + m_txPsdus[23].txDuration;
  tStart = m_txPsdus[24].txStart;

  NS_TEST_ASSERT_MSG_LT (tEnd + sifs, tStart, "Ack in response to the third QoS data frame sent too early");
  NS_TEST_ASSERT_MSG_LT (tStart, tEnd + sifs + tolerance, "Ack in response to the third QoS data frame sent too late");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[24].header.IsAck (), true, "Expected a Normal Ack");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[24].header.GetAddr1 (),
                         DynamicCast<WifiNetDevice> (m_apDevices.Get (0))->GetMac ()->GetAddress (),
                         "Expected a Normal Ack sent to the AP");
  NS_TEST_ASSERT_MSG_EQ (m_txPsdus[24].header.GetDuration (),
                         RoundDurationId (m_txPsdus[23].header.GetDuration () - sifs - m_txPsdus[24].txDuration),
                         "Duration/ID of the Ack must be derived from that of the previous frame");

  // there is no time remaining for sending a CF-End frame. This is verified by checking
  // that 25 frames are transmitted (done at the beginning of this method)

  // 3 DL packets (without RTS/CTS), 1 UL packet and 3 DL packets (with RTS/CTS)
  NS_TEST_ASSERT_MSG_EQ (m_received, 7, "Unexpected number of packets received");
}


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief wifi TXOP Test Suite
 */
class WifiTxopTestSuite : public TestSuite
{
public:
  WifiTxopTestSuite ();
};

WifiTxopTestSuite::WifiTxopTestSuite ()
  : TestSuite ("wifi-txop", UNIT)
{
  AddTestCase (new WifiTxopTest (true), TestCase::QUICK);
  AddTestCase (new WifiTxopTest (false), TestCase::QUICK);
}

static WifiTxopTestSuite g_wifiTxopTestSuite; ///< the test suite
