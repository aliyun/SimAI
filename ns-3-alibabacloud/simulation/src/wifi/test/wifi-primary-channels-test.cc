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

#include "ns3/boolean.h"
#include "ns3/enum.h"
#include "ns3/test.h"
#include "ns3/wifi-net-device.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/config.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/wifi-psdu.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/he-phy.h"
#include "ns3/he-configuration.h"
#include "ns3/ctrl-headers.h"
#include "ns3/tuple.h"
#include <bitset>
#include <algorithm>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiPrimaryChannelsTest");

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test transmissions under different primary channel settings
 *
 * This test can be repeated for different widths of the operating channel. We
 * configure as many BSSes as the number of distinct 20 MHz subchannels in the
 * operating channel, so that each BSS is assigned a distinct primary20 channel.
 * For each BSS, we test the transmission of SU PPDUs, DL MU PPDUs and HE TB PPDUs
 * of all the widths (20 MHz, 40 MHz, etc.) allowed by the operating channel.
 * Transmissions of a given type take place simultaneously in BSSes that do not
 * operate on adjacent primary channels of the considered width (so that
 * transmissions do not interfere with each other). It is also possible to
 * select whether BSSes should be assigned (distinct) BSS colors or not.
 */
class WifiPrimaryChannelsTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param channelWidth the operating channel width (in MHz)
   * \param useDistinctBssColors whether to set distinct BSS colors to BSSes
   */
  WifiPrimaryChannelsTest (uint16_t channelWidth, bool useDistinctBssColors);
  virtual ~WifiPrimaryChannelsTest ();

  /**
   * Callback invoked when PHY receives a PSDU to transmit. Used to print
   * transmitted PSDUs for debug purposes.
   *
   * \param context the context
   * \param psduMap the PSDU map
   * \param txVector the TX vector
   * \param txPowerW the tx power in Watts
   */
  void Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);
  /**
   * Have the AP of the given BSS transmit a SU PPDU using the given
   * transmission channel width
   *
   * \param bss the given BSS
   * \param txChannelWidth the given transmission channel width in MHz
   */
  void SendDlSuPpdu (uint8_t bss, uint16_t txChannelWidth);
  /**
   * Have the AP of the given BSS transmit a MU PPDU using the given
   * transmission channel width and RU type
   *
   * \param bss the given BSS
   * \param txChannelWidth the given transmission channel width in MHz
   * \param ruType the given RU type
   * \param nRus the number of RUs
   */
  void SendDlMuPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus);
  /**
   * Have the AP of the given BSS transmit a Basic Trigger Frame. This method calls
   * DoSendHeTbPpdu to actually have STAs transmit HE TB PPDUs using the given
   * transmission channel width and RU type
   *
   * \param bss the given BSS
   * \param txChannelWidth the given transmission channel width in MHz
   * \param ruType the given RU type
   * \param nRus the number of RUs
   */
  void SendHeTbPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus);
  /**
   * Have the STAs of the given BSS transmit an HE TB PPDU using the given
   * transmission channel width and RU type
   *
   * \param bss the given BSS
   * \param txChannelWidth the given transmission channel width in MHz
   * \param ruType the given RU type
   * \param nRus the number of RUs
   */
  void DoSendHeTbPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus);
  /**
   * Callback invoked when a station receives a DL PPDU.
   *
   * \param bss the BSS the receiving STA belongs to
   * \param station the receiving station
   * \param psdu the received PSDU
   * \param rxSignalInfo the info on the received signal (\see RxSignalInfo)
   * \param txVector TxVector of the received PSDU
   * \param perMpduStatus per MPDU reception status
   */
  void ReceiveDl (uint8_t bss, uint8_t station, Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo,
                  WifiTxVector txVector, std::vector<bool> perMpduStatus);
  /**
   * Callback invoked when an AP receives an UL PPDU.
   *
   * \param bss the BSS the receiving AP belongs to
   * \param psdu the received PSDU
   * \param rxSignalInfo the info on the received signal (\see RxSignalInfo)
   * \param txVector TxVector of the received PSDU
   * \param perMpduStatus per MPDU reception status
   */
  void ReceiveUl (uint8_t bss, Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo,
                  WifiTxVector txVector, std::vector<bool> perMpduStatus);
  /**
   * Check that all stations associated with an AP.
   */
  void CheckAssociation (void);
  /**
   * Check that (i) all stations belonging to the given BSSes received the SU PPDUs
   * transmitted over the given channel width; and (ii) all stations belonging to
   * the other BSSes did not receive any frame if BSS Color is set (due to BSS Color
   * filtering) or if no transmission was performed on a channel adjacent to the one
   * they operate on, otherwise.
   *
   * \param txBss the set of BSSes that transmitted an SU PPDU
   * \param txChannelWidth the given transmission channel width in MHz
   */
  void CheckReceivedSuPpdus (std::set<uint8_t> txBss, uint16_t txChannelWidth);
  /**
   * Check that (i) all stations/APs belonging to the given BSSes received the DL/UL MU PPDUs
   * transmitted over the given channel width and RU width; and (ii) stations/APs belonging to
   * the other BSSes did not receive any frame if BSS Color is set (due to BSS Color
   * filtering) or if no transmission addressed to/from stations with the same AID was
   * performed on a channel adjacent to the one they operate on, otherwise.
   *
   * \param txBss the set of BSSes that transmitted an SU PPDU
   * \param txChannelWidth the given transmission channel width in MHz
   * \param ruType the given RU type
   * \param nRus the number of RUs
   * \param isDlMu true for DL MU PPDU, false for HE TB PPDU
   */
  void CheckReceivedMuPpdus (std::set<uint8_t> txBss, uint16_t txChannelWidth, HeRu::RuType ruType,
                             std::size_t nRus, bool isDlMu);
  /**
   * Check that (i) all stations belonging to the given BSSes received the transmitted
   * Trigger Frame; and (ii) all stations belonging to the other BSSes did not receive
   * any frame Trigger Frame (given that a Trigger Frame is transmitted on the primary20
   * channel and all the primary20 channels are distinct).
   *
   * \param txBss the set of BSSes that transmitted a Trigger Frame
   * \param txChannelWidth the given transmission channel width in MHz
   */
  void CheckReceivedTriggerFrames (std::set<uint8_t> txBss, uint16_t txChannelWidth);

private:
  void DoSetup (void) override;
  void DoRun (void) override;

  uint16_t m_channelWidth;                       ///< operating channel width in MHz
  bool m_useDistinctBssColors;                   ///< true to set distinct BSS colors to BSSes
  uint8_t m_nBss;                                ///< number of BSSes
  uint8_t m_nStationsPerBss;                     ///< number of stations per AP
  std::vector<NetDeviceContainer> m_staDevices;  ///< containers for stations' NetDevices
  NetDeviceContainer m_apDevices;                ///< container for AP's NetDevice
  std::vector<std::bitset<74>> m_received;       /**< whether the last packet transmitted to/from each of
                                                      the (up to 74 per BSS) stations was received */
  std::vector<std::bitset<74>> m_processed;      /**< whether the last packet transmitted to/from each of
                                                      the (up to 74 per BSS) stations was processed */
  Time m_time;                                   ///< the time when the current action is executed
  Ptr<WifiPsdu> m_trigger;                       ///< Basic Trigger Frame
  WifiTxVector m_triggerTxVector;                ///< TX vector for Basic Trigger Frame
  Time m_triggerTxDuration;                      ///< TX duration for Basic Trigger Frame
};

WifiPrimaryChannelsTest::WifiPrimaryChannelsTest (uint16_t channelWidth, bool useDistinctBssColors)
  : TestCase ("Check correct transmissions for various primary channel settings"),
    m_channelWidth (channelWidth),
    m_useDistinctBssColors (useDistinctBssColors)
{
}

WifiPrimaryChannelsTest::~WifiPrimaryChannelsTest ()
{
}

void
WifiPrimaryChannelsTest::Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
  for (const auto& psduPair : psduMap)
    {
      std::stringstream ss;

      if (psduPair.first != SU_STA_ID)
        {
          ss << " STA-ID " << psduPair.first;
        }
      ss << " " << psduPair.second->GetHeader (0).GetTypeString ()
         << " seq " << psduPair.second->GetHeader (0).GetSequenceNumber ()
         << " from " << psduPair.second->GetAddr2 ()
         << " to " << psduPair.second->GetAddr1 ();
      NS_LOG_INFO (ss.str ());
    }
  NS_LOG_INFO (" TXVECTOR " << txVector);
}

void
WifiPrimaryChannelsTest::ReceiveDl (uint8_t bss, uint8_t station, Ptr<WifiPsdu> psdu,
                                    RxSignalInfo rxSignalInfo, WifiTxVector txVector,
                                    std::vector<bool> perMpduStatus)
{
  if (psdu->GetNMpdus () == 1)
    {
      const WifiMacHeader& hdr = psdu->GetHeader (0);

      if (hdr.IsQosData () || hdr.IsTrigger ())
        {
          NS_LOG_INFO ("RECEIVED BY BSS=" << +bss << " STA=" << +station << "  " << *psdu);
          // the MAC received a PSDU from the PHY
          NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (station), false, "Station [" << +bss << "]["
                                 << +station << "] received a frame twice");
          m_received[bss].set (station);
          // check if we are the intended destination of the PSDU
          auto dev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (station));
          if ((hdr.IsQosData () && hdr.GetAddr1 () == dev->GetMac ()->GetAddress ())
              || (hdr.IsTrigger () && hdr.GetAddr1 () == Mac48Address::GetBroadcast ()))
            {
              NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (station), false, "Station [" << +bss << "]["
                                     << +station << "] processed a frame twice");
              m_processed[bss].set (station);
            }
        }
    }
}

void
WifiPrimaryChannelsTest::ReceiveUl (uint8_t bss, Ptr<WifiPsdu> psdu, RxSignalInfo rxSignalInfo,
                                    WifiTxVector txVector, std::vector<bool> perMpduStatus)
{
  // if the BSS color is zero, this AP might receive the frame sent by another AP. Given that
  // stations only send TB PPDUs, we ignore this frame if the TX vector is not UL MU.
  if (psdu->GetNMpdus () == 1 && psdu->GetHeader (0).IsQosData () && txVector.IsUlMu ())
    {
      auto dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));

      uint16_t staId = txVector.GetHeMuUserInfoMap ().begin ()->first;
      uint8_t station = staId - 1;
      NS_LOG_INFO ("RECEIVED FROM BSSID=" << psdu->GetHeader (0).GetAddr3 () << " STA=" << +station
                   << "  " << *psdu);
      // the MAC received a PSDU containing a QoS data frame from the PHY
      NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (station), false, "AP of BSS " << +bss
                             << " received a frame from station " << +station << " twice");
      m_received[bss].set (station);
      // check if we are the intended destination of the PSDU
      if (psdu->GetHeader (0).GetAddr1 () == dev->GetMac ()->GetAddress ())
        {
          NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (station), false, "AP of BSS " << +bss
                                 << " received a frame from station " << +station << " twice");
          m_processed[bss].set (station);
        }
    }
}

void
WifiPrimaryChannelsTest::DoSetup (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (40);
  int64_t streamNumber = 100;
  uint8_t channelNum;

  // we create as many stations per BSS as the number of 26-tone RUs in a channel
  // of the configured width
  switch (m_channelWidth)
    {
      case 20:
        m_nStationsPerBss = 9;
        channelNum = 36;
        break;
      case 40:
        m_nStationsPerBss = 18;
        channelNum = 38;
        break;
      case 80:
        m_nStationsPerBss = 37;
        channelNum = 42;
        break;
      case 160:
        m_nStationsPerBss = 74;
        channelNum= 50;
        break;
      default:
        NS_ABORT_MSG ("Channel width (" << m_channelWidth << ") not allowed");
    }

  // we create as many BSSes as the number of 20 MHz subchannels
  m_nBss = m_channelWidth / 20;

  NodeContainer wifiApNodes;
  wifiApNodes.Create (m_nBss);

  std::vector<NodeContainer> wifiStaNodes (m_nBss);
  for (auto& container : wifiStaNodes)
    {
      container.Create (m_nStationsPerBss);
    }

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  SpectrumWifiPhyHelper phy;
  phy.SetChannel (spectrumChannel);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager");

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (Ssid ("non-existent-ssid")),
               "WaitBeaconTimeout", TimeValue (MicroSeconds (102400))); // same as BeaconInterval

  TupleValue<UintegerValue, UintegerValue, EnumValue, UintegerValue> channelValue;

  // Each BSS uses a distinct primary20 channel
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      channelValue.Set (WifiPhy::ChannelTuple {channelNum, m_channelWidth, WIFI_PHY_BAND_5GHZ, bss});
      phy.Set ("ChannelSettings", channelValue);

      m_staDevices.push_back (wifi.Install (phy, mac, wifiStaNodes[bss]));
    }

  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      channelValue.Set (WifiPhy::ChannelTuple {channelNum, m_channelWidth, WIFI_PHY_BAND_5GHZ, bss});
      phy.Set ("ChannelSettings", channelValue);

      mac.SetType ("ns3::ApWifiMac",
                  "Ssid", SsidValue (Ssid ("wifi-ssid-" + std::to_string (bss))),
                  "BeaconInterval", TimeValue (MicroSeconds (102400)),
                  "EnableBeaconJitter", BooleanValue (false));

      m_apDevices.Add (wifi.Install (phy, mac, wifiApNodes.Get (bss)));
    }

  // Assign fixed streams to random variables in use
  streamNumber = wifi.AssignStreams (m_apDevices, streamNumber);
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      streamNumber = wifi.AssignStreams (m_staDevices[bss], streamNumber);
    }

  // set BSS color
  if (m_useDistinctBssColors)
    {
      for (uint8_t bss = 0; bss < m_nBss; bss++)
        {
          auto dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
          dev->GetHeConfiguration ()->SetBssColor (bss + 1);
        }
    }

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));   // all stations are co-located
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNodes);
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      mobility.Install (wifiStaNodes[bss]);
    }

  m_received.resize (m_nBss);
  m_processed.resize (m_nBss);

  // pre-compute the Basic Trigger Frame to send
  auto apDev = DynamicCast<WifiNetDevice> (m_apDevices.Get (0));

  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_TRIGGER);
  hdr.SetAddr1 (Mac48Address::GetBroadcast ());
  // Addr2 has to be set
  hdr.SetSequenceNumber (1);

  Ptr<Packet> pkt = Create<Packet> ();
  CtrlTriggerHeader trigger;
  trigger.SetType (TriggerFrameType::BASIC_TRIGGER);
  pkt->AddHeader (trigger);

  m_triggerTxVector = WifiTxVector (OfdmPhy::GetOfdmRate6Mbps (), 0, WIFI_PREAMBLE_LONG, 800, 1, 1, 0,
                                    20, false, false, false);
  m_trigger = Create<WifiPsdu> (pkt, hdr);

  m_triggerTxDuration = WifiPhy::CalculateTxDuration (m_trigger->GetSize (), m_triggerTxVector,
                                                      apDev->GetMac ()->GetWifiPhy ()->GetPhyBand ());
}

void
WifiPrimaryChannelsTest::DoRun (void)
{
  // schedule association requests at different times
  Ptr<WifiNetDevice> dev;

  for (uint16_t i = 0; i < m_nStationsPerBss; i++)
    {
      // association can be done in parallel over the multiple BSSes
      for (uint8_t bss = 0; bss < m_nBss; bss++)
        {
          dev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (i));
          Simulator::Schedule (i * MicroSeconds (102400), &WifiMac::SetSsid,
                               dev->GetMac (), Ssid ("wifi-ssid-" + std::to_string (bss)));
        }
    }

  // just before sending the beacon preceding the last association, increase the beacon
  // interval (to the max allowed value) so that beacons do not interfere with data frames
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
      auto mac = DynamicCast<ApWifiMac> (dev->GetMac ());

      Simulator::Schedule ((m_nStationsPerBss - 1) * MicroSeconds (102400),
                           &ApWifiMac::SetBeaconInterval, mac, MicroSeconds (1024 * 65535));
    }

  m_time = (m_nStationsPerBss + 1) * MicroSeconds (102400);

  Simulator::Schedule (m_time, &WifiPrimaryChannelsTest::CheckAssociation, this);

  // we are done with association. We now intercept frames received by the
  // PHY layer on stations and APs, which will no longer be passed to the FEM.
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      for (uint8_t i = 0; i < m_nStationsPerBss; i++)
        {
          auto dev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (i));
          Simulator::Schedule (m_time, &WifiPhy::SetReceiveOkCallback, dev->GetPhy (),
                               MakeCallback (&WifiPrimaryChannelsTest::ReceiveDl, this)
                                             .TwoBind (bss, i));
        }
      auto dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
      Simulator::Schedule (m_time, &WifiPhy::SetReceiveOkCallback, dev->GetPhy (),
                           MakeCallback (&WifiPrimaryChannelsTest::ReceiveUl, this)
                                         .Bind (bss));
    }

  /*
   * We start generating (downlink) SU PPDUs.
   * First, APs operating on non-adjacent primary20 channels send a frame simultaneously
   * in their primary20. This is done in two rounds. As an example, we consider the
   * case of an 160 MHz operating channel:
   *
   *   AP0         AP2         AP4         AP6
   * |-----|     |-----|     |-----|     |-----|     |
   *
   *         AP1         AP3         AP5         AP7
   * |     |-----|     |-----|     |-----|     |-----|
   *
   * Then, we double the transmission channel width. We will have four rounds
   * of transmissions. We avoid using adjacent channels to avoid interfence
   * among transmissions:
   *
   *      AP0                     AP4
   * |-----------|           |-----------|           |
   *      AP1                     AP5
   * |-----------|           |-----------|           |
   *                  AP2                     AP6
   * |           |-----------|           |-----------|
   *                  AP3                     AP7
   * |           |-----------|           |-----------|
   *
   * We double the transmission channel width again. We will have eight rounds
   * of transmissions:
   *
   *            AP0
   * |-----------------------|                       |
   *            AP1
   * |-----------------------|                       |
   *            AP2
   * |-----------------------|                       |
   *            AP3
   * |-----------------------|                       |
   *                                    AP4
   * |                       |-----------------------|
   *                                    AP5
   * |                       |-----------------------|
   *                                    AP6
   * |                       |-----------------------|
   *                                    AP7
   * |                       |-----------------------|
   *
   * We double the transmission channel width again. We will have eight rounds
   * of transmissions:
   *
   *                        AP0
   * |-----------------------------------------------|
   *                        AP1
   * |-----------------------------------------------|
   *                        AP2
   * |-----------------------------------------------|
   *                        AP3
   * |-----------------------------------------------|
   *                        AP4
   * |-----------------------------------------------|
   *                        AP5
   * |-----------------------------------------------|
   *                        AP6
   * |-----------------------------------------------|
   *                        AP7
   * |-----------------------------------------------|
   *
   * The transmission channel width reached the operating channel width, we are done.
   */

  Time roundDuration = MilliSeconds (5);  // upper bound on the duration of a round

  // To have simultaneous transmissions on adjacent channels, just initialize
  // nRounds to 1 and nApsPerRound to m_channelWidth / 20. Of course, the test
  // will fail because some stations will not receive some frames due to interfence
  for (uint16_t txChannelWidth = 20, nRounds = 2, nApsPerRound = m_channelWidth / 20 / 2;
       txChannelWidth <= m_channelWidth;
       txChannelWidth *= 2, nRounds *= 2, nApsPerRound /= 2)
    {
      nRounds = std::min<uint16_t> (nRounds, m_nBss);
      nApsPerRound = std::max<uint16_t> (nApsPerRound, 1);

      for (uint16_t round = 0; round < nRounds; round++)
        {
          std::set<uint8_t> txBss;

          for (uint16_t i = 0; i < nApsPerRound; i++)
            {
              uint16_t ap = round + i * nRounds;
              txBss.insert (ap);
              Simulator::Schedule (m_time, &WifiPrimaryChannelsTest::SendDlSuPpdu, this,
                                   ap, txChannelWidth);
            }
          // check that the SU frames were correctly received
          Simulator::Schedule (m_time + roundDuration, &WifiPrimaryChannelsTest::CheckReceivedSuPpdus,
                                this, txBss, txChannelWidth);
          m_time += roundDuration;
        }
    }

  /*
   * Repeat the same scheme as before with DL MU transmissions. For each transmission
   * channel width, every round is repeated as many times as the number of ways in
   * which we can partition the transmission channel width in equal sized RUs.
   */
  for (uint16_t txChannelWidth = 20, nRounds = 2, nApsPerRound = m_channelWidth / 20 / 2;
       txChannelWidth <= m_channelWidth;
       txChannelWidth *= 2, nRounds *= 2, nApsPerRound /= 2)
    {
      nRounds = std::min<uint16_t> (nRounds, m_nBss);
      nApsPerRound = std::max<uint16_t> (nApsPerRound, 1);

      for (uint16_t round = 0; round < nRounds; round++)
        {
          for (unsigned int type = 0; type < 7; type++)
            {
              HeRu::RuType ruType = static_cast <HeRu::RuType> (type);
              std::size_t nRus = HeRu::GetNRus (txChannelWidth, ruType);
              std::set<uint8_t> txBss;
              if (nRus > 0)
                {
                  for (uint16_t i = 0; i < nApsPerRound; i++)
                    {
                      uint16_t ap = round + i * nRounds;
                      txBss.insert (ap);
                      Simulator::Schedule (m_time, &WifiPrimaryChannelsTest::SendDlMuPpdu, this,
                                           ap, txChannelWidth, ruType, nRus);
                    }
                  // check that the MU frame was correctly received
                  Simulator::Schedule (m_time + roundDuration, &WifiPrimaryChannelsTest::CheckReceivedMuPpdus,
                                      this, txBss, txChannelWidth, ruType, nRus, /* isDlMu */ true);
                  m_time += roundDuration;
                }
            }
        }
    }

  /*
   * Repeat the same scheme as before with UL MU transmissions. For each transmission
   * channel width, every round is repeated as many times as the number of ways in
   * which we can partition the transmission channel width in equal sized RUs.
   */
  for (uint16_t txChannelWidth = 20, nRounds = 2, nApsPerRound = m_channelWidth / 20 / 2;
       txChannelWidth <= m_channelWidth;
       txChannelWidth *= 2, nRounds *= 2, nApsPerRound /= 2)
    {
      nRounds = std::min<uint16_t> (nRounds, m_nBss);
      nApsPerRound = std::max<uint16_t> (nApsPerRound, 1);

      for (uint16_t round = 0; round < nRounds; round++)
        {
          for (unsigned int type = 0; type < 7; type++)
            {
              HeRu::RuType ruType = static_cast <HeRu::RuType> (type);
              std::size_t nRus = HeRu::GetNRus (txChannelWidth, ruType);
              std::set<uint8_t> txBss;
              if (nRus > 0)
                {
                  for (uint16_t i = 0; i < nApsPerRound; i++)
                    {
                      uint16_t ap = round + i * nRounds;
                      txBss.insert (ap);
                      Simulator::Schedule (m_time, &WifiPrimaryChannelsTest::SendHeTbPpdu, this,
                                           ap, txChannelWidth, ruType, nRus);
                    }
                  // check that Trigger Frames and TB PPDUs were correctly received
                  Simulator::Schedule (m_time + m_triggerTxDuration + MicroSeconds (10), /* during SIFS */
                                       &WifiPrimaryChannelsTest::CheckReceivedTriggerFrames,
                                      this, txBss, txChannelWidth);
                  Simulator::Schedule (m_time + roundDuration, &WifiPrimaryChannelsTest::CheckReceivedMuPpdus,
                                      this, txBss, txChannelWidth, ruType, nRus, /* isDlMu */ false);
                  m_time += roundDuration;
                }
            }
        }
    }

  // Trace PSDUs passed to the PHY on all devices
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                   MakeCallback (&WifiPrimaryChannelsTest::Transmit, this));

  Simulator::Stop (m_time);
  Simulator::Run ();

  Simulator::Destroy ();
}

void
WifiPrimaryChannelsTest::SendDlSuPpdu (uint8_t bss, uint16_t txChannelWidth)
{
  NS_LOG_INFO ("*** BSS " << +bss << " transmits on primary " << txChannelWidth << " MHz channel");

  auto apDev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
  auto staDev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (0));

  uint8_t bssColor = apDev->GetHeConfiguration ()->GetBssColor ();
  WifiTxVector txVector = WifiTxVector (HePhy::GetHeMcs8 (), 0, WIFI_PREAMBLE_HE_SU, 800, 1, 1, 0, txChannelWidth, false, false, false, bssColor);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_QOSDATA);
  hdr.SetQosTid (0);
  hdr.SetAddr1 (staDev->GetMac ()->GetAddress ());
  hdr.SetAddr2 (apDev->GetMac ()->GetAddress ());
  hdr.SetAddr3 (apDev->GetMac ()->GetBssid ());
  hdr.SetSequenceNumber (1);
  Ptr<WifiPsdu> psdu = Create<WifiPsdu> (Create<Packet> (1000), hdr);
  apDev->GetPhy ()->Send (WifiConstPsduMap ({std::make_pair (SU_STA_ID, psdu)}), txVector);
}

void
WifiPrimaryChannelsTest::SendDlMuPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus)
{
  NS_LOG_INFO ("*** BSS " << +bss << " transmits on primary " << txChannelWidth
               << " MHz channel a DL MU PPDU " << "addressed to " << nRus
               << " stations (RU type: " << ruType << ")");

  auto apDev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
  uint8_t bssColor = apDev->GetHeConfiguration ()->GetBssColor ();

  WifiTxVector txVector = WifiTxVector (HePhy::GetHeMcs8 (), 0, WIFI_PREAMBLE_HE_MU, 800, 1, 1, 0, txChannelWidth, false, false, false, bssColor);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_QOSDATA);
  hdr.SetQosTid (0);
  hdr.SetAddr2 (apDev->GetMac ()->GetAddress ());
  hdr.SetAddr3 (apDev->GetMac ()->GetBssid ());
  hdr.SetSequenceNumber (1);

  WifiConstPsduMap psduMap;

  for (std::size_t i = 1; i <= nRus; i++)
    {
      bool primary80 = !(txChannelWidth == 160 && i > nRus / 2);
      std::size_t index = (primary80 ? i : i - nRus / 2);

      auto staDev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (i - 1));
      uint16_t staId = DynamicCast<StaWifiMac> (staDev->GetMac ())->GetAssociationId ();
      txVector.SetHeMuUserInfo (staId, {{ruType, index, primary80}, HePhy::GetHeMcs8 (), 1});
      hdr.SetAddr1 (staDev->GetMac ()->GetAddress ());
      psduMap[staId] = Create<const WifiPsdu> (Create<Packet> (1000), hdr);
    }
  apDev->GetPhy ()->Send (psduMap, txVector);
}

void
WifiPrimaryChannelsTest::SendHeTbPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus)
{
  NS_LOG_INFO ("*** BSS " << +bss << " transmits a Basic Trigger Frame");

  auto apDev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));

  m_trigger->GetHeader (0).SetAddr2 (apDev->GetMac ()->GetAddress ());

  apDev->GetPhy ()->Send (m_trigger, m_triggerTxVector);

  // schedule the transmission of HE TB PPDUs
  Simulator::Schedule (m_triggerTxDuration + apDev->GetPhy ()->GetSifs (),
                       &WifiPrimaryChannelsTest::DoSendHeTbPpdu, this,
                       bss, txChannelWidth, ruType, nRus);
}

void
WifiPrimaryChannelsTest::DoSendHeTbPpdu (uint8_t bss, uint16_t txChannelWidth, HeRu::RuType ruType, std::size_t nRus)
{
  auto apDev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
  uint8_t bssColor = apDev->GetHeConfiguration ()->GetBssColor ();

  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_QOSDATA);
  hdr.SetQosTid (0);
  hdr.SetAddr1 (apDev->GetMac ()->GetAddress ());
  hdr.SetAddr3 (apDev->GetMac ()->GetBssid ());
  hdr.SetSequenceNumber (1);

  Time duration = Seconds (0);
  uint16_t length = 0;
  WifiTxVector trigVector (HePhy::GetHeMcs8 (), 0, WIFI_PREAMBLE_HE_TB, 3200, 1, 1, 0, txChannelWidth, false, false, false, bssColor);

  for (std::size_t i = 1; i <= nRus; i++)
    {
      NS_LOG_INFO ("*** BSS " << +bss << " STA " << i - 1 << " transmits on primary "
                   << txChannelWidth << " MHz channel an HE TB PPDU (RU type: " << ruType << ")");

      bool primary80 = !(txChannelWidth == 160 && i > nRus / 2);
      std::size_t index = (primary80 ? i : i - nRus / 2);

      auto staDev = DynamicCast<WifiNetDevice> (m_staDevices[bss].Get (i - 1));
      uint16_t staId = DynamicCast<StaWifiMac> (staDev->GetMac ())->GetAssociationId ();

      WifiTxVector txVector (HePhy::GetHeMcs8 (), 0, WIFI_PREAMBLE_HE_TB, 3200, 1, 1, 0, txChannelWidth, false, false, false, bssColor);
      txVector.SetHeMuUserInfo (staId, {{ruType, index, primary80}, HePhy::GetHeMcs8 (), 1});
      trigVector.SetHeMuUserInfo (staId, {{ruType, index, primary80}, HePhy::GetHeMcs8 (), 1});

      hdr.SetAddr2 (staDev->GetMac ()->GetAddress ());
      Ptr<const WifiPsdu> psdu = Create<const WifiPsdu> (Create<Packet> (1000), hdr);

      if (duration.IsZero ())
        {
          // calculate just once
          duration = WifiPhy::CalculateTxDuration (psdu->GetSize (), txVector,
                                                   staDev->GetMac ()->GetWifiPhy ()->GetPhyBand (), staId);
          std::tie (length, duration) = HePhy::ConvertHeTbPpduDurationToLSigLength (duration, txVector,
                                                                                    staDev->GetMac ()->GetWifiPhy ()->GetPhyBand ());
        }
      txVector.SetLength (length);

      staDev->GetPhy ()->Send (WifiConstPsduMap {{staId, psdu}}, txVector);
    }

  // AP's PHY expects to receive a TRIGVECTOR (just once)
  trigVector.SetLength (length);
  auto apHePhy = StaticCast<HePhy> (apDev->GetPhy ()->GetPhyEntity (WIFI_MOD_CLASS_HE));
  apHePhy->SetTrigVector (trigVector, duration);
}

void
WifiPrimaryChannelsTest::CheckAssociation (void)
{
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      auto dev = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss));
      auto mac = DynamicCast<ApWifiMac> (dev->GetMac ());
      NS_TEST_EXPECT_MSG_EQ (mac->GetStaList ().size (), m_nStationsPerBss,
                             "Not all the stations completed association");
    }
}

void
WifiPrimaryChannelsTest::CheckReceivedSuPpdus (std::set<uint8_t> txBss, uint16_t txChannelWidth)
{
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      if (txBss.find (bss) != txBss.end ())
        {
          // Every station in the BSS of an AP that transmitted the frame hears (i.e.,
          // passes to the MAC) the frame
          for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), true, "Station [" << +bss << "][" << +sta
                                    << "] did not receive the SU frame on primary" << txChannelWidth << " channel");
            }
          // only the first station actually processed the frames
          NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (0), true, "Station [" << +bss << "][0]"
                                << " did not process the SU frame on primary" << txChannelWidth << " channel");
          for (uint8_t sta = 1; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), false, "Station [" << +bss << "][" << +sta
                                    << "] processed the SU frame on primary" << txChannelWidth << " channel");
            }
        }
      else
        {
          // There was no transmission in this BSS. If BSS Color filtering is enabled or no frame
          // transmission overlaps with the primary20 channel of this BSS, stations in this BSS
          // did not hear any frame.
          if (m_useDistinctBssColors
              || std::none_of (txBss.begin (), txBss.end (),
                               [&](const uint8_t& txAp)
                               {
                                 auto txApPhy = DynamicCast<WifiNetDevice> (m_apDevices.Get (txAp))->GetPhy ();
                                 auto thisApPhy = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss))->GetPhy ();
                                 return txApPhy->GetOperatingChannel ().GetPrimaryChannelIndex (txChannelWidth)
                                        == thisApPhy->GetOperatingChannel ().GetPrimaryChannelIndex (txChannelWidth);
                               }))
            {
              for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), false, "Station [" << +bss << "][" << +sta
                                        << "] received the SU frame on primary" << txChannelWidth << " channel");
                }
            }
          else
            {
              // all stations heard the frame but no station processed it
              for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), true, "Station [" << +bss << "][" << +sta
                                        << "] did not receive the SU frame on primary" << txChannelWidth << " channel");
                  NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), false, "Station [" << +bss << "][" << +sta
                                        << "] processed the SU frame on primary" << txChannelWidth << " channel");
                }
            }
        }
      // reset bitmaps
      m_received[bss].reset ();
      m_processed[bss].reset ();
    }
}

void
WifiPrimaryChannelsTest::CheckReceivedMuPpdus (std::set<uint8_t> txBss, uint16_t txChannelWidth,
                                               HeRu::RuType ruType, std::size_t nRus, bool isDlMu)
{
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      if (txBss.find (bss) != txBss.end ())
        {
          // There was a transmission in this BSS.
          // [DL] Due to AID filtering, only stations that are addressed by the MU PPDU do hear the frame
          // [UL] The AP hears a TB PPDU sent by all and only the solicited stations
          for (uint8_t sta = 0; sta < nRus; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), true,
                                     (isDlMu ? "A DL MU PPDU transmitted to" : "An HE TB PPDU transmitted by")
                                     << " station [" << +bss << "][" << +sta << "] on primary"
                                     << txChannelWidth << " channel, RU type " << ruType
                                     << " was not received");
            }
          for (uint8_t sta = nRus; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), false,
                                     (isDlMu ? "A DL MU PPDU" : "An HE TB PPDU")
                                     << " transmitted on primary" << txChannelWidth
                                     << " channel, RU type " << ruType << " was received "
                                     << (isDlMu ? "by" : "from") << " station [" << +bss << "]["
                                     << +sta << "]");
            }
          // [DL] Only the addressed stations actually processed the frames
          // [UL] The AP processed the frames sent by all and only the addressed stations
          for (uint8_t sta = 0; sta < nRus; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), true,
                                     (isDlMu ? "A DL MU PPDU transmitted to" : "An HE TB PPDU transmitted by")
                                     << " station [" << +bss << "][" << +sta << "] on primary"
                                     << txChannelWidth << " channel, RU type " << ruType
                                     << " was not processed");
            }
          for (uint8_t sta = nRus; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), false,
                                     (isDlMu ? "A DL MU PPDU" : "An HE TB PPDU")
                                     << " transmitted on primary" << txChannelWidth
                                     << " channel, RU type " << ruType << " was received "
                                     << (isDlMu ? "by" : "from") << " station [" << +bss << "]["
                                     << +sta << "] and processed");
            }
        }
      else
        {
          // There was no transmission in this BSS.
          // [DL] If BSS Color filtering is enabled or no frame transmission overlaps with
          // the primary20 channel of this BSS, stations in this BSS did not hear any frame.
          // [UL] The AP did not hear any TB PPDU because no TRIGVECTOR was passed to the PHY
          if (!isDlMu || m_useDistinctBssColors
              || std::none_of (txBss.begin (), txBss.end (),
                               [&](const uint8_t& txAp)
                               {
                                 auto txApPhy = DynamicCast<WifiNetDevice> (m_apDevices.Get (txAp))->GetPhy ();
                                 auto thisApPhy = DynamicCast<WifiNetDevice> (m_apDevices.Get (bss))->GetPhy ();
                                 return txApPhy->GetOperatingChannel ().GetPrimaryChannelIndex (txChannelWidth)
                                        == thisApPhy->GetOperatingChannel ().GetPrimaryChannelIndex (txChannelWidth);
                               }))
            {
              for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), false,
                                         (isDlMu ? "A DL MU PPDU" : "An HE TB PPDU")
                                         << " transmitted on primary" << txChannelWidth
                                         << " channel, RU type " << ruType << " was received "
                                         << (isDlMu ? "by" : "from") << " station [" << +bss << "]["
                                         << +sta << "]");
                }
            }
          else
            {
              // [DL] Stations having the same AID of the stations addressed by the MU PPDU received the frame
              for (uint8_t sta = 0; sta < nRus; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), true,
                                         (isDlMu ? "A DL MU PPDU transmitted to" : "An HE TB PPDU transmitted by")
                                         << " station [" << +bss << "][" << +sta << "] on primary"
                                         << txChannelWidth << " channel, RU type " << ruType
                                         << " was not received");
                }
              for (uint8_t sta = nRus; sta < m_nStationsPerBss; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), false,
                                         (isDlMu ? "A DL MU PPDU" : "An HE TB PPDU")
                                         << " transmitted on primary" << txChannelWidth
                                         << " channel, RU type " << ruType << " was received "
                                         << (isDlMu ? "by" : "from") << " station [" << +bss << "]["
                                         << +sta << "]");
                }
              // no station processed the frame
              for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
                {
                  NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), false,
                                         (isDlMu ? "A DL MU PPDU" : "An HE TB PPDU")
                                         << " transmitted on primary" << txChannelWidth
                                         << " channel, RU type " << ruType << " was received "
                                         << (isDlMu ? "by" : "from") << " station [" << +bss << "]["
                                         << +sta << "] and processed");
                }
            }
        }
      // reset bitmaps
      m_received[bss].reset ();
      m_processed[bss].reset ();
    }
}

void
WifiPrimaryChannelsTest::CheckReceivedTriggerFrames (std::set<uint8_t> txBss, uint16_t txChannelWidth)
{
  for (uint8_t bss = 0; bss < m_nBss; bss++)
    {
      if (txBss.find (bss) != txBss.end ())
        {
          // Every station in the BSS of an AP that transmitted the Trigger Frame hears (i.e.,
          // passes to the MAC) and processes the frame
          for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), true, "Station [" << +bss << "][" << +sta
                                    << "] did not receive the Trigger Frame soliciting a transmission on primary"
                                    << txChannelWidth << " channel");
              NS_TEST_EXPECT_MSG_EQ (m_processed[bss].test (sta), true, "Station [" << +bss << "][" << +sta
                                    << "] did not process the Trigger Frame soliciting a transmission on primary"
                                    << txChannelWidth << " channel");
            }
        }
      else
        {
          // Given that a Trigger Frame is transmitted on the primary20 channel and all the
          // primary20 channels are distinct, stations in other BSSes did not hear the frame
          for (uint8_t sta = 0; sta < m_nStationsPerBss; sta++)
            {
              NS_TEST_EXPECT_MSG_EQ (m_received[bss].test (sta), false, "Station [" << +bss << "][" << +sta
                                    << "] received the Trigger Frame soliciting a transmission on primary"
                                    << txChannelWidth << " channel");
            }
        }
      // reset bitmaps
      m_received[bss].reset ();
      m_processed[bss].reset ();
    }
}


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief wifi primary channels test suite
 */
class WifiPrimaryChannelsTestSuite : public TestSuite
{
public:
  WifiPrimaryChannelsTestSuite ();
};

WifiPrimaryChannelsTestSuite::WifiPrimaryChannelsTestSuite ()
  : TestSuite ("wifi-primary-channels", UNIT)
{
  // Test cases for 20 MHz can be added, but are not that useful (there would be a single BSS)
  AddTestCase (new WifiPrimaryChannelsTest (40, true), TestCase::QUICK);
  AddTestCase (new WifiPrimaryChannelsTest (40, false), TestCase::QUICK);
  AddTestCase (new WifiPrimaryChannelsTest (80, true), TestCase::EXTENSIVE);
  AddTestCase (new WifiPrimaryChannelsTest (80, false), TestCase::EXTENSIVE);
  AddTestCase (new WifiPrimaryChannelsTest (160, true), TestCase::TAKES_FOREVER);
  AddTestCase (new WifiPrimaryChannelsTest (160, false), TestCase::TAKES_FOREVER);
}

static WifiPrimaryChannelsTestSuite g_wifiPrimaryChannelsTestSuite; ///< the test suite
