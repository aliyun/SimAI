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
#include "ns3/qos-utils.h"
#include "ns3/packet.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/he-frame-exchange-manager.h"
#include "ns3/wifi-acknowledgment.h"
#include "ns3/wifi-protection.h"
#include "ns3/he-configuration.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/config.h"
#include "ns3/pointer.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/wifi-psdu.h"
#include "ns3/multi-user-scheduler.h"
#include "ns3/he-phy.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiMacOfdmaTestSuite");

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Dummy Multi User Scheduler used to test OFDMA ack sequences
 *
 * This Multi User Scheduler returns SU_TX until the simulation time reaches 1.5 seconds
 * (when all BA agreements have been established). Afterwards, it cycles through UL_MU_TX
 * (with a BSRP Trigger Frame), UL_MU_TX (with a Basic Trigger Frame) and DL_MU_TX.
 * This scheduler requires that 4 stations are associated with the AP.
 *
 */
class TestMultiUserScheduler : public MultiUserScheduler
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TestMultiUserScheduler ();
  virtual ~TestMultiUserScheduler ();

private:
  // Implementation of pure virtual methods of MultiUserScheduler class
  TxFormat SelectTxFormat (void) override;
  DlMuInfo ComputeDlMuInfo (void) override;
  UlMuInfo ComputeUlMuInfo (void) override;

  /**
   * Compute the TX vector to use for MU PPDUs.
   */
  void ComputeWifiTxVector (void);

  TxFormat m_txFormat;              //!< the format of next transmission
  TriggerFrameType m_ulTriggerType; //!< Trigger Frame type for UL MU
  CtrlTriggerHeader m_trigger;      //!< Trigger Frame to send
  WifiMacHeader m_triggerHdr;       //!< MAC header for Trigger Frame
  WifiTxVector m_txVector;          //!< the TX vector for MU PPDUs
  WifiTxParameters m_txParams;      //!< TX parameters
  WifiPsduMap m_psduMap;            //!< the DL MU PPDU to transmit
};


NS_OBJECT_ENSURE_REGISTERED (TestMultiUserScheduler);

TypeId
TestMultiUserScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TestMultiUserScheduler")
    .SetParent<MultiUserScheduler> ()
    .SetGroupName ("Wifi")
    .AddConstructor<TestMultiUserScheduler> ()
  ;
  return tid;
}

TestMultiUserScheduler::TestMultiUserScheduler ()
  : m_txFormat (SU_TX),
    m_ulTriggerType (TriggerFrameType::BSRP_TRIGGER)
{
  NS_LOG_FUNCTION (this);
}

TestMultiUserScheduler::~TestMultiUserScheduler ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

MultiUserScheduler::TxFormat
TestMultiUserScheduler::SelectTxFormat (void)
{
  NS_LOG_FUNCTION (this);

  // Do not use OFDMA if a BA agreement has not been established with all the stations
  if (Simulator::Now () < Seconds (1.5))
    {
      NS_LOG_DEBUG ("Return SU_TX");
      return SU_TX;
    }

  ComputeWifiTxVector ();

  if (m_txFormat == SU_TX || m_txFormat == DL_MU_TX
      || (m_txFormat == UL_MU_TX && m_ulTriggerType == TriggerFrameType::BSRP_TRIGGER))
    {
      // try to send a Trigger Frame
      TriggerFrameType ulTriggerType = (m_txFormat == SU_TX || m_txFormat == DL_MU_TX
                                        ? TriggerFrameType::BSRP_TRIGGER
                                        : TriggerFrameType::BASIC_TRIGGER);

      m_trigger = CtrlTriggerHeader (ulTriggerType, m_txVector);

      WifiTxVector txVector = m_txVector;
      txVector.SetGuardInterval (m_trigger.GetGuardInterval ());

      uint32_t ampduSize = (ulTriggerType == TriggerFrameType::BSRP_TRIGGER ? m_sizeOf8QosNull : 3500);

      Time duration = WifiPhy::CalculateTxDuration (ampduSize, txVector,
                                                    m_apMac->GetWifiPhy ()->GetPhyBand (),
                                                    m_apMac->GetStaList ().begin ()->first);

      uint16_t length;
      std::tie (length, duration) = HePhy::ConvertHeTbPpduDurationToLSigLength (duration,
                                                                                m_trigger.GetHeTbTxVector (m_trigger.begin ()->GetAid12 ()),
                                                                                m_apMac->GetWifiPhy ()->GetPhyBand ());
      m_trigger.SetUlLength (length);

      Ptr<Packet> packet = Create<Packet> ();
      packet->AddHeader (m_trigger);

      m_triggerHdr = WifiMacHeader (WIFI_MAC_CTL_TRIGGER);
      m_triggerHdr.SetAddr1 (Mac48Address::GetBroadcast ());
      m_triggerHdr.SetAddr2 (m_apMac->GetAddress ());
      m_triggerHdr.SetDsNotTo ();
      m_triggerHdr.SetDsNotFrom ();

      auto item = Create<WifiMacQueueItem> (packet, m_triggerHdr);

      m_txParams.Clear ();
      // set the TXVECTOR used to send the Trigger Frame
      m_txParams.m_txVector = m_apMac->GetWifiRemoteStationManager ()->GetRtsTxVector (m_triggerHdr.GetAddr1 ());

      if (!m_heFem->TryAddMpdu (item, m_txParams, m_availableTime)
          || (m_availableTime != Time::Min ()
              && m_txParams.m_protection->protectionTime
                 + m_txParams.m_txDuration     // TF tx time
                 + m_apMac->GetWifiPhy ()->GetSifs ()
                 + duration
                 + m_txParams.m_acknowledgment->acknowledgmentTime
                 > m_availableTime ))
        {
          NS_LOG_DEBUG ("Remaining TXOP duration is not enough for BSRP TF exchange");
          return SU_TX;
        }

      m_txFormat = UL_MU_TX;
      m_ulTriggerType = ulTriggerType;
    }
  else if (m_txFormat == UL_MU_TX)
    {
      // try to send a DL MU PPDU
      m_psduMap.clear ();
      const std::map<uint16_t, Mac48Address>& staList = m_apMac->GetStaList ();
      NS_ABORT_MSG_IF (staList.size () != 4, "There must be 4 associated stations");

      /* Initialize TX params */
      m_txParams.Clear ();
      m_txParams.m_txVector = m_txVector;

      for (auto& sta : staList)
        {
          Ptr<const WifiMacQueueItem> peeked = m_apMac->GetQosTxop (AC_BE)->PeekNextMpdu (0, sta.second);

          if (peeked == 0)
            {
              NS_LOG_DEBUG ("No frame to send");
              return SU_TX;
            }

          Ptr<WifiMacQueueItem> mpdu = m_apMac->GetQosTxop (AC_BE)->GetNextMpdu (peeked, m_txParams,
                                                                                 m_availableTime,
                                                                                 m_initialFrame);
          if (mpdu == 0)
            {
              NS_LOG_DEBUG ("Not enough time to send frames to all the stations");
              return SU_TX;
            }

          std::vector<Ptr<WifiMacQueueItem>> mpduList;
          mpduList = m_heFem->GetMpduAggregator ()->GetNextAmpdu (mpdu, m_txParams, m_availableTime);

          if (mpduList.size () > 1)
            {
              m_psduMap[sta.first] = Create<WifiPsdu> (std::move (mpduList));
            }
          else
            {
              m_psduMap[sta.first] = Create<WifiPsdu> (mpdu, true);
            }
        }

      m_txFormat = DL_MU_TX;
    }
  else
    {
      NS_ABORT_MSG ("Cannot get here.");
    }

  NS_LOG_DEBUG ("Return " << m_txFormat);
  return m_txFormat;
}

void
TestMultiUserScheduler::ComputeWifiTxVector (void)
{
  if (m_txVector.GetPreambleType () == WIFI_PREAMBLE_HE_MU)
    {
      // the TX vector has been already computed
      return;
    }

  uint16_t bw = m_apMac->GetWifiPhy ()->GetChannelWidth ();

  m_txVector.SetPreambleType (WIFI_PREAMBLE_HE_MU);
  m_txVector.SetChannelWidth (bw);
  m_txVector.SetGuardInterval (m_apMac->GetHeConfiguration ()->GetGuardInterval ().GetNanoSeconds ());
  m_txVector.SetTxPowerLevel (GetWifiRemoteStationManager ()->GetDefaultTxPowerLevel ());

  const std::map<uint16_t, Mac48Address>& staList = m_apMac->GetStaList ();
  NS_ABORT_MSG_IF (staList.size () != 4, "There must be 4 associated stations");

  HeRu::RuType ruType;
  switch (bw)
    {
      case 20:
        ruType = HeRu::RU_52_TONE;
        break;
      case 40:
        ruType = HeRu::RU_106_TONE;
        break;
      case 80:
        ruType = HeRu::RU_242_TONE;
        break;
      case 160:
        ruType = HeRu::RU_484_TONE;
        break;
      default:
        NS_ABORT_MSG ("Unsupported channel width");
    }

  bool primary80 = true;
  std::size_t ruIndex = 1;

  for (auto& sta : staList)
    {
      if (bw == 160 && ruIndex == 3)
        {
          ruIndex = 1;
          primary80 = false;
        }
      m_txVector.SetHeMuUserInfo (sta.first,
                                  {{ruType, ruIndex++, primary80}, WifiMode ("HeMcs11"), 1});
    }
}

MultiUserScheduler::DlMuInfo
TestMultiUserScheduler::ComputeDlMuInfo (void)
{
  NS_LOG_FUNCTION (this);
  return DlMuInfo {m_psduMap, std::move (m_txParams)};
}

MultiUserScheduler::UlMuInfo
TestMultiUserScheduler::ComputeUlMuInfo (void)
{
  NS_LOG_FUNCTION (this);
  return UlMuInfo {m_trigger, m_triggerHdr, std::move (m_txParams)};
}


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test OFDMA acknowledgment sequences
 *
 * Run this test with:
 *
 * NS_LOG="WifiMacOfdmaTestSuite=info|prefix_time|prefix_node" ./ns3 run "test-runner --suite=wifi-mac-ofdma"
 *
 * to print the list of transmitted frames only, along with the TX time and the
 * node prefix. Replace 'info' with 'debug' if you want to print the debug messages
 * from the test multi-user scheduler only. Replace 'info' with 'level_debug' if
 * you want to print both the transmitted frames and the debug messages.
 */
class OfdmaAckSequenceTest : public TestCase
{
public:
  /**
   * MU EDCA Parameter Set
   */
  struct MuEdcaParameterSet
  {
    uint8_t muAifsn;   //!< MU AIFS (0 to disable EDCA)
    uint16_t muCwMin;  //!< MU CW min
    uint16_t muCwMax;  //!< MU CW max
    uint8_t muTimer;   //!< MU EDCA Timer in units of 8192 microseconds (0 not to use MU EDCA)
  };

  /**
   * Constructor
   * \param width the PHY channel bandwidth in MHz
   * \param dlType the DL MU ack sequence type
   * \param maxAmpduSize the maximum A-MPDU size in bytes
   * \param txopLimit the TXOP limit in microseconds
   * \param nPktsPerSta number of packets to send to/receive from each station
   * \param muEdcaParameterSet the MU EDCA Parameter Set
   */
  OfdmaAckSequenceTest (uint16_t width, WifiAcknowledgment::Method dlType, uint32_t maxAmpduSize,
                        uint16_t txopLimit, uint16_t nPktsPerSta,
                        MuEdcaParameterSet muEdcaParameterSet);
  virtual ~OfdmaAckSequenceTest ();

  /**
   * Function to trace packets received by the server application
   * \param context the context
   * \param p the packet
   * \param addr the address
   */
  void L7Receive (std::string context, Ptr<const Packet> p, const Address &addr);
  /**
   * Function to trace CW value used by the given station after the MU exchange
   * \param staIndex the index of the given station
   * \param oldCw the previous Contention Window value
   * \param cw the current Contention Window value
   */
  void TraceCw (uint32_t staIndex, uint32_t oldCw, uint32_t cw);
  /**
   * Callback invoked when FrameExchangeManager passes PSDUs to the PHY
   * \param context the context
   * \param psduMap the PSDU map
   * \param txVector the TX vector
   * \param txPowerW the tx power in Watts
   */
  void Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);
  /**
   * Check correctness of transmitted frames
   * \param sifs the SIFS duration
   * \param slotTime a slot duration
   * \param aifsn the AIFSN
   */
  void CheckResults (Time sifs, Time slotTime, uint8_t aifsn);

private:
  void DoRun (void) override;

  /// Information about transmitted frames
  struct FrameInfo
  {
    Time startTx;                            ///< start TX time
    Time endTx;                              ///< end TX time
    WifiConstPsduMap psduMap;                ///< transmitted PSDU map
    WifiTxVector txVector;                   ///< TXVECTOR
  };

  uint16_t m_nStations;                      ///< number of stations
  NetDeviceContainer m_staDevices;           ///< stations' devices
  Ptr<NetDevice> m_apDevice;                 ///< AP's device
  uint16_t m_channelWidth;                   ///< PHY channel bandwidth in MHz
  std::vector<FrameInfo> m_txPsdus;          ///< transmitted PSDUs
  WifiAcknowledgment::Method m_dlMuAckType;  ///< DL MU ack sequence type
  uint32_t m_maxAmpduSize;                   ///< maximum A-MPDU size in bytes
  uint16_t m_txopLimit;                      ///< TXOP limit in microseconds
  uint16_t m_nPktsPerSta;                    ///< number of packets to send to each station
  MuEdcaParameterSet m_muEdcaParameterSet;   ///< MU EDCA Parameter Set
  uint16_t m_received;                       ///< number of packets received by the stations
  uint16_t m_flushed;                        ///< number of DL packets flushed after DL MU PPDU
  Time m_edcaDisabledStartTime;              ///< time when disabling EDCA started
  std::vector<uint32_t> m_cwValues;          ///< CW used by stations after MU exchange
};

OfdmaAckSequenceTest::OfdmaAckSequenceTest (uint16_t width, WifiAcknowledgment::Method dlType,
                                            uint32_t maxAmpduSize, uint16_t txopLimit,
                                            uint16_t nPktsPerSta,
                                            MuEdcaParameterSet muEdcaParameterSet)
  : TestCase ("Check correct operation of DL OFDMA acknowledgment sequences"),
    m_nStations (4),
    m_channelWidth (width),
    m_dlMuAckType (dlType),
    m_maxAmpduSize (maxAmpduSize),
    m_txopLimit (txopLimit),
    m_nPktsPerSta (nPktsPerSta),
    m_muEdcaParameterSet (muEdcaParameterSet),
    m_received (0),
    m_flushed (0),
    m_edcaDisabledStartTime (Seconds (0)),
    m_cwValues (std::vector<uint32_t> (m_nStations, 2))  // 2 is an invalid CW value
{
}

OfdmaAckSequenceTest::~OfdmaAckSequenceTest ()
{
}

void
OfdmaAckSequenceTest::L7Receive (std::string context, Ptr<const Packet> p, const Address &addr)
{
  if (p->GetSize () >= 1400 && Simulator::Now () > Seconds (1.5))
    {
      m_received++;
    }
}

void
OfdmaAckSequenceTest::TraceCw (uint32_t staIndex, uint32_t oldCw, uint32_t cw)
{
  if (m_cwValues.at (staIndex) == 2)
    {
      // store the first CW used after MU exchange (the last one may be used after
      // the MU EDCA timer expired)
      m_cwValues[staIndex] = cw;
    }
}

void
OfdmaAckSequenceTest::Transmit (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
  // skip beacon frames and frames transmitted before 1.5s (association
  // request/response, ADDBA request, ...)
  if (!psduMap.begin ()->second->GetHeader (0).IsBeacon ()
      && Simulator::Now () > Seconds (1.5))
    {
      Time txDuration = WifiPhy::CalculateTxDuration (psduMap, txVector, WIFI_PHY_BAND_5GHZ);
      m_txPsdus.push_back ({Simulator::Now (),
                            Simulator::Now () + txDuration,
                            psduMap, txVector});

      for (const auto& psduPair : psduMap)
        {
          NS_LOG_INFO ("Sending " << psduPair.second->GetHeader (0).GetTypeString ()
                        << " #MPDUs " << psduPair.second->GetNMpdus ()
                        << " txDuration " << txDuration
                        << " duration/ID " << psduPair.second->GetHeader (0).GetDuration ()
                        << " #TX PSDUs = " << m_txPsdus.size ());
        }
    }

  // Flush the MAC queue of the AP after sending a DL MU PPDU (no need for
  // further transmissions)
  if (txVector.GetPreambleType () == WIFI_PREAMBLE_HE_MU)
    {
      auto dev = DynamicCast<WifiNetDevice> (m_apDevice);
      Ptr<WifiMacQueue> queue = dev->GetMac ()->GetQosTxop (AC_BE)->GetWifiMacQueue ();
      m_flushed = 0;
      for (auto it = queue->begin (); it != queue->end (); )
        {
          auto tmp = it++;
          if (!(*tmp)->IsInFlight ())
            {
              queue->Remove (*tmp, false);
              m_flushed++;
            }
        }
    }
  else if (txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
           && psduMap.begin ()->second->GetHeader (0).HasData ())
    {
      Mac48Address sender = psduMap.begin ()->second->GetAddr2 ();

      for (uint32_t i = 0; i < m_staDevices.GetN (); i++)
        {
          auto dev = DynamicCast<WifiNetDevice> (m_staDevices.Get (i));

          if (dev->GetAddress () == sender)
            {
              Ptr<QosTxop> qosTxop = dev->GetMac ()->GetQosTxop (AC_BE);

              if (m_muEdcaParameterSet.muTimer > 0 && m_muEdcaParameterSet.muAifsn > 0)
                {
                  // stations use worse access parameters, trace CW. MU AIFSN must be large
                  // enough to avoid collisions between stations trying to transmit using EDCA
                  // right after the UL MU transmission and the AP trying to send a DL MU PPDU
                  qosTxop->TraceConnectWithoutContext ("CwTrace",
                                                       MakeCallback (&OfdmaAckSequenceTest::TraceCw,
                                                                     this).Bind (i));
                }
              else
                {
                  // there is no "protection" against collisions from stations, hence flush
                  // their MAC queues after sending an HE TB PPDU containing QoS data frames,
                  // so that the AP can send a DL MU PPDU
                  qosTxop->GetWifiMacQueue ()->Flush ();
                }
              break;
            }
        }
    }
  else if (!txVector.IsMu () && psduMap.begin ()->second->GetHeader (0).IsBlockAck ()
           && psduMap.begin ()->second->GetHeader (0).GetAddr2 () == m_apDevice->GetAddress ()
           && m_muEdcaParameterSet.muTimer > 0 && m_muEdcaParameterSet.muAifsn == 0)
    {
      CtrlBAckResponseHeader blockAck;
      psduMap.begin ()->second->GetPayload (0)->PeekHeader (blockAck);

      if (blockAck.IsMultiSta ())
        {
          // AP is transmitting a multi-STA BlockAck and stations have to disable EDCA,
          // record the starting time
          m_edcaDisabledStartTime = Simulator::Now () + m_txPsdus.back ().endTx - m_txPsdus.back ().startTx;
        }
    }
}

void
OfdmaAckSequenceTest::CheckResults (Time sifs, Time slotTime, uint8_t aifsn)
{
  CtrlTriggerHeader trigger;
  CtrlBAckResponseHeader blockAck;
  Time tEnd,                           // TX end for a frame
       tStart,                         // TX start fot the next frame
       tolerance = NanoSeconds (500),  // due to propagation delay
       ifs = (m_txopLimit > 0 ? sifs : sifs + aifsn * slotTime),
       navEnd;

  /*
   *                                                  |--------------NAV------------------>|
   *                |------NAV------->|               |                 |------NAV-------->|
   *      |---------|      |----------|     |---------|      |----------|      |-----------|
   *      |         |      |QoS Null 1|     |         |      |QoS Data 1|      |           |
   *      |  BSRP   |      |----------|     |  Basic  |      |----------|      | Multi-STA |
   *      | Trigger |      |QoS Null 2|     | Trigger |      |QoS Data 2|      | Block Ack |
   *      |  Frame  |<SIFS>|----------|<IFS>|  Frame  |<SIFS>|----------|<SIFS>|           |
   *      |         |      |QoS Null 3|     |         |      |QoS Data 3|      |           |
   *      |         |      |----------|     |         |      |----------|      |           |
   *      |         |      |QoS Null 4|     |         |      |QoS Data 4|      |           |
   * -----------------------------------------------------------------------------------------
   * From:     AP                                AP                                 AP
   *   To:    all               AP              all               AP               all
   */

  // the first packet sent after 1.5s is a BSRP Trigger Frame
  NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 5, "Expected at least 5 transmitted packet");
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[0].psduMap.size () == 1
                          && m_txPsdus[0].psduMap[SU_STA_ID]->GetHeader (0).IsTrigger ()
                          && m_txPsdus[0].psduMap[SU_STA_ID]->GetHeader (0).GetAddr1 ().IsBroadcast ()),
                         true, "Expected a Trigger Frame");
  m_txPsdus[0].psduMap[SU_STA_ID]->GetPayload (0)->PeekHeader (trigger);
  NS_TEST_EXPECT_MSG_EQ (trigger.IsBsrp (), true, "Expected a BSRP Trigger Frame");
  NS_TEST_EXPECT_MSG_EQ (trigger.GetNUserInfoFields (), 4, "Expected one User Info field per station");

  // A first STA sends 8 QoS Null frames in an HE TB PPDU a SIFS after the reception of the BSRP TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[1].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[1].psduMap.size () == 1
                          && m_txPsdus[1].psduMap.begin ()->second->GetNMpdus () == 8),
                         true, "Expected 8 QoS Null frames in an HE TB PPDU");
  for (uint8_t i = 0; i < 8; i++)
    {
      const WifiMacHeader& hdr = m_txPsdus[1].psduMap.begin ()->second->GetHeader (i);
      NS_TEST_EXPECT_MSG_EQ (hdr.GetType (), WIFI_MAC_QOSDATA_NULL, "Expected a QoS Null frame");
      uint8_t tid = hdr.GetQosTid ();
      if (tid == 0)
        {
          NS_TEST_EXPECT_MSG_GT (+hdr.GetQosQueueSize (), 0, "Expected a non null queue size for TID " << +tid);
        }
      else
        {
          NS_TEST_EXPECT_MSG_EQ (+hdr.GetQosQueueSize (), 0, "Expected a null queue size for TID " << +tid);
        }
    }
  tEnd = m_txPsdus[0].endTx;
  navEnd = tEnd + m_txPsdus[0].psduMap[SU_STA_ID]->GetDuration ();
  tStart = m_txPsdus[1].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS Null frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS Null frames in HE TB PPDU sent too late");
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[1].endTx, "Duration/ID in BSRP Trigger Frame is too short");

  // A second STA sends 8 QoS Null frames in an HE TB PPDU a SIFS after the reception of the BSRP TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[2].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[2].psduMap.size () == 1
                          && m_txPsdus[2].psduMap.begin ()->second->GetNMpdus () == 8),
                         true, "Expected 8 QoS Null frames in an HE TB PPDU");
  for (uint8_t i = 0; i < 8; i++)
    {
      const WifiMacHeader& hdr = m_txPsdus[2].psduMap.begin ()->second->GetHeader (i);
      NS_TEST_EXPECT_MSG_EQ (hdr.GetType (), WIFI_MAC_QOSDATA_NULL, "Expected a QoS Null frame");
      uint8_t tid = hdr.GetQosTid ();
      if (tid == 0)
        {
          NS_TEST_EXPECT_MSG_GT (+hdr.GetQosQueueSize (), 0, "Expected a non null queue size for TID " << +tid);
        }
      else
        {
          NS_TEST_EXPECT_MSG_EQ (+hdr.GetQosQueueSize (), 0, "Expected a null queue size for TID " << +tid);
        }
    }
  tStart = m_txPsdus[2].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS Null frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS Null frames in HE TB PPDU sent too late");
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[2].endTx, "Duration/ID in BSRP Trigger Frame is too short");

  // A third STA sends 8 QoS Null frames in an HE TB PPDU a SIFS after the reception of the BSRP TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[3].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[3].psduMap.size () == 1
                          && m_txPsdus[3].psduMap.begin ()->second->GetNMpdus () == 8),
                         true, "Expected 8 QoS Null frames in an HE TB PPDU");
  for (uint8_t i = 0; i < 8; i++)
    {
      const WifiMacHeader& hdr = m_txPsdus[3].psduMap.begin ()->second->GetHeader (i);
      NS_TEST_EXPECT_MSG_EQ (hdr.GetType (), WIFI_MAC_QOSDATA_NULL, "Expected a QoS Null frame");
      uint8_t tid = hdr.GetQosTid ();
      if (tid == 0)
        {
          NS_TEST_EXPECT_MSG_GT (+hdr.GetQosQueueSize (), 0, "Expected a non null queue size for TID " << +tid);
        }
      else
        {
          NS_TEST_EXPECT_MSG_EQ (+hdr.GetQosQueueSize (), 0, "Expected a null queue size for TID " << +tid);
        }
    }
  tStart = m_txPsdus[3].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS Null frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS Null frames in HE TB PPDU sent too late");
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[3].endTx, "Duration/ID in BSRP Trigger Frame is too short");

  // A fourth STA sends 8 QoS Null frames in an HE TB PPDU a SIFS after the reception of the BSRP TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[4].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[4].psduMap.size () == 1
                          && m_txPsdus[4].psduMap.begin ()->second->GetNMpdus () == 8),
                         true, "Expected 8 QoS Null frames in an HE TB PPDU");
  for (uint8_t i = 0; i < 8; i++)
    {
      const WifiMacHeader& hdr = m_txPsdus[4].psduMap.begin ()->second->GetHeader (i);
      NS_TEST_EXPECT_MSG_EQ (hdr.GetType (), WIFI_MAC_QOSDATA_NULL, "Expected a QoS Null frame");
      uint8_t tid = hdr.GetQosTid ();
      if (tid == 0)
        {
          NS_TEST_EXPECT_MSG_GT (+hdr.GetQosQueueSize (), 0, "Expected a non null queue size for TID " << +tid);
        }
      else
        {
          NS_TEST_EXPECT_MSG_EQ (+hdr.GetQosQueueSize (), 0, "Expected a null queue size for TID " << +tid);
        }
    }
  tStart = m_txPsdus[4].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS Null frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS Null frames in HE TB PPDU sent too late");
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[4].endTx, "Duration/ID in BSRP Trigger Frame is too short");

  // the AP sends a Basic Trigger Frame to solicit QoS data frames
  NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 11, "Expected at least 11 transmitted packet");
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[5].psduMap.size () == 1
                          && m_txPsdus[5].psduMap[SU_STA_ID]->GetHeader (0).IsTrigger ()
                          && m_txPsdus[5].psduMap[SU_STA_ID]->GetHeader (0).GetAddr1 ().IsBroadcast ()),
                         true, "Expected a Trigger Frame");
  m_txPsdus[5].psduMap[SU_STA_ID]->GetPayload (0)->PeekHeader (trigger);
  NS_TEST_EXPECT_MSG_EQ (trigger.IsBasic (), true, "Expected a Basic Trigger Frame");
  NS_TEST_EXPECT_MSG_EQ (trigger.GetNUserInfoFields (), 4, "Expected one User Info field per station");
  tEnd = m_txPsdus[1].endTx;
  tStart = m_txPsdus[5].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + ifs, tStart, "Basic Trigger Frame sent too early");
  if (m_txopLimit > 0)
    {
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Basic Trigger Frame sent too late");
    }

  // A first STA sends QoS data frames in an HE TB PPDU a SIFS after the reception of the Basic TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[6].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[6].psduMap.size () == 1
                          && m_txPsdus[6].psduMap.begin ()->second->GetHeader (0).IsQosData ()),
                         true, "Expected QoS data frames in an HE TB PPDU");
  tEnd = m_txPsdus[5].endTx;
  tStart = m_txPsdus[6].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS data frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS data frames in HE TB PPDU sent too late");

  // A second STA sends QoS data frames in an HE TB PPDU a SIFS after the reception of the Basic TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[7].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[7].psduMap.size () == 1
                          && m_txPsdus[7].psduMap.begin ()->second->GetHeader (0).IsQosData ()),
                         true, "Expected QoS data frames in an HE TB PPDU");
  tEnd = m_txPsdus[5].endTx;
  tStart = m_txPsdus[7].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS data frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS data frames in HE TB PPDU sent too late");

  // A third STA sends QoS data frames in an HE TB PPDU a SIFS after the reception of the Basic TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[8].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[8].psduMap.size () == 1
                          && m_txPsdus[8].psduMap.begin ()->second->GetHeader (0).IsQosData ()),
                         true, "Expected QoS data frames in an HE TB PPDU");
  tEnd = m_txPsdus[5].endTx;
  tStart = m_txPsdus[8].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS data frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS data frames in HE TB PPDU sent too late");

  // A fourth STA sends QoS data frames in an HE TB PPDU a SIFS after the reception of the Basic TF
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[9].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                          && m_txPsdus[9].psduMap.size () == 1
                          && m_txPsdus[9].psduMap.begin ()->second->GetHeader (0).IsQosData ()),
                         true, "Expected QoS data frames in an HE TB PPDU");
  tEnd = m_txPsdus[5].endTx;
  tStart = m_txPsdus[9].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "QoS data frames in HE TB PPDU sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "QoS data frames in HE TB PPDU sent too late");

  // the AP sends a Multi-STA Block Ack
  NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[10].psduMap.size () == 1
                          && m_txPsdus[10].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAck ()
                          && m_txPsdus[10].psduMap[SU_STA_ID]->GetHeader (0).GetAddr1 ().IsBroadcast ()),
                         true, "Expected a Block Ack");
  m_txPsdus[10].psduMap[SU_STA_ID]->GetPayload (0)->PeekHeader (blockAck);
  NS_TEST_EXPECT_MSG_EQ (blockAck.IsMultiSta (), true, "Expected a Multi-STA Block Ack");
  NS_TEST_EXPECT_MSG_EQ (blockAck.GetNPerAidTidInfoSubfields (), 4,
                         "Expected one Per AID TID Info subfield per station");
  for (uint8_t i = 0; i < 4; i++)
    {
      NS_TEST_EXPECT_MSG_EQ (blockAck.GetAckType (i), true, "Expected All-ack context");
      NS_TEST_EXPECT_MSG_EQ (+blockAck.GetTidInfo (i), 14, "Expected All-ack context");
    }
  tEnd = m_txPsdus[9].endTx;
  tStart = m_txPsdus[10].startTx;
  NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Multi-STA Block Ack sent too early");
  NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Multi-STA Block Ack sent too late");

  navEnd = m_txPsdus[5].endTx + m_txPsdus[5].psduMap[SU_STA_ID]->GetDuration ();  // Basic TF's NAV
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[10].endTx, "Duration/ID in Basic Trigger Frame is too short");
  navEnd = m_txPsdus[6].endTx + m_txPsdus[6].psduMap.begin ()->second->GetDuration ();  // 1st QoS Data frame's NAV
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[10].endTx, "Duration/ID in 1st QoS Data frame is too short");
  navEnd = m_txPsdus[7].endTx + m_txPsdus[7].psduMap.begin ()->second->GetDuration ();  // 2nd QoS Data frame's NAV
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[10].endTx, "Duration/ID in 2nd QoS Data frame is too short");
  navEnd = m_txPsdus[8].endTx + m_txPsdus[8].psduMap.begin ()->second->GetDuration ();  // 3rd QoS Data frame's NAV
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[10].endTx, "Duration/ID in 3rd QoS Data frame is too short");
  navEnd = m_txPsdus[9].endTx + m_txPsdus[9].psduMap.begin ()->second->GetDuration ();  // 4th QoS Data frame's NAV
  NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[10].endTx, "Duration/ID in 4th QoS Data frame is too short");

  // the AP sends a DL MU PPDU
  NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 12, "Expected at least 12 transmitted packet");
  NS_TEST_EXPECT_MSG_EQ (m_txPsdus[11].txVector.GetPreambleType (), WIFI_PREAMBLE_HE_MU,
                         "Expected a DL MU PPDU");
  NS_TEST_EXPECT_MSG_EQ (m_txPsdus[11].psduMap.size (), 4, "Expected 4 PSDUs within the DL MU PPDU");
  // the TX duration cannot exceed the maximum PPDU duration
  NS_TEST_EXPECT_MSG_LT_OR_EQ (m_txPsdus[11].endTx - m_txPsdus[11].startTx,
                               GetPpduMaxTime (m_txPsdus[11].txVector.GetPreambleType ()),
                               "TX duration cannot exceed max PPDU duration");
  for (auto& psdu : m_txPsdus[11].psduMap)
    {
      NS_TEST_EXPECT_MSG_LT_OR_EQ (psdu.second->GetSize (), m_maxAmpduSize, "Max A-MPDU size exceeded");
    }
  tEnd = m_txPsdus[10].endTx;
  tStart = m_txPsdus[11].startTx;
  NS_TEST_EXPECT_MSG_LT_OR_EQ (tEnd + ifs, tStart, "DL MU PPDU sent too early");

  // The Duration/ID field is the same for all the PSDUs
  navEnd = m_txPsdus[11].endTx;
  for (auto& psdu : m_txPsdus[11].psduMap)
    {
      if (navEnd == m_txPsdus[11].endTx)
        {
          navEnd += psdu.second->GetDuration ();
        }
      else
        {
          NS_TEST_EXPECT_MSG_EQ (m_txPsdus[11].endTx + psdu.second->GetDuration (), navEnd,
                                  "Duration/ID must be the same for all PSDUs");
        }
    }
  NS_TEST_EXPECT_MSG_GT (navEnd, m_txPsdus[11].endTx, "Duration/ID of the DL MU PPDU cannot be zero");

  std::size_t nTxPsdus = 0;

  if (m_dlMuAckType == WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE)
    {
      /*
       *        |----------------------------------------NAV---------------------------------------------->|
       *        |                         |--------------------------NAV---------------------------------->|
       *        |                         |                         |-------------------NAV--------------->|
       *        |                         |                         |                         |----NAV---->|
       * |------|      |-----|      |-----|      |-----|      |-----|      |-----|      |-----|      |-----|
       * |PSDU 1|      |     |      |     |      |     |      |     |      |     |      |     |      |     |
       * |------|      |Block|      |Block|      |Block|      |Block|      |Block|      |Block|      |Block|
       * |PSDU 2|      | Ack |      | Ack |      | Ack |      | Ack |      | Ack |      | Ack |      | Ack |
       * |------|<SIFS>|     |<SIFS>| Req |<SIFS>|     |<SIFS>| Req |<SIFS>|     |<SIFS>| Req |<SIFS>|     |
       * |PSDU 3|      |     |      |     |      |     |      |     |      |     |      |     |      |     |
       * |------|      |     |      |     |      |     |      |     |      |     |      |     |      |     |
       * |PSDU 4|      |     |      |     |      |     |      |     |      |     |      |     |      |     |
       * ---------------------------------------------------------------------------------------------------
       * From: AP       STA 1         AP          STA 2         AP          STA 3         AP          STA 4
       *   To:           AP          STA 2         AP          STA 3         AP          STA 4         AP
       */
      NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 19, "Expected at least 19 packets");

      // A first STA sends a Block Ack a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[12].psduMap.size () == 1
                              && m_txPsdus[12].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[12].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "First Block Ack sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "First Block Ack sent too late");

      // the AP transmits a Block Ack Request an IFS after the reception of the Block Ack
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[13].psduMap.size () == 1
                              && m_txPsdus[13].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAckReq ()),
                             true, "Expected a Block Ack Request");
      tEnd = m_txPsdus[12].endTx;
      tStart = m_txPsdus[13].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "First Block Ack Request sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "First Block Ack Request sent too late");

      // A second STA sends a Block Ack a SIFS after the reception of the Block Ack Request
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[14].psduMap.size () == 1
                              && m_txPsdus[14].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[13].endTx;
      tStart = m_txPsdus[14].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Second Block Ack sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Second Block Ack sent too late");

      // the AP transmits a Block Ack Request an IFS after the reception of the Block Ack
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[15].psduMap.size () == 1
                              && m_txPsdus[15].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAckReq ()),
                             true, "Expected a Block Ack Request");
      tEnd = m_txPsdus[14].endTx;
      tStart = m_txPsdus[15].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Second Block Ack Request sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Second Block Ack Request sent too late");

      // A third STA sends a Block Ack a SIFS after the reception of the Block Ack Request
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[16].psduMap.size () == 1
                              && m_txPsdus[16].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[15].endTx;
      tStart = m_txPsdus[16].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Third Block Ack sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Third Block Ack sent too late");

      // the AP transmits a Block Ack Request an IFS after the reception of the Block Ack
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[17].psduMap.size () == 1
                              && m_txPsdus[17].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAckReq ()),
                             true, "Expected a Block Ack Request");
      tEnd = m_txPsdus[16].endTx;
      tStart = m_txPsdus[17].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Third Block Ack Request sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Third Block Ack Request sent too late");

      // A fourth STA sends a Block Ack a SIFS after the reception of the Block Ack Request
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[18].psduMap.size () == 1
                              && m_txPsdus[18].psduMap[SU_STA_ID]->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[17].endTx;
      tStart = m_txPsdus[18].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Fourth Block Ack sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Fourth Block Ack sent too late");

      if (m_txopLimit > 0)
        {
          // DL MU PPDU's NAV
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[18].endTx, "Duration/ID in the DL MU PPDU is too short");
          // 1st BlockAckReq's NAV
          navEnd = m_txPsdus[13].endTx + m_txPsdus[13].psduMap[SU_STA_ID]->GetDuration ();
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[18].endTx, "Duration/ID in the 1st BlockAckReq is too short");
          // 2nd BlockAckReq's NAV
          navEnd = m_txPsdus[15].endTx + m_txPsdus[15].psduMap[SU_STA_ID]->GetDuration ();
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[18].endTx, "Duration/ID in the 2nd BlockAckReq is too short");
          // 3rd BlockAckReq's NAV
          navEnd = m_txPsdus[17].endTx + m_txPsdus[17].psduMap[SU_STA_ID]->GetDuration ();
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[18].endTx, "Duration/ID in the 3rd BlockAckReq is too short");
        }
      nTxPsdus = 19;
    }
  else if (m_dlMuAckType == WifiAcknowledgment::DL_MU_TF_MU_BAR)
    {
      /*
       *             |----------------NAV--------------->|
       *             |                |--------NAV------>|
       *      |------|      |---------|      |-----------|
       *      |PSDU 1|      |         |      |Block Ack 1|
       *      |------|      | MU-BAR  |      |-----------|
       *      |PSDU 2|      | Trigger |      |Block Ack 2|
       *      |------|<SIFS>|  Frame  |<SIFS>|-----------|
       *      |PSDU 3|      |         |      |Block Ack 3|
       *      |------|      |         |      |-----------|
       *      |PSDU 4|      |         |      |Block Ack 4|
       * --------------------------------------------------
       * From:   AP            AP
       *   To:                all                 AP
       */
      NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 17, "Expected at least 17 packets");

      // the AP transmits a MU-BAR Trigger Frame an IFS after the transmission of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[12].psduMap.size () == 1
                              && m_txPsdus[12].psduMap[SU_STA_ID]->GetHeader (0).IsTrigger ()),
                             true, "Expected a MU-BAR Trigger Frame");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[12].startTx;
      NS_TEST_EXPECT_MSG_EQ (tStart, tEnd + sifs, "MU-BAR Trigger Frame sent at wrong time");

      // A first STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[13].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[13].psduMap.size () == 1
                              && m_txPsdus[13].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[12].endTx;
      tStart = m_txPsdus[13].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");

      // A second STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[14].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[14].psduMap.size () == 1
                              && m_txPsdus[14].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[12].endTx;
      tStart = m_txPsdus[14].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");

      // A third STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[15].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[15].psduMap.size () == 1
                              && m_txPsdus[15].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[12].endTx;
      tStart = m_txPsdus[15].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");

      // A fourth STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[16].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[16].psduMap.size () == 1
                              && m_txPsdus[16].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[12].endTx;
      tStart = m_txPsdus[16].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");

      if (m_txopLimit > 0)
        {
          // DL MU PPDU's NAV
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[13].endTx, "Duration/ID in the DL MU PPDU is too short");
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[14].endTx, "Duration/ID in the DL MU PPDU is too short");
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[15].endTx, "Duration/ID in the DL MU PPDU is too short");
          NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[16].endTx, "Duration/ID in the DL MU PPDU is too short");
        }
      // MU-BAR Trigger Frame's NAV
      navEnd = m_txPsdus[12].endTx + m_txPsdus[12].psduMap[SU_STA_ID]->GetDuration ();
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[13].endTx, "Duration/ID in the MU-BAR Trigger Frame is too short");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[14].endTx, "Duration/ID in the MU-BAR Trigger Frame is too short");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[15].endTx, "Duration/ID in the MU-BAR Trigger Frame is too short");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[16].endTx, "Duration/ID in the MU-BAR Trigger Frame is too short");

      nTxPsdus = 17;
    }
  else if (m_dlMuAckType == WifiAcknowledgment::DL_MU_AGGREGATE_TF)
    {
      /*
       *                         |--------NAV------>|
       *      |------|-----------|      |-----------|
       *      |PSDU 1|MU-BAR TF 1|      |Block Ack 1|
       *      |------------------|      |-----------|
       *      |PSDU 2|MU-BAR TF 2|      |Block Ack 2|
       *      |------------------|<SIFS>|-----------|
       *      |PSDU 3|MU-BAR TF 3|      |Block Ack 3|
       *      |------------------|      |-----------|
       *      |PSDU 4|MU-BAR TF 4|      |Block Ack 4|
       * --------------------------------------------------
       * From:         AP
       *   To:                               AP
       */
      NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus.size (), 16, "Expected at least 16 packets");

      // The last MPDU in each PSDU is a MU-BAR Trigger Frame
      for (auto& psdu : m_txPsdus[11].psduMap)
        {
          NS_TEST_EXPECT_MSG_EQ ((*(--psdu.second->end ()))->GetHeader ().IsTrigger (), true,
                                 "Expected an aggregated MU-BAR Trigger Frame");
        }

      // A first STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[12].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[12].psduMap.size () == 1
                              && m_txPsdus[12].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[12].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[12].endTx, "Duration/ID in the DL MU PPDU is too short");

      // A second STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[13].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[13].psduMap.size () == 1
                              && m_txPsdus[13].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[13].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[13].endTx, "Duration/ID in the DL MU PPDU is too short");

      // A third STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[14].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[14].psduMap.size () == 1
                              && m_txPsdus[14].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[14].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[14].endTx, "Duration/ID in the DL MU PPDU is too short");

      // A fourth STA sends a Block Ack in an HE TB PPDU a SIFS after the reception of the DL MU PPDU
      NS_TEST_EXPECT_MSG_EQ ((m_txPsdus[15].txVector.GetPreambleType () == WIFI_PREAMBLE_HE_TB
                              && m_txPsdus[15].psduMap.size () == 1
                              && m_txPsdus[15].psduMap.begin ()->second->GetHeader (0).IsBlockAck ()),
                             true, "Expected a Block Ack");
      tEnd = m_txPsdus[11].endTx;
      tStart = m_txPsdus[15].startTx;
      NS_TEST_EXPECT_MSG_LT (tEnd + sifs, tStart, "Block Ack in HE TB PPDU sent too early");
      NS_TEST_EXPECT_MSG_LT (tStart, tEnd + sifs + tolerance, "Block Ack in HE TB PPDU sent too late");
      NS_TEST_EXPECT_MSG_GT_OR_EQ (navEnd + tolerance, m_txPsdus[15].endTx, "Duration/ID in the DL MU PPDU is too short");

      nTxPsdus = 16;
    }

  NS_TEST_EXPECT_MSG_EQ (m_received, m_nPktsPerSta * m_nStations - m_flushed,
                         "Not all DL packets have been received");

  if (m_muEdcaParameterSet.muTimer > 0 && m_muEdcaParameterSet.muAifsn == 0)
    {
      // EDCA disabled, find the first PSDU transmitted by a station not in an
      // HE TB PPDU and check that it was not transmitted before the MU EDCA
      // timer expired
      for (std::size_t i = nTxPsdus; i < m_txPsdus.size (); ++i)
        {
          if (m_txPsdus[i].psduMap.size () == 1
              && m_txPsdus[i].psduMap.begin ()->second->GetHeader (0).GetAddr2 () != m_apDevice->GetAddress ()
              && !m_txPsdus[i].txVector.IsUlMu ())
            {
              NS_TEST_EXPECT_MSG_GT_OR_EQ (m_txPsdus[i].startTx.GetMicroSeconds (),
                                           m_edcaDisabledStartTime.GetMicroSeconds ()
                                           + m_muEdcaParameterSet.muTimer * 8192,
                                           "A station transmitted before the MU EDCA timer expired");
              break;
            }
        }
    }
  else if (m_muEdcaParameterSet.muTimer > 0 && m_muEdcaParameterSet.muAifsn > 0)
    {
      // stations used worse access parameters after successful UL MU transmission
      for (const auto& cwValue : m_cwValues)
        {
          NS_TEST_EXPECT_MSG_EQ ((cwValue == 2 || cwValue >= m_muEdcaParameterSet.muCwMin),
                                 true, "A station did not set the correct MU CW min");
        }
    }

  m_txPsdus.clear ();
}

void
OfdmaAckSequenceTest::DoRun (void)
{
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (1);
  int64_t streamNumber = 20;

  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (m_nStations);

  Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel> ();
  Ptr<FriisPropagationLossModel> lossModel = CreateObject<FriisPropagationLossModel> ();
  spectrumChannel->AddPropagationLossModel (lossModel);
  Ptr<ConstantSpeedPropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel> ();
  spectrumChannel->SetPropagationDelayModel (delayModel);

  SpectrumWifiPhyHelper phy;
  phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
  phy.SetErrorRateModel ("ns3::NistErrorRateModel");
  phy.SetChannel (spectrumChannel);
  switch (m_channelWidth)
    {
      case 20:
        phy.Set ("ChannelSettings", StringValue ("{36, 20, BAND_5GHZ, 0}"));
        break;
      case 40:
        phy.Set ("ChannelSettings", StringValue ("{38, 40, BAND_5GHZ, 0}"));
        break;
      case 80:
        phy.Set ("ChannelSettings", StringValue ("{42, 80, BAND_5GHZ, 0}"));
        break;
      case 160:
        phy.Set ("ChannelSettings", StringValue ("{50, 160, BAND_5GHZ, 0}"));
        break;
      default:
        NS_ABORT_MSG ("Invalid channel bandwidth (must be 20, 40, 80 or 160)");
    }

  Config::SetDefault ("ns3::HeConfiguration::MuBeAifsn",
                      UintegerValue (m_muEdcaParameterSet.muAifsn));
  Config::SetDefault ("ns3::HeConfiguration::MuBeCwMin",
                      UintegerValue (m_muEdcaParameterSet.muCwMin));
  Config::SetDefault ("ns3::HeConfiguration::MuBeCwMax",
                      UintegerValue (m_muEdcaParameterSet.muCwMax));
  Config::SetDefault ("ns3::HeConfiguration::BeMuEdcaTimer",
                      TimeValue (MicroSeconds (8192 * m_muEdcaParameterSet.muTimer)));
  // MU EDCA timers must be either all null or all non-null
  Config::SetDefault ("ns3::HeConfiguration::BkMuEdcaTimer",
                      TimeValue (MicroSeconds (8192 * m_muEdcaParameterSet.muTimer)));
  Config::SetDefault ("ns3::HeConfiguration::ViMuEdcaTimer",
                      TimeValue (MicroSeconds (8192 * m_muEdcaParameterSet.muTimer)));
  Config::SetDefault ("ns3::HeConfiguration::VoMuEdcaTimer",
                      TimeValue (MicroSeconds (8192 * m_muEdcaParameterSet.muTimer)));

  // increase MSDU lifetime so that it does not expire before the MU EDCA timer ends
  Config::SetDefault ("ns3::WifiMacQueue::MaxDelay", TimeValue (Seconds (2)));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::StaWifiMac",
               "BE_MaxAmsduSize", UintegerValue (0),
               "BE_MaxAmpduSize", UintegerValue (0),
               "Ssid", SsidValue (ssid),
               /* setting blockack threshold for sta's BE queue */
               "BE_BlockAckThreshold", UintegerValue (2),
               "ActiveProbing", BooleanValue (false));

  m_staDevices = wifi.Install (phy, mac, wifiStaNodes);

  mac.SetType ("ns3::ApWifiMac",
               "BE_MaxAmsduSize", UintegerValue (0),
               "BE_MaxAmpduSize", UintegerValue (0),
               "Ssid", SsidValue (ssid),
               "BeaconGeneration", BooleanValue (true));
  mac.SetMultiUserScheduler ("ns3::TestMultiUserScheduler");
  mac.SetAckManager ("ns3::WifiDefaultAckManager", "DlMuAckSequenceType", EnumValue (m_dlMuAckType));

  m_apDevice = wifi.Install (phy, mac, wifiApNode).Get (0);

  // Assign fixed streams to random variables in use
  streamNumber += wifi.AssignStreams (NetDeviceContainer (m_apDevice), streamNumber);
  streamNumber += wifi.AssignStreams (m_staDevices, streamNumber);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (1.0, 0.0, 0.0));
  positionAlloc->Add (Vector (0.0, 1.0, 0.0));
  positionAlloc->Add (Vector (-1.0, 0.0, 0.0));
  positionAlloc->Add (Vector (-1.0, -1.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNodes);

  Ptr<WifiNetDevice> dev;

  // Set maximum A-MPDU size
  for (uint32_t i = 0; i < m_staDevices.GetN (); i++)
    {
      dev = DynamicCast<WifiNetDevice> (m_staDevices.Get (i));
      dev->GetMac ()->SetAttribute ("BE_MaxAmpduSize", UintegerValue (m_maxAmpduSize));
    }
  dev = DynamicCast<WifiNetDevice> (m_apDevice);
  dev->GetMac ()->SetAttribute ("BE_MaxAmpduSize", UintegerValue (m_maxAmpduSize));

  PointerValue ptr;
  dev->GetMac ()->GetAttribute ("BE_Txop", ptr);
  Ptr<QosTxop> apBeQosTxop = ptr.Get<QosTxop> ();
  // set the TXOP limit on BE AC
  apBeQosTxop->SetTxopLimit (MicroSeconds (m_txopLimit));

  PacketSocketHelper packetSocket;
  packetSocket.Install (wifiApNode);
  packetSocket.Install (wifiStaNodes);

  // DL Traffic
  for (uint16_t i = 0; i < m_nStations; i++)
    {
      PacketSocketAddress socket;
      socket.SetSingleDevice (m_apDevice->GetIfIndex ());
      socket.SetPhysicalAddress (m_staDevices.Get (i)->GetAddress ());
      socket.SetProtocol (1);

      // the first client application generates two packets in order
      // to trigger the establishment of a Block Ack agreement
      Ptr<PacketSocketClient> client1 = CreateObject<PacketSocketClient> ();
      client1->SetAttribute ("PacketSize", UintegerValue (1400));
      client1->SetAttribute ("MaxPackets", UintegerValue (2));
      client1->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
      client1->SetRemote (socket);
      wifiApNode.Get (0)->AddApplication (client1);
      client1->SetStartTime (Seconds (1) + i * MilliSeconds (1));
      client1->SetStopTime (Seconds (2.0));

      // the second client application generates the selected number of packets,
      // which are sent in DL MU PPDUs.
      Ptr<PacketSocketClient> client2 = CreateObject<PacketSocketClient> ();
      client2->SetAttribute ("PacketSize", UintegerValue (1400 + i * 100));
      client2->SetAttribute ("MaxPackets", UintegerValue (m_nPktsPerSta));
      client2->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
      client2->SetRemote (socket);
      wifiApNode.Get (0)->AddApplication (client2);
      client2->SetStartTime (Seconds (1.5));
      client2->SetStopTime (Seconds (2.5));

      Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
      server->SetLocal (socket);
      wifiStaNodes.Get (i)->AddApplication (server);
      server->SetStartTime (Seconds (0.0));
      server->SetStopTime (Seconds (3.0));
    }

  // UL Traffic
  for (uint16_t i = 0; i < m_nStations; i++)
    {
      PacketSocketAddress socket;
      socket.SetSingleDevice (m_staDevices.Get (i)->GetIfIndex ());
      socket.SetPhysicalAddress (m_apDevice->GetAddress ());
      socket.SetProtocol (1);

      // the first client application generates two packets in order
      // to trigger the establishment of a Block Ack agreement
      Ptr<PacketSocketClient> client1 = CreateObject<PacketSocketClient> ();
      client1->SetAttribute ("PacketSize", UintegerValue (1400));
      client1->SetAttribute ("MaxPackets", UintegerValue (2));
      client1->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
      client1->SetRemote (socket);
      wifiStaNodes.Get (i)->AddApplication (client1);
      client1->SetStartTime (Seconds (1.005) + i * MilliSeconds (1));
      client1->SetStopTime (Seconds (2.0));

      // the second client application generates the selected number of packets,
      // which are sent in HE TB PPDUs.
      Ptr<PacketSocketClient> client2 = CreateObject<PacketSocketClient> ();
      client2->SetAttribute ("PacketSize", UintegerValue (1400 + i * 100));
      client2->SetAttribute ("MaxPackets", UintegerValue (m_nPktsPerSta));
      client2->SetAttribute ("Interval", TimeValue (MicroSeconds (0)));
      client2->SetRemote (socket);
      wifiStaNodes.Get (i)->AddApplication (client2);
      client2->SetStartTime (Seconds (1.50011));  // start before sending QoS Null frames
      client2->SetStopTime (Seconds (2.5));

      Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer> ();
      server->SetLocal (socket);
      wifiApNode.Get (0)->AddApplication (server);
      server->SetStartTime (Seconds (0.0));
      server->SetStopTime (Seconds (3.0));
    }

  Config::Connect ("/NodeList/*/ApplicationList/0/$ns3::PacketSocketServer/Rx",
                   MakeCallback (&OfdmaAckSequenceTest::L7Receive, this));
  // Trace PSDUs passed to the PHY on all devices
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                   MakeCallback (&OfdmaAckSequenceTest::Transmit, this));

  Simulator::Stop (Seconds (3));
  Simulator::Run ();

  CheckResults (dev->GetMac ()->GetWifiPhy ()->GetSifs (), dev->GetMac ()->GetWifiPhy ()->GetSlot (),
                apBeQosTxop->GetAifsn ());

  Simulator::Destroy ();
}


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief wifi MAC OFDMA Test Suite
 */
class WifiMacOfdmaTestSuite : public TestSuite
{
public:
  WifiMacOfdmaTestSuite ();
};

WifiMacOfdmaTestSuite::WifiMacOfdmaTestSuite ()
  : TestSuite ("wifi-mac-ofdma", UNIT)
{
  using MuEdcaParams = std::initializer_list<OfdmaAckSequenceTest::MuEdcaParameterSet>;

  for (auto& muEdcaParameterSet : MuEdcaParams {{0, 0, 0, 0}         /* no MU EDCA */,
                                                {0, 127, 2047, 100}  /* EDCA disabled */,
                                                {10, 127, 2047, 100} /* worse parameters */})
    {
      AddTestCase (new OfdmaAckSequenceTest (20, WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE, 10000, 5440, 15, muEdcaParameterSet), TestCase::QUICK);
      AddTestCase (new OfdmaAckSequenceTest (20, WifiAcknowledgment::DL_MU_AGGREGATE_TF, 10000, 5440, 15, muEdcaParameterSet), TestCase::QUICK);
      AddTestCase (new OfdmaAckSequenceTest (20, WifiAcknowledgment::DL_MU_TF_MU_BAR, 10000, 5440, 15, muEdcaParameterSet), TestCase::QUICK);
      AddTestCase (new OfdmaAckSequenceTest (40, WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE, 10000, 0, 15, muEdcaParameterSet), TestCase::QUICK);
      AddTestCase (new OfdmaAckSequenceTest (40, WifiAcknowledgment::DL_MU_AGGREGATE_TF, 10000, 0, 15, muEdcaParameterSet), TestCase::QUICK);
      AddTestCase (new OfdmaAckSequenceTest (40, WifiAcknowledgment::DL_MU_TF_MU_BAR, 10000, 0, 15, muEdcaParameterSet), TestCase::QUICK);
    }
}

static WifiMacOfdmaTestSuite g_wifiMacOfdmaTestSuite; ///< the test suite
