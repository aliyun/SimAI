/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Duy Nguyen
 * Copyright (c) 2015 Ghada Badawy
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
 * Authors: Duy Nguyen <duy@soe.ucsc.edu>
 *          Ghada Badawy <gbadawy@gmail.com>
 *          Matias Richart <mrichart@fing.edu.uy>
 *
 * Some Comments:
 *
 * 1) By default, Minstrel applies the multi-rate retry (the core of Minstrel
 *    algorithm). Otherwise, please use ConstantRateWifiManager instead.
 *
 * 2) Sampling is done differently from legacy Minstrel. Minstrel-HT tries
 * to sample all rates in all groups at least once and to avoid many
 * consecutive samplings.
 *
 * 3) Sample rate is tried only once, at first place of the MRR chain.
 *
 * reference: http://lwn.net/Articles/376765/
 */

#include <iomanip>
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/random-variable-stream.h"
#include "minstrel-ht-wifi-manager.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-phy.h"

#define Min(a,b) ((a < b) ? a : b)
#define Max(a,b) ((a > b) ? a : b)

NS_LOG_COMPONENT_DEFINE ("MinstrelHtWifiManager");

namespace ns3 {

///MinstrelHtWifiRemoteStation structure
struct MinstrelHtWifiRemoteStation : MinstrelWifiRemoteStation
{
  uint8_t m_sampleGroup;     //!< The group that the sample rate belongs to.

  uint32_t m_sampleWait;      //!< How many transmission attempts to wait until a new sample.
  uint32_t m_sampleTries;     //!< Number of sample tries after waiting sampleWait.
  uint32_t m_sampleCount;     //!< Max number of samples per update interval.
  uint32_t m_numSamplesSlow;  //!< Number of times a slow rate was sampled.

  uint32_t m_avgAmpduLen;      //!< Average number of MPDUs in an A-MPDU.
  uint32_t m_ampduLen;         //!< Number of MPDUs in an A-MPDU.
  uint32_t m_ampduPacketCount; //!< Number of A-MPDUs transmitted.

  McsGroupData m_groupsTable;  //!< Table of groups with stats.
  bool m_isHt;                 //!< If the station is HT capable.

  std::ofstream m_statsFile;   //!< File where statistics table is written.
};

NS_OBJECT_ENSURE_REGISTERED (MinstrelHtWifiManager);

TypeId
MinstrelHtWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MinstrelHtWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .AddConstructor<MinstrelHtWifiManager> ()
    .SetGroupName ("Wifi")
    .AddAttribute ("UpdateStatistics",
                   "The interval between updating statistics table ",
                   TimeValue (MilliSeconds (50)),
                   MakeTimeAccessor (&MinstrelHtWifiManager::m_updateStats),
                   MakeTimeChecker ())
    .AddAttribute ("LegacyUpdateStatistics",
                   "The interval between updating statistics table (for legacy Minstrel) ",
                   TimeValue (MilliSeconds (100)),
                   MakeTimeAccessor (&MinstrelHtWifiManager::m_legacyUpdateStats),
                   MakeTimeChecker ())
    .AddAttribute ("LookAroundRate",
                   "The percentage to try other rates (for legacy Minstrel)",
                   UintegerValue (10),
                   MakeUintegerAccessor (&MinstrelHtWifiManager::m_lookAroundRate),
                   MakeUintegerChecker<uint8_t>(0, 100))
    .AddAttribute ("EWMA",
                   "EWMA level",
                   UintegerValue (75),
                   MakeUintegerAccessor (&MinstrelHtWifiManager::m_ewmaLevel),
                   MakeUintegerChecker<uint8_t>(0, 100))
    .AddAttribute ("SampleColumn",
                   "The number of columns used for sampling",
                   UintegerValue (10),
                   MakeUintegerAccessor (&MinstrelHtWifiManager::m_nSampleCol),
                   MakeUintegerChecker <uint8_t> ())
    .AddAttribute ("PacketLength",
                   "The packet length used for calculating mode TxTime (bytes)",
                   UintegerValue (1200),
                   MakeUintegerAccessor (&MinstrelHtWifiManager::m_frameLength),
                   MakeUintegerChecker <uint32_t> ())
    .AddAttribute ("UseLatestAmendmentOnly",
                   "Use only the latest amendment when it is supported by both peers",
                   BooleanValue (true),
                   MakeBooleanAccessor (&MinstrelHtWifiManager::m_useLatestAmendmentOnly),
                   MakeBooleanChecker ())
    .AddAttribute ("PrintStats",
                   "Control the printing of the statistics table",
                   BooleanValue (false),
                   MakeBooleanAccessor (&MinstrelHtWifiManager::m_printStats),
                   MakeBooleanChecker ())
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&MinstrelHtWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

MinstrelHtWifiManager::MinstrelHtWifiManager ()
  : m_numGroups (0),
    m_numRates (0),
    m_currentRate (0)
{
  NS_LOG_FUNCTION (this);
  m_uniformRandomVariable = CreateObject<UniformRandomVariable> ();
  /**
   *  Create the legacy Minstrel manager in case HT is not supported by the device
   *  or non-HT stations want to associate.
   */
  m_legacyManager = CreateObject<MinstrelWifiManager> ();
}

MinstrelHtWifiManager::~MinstrelHtWifiManager ()
{
  NS_LOG_FUNCTION (this);
  for (uint8_t i = 0; i < m_numGroups; i++)
    {
      m_minstrelGroups[i].ratesFirstMpduTxTimeTable.clear ();
      m_minstrelGroups[i].ratesTxTimeTable.clear ();
    }
}

int64_t
MinstrelHtWifiManager::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  int64_t numStreamsAssigned = 0;
  m_uniformRandomVariable->SetStream (stream);
  numStreamsAssigned++;
  numStreamsAssigned += m_legacyManager->AssignStreams (stream);
  return numStreamsAssigned;
}

void
MinstrelHtWifiManager::SetupPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  // Setup PHY for legacy manager.
  m_legacyManager->SetupPhy (phy);
  WifiRemoteStationManager::SetupPhy (phy);
}

void
MinstrelHtWifiManager::SetupMac (const Ptr<WifiMac> mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_legacyManager->SetupMac (mac);
  WifiRemoteStationManager::SetupMac (mac);
}

void
MinstrelHtWifiManager::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  /**
   * Here we initialize m_minstrelGroups with all the possible groups.
   * If a group is not supported by the device, then it is marked as not supported.
   * Then, after all initializations are finished, we check actual support for each receiving station.
   */

  if (GetHtSupported ())
    {
      m_numGroups = MAX_HT_SUPPORTED_STREAMS * MAX_HT_STREAM_GROUPS;
      m_numRates = MAX_HT_GROUP_RATES;
      if (GetVhtSupported ())
        {
          m_numGroups += MAX_VHT_SUPPORTED_STREAMS * MAX_VHT_STREAM_GROUPS;
          m_numRates = MAX_VHT_GROUP_RATES;
        }
      if (GetHeSupported ())
        {
          m_numGroups += MAX_HE_SUPPORTED_STREAMS * MAX_HE_STREAM_GROUPS;
          m_numRates = MAX_HE_GROUP_RATES;
        }

      /**
       *  Initialize the groups array.
       *  The HT groups come first, then the VHT ones, and finally the HE ones.
       *  Minstrel maintains different types of indexes:
       *  - A global continuous index, which identifies all rates within all groups, in [0, m_numGroups * m_numRates]
       *  - A groupId, which indexes a group in the array, in [0, m_numGroups]
       *  - A rateId, which identifies a rate within a group, in [0, m_numRates]
       *  - A deviceIndex, which indexes a MCS in the PHY MCS array.
       *  - A mcsIndex, which indexes a MCS in the wifi-remote-station-manager supported MCSs array.
       */
      NS_LOG_DEBUG ("Initialize MCS Groups:");
      m_minstrelGroups = MinstrelMcsGroups (m_numGroups);

      // Initialize all HT groups
      for (uint16_t chWidth = 20; chWidth <= MAX_HT_WIDTH; chWidth *= 2)
        {
          for (int gi = 800; gi >= 400; )
            {
              for (uint8_t streams = 1; streams <= MAX_HT_SUPPORTED_STREAMS; streams++)
                {
                  uint8_t groupId = GetHtGroupId (streams, gi, chWidth);

                  m_minstrelGroups[groupId].streams = streams;
                  m_minstrelGroups[groupId].gi = gi;
                  m_minstrelGroups[groupId].chWidth = chWidth;
                  m_minstrelGroups[groupId].type = GROUP_HT;
                  m_minstrelGroups[groupId].isSupported = false;

                  // Check capabilities of the device
                  if (!(!GetShortGuardIntervalSupported () && (gi == 400)) ///Is SGI supported by the transmitter?
                      && (GetPhy ()->GetChannelWidth () >= chWidth) ///Is channel width supported by the transmitter?
                      && (GetPhy ()->GetMaxSupportedTxSpatialStreams () >= streams)) ///Are streams supported by the transmitter?
                    {
                      m_minstrelGroups[groupId].isSupported = true;

                      // Calculate TX time for all rates of the group
                      WifiModeList htMcsList = GetHtDeviceMcsList ();
                      for (uint8_t i = 0; i < MAX_HT_GROUP_RATES; i++)
                        {
                          uint16_t deviceIndex = i + (m_minstrelGroups[groupId].streams - 1) * 8;
                          WifiMode mode =  htMcsList[deviceIndex];
                          AddFirstMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, FIRST_MPDU_IN_AGGREGATE));
                          AddMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, MIDDLE_MPDU_IN_AGGREGATE));
                        }
                      NS_LOG_DEBUG ("Initialized group " << +groupId << ": (" << +streams << "," << gi << "," << chWidth << ")");
                    }
                }
              gi /= 2;
            }
        }

      if (GetVhtSupported ())
        {
          // Initialize all VHT groups
          for (uint16_t chWidth = 20; chWidth <= MAX_VHT_WIDTH; chWidth *= 2)
            {
              for (int gi = 800; gi >= 400; )
                {
                  for (uint8_t streams = 1; streams <= MAX_VHT_SUPPORTED_STREAMS; streams++)
                    {
                      uint8_t groupId = GetVhtGroupId (streams, gi, chWidth);

                      m_minstrelGroups[groupId].streams = streams;
                      m_minstrelGroups[groupId].gi = gi;
                      m_minstrelGroups[groupId].chWidth = chWidth;
                      m_minstrelGroups[groupId].type = GROUP_VHT;
                      m_minstrelGroups[groupId].isSupported = false;

                      // Check capabilities of the device
                      if (!(!GetShortGuardIntervalSupported () && (gi == 400)) ///Is SGI supported by the transmitter?
                          && (GetPhy ()->GetChannelWidth () >= chWidth) ///Is channel width supported by the transmitter?
                          && (GetPhy ()->GetMaxSupportedTxSpatialStreams () >= streams)) ///Are streams supported by the transmitter?
                        {
                          m_minstrelGroups[groupId].isSupported = true;

                          // Calculate TX time for all rates of the group
                          WifiModeList vhtMcsList = GetVhtDeviceMcsList ();
                          for (uint8_t i = 0; i < MAX_VHT_GROUP_RATES; i++)
                            {
                              WifiMode mode = vhtMcsList[i];
                              // Check for invalid VHT MCSs and do not add time to array.
                              if (IsValidMcs (GetPhy (), streams, chWidth, mode))
                                {
                                  AddFirstMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, FIRST_MPDU_IN_AGGREGATE));
                                  AddMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, MIDDLE_MPDU_IN_AGGREGATE));
                                }
                            }
                          NS_LOG_DEBUG ("Initialized group " << +groupId << ": (" << +streams << "," << gi << "," << chWidth << ")");
                        }
                    }
                  gi /= 2;
                }
            }
        }

      if (GetHeSupported ())
        {
          // Initialize all HE groups
          for (uint16_t chWidth = 20; chWidth <= MAX_HE_WIDTH; chWidth *= 2)
            {
              for (int gi = 3200; gi >= 800; )
                {
                  for (uint8_t streams = 1; streams <= MAX_HE_SUPPORTED_STREAMS; streams++)
                    {
                      uint8_t groupId = GetHeGroupId (streams, gi, chWidth);

                      m_minstrelGroups[groupId].streams = streams;
                      m_minstrelGroups[groupId].gi = gi;
                      m_minstrelGroups[groupId].chWidth = chWidth;
                      m_minstrelGroups[groupId].type = GROUP_HE;
                      m_minstrelGroups[groupId].isSupported = false;

                      // Check capabilities of the device
                      if ((GetGuardInterval () <= gi) ///Is GI supported by the transmitter?
                          && (GetPhy ()->GetChannelWidth () >= chWidth) ///Is channel width supported by the transmitter?
                          && (GetPhy ()->GetMaxSupportedTxSpatialStreams () >= streams)) ///Are streams supported by the transmitter?
                        {
                          m_minstrelGroups[groupId].isSupported = true;

                          // Calculate tx time for all rates of the group
                          WifiModeList heMcsList = GetHeDeviceMcsList ();
                          for (uint8_t i = 0; i < MAX_HE_GROUP_RATES; i++)
                            {
                              WifiMode mode = heMcsList.at (i);
                              // Check for invalid HE MCSs and do not add time to array.
                              if (IsValidMcs (GetPhy (), streams, chWidth, mode))
                                {
                                  AddFirstMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, FIRST_MPDU_IN_AGGREGATE));
                                  AddMpduTxTime (groupId, mode, CalculateMpduTxDuration (GetPhy (), streams, gi, chWidth, mode, MIDDLE_MPDU_IN_AGGREGATE));
                                }
                            }
                          NS_LOG_DEBUG ("Initialized group " << +groupId << ": (" << +streams << "," << gi << "," << chWidth << ")");
                        }
                    }
                  gi /= 2;
                }
            }
        }
    }
}

bool
MinstrelHtWifiManager::IsValidMcs (Ptr<WifiPhy> phy, uint8_t streams, uint16_t chWidth, WifiMode mode)
{
  NS_LOG_FUNCTION (this << phy << +streams << chWidth << mode);
  WifiTxVector txvector;
  txvector.SetNss (streams);
  txvector.SetChannelWidth (chWidth);
  txvector.SetMode (mode);
  return txvector.IsValid ();
}

Time
MinstrelHtWifiManager::CalculateMpduTxDuration (Ptr<WifiPhy> phy, uint8_t streams, uint16_t gi,
                                                uint16_t chWidth, WifiMode mode, MpduType mpduType)
{
  NS_LOG_FUNCTION (this << phy << +streams << gi << chWidth << mode << mpduType);
  WifiTxVector txvector;
  txvector.SetNss (streams);
  txvector.SetGuardInterval (gi);
  txvector.SetChannelWidth (chWidth);
  txvector.SetNess (0);
  txvector.SetStbc (0);
  txvector.SetMode (mode);
  txvector.SetPreambleType (WIFI_PREAMBLE_HT_MF);
  return WifiPhy::CalculatePhyPreambleAndHeaderDuration (txvector)
         + WifiPhy::GetPayloadDuration (m_frameLength, txvector, phy->GetPhyBand (), mpduType);
}

Time
MinstrelHtWifiManager::GetFirstMpduTxTime (uint8_t groupId, WifiMode mode) const
{
  NS_LOG_FUNCTION (this << +groupId << mode);
  auto it = m_minstrelGroups[groupId].ratesFirstMpduTxTimeTable.find (mode);
  NS_ASSERT (it != m_minstrelGroups[groupId].ratesFirstMpduTxTimeTable.end ());
  return it->second;
}

void
MinstrelHtWifiManager::AddFirstMpduTxTime (uint8_t groupId, WifiMode mode, Time t)
{
  NS_LOG_FUNCTION (this << +groupId << mode << t);
  m_minstrelGroups[groupId].ratesFirstMpduTxTimeTable.insert (std::make_pair (mode, t));
}

Time
MinstrelHtWifiManager::GetMpduTxTime (uint8_t groupId, WifiMode mode) const
{
  NS_LOG_FUNCTION (this << +groupId << mode);
  auto it = m_minstrelGroups[groupId].ratesTxTimeTable.find (mode);
  NS_ASSERT (it != m_minstrelGroups[groupId].ratesTxTimeTable.end ());
  return it->second;
}

void
MinstrelHtWifiManager::AddMpduTxTime (uint8_t groupId, WifiMode mode, Time t)
{
  NS_LOG_FUNCTION (this << +groupId << mode << t);
  m_minstrelGroups[groupId].ratesTxTimeTable.insert (std::make_pair (mode, t));
}

WifiRemoteStation *
MinstrelHtWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  MinstrelHtWifiRemoteStation *station = new MinstrelHtWifiRemoteStation ();

  // Initialize variables common to both stations.
  station->m_nextStatsUpdate = Simulator::Now () + m_updateStats;
  station->m_col = 0;
  station->m_index = 0;
  station->m_maxTpRate = 0;
  station->m_maxTpRate2 = 0;
  station->m_maxProbRate = 0;
  station->m_nModes = 0;
  station->m_totalPacketsCount = 0;
  station->m_samplePacketsCount = 0;
  station->m_isSampling = false;
  station->m_sampleRate = 0;
  station->m_sampleDeferred = false;
  station->m_shortRetry = 0;
  station->m_longRetry = 0;
  station->m_txrate = 0;
  station->m_initialized = false;

  // Variables specific to HT station
  station->m_sampleGroup = 0;
  station->m_numSamplesSlow = 0;
  station->m_sampleCount = 16;
  station->m_sampleWait = 0;
  station->m_sampleTries = 4;

  station->m_avgAmpduLen = 1;
  station->m_ampduLen = 0;
  station->m_ampduPacketCount = 0;

  // If the device supports HT
  if (GetHtSupported ())
    {
      /**
       * Assume the station is HT.
       * When correct information available it will be checked.
       */
      station->m_isHt = true;
    }
  // Use the variable in the station to indicate that the device do not support HT
  else
    {
      station->m_isHt = false;
    }

  return station;
}

void
MinstrelHtWifiManager::CheckInit (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  // Note: we appear to be doing late initialization of the table
  // to make sure that the set of supported rates has been initialized
  // before we perform our own initialization.
  if (!station->m_initialized)
    {
      /**
       *  Check if the station supports HT.
       *  Assume that if the device do not supports HT then
       *  the station will not support HT either.
       *  We save from using another check and variable.
       */
      if (!GetHtSupported (station))
        {
          NS_LOG_INFO ("non-HT station " << station);
          station->m_isHt = false;
          // We will use non-HT minstrel for this station. Initialize the manager.
          m_legacyManager->SetAttribute ("UpdateStatistics", TimeValue (m_legacyUpdateStats));
          m_legacyManager->SetAttribute ("LookAroundRate", UintegerValue (m_lookAroundRate));
          m_legacyManager->SetAttribute ("EWMA", UintegerValue (m_ewmaLevel));
          m_legacyManager->SetAttribute ("SampleColumn", UintegerValue (m_nSampleCol));
          m_legacyManager->SetAttribute ("PacketLength", UintegerValue (m_frameLength));
          m_legacyManager->SetAttribute ("PrintStats", BooleanValue (m_printStats));
          m_legacyManager->CheckInit (station);
        }
      else
        {
          NS_LOG_DEBUG ("HT station " << station);
          station->m_isHt = true;
          station->m_nModes = GetNMcsSupported (station);
          station->m_minstrelTable = MinstrelRate (station->m_nModes);
          station->m_sampleTable = SampleRate (m_numRates, std::vector<uint8_t> (m_nSampleCol));
          InitSampleTable (station);
          RateInit (station);
          station->m_initialized = true;
        }
    }
}

void
MinstrelHtWifiManager::DoReportRxOk (WifiRemoteStation *st, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << st);
  NS_LOG_DEBUG ("DoReportRxOk m_txrate=" << static_cast<MinstrelHtWifiRemoteStation*> (st)->m_txrate);
}

void
MinstrelHtWifiManager::DoReportRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);
  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }
  NS_LOG_DEBUG ("DoReportRtsFailed m_txrate = " << station->m_txrate);
  station->m_shortRetry++;
}

void
MinstrelHtWifiManager::DoReportRtsOk (WifiRemoteStation *st, double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st);
}

void
MinstrelHtWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);
  NS_LOG_DEBUG ("Final RTS failed");
  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }
  UpdateRetry (station);
}

void
MinstrelHtWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }

  NS_LOG_DEBUG ("DoReportDataFailed " << station << "\t rate " << station->m_txrate << "\tlongRetry \t" << station->m_longRetry);

  if (!station->m_isHt)
    {
      m_legacyManager->UpdateRate (station);
    }
  else if (station->m_longRetry < CountRetries (station))
    {
      uint8_t rateId = GetRateId (station->m_txrate);
      uint8_t groupId = GetGroupId (station->m_txrate);
      station->m_groupsTable[groupId].m_ratesTable[rateId].numRateAttempt++; // Increment the attempts counter for the rate used.
      UpdateRate (station);
    }
}

void
MinstrelHtWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                       double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }

  NS_LOG_DEBUG ("DoReportDataOk m_txrate = " << station->m_txrate << ", attempt = " << station->m_minstrelTable[station->m_txrate].numRateAttempt << ", success = " << station->m_minstrelTable[station->m_txrate].numRateSuccess << " (before update).");

  if (!station->m_isHt)
    {
      station->m_minstrelTable[station->m_txrate].numRateSuccess++;
      station->m_minstrelTable[station->m_txrate].numRateAttempt++;

      m_legacyManager->UpdatePacketCounters (station);

      NS_LOG_DEBUG ("DoReportDataOk m_txrate = " << station->m_txrate << ", attempt = " << station->m_minstrelTable[station->m_txrate].numRateAttempt << ", success = " << station->m_minstrelTable[station->m_txrate].numRateSuccess << " (after update).");

      UpdateRetry (station);
      m_legacyManager->UpdateStats (station);

      if (station->m_nModes >= 1)
        {
          station->m_txrate = m_legacyManager->FindRate (station);
        }
    }
  else
    {
      uint8_t rateId = GetRateId (station->m_txrate);
      uint8_t groupId = GetGroupId (station->m_txrate);
      station->m_groupsTable[groupId].m_ratesTable[rateId].numRateSuccess++;
      station->m_groupsTable[groupId].m_ratesTable[rateId].numRateAttempt++;

      UpdatePacketCounters (station, 1, 0);

      NS_LOG_DEBUG ("DoReportDataOk m_txrate = " << station->m_txrate << ", attempt = " << station->m_minstrelTable[station->m_txrate].numRateAttempt << ", success = " << station->m_minstrelTable[station->m_txrate].numRateSuccess << " (after update).");

      station->m_isSampling = false;
      station->m_sampleDeferred = false;

      UpdateRetry (station);
      if (Simulator::Now () >=  station->m_nextStatsUpdate)
        {
          UpdateStats (station);
        }

      if (station->m_nModes >= 1)
        {
          station->m_txrate = FindRate (station);
        }
    }

  NS_LOG_DEBUG ("Next rate to use TxRate = " << station->m_txrate  );
}

void
MinstrelHtWifiManager::DoReportFinalDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }

  NS_LOG_DEBUG ("DoReportFinalDataFailed - TxRate=" << station->m_txrate);

  if (!station->m_isHt)
    {
      m_legacyManager->UpdatePacketCounters (station);

      UpdateRetry (station);

      m_legacyManager->UpdateStats (station);
      if (station->m_nModes >= 1)
        {
          station->m_txrate = m_legacyManager->FindRate (station);
        }
    }
  else
    {
      UpdatePacketCounters (station, 0, 1);

      station->m_isSampling = false;
      station->m_sampleDeferred = false;

      UpdateRetry (station);
      if (Simulator::Now () >=  station->m_nextStatsUpdate)
        {
          UpdateStats (station);
        }

      if (station->m_nModes >= 1)
        {
          station->m_txrate = FindRate (station);
        }
    }
  NS_LOG_DEBUG ("Next rate to use TxRate = " << station->m_txrate);
}

void
MinstrelHtWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *st, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus,
                                              double rxSnr, double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr << dataChannelWidth << +dataNss);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }

  NS_ASSERT_MSG (station->m_isHt, "A-MPDU Tx Status called but this is a non-HT STA.");

  NS_LOG_DEBUG ("DoReportAmpduTxStatus. TxRate=" << station->m_txrate << " SuccMpdus=" <<
                nSuccessfulMpdus << " FailedMpdus=" << nFailedMpdus);

  station->m_ampduPacketCount++;
  station->m_ampduLen += nSuccessfulMpdus + nFailedMpdus;

  UpdatePacketCounters (station, nSuccessfulMpdus, nFailedMpdus);

  uint8_t rateId = GetRateId (station->m_txrate);
  uint8_t groupId = GetGroupId (station->m_txrate);
  station->m_groupsTable[groupId].m_ratesTable[rateId].numRateSuccess += nSuccessfulMpdus;
  station->m_groupsTable[groupId].m_ratesTable[rateId].numRateAttempt += nSuccessfulMpdus + nFailedMpdus;

  if (nSuccessfulMpdus == 0 && station->m_longRetry < CountRetries (station))
    {
      // We do not receive a BlockAck. The entire AMPDU fail.
      UpdateRate (station);
    }
  else
    {
      station->m_isSampling = false;
      station->m_sampleDeferred = false;

      UpdateRetry (station);
      if (Simulator::Now () >=  station->m_nextStatsUpdate)
        {
          UpdateStats (station);
        }

      if (station->m_nModes >= 1)
        {
          station->m_txrate = FindRate (station);
        }
      NS_LOG_DEBUG ("Next rate to use TxRate = " << station->m_txrate);
    }
}

void
MinstrelHtWifiManager::UpdateRate (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  /**
   * Retry Chain table is implemented here.
   *
   * FIXME
   * Currently, NS3 does not retransmit an entire A-MPDU when BACK is missing
   * but retransmits each MPDU until MPDUs lifetime expires (or a BACK is received).
   * Then, there is no way to control A-MPDU retries (no call to NeedDataRetransmission).
   * So, it is possible that the A-MPDU keeps retrying after longRetry reaches its limit.
   *
   *
   * Try |     LOOKAROUND RATE     | NORMAL RATE
   * -------------------------------------------------------
   *  1  |  Random rate            | Best throughput
   *  2  |  Next best throughput   | Next best throughput
   *  3  |  Best probability       | Best probability
   *
   * Note: For clarity, multiple blocks of if's and else's are used
   * Following implementation in Linux, in MinstrelHT lowest base rate is not used.
   * Explanation can be found here: http://marc.info/?l=linux-wireless&m=144602778611966&w=2
   */

  CheckInit (station);
  if (!station->m_initialized)
    {
      return;
    }
  station->m_longRetry++;

  /**
   * Get the IDs for all rates.
   */
  uint8_t maxTpRateId = GetRateId (station->m_maxTpRate);
  uint8_t maxTpGroupId = GetGroupId (station->m_maxTpRate);
  uint8_t maxTp2RateId = GetRateId (station->m_maxTpRate2);
  uint8_t maxTp2GroupId = GetGroupId (station->m_maxTpRate2);
  uint8_t maxProbRateId = GetRateId (station->m_maxProbRate);
  uint8_t maxProbGroupId = GetGroupId (station->m_maxProbRate);

  /// For normal rate, we're not currently sampling random rates.
  if (!station->m_isSampling)
    {
      /// Use best throughput rate.
      if (station->m_longRetry <  station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].retryCount)
        {
          NS_LOG_DEBUG ("Not Sampling; use the same rate again");
          station->m_txrate = station->m_maxTpRate;  //!<  There are still a few retries.
        }

      /// Use second best throughput rate.
      else if (station->m_longRetry < ( station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].retryCount +
                                        station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].retryCount))
        {
          NS_LOG_DEBUG ("Not Sampling; use the Max TP2");
          station->m_txrate = station->m_maxTpRate2;
        }

      /// Use best probability rate.
      else if (station->m_longRetry <= ( station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].retryCount +
                                         station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].retryCount +
                                         station->m_groupsTable[maxProbGroupId].m_ratesTable[maxProbRateId].retryCount))
        {
          NS_LOG_DEBUG ("Not Sampling; use Max Prob");
          station->m_txrate = station->m_maxProbRate;
        }
      else
        {
          NS_FATAL_ERROR ("Max retries reached and m_longRetry not cleared properly. longRetry= " << station->m_longRetry);
        }
    }

  /// We're currently sampling random rates.
  else
    {
      /// Sample rate is used only once
      /// Use the best rate.
      if (station->m_longRetry < 1 + station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTp2RateId].retryCount)
        {
          NS_LOG_DEBUG ("Sampling use the MaxTP rate");
          station->m_txrate = station->m_maxTpRate2;
        }

      /// Use the best probability rate.
      else if (station->m_longRetry <= 1 + station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTp2RateId].retryCount +
               station->m_groupsTable[maxProbGroupId].m_ratesTable[maxProbRateId].retryCount)
        {
          NS_LOG_DEBUG ("Sampling use the MaxProb rate");
          station->m_txrate = station->m_maxProbRate;
        }
      else
        {
          NS_FATAL_ERROR ("Max retries reached and m_longRetry not cleared properly. longRetry= " << station->m_longRetry);
        }
    }
  NS_LOG_DEBUG ("Next rate to use TxRate = " << station->m_txrate);
}

void
MinstrelHtWifiManager::UpdateRetry (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_shortRetry = 0;
  station->m_longRetry = 0;
}

void
MinstrelHtWifiManager::UpdatePacketCounters (MinstrelHtWifiRemoteStation *station, uint16_t nSuccessfulMpdus, uint16_t nFailedMpdus)
{
  NS_LOG_FUNCTION (this << station << nSuccessfulMpdus << nFailedMpdus);

  station->m_totalPacketsCount += nSuccessfulMpdus + nFailedMpdus;
  if (station->m_isSampling)
    {
      station->m_samplePacketsCount += nSuccessfulMpdus + nFailedMpdus;
    }
  if (station->m_totalPacketsCount == ~0)
    {
      station->m_samplePacketsCount = 0;
      station->m_totalPacketsCount = 0;
    }

  if (!station->m_sampleWait && !station->m_sampleTries && station->m_sampleCount > 0)
    {
      station->m_sampleWait = 16 + 2 * station->m_avgAmpduLen;
      station->m_sampleTries = 1;
      station->m_sampleCount--;
    }
}

WifiTxVector
MinstrelHtWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  if (!station->m_initialized)
    {
      CheckInit (station);
    }

  if (!station->m_isHt)
    {
      WifiTxVector vector = m_legacyManager->GetDataTxVector (station);
      uint64_t dataRate = vector.GetMode ().GetDataRate (vector);
      if (m_currentRate != dataRate && !station->m_isSampling)
        {
          NS_LOG_DEBUG ("New datarate: " << dataRate);
          m_currentRate = dataRate;
        }
      return vector;
    }
  else
    {
      NS_LOG_DEBUG ("DoGetDataMode m_txrate= " << station->m_txrate);

      uint8_t rateId = GetRateId (station->m_txrate);
      uint8_t groupId = GetGroupId (station->m_txrate);
      uint8_t mcsIndex = station->m_groupsTable[groupId].m_ratesTable[rateId].mcsIndex;

      NS_LOG_DEBUG ("DoGetDataMode rateId= " << +rateId << " groupId= " << +groupId << " mode= " << GetMcsSupported (station, mcsIndex));

      McsGroup group = m_minstrelGroups[groupId];

      // Check consistency of rate selected.
      if (((group.type == GROUP_HE) && (group.gi < GetGuardInterval (station))) ||
          (((group.type == GROUP_HT) || (group.type == GROUP_VHT)) && (group.gi == 400) && !GetShortGuardIntervalSupported (station)) ||
          (group.chWidth > GetChannelWidth (station)) ||
          (group.streams > GetNumberOfSupportedStreams (station)))
        {
          NS_FATAL_ERROR ("Inconsistent group selected. Group: (" << +group.streams <<
                         "," << group.gi << "," << group.chWidth << ")" <<
                         " Station capabilities: (" << GetNumberOfSupportedStreams (station) <<
                         "," << ((group.type == GROUP_HE) ? GetGuardInterval (station) : (GetShortGuardIntervalSupported (station) ? 400 : 800)) <<
                         "," << GetChannelWidth (station) << ")");
        }
      WifiMode mode = GetMcsSupported (station, mcsIndex);
      WifiTxVector txVector {mode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), group.gi, GetNumberOfAntennas (), group.streams, GetNess (station), GetChannelWidthForTransmission (mode, group.chWidth), GetAggregation (station) && !station->m_isSampling};
      uint64_t dataRate = mode.GetDataRate (txVector);
      if (m_currentRate != dataRate && !station->m_isSampling)
        {
          NS_LOG_DEBUG ("New datarate: " << dataRate);
          m_currentRate = dataRate;
        }
      return txVector;
    }
}

WifiTxVector
MinstrelHtWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  if (!station->m_initialized)
    {
      CheckInit (station);
    }

  if (!station->m_isHt)
    {
      return m_legacyManager->GetRtsTxVector (station);
    }
  else
    {
      NS_LOG_DEBUG ("DoGetRtsMode m_txrate=" << station->m_txrate);

      /* RTS is sent in a non-HT frame. RTS with HT is not supported yet in NS3.
       * When supported, decision of using HT has to follow rules in Section 9.7.6 from 802.11-2012.
       * From Sec. 9.7.6.5: "A frame other than a BlockAckReq or BlockAck that is carried in a
       * non-HT PPDU shall be transmitted by the STA using a rate no higher than the highest
       * rate in  the BSSBasicRateSet parameter that is less than or equal to the rate or
       * non-HT reference rate (see 9.7.9) of the previously transmitted frame that was
       * directed to the same receiving STA. If no rate in the BSSBasicRateSet parameter meets
       * these conditions, the control frame shall be transmitted at a rate no higher than the
       * highest mandatory rate of the attached PHY that is less than or equal to the rate
       * or non-HT reference rate (see 9.7.9) of the previously transmitted frame that was
       * directed to the same receiving STA."
       */

      // As we are in Minstrel HT, assume the last rate was an HT rate.
      uint8_t rateId = GetRateId (station->m_txrate);
      uint8_t groupId = GetGroupId (station->m_txrate);
      uint8_t mcsIndex = station->m_groupsTable[groupId].m_ratesTable[rateId].mcsIndex;

      WifiMode lastRate = GetMcsSupported (station, mcsIndex);
      uint64_t lastDataRate = lastRate.GetNonHtReferenceRate ();
      uint8_t nBasicRates = GetNBasicModes ();

      WifiMode rtsRate;
      bool rateFound = false;

      for (uint8_t i = 0; i < nBasicRates; i++)
        {
          uint64_t rate = GetBasicMode (i).GetDataRate (20);
          if (rate <= lastDataRate)
            {
              rtsRate = GetBasicMode (i);
              rateFound = true;
            }
        }

      if (!rateFound)
        {
          Ptr<WifiPhy> phy = GetPhy ();
          for (const auto & mode : phy->GetModeList ())
            {
              uint64_t rate = mode.GetDataRate (20);
              if (rate <= lastDataRate)
                {
                  rtsRate = mode;
                  rateFound = true;
                }
            }
        }

      NS_ASSERT (rateFound);

      return WifiTxVector (rtsRate, GetDefaultTxPowerLevel (), GetPreambleForTransmission (rtsRate.GetModulationClass (), GetShortPreambleEnabled ()),
                           800, 1, 1, 0, GetChannelWidthForTransmission (rtsRate, GetPhy ()->GetChannelWidth (), GetChannelWidth (station)), GetAggregation (station));
    }
}

bool
MinstrelHtWifiManager::DoNeedRetransmission (WifiRemoteStation *st, Ptr<const Packet> packet, bool normally)
{
  NS_LOG_FUNCTION (this << st << packet << normally);

  MinstrelHtWifiRemoteStation *station = static_cast<MinstrelHtWifiRemoteStation*> (st);

  CheckInit (station);
  if (!station->m_initialized)
    {
      return normally;
    }

  uint32_t maxRetries;

  if (!station->m_isHt)
    {
      maxRetries = m_legacyManager->CountRetries (station);
    }
  else
    {
      maxRetries = CountRetries (station);
    }

  if (station->m_longRetry >= maxRetries)
    {
      NS_LOG_DEBUG ("No re-transmission allowed. Retries: " <<  station->m_longRetry << " Max retries: " << maxRetries);
      return false;
    }
  else
    {
      NS_LOG_DEBUG ("Re-transmit. Retries: " <<  station->m_longRetry << " Max retries: " << maxRetries);
      return true;
    }
}

uint32_t
MinstrelHtWifiManager::CountRetries (MinstrelHtWifiRemoteStation *station)
{
  uint8_t maxProbRateId = GetRateId (station->m_maxProbRate);
  uint8_t maxProbGroupId = GetGroupId (station->m_maxProbRate);
  uint8_t maxTpRateId = GetRateId (station->m_maxTpRate);
  uint8_t maxTpGroupId = GetGroupId (station->m_maxTpRate);
  uint8_t maxTp2RateId = GetRateId (station->m_maxTpRate2);
  uint8_t maxTp2GroupId = GetGroupId (station->m_maxTpRate2);

  if (!station->m_isSampling)
    {
      return station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].retryCount +
             station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].retryCount +
             station->m_groupsTable[maxProbGroupId].m_ratesTable[maxProbRateId].retryCount;
    }
  else
    {
      return 1 + station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTp2RateId].retryCount +
             station->m_groupsTable[maxProbGroupId].m_ratesTable[maxProbRateId].retryCount;
    }
}

uint16_t
MinstrelHtWifiManager::GetNextSample (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  uint8_t sampleGroup = station->m_sampleGroup;
  uint8_t index = station->m_groupsTable[sampleGroup].m_index;
  uint8_t col = station->m_groupsTable[sampleGroup].m_col;
  uint8_t sampleIndex = station->m_sampleTable[index][col];
  uint16_t rateIndex = GetIndex (sampleGroup, sampleIndex);
  NS_LOG_DEBUG ("Next Sample is " << rateIndex);
  SetNextSample (station); //Calculate the next sample rate.
  return rateIndex;
}

void
MinstrelHtWifiManager::SetNextSample (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  do
    {
      station->m_sampleGroup++;
      station->m_sampleGroup %= m_numGroups;
    }
  while (!station->m_groupsTable[station->m_sampleGroup].m_supported);

  station->m_groupsTable[station->m_sampleGroup].m_index++;

  uint8_t sampleGroup = station->m_sampleGroup;
  uint8_t index = station->m_groupsTable[station->m_sampleGroup].m_index;
  uint8_t col = station->m_groupsTable[sampleGroup].m_col;

  if (index >= m_numRates)
    {
      station->m_groupsTable[station->m_sampleGroup].m_index = 0;
      station->m_groupsTable[station->m_sampleGroup].m_col++;
      if (station->m_groupsTable[station->m_sampleGroup].m_col >= m_nSampleCol)
        {
          station->m_groupsTable[station->m_sampleGroup].m_col = 0;
        }
      index = station->m_groupsTable[station->m_sampleGroup].m_index;
      col = station->m_groupsTable[sampleGroup].m_col;
    }
  NS_LOG_DEBUG ("New sample set: group= " << +sampleGroup << " index= " << +station->m_sampleTable[index][col]);
}

uint16_t
MinstrelHtWifiManager::FindRate (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  NS_LOG_DEBUG ("FindRate packet=" << station->m_totalPacketsCount);

  if ((station->m_samplePacketsCount + station->m_totalPacketsCount) == 0)
    {
      return station->m_maxTpRate;
    }

  // If we have waited enough, then sample.
  if (station->m_sampleWait == 0 && station->m_sampleTries != 0)
    {
      //SAMPLING
      NS_LOG_DEBUG ("Obtaining a sampling rate");
      /// Now go through the table and find an index rate.
      uint16_t sampleIdx = GetNextSample (station);
      NS_LOG_DEBUG ("Sampling rate = " << sampleIdx);

      //Evaluate if the sampling rate selected should be used.
      uint8_t sampleGroupId = GetGroupId (sampleIdx);
      uint8_t sampleRateId = GetRateId (sampleIdx);

      // If the rate selected is not supported, then don't sample.
      if (station->m_groupsTable[sampleGroupId].m_supported && station->m_groupsTable[sampleGroupId].m_ratesTable[sampleRateId].supported)
        {
          /**
           * Sampling might add some overhead to the frame.
           * Hence, don't use sampling for the currently used rates.
           *
           * Also do not sample if the probability is already higher than 95%
           * to avoid wasting airtime.
           */
          MinstrelHtRateInfo sampleRateInfo = station->m_groupsTable[sampleGroupId].m_ratesTable[sampleRateId];

          NS_LOG_DEBUG ("Use sample rate? MaxTpRate= " << station->m_maxTpRate << " CurrentRate= " << station->m_txrate <<
                        " SampleRate= " << sampleIdx << " SampleProb= " << sampleRateInfo.ewmaProb);

          if (sampleIdx != station->m_maxTpRate && sampleIdx != station->m_maxTpRate2
              && sampleIdx != station->m_maxProbRate && sampleRateInfo.ewmaProb <= 95)
            {

              /**
               * Make sure that lower rates get sampled only occasionally,
               * if the link is working perfectly.
               */

              uint8_t maxTpGroupId = GetGroupId (station->m_maxTpRate);
              uint8_t maxTp2GroupId = GetGroupId (station->m_maxTpRate2);
              uint8_t maxTp2RateId = GetRateId (station->m_maxTpRate2);
              uint8_t maxProbGroupId = GetGroupId (station->m_maxProbRate);
              uint8_t maxProbRateId = GetRateId (station->m_maxProbRate);

              uint8_t maxTpStreams = m_minstrelGroups[maxTpGroupId].streams;
              uint8_t sampleStreams = m_minstrelGroups[sampleGroupId].streams;

              Time sampleDuration = sampleRateInfo.perfectTxTime;
              Time maxTp2Duration = station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].perfectTxTime;
              Time maxProbDuration = station->m_groupsTable[maxProbGroupId].m_ratesTable[maxProbRateId].perfectTxTime;

              NS_LOG_DEBUG ("Use sample rate? SampleDuration= " << sampleDuration << " maxTp2Duration= " << maxTp2Duration <<
                            " maxProbDuration= " << maxProbDuration << " sampleStreams= " << +sampleStreams <<
                            " maxTpStreams= " << +maxTpStreams);
              if (sampleDuration < maxTp2Duration || (sampleStreams < maxTpStreams && sampleDuration < maxProbDuration))
                {
                  /// Set flag that we are currently sampling.
                  station->m_isSampling = true;

                  /// set the rate that we're currently sampling
                  station->m_sampleRate = sampleIdx;

                  NS_LOG_DEBUG ("FindRate " << "sampleRate=" << sampleIdx);
                  station->m_sampleTries--;
                  return sampleIdx;
                }
              else
                {
                  station->m_numSamplesSlow++;
                  if (sampleRateInfo.numSamplesSkipped >= 20 && station->m_numSamplesSlow <= 2)
                    {
                      /// Set flag that we are currently sampling.
                      station->m_isSampling = true;

                      /// set the rate that we're currently sampling
                      station->m_sampleRate = sampleIdx;

                      NS_LOG_DEBUG ("FindRate " << "sampleRate=" << sampleIdx);
                      station->m_sampleTries--;
                      return sampleIdx;
                    }
                }
            }
        }
    }
  if (station->m_sampleWait > 0)
    {
      station->m_sampleWait--;
    }

  ///	Continue using the best rate.

  NS_LOG_DEBUG ("FindRate " << "maxTpRrate=" << station->m_maxTpRate);
  return station->m_maxTpRate;
}
void
MinstrelHtWifiManager::UpdateStats (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  station->m_nextStatsUpdate = Simulator::Now () + m_updateStats;

  station->m_numSamplesSlow = 0;
  station->m_sampleCount = 0;

  double tempProb;

  if (station->m_ampduPacketCount > 0)
    {
      uint32_t newLen = station->m_ampduLen / station->m_ampduPacketCount;
      station->m_avgAmpduLen = ( newLen * (100 - m_ewmaLevel) + (station->m_avgAmpduLen * m_ewmaLevel) ) / 100;
      station->m_ampduLen = 0;
      station->m_ampduPacketCount = 0;
    }

  /* Initialize global rate indexes */
  station->m_maxTpRate = GetLowestIndex (station);
  station->m_maxTpRate2 = GetLowestIndex (station);
  station->m_maxProbRate = GetLowestIndex (station);

  /// Update throughput and EWMA for each rate inside each group.
  for (uint8_t j = 0; j < m_numGroups; j++)
    {
      if (station->m_groupsTable[j].m_supported)
        {
          station->m_sampleCount++;

          /* (re)Initialize group rate indexes */
          station->m_groupsTable[j].m_maxTpRate = GetLowestIndex (station, j);
          station->m_groupsTable[j].m_maxTpRate2 = GetLowestIndex (station, j);
          station->m_groupsTable[j].m_maxProbRate = GetLowestIndex (station, j);

          for (uint8_t i = 0; i < m_numRates; i++)
            {
              if (station->m_groupsTable[j].m_ratesTable[i].supported)
                {
                  station->m_groupsTable[j].m_ratesTable[i].retryUpdated = false;

                  NS_LOG_DEBUG (+i << " " << GetMcsSupported (station, station->m_groupsTable[j].m_ratesTable[i].mcsIndex) <<
                                "\t attempt=" << station->m_groupsTable[j].m_ratesTable[i].numRateAttempt <<
                                "\t success=" << station->m_groupsTable[j].m_ratesTable[i].numRateSuccess);

                  /// If we've attempted something.
                  if (station->m_groupsTable[j].m_ratesTable[i].numRateAttempt > 0)
                    {
                      station->m_groupsTable[j].m_ratesTable[i].numSamplesSkipped = 0;
                      /**
                       * Calculate the probability of success.
                       * Assume probability scales from 0 to 100.
                       */
                      tempProb = (100 * station->m_groupsTable[j].m_ratesTable[i].numRateSuccess) / station->m_groupsTable[j].m_ratesTable[i].numRateAttempt;

                      /// Bookkeeping.
                      station->m_groupsTable[j].m_ratesTable[i].prob = tempProb;

                      if (station->m_groupsTable[j].m_ratesTable[i].successHist == 0)
                        {
                          station->m_groupsTable[j].m_ratesTable[i].ewmaProb = tempProb;
                        }
                      else
                        {
                          station->m_groupsTable[j].m_ratesTable[i].ewmsdProb = CalculateEwmsd (station->m_groupsTable[j].m_ratesTable[i].ewmsdProb,
                                                                                                tempProb, station->m_groupsTable[j].m_ratesTable[i].ewmaProb,
                                                                                                m_ewmaLevel);
                          /// EWMA probability
                          tempProb = (tempProb * (100 - m_ewmaLevel) + station->m_groupsTable[j].m_ratesTable[i].ewmaProb * m_ewmaLevel)  / 100;
                          station->m_groupsTable[j].m_ratesTable[i].ewmaProb = tempProb;
                        }

                      station->m_groupsTable[j].m_ratesTable[i].throughput = CalculateThroughput (station, j, i, tempProb);

                      station->m_groupsTable[j].m_ratesTable[i].successHist += station->m_groupsTable[j].m_ratesTable[i].numRateSuccess;
                      station->m_groupsTable[j].m_ratesTable[i].attemptHist += station->m_groupsTable[j].m_ratesTable[i].numRateAttempt;
                    }
                  else
                    {
                      station->m_groupsTable[j].m_ratesTable[i].numSamplesSkipped++;
                    }

                  /// Bookkeeping.
                  station->m_groupsTable[j].m_ratesTable[i].prevNumRateSuccess = station->m_groupsTable[j].m_ratesTable[i].numRateSuccess;
                  station->m_groupsTable[j].m_ratesTable[i].prevNumRateAttempt = station->m_groupsTable[j].m_ratesTable[i].numRateAttempt;
                  station->m_groupsTable[j].m_ratesTable[i].numRateSuccess = 0;
                  station->m_groupsTable[j].m_ratesTable[i].numRateAttempt = 0;

                  if (station->m_groupsTable[j].m_ratesTable[i].throughput != 0)
                    {
                      SetBestStationThRates (station, GetIndex (j, i));
                      SetBestProbabilityRate (station, GetIndex (j, i));
                    }

                }
            }
        }
    }

  //Try to sample all available rates during each interval.
  station->m_sampleCount *= 8;

  //Recalculate retries for the rates selected.
  CalculateRetransmits (station, station->m_maxTpRate);
  CalculateRetransmits (station, station->m_maxTpRate2);
  CalculateRetransmits (station, station->m_maxProbRate);

  NS_LOG_DEBUG ("max tp=" << station->m_maxTpRate << "\nmax tp2=" <<  station->m_maxTpRate2 << "\nmax prob=" << station->m_maxProbRate);
  if (m_printStats)
    {
      PrintTable (station);
    }
}

double
MinstrelHtWifiManager::CalculateThroughput (MinstrelHtWifiRemoteStation *station, uint8_t groupId, uint8_t rateId, double ewmaProb)
{
  /**
  * Calculating throughput.
  * Do not account throughput if probability of success is below 10%
  * (as done in minstrel_ht linux implementation).
  */
  if (ewmaProb < 10)
    {
      return 0;
    }
  else
    {
      /**
       * For the throughput calculation, limit the probability value to 90% to
       * account for collision related packet error rate fluctuation.
       */
      Time txTime =  station->m_groupsTable[groupId].m_ratesTable[rateId].perfectTxTime;
      if (ewmaProb > 90)
        {
          return 90 / txTime.GetSeconds ();
        }
      else
        {
          return ewmaProb / txTime.GetSeconds ();
        }
    }
}

void
MinstrelHtWifiManager::SetBestProbabilityRate (MinstrelHtWifiRemoteStation *station, uint16_t index)
{
  GroupInfo *group;
  MinstrelHtRateInfo rate;
  uint8_t tmpGroupId, tmpRateId;
  double tmpTh, tmpProb;
  uint8_t groupId, rateId;
  double currentTh;
  // maximum group probability (GP) variables
  uint8_t maxGPGroupId, maxGPRateId;
  double maxGPTh;

  groupId = GetGroupId (index);
  rateId = GetRateId (index);
  group = &station->m_groupsTable[groupId];
  rate = group->m_ratesTable[rateId];

  tmpGroupId = GetGroupId (station->m_maxProbRate);
  tmpRateId = GetRateId (station->m_maxProbRate);
  tmpProb = station->m_groupsTable[tmpGroupId].m_ratesTable[tmpRateId].ewmaProb;
  tmpTh =  station->m_groupsTable[tmpGroupId].m_ratesTable[tmpRateId].throughput;

  if (rate.ewmaProb > 75)
    {
      currentTh = station->m_groupsTable[groupId].m_ratesTable[rateId].throughput;
      if (currentTh > tmpTh)
        {
          station->m_maxProbRate = index;
        }

      maxGPGroupId = GetGroupId (group->m_maxProbRate);
      maxGPRateId = GetRateId (group->m_maxProbRate);
      maxGPTh = station->m_groupsTable[maxGPGroupId].m_ratesTable[maxGPRateId].throughput;

      if (currentTh > maxGPTh)
        {
          group->m_maxProbRate = index;
        }
    }
  else
    {
      if (rate.ewmaProb > tmpProb)
        {
          station->m_maxProbRate = index;
        }
      maxGPRateId = GetRateId (group->m_maxProbRate);
      if (rate.ewmaProb > group->m_ratesTable[maxGPRateId].ewmaProb)
        {
          group->m_maxProbRate = index;
        }
    }
}

/*
 * Find & sort topmost throughput rates
 *
 * If multiple rates provide equal throughput the sorting is based on their
 * current success probability. Higher success probability is preferred among
 * MCS groups.
 */
void
MinstrelHtWifiManager::SetBestStationThRates (MinstrelHtWifiRemoteStation *station, uint16_t index)
{
  uint8_t groupId, rateId;
  double th, prob;
  uint8_t maxTpGroupId, maxTpRateId;
  uint8_t maxTp2GroupId, maxTp2RateId;
  double maxTpTh, maxTpProb;
  double maxTp2Th, maxTp2Prob;

  groupId = GetGroupId (index);
  rateId = GetRateId (index);
  prob = station->m_groupsTable[groupId].m_ratesTable[rateId].ewmaProb;
  th = station->m_groupsTable[groupId].m_ratesTable[rateId].throughput;

  maxTpGroupId = GetGroupId (station->m_maxTpRate);
  maxTpRateId = GetRateId (station->m_maxTpRate);
  maxTpProb = station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].ewmaProb;
  maxTpTh = station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].throughput;

  maxTp2GroupId = GetGroupId (station->m_maxTpRate2);
  maxTp2RateId = GetRateId (station->m_maxTpRate2);
  maxTp2Prob = station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].ewmaProb;
  maxTp2Th = station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].throughput;

  if (th > maxTpTh || (th == maxTpTh && prob > maxTpProb))
    {
      station->m_maxTpRate2 = station->m_maxTpRate;
      station->m_maxTpRate = index;
    }
  else if (th > maxTp2Th || (th == maxTp2Th && prob > maxTp2Prob))
    {
      station->m_maxTpRate2 = index;
    }

  //Find best rates per group

  GroupInfo *group = &station->m_groupsTable[groupId];
  maxTpGroupId = GetGroupId (group->m_maxTpRate);
  maxTpRateId = GetRateId (group->m_maxTpRate);
  maxTpProb = group->m_ratesTable[maxTpRateId].ewmaProb;
  maxTpTh = station->m_groupsTable[maxTpGroupId].m_ratesTable[maxTpRateId].throughput;

  maxTp2GroupId = GetGroupId (group->m_maxTpRate2);
  maxTp2RateId = GetRateId (group->m_maxTpRate2);
  maxTp2Prob = group->m_ratesTable[maxTp2RateId].ewmaProb;
  maxTp2Th = station->m_groupsTable[maxTp2GroupId].m_ratesTable[maxTp2RateId].throughput;

  if (th > maxTpTh || (th == maxTpTh && prob > maxTpProb))
    {
      group->m_maxTpRate2 = group->m_maxTpRate;
      group->m_maxTpRate = index;
    }
  else if (th > maxTp2Th || (th == maxTp2Th && prob > maxTp2Prob))
    {
      group->m_maxTpRate2 = index;
    }
}

void
MinstrelHtWifiManager::RateInit (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  station->m_groupsTable = McsGroupData (m_numGroups);

  /**
  * Initialize groups supported by the receiver.
  */
  NS_LOG_DEBUG ("Supported groups by station:");
  bool noSupportedGroupFound = true;
  for (uint8_t groupId = 0; groupId < m_numGroups; groupId++)
    {
      if (m_minstrelGroups[groupId].isSupported)
        {
          station->m_groupsTable[groupId].m_supported = false;

          if ((m_minstrelGroups[groupId].type == GROUP_HE) && !GetHeSupported (station))
            {
              //It is a HE group but the receiver does not support HE: skip
              continue;
            }
          if ((m_minstrelGroups[groupId].type == GROUP_VHT) && !GetVhtSupported (station))
            {
              //It is a VHT group but the receiver does not support VHT: skip
              continue;
            }
          if ((m_minstrelGroups[groupId].type != GROUP_HE) && GetHeSupported (station) && m_useLatestAmendmentOnly)
            {
              //It is not a HE group and the receiver supports HE: skip since UseLatestAmendmentOnly attribute is enabled
              continue;
            }
          if (!GetHeSupported (station) && (m_minstrelGroups[groupId].type != GROUP_VHT) && GetVhtSupported (station) && m_useLatestAmendmentOnly)
            {
              //It is not a VHT group and the receiver supports VHT (but not HE): skip since UseLatestAmendmentOnly attribute is enabled
              continue;
            }
          if (((m_minstrelGroups[groupId].type == GROUP_HT) || (m_minstrelGroups[groupId].type == GROUP_VHT))
              && (m_minstrelGroups[groupId].gi == 400) && !GetShortGuardIntervalSupported (station))
            {
              //It is a SGI group but the receiver does not support SGI: skip
              continue;
            }
          if ((m_minstrelGroups[groupId].type == GROUP_HE) && (m_minstrelGroups[groupId].gi < GetGuardInterval (station)))
            {
              //The receiver does not support the GI: skip
              continue;
            }
          if (GetChannelWidth (station) < m_minstrelGroups[groupId].chWidth)
            {
              //The receiver does not support the channel width: skip
              continue;
            }
          if (GetNumberOfSupportedStreams (station) < m_minstrelGroups[groupId].streams)
            {
              //The receiver does not support the number of spatial streams: skip
              continue;
            }

          NS_LOG_DEBUG ("Group: " << +groupId
                                 << " type: " << m_minstrelGroups[groupId].type
                                 << " streams: " << +m_minstrelGroups[groupId].streams
                                 << " GI: " << m_minstrelGroups[groupId].gi
                                 << " width: " << m_minstrelGroups[groupId].chWidth);

          noSupportedGroupFound = false;
          station->m_groupsTable[groupId].m_supported = true;
          station->m_groupsTable[groupId].m_col = 0;
          station->m_groupsTable[groupId].m_index = 0;

          station->m_groupsTable[groupId].m_ratesTable = MinstrelHtRate (m_numRates); ///Create the rate list for the group.
          for (uint8_t i = 0; i < m_numRates; i++)
            {
              station->m_groupsTable[groupId].m_ratesTable[i].supported = false;
            }

          // Initialize all modes supported by the remote station that belong to the current group.
          for (uint8_t i = 0; i < station->m_nModes; i++)
            {
              WifiMode mode = GetMcsSupported (station, i);

              ///Use the McsValue as the index in the rate table.
              ///This way, MCSs not supported are not initialized.
              uint8_t rateId = mode.GetMcsValue ();
              if (mode.GetModulationClass () == WIFI_MOD_CLASS_HT)
                {
                  rateId %= MAX_HT_GROUP_RATES;
                }

              if (((m_minstrelGroups[groupId].type == GROUP_HE)
                    && (mode.GetModulationClass () == WIFI_MOD_CLASS_HE)                                                   ///If it is a HE MCS only add to a HE group.
                    && IsValidMcs (GetPhy (), m_minstrelGroups[groupId].streams, m_minstrelGroups[groupId].chWidth, mode)) ///Check validity of the HE MCS
                  || ((m_minstrelGroups[groupId].type == GROUP_VHT)
                    && (mode.GetModulationClass () == WIFI_MOD_CLASS_VHT)                                                  ///If it is a VHT MCS only add to a VHT group.
                    && IsValidMcs (GetPhy (), m_minstrelGroups[groupId].streams, m_minstrelGroups[groupId].chWidth, mode)) ///Check validity of the VHT MCS
                  || ((m_minstrelGroups[groupId].type == GROUP_HT)
                      && (mode.GetModulationClass () == WIFI_MOD_CLASS_HT)                                                 ///If it is a HT MCS only add to a HT group.
                      && (mode.GetMcsValue () < (m_minstrelGroups[groupId].streams * 8))                                     ///Check if the HT MCS corresponds to groups number of streams.
                      && (mode.GetMcsValue () >= ((m_minstrelGroups[groupId].streams - 1) * 8))))
                {
                  NS_LOG_DEBUG ("Mode " << +i << ": " << mode);

                  station->m_groupsTable[groupId].m_ratesTable[rateId].supported = true;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].mcsIndex = i; ///Mapping between rateId and operationalMcsSet
                  station->m_groupsTable[groupId].m_ratesTable[rateId].numRateAttempt = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].numRateSuccess = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].prob = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].ewmaProb = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].prevNumRateAttempt = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].prevNumRateSuccess = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].numSamplesSkipped = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].successHist = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].attemptHist = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].throughput = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].perfectTxTime = GetFirstMpduTxTime (groupId, GetMcsSupported (station, i));
                  station->m_groupsTable[groupId].m_ratesTable[rateId].retryCount = 0;
                  station->m_groupsTable[groupId].m_ratesTable[rateId].adjustedRetryCount = 0;
                  CalculateRetransmits (station, groupId, rateId);
                }
            }
        }
    }
  ///make sure at least one group is supported, otherwise we end up with an infinite loop in SetNextSample
  if (noSupportedGroupFound)
    {
      NS_FATAL_ERROR ("No supported group has been found");
    }
  SetNextSample (station);                  /// Select the initial sample index.
  UpdateStats (station);                    /// Calculate the initial high throughput rates.
  station->m_txrate = FindRate (station);   /// Select the rate to use.
}

void
MinstrelHtWifiManager::CalculateRetransmits (MinstrelHtWifiRemoteStation *station, uint16_t index)
{
  NS_LOG_FUNCTION (this << station << index);
  uint8_t groupId = GetGroupId (index);
  uint8_t rateId = GetRateId (index);
  if (!station->m_groupsTable[groupId].m_ratesTable[rateId].retryUpdated)
    {
      CalculateRetransmits (station, groupId, rateId);
    }
}

void
MinstrelHtWifiManager::CalculateRetransmits (MinstrelHtWifiRemoteStation *station, uint8_t groupId, uint8_t rateId)
{
  NS_LOG_FUNCTION (this << station << +groupId << +rateId);

  uint32_t cw = 15;                     // Is an approximation.
  uint32_t cwMax = 1023;
  Time cwTime, txTime, dataTxTime;
  Time slotTime = GetPhy ()->GetSlot ();
  Time ackTime = GetPhy ()->GetSifs () + GetPhy ()->GetBlockAckTxTime ();

  if (station->m_groupsTable[groupId].m_ratesTable[rateId].ewmaProb < 1)
    {
      station->m_groupsTable[groupId].m_ratesTable[rateId].retryCount = 1;
    }
  else
    {
      station->m_groupsTable[groupId].m_ratesTable[rateId].retryCount = 2;
      station->m_groupsTable[groupId].m_ratesTable[rateId].retryUpdated = true;

      dataTxTime = GetFirstMpduTxTime (groupId, GetMcsSupported (station, station->m_groupsTable[groupId].m_ratesTable[rateId].mcsIndex)) +
        GetMpduTxTime (groupId, GetMcsSupported (station, station->m_groupsTable[groupId].m_ratesTable[rateId].mcsIndex)) * (station->m_avgAmpduLen - 1);

      /* Contention time for first 2 tries */
      cwTime = (cw / 2) * slotTime;
      cw = Min ((cw + 1) * 2, cwMax);
      cwTime += (cw / 2) * slotTime;
      cw = Min ((cw + 1) * 2, cwMax);

      /* Total TX time for data and Contention after first 2 tries */
      txTime = cwTime + 2 * (dataTxTime + ackTime);

      /* See how many more tries we can fit inside segment size */
      do
        {
          /* Contention time for this try */
          cwTime = (cw / 2) * slotTime;
          cw = Min ((cw + 1) * 2, cwMax);

          /* Total TX time after this try */
          txTime += cwTime + ackTime + dataTxTime;
        }
      while ((txTime < MilliSeconds (6))
             && (++station->m_groupsTable[groupId].m_ratesTable[rateId].retryCount < 7));
    }
}

double
MinstrelHtWifiManager::CalculateEwmsd (double oldEwmsd, double currentProb, double ewmaProb, double weight)
{
  double diff, incr, tmp;

  /* calculate exponential weighted moving variance */
  diff = currentProb - ewmaProb;
  incr = (100 - weight) * diff / 100;
  tmp = oldEwmsd * oldEwmsd;
  tmp = weight * (tmp + diff * incr) / 100;

  /* return standard deviation */
  return sqrt (tmp);
}

void
MinstrelHtWifiManager::InitSampleTable (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_col = station->m_index = 0;

  //for off-setting to make rates fall between 0 and nModes
  uint8_t numSampleRates = m_numRates;

  uint16_t newIndex;
  for (uint8_t col = 0; col < m_nSampleCol; col++)
    {
      for (uint8_t i = 0; i < numSampleRates; i++ )
        {
          /**
           * The next two lines basically tries to generate a random number
           * between 0 and the number of available rates
           */
          int uv = m_uniformRandomVariable->GetInteger (0, numSampleRates);
          newIndex = (i + uv) % numSampleRates;

          //this loop is used for filling in other uninitialized places
          while (station->m_sampleTable[newIndex][col] != 0)
            {
              newIndex = (newIndex + 1) % m_numRates;
            }
          station->m_sampleTable[newIndex][col] = i;
        }
    }
}

void
MinstrelHtWifiManager::PrintTable (MinstrelHtWifiRemoteStation *station)
{
  if (!station->m_statsFile.is_open ())
    {
      std::ostringstream tmp;
      tmp << "minstrel-ht-stats-" << station->m_state->m_address << ".txt";
      station->m_statsFile.open (tmp.str ().c_str (), std::ios::out);
    }

  station->m_statsFile << "               best   ____________rate__________    ________statistics________    ________last_______    ______sum-of________\n" <<
    " mode guard #  rate  [name   idx airtime  max_tp]  [avg(tp) avg(prob) sd(prob)]  [prob.|retry|suc|att]  [#success | #attempts]\n";
  for (uint8_t i = 0; i < m_numGroups; i++)
    {
      StatsDump (station, i, station->m_statsFile);
    }

  station->m_statsFile << "\nTotal packet count::    ideal " << Max (0, station->m_totalPacketsCount - station->m_samplePacketsCount) <<
    "              lookaround " << station->m_samplePacketsCount << "\n";
  station->m_statsFile << "Average # of aggregated frames per A-MPDU: " << station->m_avgAmpduLen << "\n\n";

  station->m_statsFile.flush ();
}

void
MinstrelHtWifiManager::StatsDump (MinstrelHtWifiRemoteStation *station, uint8_t groupId, std::ofstream &of)
{
  uint8_t numRates = m_numRates;
  McsGroup group = m_minstrelGroups[groupId];
  Time txTime;
  for (uint8_t i = 0; i < numRates; i++)
    {
      if (station->m_groupsTable[groupId].m_supported && station->m_groupsTable[groupId].m_ratesTable[i].supported)
        {
          of << group.type << " " << group.chWidth << "   " << group.gi << "  " << +group.streams << "   ";

          uint16_t maxTpRate = station->m_maxTpRate;
          uint16_t maxTpRate2 = station->m_maxTpRate2;
          uint16_t maxProbRate = station->m_maxProbRate;

          uint16_t idx = GetIndex (groupId, i);
          if (idx == maxTpRate)
            {
              of << 'A';
            }
          else
            {
              of << ' ';
            }
          if (idx == maxTpRate2)
            {
              of << 'B';
            }
          else
            {
              of << ' ';
            }
          if (idx == maxProbRate)
            {
              of << 'P';
            }
          else
            {
              of << ' ';
            }

          if (group.type == GROUP_HT)
            {
              of << std::setw (4) << "   MCS" << (group.streams - 1) * 8 + i;
            }
          else
            {
              of << std::setw (7) << "   MCS" << +i << "/" << static_cast<int>(group.streams);
            }

          of << "  " << std::setw (3) << +idx << "  ";

          /* tx_time[rate(i)] in usec */
          txTime = GetFirstMpduTxTime (groupId, GetMcsSupported (station, station->m_groupsTable[groupId].m_ratesTable[i].mcsIndex));
          of << std::setw (6) << txTime.GetMicroSeconds () << "  ";

          of << std::setw (7) << CalculateThroughput (station, groupId, i, 100) / 100 << "   " <<
            std::setw (7) << station->m_groupsTable[groupId].m_ratesTable[i].throughput / 100 << "   " <<
            std::setw (7) << station->m_groupsTable[groupId].m_ratesTable[i].ewmaProb << "  " <<
            std::setw (7) << station->m_groupsTable[groupId].m_ratesTable[i].ewmsdProb << "  " <<
            std::setw (7) << station->m_groupsTable[groupId].m_ratesTable[i].prob << "  " <<
            std::setw (2) << station->m_groupsTable[groupId].m_ratesTable[i].retryCount << "   " <<
            std::setw (3) << station->m_groupsTable[groupId].m_ratesTable[i].prevNumRateSuccess << "  " <<
            std::setw (3) << station->m_groupsTable[groupId].m_ratesTable[i].prevNumRateAttempt << "   " <<
            std::setw (9) << station->m_groupsTable[groupId].m_ratesTable[i].successHist << "   " <<
            std::setw (9) << station->m_groupsTable[groupId].m_ratesTable[i].attemptHist << "\n";
        }
    }
}
uint16_t
MinstrelHtWifiManager::GetIndex (uint8_t groupId, uint8_t rateId)
{
  NS_LOG_FUNCTION (this << +groupId << +rateId);
  uint16_t index;
  index = groupId * m_numRates + rateId;
  return index;
}

uint8_t
MinstrelHtWifiManager::GetRateId (uint16_t index)
{
  NS_LOG_FUNCTION (this << index);
  uint8_t id;
  id = index % m_numRates;
  return id;
}

uint8_t
MinstrelHtWifiManager::GetGroupId (uint16_t index)
{
  NS_LOG_FUNCTION (this << index);
  return index / m_numRates;
}

uint8_t
MinstrelHtWifiManager::GetHtGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth)
{
  NS_LOG_FUNCTION (this << +txstreams << gi << chWidth);
  uint8_t giIndex = (gi == 400) ? 1 : 0;
  uint8_t widthIndex = (chWidth == 40) ? 1 : 0;
  return (MAX_HT_SUPPORTED_STREAMS * 2 * widthIndex) + (MAX_HT_SUPPORTED_STREAMS * giIndex) + txstreams - 1;
}

uint8_t
MinstrelHtWifiManager::GetVhtGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth)
{
  NS_LOG_FUNCTION (this << +txstreams << gi << chWidth);
  uint8_t giIndex = (gi == 400) ? 1 : 0;
  uint8_t widthIndex;
  if (chWidth == 160)
    {
      widthIndex = 3;
    }
  else if (chWidth == 80)
    {
      widthIndex = 2;
    }
  else if (chWidth == 40)
    {
      widthIndex = 1;
    }
  else //20 MHz
    {
      widthIndex = 0;
    }
  uint8_t groupId = (MAX_HT_STREAM_GROUPS * MAX_HT_SUPPORTED_STREAMS); ///add all HT groups
  groupId += (MAX_VHT_SUPPORTED_STREAMS * 2 * widthIndex) + (MAX_VHT_SUPPORTED_STREAMS * giIndex) + txstreams - 1;
  return groupId;
}

uint8_t
MinstrelHtWifiManager::GetHeGroupId (uint8_t txstreams, uint16_t gi, uint16_t chWidth)
{
  NS_LOG_FUNCTION (this << +txstreams << gi << chWidth);
  uint8_t giIndex;
  if (gi == 800)
    {
      giIndex = 2;
    }
  else if (gi == 1600)
    {
      giIndex = 1;
    }
  else //3200 ns
    {
      giIndex = 0;
    }
  uint8_t widthIndex;
  if (chWidth == 160)
    {
      widthIndex = 3;
    }
  else if (chWidth == 80)
    {
      widthIndex = 2;
    }
  else if (chWidth == 40)
    {
      widthIndex = 1;
    }
  else //20 MHz
    {
      widthIndex = 0;
    }
  uint8_t groupId = (MAX_HT_STREAM_GROUPS * MAX_HT_SUPPORTED_STREAMS); ///add all HT groups
  if (GetVhtSupported ()) ///This check is needed since we do not support VHT in 2.4 GHz band
    {
      groupId += MAX_VHT_STREAM_GROUPS * MAX_VHT_SUPPORTED_STREAMS; ///add all VHT groups
    }
  groupId += (MAX_HE_SUPPORTED_STREAMS * 3 * widthIndex) + (MAX_HE_SUPPORTED_STREAMS * giIndex) + txstreams - 1;
  return groupId;
}

uint16_t
MinstrelHtWifiManager::GetLowestIndex (MinstrelHtWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);

  uint8_t groupId = 0;
  uint8_t rateId = 0;
  while (groupId < m_numGroups && !station->m_groupsTable[groupId].m_supported)
    {
      groupId++;
    }
  while (rateId < m_numRates && !station->m_groupsTable[groupId].m_ratesTable[rateId].supported)
    {
      rateId++;
    }
  NS_ASSERT (station->m_groupsTable[groupId].m_supported && station->m_groupsTable[groupId].m_ratesTable[rateId].supported);
  return GetIndex (groupId, rateId);
}

uint16_t
MinstrelHtWifiManager::GetLowestIndex (MinstrelHtWifiRemoteStation *station, uint8_t groupId)
{
  NS_LOG_FUNCTION (this << station << +groupId);

  uint8_t rateId = 0;
  while (rateId < m_numRates && !station->m_groupsTable[groupId].m_ratesTable[rateId].supported)
    {
      rateId++;
    }
  NS_ASSERT (station->m_groupsTable[groupId].m_supported && station->m_groupsTable[groupId].m_ratesTable[rateId].supported);
  return GetIndex (groupId, rateId);
}

WifiModeList
MinstrelHtWifiManager::GetHeDeviceMcsList () const
{
  WifiModeList heMcsList;
  for (const auto & mode : GetPhy ()->GetMcsList (WIFI_MOD_CLASS_HE))
    {
      heMcsList.push_back (mode);
    }
  return heMcsList;
}

WifiModeList
MinstrelHtWifiManager::GetVhtDeviceMcsList (void) const
{
  WifiModeList vhtMcsList;
  for (const auto & mode : GetPhy ()->GetMcsList (WIFI_MOD_CLASS_VHT))
    {
      vhtMcsList.push_back (mode);
    }
  return vhtMcsList;
}

WifiModeList
MinstrelHtWifiManager::GetHtDeviceMcsList (void) const
{
  WifiModeList htMcsList;
  for (const auto & mode : GetPhy ()->GetMcsList (WIFI_MOD_CLASS_HT))
    {
      htMcsList.push_back (mode);
    }
  return htMcsList;
}

} // namespace ns3
