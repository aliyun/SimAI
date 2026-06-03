/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universidad de la República - Uruguay
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
 * Author: Matías Richart <mrichart@fing.edu.uy>
 */

#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"
#include "ns3/data-rate.h"
#include "rrpaa-wifi-manager.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-mac.h"

NS_LOG_COMPONENT_DEFINE ("RrpaaWifiManager");

namespace ns3 {

/**
 * Hold per-remote-station state for RRPAA Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the APARF Wifi manager
 */
struct RrpaaWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_counter;            //!< Counter for transmission attempts.
  uint32_t m_nFailed;            //!< Number of failed transmission attempts.
  uint32_t m_adaptiveRtsWnd;     //!< Window size for the Adaptive RTS mechanism.
  uint32_t m_rtsCounter;         //!< Counter for RTS transmission attempts.
  Time m_lastReset;              //!< Time of the last reset.
  bool m_adaptiveRtsOn;          //!< Check if Adaptive RTS mechanism is on.
  bool m_lastFrameFail;          //!< Flag if the last frame sent has failed.
  bool m_initialized;            //!< For initializing variables.
  uint8_t m_nRate;               //!< Number of supported rates.
  uint8_t m_prevRateIndex;       //!< Rate index of the previous transmission.
  uint8_t m_rateIndex;           //!< Current rate index.
  uint8_t m_prevPowerLevel;      //!< Power level of the previous transmission.
  uint8_t m_powerLevel;          //!< Current power level.
  RrpaaThresholdsTable m_thresholds;  //!< RRPAA thresholds for this station.
  RrpaaProbabilitiesTable m_pdTable;  //!< Probability table for power and rate changes.
};

NS_OBJECT_ENSURE_REGISTERED (RrpaaWifiManager);

TypeId
RrpaaWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RrpaaWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<RrpaaWifiManager> ()
    .AddAttribute ("Basic",
                   "If true the RRPAA-BASIC algorithm will be used, otherwise the RRPAA will be used.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&RrpaaWifiManager::m_basic),
                   MakeBooleanChecker ())
    .AddAttribute ("Timeout",
                   "Timeout for the RRPAA-BASIC loss estimation block.",
                   TimeValue (MilliSeconds (500)),
                   MakeTimeAccessor (&RrpaaWifiManager::m_timeout),
                   MakeTimeChecker ())
    .AddAttribute ("FrameLength",
                   "The Data frame length (in bytes) used for calculating mode TxTime.",
                   UintegerValue (1420),
                   MakeUintegerAccessor (&RrpaaWifiManager::m_frameLength),
                   MakeUintegerChecker <uint32_t> ())
    .AddAttribute ("AckFrameLength",
                   "The Ack frame length (in bytes) used for calculating mode TxTime.",
                   UintegerValue (14),
                   MakeUintegerAccessor (&RrpaaWifiManager::m_ackLength),
                   MakeUintegerChecker <uint32_t> ())
    .AddAttribute ("Alpha",
                   "Constant for calculating the MTL threshold.",
                   DoubleValue (1.25),
                   MakeDoubleAccessor (&RrpaaWifiManager::m_alpha),
                   MakeDoubleChecker<double> (1))
    .AddAttribute ("Beta",
                   "Constant for calculating the ORI threshold.",
                   DoubleValue (2),
                   MakeDoubleAccessor (&RrpaaWifiManager::m_beta),
                   MakeDoubleChecker<double> (1))
    .AddAttribute ("Tau",
                   "Constant for calculating the EWND size.",
                   DoubleValue (0.015),
                   MakeDoubleAccessor (&RrpaaWifiManager::m_tau),
                   MakeDoubleChecker<double> (0))
    .AddAttribute ("Gamma",
                   "Constant for Probabilistic Decision Table decrements.",
                   DoubleValue (2),
                   MakeDoubleAccessor (&RrpaaWifiManager::m_gamma),
                   MakeDoubleChecker<double> (1))
    .AddAttribute ("Delta",
                   "Constant for Probabilistic Decision Table increments.",
                   DoubleValue (1.0905),
                   MakeDoubleAccessor (&RrpaaWifiManager::m_delta),
                   MakeDoubleChecker<double> (1))
    .AddTraceSource ("RateChange",
                     "The transmission rate has change.",
                     MakeTraceSourceAccessor (&RrpaaWifiManager::m_rateChange),
                     "ns3::WifiRemoteStationManager::RateChangeTracedCallback")
    .AddTraceSource ("PowerChange",
                     "The transmission power has change.",
                     MakeTraceSourceAccessor (&RrpaaWifiManager::m_powerChange),
                     "ns3::WifiRemoteStationManager::PowerChangeTracedCallback")
  ;
  return tid;
}


RrpaaWifiManager::RrpaaWifiManager ()
{
  NS_LOG_FUNCTION (this);
  m_uniformRandomVariable = CreateObject<UniformRandomVariable> ();
}

RrpaaWifiManager::~RrpaaWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

int64_t
RrpaaWifiManager::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_uniformRandomVariable->SetStream (stream);
  return 1;
}

void
RrpaaWifiManager::SetupPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_sifs = phy->GetSifs ();
  m_difs = m_sifs + 2 * phy->GetSlot ();
  m_nPowerLevels = phy->GetNTxPower ();
  m_maxPowerLevel = m_nPowerLevels  - 1;
  m_minPowerLevel = 0;
  for (const auto & mode : phy->GetModeList ())
    {
      WifiTxVector txVector;
      txVector.SetMode (mode);
      txVector.SetPreambleType (WIFI_PREAMBLE_LONG);
      /* Calculate the TX Time of the Data and the corresponding Ack */
      Time dataTxTime = phy->CalculateTxDuration (m_frameLength, txVector, phy->GetPhyBand ());
      Time ackTxTime = phy->CalculateTxDuration (m_ackLength, txVector, phy->GetPhyBand ());
      NS_LOG_DEBUG ("Calculating TX times: Mode= " << mode << " DataTxTime= " << dataTxTime << " AckTxTime= " << ackTxTime);
      AddCalcTxTime (mode, dataTxTime + ackTxTime);
    }
  WifiRemoteStationManager::SetupPhy (phy);
}

void
RrpaaWifiManager::SetupMac (const Ptr<WifiMac> mac)
{
  NS_LOG_FUNCTION (this << mac);
  WifiRemoteStationManager::SetupMac (mac);
}

void
RrpaaWifiManager::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  if (GetHtSupported ())
    {
      NS_FATAL_ERROR ("WifiRemoteStationManager selected does not support HT rates");
    }
  if (GetVhtSupported ())
    {
      NS_FATAL_ERROR ("WifiRemoteStationManager selected does not support VHT rates");
    }
  if (GetHeSupported ())
    {
      NS_FATAL_ERROR ("WifiRemoteStationManager selected does not support HE rates");
    }
}

Time
RrpaaWifiManager::GetCalcTxTime (WifiMode mode) const
{
  NS_LOG_FUNCTION (this << mode);
  for (TxTime::const_iterator i = m_calcTxTime.begin (); i != m_calcTxTime.end (); i++)
    {
      if (mode == i->second)
        {
          return i->first;
        }
    }
  NS_ASSERT (false);
  return Seconds (0);
}

void
RrpaaWifiManager::AddCalcTxTime (WifiMode mode, Time t)
{
  NS_LOG_FUNCTION (this << mode << t);
  m_calcTxTime.push_back (std::make_pair (t, mode));
}

WifiRrpaaThresholds
RrpaaWifiManager::GetThresholds (RrpaaWifiRemoteStation *station, WifiMode mode) const
{
  NS_LOG_FUNCTION (this << station << mode);
  struct WifiRrpaaThresholds threshold;
  for (RrpaaThresholdsTable::const_iterator i = station->m_thresholds.begin (); i != station->m_thresholds.end (); i++)
    {
      if (mode == i->second)
        {
          return i->first;
        }
    }
  NS_ABORT_MSG ("No thresholds for mode " << mode << " found");
  return threshold; // Silence compiler warning
}

WifiRemoteStation *
RrpaaWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  RrpaaWifiRemoteStation *station = new RrpaaWifiRemoteStation ();
  station->m_adaptiveRtsWnd = 0;
  station->m_rtsCounter = 0;
  station->m_adaptiveRtsOn = false;
  station->m_lastFrameFail = false;
  station->m_initialized = false;
  return station;
}

void
RrpaaWifiManager::CheckInit (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (!station->m_initialized)
    {
      //Note: we appear to be doing late initialization of the table
      //to make sure that the set of supported rates has been initialized
      //before we perform our own initialization.
      station->m_nRate = GetNSupported (station);
      //Initialize at minimal rate and maximal power.
      station->m_prevRateIndex = 0;
      station->m_rateIndex = 0;
      station->m_prevPowerLevel = m_maxPowerLevel;
      station->m_powerLevel = m_maxPowerLevel;
      WifiMode mode = GetSupported (station, 0);
      uint16_t channelWidth = GetChannelWidth (station);
      DataRate rate = DataRate (mode.GetDataRate (channelWidth));
      double power = GetPhy ()->GetPowerDbm (station->m_powerLevel);
      m_rateChange (rate, rate, station->m_state->m_address);
      m_powerChange (power, power, station->m_state->m_address);

      station->m_pdTable = RrpaaProbabilitiesTable (station->m_nRate, std::vector<double> (m_nPowerLevels));
      NS_LOG_DEBUG ("Initializing pdTable");
      for (uint8_t i = 0; i < station->m_nRate; i++)
        {
          for (uint8_t j = 0; j < m_nPowerLevels; j++)
            {
              station->m_pdTable[i][j] = 1;
            }
        }

      station->m_initialized = true;

      station->m_thresholds = RrpaaThresholdsTable (station->m_nRate);
      InitThresholds (station);
      ResetCountersBasic (station);
    }
}

void
RrpaaWifiManager::InitThresholds (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  double nextCritical = 0;
  double nextMtl = 0;
  double mtl = 0;
  double ori = 0;
  for (uint8_t i = 0; i < station->m_nRate; i++)
    {
      WifiMode mode = GetSupported (station, i);
      Time totalTxTime = GetCalcTxTime (mode) + m_sifs + m_difs;
      if (i == station->m_nRate - 1)
        {
          ori = 0;
        }
      else
        {
          WifiMode nextMode = GetSupported (station, i + 1);
          Time nextTotalTxTime = GetCalcTxTime (nextMode) + m_sifs + m_difs;
          nextCritical = 1 - (nextTotalTxTime.GetSeconds () / totalTxTime.GetSeconds ());
          nextMtl = m_alpha * nextCritical;
          ori = nextMtl / m_beta;
        }
      if (i == 0)
        {
          mtl = nextMtl;
        }
      WifiRrpaaThresholds th;
      th.m_ewnd = static_cast<uint32_t> (ceil (m_tau / totalTxTime.GetSeconds ()));
      th.m_ori = ori;
      th.m_mtl = mtl;
      station->m_thresholds.push_back (std::make_pair (th, mode));
      mtl = nextMtl;
      NS_LOG_DEBUG (mode << " " << th.m_ewnd << " " << th.m_mtl << " " << th.m_ori);
    }
}

void
RrpaaWifiManager::ResetCountersBasic (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_nFailed = 0;
  station->m_counter = GetThresholds (station, station->m_rateIndex).m_ewnd;
  station->m_lastReset = Simulator::Now ();
}

void
RrpaaWifiManager::DoReportRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}

void
RrpaaWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RrpaaWifiRemoteStation *station = static_cast<RrpaaWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_lastFrameFail = true;
  CheckTimeout (station);
  station->m_counter--;
  station->m_nFailed++;
  RunBasicAlgorithm (station);
}

void
RrpaaWifiManager::DoReportRxOk (WifiRemoteStation *st,
                                double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << st << rxSnr << txMode);
}

void
RrpaaWifiManager::DoReportRtsOk (WifiRemoteStation *st,
                                 double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
}

void
RrpaaWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                  double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  RrpaaWifiRemoteStation *station = static_cast<RrpaaWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_lastFrameFail = false;
  CheckTimeout (station);
  station->m_counter--;
  RunBasicAlgorithm (station);
}
void
RrpaaWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}
void
RrpaaWifiManager::DoReportFinalDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}

WifiTxVector
RrpaaWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RrpaaWifiRemoteStation *station = static_cast<RrpaaWifiRemoteStation*> (st);
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  CheckInit (station);
  WifiMode mode = GetSupported (station, station->m_rateIndex);
  DataRate rate = DataRate (mode.GetDataRate (channelWidth));
  DataRate prevRate = DataRate (GetSupported (station, station->m_prevRateIndex).GetDataRate (channelWidth));
  double power = GetPhy ()->GetPowerDbm (station->m_powerLevel);
  double prevPower = GetPhy ()->GetPowerDbm (station->m_prevPowerLevel);
  if (station->m_prevRateIndex != station->m_rateIndex)
    {
      m_rateChange (prevRate, rate, station->m_state->m_address);
      station->m_prevRateIndex = station->m_rateIndex;
    }
  if (station->m_prevPowerLevel != station->m_powerLevel)
    {
      m_powerChange (prevPower, power, station->m_state->m_address);
      station->m_prevPowerLevel = station->m_powerLevel;
    }
  return WifiTxVector (mode, station->m_powerLevel, GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}
WifiTxVector
RrpaaWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RrpaaWifiRemoteStation *station = static_cast<RrpaaWifiRemoteStation*> (st);
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  WifiMode mode;
  if (GetUseNonErpProtection () == false)
    {
      mode = GetSupported (station, 0);
    }
  else
    {
      mode = GetNonErpSupported (station, 0);
    }
  return WifiTxVector (mode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}

bool
RrpaaWifiManager::DoNeedRts (WifiRemoteStation *st,
                             uint32_t size, bool normally)
{
  NS_LOG_FUNCTION (this << st << size << normally);
  RrpaaWifiRemoteStation *station = static_cast<RrpaaWifiRemoteStation*> (st);
  CheckInit (station);
  if (m_basic)
    {
      return normally;
    }
  RunAdaptiveRtsAlgorithm (station);
  return station->m_adaptiveRtsOn;
}

void
RrpaaWifiManager::CheckTimeout (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  Time d = Simulator::Now () - station->m_lastReset;
  if (station->m_counter == 0 || d > m_timeout)
    {
      ResetCountersBasic (station);
    }
}

void
RrpaaWifiManager::RunBasicAlgorithm (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  WifiRrpaaThresholds thresholds = GetThresholds (station, station->m_rateIndex);
  double bploss = (static_cast<double> (station->m_nFailed) / thresholds.m_ewnd);
  double wploss = (static_cast<double> (station->m_counter + station->m_nFailed) / thresholds.m_ewnd);
  NS_LOG_DEBUG ("Best loss prob= " << bploss);
  NS_LOG_DEBUG ("Worst loss prob= " << wploss);
  if (bploss >= thresholds.m_mtl)
    {
      if (station->m_powerLevel < m_maxPowerLevel)
        {
          NS_LOG_DEBUG ("bploss >= MTL and power < maxPower => Increase Power");
          station->m_pdTable[station->m_rateIndex][station->m_powerLevel] /= m_gamma;
          NS_LOG_DEBUG ("pdTable[" << +station->m_rateIndex << "][" << station->m_powerLevel << "] = " << station->m_pdTable[station->m_rateIndex][station->m_powerLevel]);
          station->m_powerLevel++;
          ResetCountersBasic (station);
        }
      else if (station->m_rateIndex != 0)
        {
          NS_LOG_DEBUG ("bploss >= MTL and power = maxPower => Decrease Rate");
          station->m_pdTable[station->m_rateIndex][station->m_powerLevel] /= m_gamma;
          NS_LOG_DEBUG ("pdTable[" << +station->m_rateIndex << "][" << station->m_powerLevel << "] = " << station->m_pdTable[station->m_rateIndex][station->m_powerLevel]);
          station->m_rateIndex--;
          ResetCountersBasic (station);
        }
      else
        {
          NS_LOG_DEBUG ("bploss >= MTL but already at maxPower and minRate");
        }
    }
  else if (wploss <= thresholds.m_ori)
    {
      if (station->m_rateIndex < station->m_nRate - 1)
        {
          NS_LOG_DEBUG ("wploss <= ORI and rate < maxRate => Probabilistic Rate Increase");

          // Recalculate probabilities of lower rates.
          for (uint8_t i = 0; i <= station->m_rateIndex; i++)
            {
              station->m_pdTable[i][station->m_powerLevel] *= m_delta;
              if (station->m_pdTable[i][station->m_powerLevel] > 1)
                {
                  station->m_pdTable[i][station->m_powerLevel] = 1;
                }
              NS_LOG_DEBUG ("pdTable[" << i << "][" << (int)station->m_powerLevel << "] = " << station->m_pdTable[i][station->m_powerLevel]);
            }
          double rand = m_uniformRandomVariable->GetValue (0,1);
          if (rand < station->m_pdTable[station->m_rateIndex + 1][station->m_powerLevel])
            {
              NS_LOG_DEBUG ("Increase Rate");
              station->m_rateIndex++;
            }
        }
      else if (station->m_powerLevel > m_minPowerLevel)
        {
          NS_LOG_DEBUG ("wploss <= ORI and rate = maxRate => Probabilistic Power Decrease");

          // Recalculate probabilities of higher powers.
          for (uint32_t i = m_maxPowerLevel; i > station->m_powerLevel; i--)
            {
              station->m_pdTable[station->m_rateIndex][i] *= m_delta;
              if (station->m_pdTable[station->m_rateIndex][i] > 1)
                {
                  station->m_pdTable[station->m_rateIndex][i] = 1;
                }
              NS_LOG_DEBUG ("pdTable[" << +station->m_rateIndex << "][" << i << "] = " << station->m_pdTable[station->m_rateIndex][i]);
            }
          double rand = m_uniformRandomVariable->GetValue (0,1);
          if (rand < station->m_pdTable[station->m_rateIndex][station->m_powerLevel - 1])
            {
              NS_LOG_DEBUG ("Decrease Power");
              station->m_powerLevel--;
            }
        }
      ResetCountersBasic (station);
    }
  else if (bploss > thresholds.m_ori && wploss < thresholds.m_mtl)
    {
      if (station->m_powerLevel > m_minPowerLevel)
        {
          NS_LOG_DEBUG ("loss between ORI and MTL and power > minPowerLevel => Probabilistic Power Decrease");

          // Recalculate probabilities of higher powers.
          for (uint32_t i = m_maxPowerLevel; i >= station->m_powerLevel; i--)
            {
              station->m_pdTable[station->m_rateIndex][i] *= m_delta;
              if (station->m_pdTable[station->m_rateIndex][i] > 1)
                {
                  station->m_pdTable[station->m_rateIndex][i] = 1;
                }
              NS_LOG_DEBUG ("pdTable[" << +station->m_rateIndex << "][" << i << "] = " << station->m_pdTable[station->m_rateIndex][i]);
            }
          double rand = m_uniformRandomVariable->GetValue (0,1);
          if (rand < station->m_pdTable[station->m_rateIndex][station->m_powerLevel - 1])
            {
              NS_LOG_DEBUG ("Decrease Power");
              station->m_powerLevel--;
            }
          ResetCountersBasic (station);
        }
    }
  if (station->m_counter == 0)
    {
      ResetCountersBasic (station);
    }
}

void
RrpaaWifiManager::RunAdaptiveRtsAlgorithm (RrpaaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (!station->m_adaptiveRtsOn
      && station->m_lastFrameFail)
    {
      station->m_adaptiveRtsWnd += 2;
      station->m_rtsCounter = station->m_adaptiveRtsWnd;
    }
  else if ((station->m_adaptiveRtsOn && station->m_lastFrameFail)
           || (!station->m_adaptiveRtsOn && !station->m_lastFrameFail))
    {
      station->m_adaptiveRtsWnd = station->m_adaptiveRtsWnd / 2;
      station->m_rtsCounter = station->m_adaptiveRtsWnd;
    }
  if (station->m_rtsCounter > 0)
    {
      station->m_adaptiveRtsOn = true;
      station->m_rtsCounter--;
    }
  else
    {
      station->m_adaptiveRtsOn = false;
    }
}

WifiRrpaaThresholds
RrpaaWifiManager::GetThresholds (RrpaaWifiRemoteStation *station, uint8_t index) const
{
  NS_LOG_FUNCTION (this << station << +index);
  WifiMode mode = GetSupported (station, index);
  return GetThresholds (station, mode);
}

} // namespace ns3
