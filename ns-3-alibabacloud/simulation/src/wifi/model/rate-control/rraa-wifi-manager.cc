/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2004,2005,2006 INRIA
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
 * Author: Federico Maguolo <maguolof@dei.unipd.it>
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "rraa-wifi-manager.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-mac.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RraaWifiManager");

/**
 * \brief hold per-remote-station state for RRAA Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the RRAA Wifi manager
 */
struct RraaWifiRemoteStation : public WifiRemoteStation
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
  uint8_t m_rateIndex;           //!< Current rate index.

  RraaThresholdsTable m_thresholds; //!< RRAA thresholds for this station.
};

NS_OBJECT_ENSURE_REGISTERED (RraaWifiManager);

TypeId
RraaWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RraaWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<RraaWifiManager> ()
    .AddAttribute ("Basic",
                   "If true the RRAA-BASIC algorithm will be used, otherwise the RRAA will be used",
                   BooleanValue (false),
                   MakeBooleanAccessor (&RraaWifiManager::m_basic),
                   MakeBooleanChecker ())
    .AddAttribute ("Timeout",
                   "Timeout for the RRAA BASIC loss estimation block",
                   TimeValue (Seconds (0.05)),
                   MakeTimeAccessor (&RraaWifiManager::m_timeout),
                   MakeTimeChecker ())
    .AddAttribute ("FrameLength",
                   "The Data frame length (in bytes) used for calculating mode TxTime.",
                   UintegerValue (1420),
                   MakeUintegerAccessor (&RraaWifiManager::m_frameLength),
                   MakeUintegerChecker <uint32_t> ())
    .AddAttribute ("AckFrameLength",
                   "The Ack frame length (in bytes) used for calculating mode TxTime.",
                   UintegerValue (14),
                   MakeUintegerAccessor (&RraaWifiManager::m_ackLength),
                   MakeUintegerChecker <uint32_t> ())
    .AddAttribute ("Alpha",
                   "Constant for calculating the MTL threshold.",
                   DoubleValue (1.25),
                   MakeDoubleAccessor (&RraaWifiManager::m_alpha),
                   MakeDoubleChecker<double> (1))
    .AddAttribute ("Beta",
                   "Constant for calculating the ORI threshold.",
                   DoubleValue (2),
                   MakeDoubleAccessor (&RraaWifiManager::m_beta),
                   MakeDoubleChecker<double> (1))
    .AddAttribute ("Tau",
                   "Constant for calculating the EWND size.",
                   DoubleValue (0.012),
                   MakeDoubleAccessor (&RraaWifiManager::m_tau),
                   MakeDoubleChecker<double> (0))
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&RraaWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

RraaWifiManager::RraaWifiManager ()
  : WifiRemoteStationManager (),
    m_currentRate (0)
{
  NS_LOG_FUNCTION (this);
}

RraaWifiManager::~RraaWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
RraaWifiManager::SetupPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_sifs = phy->GetSifs ();
  m_difs = m_sifs + 2 * phy->GetSlot ();
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
RraaWifiManager::SetupMac (const Ptr<WifiMac> mac)
{
  NS_LOG_FUNCTION (this);
  WifiRemoteStationManager::SetupMac (mac);
}

void
RraaWifiManager::DoInitialize ()
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
RraaWifiManager::GetCalcTxTime (WifiMode mode) const
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
RraaWifiManager::AddCalcTxTime (WifiMode mode, Time t)
{
  NS_LOG_FUNCTION (this << mode << t);
  m_calcTxTime.push_back (std::make_pair (t, mode));
}

WifiRraaThresholds
RraaWifiManager::GetThresholds (RraaWifiRemoteStation *station, WifiMode mode) const
{
  NS_LOG_FUNCTION (this << station << mode);
  struct WifiRraaThresholds threshold;
  for (RraaThresholdsTable::const_iterator i = station->m_thresholds.begin (); i != station->m_thresholds.end (); i++)
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
RraaWifiManager::DoCreateStation (void) const
{
  RraaWifiRemoteStation *station = new RraaWifiRemoteStation ();
  station->m_initialized = false;
  station->m_adaptiveRtsWnd = 0;
  station->m_rtsCounter = 0;
  station->m_adaptiveRtsOn = false;
  station->m_lastFrameFail = false;
  return station;
}

void
RraaWifiManager::CheckInit (RraaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (!station->m_initialized)
    {
      //Note: we appear to be doing late initialization of the table
      //to make sure that the set of supported rates has been initialized
      //before we perform our own initialization.
      station->m_nRate = GetNSupported (station);
      //Initialize at maximal rate
      station->m_rateIndex = GetMaxRate (station);

      station->m_initialized = true;

      station->m_thresholds = RraaThresholdsTable (station->m_nRate);
      InitThresholds (station);
      ResetCountersBasic (station);
    }
}

void
RraaWifiManager::InitThresholds (RraaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  NS_LOG_DEBUG ("InitThresholds = " << station);

  double nextCritical = 0;
  double nextMtl = 0;
  double mtl = 0;
  double ori = 0;
  for (uint8_t i = 0; i < station->m_nRate; i++)
    {
      WifiMode mode = GetSupported (station, i);
      Time totalTxTime = GetCalcTxTime (mode) + m_sifs + m_difs;
      if (i == GetMaxRate (station))
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
          mtl = 1;
        }
      WifiRraaThresholds th;
      th.m_ewnd = static_cast<uint32_t> (ceil (m_tau / totalTxTime.GetSeconds ()));
      th.m_ori = ori;
      th.m_mtl = mtl;
      station->m_thresholds.push_back (std::make_pair (th, mode));
      mtl = nextMtl;
      NS_LOG_DEBUG (mode << " " << th.m_ewnd << " " << th.m_mtl << " " << th.m_ori);
    }
}

void
RraaWifiManager::ResetCountersBasic (RraaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_nFailed = 0;
  station->m_counter = GetThresholds (station, station->m_rateIndex).m_ewnd;
  station->m_lastReset = Simulator::Now ();
}

uint8_t
RraaWifiManager::GetMaxRate (RraaWifiRemoteStation *station) const
{
  return station->m_nRate - 1;
}

void
RraaWifiManager::DoReportRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}

void
RraaWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RraaWifiRemoteStation *station = static_cast<RraaWifiRemoteStation*> (st);
  station->m_lastFrameFail = true;
  CheckTimeout (station);
  station->m_counter--;
  station->m_nFailed++;
  RunBasicAlgorithm (station);
}

void
RraaWifiManager::DoReportRxOk (WifiRemoteStation *st,
                               double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << st << rxSnr << txMode);
}

void
RraaWifiManager::DoReportRtsOk (WifiRemoteStation *st,
                                double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
}

void
RraaWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                 double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  RraaWifiRemoteStation *station = static_cast<RraaWifiRemoteStation*> (st);
  station->m_lastFrameFail = false;
  CheckTimeout (station);
  station->m_counter--;
  RunBasicAlgorithm (station);
}

void
RraaWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}

void
RraaWifiManager::DoReportFinalDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
}

WifiTxVector
RraaWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RraaWifiRemoteStation *station = static_cast<RraaWifiRemoteStation*> (st);
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  CheckInit (station);
  WifiMode mode = GetSupported (station, station->m_rateIndex);
  uint64_t rate = mode.GetDataRate (channelWidth);
  if (m_currentRate != rate)
    {
      NS_LOG_DEBUG ("New datarate: " << rate);
      m_currentRate = rate;
    }
  return WifiTxVector (mode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}

WifiTxVector
RraaWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  RraaWifiRemoteStation *station = static_cast<RraaWifiRemoteStation*> (st);
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
RraaWifiManager::DoNeedRts (WifiRemoteStation *st,
                            uint32_t size, bool normally)
{
  NS_LOG_FUNCTION (this << st << size << normally);
  RraaWifiRemoteStation *station = static_cast<RraaWifiRemoteStation*> (st);
  CheckInit (station);
  if (m_basic)
    {
      return normally;
    }
  ARts (station);
  return station->m_adaptiveRtsOn;
}

void
RraaWifiManager::CheckTimeout (RraaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  Time d = Simulator::Now () - station->m_lastReset;
  if (station->m_counter == 0 || d > m_timeout)
    {
      ResetCountersBasic (station);
    }
}

void
RraaWifiManager::RunBasicAlgorithm (RraaWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  WifiRraaThresholds thresholds = GetThresholds (station, station->m_rateIndex);
  double ploss = (station->m_nFailed / thresholds.m_ewnd);
  if (station->m_counter == 0
      || ploss > thresholds.m_mtl)
    {
      if (ploss > thresholds.m_mtl)
        {
          station->m_rateIndex--;
        }
      else if (station->m_rateIndex < GetMaxRate (station)
               && ploss < thresholds.m_ori)
        {
          station->m_rateIndex++;
        }
      ResetCountersBasic (station);
    }
}

void
RraaWifiManager::ARts (RraaWifiRemoteStation *station)
{
  if (!station->m_adaptiveRtsOn
      && station->m_lastFrameFail)
    {
      station->m_adaptiveRtsWnd++;
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

WifiRraaThresholds
RraaWifiManager::GetThresholds (RraaWifiRemoteStation *station, uint8_t index) const
{
  NS_LOG_FUNCTION (this << station << +index);
  WifiMode mode = GetSupported (station, index);
  return GetThresholds (station, mode);
}

} //namespace ns3
