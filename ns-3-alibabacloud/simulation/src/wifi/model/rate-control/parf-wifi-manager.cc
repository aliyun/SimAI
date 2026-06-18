/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universidad de la República - Uruguay
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
 * Author: Matias Richart <mrichart@fing.edu.uy>
 */

#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ns3/data-rate.h"
#include "parf-wifi-manager.h"
#include "ns3/wifi-phy.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ParfWifiManager");

/**
 * Hold per-remote-station state for PARF Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the PARF Wifi manager
 */
struct ParfWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_nAttempt;       //!< Number of transmission attempts.
  uint32_t m_nSuccess;       //!< Number of successful transmission attempts.
  uint32_t m_nFail;          //!< Number of failed transmission attempts.
  bool m_usingRecoveryRate;  //!< If using recovery rate.
  bool m_usingRecoveryPower; //!< If using recovery power.
  uint32_t m_nRetry;         //!< Number of transmission retries.
  uint8_t m_prevRateIndex;   //!< Rate index of the previous transmission.
  uint8_t m_rateIndex;       //!< Current rate index used by the remote station.
  uint8_t m_prevPowerLevel;  //!< Power level of the previous transmission.
  uint8_t m_powerLevel;      //!< Current power level used by the remote station.
  uint8_t m_nSupported;      //!< Number of supported rates by the remote station.
  bool m_initialized;        //!< For initializing variables.
};

NS_OBJECT_ENSURE_REGISTERED (ParfWifiManager);

TypeId
ParfWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ParfWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ParfWifiManager> ()
    .AddAttribute ("AttemptThreshold",
                   "The minimum number of transmission attempts to try a new power or rate.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&ParfWifiManager::m_attemptThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("SuccessThreshold",
                   "The minimum number of successful transmissions to try a new power or rate.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&ParfWifiManager::m_successThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("PowerChange",
                     "The transmission power has change",
                     MakeTraceSourceAccessor (&ParfWifiManager::m_powerChange),
                     "ns3::WifiRemoteStationManager::PowerChangeTracedCallback")
    .AddTraceSource ("RateChange",
                     "The transmission rate has change",
                     MakeTraceSourceAccessor (&ParfWifiManager::m_rateChange),
                     "ns3::WifiRemoteStationManager::RateChangeTracedCallback")
  ;
  return tid;
}

ParfWifiManager::ParfWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

ParfWifiManager::~ParfWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
ParfWifiManager::SetupPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_minPower = 0;
  m_maxPower = phy->GetNTxPower () - 1;
  WifiRemoteStationManager::SetupPhy (phy);
}

void
ParfWifiManager::DoInitialize ()
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

WifiRemoteStation *
ParfWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  ParfWifiRemoteStation *station = new ParfWifiRemoteStation ();

  station->m_nSuccess = 0;
  station->m_nFail = 0;
  station->m_usingRecoveryRate = false;
  station->m_usingRecoveryPower = false;
  station->m_initialized = false;
  station->m_nRetry = 0;
  station->m_nAttempt = 0;

  NS_LOG_DEBUG ("create station=" << station << ", timer=" << station->m_nAttempt
                                  << ", rate=" << +station->m_rateIndex << ", power=" << +station->m_powerLevel);

  return station;
}

void
ParfWifiManager::CheckInit (ParfWifiRemoteStation *station)
{
  if (!station->m_initialized)
    {
      station->m_nSupported = GetNSupported (station);
      station->m_rateIndex = station->m_nSupported - 1;
      station->m_prevRateIndex = station->m_nSupported - 1;
      station->m_powerLevel = m_maxPower;
      station->m_prevPowerLevel = m_maxPower;
      WifiMode mode = GetSupported (station, station->m_rateIndex);
      uint16_t channelWidth = GetChannelWidth (station);
      DataRate rate = DataRate (mode.GetDataRate (channelWidth));
      double power = GetPhy ()->GetPowerDbm (m_maxPower);
      m_powerChange (power, power, station->m_state->m_address);
      m_rateChange (rate, rate, station->m_state->m_address);
      station->m_initialized = true;
    }
}

void
ParfWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

/*
 * It is important to realize that "recovery" mode starts after failure of
 * the first transmission after a rate increase and ends at the first successful
 * transmission. Specifically, recovery mode spans retransmissions boundaries.
 * Fundamentally, ARF handles each data transmission independently, whether it
 * is the initial transmission of a packet or the retransmission of a packet.
 * The fundamental reason for this is that there is a backoff between each data
 * transmission, be it an initial transmission or a retransmission.
 */
void
ParfWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  ParfWifiRemoteStation *station = static_cast<ParfWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_nAttempt++;
  station->m_nFail++;
  station->m_nRetry++;
  station->m_nSuccess = 0;

  NS_LOG_DEBUG ("station=" << station << " data fail retry=" << station->m_nRetry << ", timer=" << station->m_nAttempt
                           << ", rate=" << +station->m_rateIndex << ", power=" << +station->m_powerLevel);
  if (station->m_usingRecoveryRate)
    {
      NS_ASSERT (station->m_nRetry >= 1);
      if (station->m_nRetry == 1)
        {
          //need recovery fallback
          if (station->m_rateIndex != 0)
            {
              NS_LOG_DEBUG ("station=" << station << " dec rate");
              station->m_rateIndex--;
              station->m_usingRecoveryRate = false;
            }
        }
      station->m_nAttempt = 0;
    }
  else if (station->m_usingRecoveryPower)
    {
      NS_ASSERT (station->m_nRetry >= 1);
      if (station->m_nRetry == 1)
        {
          //need recovery fallback
          if (station->m_powerLevel < m_maxPower)
            {
              NS_LOG_DEBUG ("station=" << station << " inc power");
              station->m_powerLevel++;
              station->m_usingRecoveryPower = false;
            }
        }
      station->m_nAttempt = 0;
    }
  else
    {
      NS_ASSERT (station->m_nRetry >= 1);
      if (((station->m_nRetry - 1) % 2) == 1)
        {
          //need normal fallback
          if (station->m_powerLevel == m_maxPower)
            {
              if (station->m_rateIndex != 0)
                {
                  NS_LOG_DEBUG ("station=" << station << " dec rate");
                  station->m_rateIndex--;
                }
            }
          else
            {
              NS_LOG_DEBUG ("station=" << station << " inc power");
              station->m_powerLevel++;
            }
        }
      if (station->m_nRetry >= 2)
        {
          station->m_nAttempt = 0;
        }
    }
}

void
ParfWifiManager::DoReportRxOk (WifiRemoteStation *station,
                               double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void ParfWifiManager::DoReportRtsOk (WifiRemoteStation *station,
                                     double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << station << ctsSnr << ctsMode << rtsSnr);
}

void ParfWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                      double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  ParfWifiRemoteStation *station = static_cast<ParfWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_nAttempt++;
  station->m_nSuccess++;
  station->m_nFail = 0;
  station->m_usingRecoveryRate = false;
  station->m_usingRecoveryPower = false;
  station->m_nRetry = 0;
  NS_LOG_DEBUG ("station=" << station << " data ok success=" << station->m_nSuccess << ", timer=" << station->m_nAttempt << ", rate=" << +station->m_rateIndex << ", power=" << +station->m_powerLevel);
  if ((station->m_nSuccess == m_successThreshold
       || station->m_nAttempt == m_attemptThreshold)
      && (station->m_rateIndex < (station->m_state->m_operationalRateSet.size () - 1)))
    {
      NS_LOG_DEBUG ("station=" << station << " inc rate");
      station->m_rateIndex++;
      station->m_nAttempt = 0;
      station->m_nSuccess = 0;
      station->m_usingRecoveryRate = true;
    }
  else if (station->m_nSuccess == m_successThreshold || station->m_nAttempt == m_attemptThreshold)
    {
      //we are at the maximum rate, we decrease power
      if (station->m_powerLevel != m_minPower)
        {
          NS_LOG_DEBUG ("station=" << station << " dec power");
          station->m_powerLevel--;
        }
      station->m_nAttempt = 0;
      station->m_nSuccess = 0;
      station->m_usingRecoveryPower = true;
    }
}

void
ParfWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ParfWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
ParfWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  ParfWifiRemoteStation *station = static_cast<ParfWifiRemoteStation*> (st);
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
  if (station->m_prevPowerLevel != station->m_powerLevel)
    {
      m_powerChange (prevPower, power, station->m_state->m_address);
      station->m_prevPowerLevel = station->m_powerLevel;
    }
  if (station->m_prevRateIndex != station->m_rateIndex)
    {
      m_rateChange (prevRate, rate, station->m_state->m_address);
      station->m_prevRateIndex = station->m_rateIndex;
    }
  return WifiTxVector (mode, station->m_powerLevel, GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}

WifiTxVector
ParfWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  /// \todo we could/should implement the ARF algorithm for
  /// RTS only by picking a single rate within the BasicRateSet.
  ParfWifiRemoteStation *station = static_cast<ParfWifiRemoteStation*> (st);
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

} //namespace ns3
