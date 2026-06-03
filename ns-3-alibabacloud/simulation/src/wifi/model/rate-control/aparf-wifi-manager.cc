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
#include "aparf-wifi-manager.h"
#include "ns3/wifi-phy.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AparfWifiManager");

/**
 * Hold per-remote-station state for APARF Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the APARF Wifi manager
 */
struct
AparfWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_nSuccess;                  //!< Number of successful transmission attempts.
  uint32_t m_nFailed;                   //!< Number of failed transmission attempts.
  uint32_t m_pCount;                    //!< Number of power changes.
  uint32_t m_successThreshold;          //!< The minimum number of successful transmissions to try a new power or rate.
  uint32_t m_failThreshold;             //!< The minimum number of failed transmissions to try a new power or rate.
  uint8_t m_prevRateIndex;              //!< Rate index of the previous transmission.
  uint8_t m_rateIndex;                  //!< Current rate index.
  uint8_t m_critRateIndex;              //!< Critical rate.
  uint8_t m_prevPowerLevel;             //!< Power level of the previous transmission.
  uint8_t m_powerLevel;                 //!< Current power level.
  uint8_t m_nSupported;                 //!< Number of supported rates by the remote station.
  bool m_initialized;                   //!< For initializing variables.
  AparfWifiManager::State m_aparfState; //!< The estimated state of the channel.
};

NS_OBJECT_ENSURE_REGISTERED (AparfWifiManager);

TypeId
AparfWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AparfWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AparfWifiManager> ()
    .AddAttribute ("SuccessThreshold1",
                   "The minimum number of successful transmissions in \"High\" state to try a new power or rate.",
                   UintegerValue (3),
                   MakeUintegerAccessor (&AparfWifiManager::m_succesMax1),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("SuccessThreshold2",
                   "The minimum number of successful transmissions in \"Low\" state to try a new power or rate.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&AparfWifiManager::m_succesMax2),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("FailThreshold",
                   "The minimum number of failed transmissions to try a new power or rate.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AparfWifiManager::m_failMax),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("PowerThreshold",
                   "The maximum number of power changes.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&AparfWifiManager::m_powerMax),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("PowerDecrementStep",
                   "Step size for decrement the power.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AparfWifiManager::m_powerDec),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("PowerIncrementStep",
                   "Step size for increment the power.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AparfWifiManager::m_powerInc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RateDecrementStep",
                   "Step size for decrement the rate.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AparfWifiManager::m_rateDec),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RateIncrementStep",
                   "Step size for increment the rate.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AparfWifiManager::m_rateInc),
                   MakeUintegerChecker<uint8_t> ())
    .AddTraceSource ("PowerChange",
                     "The transmission power has change",
                     MakeTraceSourceAccessor (&AparfWifiManager::m_powerChange),
                     "ns3::WifiRemoteStationManager::PowerChangeTracedCallback")
    .AddTraceSource ("RateChange",
                     "The transmission rate has change",
                     MakeTraceSourceAccessor (&AparfWifiManager::m_rateChange),
                     "ns3::WifiRemoteStationManager::RateChangeTracedCallback")
  ;
  return tid;
}

AparfWifiManager::AparfWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

AparfWifiManager::~AparfWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
AparfWifiManager::SetupPhy (const Ptr<WifiPhy> phy)
{
  NS_LOG_FUNCTION (this << phy);
  m_minPower = 0;
  m_maxPower = phy->GetNTxPower () - 1;
  WifiRemoteStationManager::SetupPhy (phy);
}

void
AparfWifiManager::DoInitialize ()
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
AparfWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  AparfWifiRemoteStation *station = new AparfWifiRemoteStation ();

  station->m_successThreshold = m_succesMax1;
  station->m_failThreshold = m_failMax;
  station->m_nSuccess = 0;
  station->m_nFailed = 0;
  station->m_pCount = 0;
  station->m_aparfState = AparfWifiManager::High;
  station->m_initialized = false;

  NS_LOG_DEBUG ("create station=" << station << ", rate=" << +station->m_rateIndex
                                  << ", power=" << +station->m_powerLevel);

  return station;
}

void
AparfWifiManager::CheckInit (AparfWifiRemoteStation *station)
{
  if (!station->m_initialized)
    {
      station->m_nSupported = GetNSupported (station);
      station->m_rateIndex = station->m_nSupported - 1;
      station->m_prevRateIndex = station->m_nSupported - 1;
      station->m_powerLevel = m_maxPower;
      station->m_prevPowerLevel = m_maxPower;
      station->m_critRateIndex = 0;
      WifiMode mode = GetSupported (station, station->m_rateIndex);
      uint16_t channelWidth = GetChannelWidth (station);
      DataRate rate = DataRate (mode.GetDataRate (channelWidth));
      double power = GetPhy ()->GetPowerDbm (m_maxPower);
      m_powerChange (power, power, station->m_state->m_address);
      m_rateChange (rate, rate, station->m_state->m_address);
      station->m_initialized = true;
    }
}

void AparfWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void AparfWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AparfWifiRemoteStation *station = static_cast<AparfWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_nFailed++;
  station->m_nSuccess = 0;
  NS_LOG_DEBUG ("station=" << station << ", rate=" << station->m_rateIndex
                           << ", power=" << (int)station->m_powerLevel);

  if (station->m_aparfState == AparfWifiManager::Low)
    {
      station->m_aparfState = AparfWifiManager::High;
      station->m_successThreshold = m_succesMax1;
    }
  else if (station->m_aparfState == AparfWifiManager::Spread)
    {
      station->m_aparfState = AparfWifiManager::Low;
      station->m_successThreshold = m_succesMax2;
    }

  if (station->m_nFailed == station->m_failThreshold)
    {
      station->m_nFailed = 0;
      station->m_nSuccess = 0;
      station->m_pCount = 0;
      if (station->m_powerLevel == m_maxPower)
        {
          station->m_critRateIndex = station->m_rateIndex;
          if (station->m_rateIndex != 0)
            {
              NS_LOG_DEBUG ("station=" << station << " dec rate");
              station->m_rateIndex -= m_rateDec;
            }
        }
      else
        {
          NS_LOG_DEBUG ("station=" << station << " inc power");
          station->m_powerLevel += m_powerInc;
        }
    }
}

void
AparfWifiManager::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
AparfWifiManager::DoReportRtsOk (WifiRemoteStation *station, double ctsSnr,
                                 WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << station << ctsSnr << ctsMode << rtsSnr);
}

void
AparfWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                  double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  AparfWifiRemoteStation *station = static_cast<AparfWifiRemoteStation*> (st);
  CheckInit (station);
  station->m_nSuccess++;
  station->m_nFailed = 0;
  NS_LOG_DEBUG ("station=" << station << " data ok success=" << station->m_nSuccess << ", rate=" << +station->m_rateIndex << ", power=" << +station->m_powerLevel);

  if ((station->m_aparfState == AparfWifiManager::High) && (station->m_nSuccess >= station->m_successThreshold))
    {
      station->m_aparfState = AparfWifiManager::Spread;
    }
  else if ((station->m_aparfState == AparfWifiManager::Low) && (station->m_nSuccess >= station->m_successThreshold))
    {
      station->m_aparfState = AparfWifiManager::Spread;
    }
  else if (station->m_aparfState == AparfWifiManager::Spread)
    {
      station->m_aparfState = AparfWifiManager::High;
      station->m_successThreshold = m_succesMax1;
    }

  if (station->m_nSuccess == station->m_successThreshold)
    {
      station->m_nSuccess = 0;
      station->m_nFailed = 0;
      if (station->m_rateIndex == (station->m_state->m_operationalRateSet.size () - 1))
        {
          if (station->m_powerLevel != m_minPower)
            {
              NS_LOG_DEBUG ("station=" << station << " dec power");
              station->m_powerLevel -= m_powerDec;
            }
        }
      else
        {
          if (station->m_critRateIndex == 0)
            {
              if (station->m_rateIndex != (station->m_state->m_operationalRateSet.size () - 1))
                {
                  NS_LOG_DEBUG ("station=" << station << " inc rate");
                  station->m_rateIndex += m_rateInc;
                }
            }
          else
            {
              if (station->m_pCount == m_powerMax)
                {
                  station->m_powerLevel = m_maxPower;
                  station->m_rateIndex = station->m_critRateIndex;
                  station->m_pCount = 0;
                  station->m_critRateIndex = 0;
                }
              else
                {
                  if (station->m_powerLevel != m_minPower)
                    {
                      station->m_powerLevel -= m_powerDec;
                      station->m_pCount++;
                    }
                }
            }
        }
    }
}

void
AparfWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
AparfWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
AparfWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AparfWifiRemoteStation *station = static_cast<AparfWifiRemoteStation*> (st);
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
AparfWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  /// \todo we could/should implement the ARF algorithm for
  /// RTS only by picking a single rate within the BasicRateSet.
  AparfWifiRemoteStation *station = static_cast<AparfWifiRemoteStation*> (st);
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
