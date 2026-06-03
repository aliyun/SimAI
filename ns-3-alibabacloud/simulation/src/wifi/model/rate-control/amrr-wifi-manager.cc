/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2003,2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "amrr-wifi-manager.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AmrrWifiManager");

/**
 * \brief hold per-remote-station state for AMRR Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the AMRR Wifi manager
 */
struct AmrrWifiRemoteStation : public WifiRemoteStation
{
  Time m_nextModeUpdate; ///< next mode update time
  uint32_t m_tx_ok;      ///< transmit OK
  uint32_t m_tx_err;     ///< transmit error
  uint32_t m_tx_retr;    ///< transmit retry
  uint32_t m_retry;      ///< retry
  uint8_t m_txrate;      ///< transmit rate
  uint32_t m_successThreshold; ///< success threshold
  uint32_t m_success;    ///< success
  bool m_recovery;       ///< recovery
};


NS_OBJECT_ENSURE_REGISTERED (AmrrWifiManager);

TypeId
AmrrWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AmrrWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AmrrWifiManager> ()
    .AddAttribute ("UpdatePeriod",
                   "The interval between decisions about rate control changes",
                   TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&AmrrWifiManager::m_updatePeriod),
                   MakeTimeChecker ())
    .AddAttribute ("FailureRatio",
                   "Ratio of minimum erroneous transmissions needed to switch to a lower rate",
                   DoubleValue (1.0 / 3.0),
                   MakeDoubleAccessor (&AmrrWifiManager::m_failureRatio),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("SuccessRatio",
                   "Ratio of maximum erroneous transmissions needed to switch to a higher rate",
                   DoubleValue (1.0 / 10.0),
                   MakeDoubleAccessor (&AmrrWifiManager::m_successRatio),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("MaxSuccessThreshold",
                   "Maximum number of consecutive success periods needed to switch to a higher rate",
                   UintegerValue (10),
                   MakeUintegerAccessor (&AmrrWifiManager::m_maxSuccessThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinSuccessThreshold",
                   "Minimum number of consecutive success periods needed to switch to a higher rate",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AmrrWifiManager::m_minSuccessThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&AmrrWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

AmrrWifiManager::AmrrWifiManager ()
  : WifiRemoteStationManager (),
    m_currentRate (0)
{
  NS_LOG_FUNCTION (this);
}

AmrrWifiManager::~AmrrWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
AmrrWifiManager::DoInitialize ()
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
AmrrWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  AmrrWifiRemoteStation *station = new AmrrWifiRemoteStation ();
  station->m_nextModeUpdate = Simulator::Now () + m_updatePeriod;
  station->m_tx_ok = 0;
  station->m_tx_err = 0;
  station->m_tx_retr = 0;
  station->m_retry = 0;
  station->m_txrate = 0;
  station->m_successThreshold = m_minSuccessThreshold;
  station->m_success = 0;
  station->m_recovery = false;
  return station;
}

void
AmrrWifiManager::DoReportRxOk (WifiRemoteStation *station,
                               double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
AmrrWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
AmrrWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AmrrWifiRemoteStation *station = static_cast<AmrrWifiRemoteStation*> (st);
  station->m_retry++;
  station->m_tx_retr++;
}

void
AmrrWifiManager::DoReportRtsOk (WifiRemoteStation *st,
                                double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
}

void
AmrrWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                 double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  AmrrWifiRemoteStation *station = static_cast<AmrrWifiRemoteStation*> (st);
  station->m_retry = 0;
  station->m_tx_ok++;
}

void
AmrrWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
AmrrWifiManager::DoReportFinalDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AmrrWifiRemoteStation *station = static_cast<AmrrWifiRemoteStation*> (st);
  station->m_retry = 0;
  station->m_tx_err++;
}

bool
AmrrWifiManager::IsMinRate (AmrrWifiRemoteStation *station) const
{
  NS_LOG_FUNCTION (this << station);
  return (station->m_txrate == 0);
}

bool
AmrrWifiManager::IsMaxRate (AmrrWifiRemoteStation *station) const
{
  NS_LOG_FUNCTION (this << station);
  NS_ASSERT (station->m_txrate + 1 <= GetNSupported (station));
  return (station->m_txrate + 1 == GetNSupported (station));
}

bool
AmrrWifiManager::IsSuccess (AmrrWifiRemoteStation *station) const
{
  NS_LOG_FUNCTION (this << station);
  return (station->m_tx_retr + station->m_tx_err) < station->m_tx_ok * m_successRatio;
}

bool
AmrrWifiManager::IsFailure (AmrrWifiRemoteStation *station) const
{
  NS_LOG_FUNCTION (this << station);
  return (station->m_tx_retr + station->m_tx_err) > station->m_tx_ok * m_failureRatio;
}

bool
AmrrWifiManager::IsEnough (AmrrWifiRemoteStation *station) const
{
  NS_LOG_FUNCTION (this << station);
  return (station->m_tx_retr + station->m_tx_err + station->m_tx_ok) > 10;
}

void
AmrrWifiManager::ResetCnt (AmrrWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_tx_ok = 0;
  station->m_tx_err = 0;
  station->m_tx_retr = 0;
}

void
AmrrWifiManager::IncreaseRate (AmrrWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_txrate++;
  NS_ASSERT (station->m_txrate < GetNSupported (station));
}

void
AmrrWifiManager::DecreaseRate (AmrrWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_txrate--;
}

void
AmrrWifiManager::UpdateMode (AmrrWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (Simulator::Now () < station->m_nextModeUpdate)
    {
      return;
    }
  station->m_nextModeUpdate = Simulator::Now () + m_updatePeriod;
  NS_LOG_DEBUG ("Update");

  bool needChange = false;

  if (IsSuccess (station) && IsEnough (station))
    {
      station->m_success++;
      NS_LOG_DEBUG ("++ success=" << station->m_success << " successThreshold=" << station->m_successThreshold <<
                    " tx_ok=" << station->m_tx_ok << " tx_err=" << station->m_tx_err << " tx_retr=" << station->m_tx_retr <<
                    " rate=" << +station->m_txrate << " n-supported-rates=" << +GetNSupported (station));
      if (station->m_success >= station->m_successThreshold
          && !IsMaxRate (station))
        {
          station->m_recovery = true;
          station->m_success = 0;
          IncreaseRate (station);
          needChange = true;
        }
      else
        {
          station->m_recovery = false;
        }
    }
  else if (IsFailure (station))
    {
      station->m_success = 0;
      NS_LOG_DEBUG ("-- success=" << station->m_success << " successThreshold=" << station->m_successThreshold <<
                    " tx_ok=" << station->m_tx_ok << " tx_err=" << station->m_tx_err << " tx_retr=" << station->m_tx_retr <<
                    " rate=" << +station->m_txrate << " n-supported-rates=" << +GetNSupported (station));
      if (!IsMinRate (station))
        {
          if (station->m_recovery)
            {
              station->m_successThreshold *= 2;
              station->m_successThreshold = std::min (station->m_successThreshold,
                                                      m_maxSuccessThreshold);
            }
          else
            {
              station->m_successThreshold = m_minSuccessThreshold;
            }
          station->m_recovery = false;
          DecreaseRate (station);
          needChange = true;
        }
      else
        {
          station->m_recovery = false;
        }
    }
  if (IsEnough (station) || needChange)
    {
      NS_LOG_DEBUG ("Reset");
      ResetCnt (station);
    }
}

WifiTxVector
AmrrWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AmrrWifiRemoteStation *station = static_cast<AmrrWifiRemoteStation*> (st);
  UpdateMode (station);
  NS_ASSERT (station->m_txrate < GetNSupported (station));
  uint8_t rateIndex;
  if (station->m_retry < 1)
    {
      rateIndex = station->m_txrate;
    }
  else if (station->m_retry < 2)
    {
      if (station->m_txrate > 0)
        {
          rateIndex = station->m_txrate - 1;
        }
      else
        {
          rateIndex = station->m_txrate;
        }
    }
  else if (station->m_retry < 3)
    {
      if (station->m_txrate > 1)
        {
          rateIndex = station->m_txrate - 2;
        }
      else
        {
          rateIndex = station->m_txrate;
        }
    }
  else
    {
      if (station->m_txrate > 2)
        {
          rateIndex = station->m_txrate - 3;
        }
      else
        {
          rateIndex = station->m_txrate;
        }
    }
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  WifiMode mode = GetSupported (station, rateIndex);
  uint64_t rate = mode.GetDataRate (channelWidth);
  if (m_currentRate != rate)
    {
      NS_LOG_DEBUG ("New datarate: " << rate);
      m_currentRate = rate;
    }
  return WifiTxVector (mode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}

WifiTxVector
AmrrWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AmrrWifiRemoteStation *station = static_cast<AmrrWifiRemoteStation*> (st);
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  UpdateMode (station);
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
