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
#include "aarfcd-wifi-manager.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a,b) ((a < b) ? a : b)
#define Max(a,b) ((a > b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AarfcdWifiManager");

/**
 * \brief hold per-remote-station state for AARF-CD Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the AARF-CD Wifi manager
 */
struct AarfcdWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_timer; ///< timer
  uint32_t m_success; ///< success
  uint32_t m_failed; ///< failed
  bool m_recovery; ///< recovery
  bool m_justModifyRate; ///< just modify rate
  uint32_t m_successThreshold; ///< success threshold
  uint32_t m_timerTimeout; ///< timer timeout
  uint8_t m_rate; ///< rate
  bool m_rtsOn; ///< RTS on
  uint32_t m_rtsWnd; ///< RTS window
  uint32_t m_rtsCounter; ///< RTS counter
  bool m_haveASuccess; ///< have a success
};

NS_OBJECT_ENSURE_REGISTERED (AarfcdWifiManager);

TypeId
AarfcdWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AarfcdWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AarfcdWifiManager> ()
    .AddAttribute ("SuccessK", "Multiplication factor for the success threshold in the AARF algorithm.",
                   DoubleValue (2.0),
                   MakeDoubleAccessor (&AarfcdWifiManager::m_successK),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("TimerK",
                   "Multiplication factor for the timer threshold in the AARF algorithm.",
                   DoubleValue (2.0),
                   MakeDoubleAccessor (&AarfcdWifiManager::m_timerK),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("MaxSuccessThreshold",
                   "Maximum value of the success threshold in the AARF algorithm.",
                   UintegerValue (60),
                   MakeUintegerAccessor (&AarfcdWifiManager::m_maxSuccessThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinTimerThreshold",
                   "The minimum value for the 'timer' threshold in the AARF algorithm.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&AarfcdWifiManager::m_minTimerThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinSuccessThreshold",
                   "The minimum value for the success threshold in the AARF algorithm.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&AarfcdWifiManager::m_minSuccessThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinRtsWnd",
                   "Minimum value for RTS window of AARF-CD",
                   UintegerValue (1),
                   MakeUintegerAccessor (&AarfcdWifiManager::m_minRtsWnd),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MaxRtsWnd",
                   "Maximum value for RTS window of AARF-CD",
                   UintegerValue (40),
                   MakeUintegerAccessor (&AarfcdWifiManager::m_maxRtsWnd),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("TurnOffRtsAfterRateDecrease",
                   "If true the RTS mechanism will be turned off when the rate will be decreased",
                   BooleanValue (true),
                   MakeBooleanAccessor (&AarfcdWifiManager::m_turnOffRtsAfterRateDecrease),
                   MakeBooleanChecker ())
    .AddAttribute ("TurnOnRtsAfterRateIncrease",
                   "If true the RTS mechanism will be turned on when the rate will be increased",
                   BooleanValue (true),
                   MakeBooleanAccessor (&AarfcdWifiManager::m_turnOnRtsAfterRateIncrease),
                   MakeBooleanChecker ())
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&AarfcdWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

AarfcdWifiManager::AarfcdWifiManager ()
  : WifiRemoteStationManager (),
    m_currentRate (0)
{
  NS_LOG_FUNCTION (this);
}

AarfcdWifiManager::~AarfcdWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
AarfcdWifiManager::DoInitialize ()
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
AarfcdWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  AarfcdWifiRemoteStation *station = new AarfcdWifiRemoteStation ();

  //AARF fields below
  station->m_successThreshold = m_minSuccessThreshold;
  station->m_timerTimeout = m_minTimerThreshold;
  station->m_rate = 0;
  station->m_success = 0;
  station->m_failed = 0;
  station->m_recovery = false;
  station->m_timer = 0;

  //AARF-CD specific fields below
  station->m_rtsOn = false;
  station->m_rtsWnd = m_minRtsWnd;
  station->m_rtsCounter = 0;
  station->m_justModifyRate = true;
  station->m_haveASuccess = false;

  return station;
}

void
AarfcdWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
AarfcdWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
  station->m_timer++;
  station->m_failed++;
  station->m_success = 0;

  if (!station->m_rtsOn)
    {
      TurnOnRts (station);
      if (!station->m_justModifyRate && !station->m_haveASuccess)
        {
          IncreaseRtsWnd (station);
        }
      else
        {
          ResetRtsWnd (station);
        }
      station->m_rtsCounter = station->m_rtsWnd;
      if (station->m_failed >= 2)
        {
          station->m_timer = 0;
        }
    }
  else if (station->m_recovery)
    {
      NS_ASSERT (station->m_failed >= 1);
      station->m_justModifyRate = false;
      station->m_rtsCounter = station->m_rtsWnd;
      if (station->m_failed == 1)
        {
          //need recovery fallback
          if (m_turnOffRtsAfterRateDecrease)
            {
              TurnOffRts (station);
            }
          station->m_justModifyRate = true;
          station->m_successThreshold = (int)(Min (station->m_successThreshold * m_successK,
                                                   m_maxSuccessThreshold));
          station->m_timerTimeout = (int)(Max (station->m_timerTimeout * m_timerK,
                                               m_minSuccessThreshold));
          if (station->m_rate != 0)
            {
              station->m_rate--;
            }
        }
      station->m_timer = 0;
    }
  else
    {
      NS_ASSERT (station->m_failed >= 1);
      station->m_justModifyRate = false;
      station->m_rtsCounter = station->m_rtsWnd;
      if (((station->m_failed - 1) % 2) == 1)
        {
          //need normal fallback
          if (m_turnOffRtsAfterRateDecrease)
            {
              TurnOffRts (station);
            }
          station->m_justModifyRate = true;
          station->m_timerTimeout = m_minTimerThreshold;
          station->m_successThreshold = m_minSuccessThreshold;
          if (station->m_rate != 0)
            {
              station->m_rate--;
            }
        }
      if (station->m_failed >= 2)
        {
          station->m_timer = 0;
        }
    }
  CheckRts (station);
}

void
AarfcdWifiManager::DoReportRxOk (WifiRemoteStation *station,
                                 double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
AarfcdWifiManager::DoReportRtsOk (WifiRemoteStation *st,
                                  double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
  NS_LOG_DEBUG ("station=" << station << " rts ok");
  station->m_rtsCounter--;
}

void
AarfcdWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                   double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
  station->m_timer++;
  station->m_success++;
  station->m_failed = 0;
  station->m_recovery = false;
  station->m_justModifyRate = false;
  station->m_haveASuccess = true;
  NS_LOG_DEBUG ("station=" << station << " data ok success=" << station->m_success << ", timer=" << station->m_timer);
  if ((station->m_success == station->m_successThreshold
       || station->m_timer == station->m_timerTimeout)
      && (station->m_rate < (GetNSupported (station) - 1)))
    {
      NS_LOG_DEBUG ("station=" << station << " inc rate");
      station->m_rate++;
      station->m_timer = 0;
      station->m_success = 0;
      station->m_recovery = true;
      station->m_justModifyRate = true;
      if (m_turnOnRtsAfterRateIncrease)
        {
          TurnOnRts (station);
          ResetRtsWnd (station);
          station->m_rtsCounter = station->m_rtsWnd;
        }
    }
  CheckRts (station);
}

void
AarfcdWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
AarfcdWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
AarfcdWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
  uint16_t channelWidth = GetChannelWidth (station);
  if (channelWidth > 20 && channelWidth != 22)
    {
      channelWidth = 20;
    }
  WifiMode mode = GetSupported (station, station->m_rate);
  uint64_t rate = mode.GetDataRate (channelWidth);
  if (m_currentRate != rate)
    {
      NS_LOG_DEBUG ("New datarate: " << rate);
      m_currentRate = rate;
    }
  return WifiTxVector (mode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()), 800, 1, 1, 0, channelWidth, GetAggregation (station));
}

WifiTxVector
AarfcdWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  /// \todo we could/should implement the AARF algorithm for
  /// RTS only by picking a single rate within the BasicRateSet.
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
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
AarfcdWifiManager::DoNeedRts (WifiRemoteStation *st,
                              uint32_t size, bool normally)
{
  NS_LOG_FUNCTION (this << st << size << normally);
  AarfcdWifiRemoteStation *station = static_cast<AarfcdWifiRemoteStation*> (st);
  NS_LOG_INFO ("" << station << " rate=" << station->m_rate << " rts=" << (station->m_rtsOn ? "RTS" : "BASIC") <<
               " rtsCounter=" << station->m_rtsCounter);
  return station->m_rtsOn;
}

void
AarfcdWifiManager::CheckRts (AarfcdWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (station->m_rtsCounter == 0 && station->m_rtsOn)
    {
      TurnOffRts (station);
    }
}

void
AarfcdWifiManager::TurnOffRts (AarfcdWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_rtsOn = false;
  station->m_haveASuccess = false;
}

void
AarfcdWifiManager::TurnOnRts (AarfcdWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_rtsOn = true;
}

void
AarfcdWifiManager::IncreaseRtsWnd (AarfcdWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  if (station->m_rtsWnd == m_maxRtsWnd)
    {
      return;
    }

  station->m_rtsWnd *= 2;
  if (station->m_rtsWnd > m_maxRtsWnd)
    {
      station->m_rtsWnd = m_maxRtsWnd;
    }
}

void
AarfcdWifiManager::ResetRtsWnd (AarfcdWifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
  station->m_rtsWnd = m_minRtsWnd;
}

} //namespace ns3
