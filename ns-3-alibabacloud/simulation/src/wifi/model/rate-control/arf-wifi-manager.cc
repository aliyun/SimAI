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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/log.h"
#include "arf-wifi-manager.h"
#include "ns3/wifi-tx-vector.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ArfWifiManager");

/**
 * \brief hold per-remote-station state for ARF Wifi manager.
 *
 * This struct extends from WifiRemoteStation struct to hold additional
 * information required by the ARF Wifi manager
 */
struct ArfWifiRemoteStation : public WifiRemoteStation
{
  uint32_t m_timer; ///< timer value
  uint32_t m_success; ///< success count
  uint32_t m_failed; ///< failed count
  bool m_recovery; ///< recovery
  uint32_t m_timerTimeout; ///< timer timeout
  uint32_t m_successThreshold; ///< success threshold
  uint8_t m_rate; ///< rate
};

NS_OBJECT_ENSURE_REGISTERED (ArfWifiManager);

TypeId
ArfWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ArfWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ArfWifiManager> ()
    .AddAttribute ("TimerThreshold", "The 'timer' threshold in the ARF algorithm.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&ArfWifiManager::m_timerThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("SuccessThreshold",
                   "The minimum number of successful transmissions to try a new rate.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&ArfWifiManager::m_successThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&ArfWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

ArfWifiManager::ArfWifiManager ()
  : WifiRemoteStationManager (),
    m_currentRate (0)
{
  NS_LOG_FUNCTION (this);
}

ArfWifiManager::~ArfWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

void
ArfWifiManager::DoInitialize ()
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
ArfWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  ArfWifiRemoteStation *station = new ArfWifiRemoteStation ();

  station->m_successThreshold = m_successThreshold;
  station->m_timerTimeout = m_timerThreshold;
  station->m_rate = 0;
  station->m_success = 0;
  station->m_failed = 0;
  station->m_recovery = false;
  station->m_timer = 0;

  return station;
}

void
ArfWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

/**
 * It is important to realize that "recovery" mode starts after failure of
 * the first transmission after a rate increase and ends at the first successful
 * transmission. Specifically, recovery mode transcends retransmissions boundaries.
 * Fundamentally, ARF handles each data transmission independently, whether it
 * is the initial transmission of a packet or the retransmission of a packet.
 * The fundamental reason for this is that there is a backoff between each data
 * transmission, be it an initial transmission or a retransmission.
 *
 * \param st the station that we failed to send Data
 */
void
ArfWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  ArfWifiRemoteStation *station = static_cast<ArfWifiRemoteStation*> (st);
  station->m_timer++;
  station->m_failed++;
  station->m_success = 0;

  if (station->m_recovery)
    {
      NS_ASSERT (station->m_failed >= 1);
      if (station->m_failed == 1)
        {
          //need recovery fallback
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
      if (((station->m_failed - 1) % 2) == 1)
        {
          //need normal fallback
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
}

void
ArfWifiManager::DoReportRxOk (WifiRemoteStation *station,
                              double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void ArfWifiManager::DoReportRtsOk (WifiRemoteStation *station,
                                    double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << station << ctsSnr << ctsMode << rtsSnr);
  NS_LOG_DEBUG ("station=" << station << " rts ok");
}

void ArfWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                     double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
  ArfWifiRemoteStation *station = static_cast<ArfWifiRemoteStation*> (st);
  station->m_timer++;
  station->m_success++;
  station->m_failed = 0;
  station->m_recovery = false;
  NS_LOG_DEBUG ("station=" << station << " data ok success=" << station->m_success << ", timer=" << station->m_timer);
  if ((station->m_success == m_successThreshold
       || station->m_timer == m_timerThreshold)
      && (station->m_rate < (station->m_state->m_operationalRateSet.size () - 1)))
    {
      NS_LOG_DEBUG ("station=" << station << " inc rate");
      station->m_rate++;
      station->m_timer = 0;
      station->m_success = 0;
      station->m_recovery = true;
    }
}

void
ArfWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ArfWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
ArfWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  ArfWifiRemoteStation *station = static_cast<ArfWifiRemoteStation*> (st);
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
ArfWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  /// \todo we could/should implement the ARF algorithm for
  /// RTS only by picking a single rate within the BasicRateSet.
  ArfWifiRemoteStation *station = static_cast<ArfWifiRemoteStation*> (st);
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
