/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2004,2005 INRIA
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

#include "ns3/string.h"
#include "ns3/log.h"
#include "constant-rate-wifi-manager.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-tx-vector.h"
#include "ns3/wifi-utils.h"

#define Min(a,b) ((a < b) ? a : b)

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ConstantRateWifiManager");

NS_OBJECT_ENSURE_REGISTERED (ConstantRateWifiManager);

TypeId
ConstantRateWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ConstantRateWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ConstantRateWifiManager> ()
    .AddAttribute ("DataMode", "The transmission mode to use for every data packet transmission",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&ConstantRateWifiManager::m_dataMode),
                   MakeWifiModeChecker ())
    .AddAttribute ("ControlMode", "The transmission mode to use for every RTS packet transmission.",
                   StringValue ("OfdmRate6Mbps"),
                   MakeWifiModeAccessor (&ConstantRateWifiManager::m_ctlMode),
                   MakeWifiModeChecker ())
  ;
  return tid;
}

ConstantRateWifiManager::ConstantRateWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

ConstantRateWifiManager::~ConstantRateWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiRemoteStation *
ConstantRateWifiManager::DoCreateStation (void) const
{
  NS_LOG_FUNCTION (this);
  WifiRemoteStation *station = new WifiRemoteStation ();
  return station;
}

void
ConstantRateWifiManager::DoReportRxOk (WifiRemoteStation *station,
                                       double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
ConstantRateWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ConstantRateWifiManager::DoReportDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ConstantRateWifiManager::DoReportRtsOk (WifiRemoteStation *st,
                                        double ctsSnr, WifiMode ctsMode, double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode << rtsSnr);
}

void
ConstantRateWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                         double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode << dataSnr << dataChannelWidth << +dataNss);
}

void
ConstantRateWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ConstantRateWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

WifiTxVector
ConstantRateWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  uint8_t nss = Min (GetMaxNumberOfTransmitStreams (), GetNumberOfSupportedStreams (st));
  if (m_dataMode.GetModulationClass () == WIFI_MOD_CLASS_HT)
    {
      nss = 1 + (m_dataMode.GetMcsValue () / 8);
    }
  return WifiTxVector (m_dataMode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (m_dataMode.GetModulationClass (), GetShortPreambleEnabled ()), ConvertGuardIntervalToNanoSeconds (m_dataMode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))), GetNumberOfAntennas (), nss, 0, GetChannelWidthForTransmission (m_dataMode, GetPhy ()->GetChannelWidth (), GetChannelWidth (st)), GetAggregation (st));
}

WifiTxVector
ConstantRateWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  return WifiTxVector (m_ctlMode, GetDefaultTxPowerLevel (), GetPreambleForTransmission (m_ctlMode.GetModulationClass (), GetShortPreambleEnabled ()), ConvertGuardIntervalToNanoSeconds (m_ctlMode, GetShortGuardIntervalSupported (st), NanoSeconds (GetGuardInterval (st))), 1, 1, 0, GetChannelWidthForTransmission (m_ctlMode, GetPhy ()->GetChannelWidth (), GetChannelWidth (st)), GetAggregation (st));
}

} //namespace ns3
