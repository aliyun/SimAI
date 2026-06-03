/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 IITP RAS
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 */

#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/core-module.h"
#include "ns3/packet.h"

#include "ns3/wifi-phy.h"

#include "thompson-sampling-wifi-manager.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace ns3 {

/**
 * A structure containing parameters of a single rate and its
 * statistics.
 */
struct RateStats {
  WifiMode mode; ///< MCS
  uint16_t channelWidth; ///< channel width in MHz
  uint8_t nss; ///< Number of spatial streams

  double success{0.0}; ///< averaged number of successful transmissions
  double fails{0.0}; ///< averaged number of failed transmissions
  Time lastDecay{0}; ///< last time exponential decay was applied to this rate
};

/**
 * Holds station state and collected statistics.
 *
 * This struct extends from WifiRemoteStation to hold additional
 * information required by ThompsonSamplingWifiManager.
 */
struct ThompsonSamplingWifiRemoteStation : public WifiRemoteStation
{
  size_t m_nextMode; //!< Mode to select for the next transmission
  size_t m_lastMode; //!< Most recently used mode, used to write statistics

  std::vector<RateStats> m_mcsStats; //!< Collected statistics
};

NS_OBJECT_ENSURE_REGISTERED (ThompsonSamplingWifiManager);

NS_LOG_COMPONENT_DEFINE ("ThompsonSamplingWifiManager");

TypeId
ThompsonSamplingWifiManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThompsonSamplingWifiManager")
    .SetParent<WifiRemoteStationManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<ThompsonSamplingWifiManager> ()
    .AddAttribute ("Decay",
                   "Exponential decay coefficient, Hz; zero is a valid value for static scenarios",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&ThompsonSamplingWifiManager::m_decay),
                   MakeDoubleChecker<double> (0.0))
    .AddTraceSource ("Rate",
                     "Traced value for rate changes (b/s)",
                     MakeTraceSourceAccessor (&ThompsonSamplingWifiManager::m_currentRate),
                     "ns3::TracedValueCallback::Uint64")
  ;
  return tid;
}

ThompsonSamplingWifiManager::ThompsonSamplingWifiManager ()
  : m_currentRate{0}
{
  NS_LOG_FUNCTION (this);

  m_gammaRandomVariable = CreateObject<GammaRandomVariable> ();
}

ThompsonSamplingWifiManager::~ThompsonSamplingWifiManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiRemoteStation *
ThompsonSamplingWifiManager::DoCreateStation () const
{
  NS_LOG_FUNCTION (this);
  ThompsonSamplingWifiRemoteStation *station = new ThompsonSamplingWifiRemoteStation ();
  station->m_nextMode = 0;
  station->m_lastMode = 0;
  return station;
}

void
ThompsonSamplingWifiManager::InitializeStation (WifiRemoteStation *st) const
{
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);
  if (!station->m_mcsStats.empty ())
    {
      return;
    }

  // Add HT, VHT or HE MCSes
  for (const auto &mode : GetPhy ()->GetMcsList ())
    {
      for (uint16_t j = 20; j <= GetPhy ()->GetChannelWidth (); j *= 2)
        {
          WifiModulationClass modulationClass = WIFI_MOD_CLASS_HT;
          if (GetVhtSupported ())
            {
              modulationClass = WIFI_MOD_CLASS_VHT;
            }
          if (GetHeSupported ())
            {
              modulationClass = WIFI_MOD_CLASS_HE;
            }
          if (mode.GetModulationClass () == modulationClass)
            {
              for (uint8_t k = 1; k <= GetPhy ()->GetMaxSupportedTxSpatialStreams (); k++)
                {
                  if (mode.IsAllowed (j, k))
                    {
                      RateStats stats;
                      stats.mode = mode;
                      stats.channelWidth = j;
                      stats.nss = k;

                      station->m_mcsStats.push_back (stats);
                    }
                }
            }
        }
    }

  if (station->m_mcsStats.empty ())
    {
      // Add legacy non-HT modes.
      for (uint8_t i = 0; i < GetNSupported (station); i++)
        {
          RateStats stats;
          stats.mode = GetSupported (station, i);
          if (stats.mode.GetModulationClass () == WIFI_MOD_CLASS_DSSS
              || stats.mode.GetModulationClass () == WIFI_MOD_CLASS_HR_DSSS)
            {
              stats.channelWidth = 22;
            }
          else
            {
              stats.channelWidth = 20;
            }
          stats.nss = 1;
          station->m_mcsStats.push_back (stats);
        }
    }

  NS_ASSERT_MSG (!station->m_mcsStats.empty (), "No usable MCS found");

  UpdateNextMode (st);
}

void
ThompsonSamplingWifiManager::DoReportRxOk (WifiRemoteStation *station, double rxSnr, WifiMode txMode)
{
  NS_LOG_FUNCTION (this << station << rxSnr << txMode);
}

void
ThompsonSamplingWifiManager::DoReportRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ThompsonSamplingWifiManager::DoReportDataFailed (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);
  Decay (st, station->m_lastMode);
  station->m_mcsStats.at (station->m_lastMode).fails++;
  UpdateNextMode (st);
}

void
ThompsonSamplingWifiManager::DoReportRtsOk (WifiRemoteStation *st, double ctsSnr, WifiMode ctsMode,
                                     double rtsSnr)
{
  NS_LOG_FUNCTION (this << st << ctsSnr << ctsMode.GetUniqueName () << rtsSnr);
}

void
ThompsonSamplingWifiManager::UpdateNextMode (WifiRemoteStation *st) const
{
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);

  double maxThroughput = 0.0;
  double frameSuccessRate = 1.0;

  NS_ASSERT (!station->m_mcsStats.empty ());

  // Use the most robust MCS if frameSuccessRate is 0 for all MCS.
  station->m_nextMode = 0;

  for (uint32_t i = 0; i < station->m_mcsStats.size (); i++)
    {
      Decay (st, i);
      const WifiMode mode{station->m_mcsStats.at (i).mode};

      uint16_t guardInterval = GetModeGuardInterval (st, mode);
      double rate = mode.GetDataRate (station->m_mcsStats.at (i).channelWidth,
                                      guardInterval,
                                      station->m_mcsStats.at (i).nss);

      // Thompson sampling
      frameSuccessRate = SampleBetaVariable (1.0 + station->m_mcsStats.at (i).success,
                                             1.0 + station->m_mcsStats.at (i).fails);
      NS_LOG_DEBUG ("Draw"
                    << " success=" << station->m_mcsStats.at (i).success
                    << " fails=" << station->m_mcsStats.at (i).fails
                    << " frameSuccessRate=" << frameSuccessRate
                    << " mode=" << mode);
      if (frameSuccessRate * rate > maxThroughput)
        {
          maxThroughput = frameSuccessRate * rate;
          station->m_nextMode = i;
        }
    }
}

void
ThompsonSamplingWifiManager::DoReportDataOk (WifiRemoteStation *st, double ackSnr, WifiMode ackMode,
                                      double dataSnr, uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << ackSnr << ackMode.GetUniqueName () << dataSnr);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);
  Decay (st, station->m_lastMode);
  station->m_mcsStats.at (station->m_lastMode).success++;
  UpdateNextMode (st);
}

void
ThompsonSamplingWifiManager::DoReportAmpduTxStatus (WifiRemoteStation *st, uint16_t nSuccessfulMpdus,
                                             uint16_t nFailedMpdus, double rxSnr, double dataSnr,
                                             uint16_t dataChannelWidth, uint8_t dataNss)
{
  NS_LOG_FUNCTION (this << st << nSuccessfulMpdus << nFailedMpdus << rxSnr << dataSnr);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);

  Decay (st, station->m_lastMode);
  station->m_mcsStats.at (station->m_lastMode).success += nSuccessfulMpdus;
  station->m_mcsStats.at (station->m_lastMode).fails += nFailedMpdus;

  UpdateNextMode (st);
}

void
ThompsonSamplingWifiManager::DoReportFinalRtsFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

void
ThompsonSamplingWifiManager::DoReportFinalDataFailed (WifiRemoteStation *station)
{
  NS_LOG_FUNCTION (this << station);
}

uint16_t
ThompsonSamplingWifiManager::GetModeGuardInterval (WifiRemoteStation *st, WifiMode mode) const
{
  if (mode.GetModulationClass () == WIFI_MOD_CLASS_HE)
    {
      return std::max (GetGuardInterval (st), GetGuardInterval ());
    }
  else if ((mode.GetModulationClass () == WIFI_MOD_CLASS_HT) ||
           (mode.GetModulationClass () == WIFI_MOD_CLASS_VHT))
    {
      return std::max<uint16_t> (GetShortGuardIntervalSupported (st) ? 400 : 800,
                                 GetShortGuardIntervalSupported () ? 400 : 800);
    }
  else
    {
      return 800;
    }
}

WifiTxVector
ThompsonSamplingWifiManager::DoGetDataTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);

  auto &stats = station->m_mcsStats.at (station->m_nextMode);
  WifiMode mode = stats.mode;
  uint16_t channelWidth = std::min (stats.channelWidth, GetPhy ()->GetChannelWidth ());
  uint8_t nss = stats.nss;
  uint16_t guardInterval = GetModeGuardInterval (st, mode);

  station->m_lastMode = station->m_nextMode;

  NS_LOG_DEBUG ("Using"
                << " mode=" << mode
                << " channelWidth=" << channelWidth
                << " nss=" << +nss
                << " guardInterval=" << guardInterval);

  uint64_t rate = mode.GetDataRate (channelWidth, guardInterval, nss);
  if (m_currentRate != rate)
    {
      NS_LOG_DEBUG ("New datarate: " << rate);
      m_currentRate = rate;
    }

  return WifiTxVector (
      mode,
      GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (mode.GetModulationClass (),
                                  GetShortPreambleEnabled ()),
      GetModeGuardInterval (st, mode),
      GetNumberOfAntennas (),
      nss,
      0, // NESS
      GetChannelWidthForTransmission (mode, channelWidth),
      GetAggregation (station),
      false);
}

WifiTxVector
ThompsonSamplingWifiManager::DoGetRtsTxVector (WifiRemoteStation *st)
{
  NS_LOG_FUNCTION (this << st);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);

  // Use the most robust MCS for the control channel.
  auto &stats = station->m_mcsStats.at (0);
  WifiMode mode = stats.mode;
  uint16_t channelWidth = std::min (stats.channelWidth, GetPhy ()->GetChannelWidth ());
  uint8_t nss = stats.nss;

  // Make sure control frames are sent using 1 spatial stream.
  NS_ASSERT (nss == 1);

  return WifiTxVector (
      mode, GetDefaultTxPowerLevel (),
      GetPreambleForTransmission (mode.GetModulationClass (), GetShortPreambleEnabled ()),
      GetModeGuardInterval (st, mode),
      GetNumberOfAntennas (),
      nss,
      0, // NESS
      GetChannelWidthForTransmission (mode, channelWidth),
      GetAggregation (station),
      false);
}

double
ThompsonSamplingWifiManager::SampleBetaVariable (uint64_t alpha, uint64_t beta) const
{
  double X = m_gammaRandomVariable->GetValue (alpha, 1.0);
  double Y = m_gammaRandomVariable->GetValue (beta, 1.0);
  return X / (X + Y);
}

void
ThompsonSamplingWifiManager::Decay (WifiRemoteStation *st, size_t i) const
{
  NS_LOG_FUNCTION (this << st << i);
  InitializeStation (st);
  auto station = static_cast<ThompsonSamplingWifiRemoteStation *> (st);

  Time now = Simulator::Now ();
  auto &stats = station->m_mcsStats.at (i);
  if (now > stats.lastDecay)
    {
      const double coefficient =
          std::exp (m_decay * (stats.lastDecay - now).GetSeconds ());

      stats.success *= coefficient;
      stats.fails *= coefficient;
      stats.lastDecay = now;
    }
}

int64_t
ThompsonSamplingWifiManager::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_gammaRandomVariable->SetStream (stream);
  return 1;
}

} //namespace ns3
