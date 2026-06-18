/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "wifi-phy-common.h"
#include "wifi-mode.h"
#include "wifi-net-device.h"
#include "ns3/ht-configuration.h"
#include "ns3/he-configuration.h"

namespace ns3 {

uint16_t
ConvertGuardIntervalToNanoSeconds (WifiMode mode, const Ptr<WifiNetDevice> device)
{
  uint16_t gi = 800;
  if (mode.GetModulationClass () == WIFI_MOD_CLASS_HE)
    {
      Ptr<HeConfiguration> heConfiguration = device->GetHeConfiguration ();
      NS_ASSERT (heConfiguration); //If HE modulation is used, we should have a HE configuration attached
      gi = static_cast<uint16_t> (heConfiguration->GetGuardInterval ().GetNanoSeconds ());
    }
  else if (mode.GetModulationClass () == WIFI_MOD_CLASS_HT || mode.GetModulationClass () == WIFI_MOD_CLASS_VHT)
    {
      Ptr<HtConfiguration> htConfiguration = device->GetHtConfiguration ();
      NS_ASSERT (htConfiguration); //If HT/VHT modulation is used, we should have a HT configuration attached
      gi = htConfiguration->GetShortGuardIntervalSupported () ? 400 : 800;
    }
  return gi;
}

uint16_t
ConvertGuardIntervalToNanoSeconds (WifiMode mode, bool htShortGuardInterval, Time heGuardInterval)
{
  uint16_t gi;
  if (mode.GetModulationClass () == WIFI_MOD_CLASS_HE)
    {
      gi = static_cast<uint16_t> (heGuardInterval.GetNanoSeconds ());
    }
  else if (mode.GetModulationClass () == WIFI_MOD_CLASS_HT || mode.GetModulationClass () == WIFI_MOD_CLASS_VHT)
    {
      gi = htShortGuardInterval ? 400 : 800;
    }
  else
    {
      gi = 800;
    }
  return gi;
}

uint16_t
GetChannelWidthForTransmission (WifiMode mode, uint16_t maxAllowedChannelWidth)
{
  WifiModulationClass modulationClass = mode.GetModulationClass ();
  if (maxAllowedChannelWidth > 20
      && (modulationClass == WifiModulationClass::WIFI_MOD_CLASS_OFDM // all non-HT OFDM control and management frames
          || modulationClass == WifiModulationClass::WIFI_MOD_CLASS_ERP_OFDM)) // special case of beacons at 2.4 GHz
    {
      return 20;
    }
  //at 2.4 GHz basic rate can be non-ERP DSSS
  if (modulationClass == WifiModulationClass::WIFI_MOD_CLASS_DSSS
      || modulationClass == WifiModulationClass::WIFI_MOD_CLASS_HR_DSSS)
    {
      return 22;
    }
  return maxAllowedChannelWidth;
}

uint16_t
GetChannelWidthForTransmission (WifiMode mode, uint16_t operatingChannelWidth,
                                uint16_t maxSupportedChannelWidth)
{
  return GetChannelWidthForTransmission (mode, (operatingChannelWidth < maxSupportedChannelWidth)
                                                ? operatingChannelWidth : maxSupportedChannelWidth);
}

WifiPreamble
GetPreambleForTransmission (WifiModulationClass modulation, bool useShortPreamble)
{
  if (modulation == WIFI_MOD_CLASS_HE)
    {
      return WIFI_PREAMBLE_HE_SU;
    }
  else if (modulation == WIFI_MOD_CLASS_VHT)
    {
      return WIFI_PREAMBLE_VHT_SU;
    }
  else if (modulation == WIFI_MOD_CLASS_HT)
    {
      return WIFI_PREAMBLE_HT_MF; // HT_GF has been removed
    }
  else if (modulation == WIFI_MOD_CLASS_HR_DSSS && useShortPreamble) //ERP_DSSS is modeled through HR_DSSS (since same preamble and modulation)
    {
      return WIFI_PREAMBLE_SHORT;
    }
  else
    {
      return WIFI_PREAMBLE_LONG;
    }
}

bool
IsAllowedControlAnswerModulationClass (WifiModulationClass modClassReq, WifiModulationClass modClassAnswer)
{
  switch (modClassReq)
    {
    case WIFI_MOD_CLASS_DSSS:
      return (modClassAnswer == WIFI_MOD_CLASS_DSSS);
    case WIFI_MOD_CLASS_HR_DSSS:
      return (modClassAnswer == WIFI_MOD_CLASS_DSSS || modClassAnswer == WIFI_MOD_CLASS_HR_DSSS);
    case WIFI_MOD_CLASS_ERP_OFDM:
      return (modClassAnswer == WIFI_MOD_CLASS_DSSS || modClassAnswer == WIFI_MOD_CLASS_HR_DSSS || modClassAnswer == WIFI_MOD_CLASS_ERP_OFDM);
    case WIFI_MOD_CLASS_OFDM:
      return (modClassAnswer == WIFI_MOD_CLASS_OFDM);
    case WIFI_MOD_CLASS_HT:
    case WIFI_MOD_CLASS_VHT:
    case WIFI_MOD_CLASS_HE:
      return true;
    default:
      NS_FATAL_ERROR ("Modulation class not defined");
      return false;
    }
}

Time
GetPpduMaxTime (WifiPreamble preamble)
{
  Time duration;

  switch (preamble)
    {
      case WIFI_PREAMBLE_HT_MF:
      case WIFI_PREAMBLE_VHT_SU:
      case WIFI_PREAMBLE_VHT_MU:
      case WIFI_PREAMBLE_HE_SU:
      case WIFI_PREAMBLE_HE_ER_SU:
      case WIFI_PREAMBLE_HE_MU:
      case WIFI_PREAMBLE_HE_TB:
        duration = MicroSeconds (5484);
        break;
      default:
        duration = MicroSeconds (0);
        break;
    }
  return duration;
}

bool
IsMu (WifiPreamble preamble)
{
  return (IsDlMu (preamble) || IsUlMu (preamble));
}

bool
IsDlMu (WifiPreamble preamble)
{
  return (preamble == WIFI_PREAMBLE_HE_MU);
}

bool
IsUlMu (WifiPreamble preamble)
{
  return (preamble == WIFI_PREAMBLE_HE_TB);
}

} //namespace ns3
