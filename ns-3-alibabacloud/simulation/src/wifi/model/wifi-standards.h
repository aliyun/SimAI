/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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

#ifndef WIFI_STANDARD_H
#define WIFI_STANDARD_H

#include <map>
#include <list>
#include "wifi-phy-band.h"
#include "ns3/abort.h"

namespace ns3 {

/**
 * \ingroup wifi
 * Identifies the IEEE 802.11 specifications that a Wifi device can be configured to use.
 */
enum WifiStandard
{
  WIFI_STANDARD_UNSPECIFIED,
  WIFI_STANDARD_80211a,
  WIFI_STANDARD_80211b,
  WIFI_STANDARD_80211g,
  WIFI_STANDARD_80211p,
  WIFI_STANDARD_80211n,
  WIFI_STANDARD_80211ac,
  WIFI_STANDARD_80211ax
};

/**
* \brief Stream insertion operator.
*
* \param os the stream
* \param standard the standard
* \returns a reference to the stream
*/
inline std::ostream& operator<< (std::ostream& os, WifiStandard standard)
{
  switch (standard)
    {
    case WIFI_STANDARD_80211a:
      return (os << "802.11a");
    case WIFI_STANDARD_80211b:
      return (os << "802.11b");
    case WIFI_STANDARD_80211g:
      return (os << "802.11g");
    case WIFI_STANDARD_80211p:
      return (os << "802.11p");
    case WIFI_STANDARD_80211n:
      return (os << "802.11n");
    case WIFI_STANDARD_80211ac:
      return (os << "802.11ac");
    case WIFI_STANDARD_80211ax:
      return (os << "802.11ax");
    default:
      return (os << "UNSPECIFIED");
    }
}


/**
 * \brief map a given standard configured by the user to the allowed PHY bands
 */
const std::map<WifiStandard, std::list<WifiPhyBand>> wifiStandards =
{
  { WIFI_STANDARD_80211a, { WIFI_PHY_BAND_5GHZ } },
  { WIFI_STANDARD_80211b, { WIFI_PHY_BAND_2_4GHZ } },
  { WIFI_STANDARD_80211g, { WIFI_PHY_BAND_2_4GHZ } },
  { WIFI_STANDARD_80211p, { WIFI_PHY_BAND_5GHZ } },
  { WIFI_STANDARD_80211n, { WIFI_PHY_BAND_2_4GHZ, WIFI_PHY_BAND_5GHZ } },
  { WIFI_STANDARD_80211ac, { WIFI_PHY_BAND_5GHZ } },
  { WIFI_STANDARD_80211ax, { WIFI_PHY_BAND_2_4GHZ, WIFI_PHY_BAND_5GHZ, WIFI_PHY_BAND_6GHZ } }
};

/**
 * \ingroup wifi
 * \brief Enumeration of frequency channel types
 */
enum FrequencyChannelType : uint8_t
{
  WIFI_PHY_DSSS_CHANNEL = 0,
  WIFI_PHY_OFDM_CHANNEL,
  WIFI_PHY_80211p_CHANNEL
};

/**
 * Get the type of the frequency channel for the given standard
 *
 * \param standard the standard
 * \return the type of the frequency channel for the given standard
 */
inline FrequencyChannelType GetFrequencyChannelType (WifiStandard standard)
{
  switch (standard)
    {
      case WIFI_STANDARD_80211b:
        return WIFI_PHY_DSSS_CHANNEL;
      case WIFI_STANDARD_80211p:
        return WIFI_PHY_80211p_CHANNEL;
      default:
        return WIFI_PHY_OFDM_CHANNEL;
    }
}

/**
 * Get the maximum channel width in MHz allowed for the given standard.
 *
 * \param standard the standard
 * \return the maximum channel width in MHz allowed for the given standard
 */
inline uint16_t GetMaximumChannelWidth (WifiStandard standard)
{
  switch (standard)
    {
      case WIFI_STANDARD_80211b:
        return 22;
      case WIFI_STANDARD_80211p:
        return 10;
      case WIFI_STANDARD_80211a:
      case WIFI_STANDARD_80211g:
        return 20;
      case WIFI_STANDARD_80211n:
        return 40;
      case WIFI_STANDARD_80211ac:
      case WIFI_STANDARD_80211ax:
        return 160;
      default:
        NS_ABORT_MSG ("Unknown standard: " << standard);
        return 0;
    }
}

/**
 * Get the default channel width for the given PHY standard and band.
 *
 * \param standard the given standard
 * \param band the given PHY band
 * \return the default channel width (MHz) for the given standard
 */
inline uint16_t GetDefaultChannelWidth (WifiStandard standard, WifiPhyBand band)
{
  switch (standard)
    {
    case WIFI_STANDARD_80211b:
      return 22;
    case WIFI_STANDARD_80211p:
      return 10;
    case WIFI_STANDARD_80211ac:
      return 80;
    case WIFI_STANDARD_80211ax:
      return (band == WIFI_PHY_BAND_2_4GHZ ? 20 : 80);
    default:
      return 20;
    }
}

/**
 * Get the default PHY band for the given standard.
 *
 * \param standard the given standard
 * \return the default PHY band for the given standard
 */
inline WifiPhyBand GetDefaultPhyBand (WifiStandard standard)
{
  switch (standard)
    {
    case WIFI_STANDARD_80211p:
    case WIFI_STANDARD_80211a:
    case WIFI_STANDARD_80211ac:
    case WIFI_STANDARD_80211ax:
      return WIFI_PHY_BAND_5GHZ;
    default:
      return WIFI_PHY_BAND_2_4GHZ;
    }
}

} //namespace ns3

#endif /* WIFI_STANDARD_H */
