/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020
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

#ifndef WIFI_PHY_BAND_H
#define WIFI_PHY_BAND_H

#include <iostream>

namespace ns3 {

/**
 * \ingroup wifi
 * Identifies the PHY band.
 */
enum WifiPhyBand
{
  /** The 2.4 GHz band */
  WIFI_PHY_BAND_2_4GHZ,
  /** The 5 GHz band */
  WIFI_PHY_BAND_5GHZ,
  /** The 6 GHz band */
  WIFI_PHY_BAND_6GHZ,
  /** Unspecified */
  WIFI_PHY_BAND_UNSPECIFIED
};

/**
* \brief Stream insertion operator.
*
* \param os the stream
* \param band the band
* \returns a reference to the stream
*/
inline std::ostream& operator<< (std::ostream& os, WifiPhyBand band)
{
  switch (band)
    {
    case WIFI_PHY_BAND_2_4GHZ:
      return (os << "2.4GHz");
    case WIFI_PHY_BAND_5GHZ:
      return (os << "5GHz");
    case WIFI_PHY_BAND_6GHZ:
      return (os << "6GHz");
    default:
      return (os << "INVALID");
    }
}

} //namespace ns3

#endif /* WIFI_PHY_BAND_H */
