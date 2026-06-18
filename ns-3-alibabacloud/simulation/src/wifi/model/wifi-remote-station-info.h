/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
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

#ifndef WIFI_REMOTE_STATION_INFO_H
#define WIFI_REMOTE_STATION_INFO_H

#include "ns3/nstime.h"
#include "ns3/uinteger.h"

namespace ns3 {

/**
 * \brief TID independent remote station statistics
 *
 * Structure is similar to struct sta_info in Linux kernel (see
 * net/mac80211/sta_info.h)
 */
class WifiRemoteStationInfo
{
public:
  WifiRemoteStationInfo ();
  virtual ~WifiRemoteStationInfo ();

  /**
   * \brief Updates average frame error rate when data or RTS was transmitted successfully.
   *
   * \param retryCounter is SLRC or SSRC value at the moment of success transmission.
   */
  void NotifyTxSuccess (uint32_t retryCounter);
  /**
   * Updates average frame error rate when final data or RTS has failed.
   */
  void NotifyTxFailed ();
  /**
   * Return frame error rate (probability that frame is corrupted due to transmission error).
   * \returns the frame error rate
   */
  double GetFrameErrorRate () const;
private:
  /**
   * \brief Calculate averaging coefficient for frame error rate. Depends on time of the last update.
   *
   * \attention Calling this method twice gives different results,
   * because it resets time of last update.
   *
   * \return average coefficient for frame error rate
   */
  double CalculateAveragingCoefficient ();

  Time m_memoryTime; ///< averaging coefficient depends on the memory time
  Time m_lastUpdate; ///< when last update has occurred
  double m_failAvg;  ///< moving percentage of failed frames
};

} //namespace ns3

#endif /* WIFI_REMOTE_STATION_INFO_H */
