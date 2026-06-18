/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef AIRTIME_METRIC_H
#define AIRTIME_METRIC_H

#include "ns3/mesh-wifi-interface-mac.h"
#include "ns3/wifi-mac-header.h"

namespace ns3 {
namespace dot11s {
/**
 * \ingroup dot11s
 *
 * \brief Airtime link metric calculator
 *
 * Airtime link metric is defined in 11B.10 of 802.11s Draft D3.0 as:
 *
 * airtime = (O + Bt/r)* (1 + average retry counter), where:
 * - o  -- the PHY dependent channel access which includes frame headers, training sequences,
 *       access protocol frames, etc.
 * - bt -- the test packet length in bits (8192 by default),
 * - r  -- the current bitrate of the packet,
 *
 * Final result is expressed in units of 0.01 Time Unit = 10.24 us (as required by 802.11s draft)
 */
class AirtimeLinkMetricCalculator : public Object
{
public:
  AirtimeLinkMetricCalculator ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  /**
   * Airtime link metric is defined in Section 13.9 of IEEE 802.11-2012 as:
   *
   * airtime = (O + Bt/r) /  (1 - frame error rate), where
   * o  -- the PHY dependent channel access which includes frame headers, training sequences,
   *       access protocol frames, etc.
   * bt -- the test packet length in bits (8192 by default),
   * r  -- the current bitrate of the packet,
   *
   * Final result is expressed encoded as an unsigned integer in units of 
   * 0.01 TU = 10.24 us
   *
   * \param peerAddress The peer MAC address
   * \param mac The MAC layer
   * \returns the calculated metric in units of 0.01 TU
   */
  uint32_t CalculateMetric (Mac48Address peerAddress, Ptr<MeshWifiInterfaceMac> mac);
private:
  /**
   * Set number of bytes in test frame (a constant 1024 in the standard)
   * \param testLength the test length in bits
   */
  void SetTestLength (uint16_t testLength);
  /**
   * Set header TID from the Dot11MetricTid attribute
   * \param tid the header TID to use to calculate data rate
   */
  void SetHeaderTid (uint8_t tid);

  Ptr<Packet> m_testFrame; ///< test frame
  WifiMacHeader m_testHeader; ///< test header
};
} // namespace dot11s
} // namespace ns3
#endif
