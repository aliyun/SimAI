/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 North Carolina State University
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
 * Author: Scott E. Carpenter <scarpen@ncsu.edu>
 *
 */

#ifndef WAVE_BSM_STATS_H
#define WAVE_BSM_STATS_H

#include "ns3/object.h"
#include <vector>

namespace ns3 {
/**
 * \ingroup wave
 * \brief The WaveBsmStats class implements a stats collector for
 * IEEE 1609 WAVE (Wireless Access in Vehicular Environments)
 * Basic Safety Messages (BSMs).  The BSM is a ~200-byte packet that is
 * generally broadcast from every vehicle at a nominal rate of 10 Hz.
 *
 * \internal
 * Note:  This class collects data elements and accessors
 * along with methods that calculate metrics from the data
 * elements.  The data and metrics calculation algorithms
 * are collected together here purely to keep them together.
 * Future work may need to add additional metric calculations,
 * and for now, we are trying to keep all related data and
 * algorithms together, although these could easily be
 * refactored in the future and moved to separate classes.
 * However, it seems that for now, moving the data elements
 * or the algorithms separately into different classes could
 * lead to confusion over usage.
 */
class WaveBsmStats : public Object
{
public:
  /**
   * \brief Constructor
   */
  WaveBsmStats ();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * \brief Increments the count of transmitted packets
   */
  void IncTxPktCount ();

  /**
   * \brief Returns the count of transmitted packets
   * \return count of packets transmitted
   */
  int GetTxPktCount ();

  /*
   * Note:
   * The WAVE Basic Safety Message (BSM) is broadcast and
   * unacknowledged. In order to calculate packet delivery
   * ratio (PDR), we must count i) the packets that are
   * actually received and ii) the transmitted packets that
   * are expected to be received.  Both are relative to a
   * specified (circular) coverage area.
   *
   * For example:  Say we have three nodes, A, B, and C, each
   * separated by 40m, as follows:
   *
   *    A --<40m>-- B --<40m>-- C
   *
   * Let's assume that the transmission range is 50m, and only
   * A is transmitting (i.e. broadcasting).  B can receive A's
   * broadcasts, while C cannot.  Let's assume no dropped packets.
   * If we set the coverage area to  100m, then the PDR is 50%,
   * because B receives every transmission from A, while C receives
   * none of them.  However, if we change the effective
   * coverage area to 75m then the PDR improves to 100%, because
   * B receives 100% of A's transmissions, and C is outside of the
   * coverage area, and so does not factor in to the PDR.
   */

  /**
   * \brief Increments the count of (broadcast) packets expected
   * to be received within the coverage area1.  Broadcast packets
   * (i.e. WAVE Basic Safety Messages) are not ACK'd.  For packet
   * delivery ratio (PDR), we need to count transmitted packets that
   * are expected to be received within the coverage area, even
   * though they may not be physically received (due to collisions
   * or receiver power thresholds).
   * \param index index for statistics
   */
  void IncExpectedRxPktCount (int index);

  /**
   * \brief Increments the count of actual packets received
   * (regardless of coverage area).
   */
  void IncRxPktCount ();

  /**
   * \brief Increments the count of actual packets received within
   * the coverage area(index).  Broadcast packets
   * (i.e. WAVE Basic Safety Messages) are not ACK'd.  For packet
   * delivery ratio (PDR), we need to count only those received packets
   * that are actually received within the (circular) coverage area.
   * \param index index for statistics
   */
  void IncRxPktInRangeCount (int index);

  /**
   * \brief Returns the count of packets received
   * \return the count of packets received
   */
  int GetRxPktCount ();

  /**
   * \brief Returns the count of expected packets received within range(index)
   * \param index index for statistics
   * \return the count of expected packets received within range(index)
   */
  int GetExpectedRxPktCount (int index);

  /**
   * \brief Increments the count of actual packets received within range(index)
   * \param index index for statistics
   * \return the count of actual packets received within range(index)
   */
  int GetRxPktInRangeCount (int index);

  /**
   * \brief Sets the count of packets expected to received
   * \param index index for statistics
   * \param count the count of packets
   */
  void SetExpectedRxPktCount (int index, int count);

  /**
   * \brief Sets the count of packets within range that are received
   * \param index index for statistics
   * \param count the count of packets
   */
  void SetRxPktInRangeCount (int index, int count);

  /**
   * \brief Resets the count of total packets
   * expected and/or within range(index) that are received
   * \param index index for statistics
   */
  void ResetTotalRxPktCounts (int index);

  /**
   * \brief Sets the count of packets transmitted
   * \param count the count of packets transmitted
   */
  void SetTxPktCount (int count);

  /**
   * \brief Sets the count of packets received
   * \param count the count of packets received
   */
  void SetRxPktCount (int count);

  /**
   * \brief Increments the count of (application data) bytes transmitted
   * not including MAC/PHY overhead
   * \param bytes the bytes of application-data transmitted
   */
  void IncTxByteCount (int bytes);

  /**
   * \brief Returns the count of (application data) bytes transmitted
   * not include MAC/PHY overhead
   * \return number of bytes of application-data transmitted
   */
  int GetTxByteCount ();

  /**
   * \brief Returns the BSM Packet Delivery Ratio (PDR)
   * which is the percent of expected packets within range(index) that
   * are actually received
   * \param index index for statistics
   * \return the packet delivery ratio (PDR) of BSMs.
   */
  double GetBsmPdr (int index);

  /**
   * \brief Returns the cumulative BSM Packet Delivery Ratio (PDR)
   * which is the percent of cumulative expected packets within range(index)
   * that are actually received
   * \param index index for statistics
   * \return the packet delivery ratio (PDR) of BSMs.
   */
  double GetCumulativeBsmPdr (int index);

  /**
   * \brief Enables/disables logging
   * \param log
   */
  void SetLogging (int log);

  /**
   * \brief Gets logging state
   * \return logging state
   */
  int GetLogging ();

private:
  int m_wavePktSendCount; ///< packet sent count
  int m_waveByteSendCount; ///< byte sent count
  int m_wavePktReceiveCount; ///< packet receive count 
  std::vector <int> m_wavePktInCoverageReceiveCounts; ///< packet in ceoverage receive counts
  std::vector <int> m_wavePktExpectedReceiveCounts; ///< packet expected receive counts
  std::vector <int> m_waveTotalPktInCoverageReceiveCounts; ///< total packet in coverage receive counts
  std::vector <int> m_waveTotalPktExpectedReceiveCounts; ///< total packet expected receive counts
  int m_log; ///< log
};

} // namespace ns3

#endif /* WAVE_BSM_STATS_H*/
