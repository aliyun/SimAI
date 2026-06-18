/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef MSDU_AGGREGATOR_H
#define MSDU_AGGREGATOR_H

#include "ns3/object.h"
#include "ns3/nstime.h"
#include "wifi-mode.h"
#include "qos-utils.h"
#include "wifi-mac-queue-item.h"
#include <map>

namespace ns3 {

class Packet;
class QosTxop;
class WifiTxVector;
class WifiMac;
class HtFrameExchangeManager;
class WifiTxParameters;

/**
 * \brief Aggregator used to construct A-MSDUs
 * \ingroup wifi
 */
class MsduAggregator : public Object
{
public:
  /// EDCA queues typedef
  typedef std::map<AcIndex, Ptr<QosTxop> > EdcaQueues;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  MsduAggregator ();
  virtual ~MsduAggregator ();

  /**
   * Compute the size of the A-MSDU resulting from the aggregation of an MSDU of
   * size <i>msduSize</i> and an A-MSDU of size <i>amsduSize</i>.
   * Note that only the basic A-MSDU subframe format (section 9.3.2.2.2 of IEEE
   * 802.11-2016) is supported.
   *
   * \param msduSize the MSDU size in bytes.
   * \param amsduSize the A-MSDU size in bytes.
   * \return the size of the resulting A-MSDU in bytes.
   */
  static uint16_t GetSizeIfAggregated (uint16_t msduSize, uint16_t amsduSize);

  /**
   * Attempt to aggregate other MSDUs to the given A-MSDU while meeting the
   * following constraints:
   *
   * - the A-MSDU size does not exceed the maximum A-MSDU size as determined for
   * the modulation class indicated by the given TxVector
   *
   * - the size of the A-MPDU resulting from the aggregation of the MPDU in which
   * the A-MSDU will be embedded and the current A-MPDU (as specified by the given
   * TX parameters) does not exceed the maximum A-MPDU size as determined for
   * the modulation class indicated by the given TxVector
   *
   * - the time to transmit the resulting PPDU, according to the given TxVector,
   * does not exceed the maximum PPDU duration allowed by the corresponding
   * modulation class (if any)
   *
   * - the time to transmit the resulting PPDU and to carry out protection and
   * acknowledgment, as specified by the given TX parameters, does not exceed the
   * given available time (if distinct from Time::Min ())
   *
   * If aggregation succeeds (it was possible to aggregate at least an MSDU to the
   * given MSDU), all the aggregated MSDUs are dequeued and an MPDU containing the
   * A-MSDU is enqueued in the queue (replacing the given MPDU) and returned.
   * Otherwise, no MSDU is dequeued from the EDCA queue and a null pointer is returned.
   *
   * \param peekedItem the MSDU which we attempt to aggregate other MSDUs to
   * \param txParams the TX parameters for the current frame
   * \param availableTime the time available for the frame exchange
   * \return the resulting A-MSDU, if aggregation is possible, a null pointer otherwise.
   */
  Ptr<WifiMacQueueItem> GetNextAmsdu (Ptr<const WifiMacQueueItem> peekedItem, WifiTxParameters& txParams,
                                      Time availableTime) const;

  /**
   * Determine the maximum size for an A-MSDU of the given TID that can be sent
   * to the given receiver when using the given modulation class.
   *
   * \param recipient the receiver station address.
   * \param tid the TID.
   * \param modulation the modulation class.
   * \return the maximum A-MSDU size in bytes.
   */
  uint16_t GetMaxAmsduSize (Mac48Address recipient, uint8_t tid,
                            WifiModulationClass modulation) const;

  /**
   *
   * \param aggregatedPacket the aggregated packet.
   * \returns DeaggregatedMsdus.
   */
  static WifiMacQueueItem::DeaggregatedMsdus Deaggregate (Ptr<Packet> aggregatedPacket);

  /**
   * Set the MAC layer to use.
   *
   * \param mac the MAC layer to use
   */
  void SetWifiMac (const Ptr<WifiMac> mac);

  /**
   * Calculate how much padding must be added to the end of an A-MSDU of the
   * given size if a new MSDU is added.
   * Each A-MSDU subframe is padded so that its length is multiple of 4 octets.
   *
   * \param amsduSize the size of the A-MSDU
   *
   * \return the number of octets required for padding
   */
  static uint8_t CalculatePadding (uint16_t amsduSize);

protected:
  void DoDispose () override;

private:
  Ptr<WifiMac> m_mac;                   //!< the MAC of this station
  Ptr<HtFrameExchangeManager> m_htFem;  //!< the HT Frame Exchange Manager of this station
};

} //namespace ns3

#endif /* MSDU_AGGREGATOR_H */
