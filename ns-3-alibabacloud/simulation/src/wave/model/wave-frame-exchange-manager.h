/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 *         Junling Bu <linlinjavaer@gmail.com>
 *         Stefano Avallone <stavallo@unina.it>
 */

#ifndef WAVE_FRAME_EXCHANGE_MANAGER_H
#define WAVE_FRAME_EXCHANGE_MANAGER_H

#include "ns3/qos-frame-exchange-manager.h"
#include "wave-net-device.h"

namespace ns3 {

/**
 * \ingroup wave
 *
 * This class is the subclass of QosFrameExchangeManager to provide support for MAC extension
 * (1) allows higher layer control data rate and tx power level.
 *     If higher layer does not set, they will be determined by WifiRemoteStationManager
 *          of MAC layer;
 *     If higher layer sets tx parameters in non-adaptable mode, the data rate
 *        and tx power level will be used for transmission;.
 *    If higher layer sets tx parameters in adaptable mode, the data rate
 *        will be lower bound for the actual data rate, and the power level
 *        will be upper bound for the actual transmit power.
 * (2) implements the feature described in Annex C : avoid transmission at scheduled guard intervals
 *      However, the feature is extended further here that the required transmit time is determined
 *      by MAC layer itself rather than PHY layer, which contains RTS/CTS, DATA and ACK time.
 */
class WaveFrameExchangeManager : public QosFrameExchangeManager
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  WaveFrameExchangeManager ();
  virtual ~WaveFrameExchangeManager ();

  // Overridden from FrameExchangeManager
  virtual bool StartTransmission (Ptr<Txop> dcf);

  /**
   * \param device WaveNetDevice associated with WaveFrameExchangeManager
   */
  void SetWaveNetDevice (Ptr<WaveNetDevice> device);

protected:
  // Overridden from FrameExchangeManager
  virtual void DoDispose (void);

private:
  /**
   * Return a TXVECTOR for the DATA frame given the destination.
   * The function consults WifiRemoteStationManager, which controls the rate
   * to different destinations.
   *
   * \param item the item being asked for TXVECTOR
   * \return TXVECTOR for the given item
   */
  virtual WifiTxVector GetDataTxVector (Ptr<const WifiMacQueueItem> item) const;

  Ptr<ChannelScheduler> m_scheduler;     ///< the channel scheduler
  Ptr<ChannelCoordinator> m_coordinator; ///< the channel coordinator
};

} //namespace ns3

#endif /* VHT_FRAME_EXCHANGE_MANAGER_H */
