/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */

#ifndef SIMPLE_OFDM_WIMAX_CHANNEL_H
#define SIMPLE_OFDM_WIMAX_CHANNEL_H

#include <list>
#include "wimax-channel.h"
#include "bvec.h"
#include "wimax-phy.h"
#include "ns3/propagation-loss-model.h"
#include "simple-ofdm-send-param.h"

namespace ns3 {

class Packet;
class PacketBurst;
class SimpleOfdmWimaxPhy;

/**
 * \ingroup wimax
 * \brief SimpleOfdmWimaxChannel class 
 */
class SimpleOfdmWimaxChannel : public WimaxChannel
{
public:
  SimpleOfdmWimaxChannel (void);
  ~SimpleOfdmWimaxChannel (void);

  /// PropModel enumeration
  enum PropModel
  {
    RANDOM_PROPAGATION,
    FRIIS_PROPAGATION,
    LOG_DISTANCE_PROPAGATION,
    COST231_PROPAGATION
  };
  
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * \brief Creates a channel and sets the propagation model
   * \param propModel the propagation model to use
   */
  SimpleOfdmWimaxChannel (PropModel propModel);

  /**
   * \brief Sends a dummy fec block to all connected physical devices
   * \param BlockTime the time needed to send the block
   * \param burstSize the size of the burst
   * \param phy the sender device
   * \param isFirstBlock true if this block is the first one, false otherwise
   * \param isLastBlock true if this block is the last one, false otherwise
   * \param frequency the frequency on which the block is sent
   * \param modulationType the modulation used to send the fec block
   * \param direction uplink or downlink
   * \param txPowerDbm the transmission power
   * \param burst the packet burst to send
   */
  void Send (Time BlockTime,
             uint32_t burstSize, Ptr<WimaxPhy> phy, bool isFirstBlock,
             bool isLastBlock,
             uint64_t frequency, WimaxPhy::ModulationType modulationType,
             uint8_t direction, double txPowerDbm, Ptr<PacketBurst> burst);
  /**
   * \brief sets the propagation model
   * \param propModel the propagation model to used
   */
  void SetPropagationModel (PropModel propModel);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

private:
  /**
   * Attach functiion
   * \param phy the phy layer
   */
  void DoAttach (Ptr<WimaxPhy> phy);
  std::list<Ptr<SimpleOfdmWimaxPhy> > m_phyList; ///< phy list
  /**
   * Get number of devices function
   * \returns the number of devices
   */
  std::size_t DoGetNDevices (void) const;
  /**
   * End send dummy block function
   * \param rxphy the Ptr<SimpleOfdmWimaxPhy>
   * \param param the simpleOfdmSendParam *
   */
  void EndSendDummyBlock  (Ptr<SimpleOfdmWimaxPhy> rxphy, simpleOfdmSendParam * param);
  /**
   * Get device function
   * \param i the device index
   * \returns the device
   */
  Ptr<NetDevice> DoGetDevice (std::size_t i) const;
  Ptr<PropagationLossModel> m_loss; ///< loss
};

} // namespace ns3

#endif /* SIMPLE_OFDM_WIMAX_CHANNEL_H */
