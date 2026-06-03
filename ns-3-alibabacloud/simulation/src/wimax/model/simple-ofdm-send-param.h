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

#ifndef SIMPLE_OFDM_SEND_PARAM_H
#define SIMPLE_OFDM_SEND_PARAM_H

#include <list>
#include "wimax-channel.h"
#include "bvec.h"
#include "wimax-phy.h"
#include "ns3/propagation-loss-model.h"

namespace ns3 {

/**
 * \ingroup wimax
 * simpleOfdmSendParam class
 */
class simpleOfdmSendParam
{
public:
  simpleOfdmSendParam (void);
  /**
   * Constructor
   *
   * \param fecBlock FEC block
   * \param burstSize burst size
   * \param isFirstBlock is the first block
   * \param Frequency frequency
   * \param modulationType modulation type
   * \param direction the direction
   * \param rxPowerDbm receive power
   */ 
  simpleOfdmSendParam (const bvec &fecBlock, uint32_t burstSize,
                       bool isFirstBlock, uint64_t Frequency,
                       WimaxPhy::ModulationType modulationType, uint8_t direction,
                       double rxPowerDbm);
  /**
   * Constructor
   *
   * \param burstSize burst size
   * \param isFirstBlock is the first block
   * \param Frequency frequency
   * \param modulationType modulation type
   * \param direction the direction
   * \param rxPowerDbm receive power
   * \param burst packet burst object
   */ 
  simpleOfdmSendParam (uint32_t burstSize,
                       bool isFirstBlock, uint64_t Frequency,
                       WimaxPhy::ModulationType modulationType, uint8_t direction,
                       double rxPowerDbm, Ptr<PacketBurst> burst);
  ~simpleOfdmSendParam (void);
  /**
   * \brief sent the fec block to send
   * \param fecBlock the fec block to send
   */
  void SetFecBlock (const bvec &fecBlock);
  /**
   * \brief set the burst size
   * \param burstSize the burst size in bytes
   */
  void SetBurstSize (uint32_t burstSize);
  /**
   * \param isFirstBlock Set to true if this fec block is the first one in the burst, set to false otherwise
   */
  void SetIsFirstBlock (bool isFirstBlock);
  /**
   * \param Frequency set the frequency of the channel in which this fec block will be sent
   */
  void SetFrequency (uint64_t Frequency);
  /**
   * \param modulationType the modulation type used to send this fec block
   */
  void SetModulationType (WimaxPhy::ModulationType modulationType);
  /**
   * \param direction the direction on which this fec block will be sent
   */
  void SetDirection (uint8_t direction);
  /**
   * \param rxPowerDbm the received power
   */
  void SetRxPowerDbm (double rxPowerDbm);
  /**
   * \return the fec block
   */
  bvec GetFecBlock (void);
  /**
   * \return the burst size
   */
  uint32_t GetBurstSize (void);
  /**
   * \return true if this fec block is the first one in the burst, false otherwise
   */
  bool GetIsFirstBlock (void);
  /**
   * \return the frequency on which the fec block is sent/received
   */
  uint64_t GetFrequency (void);
  /**
   * \return the modulation type used to send this fec block
   */
  WimaxPhy::ModulationType GetModulationType (void);
  /**
   * \return the direction on which this fec block was sent. UP or DOWN
   */
  uint8_t GetDirection (void);
  /**
   * \return the Received power
   */
  double GetRxPowerDbm (void);
  /**
   * \return the received burst
   */
  Ptr<PacketBurst> GetBurst (void);

private:
  bvec m_fecBlock; ///< FEC block
  uint32_t m_burstSize; ///< burst size
  bool m_isFirstBlock; ///< is first block
  uint64_t m_frequency; ///< frequency
  WimaxPhy::ModulationType m_modulationType; ///< modulation type
  uint8_t m_direction; ///< direction
  double m_rxPowerDbm; ///< receive power dbM
  Ptr<PacketBurst> m_burst; ///< burst

};
} // namespace ns3

#endif /* SIMPLE_OFDM_SEND_PARAM_H */
