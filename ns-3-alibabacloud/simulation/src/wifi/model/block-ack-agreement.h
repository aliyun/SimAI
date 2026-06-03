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

#ifndef BLOCK_ACK_AGREEMENT_H
#define BLOCK_ACK_AGREEMENT_H

#include "ns3/mac48-address.h"
#include "ns3/event-id.h"
#include "block-ack-type.h"

namespace ns3 {
/**
 * \brief Maintains information for a block ack agreement.
 * \ingroup wifi
 */
class BlockAckAgreement
{
  friend class HtFrameExchangeManager;

public:
  /**
   * Constructor for BlockAckAgreement with given peer and TID.
   *
   * \param peer the peer station
   * \param tid the TID
   */
  BlockAckAgreement (Mac48Address peer, uint8_t tid);
  virtual ~BlockAckAgreement ();
  /**
   * Set buffer size.
   *
   * \param bufferSize the buffer size (in number of MPDUs)
   */
  void SetBufferSize (uint16_t bufferSize);
  /**
   * Set timeout.
   *
   * \param timeout the timeout value
   */
  void SetTimeout (uint16_t timeout);
  /**
   * Set starting sequence number.
   *
   * \param seq the starting sequence number
   */
  void SetStartingSequence (uint16_t seq);
  /**
   * Set starting sequence control.
   *
   * \param seq the starting sequence control
   */
  void SetStartingSequenceControl (uint16_t seq);
  /**
   * Set block ack policy to immediate Ack.
   */
  void SetImmediateBlockAck (void);
  /**
   * Set block ack policy to delayed Ack.
   */
  void SetDelayedBlockAck (void);
  /**
   * Enable or disable A-MSDU support.
   *
   * \param supported enable or disable A-MSDU support
   */
  void SetAmsduSupport (bool supported);
  /**
   * Return the Traffic ID (TID).
   *
   * \return TID
   */
  uint8_t GetTid (void) const;
  /**
   * Return the peer address.
   *
   * \return the peer MAC address
   */
  Mac48Address GetPeer (void) const;
  /**
   * Return the buffer size.
   *
   * \return the buffer size (in number of MPDUs)
   */
  uint16_t GetBufferSize (void) const;
  /**
   * Return the timeout.
   *
   * \return the timeout
   */
  uint16_t GetTimeout (void) const;
  /**
   * Return the starting sequence number.
   *
   * \return starting sequence number
   */
  virtual uint16_t GetStartingSequence (void) const;
  /**
   * Return the starting sequence control
   *
   * \return starting sequence control
   */
  uint16_t GetStartingSequenceControl (void) const;
  /**
   * Return the last sequence number covered by the ack window
   *
   * \return ending sequence number
   */
  uint16_t GetWinEnd (void) const;
  /**
   * Check whether the current ack policy is immediate BlockAck.
   *
   * \return true if the current ack policy is immediate BlockAck,
   *         false otherwise
   */
  bool IsImmediateBlockAck (void) const;
  /**
   * Check whether A-MSDU is supported
   *
   * \return true if A-MSDU is supported,
   *         false otherwise
   */
  bool IsAmsduSupported (void) const;
  /**
   * Enable or disable HT support.
   *
   * \param htSupported enable or disable HT support
   */
  void SetHtSupported (bool htSupported);
  /**
   * Check whether HT is supported
   *
   * \return true if HT is supported,
   *         false otherwise
   */
  bool IsHtSupported (void) const;
  /**
   * Get the type of the Block Acks sent by the recipient of this agreement.
   *
   * \return the type of the Block Acks sent by the recipient of this agreement
   */
  BlockAckType GetBlockAckType (void) const;
  /**
   * Get the type of the Block Ack Requests sent by the originator of this agreement.
   *
   * \return the type of the Block Ack Requests sent by the originator of this agreement
   */
  BlockAckReqType GetBlockAckReqType (void) const;
  /**
   * Get the distance between the given starting sequence number and the
   * given sequence number.
   *
   * \param seqNumber the given sequence number
   * \param startingSeqNumber the given starting sequence number
   * \return the distance of the given sequence number from the given starting sequence number
   */
  static std::size_t GetDistance (uint16_t seqNumber, uint16_t startingSeqNumber);


protected:
  Mac48Address m_peer;       //!< Peer address
  uint8_t m_amsduSupported;  //!< Flag whether MSDU aggregation is supported
  uint8_t m_blockAckPolicy;  //!< Type of block ack: immediate or delayed
  uint8_t m_tid;             //!< Traffic ID
  uint16_t m_bufferSize;     //!< Buffer size
  uint16_t m_timeout;        //!< Timeout
  uint16_t m_startingSeq;    //!< Starting sequence control
  uint16_t m_winEnd;         //!< Ending sequence number
  uint8_t m_htSupported;     //!< Flag whether HT is supported
  EventId m_inactivityEvent; //!< inactivity event
};

} //namespace ns3

#endif /* BLOCK_ACK_AGREEMENT_H */
