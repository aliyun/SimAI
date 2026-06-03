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
 * Author: Stefano Avallone <stavallo@unina.it>
 */
#ifndef RECIPIENT_BLOCK_ACK_AGREEMENT_H
#define RECIPIENT_BLOCK_ACK_AGREEMENT_H

#include "block-ack-agreement.h"
#include "block-ack-window.h"
#include <map>


namespace ns3 {

class WifiMacQueueItem;
class MacRxMiddle;
class CtrlBAckResponseHeader;

/**
 * \ingroup wifi
 * Maintains the scoreboard and the receive reordering buffer used by a recipient
 * of a Block Ack agreement.
 */
class RecipientBlockAckAgreement : public BlockAckAgreement
{
public:
  /**
   * Constructor
   *
   * \param originator MAC address
   * \param amsduSupported whether A-MSDU support is enabled
   * \param tid Traffic ID
   * \param bufferSize the buffer size (in number of MPDUs)
   * \param timeout the timeout value
   * \param startingSeq the starting sequence number
   * \param htSupported whether HT support is enabled
   */
  RecipientBlockAckAgreement (Mac48Address originator, bool amsduSupported, uint8_t tid,
                              uint16_t bufferSize, uint16_t timeout, uint16_t startingSeq,
                              bool htSupported);
  ~RecipientBlockAckAgreement ();

  /**
   * Set the MAC RX Middle to use.
   *
   * \param rxMiddle the MAC RX Middle to use
   */
  void SetMacRxMiddle (const Ptr<MacRxMiddle> rxMiddle);

  /**
   * Update both the scoreboard and the receive reordering buffer upon reception
   * of the given MPDU.
   *
   * \param mpdu the received MPDU
   */
  void NotifyReceivedMpdu (Ptr<WifiMacQueueItem> mpdu);
  /**
   * Update both the scoreboard and the receive reordering buffer upon reception
   * of a Block Ack Request.
   *
   * \param startingSequenceNumber the starting sequence number included in the
   *                               received Block Ack Request
   */
  void NotifyReceivedBar (uint16_t startingSequenceNumber);
  /**
   * Set the Starting Sequence Number subfield of the Block Ack Starting Sequence
   * Control subfield of the Block Ack frame and fill the block ack bitmap.
   * For Multi-STA Block Acks, <i>index</i> identifies the Per AID TID Info
   * subfield whose bitmap has to be filled.
   *
   * \param blockAckHeader the block ack header
   * \param index the index of the Per AID TID Info subfield (Multi-STA Block Ack only)
   */
  void FillBlockAckBitmap (CtrlBAckResponseHeader *blockAckHeader, std::size_t index = 0) const;
  /**
   * This is called when a Block Ack agreement is destroyed to flush the
   * received packets.
   */
  void Flush (void);

private:
  /**
   * Pass MSDUs or A-MSDUs up to the next MAC process if they are stored in
   * the buffer in order of increasing value of the Sequence Number subfield
   * starting with the MSDU or A-MSDU that has SN=WinStartB.
   * Set WinStartB to the value of the Sequence Number subfield of the last
   * MSDU or A-MSDU that was passed up to the next MAC process plus one.
   */
  void PassBufferedMpdusUntilFirstLost (void);

  /**
   * Pass any complete MSDUs or A-MSDUs stored in the buffer with Sequence Number
   * subfield values that are lower than the new value of WinStartB up to the next
   * MAC process in order of increasing Sequence Number subfield value.
   *
   * \param newWinStartB the new value of WinStartB
   */
  void PassBufferedMpdusWithSeqNumberLessThan (uint16_t newWinStartB);

  /// The key of a buffered MPDU is the pair (MPDU sequence number, pointer to WinStartB)
  typedef std::pair<uint16_t, uint16_t*> Key;

  /// Comparison functor used to sort the buffered MPDUs
  struct Compare
  {
    /**
     * Functional operator for sorting the buffered MPDUs.
     *
     * \param a the key of first buffered MPDU (\see Key)
     * \param b the key of second buffered MPDU (\see Key)
     * \return true if first key is smaller than second
     */
    bool operator() (const Key& a, const Key& b) const;
  };

  BlockAckWindow m_scoreboard;                                   ///< recipient's scoreboard
  uint16_t m_winStartB;                                          ///< starting SN for the reordering buffer
  std::size_t m_winSizeB;                                        ///< size of the receive reordering buffer
  std::map<Key, Ptr<WifiMacQueueItem>, Compare> m_bufferedMpdus; ///< buffered MPDUs sorted by Seq Number
  Ptr<MacRxMiddle> m_rxMiddle;                                   ///< the MAC RX Middle on this station
};

} //namespace ns3

#endif /* RECIPIENT_BLOCK_ACK_AGREEMENT_H */
