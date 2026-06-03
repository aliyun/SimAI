/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005, 2009 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef MAC_TX_MIDDLE_H
#define MAC_TX_MIDDLE_H

#include <map>
#include "ns3/simple-ref-count.h"

namespace ns3 {

class WifiMacHeader;
class Mac48Address;

/**
 * \ingroup wifi
 *
 * Handles sequence numbering of IEEE 802.11 data frames
 */
class MacTxMiddle : public SimpleRefCount<MacTxMiddle>
{
public:
  MacTxMiddle ();
  ~MacTxMiddle ();

  /**
   * Return the next sequence number for the given header.
   *
   * \param hdr Wi-Fi header
   * \return the next sequence number
   */
  uint16_t GetNextSequenceNumberFor (const WifiMacHeader *hdr);
  /**
   * Return the next sequence number for the Traffic ID and destination, but do not pick it (i.e. the current sequence number remains unchanged).
   * This functions is used for A-MPDU aggregation.
   *
   * \param hdr Wi-Fi header
   * \return the next sequence number
   */
  uint16_t PeekNextSequenceNumberFor (const WifiMacHeader *hdr);
  /**
   * Return the next sequence number for the Traffic ID and destination.
   *
   * \param tid Traffic ID
   * \param addr destination address
   * \return the next sequence number
   */
  uint16_t GetNextSeqNumberByTidAndAddress (uint8_t tid, Mac48Address addr) const;
  /**
   * Set the sequence number of the given MAC header as the next sequence
   * number for the Traffic ID and destination of the given MAC header.
   *
   * \param hdr the given MAC header
   */
  void SetSequenceNumberFor (const WifiMacHeader *hdr);

private:
  std::map <Mac48Address,uint16_t*> m_qosSequences; ///< QOS sequences
  uint16_t m_sequence; ///< current sequence number
};

} //namespace ns3

#endif /* MAC_TX_MIDDLE_H */
