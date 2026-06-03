/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDCAST
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
 * Author: Amine Ismail <amine.ismail@sophia.inria.fr>
 *                      <amine.ismail@udcast.com>
 *
 */

#ifndef PACKET_LOSS_COUNTER_H
#define PACKET_LOSS_COUNTER_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/address.h"

namespace ns3 {

class Socket;
class Packet;

/**
 * \ingroup udpclientserver
 *
 * \brief A class to count the number of lost packets.
 *
 * This class records the packet lost in a client/server transmission
 * leveraging a sequence number. All the packets outside a given window
 * (i.e., too old with respect to the last sequence number seen) are considered lost,
 */
class PacketLossCounter
{
public:
  /**
   * \brief Constructor
   * \param bitmapSize The window size. Must be a multiple of 8.
   */
  PacketLossCounter (uint8_t bitmapSize);
  ~PacketLossCounter ();
  /**
   * \brief Record a successfully received packet
   * \param seq the packet sequence number
   */
  void NotifyReceived (uint32_t seq);
  /**
   * \brief Get the number of lost packets.
   * \returns the number of lost packets.
   */
  uint32_t GetLost (void) const;
  /**
   * \brief Return the size of the window used to compute the packet loss.
   * \return the window size.
   */
  uint16_t GetBitMapSize (void) const;
  /**
   * \brief Set the size of the window used to compute the packet loss.
   *
   * \param size The window size. Must be a multiple of 8.
   */
  void SetBitMapSize (uint16_t size);
private:
  /**
   * \brief Check if a sequence number in the window has been received.
   * \param seqNum the sequence number.
   * \returns false if the packet has not been received.
   */
  bool GetBit (uint32_t seqNum);
  /**
   * \brief Set a sequence number to a given state.
   * \param seqNum the sequence number.
   * \param val false if the packet has not been received.
   */
  void SetBit (uint32_t seqNum, bool val);

  uint32_t m_lost; //!< Lost packets counter.
  uint16_t m_bitMapSize; //!< Window size.
  uint32_t m_lastMaxSeqNum; //!< Last sequence number seen.
  uint8_t * m_receiveBitMap; //!< Received packets in the window size.
};
}

#endif /* PACKET_LOSS_COUNTER_H */
