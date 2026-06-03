/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Adrian Sai-wah Tam
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
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */

#ifndef TCP_RX_BUFFER_H
#define TCP_RX_BUFFER_H

#include <map>
#include "ns3/traced-value.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/sequence-number.h"
#include "ns3/ptr.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-sack.h"

namespace ns3 {
class Packet;

/**
 * \ingroup tcp
 *
 * \brief Rx reordering buffer for TCP
 *
 * The class is responsible to safely store the segments, and then
 * returning them in-order to the application, where "in-order" does not means
 * "network-order", but "sender-order" : the bytes should be returned in the
 * same order that the sender application used to push them down on wire.
 *
 * The first useful sequence that this class is waiting is returned by the method
 * NextRxSequence, and could be set at the beginning through MaxRxSequence.
 *
 * The max. size of this buffer is managed through SetMaxBufferSize, and could be
 * retrieved using MaxBufferSize. The current size instead is returned by
 * Size, while the amount of in-order data that could be extracted is returned
 * by the method Available.
 *
 * To store data, use Add; for retrieving a certain amount of ordered data, use
 * the method Extract.
 *
 * SACK list
 * ---------
 *
 * An interesting feature of this class is the ability to maintain an ordered
 * SACK list, under the definition of RFC 2018. When a out-of-order segment
 * reaches this buffer, an ACK will be sent out, and the SACK list is
 * generated or updated. From RFC 2018:
 *
 * > If sent at all, SACK options SHOULD be included in all ACKs which do
 * > not ACK the highest sequence number in the data receiver's queue.
 *
 * For more information about the SACK list, please check the documentation of
 * the method GetSackList.
 *
 * \see GetSackList
 * \see UpdateSackList
 */
class TcpRxBuffer : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief Constructor
   * \param n initial Sequence number to be received
   */
  TcpRxBuffer (uint32_t n = 0);
  virtual ~TcpRxBuffer ();

  // Accessors
  /**
   * \brief Get Next Rx Sequence number
   * \returns Next Rx Sequence number
   */
  SequenceNumber32 NextRxSequence (void) const;
  /**
   * \brief Get the lowest sequence number that this TcpRxBuffer cannot accept
   * \returns the lowest sequence number that this TcpRxBuffer cannot accept
   */
  SequenceNumber32 MaxRxSequence (void) const;
  /**
   * \brief Increment the Next Sequence number
   */
  void IncNextRxSequence (void);
  /**
   * \brief Set the Next Sequence number
   * \param s the Sequence number
   */
  void SetNextRxSequence (const SequenceNumber32& s);
  /**
   * \brief Set the FIN Sequence number (i.e., the one closing the connection)
   * \param s the Sequence number
   */
  void SetFinSequence (const SequenceNumber32& s);
  /**
   * \brief Get the Maximum buffer size
   * \returns the Maximum buffer size
   */
  uint32_t MaxBufferSize (void) const;
  /**
   * \brief Set the Maximum buffer size
   * \param s the Maximum buffer size
   */
  void SetMaxBufferSize (uint32_t s);
  /**
   * \brief Get the actual buffer occupancy
   * \returns buffer occupancy (in bytes)
   */
  uint32_t Size (void) const;
  /**
   * \brief Get the actual number of bytes available to be read
   * \returns size of available data (in bytes)
   */
  uint32_t Available () const;
  /**
   * \brief Check if the buffer did receive all the data (and the connection is closed)
   * \returns true if all data have been received
   */
  bool Finished (void);

  /**
   * Insert a packet into the buffer and update the availBytes counter to
   * reflect the number of bytes ready to send to the application. This
   * function handles overlap by triming the head of the inputted packet and
   * removing data from the buffer that overlaps the tail of the inputted
   * packet
   *
   * \param p packet
   * \param tcph packet's TCP header
   * \return True when success, false otherwise.
   */
  bool Add (Ptr<Packet> p, TcpHeader const& tcph);

  /**
   * Extract data from the head of the buffer as indicated by nextRxSeq.
   * The extracted data is going to be forwarded to the application.
   *
   * \param maxSize maximum number of bytes to extract
   * \returns a packet
   */
  Ptr<Packet> Extract (uint32_t maxSize);

  /**
   * \brief Get the sack list
   *
   * The sack list can be empty, and it is updated each time Add or Extract
   * are called through the private method UpdateSackList.
   *
   * \return a list of isolated blocks
   */
  TcpOptionSack::SackList GetSackList () const;

  /**
   * \brief Get the size of Sack list
   *
   * \return the size of the sack block list; can be empty
   */
  uint32_t GetSackListSize () const;

  /**
   * \brief Says if a FIN bit has been received
   * \return true if we received a FIN bit
   */
  bool GotFin () const { return m_gotFin; }

private:
  /**
   * \brief Update the sack list, with the block seq starting at the beginning
   *
   * Note: the maximum size of the block list is 4. Caller is free to
   * drop blocks at the end to accommodate header size; from RFC 2018:
   *
   * > The data receiver SHOULD include as many distinct SACK blocks as
   * > possible in the SACK option.  Note that the maximum available
   * > option space may not be sufficient to report all blocks present in
   * > the receiver's queue.
   *
   * In fact, the maximum amount of blocks is 4, and if we consider the timestamp
   * (or other) options, it is even less. For more detail about this function,
   * please see the source code and in-line comments.
   *
   * \param head sequence number of the block at the beginning
   * \param tail sequence number of the block at the end
   */
  void UpdateSackList (const SequenceNumber32 &head, const SequenceNumber32 &tail);

  /**
   * \brief Remove old blocks from the sack list
   *
   * Used to remove blocks already delivered to the application.
   *
   * After this call, in the SACK list there will be only blocks with
   * sequence numbers greater than seq; it is perfectly safe to call this
   * function with an empty sack list.
   *
   * \param seq Last sequence to remove
   */
  void ClearSackList (const SequenceNumber32 &seq);

  TcpOptionSack::SackList m_sackList; //!< Sack list (updated constantly)

  /// container for data stored in the buffer
  typedef std::map<SequenceNumber32, Ptr<Packet> >::iterator BufIterator;
  TracedValue<SequenceNumber32> m_nextRxSeq; //!< Seqnum of the first missing byte in data (RCV.NXT)
  SequenceNumber32 m_finSeq;                 //!< Seqnum of the FIN packet
  bool m_gotFin;                             //!< Did I received FIN packet?
  uint32_t m_size;                           //!< Number of total data bytes in the buffer, not necessarily contiguous
  uint32_t m_maxBuffer;                      //!< Upper bound of the number of data bytes in buffer (RCV.WND)
  uint32_t m_availBytes;                     //!< Number of bytes available to read, i.e. contiguous block at head
  std::map<SequenceNumber32, Ptr<Packet> > m_data; //!< Corresponding data (may be null)
};

} //namespace ns3

#endif /* TCP_RX_BUFFER_H */
