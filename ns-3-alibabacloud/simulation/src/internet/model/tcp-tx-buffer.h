/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010-2015 Adrian Sai-wah Tam
 * Copyright (c) 2016 Natale Patriciello <natale.patriciello@gmail.com>
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
 * Original author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */

#ifndef TCP_TX_BUFFER_H
#define TCP_TX_BUFFER_H

#include "ns3/object.h"
#include "ns3/traced-value.h"
#include "ns3/sequence-number.h"
#include "ns3/tcp-option-sack.h"
#include "ns3/tcp-tx-item.h"

namespace ns3 {
class Packet;

/**
 * \ingroup tcp
 *
 * \brief Tcp sender buffer
 *
 * The class keeps track of all data that the application wishes to transmit to
 * the other end. When the data is acknowledged, it is removed from the buffer.
 * The buffer has a maximum size, and data is not saved if the amount exceeds
 * the limit. Packets can be added to the class through the method Add(). An
 * important thing to remember is that all the data managed is strictly
 * sequential. It can be divided into blocks, but all the data follow a strict
 * ordering. That order is managed through SequenceNumber.
 *
 * In other words, this buffer contains numbered bytes (e.g., 1,2,3), and the
 * class is allowed to return only ordered (using "<" as operator) subsets
 * (e.g. 1,2 or 2,3 or 1,2,3).
 *
 * The data structure underlying this is composed by two distinct packet lists.
 * The first (SentList) is initially empty, and it contains the packets
 * returned by the method CopyFromSequence. The second (AppList) is initially
 * empty, and it contains the packets coming from the applications, but that
 * are not transmitted yet as segments. To discover how the chunks are managed
 * and retrieved from these lists, check CopyFromSequence documentation.
 *
 * The head of the data is represented by m_firstByteSeq, and it is returned by
 * HeadSequence(). The last byte is returned by TailSequence(). In this class,
 * we also store the size (in bytes) of the packets inside the SentList in the
 * variable m_sentSize.
 *
 * SACK management
 * ---------------
 *
 * The SACK information is usually saved in a data structure referred as
 * scoreboard. In this implementation, the scoreboard is developed on top of
 * the existing classes. In particular, instead of keeping raw pointers to
 * packets in TcpTxBuffer we added the capability to store some flags
 * associated with every segment sent. This is done through the use of the
 * class TcpTxItem: instead of storing a list of packets, we store a list of
 * TcpTxItem. Each item has different flags (check the corresponding
 * documentation) and maintaining the scoreboard is a matter of travelling the
 * list and set the SACK flag on the corresponding segment sent.
 *
 * Item properties
 * ---------------
 *
 * An item (that represent a segment in flight) is not considered in flight
 * anymore when it is marked lost or sacked. A sacked item represents an
 * item which is received by the other end, but it is still impossible
 * to delete from the list because other pieces are missing at the other
 * end. A lost item never reached the other end, and retransmission is probably
 * needed. Other properties are retransmitted, that indicates if an item was
 * retransmitted, and the sequence number of the first byte of the packet. When
 * a segment is sent for the first time, only the sequence number is set, and
 * all the remaining properties are set to false. If an item is explicitly
 * sacked by the receiver, we mark it as such. Each time we receive updated
 * sack information from the other end, we perform a check to evaluate the
 * segments that can be lost (\see UpdateLostCount), and we set the flags
 * accordingly.
 *
 * Management of bytes in flight
 * -----------------------------
 *
 * Since this class manages all the output segments and the scoreboard, we can
 * do calculations about the number of bytes in flights. Earlier versions of
 * this class used algorithms copied from RFC 6675. They were inefficient
 * because they required a complete walk into the list of sent segments each
 * time a simple question, such as "Is this sequence lost?" or "How many bytes
 * are in flight?". Therefore, the class has been updated keeping in
 * consideration the RFCs (including RFC 4898) and the Linux operating
 * system. As a reference, we kept the older methods for calculating the
 * bytes in flight and if a segment is lost, renaming them as "RFC" version
 * of the methods. To have a look how the calculations are made, please see
 * BytesInFlight method.
 *
 * Lost segments
 * -------------
 *
 * After the sender receives a new SACK block, it updates the amount of segment
 * that it considers as lost, following the specifications made in RFC 6675
 * (for more detail please see the method UpdateLostCount). In case of SACKless
 * connection, the TcpSocketImplementation should provide hints through
 * the MarkHeadAsLost and AddRenoSack methods.
 *
 * \see BytesInFlight
 * \see Size
 * \see SizeFromSequence
 * \see CopyFromSequence
 */
class TcpTxBuffer : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief Constructor
   * \param n initial Sequence number to be transmitted
   */
  TcpTxBuffer (uint32_t n = 0);
  virtual ~TcpTxBuffer (void);

  // Accessors

  /**
   * \brief Get the sequence number of the buffer head
   * \returns the first byte's sequence number
   */
  SequenceNumber32 HeadSequence (void) const;

  /**
   * \brief Get the sequence number of the buffer tail (plus one)
   * \returns the last byte's sequence number + 1
   */
  SequenceNumber32 TailSequence (void) const;

  /**
   * \brief Returns total number of bytes in this buffer
   * \returns total number of bytes in this Tx buffer
   */
  uint32_t Size (void) const;

  /**
   * \brief Get the maximum buffer size
   * \returns the Tx window size (in bytes)
   */
  uint32_t MaxBufferSize (void) const;

  /**
   * \brief Set the maximum buffer size
   * \param n Tx window size (in bytes)
   */
  void SetMaxBufferSize (uint32_t n);

  /**
   * \brief check whether SACK is used on the corresponding TCP socket
   * \return true if SACK is used
   */
  bool IsSackEnabled (void) const;

  /**
   * \brief tell tx-buffer whether SACK is used on this TCP socket
   *
   * \param enabled whether sack is used
   */
  void SetSackEnabled (bool enabled);

  /**
   * \brief Returns the available capacity of this buffer
   * \returns available capacity in this Tx window
   */
  uint32_t Available (void) const;

  /**
   * \brief Set the DupAckThresh
   * \param dupAckThresh the threshold
   */
  void SetDupAckThresh (uint32_t dupAckThresh);

  /**
   * \brief Set the segment size
   * \param segmentSize the segment size
   */
  void SetSegmentSize (uint32_t segmentSize);

  /**
   * \brief Return the number of segments in the sent list that
   * have been transmitted more than once, without acknowledgment.
   *
   * This method is to support the retransmits count for determining PipeSize
   * in NewReno-style TCP.
   *
   * \returns number of segments that have been transmitted more than once, without acknowledgment
   */
  uint32_t GetRetransmitsCount (void) const;

  /**
   * \brief Get the number of segments that we believe are lost in the network
   *
   * It is calculated in UpdateLostCount.
   * \return the number of lost segment
   */
  uint32_t GetLost (void) const;

  /**
   * \brief Get the number of segments that have been explicitly sacked by the receiver.
   * \return the number of sacked segment.
   */
  uint32_t GetSacked (void) const;

  /**
   * \brief Append a data packet to the end of the buffer
   *
   * \param p The packet to be appended to the Tx buffer
   * \return Boolean to indicate success
   */
  bool Add (Ptr<Packet> p);

  /**
   * \brief Returns the number of bytes from the buffer in the range [seq, tailSequence)
   *
   * \param seq initial sequence number
   * \returns the number of bytes from the buffer in the range
   */
  uint32_t SizeFromSequence (const SequenceNumber32& seq) const;

  /**
   * \brief Copy data from the range [seq, seq+numBytes) into a packet
   *
   * In the following, we refer to the block [seq, seq+numBytes) simply as "block".
   * We check the boundary of the block, and divide the possibilities in three
   * cases:
   *
   * - the block have already been transmitted (managed in GetTransmittedSegment)
   * - the block have not been transmitted yet (managed in GetNewSegment)
   *
   * The last case is when the block is partially transmitted and partially
   * not transmitted. We trick this case by requesting the portion not transmitted
   * from GetNewSegment, and then calling GetTransmittedSegment with the full
   * block range.
   *
   * \param numBytes number of bytes to copy
   * \param seq start sequence number to extract
   * \returns a pointer to the TcpTxItem that corresponds to what requested.
   * Please do not delete the pointer, nor modify Packet data or sequence numbers.
   */
  TcpTxItem* CopyFromSequence (uint32_t numBytes, const SequenceNumber32& seq);

  /**
   * \brief Set the head sequence of the buffer
   *
   * Set the head (m_firstByteSeq) to seq. Supposed to be called only when the
   * connection is just set up and we did not send any data out yet.
   * \param seq The sequence number of the head byte
   */
  void SetHeadSequence (const SequenceNumber32& seq);

  /**
   * \brief Checks whether the ack corresponds to retransmitted data
   *
   * \param ack ACK number received
   * \return true if retransmitted data was acked
   */
  bool IsRetransmittedDataAcked (const SequenceNumber32& ack) const;

  /**
   * \brief Discard data up to but not including this sequence number.
   *
   * \param seq The first sequence number to maintain after discarding all the
   * previous sequences.
   * \param beforeDelCb Callback invoked, if it is not null, before the deletion
   * of an Item (because it was, probably, ACKed)
   */
  void DiscardUpTo (const SequenceNumber32& seq,
                    const Callback<void, TcpTxItem *> &beforeDelCb = m_nullCb);

  /**
   * \brief Update the scoreboard
   * \param list list of SACKed blocks
   * \param sackedCb Callback invoked, if it is not null, when a segment has been
   * SACKed by the receiver.
   * \returns the number of bytes newly sacked by the list of blocks
   */
  uint32_t Update (const TcpOptionSack::SackList &list,
                   const Callback<void, TcpTxItem *> &sackedCb = m_nullCb);

  /**
   * \brief Check if a segment is lost
   *
   * It does a check on the flags to determine if the segment has to be considered
   * as lost for an external class
   *
   * \param seq sequence to check
   * \return true if the sequence is supposed to be lost, false otherwise
   */
  bool IsLost (const SequenceNumber32 &seq) const;

  /**
   * \brief Get the next sequence number to transmit, according to RFC 6675
   *
   * \param seq Next sequence number to transmit, based on the scoreboard information
   * \param seqHigh Maximum sequence number to transmit, based on SMSS and/or receiver window
   * \param isRecovery true if the socket congestion state is in recovery mode
   * \return true is seq is updated, false otherwise
   */
  bool NextSeg (SequenceNumber32 *seq, SequenceNumber32 *seqHigh, bool isRecovery) const;

  /**
   * \brief Return total bytes in flight
   *
   * Counting packets in flight is pretty simple:
   *
   * \f$in_flight = sentSize - leftOut + retrans\f$
   *
   * sentsize is SND.NXT-SND.UNA, retrans is the number of retransmitted segments.
   * leftOut is the number of segment that left the network without being ACKed:
   *
   * \f$leftOut = sacked_out + lost_out\f$
   *
   * To see how we define the lost packets, look at the method UpdateLostCount.
   *
   * \returns total bytes in flight
   */
  uint32_t BytesInFlight () const;

  /**
   * \brief Set the entire sent list as lost (typically after an RTO)
   *
   * Used to set all the sent list as lost, so the bytes in flight is not counting
   * them as in flight, but we will continue to use SACK information for
   * recovering the timeout.
   *
   * Moreover, reset the retransmit flag for every item.
   * \param resetSack True if the function should reset the SACK flags.
   */
  void SetSentListLost (bool resetSack = false);

  /**
   * \brief Check if the head is retransmitted
   *
   * \return true if the head is retransmitted, false in all other cases
   * (including no segment sent)
   */
  bool IsHeadRetransmitted () const;

  /**
   * \brief DeleteRetransmittedFlagFromHead
   */
  void DeleteRetransmittedFlagFromHead ();

  /**
   * \brief Reset the sent list
   *
   */
  void ResetSentList ();

  /**
   * \brief Take the last segment sent and put it back into the un-sent list
   * (at the beginning)
   */
  void ResetLastSegmentSent ();

  /**
   * \brief Mark the head of the sent list as lost.
   */
  void MarkHeadAsLost ();

  /**
   * \brief Emulate SACKs for SACKless connection: account for a new dupack.
   *
   * The method walk the list of the sent segment until it finds a segment
   * that was not accounted in the sackedOut count. The head will never
   * be included. To reset the information added with this function (e.g.,
   * after an RTO) please use ResetRenoSack.
   *
   * The method DiscardUpTo, when invoked, will make sure to properly clean any
   * flag on the discarded item. As example, if the implementation discard an item
   * that is marked as sacked, the sackedOut count is decreased accordingly.
   */
  void AddRenoSack ();

  /**
   * \brief Reset the SACKs.
   *
   * Reset the Scoreboard from all SACK information. This method also works in
   * case the SACKs are set by the Update method.
   */
  void ResetRenoSack ();

  /**
   * \brief Set callback to obtain receiver window value
   * \param rWndCallback receiver window callback
   */
  void SetRWndCallback (Callback<uint32_t> rWndCallback);

private:
  friend std::ostream & operator<< (std::ostream & os, TcpTxBuffer const & tcpTxBuf);

  typedef std::list<TcpTxItem*> PacketList; //!< container for data stored in the buffer

  /**
   * \brief Update the lost count
   *
   * Reset lost to 0, then walk the sent list looking for lost segments.
   * We have two possible algorithms for detecting lost packets:
   *
   * - RFC 6675 algorithm, which says that if more than "Dupack thresh" (e.g., 3)
   * sacked segments above the sequence, then we can consider the sequence lost;
   * - NewReno (RFC6582): in Recovery we assume that one segment is lost
   * (classic Reno). While we are in Recovery and a partial ACK arrives,
   * we assume that one more packet is lost (NewReno).
   *
   * The {New}Reno cases, for now, are managed in TcpSocketBase through the
   * call to MarkHeadAsLost.
   * This function is, therefore, called after a SACK option has been received,
   * and updates the lost count. It can be probably optimized by not walking
   * the entire list, but a subset.
   *
   */
  void UpdateLostCount ();

  /**
   * \brief Remove the size specified from the lostOut, retrans, sacked count
   *
   * Used only in DiscardUpTo
   *
   * \param item Item that will be discarded
   * \param size size to remove (can be different from pktSize because of fragmentation)
   */
  void RemoveFromCounts (TcpTxItem *item, uint32_t size);

  /**
   * \brief Decide if a segment is lost based on RFC 6675 algorithm.
   * \param seq Sequence
   * \param segment Iterator to the sequence
   * \return true if seq is lost per RFC 6675, false otherwise
   */
  bool IsLostRFC (const SequenceNumber32 &seq, const PacketList::const_iterator &segment) const;

  /**
   * \brief Calculate the number of bytes in flight per RFC 6675
   * \return the number of bytes in flight
   */
  uint32_t BytesInFlightRFC () const;

  /**
   * \brief Get a block of data not transmitted yet and move it into SentList
   *
   * If the block is not yet transmitted, hopefully, seq is exactly the sequence
   * number of the first byte of the first packet inside AppList. We extract
   * the block from AppList and move it into the SentList, before returning the
   * block itself. We manage possible fragmentation (or merges) inside AppList
   * through GetPacketFromList.
   *
   * \see GetPacketFromList
   * \param numBytes number of bytes to copy
   *
   * \return the item that contains the right packet
   */
  TcpTxItem* GetNewSegment (uint32_t numBytes);

  /**
   * \brief Get a block of data previously transmitted
   *
   * This is clearly a retransmission, and if everything is going well,
   * the block requested is matching perfectly with another one requested
   * in the past. If not, fragmentation or merge are required. We manage
   * both inside GetPacketFromList.
   *
   * \see GetPacketFromList
   *
   * \param numBytes number of bytes to copy
   * \param seq sequence requested
   * \returns the item that contains the right packet
   */
  TcpTxItem* GetTransmittedSegment (uint32_t numBytes, const SequenceNumber32 &seq);

  /**
   * \brief Get a block (which is returned as Packet) from a list
   *
   * This function extract a block [requestedSeq,numBytes) from the list, which
   * starts at startingSeq.
   *
   * The cases we need to manage are two, and they are depicted in the following
   * image:
   *
   *\verbatim
                       |------|     |----|     |----|
                list = |      | --> |    | --> |    |
                       |------|     |----|     |----|

                       ^      ^
                       | ^ ^  |         (1)
                     seq | |  seq + numBytes
                         | |
                         | |
                      seq   seq + numBytes     (2)
   \endverbatim
   *
   * The case 1 is easy to manage: the requested block is exactly a packet
   * already stored. If one value (seq or seq + numBytes) does not align
   * to a packet boundary, or when both values does not align (case 2), it is
   * a bit more complex.
   *
   * Basically, we have two possible operations:
   *
   *  - fragment : split an existing packet in two
   *  - merge    : merge two existing packets in one
   *
   * and we reduce case (2) to case (1) through sequentially applying fragment
   * or merge. For instance:
   *
   *\verbatim
      |------|
      |      |
      |------|

      ^ ^  ^ ^
      | |  | |
  start |  | |
        |  | end
       seq |
           seq + numBytes
   \endverbatim
   *
   * To reduce to case (1), we need to perform two fragment operations:
   *
   * - fragment (start, seq)
   * - fragment (seq + numBytes, end)
   *
   * After these operations, the requested block is exactly the resulting packet.
   * Merge operation is required when the requested block span over two (or more)
   * existing packets.
   *
   * While this could be extremely slow in the worst possible scenario (one big
   * packet which is split in small packets for transmission, and merged for
   * re-transmission) that scenario is unlikely during a TCP transmission (since
   * MSS can change, but it is stable, and retransmissions do not happen for
   * each segment).
   *
   * \param list List to extract block from
   * \param startingSeq Starting sequence of the list
   * \param numBytes Bytes to extract, starting from requestedSeq
   * \param requestedSeq Requested sequence
   * \param listEdited output parameter which indicates if the list has been edited
   * \return the item that contains the right packet
   */
  TcpTxItem* GetPacketFromList (PacketList &list, const SequenceNumber32 &startingSeq,
                                uint32_t numBytes, const SequenceNumber32 &requestedSeq,
                                bool *listEdited = nullptr) const;

  /**
   * \brief Merge two TcpTxItem
   *
   * Merge t2 in t1. It consists in copying the lastSent field if t2 is more
   * recent than t1. Retransmitted field is copied only if it set in t2 but not
   * in t1. Sacked is copied only if it is true in both items.
   *
   * \param t1 first item
   * \param t2 second item
   */
  void MergeItems (TcpTxItem *t1, TcpTxItem *t2) const;

  /**
   * \brief Split one TcpTxItem
   *
   * Move "size" bytes from t2 into t1, copying all the fields.
   * Adjust the starting sequence of each item.
   *
   * \param t1 first item
   * \param t2 second item
   * \param size Size to split
   */
  void SplitItems (TcpTxItem *t1, TcpTxItem *t2, uint32_t size) const;

  /**
   * \brief Check if the values of sacked, lost, retrans, are in sync
   * with the sent list.
   */
  void ConsistencyCheck () const;

  /**
   * \brief Find the highest SACK byte
   * \return a pair with the highest byte and an iterator inside m_sentList
   */
  std::pair <TcpTxBuffer::PacketList::const_iterator, SequenceNumber32>
  FindHighestSacked () const;

  PacketList m_appList;  //!< Buffer for application data
  PacketList m_sentList; //!< Buffer for sent (but not acked) data
  uint32_t m_maxBuffer;  //!< Max number of data bytes in buffer (SND.WND)
  uint32_t m_size;       //!< Size of all data in this buffer
  uint32_t m_sentSize;   //!< Size of sent (and not discarded) segments
  Callback<uint32_t> m_rWndCallback; //!< Callback to obtain RCV.WND value

  TracedValue<SequenceNumber32> m_firstByteSeq; //!< Sequence number of the first byte in data (SND.UNA)
  std::pair <PacketList::const_iterator, SequenceNumber32> m_highestSack; //!< Highest SACK byte

  uint32_t m_lostOut   {0}; //!< Number of lost bytes
  uint32_t m_sackedOut {0}; //!< Number of sacked bytes
  uint32_t m_retrans   {0}; //!< Number of retransmitted bytes

  uint32_t m_dupAckThresh {0}; //!< Duplicate Ack threshold from TcpSocketBase
  uint32_t m_segmentSize {0}; //!< Segment size from TcpSocketBase
  bool     m_renoSack {false}; //!< Indicates if AddRenoSack was called
  bool     m_sackEnabled {true}; //!< Indicates if SACK is enabled on this connection

  static Callback<void, TcpTxItem *> m_nullCb; //!< Null callback for an item
};

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param tcpTxBuf the TcpTxBuffer to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os, TcpTxBuffer const & tcpTxBuf);

/**
 * \brief Output operator.
 * \param os The output stream.
 * \param item the item to print.
 * \returns The output stream.
 */
std::ostream & operator<< (std::ostream & os, TcpTxItem const & item);

} // namespace ns3

#endif /* TCP_TX_BUFFER_H */
