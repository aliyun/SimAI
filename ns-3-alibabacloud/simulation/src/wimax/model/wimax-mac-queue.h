/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#ifndef WIMAX_MAC_QUEUE_H
#define WIMAX_MAC_QUEUE_H

#include <queue>
#include <stdint.h>
#include "wimax-mac-header.h"
#include "ns3/packet.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"
#include "ns3/nstime.h"

namespace ns3 {

/**
 * \ingroup wimax
 * Class implementing the device packet queue
 */
class WimaxMacQueue : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  WimaxMacQueue (void);
  /**
   * Constructor
   *
   * \param maxSize maximum size of queue
   */
  WimaxMacQueue (uint32_t maxSize);
  ~WimaxMacQueue (void);
  /**
   * \brief set the maximum queue size
   * \param maxSize the max queue size
   */
  void SetMaxSize (uint32_t maxSize);
  /**
   * \return the maximum queue size
   */
  uint32_t GetMaxSize (void) const;
  /**
   * \brief Enqueue a packet
   * \param packet the packet to enqueue
   * \param hdrType the mac header type of the packet
   * \param hdr the header of the packet
   * \return true if successful
   */
  bool Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType,
                const GenericMacHeader &hdr);
  /**
   * \brief Dequeue a packet of type packetType from the queue
   * \param packetType the type of the packet
   * \return the first packet in the queue
   */
  Ptr<Packet> Dequeue (MacHeaderType::HeaderType packetType);
  /**
   * \brief Dequeue a fragment of size availableByte from the queue
   * \param packetType the type of the packet
   * \param availableByte the size of the fragment
   * \return the first packet in the queue if its size is lower than availableByte, the first availableByte of the first packet otherwise
   */
  Ptr<Packet> Dequeue (MacHeaderType::HeaderType packetType,
                       uint32_t availableByte);

  /**
   * \brief Exclusively for BS.
   *  Two versions of Peek function (with Generic MAC Header as parameter) only
   *  for BS, as BS's scheduler needs to know the CID in Generic MAC Header.
   * \param hdr the header of the packet
   * \return the first packet in the queue
   */
  Ptr<Packet> Peek (GenericMacHeader &hdr) const;
  /**
   * \brief Exclusively for BS.
   *  Two versions of Peek function (with Generic MAC Header as parameter) only
   *  for BS, as BS's scheduler needs to know the CID in Generic MAC Header.
   * \param hdr the header of the packet
   * \param timeStamp the timestamp of the packet
   * \return the first packet in the queue
   */
  Ptr<Packet> Peek (GenericMacHeader &hdr, Time &timeStamp) const;

  /**
   * \brief Two versions for SS without Generic MAC Header parameter, as SS's
   *  scheduler does not require to know the CID in Generic MAC Header.
   * \param packetType the type of the packet
   * \return the first packet in the queue
   */
  Ptr<Packet> Peek (MacHeaderType::HeaderType packetType) const;
  /**
   * \brief Two versions for SS without Generic MAC Header parameter, as SS's
   *  scheduler does not require to know the CID in Generic MAC Header.
   * \param packetType the type of the packet
   * \param timeStamp the timestamp of the packet
   * \return the first packet in the queue
   */
  Ptr<Packet> Peek (MacHeaderType::HeaderType packetType, Time &timeStamp) const;

  /**
   * Check if queue is empty
   * \returns true if empty
   */
  bool IsEmpty (void) const;

  /**
   * Exclusively for SS.
   * \return true if queue is empty of type packetType
   * \param packetType packet type to check
   */
  bool IsEmpty (MacHeaderType::HeaderType packetType) const;

  /**
   * Get size of queue
   * \returns the size
   */
  uint32_t GetSize (void) const;
  /**
   * Get number of bytes in queue
   * \returns the number of bytes
   */
  uint32_t GetNBytes (void) const;

  /**
   * Check for fragmentation of the first packet of the specified type
   * \param packetType the packet type
   * \returns true if fragmented
   */
  bool CheckForFragmentation (MacHeaderType::HeaderType packetType);
  /**
   * Get first packet header size of the specified type
   * \param packetType the packet type
   * \returns the first packet header size
   */
  uint32_t GetFirstPacketHdrSize (MacHeaderType::HeaderType packetType);
  /**
   * Get first packet payload size of the specified type
   * \param packetType the packet type
   * \returns the first packet payload size
   */
  uint32_t GetFirstPacketPayloadSize (MacHeaderType::HeaderType packetType);
  /**
   * Get required number of bytes to hold first packet of packetType
   * \param packetType the packet type
   * \returns the sum of the first packet header and payload (of specified type)
   */
  uint32_t GetFirstPacketRequiredByte (MacHeaderType::HeaderType packetType);
  /**
   * Get queue length considering also the MAC overhead
   * \return queue length
   */
  uint32_t GetQueueLengthWithMACOverhead (void);
  /// Set fragmentation function
  /**
   * Set fragmentation state for first packet of type packetType
   * \param packetType packetType to check
   */
  void SetFragmentation (MacHeaderType::HeaderType packetType);
  /**
   * Set fragment number for first packet of type packetType
   * \param packetType packetType to check
   */
  void SetFragmentNumber (MacHeaderType::HeaderType packetType);
  /**
   * Set fragment offset for first packet of type packetType
   * \param packetType packetType to check
   * \param offset offset value to set
   */
  void SetFragmentOffset (MacHeaderType::HeaderType packetType, uint32_t offset);

  /// QueueElement structure
  struct QueueElement
  {
    QueueElement (void);
    /**
     * Constructor
     *
     * \param packet the packet
     * \param hdrType MAC header type
     * \param hdr MAC header
     * \param timeStamp time stamp
     */
    QueueElement (Ptr<Packet> packet,
                  const MacHeaderType &hdrType,
                  const GenericMacHeader &hdr,
                  Time timeStamp);
    /**
     * Get size function
     * \returns the size
     */
    uint32_t GetSize (void) const;
    Ptr<Packet> m_packet; ///< packet
    MacHeaderType m_hdrType; ///< header type
    GenericMacHeader m_hdr; ///< header
    Time m_timeStamp; ///< timestamp

    /**
     To manage fragmentation feature, each QueueElement have 3 new fields:
     m_fragmentation that becomes true when the fragmentation starts;
     m_fragmentNumber that is incremented when a new fragment is sent
     m_fragmentOffset that tracks the start of the next fragment into the packet
     */
    bool m_fragmentation; ///< becomes true when the fragmentation starts
    uint32_t m_fragmentNumber; ///< incremented when a new fragment is sent
    uint32_t m_fragmentOffset; ///< tracks the start of the next fragment into the packet

    /// Set fragmentation
    void SetFragmentation (void);
    /// Set fragment number
    void SetFragmentNumber (void);
    /**
     * Set fragment offset
     * \param offset the offset
     */
    void SetFragmentOffset (uint32_t offset);
  };

private:

  /**
   In the case of non-UGS service flows at the SS side the queue will store both data packets
   and bandwidth request packets. The two are distinguished by their headers. The below two functions
   are for this purpose exclusively. The Front function returns the first packet of a specific packet
   type from the queue (which may not necessarily be at the front of the queue), and the Pop function
   pops that elements.
   * \param packetType the type of the packet
   * \returns the first packet in the queue of the specified type
   */
  WimaxMacQueue::QueueElement Front (MacHeaderType::HeaderType packetType) const;
  /**
   * Pop function
   * \param packetType the packet type
   */
  void Pop (MacHeaderType::HeaderType packetType);

  /// PacketQueue typedef
  typedef std::deque<QueueElement> PacketQueue;
  PacketQueue m_queue; ///< the queue
  uint32_t m_maxSize; ///< maximum size
  uint32_t m_bytes; ///< bytes
  uint32_t m_nrDataPackets; ///< number data packets
  uint32_t m_nrRequestPackets; ///< number request packets

  TracedCallback<Ptr<const Packet> > m_traceEnqueue; ///< enqueue trace callback
  TracedCallback<Ptr<const Packet> > m_traceDequeue; ///< dequeue trace callback
  TracedCallback<Ptr<const Packet> > m_traceDrop; ///< drop trace callback
public:
  /**
   * Get packet queue function
   * \returns the packet queue
   */
  const WimaxMacQueue::PacketQueue & GetPacketQueue (void) const;
};

} // namespace ns3

#endif /* WIMAX_MAC_QUEUE_H */
