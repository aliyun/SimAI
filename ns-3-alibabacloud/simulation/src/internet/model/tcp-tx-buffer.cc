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

#include <algorithm>
#include <iostream>

#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/tcp-option-ts.h"

#include "tcp-tx-buffer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpTxBuffer");
NS_OBJECT_ENSURE_REGISTERED (TcpTxBuffer);

Callback<void, TcpTxItem *> TcpTxBuffer::m_nullCb = MakeNullCallback<void, TcpTxItem*> ();
TypeId
TcpTxBuffer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpTxBuffer")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpTxBuffer> ()
    .AddTraceSource ("UnackSequence",
                     "First unacknowledged sequence number (SND.UNA)",
                     MakeTraceSourceAccessor (&TcpTxBuffer::m_firstByteSeq),
                     "ns3::SequenceNumber32TracedValueCallback")
  ;
  return tid;
}

/* A user is supposed to create a TcpSocket through a factory. In TcpSocket,
 * there are attributes SndBufSize and RcvBufSize to control the default Tx and
 * Rx window sizes respectively, with default of 128 KiByte. The attribute
 * SndBufSize is passed to TcpTxBuffer by TcpSocketBase::SetSndBufSize() and in
 * turn, TcpTxBuffer:SetMaxBufferSize(). Therefore, the m_maxBuffer value
 * initialized below is insignificant.
 */
TcpTxBuffer::TcpTxBuffer (uint32_t n)
  : m_maxBuffer (32768), m_size (0), m_sentSize (0), m_firstByteSeq (n)
{
  m_rWndCallback = MakeNullCallback<uint32_t> ();
}

TcpTxBuffer::~TcpTxBuffer (void)
{
  PacketList::iterator it;

  for (it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      TcpTxItem *item = *it;
      m_sentSize -= item->m_packet->GetSize ();
      delete item;
    }

  for (it = m_appList.begin (); it != m_appList.end (); ++it)
    {
      TcpTxItem *item = *it;
      m_size -= item->m_packet->GetSize ();
      delete item;
    }
}

SequenceNumber32
TcpTxBuffer::HeadSequence (void) const
{
  return m_firstByteSeq;
}

SequenceNumber32
TcpTxBuffer::TailSequence (void) const
{
  return m_firstByteSeq + SequenceNumber32 (m_size);
}

uint32_t
TcpTxBuffer::Size (void) const
{
  return m_size;
}

uint32_t
TcpTxBuffer::MaxBufferSize (void) const
{
  return m_maxBuffer;
}

void
TcpTxBuffer::SetMaxBufferSize (uint32_t n)
{
  m_maxBuffer = n;
}

bool
TcpTxBuffer::IsSackEnabled (void) const
{
  return m_sackEnabled;
}

void
TcpTxBuffer::SetSackEnabled (bool enabled)
{
  m_sackEnabled = enabled;
}

uint32_t
TcpTxBuffer::Available (void) const
{
  return m_maxBuffer - m_size;
}

void
TcpTxBuffer::SetDupAckThresh (uint32_t dupAckThresh)
{
  m_dupAckThresh = dupAckThresh;
}

void
TcpTxBuffer::SetSegmentSize (uint32_t segmentSize)
{
  m_segmentSize = segmentSize;
}

uint32_t
TcpTxBuffer::GetRetransmitsCount (void) const
{
  return m_retrans;
}

uint32_t
TcpTxBuffer::GetLost (void) const
{
  return m_lostOut;
}

uint32_t
TcpTxBuffer::GetSacked (void) const
{
  return m_sackedOut;
}

void
TcpTxBuffer::SetHeadSequence (const SequenceNumber32& seq)
{
  NS_LOG_FUNCTION (this << seq);
  m_firstByteSeq = seq;

  if (m_sentList.size () > 0)
    {
      m_sentList.front ()->m_startSeq = seq;
    }

  // if you change the head with data already sent, something bad will happen
  NS_ASSERT (m_sentList.size () == 0);
  m_highestSack = std::make_pair (m_sentList.end (), SequenceNumber32 (0));
}

bool
TcpTxBuffer::Add (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p);
  NS_LOG_LOGIC ("Try to append " << p->GetSize () << " bytes to window starting at "
                                << m_firstByteSeq << ", availSize=" << Available ());
  if (p->GetSize () <= Available ())
    {
      if (p->GetSize () > 0)
        {
          TcpTxItem *item = new TcpTxItem ();
          item->m_packet = p->Copy ();
          m_appList.insert (m_appList.end (), item);
          m_size += p->GetSize ();

          NS_LOG_LOGIC ("Updated size=" << m_size << ", lastSeq=" <<
                        m_firstByteSeq + SequenceNumber32 (m_size));
        }
      return true;
    }
  NS_LOG_LOGIC ("Rejected. Not enough room to buffer packet.");
  return false;
}

uint32_t
TcpTxBuffer::SizeFromSequence (const SequenceNumber32& seq) const
{
  NS_LOG_FUNCTION (this << seq);
  // Sequence of last byte in buffer
  SequenceNumber32 lastSeq = TailSequence ();

  if (lastSeq >= seq)
    {
      return static_cast<uint32_t> (lastSeq - seq);
    }

  NS_LOG_ERROR ("Requested a sequence beyond our space (" << seq << " > " << lastSeq <<
                "). Returning 0 for convenience.");
  return 0;
}

TcpTxItem *
TcpTxBuffer::CopyFromSequence (uint32_t numBytes, const SequenceNumber32& seq)
{
  NS_LOG_FUNCTION (this << numBytes << seq);

  NS_ABORT_MSG_IF (m_firstByteSeq > seq,
                   "Requested a sequence number which is not in the buffer anymore");
  ConsistencyCheck ();

  // Real size to extract. Insure not beyond end of data
  uint32_t s = std::min (numBytes, SizeFromSequence (seq));

  if (s == 0)
    {
      return nullptr;
    }

  TcpTxItem *outItem = nullptr;

  if (m_firstByteSeq + m_sentSize >= seq + s)
    {
      // already sent this block completely
      outItem = GetTransmittedSegment (s, seq);
      NS_ASSERT (outItem != nullptr);
      NS_ASSERT (!outItem->m_sacked);

      NS_LOG_DEBUG ("Returning already sent item " << *outItem << " from " << *this);
    }
  else if (m_firstByteSeq + m_sentSize <= seq)
    {
      NS_ABORT_MSG_UNLESS (m_firstByteSeq + m_sentSize == seq,
                           "Requesting a piece of new data with an hole");

      // this is the first time we transmit this block
      outItem = GetNewSegment (s);
      NS_ASSERT (outItem != nullptr);
      NS_ASSERT (outItem->m_retrans == false);

      NS_LOG_DEBUG ("Returning new item " << *outItem << " from " << *this);
    }
  else if (m_firstByteSeq.Get ().GetValue () + m_sentSize > seq.GetValue ()
           && m_firstByteSeq.Get ().GetValue () + m_sentSize < seq.GetValue () + s)
    {
      // Partial: a part is retransmission, the remaining data is new
      // Just return the old segment, without taking new data. Hopefully
      // TcpSocketBase will request new data

      uint32_t amount = (m_firstByteSeq.Get ().GetValue () + m_sentSize) - seq.GetValue ();

      return CopyFromSequence (amount, seq);
    }

  outItem->m_lastSent = Simulator::Now ();
  NS_ASSERT_MSG (outItem->m_startSeq >= m_firstByteSeq,
                 "Returning an item " << *outItem << " with SND.UNA as " <<
                 m_firstByteSeq);
  ConsistencyCheck ();
  return outItem;
}

TcpTxItem*
TcpTxBuffer::GetNewSegment (uint32_t numBytes)
{
  NS_LOG_FUNCTION (this << numBytes);

  SequenceNumber32 startOfAppList = m_firstByteSeq + m_sentSize;

  NS_LOG_INFO ("AppList start at " << startOfAppList << ", sentSize = " <<
               m_sentSize << " firstByte: " << m_firstByteSeq);

  TcpTxItem *item = GetPacketFromList (m_appList, startOfAppList,
                                       numBytes, startOfAppList);
  item->m_startSeq = startOfAppList;

  // Move item from AppList to SentList (should be the first, not too complex)
  auto it = std::find (m_appList.begin (), m_appList.end (), item);
  NS_ASSERT (it != m_appList.end ());

  m_appList.erase (it);
  m_sentList.insert (m_sentList.end (), item);
  m_sentSize += item->m_packet->GetSize ();

  return item;
}

TcpTxItem*
TcpTxBuffer::GetTransmittedSegment (uint32_t numBytes, const SequenceNumber32 &seq)
{
  NS_LOG_FUNCTION (this << numBytes << seq);
  NS_ASSERT (seq >= m_firstByteSeq);
  NS_ASSERT (numBytes <= m_sentSize);
  NS_ASSERT (m_sentList.size () >= 1);

  auto it = m_sentList.begin ();
  bool listEdited = false;
  uint32_t s = numBytes;

  // Avoid to merge different packet for this retransmission if flags are
  // different.
  for (; it != m_sentList.end(); ++it)
    {
      if ((*it)->m_startSeq == seq)
        {
          auto next = it;
          next++;
          if (next != m_sentList.end ())
            {
              // Next is not sacked and have the same value for m_lost ... there is the possibility to merge
              if ((! (*next)->m_sacked) && ((*it)->m_lost == (*next)->m_lost))
                {
                  s = std::min(s, (*it)->m_packet->GetSize () + (*next)->m_packet->GetSize ());
                }
              else
                {
                  // Next is sacked... better to retransmit only the first segment
                  s = std::min(s, (*it)->m_packet->GetSize ());
                }
            }
          else
            {
              s = std::min(s, (*it)->m_packet->GetSize ());
            }
          break;
        }
    }

  TcpTxItem *item = GetPacketFromList (m_sentList, m_firstByteSeq, s, seq, &listEdited);

  if (! item->m_retrans)
    {
      m_retrans += item->m_packet->GetSize ();
      item->m_retrans = true;
    }

  return item;
}

std::pair <TcpTxBuffer::PacketList::const_iterator, SequenceNumber32>
TcpTxBuffer::FindHighestSacked () const
{
  NS_LOG_FUNCTION (this);

  SequenceNumber32 beginOfCurrentPacket = m_firstByteSeq;

  auto ret = std::make_pair (m_sentList.end (), SequenceNumber32 (0));

  for (auto it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      const TcpTxItem *item = *it;
      if (item->m_sacked)
        {
          ret = std::make_pair (it, beginOfCurrentPacket);
        }
      beginOfCurrentPacket += item->m_packet->GetSize ();
    }

  return ret;
}


void
TcpTxBuffer::SplitItems (TcpTxItem *t1, TcpTxItem *t2, uint32_t size) const
{
  NS_ASSERT (t1 != nullptr && t2 != nullptr);
  NS_LOG_FUNCTION (this << *t2 << size);

  t1->m_packet = t2->m_packet->CreateFragment (0, size);
  t2->m_packet->RemoveAtStart (size);

  t1->m_startSeq = t2->m_startSeq;
  t1->m_sacked = t2->m_sacked;
  t1->m_lastSent = t2->m_lastSent;
  t1->m_retrans = t2->m_retrans;
  t1->m_lost = t2->m_lost;

  t2->m_startSeq += size;

  NS_LOG_INFO ("Split of size " << size << " result: t1 " << *t1 << " t2 " << *t2);
}

TcpTxItem*
TcpTxBuffer::GetPacketFromList (PacketList &list, const SequenceNumber32 &listStartFrom,
                                uint32_t numBytes, const SequenceNumber32 &seq,
                                bool *listEdited) const
{
  NS_LOG_FUNCTION (this << numBytes << seq);

  /*
   * Our possibilities are sketched out in the following:
   *
   *                    |------|     |----|     |----|
   * GetList (m_data) = |      | --> |    | --> |    |
   *                    |------|     |----|     |----|
   *
   *                    ^ ^ ^  ^
   *                    | | |  |         (1)
   *                  seq | |  numBytes
   *                      | |
   *                      | |
   *                    seq numBytes     (2)
   *
   * (1) seq and numBytes are the boundary of some packet
   * (2) seq and numBytes are not the boundary of some packet
   *
   * We can have mixed case (e.g. seq over the boundary while numBytes not).
   *
   * If we discover that we are in (2) or in a mixed case, we split
   * packets accordingly to the requested bounds and re-run the function.
   *
   * In (1), things are pretty easy, it's just a matter of walking the list and
   * defragment packets, if needed (e.g. seq is the beginning of the first packet
   * while maxBytes is the end of some packet next in the list).
   */

  Ptr<Packet> currentPacket = nullptr;
  TcpTxItem *currentItem = nullptr;
  TcpTxItem *outItem = nullptr;
  PacketList::iterator it = list.begin ();
  SequenceNumber32 beginOfCurrentPacket = listStartFrom;

  while (it != list.end ())
    {
      currentItem = *it;
      currentPacket = currentItem->m_packet;
      NS_ASSERT_MSG (list != m_sentList || currentItem->m_startSeq >= m_firstByteSeq,
                     "start: " << m_firstByteSeq << " currentItem start: " <<
                     currentItem->m_startSeq);

      // The objective of this snippet is to find (or to create) the packet
      // that begin with the sequence seq

      if (seq < beginOfCurrentPacket + currentPacket->GetSize ())
        {
          // seq is inside the current packet
          if (seq == beginOfCurrentPacket)
            {
              // seq is the beginning of the current packet. Hurray!
              outItem = currentItem;
              NS_LOG_INFO ("Current packet starts at seq " << seq <<
                           " ends at " << seq + currentPacket->GetSize ());
            }
          else if (seq > beginOfCurrentPacket)
            {
              // seq is inside the current packet but seq is not the beginning,
              // it's somewhere in the middle. Just fragment the beginning and
              // start again.
              NS_LOG_INFO ("we are at " << beginOfCurrentPacket <<
                           " searching for " << seq <<
                           " and now we recurse because packet ends at "
                                        << beginOfCurrentPacket + currentPacket->GetSize ());
              TcpTxItem *firstPart = new TcpTxItem ();
              SplitItems (firstPart, currentItem, seq - beginOfCurrentPacket);

              // insert firstPart before currentItem
              list.insert (it, firstPart);
              if (listEdited)
                {
                  *listEdited = true;
                }

              return GetPacketFromList (list, listStartFrom, numBytes, seq, listEdited);
            }
          else
            {
              NS_FATAL_ERROR ("seq < beginOfCurrentPacket: our data is before");
            }
        }
      else
        {
          // Walk the list, the current packet does not contain seq
          beginOfCurrentPacket += currentPacket->GetSize ();
          it++;
          continue;
        }

      NS_ASSERT (outItem != nullptr);

      // The objective of this snippet is to find (or to create) the packet
      // that ends after numBytes bytes. We are sure that outPacket starts
      // at seq.

      if (seq + numBytes <= beginOfCurrentPacket + currentPacket->GetSize ())
        {
          // the end boundary is inside the current packet
          if (numBytes == currentPacket->GetSize ())
            {
              // the end boundary is exactly the end of the current packet. Hurray!
              if (currentItem->m_packet == outItem->m_packet)
                {
                  // A perfect match!
                  return outItem;
                }
              else
                {
                  // the end is exactly the end of current packet, but
                  // current > outPacket in the list. Merge current with the
                  // previous, and recurse.
                  NS_ASSERT (it != list.begin ());
                  TcpTxItem *previous = *(--it);

                  list.erase (it);

                  MergeItems (previous, currentItem);
                  delete currentItem;
                  if (listEdited)
                    {
                      *listEdited = true;
                    }

                  return GetPacketFromList (list, listStartFrom, numBytes, seq, listEdited);
                }
            }
          else if (numBytes < currentPacket->GetSize ())
            {
              // the end is inside the current packet, but it isn't exactly
              // the packet end. Just fragment, fix the list, and return.
              TcpTxItem *firstPart = new TcpTxItem ();
              SplitItems (firstPart, currentItem, numBytes);

              // insert firstPart before currentItem
              list.insert (it, firstPart);
              if (listEdited)
                {
                  *listEdited = true;
                }

              return firstPart;
            }
        }
      else
        {
          // The end isn't inside current packet, but there is an exception for
          // the merge and recurse strategy...
          if (++it == list.end ())
            {
              // ...current is the last packet we sent. We have not more data;
              // Go for this one.
              NS_LOG_WARN ("Cannot reach the end, but this case is covered "
                           "with conditional statements inside CopyFromSequence."
                           "Something has gone wrong, report a bug");
              return outItem;
            }

          // The current packet does not contain the requested end. Merge current
          // with the packet that follows, and recurse
          TcpTxItem *next = (*it); // Please remember we have incremented it
                                   // in the previous if

          MergeItems (currentItem, next);
          list.erase (it);

          delete next;

          if (listEdited)
            {
              *listEdited = true;
            }

          return GetPacketFromList (list, listStartFrom, numBytes, seq, listEdited);
        }
    }

  NS_FATAL_ERROR ("This point is not reachable");
  return nullptr; // Silence compiler warning about lack of return value
}

void
TcpTxBuffer::MergeItems (TcpTxItem *t1, TcpTxItem *t2) const
{
  NS_ASSERT (t1 != nullptr && t2 != nullptr);
  NS_LOG_FUNCTION (this << *t1 << *t2);
  NS_LOG_INFO ("Merging " << *t2 << " into " << *t1);

  NS_ASSERT_MSG (t1->m_sacked == t2->m_sacked,
                 "Merging one sacked and another not sacked. Impossible");
  NS_ASSERT_MSG (t1->m_lost == t2->m_lost,
                 "Merging one lost and another not lost. Impossible");

  // If one is retrans and the other is not, cancel the retransmitted flag.
  // We are merging this segment for the retransmit, so the count will
  // be updated in MarkTransmittedSegment.
  if (t1->m_retrans != t2->m_retrans)
    {
      if (t1->m_retrans)
        {
          TcpTxBuffer *self = const_cast<TcpTxBuffer*> (this);
          self->m_retrans -= t1->m_packet->GetSize ();
          t1->m_retrans = false;
        }
      else
        {
          NS_ASSERT (t2->m_retrans);
          TcpTxBuffer *self = const_cast<TcpTxBuffer*> (this);
          self->m_retrans -= t2->m_packet->GetSize ();
          t2->m_retrans = false;
        }
    }

  if (t1->m_lastSent < t2->m_lastSent)
    {
      t1->m_lastSent = t2->m_lastSent;
    }

  t1->m_packet->AddAtEnd (t2->m_packet);

  NS_LOG_INFO ("Situation after the merge: " << *t1);
}

void
TcpTxBuffer::RemoveFromCounts (TcpTxItem *item, uint32_t size)
{
  NS_LOG_FUNCTION (this << *item << size);
  if (item->m_sacked)
    {
      NS_ASSERT (m_sackedOut >= size);
      m_sackedOut -= size;
    }
  if (item->m_retrans)
    {
      NS_ASSERT (m_retrans >= size);
      m_retrans -= size;
    }
  if (item->m_lost)
    {
      NS_ASSERT_MSG (m_lostOut >= size, "Trying to remove " << size <<
                     " bytes from " << m_lostOut);
      m_lostOut -= size;
    }
}

bool
TcpTxBuffer::IsRetransmittedDataAcked (const SequenceNumber32& ack) const
{
  NS_LOG_FUNCTION (this);
  for (const auto &it : m_sentList)
     {
       TcpTxItem *item = it;
       Ptr<Packet> p = item->m_packet;
       if (item->m_startSeq + p->GetSize () == ack && !item->m_sacked && item->m_retrans)
         {
           return true;
         }
     }
  return false;
}

void
TcpTxBuffer::DiscardUpTo (const SequenceNumber32& seq,
                          const Callback<void, TcpTxItem *> &beforeDelCb)
{
  NS_LOG_FUNCTION (this << seq);

  // Cases do not need to scan the buffer
  if (m_firstByteSeq >= seq)
    {
      NS_LOG_DEBUG ("Seq " << seq << " already discarded.");
      return;
    }
  NS_LOG_DEBUG ("Remove up to " << seq << " lost: " << m_lostOut <<
                " retrans: " << m_retrans << " sacked: " << m_sackedOut);

  // Scan the buffer and discard packets
  uint32_t offset = seq - m_firstByteSeq.Get ();  // Number of bytes to remove
  uint32_t pktSize;
  PacketList::iterator i = m_sentList.begin ();
  while (m_size > 0 && offset > 0)
    {
      if (i == m_sentList.end ())
        {
          // Move data from app list to sent list, so we can delete the item
          [[maybe_unused]] Ptr<Packet> p = CopyFromSequence (offset, m_firstByteSeq)->GetPacketCopy ();
          NS_ASSERT (p != nullptr);
          i = m_sentList.begin ();
          NS_ASSERT (i != m_sentList.end ());
        }
      TcpTxItem *item = *i;
      Ptr<Packet> p = item->m_packet;
      pktSize = p->GetSize ();
      NS_ASSERT_MSG (item->m_startSeq == m_firstByteSeq,
                     "Item starts at " << item->m_startSeq <<
                     " while SND.UNA is " << m_firstByteSeq << " from " << *this);

      if (offset >= pktSize)
        { // This packet is behind the seqnum. Remove this packet from the buffer
          m_size -= pktSize;
          m_sentSize -= pktSize;
          offset -= pktSize;
          m_firstByteSeq += pktSize;

          RemoveFromCounts (item, pktSize);

          i = m_sentList.erase (i);
          NS_LOG_INFO ("Removed " << *item << " lost: " << m_lostOut <<
                       " retrans: " << m_retrans << " sacked: " << m_sackedOut <<
                       ". Remaining data " << m_size);

          if (!beforeDelCb.IsNull ())
            {
              // Inform Rate algorithms only when a full packet is ACKed
              beforeDelCb (item);
            }

          delete item;
        }
      else if (offset > 0)
        { // Part of the packet is behind the seqnum. Fragment
          pktSize -= offset;
          NS_LOG_INFO (*item);
          // PacketTags are preserved when fragmenting
          item->m_packet = item->m_packet->CreateFragment (offset, pktSize);
          item->m_startSeq += offset;
          m_size -= offset;
          m_sentSize -= offset;
          m_firstByteSeq += offset;

          RemoveFromCounts (item, offset);

          NS_LOG_INFO ("Fragmented one packet by size " << offset <<
                       ", new size=" << pktSize << " resulting item is " <<
                       *item << " status: " << *this);
          break;
        }
    }
  // Catching the case of ACKing a FIN
  if (m_size == 0)
    {
      m_firstByteSeq = seq;
    }

  if (!m_sentList.empty ())
    {
      TcpTxItem *head = m_sentList.front ();
      if (head->m_sacked)
        {
          NS_ASSERT (!head->m_lost);
          // It is not possible to have the UNA sacked; otherwise, it would
          // have been ACKed. This is, most likely, our wrong guessing
          // when adding Reno dupacks in the count.
          head->m_sacked = false;
          m_sackedOut -= head->m_packet->GetSize ();
          NS_LOG_INFO ("Moving the SACK flag from the HEAD to another segment");
          AddRenoSack ();
          MarkHeadAsLost ();
        }

      NS_ASSERT_MSG (head->m_startSeq == seq,
                     "While removing up to " << seq << " we get SND.UNA to " <<
                     m_firstByteSeq << " this is the result: " << *this);
    }

  if (m_highestSack.second <= m_firstByteSeq)
    {
      m_highestSack = std::make_pair (m_sentList.end (), SequenceNumber32 (0));
    }

  NS_LOG_DEBUG ("Discarded up to " << seq << " lost: " << m_lostOut <<
                " retrans: " << m_retrans << " sacked: " << m_sackedOut);
  NS_LOG_LOGIC ("Buffer status after discarding data " << *this);
  NS_ASSERT (m_firstByteSeq >= seq);
  NS_ASSERT (m_sentSize >= m_sackedOut + m_lostOut);
  ConsistencyCheck ();
}

uint32_t
TcpTxBuffer::Update (const TcpOptionSack::SackList &list,
                     const Callback<void, TcpTxItem *> &sackedCb)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_INFO ("Updating scoreboard, got " << list.size () << " blocks to analyze");

  uint32_t bytesSacked = 0;

  for (auto option_it = list.begin (); option_it != list.end (); ++option_it)
    {
      PacketList::iterator item_it = m_sentList.begin ();
      SequenceNumber32 beginOfCurrentPacket = m_firstByteSeq;

      if (m_firstByteSeq + m_sentSize < (*option_it).first)
        {
          NS_LOG_INFO ("Not updating scoreboard, the option block is outside the sent list");
          return bytesSacked;
        }

      while (item_it != m_sentList.end ())
        {
          uint32_t pktSize = (*item_it)->m_packet->GetSize ();

          // Check the boundary of this packet ... only mark as sacked if
          // it is precisely mapped over the option. It means that if the receiver
          // is reporting as sacked single range bytes that are not mapped 1:1
          // in what we have, the option is discarded. There's room for improvement
          // here.
          if (beginOfCurrentPacket >= (*option_it).first
              && beginOfCurrentPacket + pktSize <= (*option_it).second)
            {
              if ((*item_it)->m_sacked)
                {
                  NS_ASSERT (!(*item_it)->m_lost);
                  NS_LOG_INFO ("Received block " << *option_it <<
                               ", checking sentList for block " << *(*item_it) <<
                               ", found in the sackboard already sacked");
                }
              else
                {
                  if ((*item_it)->m_lost)
                    {
                      (*item_it)->m_lost = false;
                      m_lostOut -= (*item_it)->m_packet->GetSize ();
                    }

                  (*item_it)->m_sacked = true;
                  m_sackedOut += (*item_it)->m_packet->GetSize ();
                  bytesSacked += (*item_it)->m_packet->GetSize ();

                  if (m_highestSack.first == m_sentList.end()
                      || m_highestSack.second <= beginOfCurrentPacket + pktSize)
                    {
                      m_highestSack = std::make_pair (item_it, beginOfCurrentPacket);
                    }

                  NS_LOG_INFO ("Received block " << *option_it <<
                               ", checking sentList for block " << *(*item_it) <<
                               ", found in the sackboard, sacking, current highSack: " <<
                               m_highestSack.second);

                  if (!sackedCb.IsNull ())
                    {
                      sackedCb (*item_it);
                    }
                }
            }
          else if (beginOfCurrentPacket + pktSize > (*option_it).second)
            {
              // We already passed the received block end. Exit from the loop
              NS_LOG_INFO ("Received block [" << *option_it <<
                           ", checking sentList for block " << *(*item_it) <<
                           "], not found, breaking loop");
              break;
            }

          beginOfCurrentPacket += pktSize;
          ++item_it;
        }
    }

  if (bytesSacked > 0)
    {
      NS_ASSERT_MSG (m_highestSack.first != m_sentList.end(), "Buffer status: " << *this);
      UpdateLostCount ();
    }

  NS_ASSERT ((*(m_sentList.begin ()))->m_sacked == false);
  NS_ASSERT_MSG (m_sentSize >= m_sackedOut + m_lostOut, *this);
  //NS_ASSERT (list.size () == 0 || modified);   // Assert for duplicated SACK or
                                                 // impossiblity to map the option into the sent blocks
  ConsistencyCheck ();
  return bytesSacked;
}

void
TcpTxBuffer::UpdateLostCount ()
{
  NS_LOG_FUNCTION (this);
  uint32_t sacked = 0;
  SequenceNumber32 beginOfCurrentPacket = m_highestSack.second;
  if (m_highestSack.first == m_sentList.end ())
    {
      NS_LOG_INFO ("Status before the update: " << *this <<
                   ", will start from the latest sent item");
    }
  else
    {
      NS_LOG_INFO ("Status before the update: " << *this <<
                   ", will start from item " << *(*m_highestSack.first));
    }

  for (auto it = m_highestSack.first; it != m_sentList.begin(); --it)
    {
      TcpTxItem *item = *it;
      if (item->m_sacked)
        {
          sacked++;
        }

      if (sacked >= m_dupAckThresh)
        {
          if (!item->m_sacked && !item->m_lost)
            {
              item->m_lost = true;
              m_lostOut += item->m_packet->GetSize ();
            }
        }
      beginOfCurrentPacket -= item->m_packet->GetSize ();
    }

  if (sacked >= m_dupAckThresh)
    {
      TcpTxItem *item = *m_sentList.begin ();
      if (!item->m_lost)
        {
          item->m_lost = true;
          m_lostOut += item->m_packet->GetSize ();
        }
    }
  NS_LOG_INFO ("Status after the update: " << *this);
  ConsistencyCheck ();
}

bool
TcpTxBuffer::IsLost (const SequenceNumber32 &seq) const
{
  NS_LOG_FUNCTION (this << seq);

  SequenceNumber32 beginOfCurrentPacket = m_firstByteSeq;
  PacketList::const_iterator it;

  if (seq >= m_highestSack.second)
    {
      return false;
    }

  // In theory, using a map and hints when inserting elements can improve
  // performance
  for (it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      // Search for the right iterator before calling IsLost()
      if (beginOfCurrentPacket >= seq)
        {
          if ((*it)->m_lost == true)
            {
              NS_LOG_INFO ("seq=" << seq << " is lost because of lost flag");
              return true;
            }

          if ((*it)->m_sacked == true)
            {
              NS_LOG_INFO ("seq=" << seq << " is not lost because of sacked flag");
              return false;
            }
        }

      beginOfCurrentPacket += (*it)->m_packet->GetSize ();
    }

  return false;
}

bool
TcpTxBuffer::NextSeg (SequenceNumber32 *seq, SequenceNumber32 *seqHigh, bool isRecovery) const
{
  NS_LOG_FUNCTION (this << isRecovery);
  /* RFC 6675, NextSeg definition.
   *
   * (1) If there exists a smallest unSACKed sequence number 'S2' that
   *     meets the following three criteria for determining loss, the
   *     sequence range of one segment of up to SMSS octets starting
   *     with S2 MUST be returned.
   *
   *     (1.a) S2 is greater than HighRxt.
   *
   *     (1.b) S2 is less than the highest octet covered by any
   *           received SACK.
   *
   *     (1.c) IsLost (S2) returns true.
   */
  PacketList::const_iterator it;
  TcpTxItem *item;
  SequenceNumber32 seqPerRule3;
  bool isSeqPerRule3Valid = false;
  SequenceNumber32 beginOfCurrentPkt = m_firstByteSeq;

  for (it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      item = *it;

      // Condition 1.a , 1.b , and 1.c
      if (item->m_retrans == false && item->m_sacked == false)
        {
          if (item->m_lost)
            {
              NS_LOG_INFO("IsLost, returning" << beginOfCurrentPkt);
              *seq = beginOfCurrentPkt;
              *seqHigh = *seq + m_segmentSize;
              return true;
            }
          else if (seqPerRule3.GetValue () == 0 && isRecovery)
            {
              NS_LOG_INFO ("Saving for rule 3 the seq " << beginOfCurrentPkt);
              isSeqPerRule3Valid = true;
              seqPerRule3 = beginOfCurrentPkt;
            }
        }

      // Nothing found, iterate
      beginOfCurrentPkt += item->m_packet->GetSize ();
    }

  /* (2) If no sequence number 'S2' per rule (1) exists but there
   *     exists available unsent data and the receiver's advertised
   *     window allows, the sequence range of one segment of up to SMSS
   *     octets of previously unsent data starting with sequence number
   *     HighData+1 MUST be returned.
   */
  if (SizeFromSequence (m_firstByteSeq + m_sentSize) > 0)
    {
      if (m_sentSize <= m_rWndCallback ())
        {
          NS_LOG_INFO ("There is unsent data. Send it");
          *seq = m_firstByteSeq + m_sentSize;
          *seqHigh = *seq + std::min<uint32_t> (m_segmentSize, (m_rWndCallback () - m_sentSize));
          return true;
        }
      else
        {
          NS_LOG_INFO ("There is no available receiver window to send");
          return false;
        }
    }
  else
    {
      NS_LOG_INFO ("There isn't unsent data.");
    }

  /* (3) If the conditions for rules (1) and (2) fail, but there exists
   *     an unSACKed sequence number 'S3' that meets the criteria for
   *     detecting loss given in steps (1.a) and (1.b) above
   *     (specifically excluding step (1.c)), then one segment of up to
   *     SMSS octets starting with S3 SHOULD be returned.
   */
  if (isSeqPerRule3Valid)
    {
      NS_LOG_INFO ("Rule3 valid. " << seqPerRule3);
      *seq = seqPerRule3;
      *seqHigh = *seq + m_segmentSize;
      return true;
    }

  /* (4) If the conditions for (1), (2), and (3) fail, but there exists
   *     outstanding unSACKed data, we provide the opportunity for a
   *     single "rescue" retransmission per entry into loss recovery.
   *     If HighACK is greater than RescueRxt (or RescueRxt is
   *     undefined), then one segment of up to SMSS octets that MUST
   *     include the highest outstanding unSACKed sequence number
   *     SHOULD be returned, and RescueRxt set to RecoveryPoint.
   *     HighRxt MUST NOT be updated.
   *
   * This point require too much interaction between us and TcpSocketBase.
   * We choose to not respect the SHOULD (allowed from RFC MUST/SHOULD definition)
   */
  NS_LOG_INFO ("Can't return anything");
  return false;
}

uint32_t
TcpTxBuffer::BytesInFlight () const
{
  NS_ASSERT_MSG (m_sackedOut + m_lostOut <= m_sentSize,
                 "Count of sacked " << m_sackedOut << " and lost " << m_lostOut <<
                 " is out of sync with sent list size " << m_sentSize <<
                 " " << *this);
  uint32_t leftOut = m_sackedOut + m_lostOut;
  uint32_t retrans = m_retrans;

  NS_LOG_INFO ("Sent size: " << m_sentSize << " leftOut: " << leftOut <<
               " retrans: " << retrans);
  uint32_t in_flight = m_sentSize - leftOut + retrans;

  //uint32_t rfc_in_flight = BytesInFlightRFC (3, segmentSize);
  //NS_ASSERT_MSG(in_flight == rfc_in_flight,
  //              "Calculated: " << in_flight << " RFC: " << rfc_in_flight <<
  //              "Sent size: " << m_sentSize << " leftOut: " << leftOut <<
  //                             " retrans: " << retrans);
  return in_flight;
}

uint32_t
TcpTxBuffer::BytesInFlightRFC () const
{
  PacketList::const_iterator it;
  TcpTxItem *item;
  uint32_t size = 0; // "pipe" in RFC
  SequenceNumber32 beginOfCurrentPkt = m_firstByteSeq;
  uint32_t sackedOut = 0;
  uint32_t lostOut = 0;
  uint32_t retrans = 0;
  uint32_t totalSize = 0;

  // After initializing pipe to zero, the following steps are taken for each
  // octet 'S1' in the sequence space between HighACK and HighData that has not
  // been SACKed:
  for (it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      item = *it;
      totalSize += item->m_packet->GetSize();
      if (!item->m_sacked)
        {
          bool isLost = IsLostRFC (beginOfCurrentPkt, it);
          // (a) If IsLost (S1) returns false: Pipe is incremented by 1 octet.
          if (!isLost)
            {
              size += item->m_packet->GetSize ();
            }
          // (b) If S1 <= HighRxt: Pipe is incremented by 1 octet.
          // (NOTE: we use the m_retrans flag instead of keeping and updating
          // another variable). Only if the item is not marked as lost
          else if (item->m_retrans)
            {
              size += item->m_packet->GetSize ();
            }

          if (isLost)
            {
              lostOut += item->m_packet->GetSize ();
            }
        }
      else
        {
          sackedOut += item->m_packet->GetSize ();
        }

      if (item->m_retrans)
        {
          retrans += item->m_packet->GetSize ();
        }
      beginOfCurrentPkt += item->m_packet->GetSize ();
    }

  NS_ASSERT_MSG(lostOut == m_lostOut, "Lost counted: " << lostOut << " " <<
                m_lostOut << "\n" << *this);
  NS_ASSERT_MSG(retrans == m_retrans, "Retrans Counted: " << retrans << " " <<
                m_retrans << "\n" << *this);
  NS_ASSERT_MSG(sackedOut == m_sackedOut, "Sacked counted: " << sackedOut <<
                " " << m_sackedOut << *this);
  NS_ASSERT_MSG(totalSize == m_sentSize,
                "Sent size counted: " << totalSize << " " << m_sentSize << *this);

  return size;
}

bool
TcpTxBuffer::IsLostRFC (const SequenceNumber32 &seq, const PacketList::const_iterator &segment) const
{
  NS_LOG_FUNCTION (this << seq);
  uint32_t count = 0;
  uint32_t bytes = 0;
  PacketList::const_iterator it;
  TcpTxItem *item;
  Ptr<const Packet> current;
  SequenceNumber32 beginOfCurrentPacket = seq;

  if ((*segment)->m_sacked == true)
    {
      return false;
    }

  // From RFC 6675:
  // > The routine returns true when either dupThresh discontiguous SACKed
  // > sequences have arrived above 'seq' or more than (dupThresh - 1) * SMSS bytes
  // > with sequence numbers greater than 'SeqNum' have been SACKed.  Otherwise, the
  // > routine returns false.
  for (it = segment; it != m_sentList.end (); ++it)
    {
      item = *it;
      current = item->m_packet;

      if (item->m_sacked)
        {
          NS_LOG_INFO ("Segment " << *item <<
                       " found to be SACKed while checking for " << seq);
          ++count;
          bytes += current->GetSize ();
          if ((count >= m_dupAckThresh) || (bytes > (m_dupAckThresh-1) * m_segmentSize))
            {
              NS_LOG_INFO ("seq=" << seq << " is lost because of 3 sacked blocks ahead");
              return true;
            }
        }

      if (beginOfCurrentPacket >= m_highestSack.second)
        {
          if (item->m_lost && !item->m_retrans)
            return true;

          NS_LOG_INFO ("seq=" << seq << " is not lost because there are no sacked segment ahead");
          return false;
        }

      beginOfCurrentPacket += current->GetSize ();
    }
  if (it == m_highestSack.first)
    {
      NS_LOG_INFO ("seq=" << seq << " is not lost because there are no sacked segment ahead " << m_highestSack.second);
    }
  return false;
}

void
TcpTxBuffer::ResetRenoSack ()
{
  NS_LOG_FUNCTION (this);

  m_sackedOut = 0;
  for (auto it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      (*it)->m_sacked = false;
    }

  m_highestSack = std::make_pair (m_sentList.end (), SequenceNumber32 (0));
}

void
TcpTxBuffer::SetRWndCallback (Callback<uint32_t> rWndCallback)
{
  NS_LOG_FUNCTION (this);
  m_rWndCallback = rWndCallback;
}

void
TcpTxBuffer::ResetSentList ()
{
  NS_LOG_FUNCTION (this);
  TcpTxItem *item;

  // Keep the head items; they will then marked as lost
  while (m_sentList.size () > 0)
    {
      item = m_sentList.back ();
      item->m_retrans = item->m_sacked = item->m_lost = false;
      m_appList.push_front (item);
      m_sentList.pop_back ();
    }

  m_sentSize = 0;
  m_lostOut = 0;
  m_retrans = 0;
  m_sackedOut = 0;
  m_highestSack = std::make_pair (m_sentList.end (), SequenceNumber32 (0));
}

void
TcpTxBuffer::ResetLastSegmentSent ()
{
  NS_LOG_FUNCTION (this);
  if (!m_sentList.empty ())
    {
      TcpTxItem *item = m_sentList.back ();

      m_sentList.pop_back ();
      m_sentSize -= item->m_packet->GetSize ();
      if (item->m_retrans)
        {
          m_retrans -= item->m_packet->GetSize ();
        }
      m_appList.insert (m_appList.begin (), item);
    }
  ConsistencyCheck ();
}

void
TcpTxBuffer::SetSentListLost (bool resetSack)
{
  NS_LOG_FUNCTION (this);
  m_retrans = 0;

  if (resetSack)
    {
      m_sackedOut = 0;
      m_lostOut = m_sentSize;
      m_highestSack = std::make_pair (m_sentList.end (), SequenceNumber32 (0));
    }
  else
    {
      m_lostOut = 0;
    }

  for (auto it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      if (resetSack)
        {
          (*it)->m_sacked = false;
          (*it)->m_lost = true;
        }
      else
        {
          if ((*it)->m_lost)
            {
              // Have to increment it because we set it to 0 at line 1133
              m_lostOut += (*it)->m_packet->GetSize ();
            }
          else if (!(*it)->m_sacked)
            {
              // Packet is not marked lost, nor is sacked. Then it becomes lost.
              (*it)->m_lost = true;
              m_lostOut += (*it)->m_packet->GetSize ();
            }
        }

      (*it)->m_retrans = false;
    }

  NS_LOG_INFO ("Set sent list lost, status: " << *this);
  NS_ASSERT_MSG (m_sentSize >= m_sackedOut + m_lostOut, *this);
  ConsistencyCheck ();
}

bool
TcpTxBuffer::IsHeadRetransmitted () const
{
  NS_LOG_FUNCTION (this);

  if (m_sentSize == 0)
    {
      return false;
    }

  return m_sentList.front ()->m_retrans;
}

void
TcpTxBuffer::DeleteRetransmittedFlagFromHead ()
{
  NS_LOG_FUNCTION (this);

  if (m_sentSize == 0)
    {
      return;
    }

  if (m_sentList.front ()->m_retrans)
    {
      m_sentList.front ()->m_retrans = false;
      m_retrans -= m_sentList.front ()->m_packet->GetSize ();
    }
  ConsistencyCheck ();
}

void
TcpTxBuffer::MarkHeadAsLost ()
{
  if (m_sentList.size () > 0)
    {
      // If the head is sacked (reneging by the receiver the previously sent
      // information) we revert the sacked flag.
      // A sacked head means that we should advance SND.UNA.. so it's an error.
      if (m_sentList.front ()->m_sacked)
        {
          m_sentList.front ()->m_sacked = false;
          m_sackedOut -= m_sentList.front ()->m_packet->GetSize ();
        }

      if (m_sentList.front ()->m_retrans)
        {
          m_sentList.front ()->m_retrans = false;
          m_retrans -= m_sentList.front ()->m_packet->GetSize ();
        }

      if (! m_sentList.front()->m_lost)
        {
          m_sentList.front()->m_lost = true;
          m_lostOut += m_sentList.front ()->m_packet->GetSize ();
        }
    }
  ConsistencyCheck ();
}

void
TcpTxBuffer::AddRenoSack (void)
{
  NS_LOG_FUNCTION (this);

  if (m_sackEnabled)
    {
      NS_ASSERT (m_sentList.size () > 1);
    }
  else
    {
      NS_ASSERT (m_sentList.size () > 0);
    }

  m_renoSack = true;

  // We can _never_ SACK the head, so start from the second segment sent
  auto it = ++m_sentList.begin ();

  // Find the "highest sacked" point, that is SND.UNA + m_sackedOut
  while (it != m_sentList.end () && (*it)->m_sacked)
    {
      ++it;
    }

  // Add to the sacked size the size of the first "not sacked" segment
  if (it != m_sentList.end ())
    {
      (*it)->m_sacked = true;
      m_sackedOut += (*it)->m_packet->GetSize ();
      m_highestSack = std::make_pair (it, (*it)->m_startSeq);
      NS_LOG_INFO ("Added a Reno SACK, status: " << *this);
    }
  else
    {
      NS_LOG_INFO ("Can't add a Reno SACK because we miss segments. This dupack"
                   " should be arrived from spurious retransmissions");
    }

  ConsistencyCheck ();
}

void
TcpTxBuffer::ConsistencyCheck () const
{
  static const bool enable = false;

  if (!enable)
    {
      return;
    }

  uint32_t sacked = 0;
  uint32_t lost = 0;
  uint32_t retrans = 0;

  for (auto it = m_sentList.begin (); it != m_sentList.end (); ++it)
    {
      if ((*it)->m_sacked)
        {
          sacked += (*it)->m_packet->GetSize ();
        }
      if ((*it)->m_lost)
        {
          lost += (*it)->m_packet->GetSize ();
        }
      if ((*it)->m_retrans)
        {
          retrans += (*it)->m_packet->GetSize ();
        }
    }

  NS_ASSERT_MSG (sacked == m_sackedOut, "Counted SACK: " << sacked <<
                 " stored SACK: " << m_sackedOut);
  NS_ASSERT_MSG (lost == m_lostOut, " Counted lost: " << lost <<
                 " stored lost: " << m_lostOut);
  NS_ASSERT_MSG (retrans == m_retrans, " Counted retrans: " << retrans <<
                 " stored retrans: " << m_retrans);
}

std::ostream &
operator<< (std::ostream & os, TcpTxItem const & item)
{
  item.Print (os);
  return os;
}

std::ostream &
operator<< (std::ostream & os, TcpTxBuffer const & tcpTxBuf)
{
  TcpTxBuffer::PacketList::const_iterator it;
  std::stringstream ss;
  SequenceNumber32 beginOfCurrentPacket = tcpTxBuf.m_firstByteSeq;
  uint32_t sentSize = 0, appSize = 0;

  Ptr<const Packet> p;
  for (it = tcpTxBuf.m_sentList.begin (); it != tcpTxBuf.m_sentList.end (); ++it)
    {
      p = (*it)->GetPacket ();
      ss << "{";
      (*it)->Print (ss);
      ss << "}";
      sentSize += p->GetSize ();
      beginOfCurrentPacket += p->GetSize ();
    }

  for (it = tcpTxBuf.m_appList.begin (); it != tcpTxBuf.m_appList.end (); ++it)
    {
      appSize += (*it)->GetPacket ()->GetSize ();
    }

  os << "Sent list: " << ss.str () << ", size = " << tcpTxBuf.m_sentList.size () <<
    " Total size: " << tcpTxBuf.m_size <<
    " m_firstByteSeq = " << tcpTxBuf.m_firstByteSeq <<
    " m_sentSize = " << tcpTxBuf.m_sentSize <<
    " m_retransOut = " << tcpTxBuf.m_retrans <<
    " m_lostOut = " << tcpTxBuf.m_lostOut <<
    " m_sackedOut = " << tcpTxBuf.m_sackedOut;

  NS_ASSERT (sentSize == tcpTxBuf.m_sentSize);
  NS_ASSERT (tcpTxBuf.m_size - tcpTxBuf.m_sentSize == appSize);
  return os;
}

} // namespace ns3
