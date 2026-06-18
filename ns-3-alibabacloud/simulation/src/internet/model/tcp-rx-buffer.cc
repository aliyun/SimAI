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

#include "ns3/packet.h"
#include "ns3/log.h"
#include "tcp-rx-buffer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpRxBuffer");

NS_OBJECT_ENSURE_REGISTERED (TcpRxBuffer);

TypeId
TcpRxBuffer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpRxBuffer")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpRxBuffer> ()
    .AddTraceSource ("NextRxSequence",
                     "Next sequence number expected (RCV.NXT)",
                     MakeTraceSourceAccessor (&TcpRxBuffer::m_nextRxSeq),
                     "ns3::SequenceNumber32TracedValueCallback")
  ;
  return tid;
}

/* A user is supposed to create a TcpSocket through a factory. In TcpSocket,
 * there are attributes SndBufSize and RcvBufSize to control the default Tx and
 * Rx window sizes respectively, with default of 128 KiByte. The attribute
 * RcvBufSize is passed to TcpRxBuffer by TcpSocketBase::SetRcvBufSize() and in
 * turn, TcpRxBuffer:SetMaxBufferSize(). Therefore, the m_maxBuffer value
 * initialized below is insignificant.
 */
TcpRxBuffer::TcpRxBuffer (uint32_t n)
  : m_nextRxSeq (n), m_gotFin (false), m_size (0), m_maxBuffer (32768), m_availBytes (0)
{
}

TcpRxBuffer::~TcpRxBuffer ()
{
}

SequenceNumber32
TcpRxBuffer::NextRxSequence (void) const
{
  return m_nextRxSeq;
}

void
TcpRxBuffer::SetNextRxSequence (const SequenceNumber32& s)
{
  m_nextRxSeq = s;
}

uint32_t
TcpRxBuffer::MaxBufferSize (void) const
{
  return m_maxBuffer;
}

void
TcpRxBuffer::SetMaxBufferSize (uint32_t s)
{
  m_maxBuffer = s;
}

uint32_t
TcpRxBuffer::Size (void) const
{
  return m_size;
}

uint32_t
TcpRxBuffer::Available () const
{
  return m_availBytes;
}

void
TcpRxBuffer::IncNextRxSequence ()
{
  NS_LOG_FUNCTION (this);
  // Increment nextRxSeq is valid only if we don't have any data buffered,
  // this is supposed to be called only during the three-way handshake
  NS_ASSERT (m_size == 0);
  m_nextRxSeq++;
}

// Return the lowest sequence number that this TcpRxBuffer cannot accept
SequenceNumber32
TcpRxBuffer::MaxRxSequence (void) const
{
  if (m_gotFin)
    { // No data allowed beyond FIN
      return m_finSeq;
    }
  else if (m_data.size () && m_nextRxSeq > m_data.begin ()->first)
    { // No data allowed beyond Rx window allowed
      return m_data.begin ()->first + SequenceNumber32 (m_maxBuffer);
    }
  return m_nextRxSeq + SequenceNumber32 (m_maxBuffer);
}

void
TcpRxBuffer::SetFinSequence (const SequenceNumber32& s)
{
  NS_LOG_FUNCTION (this);

  m_gotFin = true;
  m_finSeq = s;
  if (m_nextRxSeq == m_finSeq) ++m_nextRxSeq;
}

bool
TcpRxBuffer::Finished (void)
{
  return (m_gotFin && m_finSeq < m_nextRxSeq);
}

bool
TcpRxBuffer::Add (Ptr<Packet> p, TcpHeader const& tcph)
{
  NS_LOG_FUNCTION (this << p << tcph);

  uint32_t pktSize = p->GetSize ();
  SequenceNumber32 headSeq = tcph.GetSequenceNumber ();
  SequenceNumber32 tailSeq = headSeq + SequenceNumber32 (pktSize);
  NS_LOG_LOGIC ("Add pkt " << p << " len=" << pktSize << " seq=" << headSeq
                           << ", when NextRxSeq=" << m_nextRxSeq << ", buffsize=" << m_size);

  // Trim packet to fit Rx window specification
  if (headSeq < m_nextRxSeq) headSeq = m_nextRxSeq;
  if (m_data.size ())
    {
      SequenceNumber32 maxSeq = m_data.begin ()->first + SequenceNumber32 (m_maxBuffer);
      if (maxSeq < tailSeq) tailSeq = maxSeq;
      if (tailSeq < headSeq) headSeq = tailSeq;
    }
  // Remove overlapped bytes from packet
  BufIterator i = m_data.begin ();
  while (i != m_data.end () && i->first <= tailSeq)
    {
      SequenceNumber32 lastByteSeq = i->first + SequenceNumber32 (i->second->GetSize ());
      if (lastByteSeq > headSeq)
        {
          if (i->first > headSeq && lastByteSeq < tailSeq)
            { // Rare case: Existing packet is embedded fully in the new packet
              m_size -= i->second->GetSize ();
              m_data.erase (i++);
              continue;
            }
          if (i->first <= headSeq)
            { // Incoming head is overlapped
              headSeq = lastByteSeq;
            }
          if (lastByteSeq >= tailSeq)
            { // Incoming tail is overlapped
              tailSeq = i->first;
            }
        }
      ++i;
    }
  // We now know how much we are going to store, trim the packet
  if (headSeq >= tailSeq)
    {
      NS_LOG_LOGIC ("Nothing to buffer");
      return false; // Nothing to buffer anyway
    }
  else
    {
      uint32_t start = static_cast<uint32_t> (headSeq - tcph.GetSequenceNumber ());
      uint32_t length = static_cast<uint32_t> (tailSeq - headSeq);
      p = p->CreateFragment (start, length);
      NS_ASSERT (length == p->GetSize ());
    }
  // Insert packet into buffer
  NS_ASSERT (m_data.find (headSeq) == m_data.end ()); // Shouldn't be there yet
  m_data [ headSeq ] = p;

  if (headSeq > m_nextRxSeq)
    {
      // Generate a new SACK block
      UpdateSackList (headSeq, tailSeq);
    }

  NS_LOG_LOGIC ("Buffered packet of seqno=" << headSeq << " len=" << p->GetSize ());
  // Update variables
  m_size += p->GetSize ();      // Occupancy
  for (i = m_data.begin (); i != m_data.end (); ++i)
    {
      if (i->first < m_nextRxSeq)
        {
          continue;
        }
      else if (i->first > m_nextRxSeq)
        {
          break;
        };
      m_nextRxSeq = i->first + SequenceNumber32 (i->second->GetSize ());
      m_availBytes += i->second->GetSize ();
      ClearSackList (m_nextRxSeq);
    }
  NS_LOG_LOGIC ("Updated buffer occupancy=" << m_size << " nextRxSeq=" << m_nextRxSeq);
  if (m_gotFin && m_nextRxSeq == m_finSeq)
    { // Account for the FIN packet
      ++m_nextRxSeq;
    };
  return true;
}

uint32_t
TcpRxBuffer::GetSackListSize () const
{
  NS_LOG_FUNCTION (this);

  return static_cast<uint32_t> (m_sackList.size ());
}

void
TcpRxBuffer::UpdateSackList (const SequenceNumber32 &head, const SequenceNumber32 &tail)
{
  NS_LOG_FUNCTION (this << head << tail);
  NS_ASSERT (head > m_nextRxSeq);

  TcpOptionSack::SackBlock current;
  current.first = head;
  current.second = tail;

  // The block "current" has been safely stored. Now we need to build the SACK
  // list, to be advertised. From RFC 2018:
  // (a) The first SACK block (i.e., the one immediately following the
  //     kind and length fields in the option) MUST specify the contiguous
  //     block of data containing the segment which triggered this ACK,
  //     unless that segment advanced the Acknowledgment Number field in
  //     the header.  This assures that the ACK with the SACK option
  //     reflects the most recent change in the data receiver's buffer
  //     queue.
  //
  // (b) The data receiver SHOULD include as many distinct SACK blocks as
  //     possible in the SACK option.  Note that the maximum available
  //     option space may not be sufficient to report all blocks present in
  //     the receiver's queue.
  //
  // (c) The SACK option SHOULD be filled out by repeating the most
  //     recently reported SACK blocks (based on first SACK blocks in
  //     previous SACK options) that are not subsets of a SACK block
  //     already included in the SACK option being constructed.  This
  //     assures that in normal operation, any segment remaining part of a
  //     non-contiguous block of data held by the data receiver is reported
  //     in at least three successive SACK options, even for large-window
  //     TCP implementations [RFC1323]).  After the first SACK block, the
  //     following SACK blocks in the SACK option may be listed in
  //     arbitrary order.

  m_sackList.push_front (current);

  // We have inserted the block at the beginning of the list. Now, we should
  // check if any existing blocks overlap with that.
  bool updated = false;
  TcpOptionSack::SackList::iterator it = m_sackList.begin ();
  TcpOptionSack::SackBlock begin = *it;
  TcpOptionSack::SackBlock merged;
  ++it;

  // Iterates until we examined all blocks in the list (maximum 4)
  while (it != m_sackList.end ())
    {
      current = *it;

      // This is a left merge:
      // [current_first; current_second] [beg_first; beg_second]
      if (begin.first == current.second)
        {
          NS_ASSERT (current.first < begin.second);
          merged = TcpOptionSack::SackBlock (current.first, begin.second);
          updated = true;
        }
      // while this is a right merge
      // [begin_first; begin_second] [current_first; current_second]
      else if (begin.second == current.first)
        {
          NS_ASSERT (begin.first < current.second);
          merged = TcpOptionSack::SackBlock (begin.first, current.second);
          updated = true;
        }

      // If we have merged the blocks (and the result is in merged) we should
      // delete the current block (it), the first block, and insert the merged
      // one at the beginning.
      if (updated)
        {
          m_sackList.erase (it);
          m_sackList.pop_front ();
          m_sackList.push_front (merged);
          it = m_sackList.begin ();
          begin = *it;
          updated = false;
        }

      ++it;
    }

  // Since the maximum blocks that fits into a TCP header are 4, there's no
  // point on maintaining the others.
  if (m_sackList.size () > 4)
    {
      m_sackList.pop_back ();
    }

  // Please note that, if a block b is discarded and then a block contiguous
  // to b is received, only that new block (without the b part) is reported.
  // This is perfectly fine for the RFC point (a), given that we do not report any
  // overlapping blocks shortly after.
}

void
TcpRxBuffer::ClearSackList (const SequenceNumber32 &seq)
{
  NS_LOG_FUNCTION (this << seq);

  TcpOptionSack::SackList::iterator it;
  for (it = m_sackList.begin (); it != m_sackList.end (); )
    {
      TcpOptionSack::SackBlock block = *it;
      NS_ASSERT (block.first < block.second);

      if (block.second <= seq)
        {
          it = m_sackList.erase (it);
        }
      else
        {
          it++;
        }
    }
}

TcpOptionSack::SackList
TcpRxBuffer::GetSackList () const
{
  return m_sackList;
}

Ptr<Packet>
TcpRxBuffer::Extract (uint32_t maxSize)
{
  NS_LOG_FUNCTION (this << maxSize);

  uint32_t extractSize = std::min (maxSize, m_availBytes);
  NS_LOG_LOGIC ("Requested to extract " << extractSize << " bytes from TcpRxBuffer of size=" << m_size);
  if (extractSize == 0) return nullptr;  // No contiguous block to return
  NS_ASSERT (m_data.size ()); // At least we have something to extract
  Ptr<Packet> outPkt = Create<Packet> (); // The packet that contains all the data to return
  BufIterator i;
  while (extractSize)
    { // Check the buffered data for delivery
      i = m_data.begin ();
      NS_ASSERT (i->first <= m_nextRxSeq); // in-sequence data expected
      // Check if we send the whole pkt or just a partial
      uint32_t pktSize = i->second->GetSize ();
      if (pktSize <= extractSize)
        { // Whole packet is extracted
          outPkt->AddAtEnd (i->second);
          m_data.erase (i);
          m_size -= pktSize;
          m_availBytes -= pktSize;
          extractSize -= pktSize;
        }
      else
        { // Partial is extracted and done
          outPkt->AddAtEnd (i->second->CreateFragment (0, extractSize));
          m_data[i->first + SequenceNumber32 (extractSize)] = i->second->CreateFragment (extractSize, pktSize - extractSize);
          m_data.erase (i);
          m_size -= extractSize;
          m_availBytes -= extractSize;
          extractSize = 0;
        }
    }
  if (outPkt->GetSize () == 0)
    {
      NS_LOG_LOGIC ("Nothing extracted.");
      return nullptr;
    }
  NS_LOG_LOGIC ("Extracted " << outPkt->GetSize ( ) << " bytes, bufsize=" << m_size
                             << ", num pkts in buffer=" << m_data.size ());
  return outPkt;
}

} //namespace ns3
