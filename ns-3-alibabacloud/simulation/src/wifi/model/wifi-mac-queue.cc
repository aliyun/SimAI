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
 *          Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/simulator.h"
#include "wifi-mac-queue.h"
#include "qos-blocked-destinations.h"
#include <functional>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiMacQueue");

NS_OBJECT_ENSURE_REGISTERED (WifiMacQueue);
NS_OBJECT_TEMPLATE_CLASS_DEFINE (Queue, WifiMacQueueItem);

TypeId
WifiMacQueue::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiMacQueue")
    .SetParent<Queue<WifiMacQueueItem> > ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiMacQueue> ()
    .AddAttribute ("MaxSize",
                   "The max queue size",
                   QueueSizeValue (QueueSize ("500p")),
                   MakeQueueSizeAccessor (&QueueBase::SetMaxSize,
                                          &QueueBase::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("MaxDelay", "If a packet stays longer than this delay in the queue, it is dropped.",
                   TimeValue (MilliSeconds (500)),
                   MakeTimeAccessor (&WifiMacQueue::SetMaxDelay),
                   MakeTimeChecker ())
    .AddAttribute ("DropPolicy", "Upon enqueue with full queue, drop oldest (DropOldest) or newest (DropNewest) packet",
                   EnumValue (DROP_NEWEST),
                   MakeEnumAccessor (&WifiMacQueue::m_dropPolicy),
                   MakeEnumChecker (WifiMacQueue::DROP_OLDEST, "DropOldest",
                                    WifiMacQueue::DROP_NEWEST, "DropNewest"))
    .AddTraceSource ("Expired", "MPDU dropped because its lifetime expired.",
                     MakeTraceSourceAccessor (&WifiMacQueue::m_traceExpired),
                     "ns3::WifiMacQueueItem::TracedCallback")
  ;
  return tid;
}

WifiMacQueue::WifiMacQueue (AcIndex ac)
  : m_ac (ac),
    NS_LOG_TEMPLATE_DEFINE ("WifiMacQueue")
{
}

WifiMacQueue::~WifiMacQueue ()
{
  NS_LOG_FUNCTION_NOARGS ();
  m_nQueuedPackets.clear ();
  m_nQueuedBytes.clear ();
}

bool
WifiMacQueue::TtlExceeded (ConstIterator &it, const Time& now)
{
  if (now > (*it)->GetTimeStamp () + m_maxDelay)
    {
      NS_LOG_DEBUG ("Removing packet that stayed in the queue for too long (" <<
                    now - (*it)->GetTimeStamp () << ")");
      auto curr = it++;
      m_traceExpired (DoRemove (curr));
      return true;
    }
  return false;
}

bool
WifiMacQueue::TtlExceeded (Ptr<const WifiMacQueueItem> item, const Time& now)
{
  NS_ASSERT (item != nullptr && item->IsQueued ());
  auto it = item->m_queueIt;
  return TtlExceeded (it, now);
}

void
WifiMacQueue::SetMaxDelay (Time delay)
{
  NS_LOG_FUNCTION (this << delay);
  m_maxDelay = delay;
}

Time
WifiMacQueue::GetMaxDelay (void) const
{
  return m_maxDelay;
}

bool
WifiMacQueue::Enqueue (Ptr<WifiMacQueueItem> item)
{
  NS_LOG_FUNCTION (this << *item);

  return Insert (end (), item);
}

bool
WifiMacQueue::PushFront (Ptr<WifiMacQueueItem> item)
{
  NS_LOG_FUNCTION (this << *item);

  return Insert (begin (), item);
}

bool
WifiMacQueue::Insert (ConstIterator pos, Ptr<WifiMacQueueItem> item)
{
  NS_LOG_FUNCTION (this << *item);
  NS_ASSERT_MSG (GetMaxSize ().GetUnit () == QueueSizeUnit::PACKETS,
                 "WifiMacQueues must be in packet mode");

  // insert the item if the queue is not full
  if (QueueBase::GetNPackets () < GetMaxSize ().GetValue ())
    {
      return DoEnqueue (pos, item);
    }

  // the queue is full; scan the list in the attempt to remove stale packets
  ConstIterator it = begin ();
  const Time now = Simulator::Now ();
  while (it != end ())
    {
      if (it == pos && TtlExceeded (it, now))
        {
          return DoEnqueue (it, item);
        }
      if (TtlExceeded (it, now))
        {
          return DoEnqueue (pos, item);
        }
      it++;
    }

  // the queue is still full, remove the oldest item if the policy is drop oldest
  if (m_dropPolicy == DROP_OLDEST)
    {
      NS_LOG_DEBUG ("Remove the oldest item in the queue");
      if (pos == begin ())
        {
          // Avoid invalidating pos
          DoRemove (begin ());
          pos = begin ();
        }
      else
        {
          DoRemove (begin ());
        }
    }

  return DoEnqueue (pos, item);
}

Ptr<WifiMacQueueItem>
WifiMacQueue::Dequeue (void)
{
  NS_LOG_FUNCTION (this);
  const Time now = Simulator::Now ();
  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          return DoDequeue (it);
        }
    }
  NS_LOG_DEBUG ("The queue is empty");
  return 0;
}

void
WifiMacQueue::DequeueIfQueued (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  if (mpdu->IsQueued ())
    {
      NS_ASSERT (mpdu->m_queueAc == m_ac);
      NS_ASSERT (*mpdu->m_queueIt == mpdu);

      DoDequeue (mpdu->m_queueIt);
    }
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::Peek (void) const
{
  NS_LOG_FUNCTION (this);
  const Time now = Simulator::Now ();
  for (auto it = begin (); it != end (); it++)
    {
      // skip packets that stayed in the queue for too long. They will be
      // actually removed from the queue by the next call to a non-const method
      if (now <= (*it)->GetTimeStamp () + m_maxDelay)
        {
          return DoPeek (it);
        }
    }
  NS_LOG_DEBUG ("The queue is empty");
  return 0;
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::PeekByAddress (Mac48Address dest, Ptr<const WifiMacQueueItem> item) const
{
  NS_LOG_FUNCTION (this << dest << item);
  NS_ASSERT (item == nullptr || item->IsQueued ());

  ConstIterator it = (item != nullptr ? std::next (item->m_queueIt) : begin ());
  const Time now = Simulator::Now ();
  while (it != end ())
    {
      // skip packets that stayed in the queue for too long. They will be
      // actually removed from the queue by the next call to a non-const method
      if (now <= (*it)->GetTimeStamp () + m_maxDelay)
        {
          if (((*it)->GetHeader ().IsData () || (*it)->GetHeader ().IsQosData ())
              && (*it)->GetDestinationAddress () == dest)
            {
              return *it;
            }
        }
      it++;
    }
  NS_LOG_DEBUG ("The queue is empty");
  return nullptr;
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::PeekByTid (uint8_t tid, Ptr<const WifiMacQueueItem> item) const
{
  NS_LOG_FUNCTION (this << +tid << item);
  NS_ASSERT (item == nullptr || item->IsQueued ());

  ConstIterator it = (item != nullptr ? std::next (item->m_queueIt) : begin ());
  const Time now = Simulator::Now ();
  while (it != end ())
    {
      // skip packets that stayed in the queue for too long. They will be
      // actually removed from the queue by the next call to a non-const method
      if (now <= (*it)->GetTimeStamp () + m_maxDelay)
        {
          if ((*it)->GetHeader ().IsQosData () && (*it)->GetHeader ().GetQosTid () == tid)
            {
              return *it;
            }
        }
      it++;
    }
  NS_LOG_DEBUG ("The queue is empty");
  return nullptr;
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::PeekByTidAndAddress (uint8_t tid, Mac48Address dest, Ptr<const WifiMacQueueItem> item) const
{
  NS_LOG_FUNCTION (this << +tid << dest << item);
  NS_ASSERT (item == nullptr || item->IsQueued ());

  ConstIterator it = (item != nullptr ? std::next (item->m_queueIt) : begin ());
  const Time now = Simulator::Now ();
  while (it != end ())
    {
      // skip packets that stayed in the queue for too long. They will be
      // actually removed from the queue by the next call to a non-const method
      if (now <= (*it)->GetTimeStamp () + m_maxDelay)
        {
          if ((*it)->GetHeader ().IsQosData () && (*it)->GetDestinationAddress () == dest
              && (*it)->GetHeader ().GetQosTid () == tid)
            {
              return *it;
            }
        }
      it++;
    }
  NS_LOG_DEBUG ("The queue is empty");
  return nullptr;
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::PeekFirstAvailable (const Ptr<QosBlockedDestinations> blockedPackets,
                                  Ptr<const WifiMacQueueItem> item) const
{
  NS_LOG_FUNCTION (this << item);
  NS_ASSERT (item == nullptr || item->IsQueued ());

  ConstIterator it = (item != nullptr ? std::next (item->m_queueIt) : begin ());
  const Time now = Simulator::Now ();
  while (it != end ())
    {
      // skip packets that stayed in the queue for too long. They will be
      // actually removed from the queue by the next call to a non-const method
      if (now <= (*it)->GetTimeStamp () + m_maxDelay)
        {
          if (!(*it)->GetHeader ().IsQosData () || !blockedPackets
              || !blockedPackets->IsBlocked ((*it)->GetHeader ().GetAddr1 (), (*it)->GetHeader ().GetQosTid ()))
            {
              return *it;
            }
        }
      it++;
    }
  NS_LOG_DEBUG ("The queue is empty");
  return nullptr;
}

Ptr<WifiMacQueueItem>
WifiMacQueue::Remove (void)
{
  NS_LOG_FUNCTION (this);

  const Time now = Simulator::Now ();
  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          return DoRemove (it);
        }
    }
  NS_LOG_DEBUG ("The queue is empty");
  return 0;
}

Ptr<const WifiMacQueueItem>
WifiMacQueue::Remove (Ptr<const WifiMacQueueItem> item, bool removeExpired)
{
  NS_LOG_FUNCTION (this << item << removeExpired);
  NS_ASSERT (item != 0 && item->IsQueued ());

  if (!removeExpired)
    {
      ConstIterator next = std::next (item->m_queueIt);
      DoRemove (item->m_queueIt);
      return (next == end () ? nullptr : *next);
    }

  const Time now = Simulator::Now ();

  // remove stale items queued before the given position
  ConstIterator it = begin ();
  while (it != end ())
    {
      if (*it == item)
        {
          ConstIterator next = std::next (item->m_queueIt);
          DoRemove (item->m_queueIt);
          return (next == end () ? nullptr : *next);
        }
      else if (!TtlExceeded (it, now))
        {
          it++;
        }
    }
  NS_LOG_DEBUG ("Invalid iterator");
  return nullptr;
}

void
WifiMacQueue::Replace (Ptr<const WifiMacQueueItem> currentItem, Ptr<WifiMacQueueItem> newItem)
{
  NS_LOG_FUNCTION (this << *currentItem << *newItem);
  NS_ASSERT (currentItem->IsQueued ());
  NS_ASSERT (currentItem->m_queueAc == m_ac);
  NS_ASSERT (*currentItem->m_queueIt == currentItem);
  NS_ASSERT (!newItem->IsQueued ());

  auto pos = std::next (currentItem->m_queueIt);
  DoDequeue (currentItem->m_queueIt);
  bool ret = Insert (pos, newItem);
  // The size of a WifiMacQueue is measured as number of packets. We dequeued
  // one packet, so there is certainly room for inserting one packet
  NS_ABORT_IF (!ret);
}

uint32_t
WifiMacQueue::GetNPacketsByAddress (Mac48Address dest)
{
  NS_LOG_FUNCTION (this << dest);

  uint32_t nPackets = 0;
  const Time now = Simulator::Now ();

  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          if ((*it)->GetHeader ().IsData () && (*it)->GetDestinationAddress () == dest)
            {
              nPackets++;
            }

          it++;
        }
    }
  NS_LOG_DEBUG ("returns " << nPackets);
  return nPackets;
}

uint32_t
WifiMacQueue::GetNPacketsByTidAndAddress (uint8_t tid, Mac48Address dest)
{
  NS_LOG_FUNCTION (this << dest);
  uint32_t nPackets = 0;
  const Time now = Simulator::Now ();

  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          if ((*it)->GetHeader ().IsQosData () && (*it)->GetDestinationAddress () == dest
              && (*it)->GetHeader ().GetQosTid () == tid)
            {
              nPackets++;
            }

          it++;
        }
    }
  NS_LOG_DEBUG ("returns " << nPackets);
  return nPackets;
}

bool
WifiMacQueue::IsEmpty (void)
{
  NS_LOG_FUNCTION (this);
  const Time now = Simulator::Now ();

  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          NS_LOG_DEBUG ("returns false");
          return false;
        }
    }
  NS_LOG_DEBUG ("returns true");
  return true;
}

uint32_t
WifiMacQueue::GetNPackets (void)
{
  NS_LOG_FUNCTION (this);
  const Time now = Simulator::Now ();

  // remove packets that stayed in the queue for too long
  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          it++;
        }
    }
  return QueueBase::GetNPackets ();
}

uint32_t
WifiMacQueue::GetNBytes (void)
{
  NS_LOG_FUNCTION (this);
  const Time now = Simulator::Now ();

  // remove packets that stayed in the queue for too long
  for (ConstIterator it = begin (); it != end (); )
    {
      if (!TtlExceeded (it, now))
        {
          it++;
        }
    }
  return QueueBase::GetNBytes ();
}

uint32_t
WifiMacQueue::GetNPackets (uint8_t tid, Mac48Address dest) const
{
  WifiAddressTidPair addressTidPair (dest, tid);
  auto it = m_nQueuedPackets.find (addressTidPair);
  if (it == m_nQueuedPackets.end ())
    {
      return 0;
    }
  return m_nQueuedPackets.at (addressTidPair);
}

uint32_t
WifiMacQueue::GetNBytes (uint8_t tid, Mac48Address dest) const
{
  WifiAddressTidPair addressTidPair (dest, tid);
  auto it = m_nQueuedBytes.find (addressTidPair);
  if (it == m_nQueuedBytes.end ())
    {
      return 0;
    }
  return m_nQueuedBytes.at (addressTidPair);
}

bool
WifiMacQueue::DoEnqueue (ConstIterator pos, Ptr<WifiMacQueueItem> item)
{
  Iterator ret;
  if (Queue<WifiMacQueueItem>::DoEnqueue (pos, item, ret))
    {
      // update statistics about queued packets
      if (item->GetHeader ().IsQosData ())
        {
          WifiAddressTidPair addressTidPair (item->GetHeader ().GetAddr1 (), item->GetHeader ().GetQosTid ());
          auto it = m_nQueuedPackets.find (addressTidPair);
          if (it == m_nQueuedPackets.end ())
            {
              m_nQueuedPackets[addressTidPair] = 0;
              m_nQueuedBytes[addressTidPair] = 0;
            }
          m_nQueuedPackets[addressTidPair]++;
          m_nQueuedBytes[addressTidPair] += item->GetSize ();
        }
      // set item's information about its position in the queue
      item->m_queueAc = m_ac;
      item->m_queueIt = ret;
      return true;
    }
  return false;
}

Ptr<WifiMacQueueItem>
WifiMacQueue::DoDequeue (ConstIterator pos)
{
  NS_LOG_FUNCTION (this);

  Ptr<WifiMacQueueItem> item = Queue<WifiMacQueueItem>::DoDequeue (pos);

  if (item != 0 && item->GetHeader ().IsQosData ())
    {
      WifiAddressTidPair addressTidPair (item->GetHeader ().GetAddr1 (), item->GetHeader ().GetQosTid ());
      NS_ASSERT (m_nQueuedPackets.find (addressTidPair) != m_nQueuedPackets.end ());
      NS_ASSERT (m_nQueuedPackets[addressTidPair] >= 1);
      NS_ASSERT (m_nQueuedBytes[addressTidPair] >= item->GetSize ());

      m_nQueuedPackets[addressTidPair]--;
      m_nQueuedBytes[addressTidPair] -= item->GetSize ();
    }

  if (item != 0)
    {
      NS_ASSERT (item->IsQueued ());
      item->m_queueAc = AC_UNDEF;
    }

  return item;
}

Ptr<WifiMacQueueItem>
WifiMacQueue::DoRemove (ConstIterator pos)
{
  Ptr<WifiMacQueueItem> item = Queue<WifiMacQueueItem>::DoRemove (pos);

  if (item != 0 && item->GetHeader ().IsQosData ())
    {
      WifiAddressTidPair addressTidPair (item->GetHeader ().GetAddr1 (), item->GetHeader ().GetQosTid ());
      NS_ASSERT (m_nQueuedPackets.find (addressTidPair) != m_nQueuedPackets.end ());
      NS_ASSERT (m_nQueuedPackets[addressTidPair] >= 1);
      NS_ASSERT (m_nQueuedBytes[addressTidPair] >= item->GetSize ());

      m_nQueuedPackets[addressTidPair]--;
      m_nQueuedBytes[addressTidPair] -= item->GetSize ();
    }

  if (item != 0)
    {
      NS_ASSERT (item->IsQueued ());
      item->m_queueAc = AC_UNDEF;
    }

  return item;
}

} //namespace ns3
