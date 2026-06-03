/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Andrew McGregor
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
 * Codel, the COntrolled DELay Queueing discipline
 * Based on ns2 simulation code presented by Kathie Nichols
 *
 * This port based on linux kernel code by
 * Authors:	Dave Täht <d@taht.net>
 *		Eric Dumazet <edumazet@google.com>
 *
 * Ported to ns-3 by: Andrew McGregor <andrewmcgr@gmail.com>
*/

#include "ns3/log.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "ns3/abort.h"
#include "codel-queue-disc.h"
#include "ns3/object-factory.h"
#include "ns3/drop-tail-queue.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CoDelQueueDisc");

/**
 * Performs a reciprocal divide, similar to the
 * Linux kernel reciprocal_divide function
 * \param A numerator
 * \param R reciprocal of the denominator B
 * \return the value of A/B
 */
/* borrowed from the linux kernel */
static inline uint32_t ReciprocalDivide (uint32_t A, uint32_t R)
{
  return (uint32_t)(((uint64_t)A * R) >> 32);
}

/* end kernel borrowings */

/**
 * Returns the current time translated in CoDel time representation
 * \return the current time
 */
static uint32_t CoDelGetTime (void)
{
  Time time = Simulator::Now ();
  uint64_t ns = time.GetNanoSeconds ();

  return static_cast<uint32_t>(ns >> CODEL_SHIFT);
}


NS_OBJECT_ENSURE_REGISTERED (CoDelQueueDisc);

TypeId CoDelQueueDisc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::CoDelQueueDisc")
    .SetParent<QueueDisc> ()
    .SetGroupName ("TrafficControl")
    .AddConstructor<CoDelQueueDisc> ()
    .AddAttribute ("UseEcn",
                   "True to use ECN (packets are marked instead of being dropped)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&CoDelQueueDisc::m_useEcn),
                   MakeBooleanChecker ())
    .AddAttribute ("UseL4s",
                   "True to use L4S (only ECT1 packets are marked at CE threshold)",
                   BooleanValue (false),
                   MakeBooleanAccessor (&CoDelQueueDisc::m_useL4s),
                   MakeBooleanChecker ())
    .AddAttribute ("MaxSize",
                   "The maximum number of packets/bytes accepted by this queue disc.",
                   QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, 1500 * DEFAULT_CODEL_LIMIT)),
                   MakeQueueSizeAccessor (&QueueDisc::SetMaxSize,
                                          &QueueDisc::GetMaxSize),
                   MakeQueueSizeChecker ())
    .AddAttribute ("MinBytes",
                   "The CoDel algorithm minbytes parameter.",
                   UintegerValue (1500),
                   MakeUintegerAccessor (&CoDelQueueDisc::m_minBytes),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Interval",
                   "The CoDel algorithm interval",
                   StringValue ("100ms"),
                   MakeTimeAccessor (&CoDelQueueDisc::m_interval),
                   MakeTimeChecker ())
    .AddAttribute ("Target",
                   "The CoDel algorithm target queue delay",
                   StringValue ("5ms"),
                   MakeTimeAccessor (&CoDelQueueDisc::m_target),
                   MakeTimeChecker ())
    .AddAttribute ("CeThreshold",
                   "The CoDel CE threshold for marking packets",
                   TimeValue (Time::Max ()),
                   MakeTimeAccessor (&CoDelQueueDisc::m_ceThreshold),
                   MakeTimeChecker ())
    .AddTraceSource ("Count",
                     "CoDel count",
                     MakeTraceSourceAccessor (&CoDelQueueDisc::m_count),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("LastCount",
                     "CoDel lastcount",
                     MakeTraceSourceAccessor (&CoDelQueueDisc::m_lastCount),
                     "ns3::TracedValueCallback::Uint32")
    .AddTraceSource ("DropState",
                     "Dropping state",
                     MakeTraceSourceAccessor (&CoDelQueueDisc::m_dropping),
                     "ns3::TracedValueCallback::Bool")
    .AddTraceSource ("DropNext",
                     "Time until next packet drop",
                     MakeTraceSourceAccessor (&CoDelQueueDisc::m_dropNext),
                     "ns3::TracedValueCallback::Uint32")
  ;

  return tid;
}

CoDelQueueDisc::CoDelQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE),
    m_count (0),
    m_lastCount (0),
    m_dropping (false),
    m_recInvSqrt (~0U >> REC_INV_SQRT_SHIFT),
    m_firstAboveTime (0),
    m_dropNext (0)
{
  NS_LOG_FUNCTION (this);
}

CoDelQueueDisc::~CoDelQueueDisc ()
{
  NS_LOG_FUNCTION (this);
}

uint16_t
CoDelQueueDisc::NewtonStep (uint16_t recInvSqrt, uint32_t count)
{
  NS_LOG_FUNCTION_NOARGS ();
  uint32_t invsqrt = ((uint32_t) recInvSqrt) << REC_INV_SQRT_SHIFT;
  uint32_t invsqrt2 = ((uint64_t) invsqrt * invsqrt) >> 32;
  uint64_t val = (3ll << 32) - ((uint64_t) count * invsqrt2);

  val >>= 2; /* avoid overflow */
  val = (val * invsqrt) >> (32 - 2 + 1);
  return static_cast<uint16_t>(val >> REC_INV_SQRT_SHIFT);
}

uint32_t
CoDelQueueDisc::ControlLaw (uint32_t t, uint32_t interval, uint32_t recInvSqrt)
{
  NS_LOG_FUNCTION_NOARGS ();
  return t + ReciprocalDivide (interval, recInvSqrt << REC_INV_SQRT_SHIFT);
}

bool
CoDelQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  NS_LOG_FUNCTION (this << item);

  if (GetCurrentSize () + item > GetMaxSize ())
    {
      NS_LOG_LOGIC ("Queue full -- dropping pkt");
      DropBeforeEnqueue (item, OVERLIMIT_DROP);
      return false;
    }

  bool retval = GetInternalQueue (0)->Enqueue (item);

  // If Queue::Enqueue fails, QueueDisc::DropBeforeEnqueue is called by the
  // internal queue because QueueDisc::AddInternalQueue sets the trace callback

  NS_LOG_LOGIC ("Number packets " << GetInternalQueue (0)->GetNPackets ());
  NS_LOG_LOGIC ("Number bytes " << GetInternalQueue (0)->GetNBytes ());

  return retval;
}

bool
CoDelQueueDisc::OkToDrop (Ptr<QueueDiscItem> item, uint32_t now)
{
  NS_LOG_FUNCTION (this);
  bool okToDrop;

  if (!item)
    {
      m_firstAboveTime = 0;
      return false;
    }

  Time delta = Simulator::Now () - item->GetTimeStamp ();
  NS_LOG_INFO ("Sojourn time " << delta.As (Time::MS));
  uint32_t sojournTime = Time2CoDel (delta);

  if (CoDelTimeBefore (sojournTime, Time2CoDel (m_target))
      || GetInternalQueue (0)->GetNBytes () < m_minBytes)
    {
      // went below so we'll stay below for at least q->interval
      NS_LOG_LOGIC ("Sojourn time is below target or number of bytes in queue is less than minBytes; packet should not be dropped");
      m_firstAboveTime = 0;
      return false;
    }
  okToDrop = false;
  if (m_firstAboveTime == 0)
    {
      /* just went above from below. If we stay above
       * for at least q->interval we'll say it's ok to drop
       */
      NS_LOG_LOGIC ("Sojourn time has just gone above target from below, need to stay above for at least q->interval before packet can be dropped. ");
      m_firstAboveTime = now + Time2CoDel (m_interval);
    }
  else if (CoDelTimeAfter (now, m_firstAboveTime))
    {
      NS_LOG_LOGIC ("Sojourn time has been above target for at least q->interval; it's OK to (possibly) drop packet.");
      okToDrop = true;
    }
  return okToDrop;
}

Ptr<QueueDiscItem>
CoDelQueueDisc::DoDequeue (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<QueueDiscItem> item = GetInternalQueue (0)->Dequeue ();
  if (!item)
    {
      // Leave dropping state when queue is empty
      m_dropping = false;
      NS_LOG_LOGIC ("Queue empty");
      return 0;
    }
  uint32_t ldelay = Time2CoDel (Simulator::Now () - item->GetTimeStamp ());
  if (item && m_useL4s)
    {
      uint8_t tosByte = 0;
      if (item->GetUint8Value (QueueItem::IP_DSFIELD, tosByte) && (((tosByte & 0x3) == 1) || (tosByte & 0x3) == 3))
        {
          if ((tosByte & 0x3) == 1)
            {
              NS_LOG_DEBUG ("ECT1 packet " << static_cast<uint16_t> (tosByte & 0x3));
            }
          else
            {
              NS_LOG_DEBUG ("CE packet " << static_cast<uint16_t> (tosByte & 0x3));
            }

          if (CoDelTimeAfter (ldelay, Time2CoDel (m_ceThreshold)) && Mark (item, CE_THRESHOLD_EXCEEDED_MARK))
            {
              NS_LOG_LOGIC ("Marking due to CeThreshold " << m_ceThreshold.GetSeconds ());
            }
          return item;
        }
    }

  uint32_t now = CoDelGetTime ();

  NS_LOG_LOGIC ("Popped " << item);
  NS_LOG_LOGIC ("Number packets remaining " << GetInternalQueue (0)->GetNPackets ());
  NS_LOG_LOGIC ("Number bytes remaining " << GetInternalQueue (0)->GetNBytes ());

  // Determine if item should be dropped
  bool okToDrop = OkToDrop (item, now);
  bool isMarked = false;

  if (m_dropping)
    { // In the dropping state (sojourn time has gone above target and hasn't come down yet)
      // Check if we can leave the dropping state or next drop should occur
      NS_LOG_LOGIC ("In dropping state, check if it's OK to leave or next drop should occur");
      if (!okToDrop)
        {
          /* sojourn time fell below target - leave dropping state */
          NS_LOG_LOGIC ("Sojourn time goes below target, it's OK to leave dropping state.");
          m_dropping = false;
        }
      else if (CoDelTimeAfterEq (now, m_dropNext))
        {
          while (m_dropping && CoDelTimeAfterEq (now, m_dropNext))
            {
              ++m_count;
              m_recInvSqrt = NewtonStep (m_recInvSqrt, m_count);
              // It's time for the next drop. Drop the current packet and
              // dequeue the next. The dequeue might take us out of dropping
              // state. If not, schedule the next drop.
              // A large amount of packets in queue might result in drop
              // rates so high that the next drop should happen now,
              // hence the while loop.
              if (m_useEcn && Mark (item, TARGET_EXCEEDED_MARK))
                {
                  isMarked = true;
                  NS_LOG_LOGIC ("Sojourn time is still above target and it's time for next drop or mark; marking " << item);
                  NS_LOG_LOGIC ("Running ControlLaw for input m_dropNext: " << (double)m_dropNext / 1000000);
                  m_dropNext = ControlLaw (now, Time2CoDel (m_interval), m_recInvSqrt);
                  NS_LOG_LOGIC ("Scheduled next drop at " << (double) m_dropNext / 1000000);
                  goto end;
                }
              NS_LOG_LOGIC ("Sojourn time is still above target and it's time for next drop; dropping " << item);
              DropAfterDequeue (item, TARGET_EXCEEDED_DROP);

              item = GetInternalQueue (0)->Dequeue ();

              if (item)
                {
                  NS_LOG_LOGIC ("Popped " << item);
                  NS_LOG_LOGIC ("Number packets remaining " << GetInternalQueue (0)->GetNPackets ());
                  NS_LOG_LOGIC ("Number bytes remaining " << GetInternalQueue (0)->GetNBytes ());
                }

              if (!OkToDrop (item, now))
                {
                  /* leave dropping state */
                  NS_LOG_LOGIC ("Leaving dropping state");
                  m_dropping = false;
                }
              else
                {
                  /* schedule the next drop */
                  NS_LOG_LOGIC ("Running ControlLaw for input m_dropNext: " << (double)m_dropNext / 1000000);
                  m_dropNext = ControlLaw (m_dropNext, Time2CoDel (m_interval), m_recInvSqrt);
                  NS_LOG_LOGIC ("Scheduled next drop at " << (double)m_dropNext / 1000000);
                }
            }
        }
    }
  else
    {
      // Not in the dropping state
      // Decide if we have to enter the dropping state and drop the first packet
      NS_LOG_LOGIC ("Not in dropping state; decide if we have to enter the state and drop the first packet");
      if (okToDrop)
        {
          if (m_useEcn && Mark (item, TARGET_EXCEEDED_MARK))
            {
              isMarked = true;
              NS_LOG_LOGIC ("Sojourn time goes above target, marking the first packet " << item << " and entering the dropping state");
            }
          else
            {
              // Drop the first packet and enter dropping state unless the queue is empty
              NS_LOG_LOGIC ("Sojourn time goes above target, dropping the first packet " << item << " and entering the dropping state");
              DropAfterDequeue (item, TARGET_EXCEEDED_DROP);
              item = GetInternalQueue (0)->Dequeue ();
              if (item)
                {
                  NS_LOG_LOGIC ("Popped " << item);
                  NS_LOG_LOGIC ("Number packets remaining " << GetInternalQueue (0)->GetNPackets ());
                  NS_LOG_LOGIC ("Number bytes remaining " << GetInternalQueue (0)->GetNBytes ());
                }
              OkToDrop (item, now);
            }
          m_dropping = true;
          /*
           * if min went above target close to when we last went below it
           * assume that the drop rate that controlled the queue on the
           * last cycle is a good starting point to control it now.
           */
          int delta = m_count - m_lastCount;
          if (delta > 1 && CoDelTimeBefore (now - m_dropNext, 16 * Time2CoDel (m_interval)))
            {
              m_count = delta;
              m_recInvSqrt = NewtonStep (m_recInvSqrt, m_count);
            }
          else
            {
              m_count = 1;
              m_recInvSqrt = ~0U >> REC_INV_SQRT_SHIFT;
            }
          m_lastCount = m_count;
          NS_LOG_LOGIC ("Running ControlLaw for input now: " << (double)now);
          m_dropNext = ControlLaw (now, Time2CoDel (m_interval), m_recInvSqrt);
          NS_LOG_LOGIC ("Scheduled next drop at " << (double)m_dropNext / 1000000 << " now " << (double)now / 1000000);
        }
    }
  end:
  ldelay = Time2CoDel (Simulator::Now () - item->GetTimeStamp ());
  // In Linux, this branch of code is executed even if the packet has been marked
  // according to the target delay above. If the ns-3 code were to do the same here,
  // it would result in two counts of mark in the queue statistics. Therefore, we
  // use the isMarked flag to suppress a second attempt at marking.
  if (!isMarked && item && !m_useL4s && m_useEcn && CoDelTimeAfter (ldelay, Time2CoDel (m_ceThreshold)) && Mark (item, CE_THRESHOLD_EXCEEDED_MARK))
    {
      NS_LOG_LOGIC ("Marking due to CeThreshold " << m_ceThreshold.GetSeconds ());
    }
  return item;
}

Time
CoDelQueueDisc::GetTarget (void)
{
  return m_target;
}

Time
CoDelQueueDisc::GetInterval (void)
{
  return m_interval;
}

uint32_t
CoDelQueueDisc::GetDropNext (void)
{
  return m_dropNext;
}

bool
CoDelQueueDisc::CoDelTimeAfter (uint32_t a, uint32_t b)
{
  return  ((int64_t)(a) - (int64_t)(b) > 0);
}

bool
CoDelQueueDisc::CoDelTimeAfterEq (uint32_t a, uint32_t b)
{
  return ((int64_t)(a) - (int64_t)(b) >= 0);
}

bool
CoDelQueueDisc::CoDelTimeBefore (uint32_t a, uint32_t b)
{
  return  ((int64_t)(a) - (int64_t)(b) < 0);
}

bool
CoDelQueueDisc::CoDelTimeBeforeEq (uint32_t a, uint32_t b)
{
  return ((int64_t)(a) - (int64_t)(b) <= 0);
}

uint32_t
CoDelQueueDisc::Time2CoDel (Time t)
{
  return static_cast<uint32_t>(t.GetNanoSeconds () >> CODEL_SHIFT);
}

bool
CoDelQueueDisc::CheckConfig (void)
{
  NS_LOG_FUNCTION (this);
  if (GetNQueueDiscClasses () > 0)
    {
      NS_LOG_ERROR ("CoDelQueueDisc cannot have classes");
      return false;
    }

  if (GetNPacketFilters () > 0)
    {
      NS_LOG_ERROR ("CoDelQueueDisc cannot have packet filters");
      return false;
    }

  if (GetNInternalQueues () == 0)
    {
      // add a DropTail queue
      AddInternalQueue (CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> >
                          ("MaxSize", QueueSizeValue (GetMaxSize ())));
    }

  if (GetNInternalQueues () != 1)
    {
      NS_LOG_ERROR ("CoDelQueueDisc needs 1 internal queue");
      return false;
    }

  return true;
}

void
CoDelQueueDisc::InitializeParams (void)
{
  NS_LOG_FUNCTION (this);
}

} // namespace ns3

