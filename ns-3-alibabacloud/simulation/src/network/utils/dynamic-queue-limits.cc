/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
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
 * Authors: Pasquale Imputato <p.imputato@gmail.com>
 *          Stefano Avallone <stefano.avallone@unina.it>
 *
 * This code is a port of the dynamic queue limits library implemented
 * in the Linux kernel by
 * Author: Tom Herbert <therbert@google.com>
 */

#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "dynamic-queue-limits.h"

// Set some static maximums
static const uint32_t UINTMAX = std::numeric_limits<uint32_t>::max ();
static const uint32_t DQL_MAX_OBJECT = UINTMAX / 16;
static const uint32_t DQL_MAX_LIMIT = (UINTMAX / 2) - DQL_MAX_OBJECT;

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DynamicQueueLimits");

NS_OBJECT_ENSURE_REGISTERED (DynamicQueueLimits);

TypeId
DynamicQueueLimits::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DynamicQueueLimits")
    .SetParent<Object> ()
    .SetParent<QueueLimits> ()
    .SetGroupName ("Network")
    .AddConstructor<DynamicQueueLimits> ()
    .AddAttribute ("HoldTime",
                   "The DQL algorithm hold time",
                   StringValue ("1s"),
                   MakeTimeAccessor (&DynamicQueueLimits::m_slackHoldTime),
                   MakeTimeChecker ())
    .AddAttribute ("MaxLimit",
                   "Maximum limit",
                   UintegerValue (DQL_MAX_LIMIT),
                   MakeUintegerAccessor (&DynamicQueueLimits::m_maxLimit),
                   MakeUintegerChecker<uint32_t> (0, DQL_MAX_LIMIT))
    .AddAttribute ("MinLimit",
                   "Minimum limit",
                   UintegerValue (0),
                   MakeUintegerAccessor (&DynamicQueueLimits::m_minLimit),
                   MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("Limit",
                     "Limit value calculated by DQL",
                     MakeTraceSourceAccessor (&DynamicQueueLimits::m_limit),
                     "ns3::TracedValueCallback::Uint32")
  ;
  return tid;
}

DynamicQueueLimits::DynamicQueueLimits ()
{
  NS_LOG_FUNCTION (this);
  Reset ();
}

DynamicQueueLimits::~DynamicQueueLimits ()
{
  NS_LOG_FUNCTION (this);
}

void
DynamicQueueLimits::Reset ()
{
  NS_LOG_FUNCTION (this);
  // Reset all dynamic values
  m_limit = 0;
  m_numQueued = 0;
  m_numCompleted = 0;
  m_lastObjCnt = 0;
  m_prevNumQueued = 0;
  m_prevLastObjCnt = 0;
  m_prevOvlimit = 0;
  m_lowestSlack = UINTMAX;
  m_slackStartTime = Simulator::Now ();
}

void
DynamicQueueLimits::Completed (uint32_t count)
{
  NS_LOG_FUNCTION (this << count);
  uint32_t inprogress, prevInprogress, limit;
  uint32_t ovlimit, completed, numQueued;
  bool allPrevCompleted;

  numQueued = m_numQueued;

  // Can't complete more than what's in queue
  NS_ASSERT (count <= numQueued - m_numCompleted);

  completed = m_numCompleted + count;
  limit = m_limit;
  ovlimit = Posdiff (numQueued - m_numCompleted, limit);
  inprogress = numQueued - completed;
  prevInprogress = m_prevNumQueued - m_numCompleted;
  allPrevCompleted = ((int32_t)(completed - m_prevNumQueued)) >= 0;

  if ((ovlimit && !inprogress) || (m_prevOvlimit && allPrevCompleted))
    {
      NS_LOG_DEBUG ("Queue starved, increase limit");
     /*
      * Queue considered starved if:
      *   - The queue was over-limit in the last interval,
      *     and there is no more data in the queue.
      *  OR
      *   - The queue was over-limit in the previous interval and
      *     when enqueuing it was possible that all queued data
      *     had been consumed.  This covers the case when queue
      *     may have becomes starved between completion processing
      *     running and next time enqueue was scheduled.
      *
      *     When queue is starved increase the limit by the amount
      *     of bytes both sent and completed in the last interval,
      *     plus any previous over-limit.
      */
      limit += Posdiff (completed, m_prevNumQueued) + m_prevOvlimit;
      m_slackStartTime = Simulator::Now ();
      m_lowestSlack = UINTMAX;
    }
  else if (inprogress && prevInprogress && !allPrevCompleted)
    {
      NS_LOG_DEBUG ("Queue not starved, check decrease limit");
     /*
      * Queue was not starved, check if the limit can be decreased.
      * A decrease is only considered if the queue has been busy in
      * the whole interval (the check above).
      *
      * If there is slack, the amount of execess data queued above
      * the the amount needed to prevent starvation, the queue limit
      * can be decreased.  To avoid hysteresis we consider the
      * minimum amount of slack found over several iterations of the
      * completion routine.
      */
      uint32_t slack, slackLastObjs;

     /*
      * Slack is the maximum of
      *   - The queue limit plus previous over-limit minus twice
      *     the number of objects completed.  Note that two times
      *     number of completed bytes is a basis for an upper bound
      *     of the limit.
      *   - Portion of objects in the last queuing operation that
      *     was not part of non-zero previous over-limit.  That is
      *     "round down" by non-overlimit portion of the last
      *     queueing operation.
      */
      slack = Posdiff (limit + m_prevOvlimit, 2 * (completed - m_numCompleted));
      slackLastObjs = m_prevOvlimit ? Posdiff (m_prevLastObjCnt, m_prevOvlimit) : 0;

      slack = std::max (slack, slackLastObjs);

      if (slack < m_lowestSlack)
        {
          m_lowestSlack = slack;
        }

      if (Simulator::Now () > (m_slackStartTime + m_slackHoldTime))
        {
          limit = Posdiff (limit, m_lowestSlack);
          m_slackStartTime = Simulator::Now ();
          m_lowestSlack = UINTMAX;
        }
    }

  // Enforce bounds on limit
  limit = std::min ((uint32_t)std::max (limit, m_minLimit), m_maxLimit);

  if (limit != m_limit)
    {
      NS_LOG_DEBUG ("Update limit");
      m_limit = limit;
      ovlimit = 0;
    }

  m_adjLimit = limit + completed;
  m_prevOvlimit = ovlimit;
  m_prevLastObjCnt = m_lastObjCnt;
  m_numCompleted = completed;
  m_prevNumQueued = numQueued;
}

int32_t
DynamicQueueLimits::Available () const
{
  NS_LOG_FUNCTION (this);
  return (m_adjLimit - m_numQueued);
}

void
DynamicQueueLimits::Queued (uint32_t count)
{
  NS_LOG_FUNCTION (this << count);
  NS_ASSERT (count <= DQL_MAX_OBJECT);

  m_lastObjCnt = count;
  m_numQueued += count;
}

int32_t
DynamicQueueLimits::Posdiff (int32_t a, int32_t b)
{
  NS_LOG_FUNCTION (this << a << b);
  return std::max ((a - b), 0);
}

} // namespace ns3
