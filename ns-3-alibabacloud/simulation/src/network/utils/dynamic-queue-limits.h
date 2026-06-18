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

#ifndef DYNAMIC_QUEUE_LIMITS_H
#define DYNAMIC_QUEUE_LIMITS_H

#include "queue-limits.h"
#include "ns3/nstime.h"
#include "ns3/traced-value.h"
#include <limits.h>

namespace ns3 {

/**
 * \ingroup network
 *
 * DynamicQueueLimits would be used in conjunction with a producer/consumer
 * type queue (possibly a netdevice queue).
 * Such a queue would have these general properties:
 *
 *   1) Objects are queued up to some limit specified as number of objects.
 *   2) Periodically a completion process executes which retires consumed
 *      objects.
 *   3) Starvation occurs when limit has been reached, all queued data has
 *      actually been consumed, but completion processing has not yet run
 *      so queuing new data is blocked.
 *   4) Minimizing the amount of queued data is desirable.
 *
 * The goal of DynamicQueueLimits is to calculate the limit as the minimum
 * number of objects needed to prevent starvation.
 *
 * The primary functions of DynamicQueueLimits are:
 *   Completed - called at completion time to indicate how many objects
 *               were retired from the queue
 *   Available - returns how many objects are available to be queued based
 *           on the object limit and how many objects are already enqueued
 *   Queued - called when objects are enqueued to record number of objects
 *
 */

class DynamicQueueLimits : public QueueLimits {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  DynamicQueueLimits ();
  virtual ~DynamicQueueLimits ();

  virtual void Reset ();
  virtual void Completed (uint32_t count);
  virtual int32_t Available () const;
  virtual void Queued (uint32_t count);

private:
  /**
   * Calculates the difference between the two operators and
   * returns the number if positive, zero otherwise.
   * \param a First operator.
   * \param b Second operator.
   * \returns the difference between a and b if positive, zero otherwise.
   */
  int32_t Posdiff (int32_t a, int32_t b);

  // Fields accessed in enqueue path
  uint32_t m_numQueued {0};                 //!< Total ever queued
  uint32_t m_adjLimit {0};                  //!< limit + num_completed
  uint32_t m_lastObjCnt {0};                //!< Count at last queuing

  // Fields accessed only by completion path
  TracedValue<uint32_t> m_limit;            //!< Current limit
  uint32_t m_numCompleted {0};              //!< Total ever completed

  uint32_t m_prevOvlimit {0};               //!< Previous over limit
  uint32_t m_prevNumQueued {0};             //!< Previous queue total
  uint32_t m_prevLastObjCnt {0};            //!< Previous queuing cnt

  uint32_t m_lowestSlack {std::numeric_limits<uint32_t>::max ()};  //!< Lowest slack found
  Time m_slackStartTime {Seconds (0)};      //!< Time slacks seen

  // Configuration
  uint32_t m_maxLimit;                      //!< Max limit
  uint32_t m_minLimit;                      //!< Minimum limit
  Time m_slackHoldTime;                     //!< Time to measure slack
};

} // namespace ns3

#endif /* DYNAMIC_QUEUE_LIMITS_H */
