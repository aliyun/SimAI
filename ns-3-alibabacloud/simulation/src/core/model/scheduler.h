/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <stdint.h>
#include "object.h"

/**
 * \file
 * \ingroup scheduler
 * \ingroup events
 * ns3::Scheduler abstract base class, ns3::Scheduler::Event and
 * ns3::Scheduler::EventKey declarations.
 */

namespace ns3 {

class EventImpl;

/**
 * \ingroup core
 * \defgroup scheduler Scheduler and Events
 * \brief Manage the event list by creating and scheduling events.
 */
/**
 * \ingroup scheduler
 * \defgroup events Events
 */
/**
 * \ingroup scheduler
 * \brief Maintain the event list
 *
 *
 * In ns-3 the Scheduler manages the future event list.  There are several
 * different Scheduler implementations with different time and space tradeoffs.
 * Which one is "best" depends in part on the characteristics
 * of the model being executed.  For optimized production work common
 * practice is to benchmark each Scheduler on the model of interest.
 * The utility program utils/bench-simulator.cc can do simple benchmarking
 * of each SchedulerImpl against an exponential or user-provided
 * event time distribution.
 *
 * The most important Scheduler functions for time performance are (usually)
 * Scheduler::Insert (for new events) and Scheduler::RemoveNext (for pulling
 * off the next event to execute).  Simulator::Cancel is usually
 * implemented by simply setting a bit on the Event, but leaving it in the
 * Scheduler; the Simulator just skips those events as they are encountered.
 *
 * For models which need a large event list the Scheduler overhead
 * and per-event memory cost could also be important.  Some models
 * rely heavily on Scheduler::Cancel, however, and these might benefit
 * from using Scheduler::Remove instead, to reduce the size of the event
 * list, at the time cost of actually removing events from the list.
 *
 * A summary of the main characteristics
 * of each SchedulerImpl is provided below.  See the individual
 * Scheduler pages for details on the complexity of the other API calls.
 * (Memory overheads assume pointers and `std::size_t` are both 8 bytes.)
 *
 * <table class="markdownTable">
 * <tr class="markdownTableHead">
 *      <th class="markdownTableHeadCenter" colspan="2"> %Scheduler Type </th>
 *      <th class="markdownTableHeadCenter" colspan="4">Complexity</th>
 * </tr>
 * <tr class="markdownTableHead">
 *      <th class="markdownTableHeadLeft" rowspan="2"> %SchedulerImpl </th>
 *      <th class="markdownTableHeadLeft" rowspan="2"> Method </th>
 *      <th class="markdownTableHeadCenter" colspan="2"> %Time </th>
 *      <th class="markdownTableHeadCenter" colspan="2"> Space</th>
 * </tr>
 * <tr class="markdownTableHead">
 *      <th class="markdownTableHeadLeft"> %Insert()</th>
 *      <th class="markdownTableHeadLeft"> %RemoveNext()</th>
 *      <th class="markdownTableHeadLeft"> Overhead</th>
 *      <th class="markdownTableHeadLeft"> Per %Event</th>
 * </tr>
 * <tr class="markdownTableBody">
 *      <td class="markdownTableBodyLeft"> CalendarScheduler </td>
 *      <td class="markdownTableBodyLeft"> `<std::list> []` </td>
 *      <td class="markdownTableBodyLeft"> Constant </td>
 *      <td class="markdownTableBodyLeft"> Constant </td>
 *      <td class="markdownTableBodyLeft"> 24 bytes </td>
 *      <td class="markdownTableBodyLeft"> 16 bytes </td>
 * </tr>
 * <tr class="markdownTableBody">
 *      <td class="markdownTableBodyLeft"> HeapScheduler </td>
 *      <td class="markdownTableBodyLeft"> Heap on `std::vector` </td>
 *      <td class="markdownTableBodyLeft"> Logarithmic  </td>
 *      <td class="markdownTableBodyLeft"> Logarithmic </td>
 *      <td class="markdownTableBodyLeft"> 24 bytes </td>
 *      <td class="markdownTableBodyLeft"> 0 </td>
 * </tr>
 * <tr class="markdownTableBody">
 *      <td class="markdownTableBodyLeft"> ListScheduler </td>
 *      <td class="markdownTableBodyLeft"> `std::list` </td>
 *      <td class="markdownTableBodyLeft"> Linear </td>
 *      <td class="markdownTableBodyLeft"> Constant </td>
 *      <td class="markdownTableBodyLeft"> 24 bytes </td>
 *      <td class="markdownTableBodyLeft"> 16 bytes </td>
 * </tr>
 * <tr class="markdownTableBody">
 *      <td class="markdownTableBodyLeft"> MapScheduler </td>
 *      <td class="markdownTableBodyLeft"> `std::map` </td>
 *      <td class="markdownTableBodyLeft"> Logarithmic </td>
 *      <td class="markdownTableBodyLeft"> Constant </td>
 *      <td class="markdownTableBodyLeft"> 40 bytes </td>
 *      <td class="markdownTableBodyLeft"> 32 bytes </td>
 * </tr>
 * <tr class="markdownTableBody">
 *      <td class="markdownTableBodyLeft"> PriorityQueueScheduler </td>
 *      <td class="markdownTableBodyLeft"> `std::priority_queue<,std::vector>` </td>
 *      <td class="markdownTableBodyLeft"> Logarithmic  </td>
 *      <td class="markdownTableBodyLeft"> Logarithmic </td>
 *      <td class="markdownTableBodyLeft"> 24 bytes </td>
 *      <td class="markdownTableBodyLeft"> 0 </td>
 * </tr>
 * </table>
 *
 * It is possible to change the Scheduler choice during a simulation,
 * via Simulator::SetScheduler.
 *
 * The Scheduler base class specifies the interface used to maintain the
 * event list. If you want to provide a new event list scheduler,
 * you need to create a subclass of this base class and implement
 * all the pure virtual methods defined here.
 *
 * The only tricky aspect of this API is the memory management of
 * the EventImpl pointer which is a member of the Event data structure.
 * The lifetime of this pointer is assumed to always be longer than
 * the lifetime of the Scheduler class which means that the caller
 * is responsible for ensuring that this invariant holds through
 * calling EventId::Ref and SimpleRefCount::Unref at the right time.
 * Typically, EventId::Ref is called before Insert and SimpleRefCount::Unref is called
 * after a call to one of the Remove methods.
 */
class Scheduler : public Object
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \ingroup events
   * Structure for sorting and comparing Events.
   */
  struct EventKey
  {
    uint64_t m_ts;         /**< Event time stamp. */
    uint32_t m_uid;        /**< Event unique id. */
    uint32_t m_context;    /**< Event context. */
  };
  /**
   * \ingroup events
   * Scheduler event.
   *
   * An Event consists of an EventKey, used for maintaining the schedule,
   * and an EventImpl which is the actual implementation.
   */
  struct Event
  {
    EventImpl *impl;       /**< Pointer to the event implementation. */
    EventKey key;          /**< Key for sorting and ordering Events. */
  };

  /** Destructor. */
  virtual ~Scheduler () = 0;

  /**
   * Insert a new Event in the schedule.
   *
   * \param [in] ev Event to store in the event list
   */
  virtual void Insert (const Event &ev) = 0;
  /**
   * Test if the schedule is empty.
   *
   * \returns \c true if the event list is empty and \c false otherwise.
   */
  virtual bool IsEmpty (void) const = 0;
  /**
   * Get a pointer to the next event.
   *
   * This method cannot be invoked if the list is empty.
   *
   * \returns A pointer to the next earliest event. The caller
   *      takes ownership of the returned pointer.
   */
  virtual Event PeekNext (void) const = 0;
  /**
   * Remove the earliest event from the event list.
   *
   * This method cannot be invoked if the list is empty.
   *
   * \return The Event.
   */
  virtual Event RemoveNext (void) = 0;
  /**
   * Remove a specific event from the event list.
   *
   * This method cannot be invoked if the list is empty.
   *
   * \param [in] ev The event to remove
   */
  virtual void Remove (const Event &ev) = 0;
};

/**
 * \ingroup Events
 * Compare (equal) two events by EventKey.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a != \c b
 */
inline bool operator == (const Scheduler::EventKey &a,
                         const Scheduler::EventKey &b)
{
  return a.m_uid == b.m_uid;
}

/**
 * \ingroup Events
 * Compare (not equal) two events by EventKey.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a != \c b
 */
inline bool operator != (const Scheduler::EventKey &a,
                         const Scheduler::EventKey &b)
{
  return a.m_uid != b.m_uid;
}

/**
 * \ingroup Events
 * Compare (less than) two events by EventKey.
 *
 * Note the invariants which this function must provide:
 * - irreflexibility: f (x,x) is false
 * - antisymmetry: f(x,y) = !f(y,x)
 * - transitivity: f(x,y) and f(y,z) => f(x,z)
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a < \c b
 */
inline bool operator < (const Scheduler::EventKey &a,
                        const Scheduler::EventKey &b)
{
  if (a.m_ts < b.m_ts)
    {
      return true;
    }
  else if (a.m_ts == b.m_ts
           && a.m_uid < b.m_uid)
    {
      return true;
    }
  else
    {
      return false;
    }
}

/**
 * Compare (greater than) two events by EventKey.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a > \c b
 */
inline bool operator > (const Scheduler::EventKey &a,
                        const Scheduler::EventKey &b)
{
  if (a.m_ts > b.m_ts)
    {
      return true;
    }
  else if (a.m_ts == b.m_ts
           && a.m_uid > b.m_uid)
    {
      return true;
    }
  else
    {
      return false;
    }
}

/**
 * Compare (equal) two events by Event.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a == \c b
 */
inline bool operator == (const Scheduler::Event &a,
                         const Scheduler::Event &b)
{
  return a.key == b.key;
}

/**
 * Compare (not equal) two events by Event.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a != \c b
 */
inline bool operator != (const Scheduler::Event &a,
                         const Scheduler::Event &b)
{
  return a.key != b.key;
}

/**
 * Compare (less than) two events by Event.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a < \c b
 */
inline bool operator < (const Scheduler::Event &a,
                        const Scheduler::Event &b)
{
  return a.key < b.key;
}

/**
 * Compare (greater than) two events by Event.
 *
 * \param [in] a The first event.
 * \param [in] b The second event.
 * \returns \c true if \c a > \c b
 */
inline bool operator > (const Scheduler::Event &a,
                        const Scheduler::Event &b)
{
  return a.key > b.key;
}


} // namespace ns3


#endif /* SCHEDULER_H */
