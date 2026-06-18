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
#ifndef EVENT_ID_H
#define EVENT_ID_H

#include <stdint.h>
#include "ptr.h"
#include "event-impl.h"

/**
 * \file
 * \ingroup events
 * ns3::EventId declarations.
 */

namespace ns3 {

class EventImpl;

/**
 * \ingroup events
 * \brief An identifier for simulation events.
 *
 * Each EventId identifies a unique event scheduled with one
 * of the many Simulator::Schedule() methods. This EventId can
 * be used to cancel or remove events after they are scheduled
 * with Cancel(), Remove(), or Simulator::Cancel() or Simulator::Remove().
 *
 * The important thing to remember about this class is that
 * every variable of this type is _always_ in a valid state,
 * even when it has not been assigned an EventId coming from a
 * Simulator::Schedule() method:  calling Simulator::Cancel(), IsRunning(),
 * IsExpired() or passing around instances of this object
 * will not result in crashes or memory leaks.
 */
class EventId
{
public:

  /** Special values of the event UID. */
  enum UID 
    {
      /** Invalid UID value. */
      INVALID = 0,
      /** ScheduleNow() events. */
      NOW = 1,
      /** ScheduleDestroy() events. */
      DESTROY = 2,
      /** Reserved UID. */
      RESERVED = 3,
      /** Schedule(), etc. events. */
      VALID = 4
    };

  /** Default constructor. This EventId does nothing. */
  EventId ();
  /**
   * Construct a real event.
   *
   * \param [in] impl The implementation of this event.
   * \param [in] ts The virtual time stamp this event should occur.
   * \param [in] context The execution context for this event.
   * \param [in] uid The unique id for this EventId.
   */
  EventId (const Ptr<EventImpl> &impl, uint64_t ts, uint32_t context, uint32_t uid);
  /**
   * This method is syntactic sugar for the ns3::Simulator::Cancel
   * method.
   */
  void Cancel (void);
  /**
   * This method is syntactic sugar for the ns3::Simulator::Remove
   * method.
   */
  void Remove (void);
  /**
   * This method is syntactic sugar for the ns3::Simulator::IsExpired
   * method.
   * \returns \c true if the event has expired, \c false otherwise.
   */
  bool IsExpired (void) const;
  /**
   * This method is syntactic sugar for !IsExpired().
   *
   * \returns \c true if the event has not expired, \c false otherwise.
   */
  bool IsRunning (void) const;

public:
  /**
   * \name Scheduler Helpers.
   * \brief These methods are normally invoked only by
   * subclasses of the Scheduler base class.
   */
  /**@{*/
  /** \return The underlying EventImpl pointer. */
  EventImpl * PeekEventImpl (void) const;
  /** \return The virtual time stamp. */
  uint64_t GetTs (void) const;
  /** \return The event context. */
  uint32_t GetContext (void) const;
  /** \return The unique id. */
  uint32_t GetUid (void) const;
  /**@}*/

  /**
   * Test if two EventId's are equal.
   * \param [in] a The first EventId.
   * \param [in] b The second EventId.
   * \return \c true if the \pname{a} and \pname{b} represent the same event.
   */
  friend bool operator == (const EventId &a, const EventId &b);
  /**
   * Test if two EventId's are not equal.
   * \param [in] a The first EventId.
   * \param [in] b The second EventId.
   * \return \c true if the \pname{a} and \pname{b} are not the same event.
   */
  friend bool operator != (const EventId &a, const EventId &b);
  /**
   * Less than operator for two EventId's, based on time stamps.
   * \param [in] a The first EventId.
   * \param [in] b The second EventId.
   * \return \c true if \pname{a} occurs before \pname{b}.
   */
  friend bool operator <  (const EventId &a, const EventId &b);

private:
  Ptr<EventImpl> m_eventImpl;  /**< The underlying event implementation. */
  uint64_t m_ts;               /**< The virtual time stamp. */
  uint32_t m_context;          /**< The context. */
  uint32_t m_uid;              /**< The unique id. */
};

/*************************************************
 **  Inline implementations
 ************************************************/

inline
bool
operator == (const EventId &a, const EventId &b)
{
  return
    a.m_uid == b.m_uid
    && a.m_context == b.m_context
    && a.m_ts == b.m_ts
    && a.m_eventImpl == b.m_eventImpl;
}

inline
bool
operator != (const EventId &a, const EventId &b)
{
  return !(a == b);
}

inline
bool
operator <  (const EventId &a, const EventId &b)
{
  return (a.GetTs () < b.GetTs ());
}

} // namespace ns3

#endif /* EVENT_ID_H */
