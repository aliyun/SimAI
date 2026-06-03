/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
#ifndef EVENT_IMPL_H
#define EVENT_IMPL_H

#include <stdint.h>
#include "simple-ref-count.h"

/**
 * \file
 * \ingroup events
 * ns3::EventImpl declarations.
 */

namespace ns3 {

/**
 * \ingroup events
 * \brief A simulation event.
 *
 * Each subclass of this base class represents a simulation event. The
 * Invoke() method will be called by the simulation engine
 * when it reaches the time associated to this event. Most subclasses
 * are usually created by one of the many Simulator::Schedule
 * methods.
 */
class EventImpl : public SimpleRefCount<EventImpl>
{
public:
  /** Default constructor. */
  EventImpl ();
  /** Destructor. */
  virtual ~EventImpl () = 0;
  /**
   * Called by the simulation engine to notify the event that it is time
   * to execute.
   */
  void Invoke (void);
  /**
   * Marks the event as 'canceled'. The event is not removed from
   * the event list but the simulation engine will check its canceled status
   * before calling Invoke().
   */
  void Cancel (void);
  /**
   * \returns true if the event has been canceled.
   *
   * Checked by the simulation engine before calling Invoke().
   */
  bool IsCancelled (void);

protected:
  /**
   * Implementation for Invoke().
   *
   * This typically calls a method or function pointer with the
   * arguments bound by a call to one of the MakeEvent() functions.
   */
  virtual void Notify (void) = 0;

private:
  bool m_cancel;  /**< Has this event been cancelled. */
};

} // namespace ns3

#endif /* EVENT_IMPL_H */
