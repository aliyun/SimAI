/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
#ifndef TIMER_H
#define TIMER_H

#include "fatal-error.h"
#include "nstime.h"
#include "event-id.h"
#include "int-to-type.h"

/**
 * \file
 * \ingroup timer
 * ns3::Timer class declaration.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup timer Virtual Time Timer and Watchdog
 *
 * The Timer and Watchdog objects both facilitate scheduling functions
 * to execute a specified virtual time in the future.
 *
 * A Watchdog timer cannot be paused or cancelled once it has been started,
 * however it can be lengthened (delayed).  A Watchdog takes no action
 * when it is destroyed.
 *
 * A Timer can be suspended, resumed, cancelled and queried for time left,
 * but it can't be extended (except by suspending and resuming).
 * In addition, it can be configured to take different actions when the
 * Timer is destroyed.
 */

class TimerImpl;

/**
 * \ingroup timer
 * \brief A simple virtual Timer class
 *
 * A (virtual time) timer is used to hold together a delay, a function to invoke
 * when the delay expires, and a set of arguments to pass to the function
 * when the delay expires.
 *
 * A Timer can be suspended, resumed, cancelled and queried for the
 * time left, but it can't be extended (except by suspending and
 * resuming).
 *
 * A timer can also be used to enforce a set of predefined event lifetime
 * management policies. These policies are specified at construction time
 * and cannot be changed after.
 *
 * \see Watchdog for a simpler interface for a watchdog timer.
 */
class Timer
{
public:
  /**
   * The policy to use to manager the internal timer when an
   * instance of the Timer class is destroyed or suspended.
   *
   * In the case of suspension, only `CANCEL_ON_DESTROY` and
   * `REMOVE_ON_DESTROY` apply.
   *
   * These symbols have "Destroy" in their names
   * for historical reasons.
   */
  enum DestroyPolicy
  {
    /**
     * This policy cancels the event from the destructor of the Timer
     * or from Suspend().  This is typically faster than `REMOVE_ON_DESTROY`
     * but uses more memory.
     */
    CANCEL_ON_DESTROY = (1 << 3),
    /**
     * This policy removes the event from the simulation event list
     * when the destructor of the Timer is invoked, or the Timer is
     * suspended.  This is typically slower than Cancel, but frees memory.
     */
    REMOVE_ON_DESTROY = (1 << 4),
    /**
     * This policy enforces a check from the destructor of the Timer
     * to verify that the timer has already expired.
     */
    CHECK_ON_DESTROY = (1 << 5)
  };
  /** The possible states of the Timer. */
  enum State
  {
    RUNNING,    /**< Timer is currently running. */
    EXPIRED,    /**< Timer has already expired. */
    SUSPENDED,  /**< Timer is suspended. */
  };
  /**
   * Create a timer with a default event lifetime management policy:
   *  - CHECK_ON_DESTROY
   */
  Timer ();
  /**
   * \param [in] destroyPolicy the event lifetime management policies
   * to use for destroy events
   */
  Timer (enum DestroyPolicy destroyPolicy);
  ~Timer ();

  /**
   * \tparam FN \deduced The type of the function.
   * \param [in] fn the function
   *
   * Store this function in this Timer for later use by Timer::Schedule.
   */
  template <typename FN>
  void SetFunction (FN fn);

  /**
   * \tparam MEM_PTR \deduced The type of the class member function.
   * \tparam OBJ_PTR \deduced The type of the class instance pointer.
   * \param [in] memPtr the member function pointer
   * \param [in] objPtr the pointer to object
   *
   * Store this function and object in this Timer for later use by
   * Timer::Schedule.
   */
  template <typename MEM_PTR, typename OBJ_PTR>
  void SetFunction (MEM_PTR memPtr, OBJ_PTR objPtr);


  /**
   * \tparam Ts \deduced Argument types
   * \param [in] args arguments
   *
   * Store these arguments in this Timer for later use by Timer::Schedule.
   */
  template <typename... Ts>
  void SetArguments (Ts... args);

  /**
   * \param [in] delay The delay
   *
   * The next call to Schedule will schedule the timer with this delay.
   */
  void SetDelay (const Time &delay);
  /**
   * \returns The currently-configured delay for the next Schedule.
   */
  Time GetDelay (void) const;
  /**
   * \returns The amount of time left until this timer expires.
   *
   * This method returns zero if the timer is in EXPIRED state.
   */
  Time GetDelayLeft (void) const;
  /**
   * Cancel the currently-running event if there is one. Do nothing
   * otherwise.
   */
  void Cancel (void);
  /**
   * Remove from the simulation event-list the currently-running event
   * if there is one. Do nothing otherwise.
   */
  void Remove (void);
  /**
   * \return \c true if there is no currently-running event,
   * \c false otherwise.
   */
  bool IsExpired (void) const;
  /**
   * \return \c true if there is a currently-running event,
   * \c false otherwise.
   */
  bool IsRunning (void) const;
  /**
   * \returns \c true if this timer was suspended and not yet resumed,
   * \c false otherwise.
   */
  bool IsSuspended (void) const;
  /**
   * \returns The current state of the timer.
   */
  enum Timer::State GetState (void) const;
  /**
   * Schedule a new event using the currently-configured delay, function,
   * and arguments.
   */
  void Schedule (void);
  /**
   * \param [in] delay the delay to use
   *
   * Schedule a new event using the specified delay (ignore the delay set by
   * Timer::SetDelay), function, and arguments.
   */
  void Schedule (Time delay);

  /**
   * Pause the timer and save the amount of time left until it was
   * set to expire.
   *
   * Subsequently calling Resume() will restart the Timer with the
   * remaining time.
   *
   * The DestroyPolicy set at construction determines
   * whether the underlying Simulator::Event is cancelled or removed.
   *
   * Calling Suspend on a non-running timer is an error.
   */
  void Suspend (void);
  /**
   * Restart the timer to expire within the amount of time left saved
   * during Suspend.
   * Calling Resume without a prior call to Suspend is an error.
   */
  void Resume (void);

private:
  /** Internal bit marking the suspended state. */
  enum InternalSuspended
  {
    TIMER_SUSPENDED = (1 << 7)  /** Timer suspended. */
  };

  /**
   * Bitfield for Timer State, DestroyPolicy and InternalSuspended.
   *
   * \internal
   * The DestroyPolicy, State and InternalSuspended state are stored
   * in this single bitfield.  The State uses the low-order bits,
   * so the other users of the bitfield have to be careful in defining
   * their bits to avoid the State.
   */
  int m_flags;
  /** The delay configured for this Timer. */
  Time m_delay;
  /** The future event scheduled to expire the timer. */
  EventId m_event;
  /**
   * The timer implementation, which contains the bound callback
   * function and arguments.
   */
  TimerImpl *m_impl;
  /** The amount of time left on the Timer while it is suspended. */
  Time m_delayLeft;
};

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

#include "timer-impl.h"

namespace ns3 {


template <typename FN>
void
Timer::SetFunction (FN fn)
{
  delete m_impl;
  m_impl = MakeTimerImpl (fn);
}
template <typename MEM_PTR, typename OBJ_PTR>
void
Timer::SetFunction (MEM_PTR memPtr, OBJ_PTR objPtr)
{
  delete m_impl;
  m_impl = MakeTimerImpl (memPtr, objPtr);
}

template <typename... Ts>
void
Timer::SetArguments (Ts... args)
{
  if (m_impl == 0)
    {
      NS_FATAL_ERROR ("You cannot set the arguments of a Timer before setting its function.");
      return;
    }
  m_impl->SetArgs (args...);
}

} // namespace ns3

#endif /* TIMER_H */
