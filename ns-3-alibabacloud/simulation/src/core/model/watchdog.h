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
#ifndef WATCHDOG_H
#define WATCHDOG_H

#include "nstime.h"
#include "event-id.h"

/**
 * \file
 * \ingroup timer
 * ns3::Watchdog timer class declaration.
 */

namespace ns3 {

class TimerImpl;

/**
 * \ingroup timer
 * \brief A very simple watchdog operating in virtual time.
 *
 * The watchdog timer is started by calling Ping with a delay value.
 * Once started the timer cannot be suspended, cancelled or shortened.
 * It _can_ be lengthened (delayed) by calling Ping again:  if the new
 * expire time (current simulation time plus the new delay)
 * is greater than the old expire time the timer will be extended
 * to the new expire time.
 *
 * Typical usage would be to periodically Ping the Watchdog, extending
 * it's execution time.  If the owning process fails to Ping before
 * the Watchdog expires, the registered function will be invoked.
 *
 * If you don't ping the watchdog sufficiently often, it triggers its
 * listening function.
 *
 * \see Timer for a more sophisticated general purpose timer.
 */
class Watchdog
{
public:
  /** Constructor. */
  Watchdog ();
  /** Destructor. */
  ~Watchdog ();

  /**
   * Delay the timer.
   *
   * \param [in] delay The watchdog delay
   *
   * After a call to this method, the watchdog will not be triggered
   * until the delay specified has been expired. This operation is
   * sometimes named "re-arming" a watchdog in some operating systems.
   */
  void Ping (Time delay);

  /**
   * Set the function to execute when the timer expires.
   *
   * \tparam FN \deduced The type of the function.
   * \param [in] fn The function
   *
   * Store this function in this Timer for later use by Timer::Schedule.
   */
  template <typename FN>
  void SetFunction (FN fn);

  /**
   * Set the function to execute when the timer expires.
   *
   * \tparam MEM_PTR \deduced Class method function type.
   * \tparam OBJ_PTR \deduced Class type containing the function.
   * \param [in] memPtr The member function pointer
   * \param [in] objPtr The pointer to object
   *
   * Store this function and object in this Timer for later use by Timer::Schedule.
   */
  template <typename MEM_PTR, typename OBJ_PTR>
  void SetFunction (MEM_PTR memPtr, OBJ_PTR objPtr);


  /**
   * Set the arguments to be used when invoking the expire function.
   */
  /**@{*/
  /**
   * \tparam Ts \deduced Argument types.
   * \param [in] args arguments
   */
  template <typename... Ts>
  void SetArguments (Ts&&... args);
  /**@}*/

private:
  /** Internal callback invoked when the timer expires. */
  void Expire (void);
  /**
   * The timer implementation, which contains the bound callback
   * function and arguments.
   */
  TimerImpl *m_impl;
  /** The future event scheduled to expire the timer. */
  EventId m_event;
  /** The absolute time when the timer will expire. */
  Time m_end;
};

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

#include "timer-impl.h"

namespace ns3 {


template <typename FN>
void
Watchdog::SetFunction (FN fn)
{
  delete m_impl;
  m_impl = MakeTimerImpl (fn);
}
template <typename MEM_PTR, typename OBJ_PTR>
void
Watchdog::SetFunction (MEM_PTR memPtr, OBJ_PTR objPtr)
{
  delete m_impl;
  m_impl = MakeTimerImpl (memPtr, objPtr);
}

template <typename... Ts>
void
Watchdog::SetArguments (Ts&&... args)
{
  if (m_impl == 0)
    {
      NS_FATAL_ERROR ("You cannot set the arguments of a Watchdog before setting its function.");
      return;
    }
  m_impl->SetArgs (std::forward<Ts>(args)...);
}

} // namespace ns3


#endif /* WATCHDOG_H */
