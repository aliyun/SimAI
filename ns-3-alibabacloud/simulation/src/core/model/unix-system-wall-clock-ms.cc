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
 * Author: Mathieu Lacage <mathieu.lacage.inria.fr>
 */

#include "system-wall-clock-ms.h"
#include "abort.h"
#include "log.h"
#include <sys/times.h>
#include <unistd.h>

/**
 * \file
 * \ingroup system
 * ns3::SystemWallClockMs and ns3::SystemWallClockMsPrivate implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SystemWallClockMs");

/**
 * \ingroup system
 * \brief System-dependent implementation for SystemWallClockMs
 */
class SystemWallClockMsPrivate
{
public:
  /** \copydoc SystemWallClockMs::Start() */
  void Start (void);
  /** \copydoc SystemWallClockMs::End() */
  int64_t End (void);
  /** \copydoc SystemWallClockMs::GetElapsedReal() */
  int64_t GetElapsedReal (void) const;
  /** \copydoc SystemWallClockMs::GetElapsedUser() */
  int64_t GetElapsedUser (void) const;
  /** \copydoc SystemWallClockMs::GetElapsedSystem() */
  int64_t GetElapsedSystem (void) const;

private:
  struct tms m_startTimes;  //!< The native time structure.
  clock_t m_startTime;      //!< Native real time.
  int64_t m_elapsedReal;    //!< Elapsed real time, in ms.
  int64_t m_elapsedUser;    //!< Elapsed user time, in ms.
  int64_t m_elapsedSystem;  //!< Elapsed system time, in ms.
};

void
SystemWallClockMsPrivate::Start (void)
{
  NS_LOG_FUNCTION (this);
  m_startTime = times (&m_startTimes);
}

int64_t
SystemWallClockMsPrivate::End (void)
{
  //
  // We need to return the number of milliseconds that have elapsed in some
  // reasonably portable way.  The underlying function that we will use returns
  // a number of elapsed ticks.  We can look up the number of ticks per second
  // from the system configuration.
  //
  // Conceptually, we need to find the number of elapsed clock ticks and then
  // multiply the result by the milliseconds per clock tick (or divide by clock
  // ticks per millisecond).  Integer dividing by clock ticks per millisecond
  // is bad since this number is fractional on most machines and would result
  // in divide by zero errors due to integer rounding.
  //
  // Multiplying by milliseconds per clock tick works up to a clock resolution
  // of 1000 ticks per second.  If we go  past this point, we begin to get zero
  // elapsed times when millisecondsPerTick becomes fractional and another
  // rounding error appears.
  //
  // So rounding errors using integers can bite you from both direction.  Since
  // all of our targets have math coprocessors, why not just use doubles
  // internally?  Works fine, lasts a long time.
  //
  // If millisecondsPerTick becomes fractional, and an elapsed time greater than
  // a milliscond is measured, the function will work as expected.  If an elapsed
  // time is measured that turns out to be less than a millisecond, we'll just
  // return zero which would, I think, also will be expected.
  //
  NS_LOG_FUNCTION (this);
  static int64_t ticksPerSecond = sysconf (_SC_CLK_TCK);
  static double millisecondsPerTick = 1000. / ticksPerSecond;

  //
  // If sysconf () fails, we have no idea how to do the required conversion to ms.
  //
  NS_ABORT_MSG_IF (ticksPerSecond == -1, "SystemWallClockMsPrivate(): Cannot sysconf (_SC_CLK_TCK)");

  struct tms endTimes;
  clock_t endTime = times (&endTimes);

  double tmp;

  tmp = static_cast<double> (endTime - m_startTime) * millisecondsPerTick;
  m_elapsedReal = static_cast<int64_t> (tmp);

  tmp = static_cast<double> (endTimes.tms_utime - m_startTimes.tms_utime) * millisecondsPerTick;
  m_elapsedUser = static_cast<int64_t> (tmp);

  tmp = static_cast<double> (endTimes.tms_stime - m_startTimes.tms_stime) * millisecondsPerTick;
  m_elapsedSystem = static_cast<int64_t> (tmp);

  return m_elapsedReal;
}

int64_t
SystemWallClockMsPrivate::GetElapsedReal (void) const
{
  NS_LOG_FUNCTION (this);
  return m_elapsedReal;
}

int64_t
SystemWallClockMsPrivate::GetElapsedUser (void) const
{
  NS_LOG_FUNCTION (this);
  return m_elapsedUser;
}

int64_t
SystemWallClockMsPrivate::GetElapsedSystem (void) const
{
  NS_LOG_FUNCTION (this);
  return m_elapsedSystem;
}

SystemWallClockMs::SystemWallClockMs ()
  : m_priv (new SystemWallClockMsPrivate ())
{
  NS_LOG_FUNCTION (this);
}

SystemWallClockMs::~SystemWallClockMs ()
{
  NS_LOG_FUNCTION (this);
  delete m_priv;
  m_priv = 0;
}

void
SystemWallClockMs::Start (void)
{
  NS_LOG_FUNCTION (this);
  m_priv->Start ();
}

int64_t
SystemWallClockMs::End (void)
{
  NS_LOG_FUNCTION (this);
  return m_priv->End ();
}

int64_t
SystemWallClockMs::GetElapsedReal (void) const
{
  NS_LOG_FUNCTION (this);
  return m_priv->GetElapsedReal ();
}

int64_t
SystemWallClockMs::GetElapsedUser (void) const
{
  NS_LOG_FUNCTION (this);
  return m_priv->GetElapsedUser ();
}

int64_t
SystemWallClockMs::GetElapsedSystem (void) const
{
  NS_LOG_FUNCTION (this);
  return m_priv->GetElapsedSystem ();
}

} // namespace ns3
