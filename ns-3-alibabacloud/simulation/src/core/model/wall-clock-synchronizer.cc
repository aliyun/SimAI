/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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
 */

#include <ctime>       // clock_t
#include <sys/time.h>  // gettimeofday
                       // clock_getres: glibc < 2.17, link with librt

#include "log.h"
#include "wall-clock-synchronizer.h"

#include <condition_variable>
#include <chrono>
#include <mutex>

/**
 * \file
 * \ingroup realtime
 * ns3::WallClockSynchronizer implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WallClockSynchronizer");

NS_OBJECT_ENSURE_REGISTERED (WallClockSynchronizer);

TypeId
WallClockSynchronizer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WallClockSynchronizer")
    .SetParent<Synchronizer> ()
    .SetGroupName ("Core")
  ;
  return tid;
}

WallClockSynchronizer::WallClockSynchronizer ()
{
  NS_LOG_FUNCTION (this);
//
// In Linux, the basic timekeeping unit is derived from a variable called HZ
// HZ is the frequency in hertz of the system timer.  The system timer fires
// every 1/HZ seconds and a counter, called the jiffies counter is incremented
// at each tick.  The time between ticks is called a jiffy (American slang for
// a short period of time).  The ticking of the jiffies counter is how the
// the kernel tells time.
//
// Now, the shortest time the kernel can sleep is one jiffy since a timer
// has to be set to expire and trigger the process to be made ready.  The
// Posix clock CLOCK_REALTIME is defined as a 1/HZ clock, so by doing a
// clock_getres () on the realtime clock we can infer the scheduler quantum
// and the minimimum sleep time for the system.  This is most certainly NOT
// going to be one nanosecond even though clock_nanosleep () pretends it is.
//
// The reason this number is important is that we are going to schedule lots
// of waits for less time than a jiffy.  The clock_nanosleep function is
// going to guarantee that it will sleep for AT LEAST the time specified.
// The least time that it will sleep is a jiffy.
//
// In order to deal with this, we are going to do a spin-wait if the simulator
// requires a delay less than a jiffy.  This is on the order of one millisecond
// (999848 ns) on the ns-regression machine.
//
// If the underlying OS does not support posix clocks, we'll just assume a
// one millisecond quantum and deal with this as best we can

#ifdef CLOCK_REALTIME
  struct timespec ts;
  clock_getres (CLOCK_REALTIME, &ts);
  m_jiffy = ts.tv_sec * NS_PER_SEC + ts.tv_nsec;
  NS_LOG_INFO ("Jiffy is " << m_jiffy << " ns");
#else
  m_jiffy = 1000000;
#endif
}

WallClockSynchronizer::~WallClockSynchronizer ()
{
  NS_LOG_FUNCTION (this);
}

bool
WallClockSynchronizer::DoRealtime (void)
{
  NS_LOG_FUNCTION (this);
  return true;
}

uint64_t
WallClockSynchronizer::DoGetCurrentRealtime (void)
{
  NS_LOG_FUNCTION (this);
  return GetNormalizedRealtime ();
}

void
WallClockSynchronizer::DoSetOrigin (uint64_t ns)
{
  NS_LOG_FUNCTION (this << ns);
//
// In order to make sure we're really locking the simulation time to some
// wall-clock time, we need to be able to compare that simulation time to
// that wall-clock time.  The wall clock will have been running for some
// long time and will probably have a huge count of nanoseconds in it.  We
// save the real time away so we can subtract it from "now" later and get
// a count of nanoseconds in real time since the simulation started.
//
  m_realtimeOriginNano = GetRealtime ();
  NS_LOG_INFO ("origin = " << m_realtimeOriginNano);
}

int64_t
WallClockSynchronizer::DoGetDrift (uint64_t ns)
{
  NS_LOG_FUNCTION (this << ns);
//
// In order to make sure we're really locking the simulation time to some
// wall-clock time, we need to be able to compare that simulation time to
// that wall-clock time.  In DoSetOrigin we saved the real time at the start
// of the simulation away.  This is the place where we subtract it from "now"
// to a count of nanoseconds in real time since the simulation started.  We
// then subtract the current real time in normalized nanoseconds we just got
// from the normalized simulation time in nanoseconds that is passed in as
// the parameter ns.  We return an integer difference, but in reality all of
// the mechanisms that cause wall-clock to simuator time drift cause events
// to be late.  That means that the wall-clock will be higher than the
// simulation time and drift will be positive.  I would be astonished to
// see a negative drift, but the possibility is admitted for other
// implementations; and we'll use the ability to report an early result in
// DoSynchronize below.
//
  uint64_t nsNow = GetNormalizedRealtime ();

  if (nsNow > ns)
    {
//
// Real time (nsNow) is larger/later than the simulator time (ns).  We are
// behind real time and the difference (drift) is positive.
//
      return (int64_t)(nsNow - ns);
    }
  else
    {
//
// Real time (nsNow) is smaller/earlier than the simulator time (ns).  We are
// ahead of real time and the difference (drift) is negative.
//
      return -(int64_t)(ns - nsNow);
    }
}

bool
WallClockSynchronizer::DoSynchronize (uint64_t nsCurrent, uint64_t nsDelay)
{
  NS_LOG_FUNCTION (this << nsCurrent << nsDelay);
//
// This is the belly of the beast.  We have received two parameters from the
// simulator proper -- a current simulation time (nsCurrent) and a simulation
// time to delay which identifies the time the next event is supposed to fire.
//
// The first thing we need to do is to (try and) correct for any realtime
// drift that has happened in the system.  In this implementation, we realize
// that all mechanisms for drift will cause the drift to be such that the
// realtime is greater than the simulation time.  This typically happens when
// our process is put to sleep for a given time, but actually sleeps for
// longer.  So, what we want to do is to "catch up" to realtime and delay for
// less time than we are actually asked to delay.  DriftCorrect will return a
// number from 0 to nsDelay corresponding to the amount of catching-up we
// need to do.  If we are more than nsDelay behind, we do not wait at all.
//
// Note that it will be impossible to catch up if the amount of drift is
// cumulatively greater than the amount of delay between events.  The method
// GetDrift () is available to clients of the syncrhonizer to keep track of
// the cumulative drift.  The client can assert if the drift gets out of
// hand, print warning messages, or just ignore the situation and hope it will
// go away.
//
  uint64_t ns = DriftCorrect (nsCurrent, nsDelay);
  NS_LOG_INFO ("Synchronize ns = " << ns);
//
// Once we've decided on how long we need to delay, we need to split this
// time into sleep waits and busy waits.  The reason for this is described
// in the comments for the constructor where jiffies and jiffy resolution is
// explained.
//
// Here, I'll just say that we need that the jiffy is the minimum resolution
// of the system clock.  It can only sleep in blocks of time equal to a jiffy.
// If we want to be more accurate than a jiffy (we do) then we need to sleep
// for some number of jiffies and then busy wait for any leftover time.
//
  uint64_t numberJiffies = ns / m_jiffy;
  NS_LOG_INFO ("Synchronize numberJiffies = " << numberJiffies);
//
// This is where the real world interjects its very ugly head.  The code
// immediately below reflects the fact that a sleep is actually quite probably
// going to end up sleeping for some number of jiffies longer than you wanted.
// This is because your system is going to be off doing other unimportant
// stuff during that extra time like running file systems and networks.  What
// we want to do is to ask the system to sleep enough less than the requested
// delay so that it comes back early most of the time (coming back early is
// fine, coming back late is bad).  If we can convince the system to come back
// early (most of the time), then we can busy-wait until the requested
// completion time actually comes around (most of the time).
//
// The tradeoff here is, of course, that the less time we spend sleeping, the
// more accurately we will sync up; but the more CPU time we will spend busy
// waiting (doing nothing).
//
// I'm not really sure about this number -- a boss of mine once said, "pick
// a number and it'll be wrong."  But this works for now.
//
// \todo Hardcoded tunable parameter below.
//
  if (numberJiffies > 3)
    {
      NS_LOG_INFO ("SleepWait for " << numberJiffies * m_jiffy << " ns");
      NS_LOG_INFO ("SleepWait until " << nsCurrent + numberJiffies * m_jiffy
                                      << " ns");
//
// SleepWait is interruptible.  If it returns true it meant that the sleep
// went until the end.  If it returns false, it means that the sleep was
// interrupted by a Signal.  In this case, we need to return and let the
// simulator re-evaluate what to do.
//
      if (SleepWait ((numberJiffies - 3) * m_jiffy) == false)
        {
          NS_LOG_INFO ("SleepWait interrupted");
          return false;
        }
    }
  NS_LOG_INFO ("Done with SleepWait");
//
// We asked the system to sleep for some number of jiffies, but that doesn't
// mean we actually did.  Let's re-evaluate what we need to do here.  Maybe
// we're already late.  Probably the "real" delay time left has little to do
// with what we would calculate it to be naively.
//
// We are now at some Realtime.  The important question now is not, "what
// would we calculate in a mathematicians paradise," it is, "how many
// nanoseconds do we need to busy-wait until we get to the Realtime that
// corresponds to nsCurrent + nsDelay (in simulation time).  We have a handy
// function to do just that -- we ask for the time the realtime clock has
// drifted away from the simulation clock.  That's our answer.  If the drift
// is negative, we're early and we need to busy wait for that number of
// nanoseconds.  The place were we want to be is described by the parameters
// we were passed by the simulator.
//
  int64_t nsDrift = DoGetDrift (nsCurrent + nsDelay);
//
// If the drift is positive, we are already late and we need to just bail out
// of here as fast as we can.  Return true to indicate that the requested time
// has, in fact, passed.
//
  if (nsDrift >= 0)
    {
      NS_LOG_INFO ("Back from SleepWait: IML8 " << nsDrift);
      return true;
    }
//
// There are some number of nanoseconds left over and we need to wait until
// the time defined by nsDrift.  We'll do a SpinWait since the usual case
// will be that we are doing this Spinwait after we've gotten a rough delay
// using the SleepWait above.  If SpinWait completes to the end, it will
// return true; if it is interrupted by a signal it will return false.
//
  NS_LOG_INFO ("SpinWait until " << nsCurrent + nsDelay);
  return SpinWait (nsCurrent + nsDelay);
}

void
WallClockSynchronizer::DoSignal (void)
{
  NS_LOG_FUNCTION (this);

  std::unique_lock<std::mutex> lock (m_mutex);
  m_condition = true;

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details).
  // Reference: https://en.cppreference.com/w/cpp/thread/condition_variable
  lock.unlock ();
  m_conditionVariable.notify_one ();
}

void
WallClockSynchronizer::DoSetCondition (bool cond)
{
  NS_LOG_FUNCTION (this << cond);
  m_condition = cond;
}

void
WallClockSynchronizer::DoEventStart (void)
{
  NS_LOG_FUNCTION (this);
  m_nsEventStart = GetNormalizedRealtime ();
}

uint64_t
WallClockSynchronizer::DoEventEnd (void)
{
  NS_LOG_FUNCTION (this);
  return GetNormalizedRealtime () - m_nsEventStart;
}

bool
WallClockSynchronizer::SpinWait (uint64_t ns)
{
  NS_LOG_FUNCTION (this << ns);
// We just sit here and spin, wasting CPU cycles until we get to the right
// time or are told to leave.
  for (;;)
    {
      if (GetNormalizedRealtime () >= ns)
        {
          return true;
        }
      if (m_condition)
        {
          return false;
        }
    }
// Quiet compiler
  return true;
}

bool
WallClockSynchronizer::SleepWait (uint64_t ns)
{
  NS_LOG_FUNCTION (this << ns);

  std::unique_lock<std::mutex> lock (m_mutex);
  bool finishedWaiting = m_conditionVariable.wait_for (
                            lock,
                            std::chrono::nanoseconds (ns),    // Timeout
                            [this](){ return m_condition; }); // Wait condition

  return finishedWaiting;
}

uint64_t
WallClockSynchronizer::DriftCorrect (uint64_t nsNow, uint64_t nsDelay)
{
  NS_LOG_FUNCTION (this << nsNow << nsDelay);
  int64_t drift = DoGetDrift (nsNow);
//
// If we're running late, drift will be positive and we need to correct by
// delaying for less time.  If we're early for some bizarre reason, we don't
// do anything since we'll almost instantly self-correct.
//
  if (drift < 0)
    {
      return nsDelay;
    }
//
// If we've drifted out of sync by less than the requested delay, then just
// subtract the drift from the delay and fix up the drift in one go.  If we
// have more drift than delay, then we just play catch up as fast as possible
// by not delaying at all.
//
  uint64_t correction = (uint64_t)drift;
  if (correction <= nsDelay)
    {
      return nsDelay - correction;
    }
  else
    {
      return 0;
    }
}

uint64_t
WallClockSynchronizer::GetRealtime (void)
{
  NS_LOG_FUNCTION (this);
  struct timeval tvNow;
  gettimeofday (&tvNow, NULL);
  return TimevalToNs (&tvNow);
}

uint64_t
WallClockSynchronizer::GetNormalizedRealtime (void)
{
  NS_LOG_FUNCTION (this);
  return GetRealtime () - m_realtimeOriginNano;
}

void
WallClockSynchronizer::NsToTimeval (int64_t ns, struct timeval *tv)
{
  NS_LOG_FUNCTION (this << ns << tv);
  NS_ASSERT ((ns % US_PER_NS) == 0);
  tv->tv_sec = static_cast<long> (ns / NS_PER_SEC);
  tv->tv_usec = (ns % NS_PER_SEC) / US_PER_NS;
}

uint64_t
WallClockSynchronizer::TimevalToNs (struct timeval *tv)
{
  NS_LOG_FUNCTION (this << tv);
  uint64_t nsResult = tv->tv_sec * NS_PER_SEC + tv->tv_usec * US_PER_NS;
  NS_ASSERT ((nsResult % US_PER_NS) == 0);
  return nsResult;
}

void
WallClockSynchronizer::TimevalAdd (
  struct timeval *tv1,
  struct timeval *tv2,
  struct timeval *result)
{
  NS_LOG_FUNCTION (this << tv1 << tv2 << result);
  result->tv_sec = tv1->tv_sec + tv2->tv_sec;
  result->tv_usec = tv1->tv_usec + tv2->tv_usec;
  if (result->tv_usec > (int64_t)US_PER_SEC)
    {
      ++result->tv_sec;
      result->tv_usec %= US_PER_SEC;
    }
}

} // namespace ns3
