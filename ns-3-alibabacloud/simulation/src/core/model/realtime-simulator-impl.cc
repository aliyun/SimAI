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

#include "simulator.h"
#include "realtime-simulator-impl.h"
#include "wall-clock-synchronizer.h"
#include "scheduler.h"
#include "event-impl.h"
#include "synchronizer.h"

#include "ptr.h"
#include "pointer.h"
#include "assert.h"
#include "fatal-error.h"
#include "log.h"
#include "boolean.h"
#include "enum.h"

#include <cmath>
#include <mutex>
#include <thread>

/**
 * \file
 * \ingroup realtime
 * ns3::RealTimeSimulatorImpl implementation.
 */

namespace ns3 {

// Note:  Logging in this file is largely avoided due to the
// number of calls that are made to these functions and the possibility
// of causing recursions leading to stack overflow
NS_LOG_COMPONENT_DEFINE ("RealtimeSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED (RealtimeSimulatorImpl);

TypeId
RealtimeSimulatorImpl::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RealtimeSimulatorImpl")
    .SetParent<SimulatorImpl> ()
    .SetGroupName ("Core")
    .AddConstructor<RealtimeSimulatorImpl> ()
    .AddAttribute ("SynchronizationMode",
                   "What to do if the simulation cannot keep up with real time.",
                   EnumValue (SYNC_BEST_EFFORT),
                   MakeEnumAccessor (&RealtimeSimulatorImpl::SetSynchronizationMode),
                   MakeEnumChecker (SYNC_BEST_EFFORT, "BestEffort",
                                    SYNC_HARD_LIMIT, "HardLimit"))
    .AddAttribute ("HardLimit",
                   "Maximum acceptable real-time jitter (used in conjunction with SynchronizationMode=HardLimit)",
                   TimeValue (Seconds (0.1)),
                   MakeTimeAccessor (&RealtimeSimulatorImpl::m_hardLimit),
                   MakeTimeChecker ())
  ;
  return tid;
}


RealtimeSimulatorImpl::RealtimeSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);

  m_stop = false;
  m_running = false;
  m_uid = EventId::UID::VALID;
  m_currentUid = EventId::UID::INVALID;
  m_currentTs = 0;
  m_currentContext = Simulator::NO_CONTEXT;
  m_unscheduledEvents = 0;
  m_eventCount = 0;

  m_main = std::this_thread::get_id ();

  // Be very careful not to do anything that would cause a change or assignment
  // of the underlying reference counts of m_synchronizer or you will be sorry.
  m_synchronizer = CreateObject<WallClockSynchronizer> ();
}

RealtimeSimulatorImpl::~RealtimeSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);
}

void
RealtimeSimulatorImpl::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  while (!m_events->IsEmpty ())
    {
      Scheduler::Event next = m_events->RemoveNext ();
      next.impl->Unref ();
    }
  m_events = 0;
  m_synchronizer = 0;
  SimulatorImpl::DoDispose ();
}

void
RealtimeSimulatorImpl::Destroy ()
{
  NS_LOG_FUNCTION (this);

  //
  // This function is only called with the private version "disconnected" from
  // the main simulator functions.  We rely on the user not calling
  // Simulator::Destroy while there is a chance that a worker thread could be
  // accessing the current instance of the private object.  In practice this
  // means shutting down the workers and doing a Join() before calling the
  // Simulator::Destroy().
  //
  while (m_destroyEvents.empty () == false)
    {
      Ptr<EventImpl> ev = m_destroyEvents.front ().PeekEventImpl ();
      m_destroyEvents.pop_front ();
      NS_LOG_LOGIC ("handle destroy " << ev);
      if (ev->IsCancelled () == false)
        {
          ev->Invoke ();
        }
    }
}

void
RealtimeSimulatorImpl::SetScheduler (ObjectFactory schedulerFactory)
{
  NS_LOG_FUNCTION (this << schedulerFactory);

  Ptr<Scheduler> scheduler = schedulerFactory.Create<Scheduler> ();

  {
    std::unique_lock lock {m_mutex};

    if (m_events != 0)
      {
        while (m_events->IsEmpty () == false)
          {
            Scheduler::Event next = m_events->RemoveNext ();
            scheduler->Insert (next);
          }
      }
    m_events = scheduler;
  }
}

void
RealtimeSimulatorImpl::ProcessOneEvent (void)
{
  //
  // The idea here is to wait until the next event comes due.  In the case of
  // a realtime simulation, we want real time to be consumed between events.
  // It is the realtime synchronizer that causes real time to be consumed by
  // doing some kind of a wait.
  //
  // We need to be able to have external events (such as a packet reception event)
  // cause us to re-evaluate our state.  The way this works is that the synchronizer
  // gets interrupted and returns.  So, there is a possibility that things may change
  // out from under us dynamically.  In this case, we need to re-evaluate how long to
  // wait in a for-loop until we have waited successfully (until a timeout) for the
  // event at the head of the event list.
  //
  // m_synchronizer->Synchronize will return true if the wait was completed without
  // interruption, otherwise it will return false indicating that something has changed
  // out from under us.  If we sit in the for-loop trying to synchronize until
  // Synchronize() returns true, we will have successfully synchronized the execution
  // time of the next event with the wall clock time of the synchronizer.
  //

  for (;;)
    {
      uint64_t tsDelay = 0;
      uint64_t tsNext = 0;

      //
      // It is important to understand that m_currentTs is interpreted only as the
      // timestamp  of the last event we executed.  Current time can a bit of a
      // slippery concept in realtime mode.  What we have here is a discrete event
      // simulator, so the last event is, by definition, executed entirely at a single
      //  discrete time.  This is the definition of m_currentTs.  It really has
      // nothing to do with the current real time, except that we are trying to arrange
      // that at the instant of the beginning of event execution, the current real time
      // and m_currentTs coincide.
      //
      // We use tsNow as the indication of the current real time.
      //
      uint64_t tsNow;

      {
        std::unique_lock lock {m_mutex};
        //
        // Since we are in realtime mode, the time to delay has got to be the
        // difference between the current realtime and the timestamp of the next
        // event.  Since m_currentTs is actually the timestamp of the last event we
        // executed, it's not particularly meaningful for us here since real time has
        // certainly elapsed since it was last updated.
        //
        // It is possible that the current realtime has drifted past the next event
        // time so we need to be careful about that and not delay in that case.
        //
        NS_ASSERT_MSG (m_synchronizer->Realtime (),
                       "RealtimeSimulatorImpl::ProcessOneEvent (): Synchronizer reports not Realtime ()");

        //
        // tsNow is set to the normalized current real time.  When the simulation was
        // started, the current real time was effectively set to zero; so tsNow is
        // the current "real" simulation time.
        //
        // tsNext is the simulation time of the next event we want to execute.
        //
        tsNow = m_synchronizer->GetCurrentRealtime ();
        tsNext = NextTs ();

        //
        // tsDelay is therefore the real time we need to delay in order to bring the
        // real time in sync with the simulation time.  If we wait for this amount of
        // real time, we will accomplish moving the simulation time at the same rate
        // as the real time.  This is typically called "pacing" the simulation time.
        //
        // We do have to be careful if we are falling behind.  If so, tsDelay must be
        // zero.  If we're late, don't dawdle.
        //
        if (tsNext <= tsNow)
          {
            tsDelay = 0;
          }
        else
          {
            tsDelay = tsNext - tsNow;
          }

        //
        // We've figured out how long we need to delay in order to pace the
        // simulation time with the real time.  We're going to sleep, but need
        // to work with the synchronizer to make sure we're awakened if something
        // external happens (like a packet is received).  This next line resets
        // the synchronizer so that any future event will cause it to interrupt.
        //
        m_synchronizer->SetCondition (false);
      }

      //
      // We have a time to delay.  This time may actually not be valid anymore
      // since we released the critical section immediately above, and a real-time
      // ScheduleReal or ScheduleRealNow may have snuck in, well, between the
      // closing brace above and this comment so to speak.  If this is the case,
      // that schedule operation will have done a synchronizer Signal() that
      // will set the condition variable to true and cause the Synchronize call
      // below to return immediately.
      //
      // It's easiest to understand if you just consider a short tsDelay that only
      // requires a SpinWait down in the synchronizer.  What will happen is that
      // whan Synchronize calls SpinWait, SpinWait will look directly at its
      // condition variable.  Note that we set this condition variable to false
      // inside the critical section above.
      //
      // SpinWait will go into a forever loop until either the time has expired or
      // until the condition variable becomes true.  A true condition indicates that
      // the wait should stop.  The condition is set to true by one of the Schedule
      // methods of the simulator; so if we are in a wait down in Synchronize, and
      // a Simulator::ScheduleReal is done, the wait down in Synchronize will exit and
      // Synchronize will return false.  This means we have not actually synchronized
      // to the event expiration time.  If no real-time schedule operation is done
      // while down in Synchronize, the wait will time out and Synchronize will return
      // true.  This indicates that we have synchronized to the event time.
      //
      // So we need to stay in this for loop, looking for the next event timestamp and
      // attempting to sleep until its due.  If we've slept until the timestamp is due,
      // Synchronize returns true and we break out of the sync loop.  If an external
      // event happens that requires a re-schedule, Synchronize returns false and
      // we re-evaluate our timing by continuing in the loop.
      //
      // It is expected that tsDelay become shorter as external events interrupt our
      // waits.
      //
      if (m_synchronizer->Synchronize (tsNow, tsDelay))
        {
          NS_LOG_LOGIC ("Interrupted ...");
          break;
        }

      //
      // If we get to this point, we have been interrupted during a wait by a real-time
      // schedule operation.  This means all bets are off regarding tsDelay and we need
      // to re-evaluate what it is we want to do.  We'll loop back around in the
      // for-loop and start again from scratch.
      //
    }

  //
  // If we break out of the for-loop above, we have waited until the time specified
  // by the event that was at the head of the event list when we started the process.
  // Since there is a bunch of code that was executed outside a critical section (the
  // Synchronize call) we cannot be sure that the event at the head of the event list
  // is the one we think it is.  What we can be sure of is that it is time to execute
  // whatever event is at the head of this list if the list is in time order.
  //
  Scheduler::Event next;

  {
    std::unique_lock lock {m_mutex};

    //
    // We do know we're waiting for an event, so there had better be an event on the
    // event queue.  Let's pull it off.  When we release the critical section, the
    // event we're working on won't be on the list and so subsequent operations won't
    // mess with us.
    //
    NS_ASSERT_MSG (m_events->IsEmpty () == false,
                   "RealtimeSimulatorImpl::ProcessOneEvent(): event queue is empty");
    next = m_events->RemoveNext ();

    PreEventHook (EventId (next.impl, next.key.m_ts, 
                           next.key.m_context, next.key.m_uid));

    m_unscheduledEvents--;
    m_eventCount++;

    //
    // We cannot make any assumption that "next" is the same event we originally waited
    // for.  We can only assume that only that it must be due and cannot cause time
    // to move backward.
    //
    NS_ASSERT_MSG (next.key.m_ts >= m_currentTs,
                   "RealtimeSimulatorImpl::ProcessOneEvent(): "
                   "next.GetTs() earlier than m_currentTs (list order error)");
    NS_LOG_LOGIC ("handle " << next.key.m_ts);

    //
    // Update the current simulation time to be the timestamp of the event we're
    // executing.  From the rest of the simulation's point of view, simulation time
    // is frozen until the next event is executed.
    //
    m_currentTs = next.key.m_ts;
    m_currentContext = next.key.m_context;
    m_currentUid = next.key.m_uid;

    //
    // We're about to run the event and we've done our best to synchronize this
    // event execution time to real time.  Now, if we're in SYNC_HARD_LIMIT mode
    // we have to decide if we've done a good enough job and if we haven't, we've
    // been asked to commit ritual suicide.
    //
    // We check the simulation time against the current real time to make this
    // judgement.
    //
    if (m_synchronizationMode == SYNC_HARD_LIMIT)
      {
        uint64_t tsFinal = m_synchronizer->GetCurrentRealtime ();
        uint64_t tsJitter;

        if (tsFinal >= m_currentTs)
          {
            tsJitter = tsFinal - m_currentTs;
          }
        else
          {
            tsJitter = m_currentTs - tsFinal;
          }

        if (tsJitter > static_cast<uint64_t> (m_hardLimit.GetTimeStep ()))
          {
            NS_FATAL_ERROR ("RealtimeSimulatorImpl::ProcessOneEvent (): "
                            "Hard real-time limit exceeded (jitter = " << tsJitter << ")");
          }
      }
  }

  //
  // We have got the event we're about to execute completely disentangled from the
  // event list so we can execute it outside a critical section without fear of someone
  // changing things out from under us.

  EventImpl *event = next.impl;
  m_synchronizer->EventStart ();
  event->Invoke ();
  m_synchronizer->EventEnd ();
  event->Unref ();
}

bool
RealtimeSimulatorImpl::IsFinished (void) const
{
  bool rc;
  {
    std::unique_lock lock {m_mutex};
    rc = m_events->IsEmpty () || m_stop;
  }

  return rc;
}

//
// Peeks into event list.  Should be called with critical section locked.
//
uint64_t
RealtimeSimulatorImpl::NextTs (void) const
{
  NS_ASSERT_MSG (m_events->IsEmpty () == false,
                 "RealtimeSimulatorImpl::NextTs(): event queue is empty");
  Scheduler::Event ev = m_events->PeekNext ();
  return ev.key.m_ts;
}

void
RealtimeSimulatorImpl::Run (void)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (m_running == false,
                 "RealtimeSimulatorImpl::Run(): Simulator already running");

  // Set the current threadId as the main threadId
  m_main = std::this_thread::get_id ();

  m_stop = false;
  m_running = true;
  m_synchronizer->SetOrigin (m_currentTs);

  // Sleep until signalled
  uint64_t tsNow = 0;
  uint64_t tsDelay = 1000000000; // wait time of 1 second (in nanoseconds)

  while (!m_stop)
    {
      bool process = false;
      {
        std::unique_lock lock {m_mutex};

        if (!m_events->IsEmpty ())
          {
            process = true;
          }
        else
          {
            // Get current timestamp while holding the critical section
            tsNow = m_synchronizer->GetCurrentRealtime ();
          }
      }

      if (!process)
        {
          // Sleep until signalled
          tsNow = m_synchronizer->Synchronize (tsNow, tsDelay);

          // Re-check event queue
          continue;
        }

      ProcessOneEvent ();
    }

  //
  // If the simulator stopped naturally by lack of events, make a
  // consistency test to check that we didn't lose any events along the way.
  //
  {
    std::unique_lock lock {m_mutex};

    NS_ASSERT_MSG (m_events->IsEmpty () == false || m_unscheduledEvents == 0,
                   "RealtimeSimulatorImpl::Run(): Empty queue and unprocessed events");
  }

  m_running = false;
}

bool
RealtimeSimulatorImpl::Running (void) const
{
  return m_running;
}

bool
RealtimeSimulatorImpl::Realtime (void) const
{
  return m_synchronizer->Realtime ();
}

void
RealtimeSimulatorImpl::Stop (void)
{
  NS_LOG_FUNCTION (this);
  m_stop = true;
}

void
RealtimeSimulatorImpl::Stop (Time const &delay)
{
  NS_LOG_FUNCTION (this << delay);
  Simulator::Schedule (delay, &Simulator::Stop);
}

//
// Schedule an event for a _relative_ time in the future.
//
EventId
RealtimeSimulatorImpl::Schedule (Time const &delay, EventImpl *impl)
{
  NS_LOG_FUNCTION (this << delay << impl);

  Scheduler::Event ev;
  {
    std::unique_lock lock {m_mutex};
    //
    // This is the reason we had to bring the absolute time calculation in from the
    // simulator.h into the implementation.  Since the implementations may be
    // multi-threaded, we need this calculation to be atomic.  You can see it is
    // here since we are running in a CriticalSection.
    //
    Time tAbsolute = Simulator::Now () + delay;
    NS_ASSERT_MSG (delay.IsPositive (), "RealtimeSimulatorImpl::Schedule(): Negative delay");
    ev.impl = impl;
    ev.key.m_ts = (uint64_t) tAbsolute.GetTimeStep ();
    ev.key.m_context = GetContext ();
    ev.key.m_uid = m_uid;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert (ev);
    m_synchronizer->Signal ();
  }

  return EventId (impl, ev.key.m_ts, ev.key.m_context, ev.key.m_uid);
}

void
RealtimeSimulatorImpl::ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *impl)
{
  NS_LOG_FUNCTION (this << context << delay << impl);

  {
    std::unique_lock lock {m_mutex};
    uint64_t ts;

    if (m_main == std::this_thread::get_id ())
      {
        ts = m_currentTs + delay.GetTimeStep ();
      }
    else
      {
        //
        // If the simulator is running, we're pacing and have a meaningful
        // realtime clock.  If we're not, then m_currentTs is where we stopped.
        //
        ts = m_running ? m_synchronizer->GetCurrentRealtime () : m_currentTs;
        ts += delay.GetTimeStep ();
      }

    NS_ASSERT_MSG (ts >= m_currentTs, "RealtimeSimulatorImpl::ScheduleRealtime(): schedule for time < m_currentTs");
    Scheduler::Event ev;
    ev.impl = impl;
    ev.key.m_ts = ts;
    ev.key.m_context = context;
    ev.key.m_uid = m_uid;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert (ev);
    m_synchronizer->Signal ();
  }
}

EventId
RealtimeSimulatorImpl::ScheduleNow (EventImpl *impl)
{
  NS_LOG_FUNCTION (this << impl);
  return Schedule (Time (0), impl);
}

Time
RealtimeSimulatorImpl::Now (void) const
{
  return TimeStep (m_currentTs);
}

//
// Schedule an event for a _relative_ time in the future.
//
void
RealtimeSimulatorImpl::ScheduleRealtimeWithContext (uint32_t context, Time const &time, EventImpl *impl)
{
  NS_LOG_FUNCTION (this << context << time << impl);

  {
    std::unique_lock lock {m_mutex};

    uint64_t ts = m_synchronizer->GetCurrentRealtime () + time.GetTimeStep ();
    NS_ASSERT_MSG (ts >= m_currentTs, "RealtimeSimulatorImpl::ScheduleRealtime(): schedule for time < m_currentTs");
    Scheduler::Event ev;
    ev.impl = impl;
    ev.key.m_ts = ts;
    ev.key.m_uid = m_uid;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert (ev);
    m_synchronizer->Signal ();
  }
}

void
RealtimeSimulatorImpl::ScheduleRealtime (Time const &time, EventImpl *impl)
{
  NS_LOG_FUNCTION (this << time << impl);
  ScheduleRealtimeWithContext (GetContext (), time, impl);
}

void
RealtimeSimulatorImpl::ScheduleRealtimeNowWithContext (uint32_t context, EventImpl *impl)
{
  NS_LOG_FUNCTION (this << context << impl);
  {
    std::unique_lock lock {m_mutex};

    //
    // If the simulator is running, we're pacing and have a meaningful
    // realtime clock.  If we're not, then m_currentTs is were we stopped.
    //
    uint64_t ts = m_running ? m_synchronizer->GetCurrentRealtime () : m_currentTs;
    NS_ASSERT_MSG (ts >= m_currentTs,
                   "RealtimeSimulatorImpl::ScheduleRealtimeNowWithContext(): schedule for time < m_currentTs");
    Scheduler::Event ev;
    ev.impl = impl;
    ev.key.m_ts = ts;
    ev.key.m_uid = m_uid;
    ev.key.m_context = context;
    m_uid++;
    m_unscheduledEvents++;
    m_events->Insert (ev);
    m_synchronizer->Signal ();
  }
}

void
RealtimeSimulatorImpl::ScheduleRealtimeNow (EventImpl *impl)
{
  NS_LOG_FUNCTION (this << impl);
  ScheduleRealtimeNowWithContext (GetContext (), impl);
}

Time
RealtimeSimulatorImpl::RealtimeNow (void) const
{
  return TimeStep (m_synchronizer->GetCurrentRealtime ());
}

EventId
RealtimeSimulatorImpl::ScheduleDestroy (EventImpl *impl)
{
  NS_LOG_FUNCTION (this << impl);

  EventId id;
  {
    std::unique_lock lock {m_mutex};

    //
    // Time doesn't really matter here (especially in realtime mode).  It is
    // overridden by the uid of DESTROY which identifies this as an event to be
    // executed at Simulator::Destroy time.
    //
    id = EventId (Ptr<EventImpl> (impl, false), m_currentTs, 0xffffffff, 
                  EventId::UID::DESTROY);
    m_destroyEvents.push_back (id);
    m_uid++;
  }

  return id;
}

Time
RealtimeSimulatorImpl::GetDelayLeft (const EventId &id) const
{
  //
  // If the event has expired, there is no delay until it runs.  It is not the
  // case that there is a negative time until it runs.
  //
  if (IsExpired (id))
    {
      return TimeStep (0);
    }

  return TimeStep (id.GetTs () - m_currentTs);
}

void
RealtimeSimulatorImpl::Remove (const EventId &id)
{
  if (id.GetUid () == EventId::UID::DESTROY)
    {
      // destroy events.
      for (DestroyEvents::iterator i = m_destroyEvents.begin ();
           i != m_destroyEvents.end ();
           i++)
        {
          if (*i == id)
            {
              m_destroyEvents.erase (i);
              break;
            }
        }
      return;
    }
  if (IsExpired (id))
    {
      return;
    }

  {
    std::unique_lock lock {m_mutex};

    Scheduler::Event event;
    event.impl = id.PeekEventImpl ();
    event.key.m_ts = id.GetTs ();
    event.key.m_context = id.GetContext ();
    event.key.m_uid = id.GetUid ();

    m_events->Remove (event);
    m_unscheduledEvents--;
    event.impl->Cancel ();
    event.impl->Unref ();
  }
}

void
RealtimeSimulatorImpl::Cancel (const EventId &id)
{
  if (IsExpired (id) == false)
    {
      id.PeekEventImpl ()->Cancel ();
    }
}

bool
RealtimeSimulatorImpl::IsExpired (const EventId &id) const
{
  if (id.GetUid () == EventId::UID::DESTROY)
    {
      if (id.PeekEventImpl () == 0
          || id.PeekEventImpl ()->IsCancelled ())
        {
          return true;
        }
      // destroy events.
      for (DestroyEvents::const_iterator i = m_destroyEvents.begin ();
           i != m_destroyEvents.end (); i++)
        {
          if (*i == id)
            {
              return false;
            }
        }
      return true;
    }

  //
  // If the time of the event is less than the current timestamp of the
  // simulator, the simulator has gone past the invocation time of the
  // event, so the statement ev.GetTs () < m_currentTs does mean that
  // the event has been fired even in realtime mode.
  //
  // The same is true for the next line involving the m_currentUid.
  //
  if (id.PeekEventImpl () == 0
      || id.GetTs () < m_currentTs
      || (id.GetTs () == m_currentTs && id.GetUid () <= m_currentUid)
      || id.PeekEventImpl ()->IsCancelled ())
    {
      return true;
    }
  else
    {
      return false;
    }
}

Time
RealtimeSimulatorImpl::GetMaximumSimulationTime (void) const
{
  return TimeStep (0x7fffffffffffffffLL);
}

// System ID for non-distributed simulation is always zero
uint32_t
RealtimeSimulatorImpl::GetSystemId (void) const
{
  return 0;
}

uint32_t
RealtimeSimulatorImpl::GetContext (void) const
{
  return m_currentContext;
}

uint64_t
RealtimeSimulatorImpl::GetEventCount (void) const
{
  return m_eventCount;
}

void
RealtimeSimulatorImpl::SetSynchronizationMode (enum SynchronizationMode mode)
{
  NS_LOG_FUNCTION (this << mode);
  m_synchronizationMode = mode;
}

RealtimeSimulatorImpl::SynchronizationMode
RealtimeSimulatorImpl::GetSynchronizationMode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_synchronizationMode;
}

void
RealtimeSimulatorImpl::SetHardLimit (Time limit)
{
  NS_LOG_FUNCTION (this << limit);
  m_hardLimit = limit;
}

Time
RealtimeSimulatorImpl::GetHardLimit (void) const
{
  NS_LOG_FUNCTION (this);
  return m_hardLimit;
}

} // namespace ns3
