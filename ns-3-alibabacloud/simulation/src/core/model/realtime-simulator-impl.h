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

#ifndef REALTIME_SIMULATOR_IMPL_H
#define REALTIME_SIMULATOR_IMPL_H

#include "simulator-impl.h"

#include "scheduler.h"
#include "synchronizer.h"
#include "event-impl.h"

#include "ptr.h"
#include "assert.h"
#include "log.h"

#include <list>
#include <mutex>
#include <thread>

/**
 * \file
 * \ingroup realtime
 * ns3::RealtimeSimulatorImpl declaration.
 */

namespace ns3 {

/**
 * \ingroup simulator
 * \defgroup realtime Realtime Simulator
 *
 * Realtime simulator implementation.
 */

/**
 * \ingroup realtime
 *
 * Realtime version of SimulatorImpl.
 */
class RealtimeSimulatorImpl : public SimulatorImpl
{
public:
  /**
   * Get the registered TypeId for this class.
   * \returns The TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * What to do when we can't maintain real time synchrony.
   */
  enum SynchronizationMode
  {
    /**
     * Make a best effort to keep synced to real-time.
     *
     * If we fall behind, keep going.
     */
    SYNC_BEST_EFFORT,
    /**
     * Keep to real time within the hard limit tolerance configured
     * with SetHardLimit, or die trying.
     *
     * Falling behind by more than the hard limit tolerance triggers
     * a fatal error.
     * \see SetHardLimit
     */
    SYNC_HARD_LIMIT,
  };

  /** Constructor. */
  RealtimeSimulatorImpl ();
  /** Destructor. */
  ~RealtimeSimulatorImpl ();

  // Inherited from SimulatorImpl
  virtual void Destroy ();
  virtual bool IsFinished (void) const;
  virtual void Stop (void);
  virtual void Stop (const Time &delay);
  virtual EventId Schedule (const Time &delay, EventImpl *event);
  virtual void ScheduleWithContext (uint32_t context, const Time &delay, EventImpl *event);
  virtual EventId ScheduleNow (EventImpl *event);
  virtual EventId ScheduleDestroy (EventImpl *event);
  virtual void Remove (const EventId &ev);
  virtual void Cancel (const EventId &ev);
  virtual bool IsExpired (const EventId &ev) const;
  virtual void Run (void);
  virtual Time Now (void) const;
  virtual Time GetDelayLeft (const EventId &id) const;
  virtual Time GetMaximumSimulationTime (void) const;
  virtual void SetScheduler (ObjectFactory schedulerFactory);
  virtual uint32_t GetSystemId (void) const;
  virtual uint32_t GetContext (void) const;
  virtual uint64_t GetEventCount (void) const;

  /** \copydoc ScheduleWithContext(uint32_t,const Time&,EventImpl*) */
  void ScheduleRealtimeWithContext (uint32_t context, const Time &delay, EventImpl *event);
  /**
   * Schedule a future event execution (in the same context).
   *
   * @param [in] delay Delay until the event expires.
   * @param [in] event The event to schedule.
   */
  void ScheduleRealtime (const Time &delay, EventImpl *event);
  /**
   * Schedule an event to run at the current virtual time.
   *
   * \param [in] context Event context.
   * \param [in] event The event to schedule.
   */
  void ScheduleRealtimeNowWithContext (uint32_t context, EventImpl *event);
  /**
   * Schedule an event to run at the current virtual time.
   *
   * @param [in] event The event to schedule.
   */
  void ScheduleRealtimeNow (EventImpl *event);
  /**
   * Get the current real time from the synchronizer.
   * \returns The current real time.
   */
  Time RealtimeNow (void) const;

  /**
   * Set the SynchronizationMode.
   *
   * \param [in] mode The new SynchronizationMode.
   */
  void SetSynchronizationMode (RealtimeSimulatorImpl::SynchronizationMode mode);
  /**
   * Get the SynchronizationMode.
   * \returns The current SynchronizationMode.
   */
  RealtimeSimulatorImpl::SynchronizationMode GetSynchronizationMode (void) const;

  /**
   * Set the fatal error threshold for SynchronizationMode SYNC_HARD_LIMIT.
   *
   * \param [in] limit The maximum amount of real time we are allowed to fall
   *     behind before we trigger a fatal error.
   */
  void SetHardLimit (Time limit);
  /**
   * Get the current fatal error threshold for SynchronizationMode
   * SYNC_HARD_LIMIT.
   *
   * \returns The hard limit threshold.
   */
  Time GetHardLimit (void) const;

private:
  /**
   * Is the simulator running?
   * \returns \c true if we are running.
   */
  bool Running (void) const;
  /**
   * Check that the Synchronizer is locked to the real time clock.
   * \return \c true if the Synchronizer is locked.
   */
  bool Realtime (void) const;
  /**
   * Get the timestep of the next event.
   * \returns The timestep of the next event.
   */
  uint64_t NextTs (void) const;
  /** Process the next event. */
  void ProcessOneEvent (void);
  /** Destructor implementation. */
  virtual void DoDispose (void);

  /** Container type for events to be run at destroy time. */
  typedef std::list<EventId> DestroyEvents;
  /** Container for events to be run at destroy time. */
  DestroyEvents m_destroyEvents;
  /** Has the stopping condition been reached? */
  bool m_stop;
  /** Is the simulator currently running. */
  bool m_running;

  /**
   * \name Mutex-protected variables.
   *
   * These variables are protected by #m_mutex.
   */
  /**@{*/
  /** The event list. */
  Ptr<Scheduler> m_events;
  /**< Number of events in the event list. */
  int m_unscheduledEvents;
  /**< Unique id for the next event to be scheduled. */
  uint32_t m_uid;
  /**< Unique id of the current event. */
  uint32_t m_currentUid;
  /**< Timestep of the current event. */
  uint64_t m_currentTs;
  /**< Execution context. */
  uint32_t m_currentContext;
  /** The event count. */
  uint64_t m_eventCount;
  /**@}*/

  /** Mutex to control access to key state. */
  mutable std::mutex m_mutex;

  /** The synchronizer in use to track real time. */
  Ptr<Synchronizer> m_synchronizer;

  /** SynchronizationMode policy. */
  SynchronizationMode m_synchronizationMode;

  /** The maximum allowable drift from real-time in SYNC_HARD_LIMIT mode. */
  Time m_hardLimit;

  /** Main thread. */
  std::thread::id m_main;
};

} // namespace ns3

#endif /* REALTIME_SIMULATOR_IMPL_H */
