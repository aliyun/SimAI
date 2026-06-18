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

#ifndef SYNCHRONIZER_H
#define SYNCHRONIZER_H

#include <stdint.h>
#include "nstime.h"
#include "object.h"

/**
 * @file
 * @ingroup realtime
 * ns3::Synchronizer declaration.
 */

namespace ns3 {

/**
 * @ingroup realtime
 * @brief Base class used for synchronizing the simulation events to some
 * real time "wall clock."
 *
 * The simulation clock is maintained as a 64-bit integer in a unit specified
 * by the user through the Time::SetResolution function. This means that
 * it is not possible to specify event expiration times with anything better
 * than this user-specified accuracy.  We use this clock for the simulation
 * time.
 *
 * The real-time clock is maintained as a 64-bit integer count of nanoseconds.
 *
 * The synchronization between the simulation clock and the real-time clock
 * is maintained using a combination of sleep-waiting, busy-waiting and a
 * feedback loop.
 */
class Synchronizer : public Object
{
public:
  /**
   * Get the registered TypeId for this class.
   * @returns The TypeId.
   */
  static TypeId GetTypeId (void);

  /** Constructor. */
  Synchronizer ();
  /** Destructor. */
  virtual ~Synchronizer ();

  /**
   * @brief Return true if this synchronizer is actually synchronizing to a
   * realtime clock.
   *
   * The simulator sometimes needs to know this.
   *
   * @returns @c true if locked with realtime, @c false if not.
   */
  bool Realtime (void);

  /**
   * @brief Retrieve the value of the origin of the underlying normalized wall
   * clock time in simulator timestep units.
   *
   * @returns The normalized wall clock time (in Time resolution units).
   * @see SetOrigin
   */
  uint64_t GetCurrentRealtime (void);

  /**
   * @brief Establish a correspondence between a simulation time and the
   * synchronizer real time.
   *
   * This method is expected to be called at the "instant" before simulation
   * begins.  At this point, simulation time = 0, and a
   * set = 0 in this method.  We then associate this time with the current
   * value of the real time clock that will be used to actually perform the
   * synchronization.
   *
   * Subclasses are expected to implement the corresponding DoSetOrigin pure
   * virtual method to do the actual real-time-clock-specific work
   * of making the correspondence mentioned above.
   *
   * @param [in] ts The simulation time we should use as the origin (in
   *     Time resolution units).
   * @see DoSetOrigin
   */
  void SetOrigin (uint64_t ts);

  /**
   * @brief Retrieve the value of the origin of the simulation time in
   * Time.resolution units.
   *
   * @returns The simulation time used as the origin (in Time resolution units).
   * @see SetOrigin
   */
  uint64_t GetOrigin (void);

  /**
   * @brief Retrieve the difference between the real time clock used to
   * synchronize the simulation and the simulation time (in
   * Time resolution units).
   *
   * @param [in] ts Simulation time in Time resolution units.
   * @returns Simulation Time (in Time resolution units)
   *     minus the origin time (stored internally in nanosecond units).
   * @see SetOrigin
   * @see DoGetDrift
   */
  int64_t GetDrift (uint64_t ts);

  /**
   * @brief Wait until the real time is in sync with the specified simulation
   * time or until the synchronizer is Signalled.
   *
   * This is where the real work of synchronization is done.  The @c tsCurrent
   * argument is the simulation time.  The job of Synchronize is to
   * translate from simulation time to synchronizer time (in a perfect world
   * this is the same time) and then figure out how long in real-time it needs
   * to wait until that synchronizer / simulation time comes around.
   *
   * Subclasses are expected to implement the corresponding DoSynchronize pure
   * virtual method to do the actual real-time-clock-specific work of waiting
   * (either busy-waiting or sleeping, or some combination thereof) until the
   * requested simulation time.
   *
   * @param [in] tsCurrent The current simulation time (in Time resolution units).
   * @param [in] tsDelay The simulation time we need to wait for (in Time
   *     resolution units).
   * @returns @c true if the function ran to completion,
   *          @c false if it was interrupted by a Signal.
   * @see DoSynchronize
   * @see Signal
   */
  bool Synchronize (uint64_t tsCurrent, uint64_t tsDelay);

  /**
   * @brief Tell a possible simulator thread waiting in the Synchronize method
   * that an event has happened which demands a reevaluation of the wait time.
   *
   * This will cause the thread to wake and return to the simulator proper
   * where it can get its bearings.
   *
   * @see Synchronize
   * @see DoSignal
   */
  void Signal (void);

  /**
   * @brief Set the condition variable that tells a possible simulator thread
   * waiting in the Synchronize method that an event has happened which demands
   * a reevaluation of the wait time.
   *
   * @param [in] cond The new value for the condition variable.
   *
   * @see Signal
   */
  void SetCondition (bool cond);

  /**
   * @brief Ask the synchronizer to remember what time it is.
   *
   * Typically used with EventEnd to determine the real execution time
   * of a simulation event.
   *
   * @see EventEnd
   */
  void EventStart (void);

  /**
   * @brief Ask the synchronizer to return the time step between the instant
   * remembered during EventStart and now.
   *
   * Used in conjunction with EventStart to determine the real execution time
   * of a simulation event.
   *
   * @returns The elapsed real time, in ns.
   * @see EventStart
   */
  uint64_t EventEnd (void);

protected:
  /**
   * @brief Establish a correspondence between a simulation time and a
   * wall-clock (real) time.
   *
   * There are three timelines involved here:  the simulation (virtual) time,
   * the (absolute) wall-clock time and the (relative) synchronizer real time.
   * Calling this method makes a correspondence between the origin of the
   * synchronizer time and the current wall-clock time.
   *
   * This method is expected to be called at the "instant" before simulation
   * begins.  At this point, simulation time = 0, and synchronizer time is
   * set = 0 in this method.  We then associate this time with the current
   * value of the real time clock that will be used to actually perform the
   * synchronization.
   *
   * Subclasses are expected to implement this method to do the actual
   * real-time-clock-specific work of making the correspondence mentioned above.
   * for example, this is where the differences between Time parameters and
   * parameters to clock_nanosleep would be dealt with.
   *
   * @param [in] ns The simulation time we need to use as the origin (normalized to
   *    nanosecond units).
   * @see SetOrigin
   */
  virtual void DoSetOrigin (uint64_t ns) = 0;

  /**
   * @brief Return @c true if this synchronizer is actually synchronizing to a
   * realtime clock.
   *
   * The simulator sometimes needs to know this.
   *
   * Subclasses are expected to implement this method to tell the outside world
   * whether or not they are synchronizing to a realtime clock.
   *
   * @returns @c true if locked with realtime, @c false if not.
   */
  virtual bool DoRealtime (void) = 0;

  /**
   * @brief Retrieve the value of the origin of the underlying normalized wall
   * clock time in Time resolution units.
   *
   * Subclasses are expected to implement this method to do the actual
   * real-time-clock-specific work of getting the current time.
   *
   * @returns The normalized wall clock time (in nanosecond units).
   * @see SetOrigin
   */
  virtual uint64_t DoGetCurrentRealtime (void) = 0;

  /**
   * @brief Wait until the real time is in sync with the specified simulation
   * time.
   *
   * This is where the real work of synchronization is done.  The
   * @c nsCurrent argument is the simulation time (in ns).  The job of
   * DoSynchronize is to translate from simulation time to synchronizer time
   * (in a perfect world these are the same time) and then figure out
   * how long in real-time it needs to wait until that
   * synchronizer / simulation time comes around.
   *
   * Subclasses are expected to implement this method to do the actual
   * real-time-clock-specific work of waiting (either busy-waiting or sleeping,
   * or some combination) until the requested simulation time.
   *
   * @param [in] nsCurrent The current simulation time (in nanosecond units).
   * @param [in] nsDelay The simulation time we need to wait for (normalized to
   * nanosecond units).
   * @returns @c true if the function ran to completion,
   *          @c false if it was interrupted by a Signal.
   * @see Synchronize
   * @see Signal
   */
  virtual bool DoSynchronize (uint64_t nsCurrent, uint64_t nsDelay) = 0;

  /**
   * @brief Tell a possible simulator thread waiting in the
   * DoSynchronize method that an event has happened which
   * demands a reevaluation of the wait time.
   *
   * @see Signal
   */
  virtual void DoSignal (void) = 0;

  /**
   * @brief Set the condition variable to tell a possible simulator thread
   * waiting in the Synchronize method that an event has happened which
   * demands a reevaluation of the wait time.
   *
   * @param [in] cond The new value for the condition variable.
   * @see SetCondition
   */
  virtual void DoSetCondition (bool cond) = 0;

  /**
   * @brief Get the drift between the real time clock used to synchronize
   * the simulation and the current simulation time.
   *
   * @param [in] ns Simulation time in ns.
   * @returns Drift in ns units.
   * @see SetOrigin
   * @see GetDrift
   */
  virtual int64_t DoGetDrift (uint64_t ns) = 0;

  /**
   * @brief Record the normalized real time at which the current
   * event is starting execution.
   */
  virtual void DoEventStart (void) = 0;
  /**
   * @brief Return the amount of real time elapsed since the last call
   * to EventStart.
   *
   * @returns The elapsed real time, in ns.
   */
  virtual uint64_t DoEventEnd (void) = 0;

  /** The real time, in ns, when SetOrigin was called. */
  uint64_t m_realtimeOriginNano;
  /** The simulation time, in ns, when SetOrigin was called. */
  uint64_t m_simOriginNano;

private:
  /**
   * @brief Convert a simulator time step (in Time resolution units)
   * to a normalized time step in nanosecond units.
   *
   * @param [in] ts The simulation time step to be normalized.
   * @returns The simulation time step normalized to nanosecond units.
   */
  uint64_t TimeStepToNanosecond (uint64_t ts);

  /**
   * @brief Convert a normalized nanosecond time step into a
   * simulator time step (in Time resolution units).
   *
   * @param [in] ns The nanosecond count step to be converted
   * @returns The simulation time step to be interpreted in appropriate units.
   */
  uint64_t NanosecondToTimeStep (uint64_t ns);
};

} // namespace ns3

#endif /* SYNCHRONIZER_H */
