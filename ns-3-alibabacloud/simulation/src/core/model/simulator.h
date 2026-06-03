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

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "event-id.h"
#include "event-impl.h"
#include "make-event.h"
#include "nstime.h"

#include "object-factory.h"

#include <stdint.h>
#include <string>

/**
 * @file
 * @ingroup simulator
 * ns3::Simulator declaration.
 */

namespace ns3 {

class SimulatorImpl;
class Scheduler;

/**
 * @ingroup core
 * @defgroup simulator Simulator
 * @brief Control the virtual time and the execution of simulation events.
 */
/**
 * @ingroup simulator
 *
 * @brief Control the scheduling of simulation events.
 *
 * The internal simulation clock is maintained
 * as a 64-bit integer in a unit specified by the user
 * through the Time::SetResolution function. This means that it is
 * not possible to specify event expiration times with anything better
 * than this user-specified accuracy. Events whose expiration time is
 * the same modulo this accuracy are scheduled in FIFO order: the
 * first event inserted in the scheduling queue is scheduled to
 * expire first.
 *
 * A simple example of how to use the Simulator class to schedule events
 * is shown in sample-simulator.cc:
 * @include src/core/examples/sample-simulator.cc
 */
class Simulator
{
public:
  /**
   * @param [in] impl A new simulator implementation.
   *
   * The simulator provides a mechanism to swap out different implementations.
   * For example, the default implementation is a single-threaded simulator
   * that performs no realtime synchronization.  By calling this method, you
   * can substitute in a new simulator implementation that might be multi-
   * threaded and synchronize events to a realtime clock.
   *
   * The simulator implementation can be set when the simulator is not
   * running.
   */
  static void SetImplementation (Ptr<SimulatorImpl> impl);

  /**
   * @brief Get the SimulatorImpl singleton.
   *
   * @internal
   * If the SimulatorImpl singleton hasn't been created yet,
   * this function does so.  At the same time it also creates
   * the Scheduler.  Both of these respect the global values
   * which may have been set from the command line or through
   * the Config system.
   *
   * As a side effect we also call LogSetTimePrinter() and
   * LogSetNodePrinter() with the default implementations
   * since we can't really do any logging until we have
   * a SimulatorImpl and Scheduler.

   * @return The SimulatorImpl singleton.
   */
  static Ptr<SimulatorImpl> GetImplementation (void);

  /**
   * @brief Set the scheduler type with an ObjectFactory.
   * @param [in] schedulerFactory The configured ObjectFactory.
   *
   * The event scheduler can be set at any time: the events scheduled
   * in the previous scheduler will be transferred to the new scheduler
   * before we start to use it.
   */
  static void SetScheduler (ObjectFactory schedulerFactory);

  /**
   * Execute the events scheduled with ScheduleDestroy().
   *
   * This method is typically invoked at the end of a simulation
   * to avoid false-positive reports by a leak checker.
   * After this method has been invoked, it is actually possible
   * to restart a new simulation with a set of calls to Simulator::Run,
   * Simulator::Schedule and Simulator::ScheduleWithContext.
   */
  static void Destroy (void);

  /**
   * Check if the simulation should finish.
   *
   * Reasons to finish are because there are
   * no more events lefts to be scheduled, or if simulation
   * time has already reached the "stop time" (see Simulator::Stop()).
   *
   * @return @c true if no more events or stop time reached.
   */
  static bool IsFinished (void);

  /**
   * Run the simulation.
   *
   * The simulation will run until one of:
   *   - No events are present anymore
   *   - The user called Simulator::Stop
   *   - The user called Simulator::Stop with a stop time and the
   *     expiration time of the next event to be processed
   *     is greater than or equal to the stop time.
   */
  static void Run (void);

  /**
   * Tell the Simulator the calling event should be the last one
   * executed.
   *
   * If a running event invokes this method, it will be the last
   * event executed by the Simulator::Run method before
   * returning to the caller.
   */
  static void Stop (void);

  /**
   * Schedule the time delay until the Simulator should stop.
   *
   * Force the Simulator::Run method to return to the caller when the
   * expiration time of the next event to be processed is greater than
   * or equal to the stop time.  The stop time is relative to the
   * current simulation time.
   * @param [in] delay The stop time, relative to the current time.
   */
  static void Stop (const Time &delay);

  /**
   * Get the current simulation context.
   *
   * The simulation context is the ns-3 notion of a Logical Process.
   * Events in a single context should only modify state associated with
   * that context. Events for objects in other contexts should be
   * scheduled with ScheduleWithContext() to track the context switches.
   * In other words, events in different contexts should be mutually
   * thread safe, by not modify overlapping model state.
   *
   * In circumstances where the context can't be determined, such as
   * during object initialization, the \c enum value \c NO_CONTEXT
   * should be used.
   *
   * @return The current simulation context
   */
  static uint32_t GetContext (void);

  /**
   * Context enum values.
   *
   * \internal
   * This enum type is fixed to match the representation size
   * of simulation context.
   */
  enum : uint32_t
  {
    /**
     * Flag for events not associated with any particular context.
     */
    NO_CONTEXT = 0xffffffff
  };

  /**
   * Get the number of events executed.
   * \returns The total number of events executed.
   */
  static uint64_t GetEventCount (void);


  /**
   * @name Schedule events (in the same context) to run at a future time.
   */
  /** @{ */
  /**
   * Schedule an event to expire after @p delay.
   * This can be thought of as scheduling an event
   * for the current simulation time plus the @p delay  passed as a
   * parameter.
   *
   * We leverage SFINAE to discard this overload if the second argument is
   * convertible to Ptr<EventImpl> or is a function pointer.
   *
   * @tparam FUNC @deduced Template type for the function to invoke.
   * @tparam Ts @deduced Argument types.
   * @param [in] delay The relative expiration time of the event.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to MakeEvent.
   * @returns The id for the scheduled event.
   */
  template <typename FUNC,
            typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type = 0,
            typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type = 0,
            typename... Ts>
  static EventId Schedule (Time const &delay, FUNC f, Ts&&... args);

  /**
   * Schedule an event to expire after @p delay.
   * This can be thought of as scheduling an event
   * for the current simulation time plus the @p delay  passed as a
   * parameter.
   *
   * @tparam Us @deduced Formal function argument types.
   * @tparam Ts @deduced Actual function argument types.
   * When the event expires (when it becomes due to be run), the
   * function will be invoked with any supplied arguments.
   * @param [in] delay The relative expiration time of the event.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to the invoked function.
   * @returns The id for the scheduled event.
   */
  template <typename... Us, typename... Ts>
  static EventId Schedule (Time const &delay, void (*f)(Us...), Ts&&... args);
  /** @} */  // Schedule events (in the same context) to run at a future time.

  /**
   * @name Schedule events (in a different context) to run now or at a future time.
   *
   * See @ref main-test-sync.cc for example usage.
   */
  /** @{ */
  /**
   * Schedule an event with the given context.
   * A context of 0xffffffff means no context is specified.
   * This method is thread-safe: it can be called from any thread.
   *
   * We leverage SFINAE to discard this overload if the second argument is
   * convertible to Ptr<EventImpl> or is a function pointer.
   *
   * @tparam FUNC @deduced Template type for the function to invoke.
   * @tparam Ts @deduced Argument types.
   * @param [in] context User-specified context parameter
   * @param [in] delay The relative expiration time of the event.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to MakeEvent.
   */
  template <typename FUNC,
            typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type = 0,
            typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type = 0,
            typename... Ts>
  static void ScheduleWithContext (uint32_t context, Time const &delay, FUNC f, Ts&&... args);

  /**
   * Schedule an event with the given context.
   * A context of 0xffffffff means no context is specified.
   * This method is thread-safe: it can be called from any thread.
   *
   * @tparam Us @deduced Formal function argument types.
   * @tparam Ts @deduced Actual function argument types.
   * @param [in] context User-specified context parameter
   * @param [in] delay The relative expiration time of the event.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to the invoked function.
   */
  template <typename... Us, typename... Ts>
  static void ScheduleWithContext (uint32_t context, Time const &delay, void (*f)(Us...), Ts&&... args);
  /** @} */  // Schedule events (in a different context) to run now or at a future time.

  /**
   * @name Schedule events (in the same context) to run now.
   */
  /** @{ */
  /**
   * Schedule an event to expire Now. All events scheduled to
   * to expire "Now" are scheduled FIFO, after all normal events
   * have expired.
   *
   * We leverage SFINAE to discard this overload if the second argument is
   * convertible to Ptr<EventImpl> or is a function pointer.
   *
   * @tparam FUNC @deduced Template type for the function to invoke.
   * @tparam Ts @deduced Actual function argument types.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to the invoked function.
   * @return The EventId of the scheduled event.
   */
  template <typename FUNC,
            typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type = 0,
            typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type = 0,
            typename... Ts>
  static EventId ScheduleNow (FUNC f, Ts&&... args);

  /**
   * Schedule an event to expire Now. All events scheduled to
   * to expire "Now" are scheduled FIFO, after all normal events
   * have expired.
   *
   * @tparam Us @deduced Formal function argument types.
   * @tparam Ts @deduced Actual function argument types.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to MakeEvent.
   * @return The EventId of the scheduled event.
   */
  template <typename... Us, typename... Ts>
  static EventId ScheduleNow (void (*f)(Us...), Ts&&... args);

  /** @} */  // Schedule events (in the same context) to run now.

  /**
   * @name Schedule events to run at the end of the simulation, when Simulator:Destroy() is called.
   */
  /** @{ */
  /**
   * Schedule an event to run at the end of the simulation, when Simulator::Destroy() is called.
   * All events scheduled to expire at "Destroy" time are scheduled FIFO,
   * after all normal events have expired and only when
   * Simulator::Destroy is invoked.
   *
   * We leverage SFINAE to discard this overload if the second argument is
   * convertible to Ptr<EventImpl> or is a function pointer.
   *
   * @tparam FUNC @deduced Template type for the function to invoke.
   * @tparam Ts @deduced Actual function argument types.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to MakeEvent.
   * @return The EventId of the scheduled event.
   */
  template <typename FUNC,
            typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type = 0,
            typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type = 0,
            typename... Ts>
  static EventId ScheduleDestroy (FUNC f, Ts&&... args);

  /**
   * Schedule an event to run at the end of the simulation, when Simulator::Destroy() is called.
   * All events scheduled to expire at "Destroy" time are scheduled FIFO,
   * after all normal events have expired and only when
   * Simulator::Destroy is invoked.
   *
   * @tparam Us @deduced Formal function argument types.
   * @tparam Ts @deduced Actual function argument types.
   * @param [in] f The function to invoke.
   * @param [in] args Arguments to pass to MakeEvent.
   * @return The EventId of the scheduled event.
   */
  template <typename... Us, typename... Ts>
  static EventId ScheduleDestroy (void (*f)(Us...), Ts&&... args);

  /** @} */  // Schedule events to run when Simulator:Destroy() is called.

  /**
   * Remove an event from the event list.
   *
   * This method has the same visible effect as the
   * ns3::EventId::Cancel method
   * but its algorithmic complexity is much higher: it has often
   * O(log(n)) complexity, sometimes O(n), sometimes worse.
   * Note that it is not possible to remove events which were scheduled
   * for the "destroy" time. Doing so will result in a program error (crash).
   *
   * @param [in] id The event to remove from the list of scheduled events.
   */
  static void Remove (const EventId &id);

  /**
   * Set the cancel bit on this event: the event's associated function
   * will not be invoked when it expires.
   *
   * This method has the same visible effect as the
   * ns3::Simulator::Remove method but its algorithmic complexity is
   * much lower: it has O(1) complexity.
   * This method has the exact same semantics as ns3::EventId::Cancel.
   * Note that it is not possible to cancel events which were scheduled
   * for the "destroy" time. Doing so will result in a program error (crash).
   *
   * @param [in] id the event to cancel
   */
  static void Cancel (const EventId &id);

  /**
   * Check if an event has already run or been cancelled.
   *
   * This method has O(1) complexity.
   * Note that it is not possible to test for the expiration of
   * events which were scheduled for the "destroy" time. Doing so
   * will result in a program error (crash).
   * An event is said to "expire" when it starts being scheduled
   * which means that if the code executed by the event calls
   * this function, it will get true.
   *
   * @param [in] id The event to test for expiration.
   * @returns @c true if the event has expired, false otherwise.
   */
  static bool IsExpired (const EventId &id);

  /**
   * Return the current simulation virtual time.
   *
   * @returns The current virtual time.
   */
  static Time Now (void);

  /**
   * Get the remaining time until this event will execute.
   *
   * @param [in] id The event id to analyse.
   * @return The delay left until the input event id expires.
   *          if the event is not running, this method returns
   *          zero.
   */
  static Time GetDelayLeft (const EventId &id);

  /**
   * Get the maximum representable simulation time.
   *
   * @return The maximum simulation time at which an event
   *          can be scheduled.
   *
   * The returned value will always be bigger than or equal to Simulator::Now.
   */
  static Time GetMaximumSimulationTime (void);

  /**
   * Schedule a future event execution (in the same context).
   *
   * @param [in] delay Delay until the event expires.
   * @param [in] event The event to schedule.
   * @returns A unique identifier for the newly-scheduled event.
   */
  static EventId Schedule (const Time &delay, const Ptr<EventImpl> &event);

  /**
   * Schedule a future event execution (in a different context).
   * This method is thread-safe: it can be called from any thread.
   *
   * @param [in] delay Delay until the event expires.
   * @param [in] context Event context.
   * @param [in] event The event to schedule.
   */
  static void ScheduleWithContext (uint32_t context, const Time &delay, EventImpl *event);

  /**
   * Schedule an event to run at the end of the simulation, after
   * the Stop() time or condition has been reached.
   *
   * @param [in] event The event to schedule.
   * @returns A unique identifier for the newly-scheduled event.
   */
  static EventId ScheduleDestroy (const Ptr<EventImpl> &event);

  /**
   * Schedule an event to run at the current virtual time.
   *
   * @param [in] event The event to schedule.
   * @returns A unique identifier for the newly-scheduled event.
   */
  static EventId ScheduleNow (const Ptr<EventImpl> &event);

  /**
   * Get the system id of this simulator.
   *
   * The system id is the identifier for this simulator instance
   * in a distributed simulation.  For MPI this is the MPI rank.
   * @return The system id for this simulator.
   */
  static uint32_t GetSystemId (void);

private:
  /** Default constructor. */
  Simulator ();
  /** Destructor. */
  ~Simulator ();

  /**
   * Implementation of the various Schedule methods.
   * @param [in] delay Delay until the event should execute.
   * @param [in] event The event to execute.
   * @return The EventId.
   */
  static EventId DoSchedule (Time const &delay, EventImpl *event);
  /**
   * Implementation of the various ScheduleNow methods.
   * @param [in] event The event to execute.
   * @return The EventId.
   */
  static EventId DoScheduleNow (EventImpl *event);
  /**
   * Implementation of the various ScheduleDestroy methods.
   * @param [in] event The event to execute.
   * @return The EventId.
   */
  static EventId DoScheduleDestroy (EventImpl *event);

};  // class Simulator

/**
 * @ingroup simulator
 * @brief create an ns3::Time instance which contains the
 *        current simulation time.
 *
 * This is really a shortcut for the ns3::Simulator::Now method.
 * It is typically used as shown below to schedule an event
 * which expires at the absolute time "2 seconds":
 * @code
 *   Simulator::Schedule (Seconds (2.0) - Now (), &my_function);
 * @endcode
 * @return The current simulation time.
 */
Time Now (void);

} // namespace ns3


/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

namespace ns3 {

// Doxygen has trouble with static template functions in a class:
// it treats the in-class declaration as different from the
// out of class definition, so makes two entries in the member list.  Ugh

template <typename FUNC,
          typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type,
          typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type,
          typename... Ts>
EventId Simulator::Schedule (Time const &delay, FUNC f, Ts&&... args)
{
  return DoSchedule (delay, MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename... Us, typename... Ts>
EventId Simulator::Schedule (Time const &delay, void (*f)(Us...), Ts&&... args)
{
  return DoSchedule (delay, MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename FUNC,
          typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type,
          typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type,
          typename... Ts>
void Simulator::ScheduleWithContext (uint32_t context, Time const &delay, FUNC f, Ts&&... args)
{
  return ScheduleWithContext (context, delay, MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename... Us, typename... Ts>
void Simulator::ScheduleWithContext (uint32_t context, Time const &delay, void (*f)(Us...), Ts&&... args)
{
  return ScheduleWithContext (context, delay, MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename FUNC,
          typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type,
          typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type,
          typename... Ts>
EventId
Simulator::ScheduleNow (FUNC f, Ts&&... args)
{
  return DoScheduleNow (MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename... Us, typename... Ts>
EventId
Simulator::ScheduleNow (void (*f)(Us...), Ts&&... args)
{
  return DoScheduleNow (MakeEvent (f, std::forward<Ts> (args)...));
}


template <typename FUNC,
          typename std::enable_if<!std::is_convertible<FUNC, Ptr<EventImpl>>::value,int>::type,
          typename std::enable_if<!std::is_function<typename std::remove_pointer<FUNC>::type>::value,int>::type,
          typename... Ts>
EventId
Simulator::ScheduleDestroy (FUNC f, Ts&&... args)
{
  return DoScheduleDestroy (MakeEvent (f, std::forward<Ts> (args)...));
}

template <typename... Us, typename... Ts>
EventId
Simulator::ScheduleDestroy (void (*f)(Us...), Ts&&... args)
{
  return DoScheduleDestroy (MakeEvent (f, std::forward<Ts> (args)...));
}

} // namespace ns3

#endif /* SIMULATOR_H */
