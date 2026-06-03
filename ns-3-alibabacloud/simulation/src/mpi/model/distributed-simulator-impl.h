/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: George Riley <riley@ece.gatech.edu>
 *
 */

/**
 * \file
 * \ingroup mpi
 *  Declaration of classes  ns3::LbtsMessage and ns3::DistributedSimulatorImpl.
 */

#ifndef NS3_DISTRIBUTED_SIMULATOR_IMPL_H
#define NS3_DISTRIBUTED_SIMULATOR_IMPL_H

#include "ns3/simulator-impl.h"
#include "ns3/scheduler.h"
#include "ns3/event-impl.h"
#include "ns3/ptr.h"

#include <list>

namespace ns3 {

/**
 * \ingroup mpi
 *
 * \brief Structure used for all-reduce LBTS computation
 */
class LbtsMessage
{
public:
  LbtsMessage ()
    : m_txCount (0),
      m_rxCount (0),
      m_myId (0),
      m_isFinished (false)
  {
  }

  /**
   * \param rxc received count
   * \param txc transmitted count
   * \param id mpi rank
   * \param isFinished whether message is finished
   * \param t smallest time
   */
  LbtsMessage (uint32_t rxc, uint32_t txc, uint32_t id, bool isFinished, const Time& t)
    : m_txCount (txc),
      m_rxCount (rxc),
      m_myId (id),
      m_smallestTime (t),
      m_isFinished (isFinished)
  {
  }

  ~LbtsMessage ();

  /**
   * \return smallest time
   */
  Time GetSmallestTime ();
  /**
   * \return transmitted count
   */
  uint32_t GetTxCount ();
  /**
   * \return received count
   */
  uint32_t GetRxCount ();
  /**
   * \return id which corresponds to mpi rank
   */
  uint32_t GetMyId ();
  /**
   * \return true if system is finished
   */
  bool IsFinished ();

private:
  uint32_t m_txCount;         /**< Count of transmitted messages. */ 
  uint32_t m_rxCount;         /**< Count of received messages. */ 
  uint32_t m_myId;            /**< System Id of the rank sending this LBTS. */
  Time     m_smallestTime;    /**< Earliest next event timestamp. */
  bool     m_isFinished;      /**< \c true when this rank has no more events. */
};

/**
 * \ingroup simulator
 * \ingroup mpi
 *
 * \brief Distributed simulator implementation using lookahead
 */
class DistributedSimulatorImpl : public SimulatorImpl
{
public:
  /**
   *  Register this type.
   *  \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /** Default constructor. */
  DistributedSimulatorImpl ();
  /** Destructor. */
  ~DistributedSimulatorImpl ();

  // virtual from SimulatorImpl
  virtual void Destroy ();
  virtual bool IsFinished (void) const;
  virtual void Stop (void);
  virtual void Stop (Time const &delay);
  virtual EventId Schedule (Time const &delay, EventImpl *event);
  virtual void ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *event);
  virtual EventId ScheduleNow (EventImpl *event);
  virtual EventId ScheduleDestroy (EventImpl *event);
  virtual void Remove (const EventId &id);
  virtual void Cancel (const EventId &id);
  virtual bool IsExpired (const EventId &id) const;
  virtual void Run (void);
  virtual Time Now (void) const;
  virtual Time GetDelayLeft (const EventId &id) const;
  virtual Time GetMaximumSimulationTime (void) const;
  virtual void SetScheduler (ObjectFactory schedulerFactory);
  virtual uint32_t GetSystemId (void) const;
  virtual uint32_t GetContext (void) const;
  virtual uint64_t GetEventCount (void) const;

  /**
   * Add additional bound to lookahead constraints.
   *
   * This may be used if there are additional constraints on lookahead
   * in addition to the minimum inter rank latency time.  For example
   * when running ns-3 in a co-simulation setting the other simulators
   * may have tighter lookahead constraints.
   * 
   * The method may be invoked more than once, the minimum time will
   * be used to constrain lookahead.
   *
   * \param [in] lookAhead The maximum lookahead; must be > 0.
   */
  virtual void BoundLookAhead (const Time lookAhead);
  
private:
  // Inherited from Object
  virtual void DoDispose (void);

  /**
   * Calculate lookahead constraint based on network latency.
   *
   * The smallest cross-rank PointToPoint channel delay imposes
   * a constraint on the conservative PDES time window.  The
   * user may impose additional constraints on lookahead
   * using the ConstrainLookAhead() method.
   */
  void CalculateLookAhead (void);
  /**
   * Check if this rank is finished.  It's finished when there are
   * no more events or stop has been requested.
   *
   * \returns \c true when this rank is finished.
   */
  bool IsLocalFinished (void) const;

  /** Process the next event. */
  void ProcessOneEvent (void);
  /**
   * Get the timestep of the next event.
   *
   * If there are no more events the timestep is infinity.
   *
   * \return The next event timestep.
   */
  uint64_t NextTs (void) const;
  /**
   * Get the time of the next event, as returned by NextTs().
   *
   * \return The next event time stamp.
   */
  Time Next (void) const;

  /** Container type for the events to run at Simulator::Destroy(). */
  typedef std::list<EventId> DestroyEvents;

  /** The container of events to run at Destroy() */
  DestroyEvents m_destroyEvents;
  /** Flag calling for the end of the simulation. */
  bool m_stop;
  /** Are all parallel instances completed. */
  bool m_globalFinished;     
  /** The event priority queue. */
  Ptr<Scheduler> m_events;
  
  /** Next event unique id. */
  uint32_t m_uid;
  /** Unique id of the current event. */
  uint32_t m_currentUid;
  /** Timestamp of the current event. */
  uint64_t m_currentTs;
  /** Execution context of the current event. */
  uint32_t m_currentContext;
  /** The event count. */
  uint64_t m_eventCount;
  /**
   * Number of events that have been inserted but not yet scheduled,
   * not counting the "destroy" events; this is used for validation.
   */
  int m_unscheduledEvents;

  /**
   * Container for Lbts messages, one per rank.
   * Allocated once we know how many systems there are.
   */
  LbtsMessage* m_pLBTS;
  uint32_t     m_myId;        /**< MPI rank. */
  uint32_t     m_systemCount; /**< MPI communicator size. */
  Time         m_grantedTime; /**< End of current window. */
  static Time  m_lookAhead;   /**< Current window size. */

};

} // namespace ns3

#endif /* NS3_DISTRIBUTED_SIMULATOR_IMPL_H */
