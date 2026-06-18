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
 *  Implementation of classes  ns3::LbtsMessage and ns3::DistributedSimulatorImpl.
 */

#include "distributed-simulator-impl.h"
#include "granted-time-window-mpi-interface.h"
#include "mpi-interface.h"

#include "ns3/simulator.h"
#include "ns3/scheduler.h"
#include "ns3/event-impl.h"
#include "ns3/channel.h"
#include "ns3/node-container.h"
#include "ns3/ptr.h"
#include "ns3/pointer.h"
#include "ns3/assert.h"
#include "ns3/log.h"

#include <mpi.h>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DistributedSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED (DistributedSimulatorImpl);

LbtsMessage::~LbtsMessage ()
{
}

Time
LbtsMessage::GetSmallestTime ()
{
  return m_smallestTime;
}

uint32_t
LbtsMessage::GetTxCount ()
{
  return m_txCount;
}

uint32_t
LbtsMessage::GetRxCount ()
{
  return m_rxCount;
}
uint32_t
LbtsMessage::GetMyId ()
{
  return m_myId;
}

bool
LbtsMessage::IsFinished ()
{
  return m_isFinished;
}

/**
 * Initialize m_lookAhead to maximum, it will be constrained by
 * user supplied time via BoundLookAhead and the
 * minimum latency network between ranks.
 */
Time DistributedSimulatorImpl::m_lookAhead = Time::Max();

TypeId
DistributedSimulatorImpl::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DistributedSimulatorImpl")
    .SetParent<SimulatorImpl> ()
    .SetGroupName ("Mpi")
    .AddConstructor<DistributedSimulatorImpl> ()
  ;
  return tid;
}

DistributedSimulatorImpl::DistributedSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);

  m_myId = MpiInterface::GetSystemId ();
  m_systemCount = MpiInterface::GetSize ();

  // Allocate the LBTS message buffer
  m_pLBTS = new LbtsMessage[m_systemCount];
  m_grantedTime = Seconds (0);

  m_stop = false;
  m_globalFinished = false;
  m_uid = EventId::UID::VALID;
  m_currentUid = EventId::UID::INVALID;
  m_currentTs = 0;
  m_currentContext = Simulator::NO_CONTEXT;
  m_unscheduledEvents = 0;
  m_eventCount = 0;
  m_events = 0;
}

DistributedSimulatorImpl::~DistributedSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);
}

void
DistributedSimulatorImpl::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  while (!m_events->IsEmpty ())
    {
      Scheduler::Event next = m_events->RemoveNext ();
      next.impl->Unref ();
    }
  m_events = 0;
  delete [] m_pLBTS;
  SimulatorImpl::DoDispose ();
}

void
DistributedSimulatorImpl::Destroy ()
{
  NS_LOG_FUNCTION (this);

  while (!m_destroyEvents.empty ())
    {
      Ptr<EventImpl> ev = m_destroyEvents.front ().PeekEventImpl ();
      m_destroyEvents.pop_front ();
      NS_LOG_LOGIC ("handle destroy " << ev);
      if (!ev->IsCancelled ())
        {
          ev->Invoke ();
        }
    }

  MpiInterface::Destroy ();
}


void
DistributedSimulatorImpl::CalculateLookAhead (void)
{
  NS_LOG_FUNCTION (this);

  /* If runnning sequential simulation can ignore lookahead */
  if (MpiInterface::GetSize () <= 1)
    {
      m_lookAhead = Seconds (0);
    }
  else
    {
      NodeContainer c = NodeContainer::GetGlobal ();
      for (NodeContainer::Iterator iter = c.Begin (); iter != c.End (); ++iter)
        {
          if ((*iter)->GetSystemId () != MpiInterface::GetSystemId ())
            {
              continue;
            }

          for (uint32_t i = 0; i < (*iter)->GetNDevices (); ++i)
            {
              Ptr<NetDevice> localNetDevice = (*iter)->GetDevice (i);
              // only works for p2p links currently
              if (!localNetDevice->IsPointToPoint ())
                {
                  continue;
                }
              Ptr<Channel> channel = localNetDevice->GetChannel ();
              if (channel == 0)
                {
                  continue;
                }

              // grab the adjacent node
              Ptr<Node> remoteNode;
              if (channel->GetDevice (0) == localNetDevice)
                {
                  remoteNode = (channel->GetDevice (1))->GetNode ();
                }
              else
                {
                  remoteNode = (channel->GetDevice (0))->GetNode ();
                }

              // if it's not remote, don't consider it
              if (remoteNode->GetSystemId () == MpiInterface::GetSystemId ())
                {
                  continue;
                }

              // compare delay on the channel with current value of
              // m_lookAhead.  if delay on channel is smaller, make
              // it the new lookAhead.
              TimeValue delay;
              channel->GetAttribute ("Delay", delay);

              if (delay.Get () < m_lookAhead)
                {
                  m_lookAhead = delay.Get ();
                }
            }
        }
    }

  // m_lookAhead is now set
  m_grantedTime = m_lookAhead;

  /*
   * Compute the maximum inter-task latency and use that value
   * for tasks with no inter-task links.
   *
   * Special processing for edge cases.  For tasks that have no
   * nodes need to determine a reasonable lookAhead value.  Infinity
   * would work correctly but introduces a performance issue; tasks
   * with an infinite lookAhead would execute all their events
   * before doing an AllGather resulting in very bad load balance
   * during the first time window.  Since all tasks participate in
   * the AllGather it is desirable to have all the tasks advance in
   * simulation time at a similar rate assuming roughly equal events
   * per unit of simulation time in order to equalize the amount of
   * work per time window.
   */
  long sendbuf;
  long recvbuf;

  /* Tasks with no inter-task links do not contribute to max */
  if (m_lookAhead == GetMaximumSimulationTime ())
    {
      sendbuf = 0;
    }
  else
    {
      sendbuf  = m_lookAhead.GetInteger ();
    }

  MPI_Allreduce (&sendbuf, &recvbuf, 1, MPI_LONG, MPI_MAX, MpiInterface::GetCommunicator ());

  /* For nodes that did not compute a lookahead use max from ranks
   * that did compute a value.  An edge case occurs if all nodes have
   * no inter-task links (max will be 0 in this case). Use infinity so all tasks
   * will proceed without synchronization until a single AllGather
   * occurs when all tasks have finished.
   */
  if (m_lookAhead == GetMaximumSimulationTime () && recvbuf != 0)
    {
      m_lookAhead = Time (recvbuf);
      m_grantedTime = m_lookAhead;
    }
}

void
DistributedSimulatorImpl::BoundLookAhead (const Time lookAhead)
{
  if (lookAhead > Time (0))
    {
      NS_LOG_FUNCTION (this << lookAhead);
      m_lookAhead = Min(m_lookAhead, lookAhead);
    }
  else
    {
      NS_LOG_WARN ("attempted to set lookahead to a negative time: " << lookAhead);
    }
}

void
DistributedSimulatorImpl::SetScheduler (ObjectFactory schedulerFactory)
{
  NS_LOG_FUNCTION (this << schedulerFactory);

  Ptr<Scheduler> scheduler = schedulerFactory.Create<Scheduler> ();

  if (m_events != 0)
    {
      while (!m_events->IsEmpty ())
        {
          Scheduler::Event next = m_events->RemoveNext ();
          scheduler->Insert (next);
        }
    }
  m_events = scheduler;
}

void
DistributedSimulatorImpl::ProcessOneEvent (void)
{
  NS_LOG_FUNCTION (this);

  Scheduler::Event next = m_events->RemoveNext ();

  PreEventHook (EventId (next.impl, next.key.m_ts, 
                         next.key.m_context, next.key.m_uid));

  NS_ASSERT (next.key.m_ts >= m_currentTs);
  m_unscheduledEvents--;
  m_eventCount++;

  NS_LOG_LOGIC ("handle " << next.key.m_ts);
  m_currentTs = next.key.m_ts;
  m_currentContext = next.key.m_context;
  m_currentUid = next.key.m_uid;
  next.impl->Invoke ();
  next.impl->Unref ();
}

bool
DistributedSimulatorImpl::IsFinished (void) const
{
  return m_globalFinished;
}

bool
DistributedSimulatorImpl::IsLocalFinished (void) const
{
  return m_events->IsEmpty () || m_stop;
}

uint64_t
DistributedSimulatorImpl::NextTs (void) const
{
  // If local MPI task is has no more events or stop was called
  // next event time is infinity.
  if (IsLocalFinished ())
    {
      return GetMaximumSimulationTime ().GetTimeStep ();
    }
  else
    {
      Scheduler::Event ev = m_events->PeekNext ();
      return ev.key.m_ts;
    }
}

Time
DistributedSimulatorImpl::Next (void) const
{
  return TimeStep (NextTs ());
}

void
DistributedSimulatorImpl::Run (void)
{
  NS_LOG_FUNCTION (this);

  CalculateLookAhead ();
  m_stop = false;
  m_globalFinished = false;
  while (!m_globalFinished)
    {
      Time nextTime = Next ();

      // If local event is beyond grantedTime then need to synchronize
      // with other tasks to determine new time window. If local task
      // is finished then continue to participate in allgather
      // synchronizations with other tasks until all tasks have
      // completed.
      if (nextTime > m_grantedTime || IsLocalFinished () )
        {
          // Can't process next event, calculate a new LBTS
          // First receive any pending messages
          GrantedTimeWindowMpiInterface::ReceiveMessages ();
          // reset next time
          nextTime = Next ();
          // And check for send completes
          GrantedTimeWindowMpiInterface::TestSendComplete ();
          // Finally calculate the lbts
          LbtsMessage lMsg (GrantedTimeWindowMpiInterface::GetRxCount (), GrantedTimeWindowMpiInterface::GetTxCount (), 
                            m_myId, IsLocalFinished (), nextTime);
          m_pLBTS[m_myId] = lMsg;
          MPI_Allgather (&lMsg, sizeof (LbtsMessage), MPI_BYTE, m_pLBTS,
                         sizeof (LbtsMessage), MPI_BYTE, MpiInterface::GetCommunicator ());
          Time smallestTime = m_pLBTS[0].GetSmallestTime ();
          // The totRx and totTx counts insure there are no transient
          // messages;  If totRx != totTx, there are transients,
          // so we don't update the granted time.
          uint32_t totRx = m_pLBTS[0].GetRxCount ();
          uint32_t totTx = m_pLBTS[0].GetTxCount ();
          m_globalFinished = m_pLBTS[0].IsFinished ();

          for (uint32_t i = 1; i < m_systemCount; ++i)
            {
              if (m_pLBTS[i].GetSmallestTime () < smallestTime)
                {
                  smallestTime = m_pLBTS[i].GetSmallestTime ();
                }
              totRx += m_pLBTS[i].GetRxCount ();
              totTx += m_pLBTS[i].GetTxCount ();
              m_globalFinished &= m_pLBTS[i].IsFinished ();
            }

          // Global halting condition is all nodes have empty queue's and
          // no messages are in-flight.
          m_globalFinished &= totRx == totTx;
          
          if (totRx == totTx)
            {
              // If lookahead is infinite then granted time should be as well.
              // Covers the edge case if all the tasks have no inter tasks
              // links, prevents overflow of granted time.
              if (m_lookAhead == GetMaximumSimulationTime ())
                {
                  m_grantedTime = GetMaximumSimulationTime ();
                }
              else
                {
                  // Overflow is possible here if near end of representable time.
                  m_grantedTime = smallestTime + m_lookAhead;
                }
            }
        }

      // Execute next event if it is within the current time window.
      // Local task may be completed.
      if ( (nextTime <= m_grantedTime) && (!IsLocalFinished ()) )
        { // Safe to process
          ProcessOneEvent ();
        }
    }

  // If the simulator stopped naturally by lack of events, make a
  // consistency test to check that we didn't lose any events along the way.
  NS_ASSERT (!m_events->IsEmpty () || m_unscheduledEvents == 0);
}

uint32_t DistributedSimulatorImpl::GetSystemId () const
{
  return m_myId;
}

void
DistributedSimulatorImpl::Stop (void)
{
  NS_LOG_FUNCTION (this);

  m_stop = true;
}

void
DistributedSimulatorImpl::Stop (Time const &delay)
{
  NS_LOG_FUNCTION (this << delay.GetTimeStep ());

  Simulator::Schedule (delay, &Simulator::Stop);
}

//
// Schedule an event for a _relative_ time in the future.
//
EventId
DistributedSimulatorImpl::Schedule (Time const &delay, EventImpl *event)
{
  NS_LOG_FUNCTION (this << delay.GetTimeStep () << event);

  Time tAbsolute = delay + TimeStep (m_currentTs);

  NS_ASSERT (tAbsolute.IsPositive ());
  NS_ASSERT (tAbsolute >= TimeStep (m_currentTs));
  Scheduler::Event ev;
  ev.impl = event;
  ev.key.m_ts = static_cast<uint64_t> (tAbsolute.GetTimeStep ());
  ev.key.m_context = GetContext ();
  ev.key.m_uid = m_uid;
  m_uid++;
  m_unscheduledEvents++;
  m_events->Insert (ev);
  return EventId (event, ev.key.m_ts, ev.key.m_context, ev.key.m_uid);
}

void
DistributedSimulatorImpl::ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *event)
{
  NS_LOG_FUNCTION (this << context << delay.GetTimeStep () << m_currentTs << event);

  Scheduler::Event ev;
  ev.impl = event;
  ev.key.m_ts = m_currentTs + delay.GetTimeStep ();
  ev.key.m_context = context;
  ev.key.m_uid = m_uid;
  m_uid++;
  m_unscheduledEvents++;
  m_events->Insert (ev);
}

EventId
DistributedSimulatorImpl::ScheduleNow (EventImpl *event)
{
  NS_LOG_FUNCTION (this << event);
  return Schedule (Time (0), event);
}

EventId
DistributedSimulatorImpl::ScheduleDestroy (EventImpl *event)
{
  NS_LOG_FUNCTION (this << event);

  EventId id (Ptr<EventImpl> (event, false), m_currentTs, 0xffffffff, 2);
  m_destroyEvents.push_back (id);
  m_uid++;
  return id;
}

Time
DistributedSimulatorImpl::Now (void) const
{
  return TimeStep (m_currentTs);
}

Time
DistributedSimulatorImpl::GetDelayLeft (const EventId &id) const
{
  if (IsExpired (id))
    {
      return TimeStep (0);
    }
  else
    {
      return TimeStep (id.GetTs () - m_currentTs);
    }
}

void
DistributedSimulatorImpl::Remove (const EventId &id)
{
  if (id.GetUid () == EventId::UID::DESTROY)
    {
      // destroy events.
      for (DestroyEvents::iterator i = m_destroyEvents.begin (); i != m_destroyEvents.end (); i++)
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
  Scheduler::Event event;
  event.impl = id.PeekEventImpl ();
  event.key.m_ts = id.GetTs ();
  event.key.m_context = id.GetContext ();
  event.key.m_uid = id.GetUid ();
  m_events->Remove (event);
  event.impl->Cancel ();
  // whenever we remove an event from the event list, we have to unref it.
  event.impl->Unref ();

  m_unscheduledEvents--;
}

void
DistributedSimulatorImpl::Cancel (const EventId &id)
{
  if (!IsExpired (id))
    {
      id.PeekEventImpl ()->Cancel ();
    }
}

bool
DistributedSimulatorImpl::IsExpired (const EventId &id) const
{
  if (id.GetUid () == EventId::UID::DESTROY)
    {
      if (id.PeekEventImpl () == 0
          || id.PeekEventImpl ()->IsCancelled ())
        {
          return true;
        }
      // destroy events.
      for (DestroyEvents::const_iterator i = m_destroyEvents.begin (); i != m_destroyEvents.end (); i++)
        {
          if (*i == id)
            {
              return false;
            }
        }
      return true;
    }
  if (id.PeekEventImpl () == 0
      || id.GetTs () < m_currentTs
      || (id.GetTs () == m_currentTs
          && id.GetUid () <= m_currentUid)
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
DistributedSimulatorImpl::GetMaximumSimulationTime (void) const
{
  /// \todo I am fairly certain other compilers use other non-standard
  /// post-fixes to indicate 64 bit constants.
  return TimeStep (0x7fffffffffffffffLL);
}

uint32_t
DistributedSimulatorImpl::GetContext (void) const
{
  return m_currentContext;
}

uint64_t
DistributedSimulatorImpl::GetEventCount (void) const
{
  return m_eventCount;
}

} // namespace ns3
