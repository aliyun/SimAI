/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "logical-process.h"
#include "mtp-interface.h"

#include "ns3/channel.h"
#include "ns3/node-container.h"
#include "ns3/simulator.h"

#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LogicalProcess");

LogicalProcess::LogicalProcess ()
{
  m_systemId = 0;
  m_systemCount = 0;
  m_uid = EventId::UID::VALID;
  m_stop = false;
  m_currentContext = Simulator::NO_CONTEXT;
  m_currentUid = 0;
  m_currentTs = 0;
  m_eventCount = 0;
  m_pendingEventCount = 0;
  m_events = 0;
  m_lookAhead = TimeStep (0);
}

LogicalProcess::~LogicalProcess ()
{
  NS_LOG_INFO ("system " << m_systemId << " finished with event count " << m_eventCount);

  // if others hold references to event list, do not unref events
  if (m_events->GetReferenceCount () == 1)
    {
      while (!m_events->IsEmpty ())
        {
          Scheduler::Event next = m_events->RemoveNext ();
          next.impl->Unref ();
        }
    }
}

void
LogicalProcess::Enable (const uint32_t systemId, const uint32_t systemCount)
{
  m_systemId = systemId;
  m_systemCount = systemCount;
}

void
LogicalProcess::CalculateLookAhead ()
{
  NS_LOG_FUNCTION (this);

  if (m_systemId == 0)
    {
      m_lookAhead = TimeStep (0); // No lookahead for public LP
    }
  else
    {
      m_lookAhead = Time::Max () / 2 - TimeStep (1);
      NodeContainer c = NodeContainer::GetGlobal ();
      for (NodeContainer::Iterator iter = c.Begin (); iter != c.End (); ++iter)
        {
#ifdef NS3_MPI
          if (((*iter)->GetSystemId () >> 16) != m_systemId)
            {
              continue;
            }
#else
          if ((*iter)->GetSystemId () != m_systemId)
            {
              continue;
            }
#endif
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
              if (remoteNode->GetSystemId () == m_systemId)
                {
                  continue;
                }
              // compare delay on the channel with current value of m_lookAhead.
              // if delay on channel is smaller, make it the new lookAhead.
              TimeValue delay;
              channel->GetAttribute ("Delay", delay);
              if (delay.Get () < m_lookAhead)
                {
                  m_lookAhead = delay.Get ();
                }
              // add the neighbour to the mailbox
              m_mailbox[remoteNode->GetSystemId ()];
            }
        }
    }

  NS_LOG_INFO ("lookahead of system " << m_systemId << " is set to " << m_lookAhead.GetTimeStep ());
}

void
LogicalProcess::ReceiveMessages ()
{
  NS_LOG_FUNCTION (this);

  m_pendingEventCount = 0;
  for (auto &item : m_mailbox)
    {
      auto &queue = item.second;
      std::sort (queue.begin (), queue.end (), std::greater<> ());
      while (!queue.empty ())
        {
          auto &evWithTs = queue.back ();
          Scheduler::Event &ev = std::get<3> (evWithTs);
          ev.key.m_uid = m_uid++;
          m_events->Insert (ev);
          queue.pop_back ();
          m_pendingEventCount++;
        }
    }
}

void
LogicalProcess::ProcessOneRound ()
{
  NS_LOG_FUNCTION (this);

  // set thread context
  MtpInterface::SetSystem (m_systemId);

  // calculate time window
  Time grantedTime =
      Min (MtpInterface::GetSmallestTime () + m_lookAhead, MtpInterface::GetNextPublicTime ());

  auto start = std::chrono::system_clock::now ();

  // process events
  while (Next () <= grantedTime)
    {
      Scheduler::Event next = m_events->RemoveNext ();
      m_eventCount++;
      NS_LOG_LOGIC ("handle " << next.key.m_ts);

      auto oldT = m_currentTs;

      m_currentTs = next.key.m_ts;
      m_currentContext = next.key.m_context;
      m_currentUid = next.key.m_uid;
      
      next.impl->Invoke ();
      next.impl->Unref ();
    }

  auto end = std::chrono::system_clock::now ();
  m_executionTime = std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count ();
}

EventId
LogicalProcess::Schedule (Time const &delay, EventImpl *event)
{
  Scheduler::Event ev;

  ev.impl = event;
  ev.key.m_ts = m_currentTs + delay.GetTimeStep ();
  ev.key.m_context = GetContext ();
  ev.key.m_uid = m_uid++;
  m_events->Insert (ev);

  return EventId (event, ev.key.m_ts, ev.key.m_context, ev.key.m_uid);
}

void
LogicalProcess::ScheduleAt (const uint32_t context, Time const &time, EventImpl *event)
{
  Scheduler::Event ev;

  ev.impl = event;
  ev.key.m_ts = time.GetTimeStep ();
  ev.key.m_context = context;
  ev.key.m_uid = m_uid++;
  m_events->Insert (ev);
}

void
LogicalProcess::ScheduleWithContext (LogicalProcess *remote, const uint32_t context,
                                     Time const &delay, EventImpl *event)
{
  Scheduler::Event ev;

  ev.impl = event;
  ev.key.m_ts = delay.GetTimeStep () + m_currentTs;
  ev.key.m_context = context;

  if (remote == this)
    {
      ev.key.m_uid = m_uid++;
      m_events->Insert (ev);
    }
  else
    {
      ev.key.m_uid = EventId::UID::INVALID;
      remote->m_mailbox[m_systemId].push_back (
          std::make_tuple (m_currentTs, m_systemId, m_uid, ev));
    }
}

void
LogicalProcess::InvokeNow (Scheduler::Event const &ev)
{
  uint32_t oldSystemId = MtpInterface::GetSystem ()->GetSystemId ();
  MtpInterface::SetSystem (m_systemId);

  m_eventCount++;
  NS_LOG_LOGIC ("handle " << ev.key.m_ts);

  m_currentTs = ev.key.m_ts;
  m_currentContext = ev.key.m_context;
  m_currentUid = ev.key.m_uid;

  ev.impl->Invoke ();
  ev.impl->Unref ();

  // restore previous thread context
  MtpInterface::SetSystem (oldSystemId);
}

void
LogicalProcess::Remove (const EventId &id)
{
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
}

bool
LogicalProcess::IsExpired (const EventId &id) const
{
  if (id.PeekEventImpl () == 0 || id.GetTs () < m_currentTs ||
      (id.GetTs () == m_currentTs && id.GetUid () <= m_currentUid) ||
      id.PeekEventImpl ()->IsCancelled ())
    {
      return true;
    }
  else
    {
      return false;
    }
}

void
LogicalProcess::SetScheduler (ObjectFactory schedulerFactory)
{
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

Time
LogicalProcess::Next () const
{
  if (m_stop || m_events->IsEmpty ())
    {
      return Time::Max ();
    }
  else
    {
      Scheduler::Event ev = m_events->PeekNext ();
      return TimeStep (ev.key.m_ts);
    }
}

} // namespace ns3
