/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

#include "hybrid-simulator-impl.h"
#include "granted-time-window-mpi-interface.h"
#include "mpi-interface.h"

#include "ns3/channel.h"
#include "ns3/simulator.h"
#include "ns3/node.h"
#include "ns3/node-container.h"
#include "ns3/node-list.h"
#include "ns3/uinteger.h"
#include "ns3/mtp-interface.h"

#include <mpi.h>
#include <algorithm>
#include <queue>
#include <thread>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HybridSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED (HybridSimulatorImpl);

HybridSimulatorImpl::HybridSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);

  MtpInterface::Enable (1, 0);
  m_myId = MpiInterface::GetSystemId ();
  m_systemCount = MpiInterface::GetSize ();

  // Allocate the LBTS message buffer
  m_pLBTS = new LbtsMessage[m_systemCount];
  m_smallestTime = Seconds (0);
  m_globalFinished = false;
}

HybridSimulatorImpl::~HybridSimulatorImpl ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
HybridSimulatorImpl::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("ns3::HybridSimulatorImpl")
          .SetParent<SimulatorImpl> ()
          .SetGroupName ("Mtp")
          .AddConstructor<HybridSimulatorImpl> ()
          .AddAttribute ("MaxThreads", "The maximum threads used in simulation",
                         UintegerValue (std::thread::hardware_concurrency ()),
                         MakeUintegerAccessor (&HybridSimulatorImpl::m_maxThreads),
                         MakeUintegerChecker<uint32_t> (1))
          .AddAttribute ("MinLookahead", "The minimum lookahead in a partition",
                         TimeValue (TimeStep (1)),
                         MakeTimeAccessor (&HybridSimulatorImpl::m_minLookahead),
                         MakeTimeChecker (TimeStep (0)));
  return tid;
}

void
HybridSimulatorImpl::Destroy ()
{
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

  MtpInterface::Disable ();
  MpiInterface::Destroy ();
}

bool
HybridSimulatorImpl::IsFinished (void) const
{
  return m_globalFinished;
}

bool
HybridSimulatorImpl::IsLocalFinished (void) const
{
  return MtpInterface::isFinished ();
}

void
HybridSimulatorImpl::Stop (void)
{
  NS_LOG_FUNCTION (this);
  for (uint32_t i = 0; i < MtpInterface::GetSize (); i++)
    {
      MtpInterface::GetSystem (i)->Stop ();
    }
}

void
HybridSimulatorImpl::Stop (Time const &delay)
{
  NS_LOG_FUNCTION (this << delay.GetTimeStep ());
  Simulator::Schedule (delay, &Simulator::Stop);
}

EventId
HybridSimulatorImpl::Schedule (Time const &delay, EventImpl *event)
{
  NS_LOG_FUNCTION (this << delay.GetTimeStep () << event);
  return MtpInterface::GetSystem ()->Schedule (delay, event);
}

void
HybridSimulatorImpl::ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *event)
{
  NS_LOG_FUNCTION (this << context << delay.GetTimeStep () << event);

  if (MtpInterface::GetSize () == 1)
    {
      // initialization stage, do not schedule remote
      LogicalProcess *local = MtpInterface::GetSystem ();
      local->ScheduleWithContext (local, context, delay, event);
    }
  else
    {
      LogicalProcess *remote =
          MtpInterface::GetSystem (NodeList::GetNode (context)->GetSystemId () >> 16);
      MtpInterface::GetSystem ()->ScheduleWithContext (remote, context, delay, event);
    }
}

EventId
HybridSimulatorImpl::ScheduleNow (EventImpl *event)
{
  return Schedule (TimeStep (0), event);
}

EventId
HybridSimulatorImpl::ScheduleDestroy (EventImpl *event)
{
  EventId id (Ptr<EventImpl> (event, false), GetMaximumSimulationTime ().GetTimeStep (), 0xffffffff,
              EventId::DESTROY);
  MtpInterface::CriticalSection cs;
  m_destroyEvents.push_back (id);
  return id;
}

void
HybridSimulatorImpl::Remove (const EventId &id)
{
  if (id.GetUid () == EventId::DESTROY)
    {
      // destroy events.
      for (std::list<EventId>::iterator i = m_destroyEvents.begin (); i != m_destroyEvents.end ();
           i++)
        {
          if (*i == id)
            {
              m_destroyEvents.erase (i);
              break;
            }
        }
    }
  else
    {
      MtpInterface::GetSystem ()->Remove (id);
    }
}

void
HybridSimulatorImpl::Cancel (const EventId &id)
{
  if (!IsExpired (id))
    {
      id.PeekEventImpl ()->Cancel ();
    }
}

bool
HybridSimulatorImpl::IsExpired (const EventId &id) const
{
  if (id.GetUid () == EventId::DESTROY)
    {
      // destroy events.
      if (id.PeekEventImpl () == 0 || id.PeekEventImpl ()->IsCancelled ())
        {
          return true;
        }
      for (std::list<EventId>::const_iterator i = m_destroyEvents.begin ();
           i != m_destroyEvents.end (); i++)
        {
          if (*i == id)
            {
              return false;
            }
        }
      return true;
    }
  else
    {
      return MtpInterface::GetSystem ()->IsExpired (id);
    }
}

void
HybridSimulatorImpl::Run (void)
{
  NS_LOG_FUNCTION (this);

  Partition ();
  MtpInterface::RunBefore ();

  m_globalFinished = false;
  while (!m_globalFinished)
    {
      GrantedTimeWindowMpiInterface::ReceiveMessages ();
      GrantedTimeWindowMpiInterface::TestSendComplete ();
      MtpInterface::CalculateSmallestTime ();
      LbtsMessage lMsg (GrantedTimeWindowMpiInterface::GetRxCount (),
                        GrantedTimeWindowMpiInterface::GetTxCount (), m_myId, IsLocalFinished (),
                        MtpInterface::GetSmallestTime ());
      m_pLBTS[m_myId] = lMsg;
      MPI_Allgather (&lMsg, sizeof (LbtsMessage), MPI_BYTE, m_pLBTS, sizeof (LbtsMessage), MPI_BYTE,
                     MpiInterface::GetCommunicator ());
      m_smallestTime = m_pLBTS[0].GetSmallestTime ();

      // The totRx and totTx counts insure there are no transient
      // messages;  If totRx != totTx, there are transients,
      // so we don't update the granted time.
      uint32_t totRx = m_pLBTS[0].GetRxCount ();
      uint32_t totTx = m_pLBTS[0].GetTxCount ();
      m_globalFinished = m_pLBTS[0].IsFinished ();

      // calculate smallest time of all hosts
      for (uint32_t i = 1; i < m_systemCount; ++i)
        {
          if (m_pLBTS[i].GetSmallestTime () < m_smallestTime)
            {
              m_smallestTime = m_pLBTS[i].GetSmallestTime ();
            }
          totRx += m_pLBTS[i].GetRxCount ();
          totTx += m_pLBTS[i].GetTxCount ();
          m_globalFinished &= m_pLBTS[i].IsFinished ();
        }
      MtpInterface::SetSmallestTime (m_smallestTime);

      // Global halting condition is all nodes have empty queue's and
      // no messages are in-flight.
      m_globalFinished &= totRx == totTx;

      // Execute next event if it is within the current time window.
      // Local task may be completed.
      if (totRx == totTx && !IsLocalFinished ())
        { // Safe to process
          MtpInterface::ProcessOneRound ();
        }
    }

  MtpInterface::RunAfter ();
}

Time
HybridSimulatorImpl::Now (void) const
{
  // Do not add function logging here, to avoid stack overflow
  return MtpInterface::GetSystem ()->Now ();
}

Time
HybridSimulatorImpl::GetDelayLeft (const EventId &id) const
{
  if (IsExpired (id))
    {
      return TimeStep (0);
    }
  else
    {
      return MtpInterface::GetSystem ()->GetDelayLeft (id);
    }
}

Time
HybridSimulatorImpl::GetMaximumSimulationTime (void) const
{
  return Time::Max () / 2;
}

void
HybridSimulatorImpl::SetScheduler (ObjectFactory schedulerFactory)
{
  NS_LOG_FUNCTION (this << schedulerFactory);
  for (uint32_t i = 0; i < MtpInterface::GetSize (); i++)
    {
      MtpInterface::GetSystem (i)->SetScheduler (schedulerFactory);
    }
  m_schedulerTypeId = schedulerFactory.GetTypeId ();
}

uint32_t
HybridSimulatorImpl::GetSystemId () const
{
  return m_myId;
}

uint32_t
HybridSimulatorImpl::GetContext (void) const
{
  return MtpInterface::GetSystem ()->GetContext ();
}

uint64_t
HybridSimulatorImpl::GetEventCount (void) const
{
  uint64_t eventCount = 0;
  for (uint32_t i = 0; i < MtpInterface::GetSize (); i++)
    {
      eventCount += MtpInterface::GetSystem (i)->GetEventCount ();
    }
  return eventCount;
}

void
HybridSimulatorImpl::DoDispose (void)
{
  delete[] m_pLBTS;
  SimulatorImpl::DoDispose ();
}

void
HybridSimulatorImpl::Partition ()
{
  NS_LOG_FUNCTION (this);
  uint32_t localSystemId = 0;
  NodeContainer nodes = NodeContainer::GetGlobal ();
  bool *visited = new bool[nodes.GetN ()]{false};
  std::queue<Ptr<Node>> q;

  // perform a BFS on the whole network topo to assign each node a localSystemId
  for (NodeContainer::Iterator it = nodes.Begin (); it != nodes.End (); it++)
    {
      Ptr<Node> node = *it;
      if (!visited[node->GetId ()] && node->GetSystemId () == m_myId)
        {
          q.push (node);
          localSystemId++;
          while (!q.empty ())
            {
              // pop from BFS queue
              node = q.front ();
              q.pop ();
              visited[node->GetId ()] = true;
              // assign this node the current localSystemId
              node->SetSystemId (localSystemId << 16 | m_myId);
              NS_LOG_INFO ("node " << node->GetId () << " is set to local system "
                                   << localSystemId);

              for (uint32_t i = 0; i < node->GetNDevices (); i++)
                {
                  Ptr<NetDevice> localNetDevice = node->GetDevice (i);
                  Ptr<Channel> channel = localNetDevice->GetChannel ();
                  if (channel == 0)
                    {
                      continue;
                    }
                  // cut-off p2p links for partition
                  if (localNetDevice->IsPointToPoint ())
                    {
                      TimeValue delay;
                      channel->GetAttribute ("Delay", delay);
                      // if delay is below threshold, do not cut-off
                      if (delay.Get () >= m_minLookahead)
                        {
                          continue;
                        }
                    }
                  // grab the adjacent nodes
                  for (uint32_t j = 0; j < channel->GetNDevices (); j++)
                    {
                      Ptr<Node> remote = channel->GetDevice (j)->GetNode ();
                      // if it's not visited, and not remote, add it to the current partition
                      if (!visited[remote->GetId ()] && node->GetSystemId () == m_myId)
                        {
                          q.push (remote);
                        }
                    }
                }
            }
        }
    }
  delete[] visited;

  // after the partition, we finally know the system count
  uint32_t systemCount = localSystemId;
  uint32_t threadCount = std::min (m_maxThreads, systemCount);
  NS_LOG_INFO ("Partition done! " << systemCount << " systems share " << threadCount << " threads");

  // create new systems
  Ptr<Scheduler> events = MtpInterface::GetSystem ()->GetPendingEvents ();
  MtpInterface::Disable ();
  MtpInterface::Enable (threadCount, systemCount);

  // set scheduler
  ObjectFactory schedulerFactory;
  schedulerFactory.SetTypeId (m_schedulerTypeId);
  for (uint32_t i = 0; i <= systemCount; i++)
    {
      MtpInterface::GetSystem (i)->SetScheduler (schedulerFactory);
    }

  // transfer events to new system
  while (!events->IsEmpty ())
    {
      Scheduler::Event ev = events->RemoveNext ();
      // invoke initialization events (at time 0) by their insertion order
      // since they may not be in the same system, causing error
      if (ev.key.m_ts == 0)
        {
          MtpInterface::GetSystem (ev.key.m_context == Simulator::NO_CONTEXT
                                       ? 0
                                       : NodeList::GetNode (ev.key.m_context)->GetSystemId () >> 16)
              ->InvokeNow (ev);
        }
      else if (ev.key.m_context == Simulator::NO_CONTEXT)
        {
          Schedule (TimeStep (ev.key.m_ts), ev.impl);
        }
      else
        {
          ScheduleWithContext (ev.key.m_context, TimeStep (ev.key.m_ts), ev.impl);
        }
    }
}

} // namespace ns3
