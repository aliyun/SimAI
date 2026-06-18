/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef LOGICAL_PROCESS_H
#define LOGICAL_PROCESS_H

#include "ns3/event-id.h"
#include "ns3/event-impl.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/ptr.h"
#include "ns3/scheduler.h"

#include <atomic>
#include <chrono>
#include <map>
#include <tuple>
#include <vector>

namespace ns3 {

class LogicalProcess
{
public:
  LogicalProcess ();
  ~LogicalProcess ();

  void Enable (const uint32_t systemId, const uint32_t systemCount);
  void CalculateLookAhead ();
  void ReceiveMessages ();
  void ProcessOneRound ();

  inline uint64_t
  GetExecutionTime () const
  {
    return m_executionTime;
  }

  inline uint64_t
  GetPendingEventCount () const
  {
    return m_pendingEventCount;
  }

  inline Ptr<Scheduler>
  GetPendingEvents () const
  {
    return m_events;
  }

  // mapped from MultithreadedSimulatorImpl
  EventId Schedule (Time const &delay, EventImpl *event);
  void ScheduleAt (const uint32_t context, Time const &time, EventImpl *event);
  void ScheduleWithContext (LogicalProcess *remote, const uint32_t context, Time const &delay,
                            EventImpl *event);
  void InvokeNow (Scheduler::Event const &ev); // cross context immediate invocation
  void Remove (const EventId &id);
  void Cancel (const EventId &id);
  bool IsExpired (const EventId &id) const;
  void SetScheduler (ObjectFactory schedulerFactory);
  Time Next () const;

  inline bool
  isLocalFinished () const
  {
    return m_stop || m_events->IsEmpty ();
  }

  inline void
  Stop ()
  {
    m_stop = true;
  }

  inline Time
  Now () const
  {
    return TimeStep (m_currentTs);
  }

  inline Time
  GetDelayLeft (const EventId &id) const
  {
    return TimeStep (id.GetTs () - m_currentTs);
  }

  inline uint32_t
  GetSystemId (void) const
  {
    return m_systemId;
  }

  inline uint32_t
  GetContext () const
  {
    return m_currentContext;
  }

  inline uint64_t
  GetEventCount () const
  {
    return m_eventCount;
  }

private:
  uint32_t m_systemId;
  uint32_t m_systemCount;
  bool m_stop;
  uint32_t m_uid;
  uint32_t m_currentContext;
  uint32_t m_currentUid;
  uint64_t m_currentTs;
  uint64_t m_eventCount;
  uint64_t m_pendingEventCount;
  Ptr<Scheduler> m_events;
  Time m_lookAhead;

  std::map<uint32_t, std::vector<std::tuple<uint64_t, uint32_t, uint32_t, Scheduler::Event>>>
      m_mailbox; // event message mail box
  std::chrono::nanoseconds::rep m_executionTime;
};

} // namespace ns3

#endif /* LOGICAL_PROCESS_H */
