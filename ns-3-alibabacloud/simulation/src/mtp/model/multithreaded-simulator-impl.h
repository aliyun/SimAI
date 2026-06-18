/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef MULTITHREADED_SIMULATOR_IMPL_H
#define MULTITHREADED_SIMULATOR_IMPL_H

#include "ns3/event-id.h"
#include "ns3/event-impl.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/simulator-impl.h"

#include <list>

namespace ns3 {

class MultithreadedSimulatorImpl : public SimulatorImpl
{
public:
  static TypeId GetTypeId ();

  /** Default constructor. */
  MultithreadedSimulatorImpl ();
  /** Destructor. */
  ~MultithreadedSimulatorImpl ();

  // virtual from SimulatorImpl
  virtual void Destroy ();
  virtual bool IsFinished () const;
  virtual void Stop ();
  virtual void Stop (Time const &delay);
  virtual EventId Schedule (Time const &delay, EventImpl *event);
  virtual void ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *event);
  virtual EventId ScheduleNow (EventImpl *event);
  virtual EventId ScheduleDestroy (EventImpl *event);
  virtual void Remove (const EventId &id);
  virtual void Cancel (const EventId &id);
  virtual bool IsExpired (const EventId &id) const;
  virtual void Run ();
  virtual Time Now () const;
  virtual Time GetDelayLeft (const EventId &id) const;
  virtual Time GetMaximumSimulationTime () const;
  virtual void SetScheduler (ObjectFactory schedulerFactory);
  virtual uint32_t GetSystemId () const;
  virtual uint32_t GetContext () const;
  virtual uint64_t GetEventCount () const;

private:
  // Inherited from Object
  virtual void DoDispose ();

  void Partition ();

  bool m_partition;
  uint32_t m_maxThreads;
  Time m_minLookahead;
  TypeId m_schedulerTypeId;
  std::list<EventId> m_destroyEvents;
};

} // namespace ns3

#endif /* MULTITHREADED_SIMULATOR_IMPL_H */
