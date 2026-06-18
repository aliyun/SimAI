/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#ifndef NS3_HYBRID_SIMULATOR_IMPL_H
#define NS3_HYBRID_SIMULATOR_IMPL_H

#include "distributed-simulator-impl.h"

#include "ns3/event-id.h"
#include "ns3/event-impl.h"
#include "ns3/nstime.h"
#include "ns3/object-factory.h"
#include "ns3/simulator-impl.h"

#include <list>

namespace ns3 {

class HybridSimulatorImpl : public SimulatorImpl
{
public:
  static TypeId GetTypeId (void);

  /** Default constructor. */
  HybridSimulatorImpl ();
  /** Destructor. */
  ~HybridSimulatorImpl ();

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

private:
  // Inherited from Object
  virtual void DoDispose (void);

  bool IsLocalFinished (void) const;

  /** Are all parallel instances completed. */
  bool m_globalFinished;

  LbtsMessage *m_pLBTS;
  uint32_t m_myId; /**< MPI rank. */
  uint32_t m_systemCount; /**< MPI communicator size. */
  Time m_smallestTime; /**< End of current window. */

  void Partition ();

  uint32_t m_maxThreads;
  Time m_minLookahead;
  TypeId m_schedulerTypeId;
  std::list<EventId> m_destroyEvents;
};

} // namespace ns3

#endif /* NS3_HYBRID_SIMULATOR_IMPL_H */
