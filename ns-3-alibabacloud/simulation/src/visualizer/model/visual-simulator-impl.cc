/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Gustavo Carneiro
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
 * Author: Gustavo Carneiro <gjcarneiro@gmail.com> <gjc@inescporto.pt>
 */
#include <Python.h>
#undef HAVE_PTHREAD_H
#undef HAVE_SYS_STAT_H
#include "visual-simulator-impl.h"
#include "ns3/default-simulator-impl.h"
#include "ns3/log.h"
#include "ns3/packet-metadata.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("VisualSimulatorImpl");

NS_OBJECT_ENSURE_REGISTERED (VisualSimulatorImpl);

namespace
{
/**
 * Get an object factory configured to the default simulator implementation
 * \return an object factory.
 */
ObjectFactory
GetDefaultSimulatorImplFactory ()
{
  ObjectFactory factory;
  factory.SetTypeId (DefaultSimulatorImpl::GetTypeId ());
  return factory;
}
}


TypeId
VisualSimulatorImpl::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::VisualSimulatorImpl")
    .SetParent<SimulatorImpl> ()
    .SetGroupName ("Visualizer")
    .AddConstructor<VisualSimulatorImpl> ()
    .AddAttribute ("SimulatorImplFactory",
                   "Factory for the underlying simulator implementation used by the visualizer.",
                   ObjectFactoryValue (GetDefaultSimulatorImplFactory ()),
                   MakeObjectFactoryAccessor (&VisualSimulatorImpl::m_simulatorImplFactory),
                   MakeObjectFactoryChecker ())
  ;
  return tid;
}


VisualSimulatorImpl::VisualSimulatorImpl ()
{
  PacketMetadata::Enable ();
}

VisualSimulatorImpl::~VisualSimulatorImpl ()
{
}

void
VisualSimulatorImpl::DoDispose (void)
{
  if (m_simulator)
    {
      m_simulator->Dispose ();
      m_simulator = NULL;
    }
  SimulatorImpl::DoDispose ();
}

void
VisualSimulatorImpl::NotifyConstructionCompleted ()
{
  m_simulator = m_simulatorImplFactory.Create<SimulatorImpl> ();
}


void
VisualSimulatorImpl::Destroy ()
{
  m_simulator->Destroy ();
}

void
VisualSimulatorImpl::SetScheduler (ObjectFactory schedulerFactory)
{
  m_simulator->SetScheduler (schedulerFactory);
}

// System ID for non-distributed simulation is always zero
uint32_t 
VisualSimulatorImpl::GetSystemId (void) const
{
  return m_simulator->GetSystemId ();
}

bool 
VisualSimulatorImpl::IsFinished (void) const
{
  return m_simulator->IsFinished ();
}

void
VisualSimulatorImpl::Run (void)
{
  if (!Py_IsInitialized ())
    {
      #if PY_MAJOR_VERSION >= 3
        const wchar_t *argv[] = { L"python", NULL};
        Py_Initialize ();
        PySys_SetArgv (1, (wchar_t**) argv);
      #else
        const char *argv[] = { "python", NULL};
        Py_Initialize ();
        PySys_SetArgv (1, (char**) argv);
      #endif
      PyRun_SimpleString (
                          "import visualizer\n"
                          "visualizer.start();\n"
                          );
    }
  else
    {
      PyGILState_STATE __py_gil_state = PyGILState_Ensure ();
    
      PyRun_SimpleString (
                          "import visualizer\n"
                          "visualizer.start();\n"
                          );

      PyGILState_Release (__py_gil_state);
    }
}

void 
VisualSimulatorImpl::Stop (void)
{
  m_simulator->Stop ();
}

void 
VisualSimulatorImpl::Stop (Time const &delay)
{
  m_simulator->Stop (delay);
}

//
// Schedule an event for a _relative_ time in the future.
//
EventId
VisualSimulatorImpl::Schedule (Time const &delay, EventImpl *event)
{
  return m_simulator->Schedule (delay, event);
}

void
VisualSimulatorImpl::ScheduleWithContext (uint32_t context, Time const &delay, EventImpl *event)
{
  m_simulator->ScheduleWithContext (context, delay, event);
}

EventId
VisualSimulatorImpl::ScheduleNow (EventImpl *event)
{
  return m_simulator->ScheduleNow (event);
}

EventId
VisualSimulatorImpl::ScheduleDestroy (EventImpl *event)
{
  return m_simulator->ScheduleDestroy (event);
}

Time
VisualSimulatorImpl::Now (void) const
{
  return m_simulator->Now ();
}

Time 
VisualSimulatorImpl::GetDelayLeft (const EventId &id) const
{
  return m_simulator->GetDelayLeft (id);
}

void
VisualSimulatorImpl::Remove (const EventId &id)
{
  m_simulator->Remove (id);
}

void
VisualSimulatorImpl::Cancel (const EventId &id)
{
  m_simulator->Cancel (id);
}

bool
VisualSimulatorImpl::IsExpired (const EventId &id) const
{
  return m_simulator->IsExpired (id);
}

Time 
VisualSimulatorImpl::GetMaximumSimulationTime (void) const
{
  return m_simulator->GetMaximumSimulationTime ();
}

uint32_t
VisualSimulatorImpl::GetContext (void) const
{
  return m_simulator->GetContext ();
}

uint64_t
VisualSimulatorImpl::GetEventCount (void) const
{
  return m_simulator->GetEventCount ();
}

void
VisualSimulatorImpl::RunRealSimulator (void)
{
  m_simulator->Run ();
}


} // namespace ns3


