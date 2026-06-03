/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 INRIA
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
 * Author: Claudio Freire <claudio-daniel.freire@inria.fr>
 */
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/list-scheduler.h"
#include "ns3/heap-scheduler.h"
#include "ns3/map-scheduler.h"
#include "ns3/calendar-scheduler.h"
#include "ns3/config.h"
#include "ns3/string.h"

#include <chrono>  // seconds, milliseconds
#include <ctime>
#include <list>
#include <thread>  // sleep_for
#include <utility>

using namespace ns3;

/// Maximum number of threads.
constexpr int MAXTHREADS = 64;

/**
 * \file
 * \ingroup threaded-tests
 * Threaded events test suite
 */

/**
 * \ingroup core-tests
 * \defgroup threaded-tests Threaded events tests
 */

/**
 * \ingroup threaded-tests
 *  
 * \brief Check threaded event handling with various thread number, schedulers, and  simulator types.
 */
class ThreadedSimulatorEventsTestCase : public TestCase
{
public:
  /**
   * Constructor.
   * 
   * \param schedulerFactory The scheduler factory.
   * \param simulatorType The simulator type.
   * \param threads The number of threads.
   */
  ThreadedSimulatorEventsTestCase (ObjectFactory schedulerFactory, const std::string &simulatorType, unsigned int threads);
  /**
   * Event A
   * \param a The Event parameter.
   */
  void EventA (int a);
  /**
   * Event B
   * \param b The Event parameter.
   */
  void EventB (int b);
  /**
   * Event C
   * \param c The Event parameter.
   */
  void EventC (int c);
  /**
   * Event D
   * \param d The Event parameter.
   */
  void EventD (int d);
  /**
   * No-op function, records the thread that called it. 
   * \param threadno The thread number.
   */
  void DoNothing (unsigned int threadno);
  /**
   * Schedule a thread.
   * \param context The context.
   */
  static void SchedulingThread (std::pair<ThreadedSimulatorEventsTestCase *, unsigned int> context);
  /**
   * End the thread execution.
   */
  void End (void);
  uint64_t m_a; //!< The value incremented when EventA is called.
  uint64_t m_b; //!< The value incremented when EventB is called.
  uint64_t m_c; //!< The value incremented when EventC is called.
  uint64_t m_d; //!< The value incremented when EventD is called.
  unsigned int m_threads; //!< The number of threads.
  bool m_threadWaiting[MAXTHREADS]; //!< Threads waiting to be scheduled.
  bool m_stop;  //!< Stop variable.
  ObjectFactory m_schedulerFactory; //!< Scheduler factory.
  std::string m_simulatorType;      //!< Simulator type.
  std::string m_error;              //!< Error condition.
  std::list<std::thread> m_threadlist; //!< Thread list.

private:
  virtual void DoSetup (void);
  virtual void DoRun (void);
  virtual void DoTeardown (void);
};

ThreadedSimulatorEventsTestCase::ThreadedSimulatorEventsTestCase (ObjectFactory schedulerFactory, const std::string &simulatorType, unsigned int threads)
  : TestCase ("Check threaded event handling with " +
              std::to_string (threads) + " threads, " +
              schedulerFactory.GetTypeId ().GetName () + " scheduler, in " +
              simulatorType),
    m_threads (threads),
    m_schedulerFactory (schedulerFactory),
    m_simulatorType (simulatorType)
{}

void
ThreadedSimulatorEventsTestCase::End (void)
{
  m_stop = true;
  for (auto& thread : m_threadlist)
    {
      if (thread.joinable ())
        {
          thread.join ();
        }
    }
}
void
ThreadedSimulatorEventsTestCase::SchedulingThread (std::pair<ThreadedSimulatorEventsTestCase *, unsigned int> context)
{
  ThreadedSimulatorEventsTestCase *me = context.first;
  unsigned int threadno = context.second;

  while (!me->m_stop)
    {
      me->m_threadWaiting[threadno] = true;
      Simulator::ScheduleWithContext (threadno,
                                      MicroSeconds (1),
                                      &ThreadedSimulatorEventsTestCase::DoNothing, me, threadno);
      while (!me->m_stop && me->m_threadWaiting[threadno])
        {
          std::this_thread::sleep_for (std::chrono::nanoseconds (500));
        }
    }
}
void
ThreadedSimulatorEventsTestCase::DoNothing (unsigned int threadno)
{
  if (!m_error.empty ())
    {
      m_error = "Bad threaded scheduling";
    }
  m_threadWaiting[threadno] = false;
}
void
ThreadedSimulatorEventsTestCase::EventA (int a)
{
  if (m_a != m_b || m_a != m_c || m_a != m_d)
    {
      m_error = "Bad scheduling";
      Simulator::Stop ();
    }
  ++m_a;
  Simulator::Schedule (MicroSeconds (10),
                       &ThreadedSimulatorEventsTestCase::EventB, this, a + 1);
}

void
ThreadedSimulatorEventsTestCase::EventB (int b)
{
  if (m_a != (m_b + 1) || m_a != (m_c + 1) || m_a != (m_d + 1))
    {
      m_error = "Bad scheduling";
      Simulator::Stop ();
    }
  ++m_b;
  Simulator::Schedule (MicroSeconds (10),
                       &ThreadedSimulatorEventsTestCase::EventC, this, b + 1);
}

void
ThreadedSimulatorEventsTestCase::EventC (int c)
{
  if (m_a != m_b || m_a != (m_c + 1) || m_a != (m_d + 1))
    {
      m_error = "Bad scheduling";
      Simulator::Stop ();
    }
  ++m_c;
  Simulator::Schedule (MicroSeconds (10),
                       &ThreadedSimulatorEventsTestCase::EventD, this, c + 1);
}

void
ThreadedSimulatorEventsTestCase::EventD (int d)
{
  if (m_a != m_b || m_a != m_c || m_a != (m_d + 1))
    {
      m_error = "Bad scheduling";
      Simulator::Stop ();
    }
  ++m_d;
  if (m_stop)
    {
      Simulator::Stop ();
    }
  else
    {
      Simulator::Schedule (MicroSeconds (10),
                           &ThreadedSimulatorEventsTestCase::EventA, this, d + 1);
    }
}

void
ThreadedSimulatorEventsTestCase::DoSetup (void)
{
  if (!m_simulatorType.empty ())
    {
      Config::SetGlobal ("SimulatorImplementationType", StringValue (m_simulatorType));
    }

  m_error = "";

  m_a =
    m_b =
      m_c =
        m_d = 0;
}
void
ThreadedSimulatorEventsTestCase::DoTeardown (void)
{
  m_threadlist.clear ();

  Config::SetGlobal ("SimulatorImplementationType", StringValue ("ns3::DefaultSimulatorImpl"));
}
void
ThreadedSimulatorEventsTestCase::DoRun (void)
{
  m_stop = false;
  Simulator::SetScheduler (m_schedulerFactory);

  Simulator::Schedule (MicroSeconds (10), &ThreadedSimulatorEventsTestCase::EventA, this, 1);
  Simulator::Schedule (Seconds (1), &ThreadedSimulatorEventsTestCase::End, this);

  for (unsigned int i = 0; i < m_threads; ++i)
    {
      m_threadlist.push_back (
        std::thread (&ThreadedSimulatorEventsTestCase::SchedulingThread,
                     std::pair<ThreadedSimulatorEventsTestCase *, unsigned int> (this,i) ));
    }

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_EXPECT_MSG_EQ (m_error.empty (), true, m_error.c_str ());
  NS_TEST_EXPECT_MSG_EQ (m_a, m_b, "Bad scheduling");
  NS_TEST_EXPECT_MSG_EQ (m_a, m_c, "Bad scheduling");
  NS_TEST_EXPECT_MSG_EQ (m_a, m_d, "Bad scheduling");
}

/**
 * \ingroup threaded-tests
 *  
 * \brief The threaded simulator Test Suite.
 */
class ThreadedSimulatorTestSuite : public TestSuite
{
public:
  ThreadedSimulatorTestSuite ()
    : TestSuite ("threaded-simulator")
  {
    std::string simulatorTypes[] = {
#ifdef HAVE_RT
      "ns3::RealtimeSimulatorImpl",
#endif
      "ns3::DefaultSimulatorImpl"
    };
    std::string schedulerTypes[] = {
      "ns3::ListScheduler",
      "ns3::HeapScheduler",
      "ns3::MapScheduler",
      "ns3::CalendarScheduler"
    };
    unsigned int threadcounts[] = {
      0,
      2,
      10,
      20
    };
    ObjectFactory factory;

    for (unsigned int i = 0; i < (sizeof(simulatorTypes) / sizeof(simulatorTypes[0])); ++i)
      {
        for (unsigned int j = 0; j < (sizeof(threadcounts) / sizeof(threadcounts[0])); ++j)
          {
            for (unsigned int k = 0; k < (sizeof(schedulerTypes) / sizeof(schedulerTypes[0])); ++k)
              {
                factory.SetTypeId (schedulerTypes[k]);
                AddTestCase (new ThreadedSimulatorEventsTestCase (factory, simulatorTypes[i], threadcounts[j]), TestCase::QUICK);
              }
          }
      }
  }
};

/// Static variable for test initialization.
static ThreadedSimulatorTestSuite g_threadedSimulatorTestSuite; 
