/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INESC Porto
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
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */

#include "ns3/test.h"
#include "ns3/event-garbage-collector.h"

/**
 * \file
 * \ingroup core-tests
 * \ingroup events
 * \ingroup events-garbage-tests
 * EventGarbageCollector test suite.
 */

/**
 * \ingroup core-tests
 * \defgroup event-garbage-tests EventGarbageCollector test suite
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup event-garbage-tests
 * Event garbage collector test.
 */
class EventGarbageCollectorTestCase : public TestCase
{
  int m_counter; //!< Counter to trigger deletion of events.
  EventGarbageCollector *m_events; //!< Object under test.

  /** Callback to record event invocations. */
  void EventGarbageCollectorCallback ();

public:

  /** Constructor. */
  EventGarbageCollectorTestCase ();
  /** Destructor. */
  virtual ~EventGarbageCollectorTestCase ();
  virtual void DoRun (void);
};

EventGarbageCollectorTestCase::EventGarbageCollectorTestCase ()
  : TestCase ("EventGarbageCollector"), m_counter (0), m_events (0)
{}

EventGarbageCollectorTestCase::~EventGarbageCollectorTestCase ()
{}

void
EventGarbageCollectorTestCase::EventGarbageCollectorCallback ()
{
  m_counter++;
  if (m_counter == 50)
    {
      // this should cause the remaining (50) events to be cancelled
      delete m_events;
      m_events = 0;
    }
}

void EventGarbageCollectorTestCase::DoRun (void)
{
  m_events = new EventGarbageCollector ();

  for (int n = 0; n < 100; n++)
    {
      m_events->Track (Simulator::Schedule
                         (Simulator::Now (),
                         &EventGarbageCollectorTestCase::EventGarbageCollectorCallback,
                         this));
    }
  Simulator::Run ();
  NS_TEST_EXPECT_MSG_EQ (m_events, 0, "");
  NS_TEST_EXPECT_MSG_EQ (m_counter, 50, "");
  Simulator::Destroy ();
}

/**
 * \ingroup event-garbage-tests
 * Event garbage collector test suite.
 */
class EventGarbageCollectorTestSuite : public TestSuite
{
public:
  EventGarbageCollectorTestSuite ()
    : TestSuite ("event-garbage-collector")
  {
    AddTestCase (new EventGarbageCollectorTestCase ());
  }
};

/**
 * \ingroup event-garbage-tests
 * EventGarbageCollectorTestSuite instance variable.
 */
static EventGarbageCollectorTestSuite g_eventGarbageCollectorTestSuite;


}    // namespace tests

}  // namespace ns3

