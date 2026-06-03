/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/trickle-timer.h"
#include "ns3/test.h"
#include <vector>
#include <numeric>
#include <algorithm>

/**
 * \file
 * \ingroup core-tests
 * \ingroup timer
 * \ingroup timer-tests
 *
 * Trickle Timer test suite.
 *
 * This test checks that the timer, in steady-state mode (i.e., after
 * the transient period) have a frequency between [I/2, I + I/2].
 *
 * The test also checks that the redundancy works, i.e., if the timer
 * receives enough "consistent" events it will suppress its output.
 */

namespace ns3 {

namespace tests {


/**
 * \ingroup timer-tests
 *  TrickleTimer test
 */
class TrickleTimerTestCase : public TestCase
{
public:
  /** Constructor. */
  TrickleTimerTestCase ();
  virtual void DoRun (void);
  /**
   * Function to invoke when TrickleTimer expires.
   */
  void ExpireTimer (void);
  std::vector<Time> m_expiredTimes;     //!< Time when TrickleTimer expired

  /**
   * Function to signal that the transient is over
   */
  void TransientOver (void);

  /**
   * Test the steady-state
   * \param unit Minimum interval
   */
  void TestSteadyState (Time unit);

  /**
   * Test the redundancy suppression
   * \param unit Minimum interval
   */
  void TestRedundancy (Time unit);

  /**
   * Inject in the timer a consistent event
   * \param interval Interval
   * \param tricklePtr Pointer to the TrickleTimer
   */
  void ConsistentEvent (Time interval, TrickleTimer* tricklePtr);

  bool m_enableDataCollection;     //!< Collect data if true
};

TrickleTimerTestCase::TrickleTimerTestCase ()
  : TestCase ("Check the Trickle Timer algorithm")
{}

void
TrickleTimerTestCase::ExpireTimer (void)
{
  if (m_enableDataCollection==false)
    {
      return;
    }

  m_expiredTimes.push_back (Simulator::Now ());
}

void
TrickleTimerTestCase::TransientOver (void)
{
  m_enableDataCollection = true;
}

void
TrickleTimerTestCase::TestSteadyState (Time unit)
{
  m_expiredTimes.clear ();
  m_enableDataCollection = false;

  TrickleTimer trickle (unit, 4, 1);
  trickle.SetFunction (&TrickleTimerTestCase::ExpireTimer, this);
  trickle.Enable ();
  // We reset the timer to force the interval to the minimum
  trickle.Reset ();

  NS_TEST_EXPECT_MSG_EQ (trickle.GetDoublings (), 4, "The doublings re-compute mechanism is not working.");

  // The transient is over at (exp2(doublings +1) -1) * MinInterval (worst case).
  Simulator::Schedule (unit*31, &TrickleTimerTestCase::TransientOver, this);

  Simulator::Stop (unit * 50000);

  Simulator::Run ();
  Simulator::Destroy ();

  std::vector<Time> expirationFrequency;

  expirationFrequency.resize (m_expiredTimes.size ());
  std::adjacent_difference (m_expiredTimes.begin (), m_expiredTimes.end (), expirationFrequency.begin ());
  expirationFrequency.erase (expirationFrequency.begin ());

  int64x64_t min = (*std::min_element (expirationFrequency.begin (), expirationFrequency.end ()))/unit;
  int64x64_t max = (*std::max_element (expirationFrequency.begin (), expirationFrequency.end ()))/unit;

  NS_TEST_EXPECT_MSG_GT_OR_EQ (min.GetDouble (), 8, "Timer did fire too fast ??");
  NS_TEST_EXPECT_MSG_LT_OR_EQ (max.GetDouble (), 24, "Timer did fire too slow ??");
}

void
TrickleTimerTestCase::TestRedundancy (Time unit)
{
  m_expiredTimes.clear ();
  m_enableDataCollection = false;

  TrickleTimer trickle (unit, 4, 1);
  trickle.SetFunction (&TrickleTimerTestCase::ExpireTimer, this);
  trickle.Enable ();
  // We reset the timer to force the interval to the minimum
  trickle.Reset ();

  NS_TEST_EXPECT_MSG_EQ (trickle.GetDoublings (), 4, "The doublings re-compute mechanism is not working.");

  // The transient is over at (exp2(doublings +1) -1) * MinInterval (worst case).
  Simulator::Schedule (unit*31, &TrickleTimerTestCase::TransientOver, this);
  Simulator::Schedule (unit*31, &TrickleTimerTestCase::ConsistentEvent, this, unit*8, &trickle);

  Simulator::Stop (unit * 50000);

  Simulator::Run ();
  Simulator::Destroy ();

  NS_TEST_EXPECT_MSG_EQ (m_expiredTimes.size (), 0, "Timer did fire while being suppressed ??");
}

void
TrickleTimerTestCase::ConsistentEvent (Time interval, TrickleTimer* tricklePtr)
{
  tricklePtr->ConsistentEvent ();
  Simulator::Schedule (interval, &TrickleTimerTestCase::ConsistentEvent, this, interval, tricklePtr);
}


void
TrickleTimerTestCase::DoRun (void)
{
  TestSteadyState (Time (1));
  TestSteadyState (Seconds (1));
  TestRedundancy (Seconds (1));
}


/**
 * \ingroup timer-tests
 *  Trickle Timer test suite
 */
class TrickleTimerTestSuite : public TestSuite
{
public:
  /** Constructor. */
  TrickleTimerTestSuite ()
    : TestSuite ("trickle-timer")
  {
    AddTestCase (new TrickleTimerTestCase ());
  }
};

/**
 * \ingroup timer-tests
 * TrickleTimerTestSuite instance variable.
 */
static TrickleTimerTestSuite g_trickleTimerTestSuite;


}    // namespace tests

}  // namespace ns3
