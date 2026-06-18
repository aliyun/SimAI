/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "ns3/timer.h"
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"

/**
 * \file
 * \ingroup timer-tests
 * Timer test suite
 */

/**
 * \ingroup core-tests
 * \defgroup timer-tests Timer tests
 */

namespace {

/* *NS_CHECK_STYLE_OFF* */
/// Function with one int parameter.
void bari (int) {}
/// Function with two int parameters.
void bar2i (int, int) {}
/// Function with three int parameters.
void bar3i (int, int, int) {}
/// Function with four int parameters.
void bar4i (int, int, int, int) {}
/// Function with five int parameters.
void bar5i (int, int, int, int, int) {}
/// Function with one const int reference parameter.
void barcir (const int &) {}
/// Function with one int reference parameter.
void barir (int &) {}
/* *NS_CHECK_STYLE_ON* */

} // anonymous namespace

using namespace ns3;

/**
 * \ingroup timer-tests
 *  
 * \brief Check correct state transitions.
 */
class TimerStateTestCase : public TestCase
{
public:
  TimerStateTestCase ();
  virtual void DoRun (void);
};

TimerStateTestCase::TimerStateTestCase ()
  : TestCase ("Check correct state transitions")
{}
void
TimerStateTestCase::DoRun (void)
{
  Timer timer = Timer (Timer::CANCEL_ON_DESTROY);

  timer.SetFunction (&bari);
  timer.SetArguments (1);
  timer.SetDelay (Seconds (10.0));
  NS_TEST_ASSERT_MSG_EQ (!timer.IsRunning (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.IsExpired (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsSuspended (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.GetState (), Timer::EXPIRED, "");
  timer.Schedule ();
  NS_TEST_ASSERT_MSG_EQ (timer.IsRunning (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsExpired (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsSuspended (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.GetState (), Timer::RUNNING, "");
  timer.Suspend ();
  NS_TEST_ASSERT_MSG_EQ (!timer.IsRunning (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsExpired (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.IsSuspended (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.GetState (), Timer::SUSPENDED, "");
  timer.Resume ();
  NS_TEST_ASSERT_MSG_EQ (timer.IsRunning (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsExpired (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsSuspended (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.GetState (), Timer::RUNNING, "");
  timer.Cancel ();
  NS_TEST_ASSERT_MSG_EQ (!timer.IsRunning (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.IsExpired (), true, "");
  NS_TEST_ASSERT_MSG_EQ (!timer.IsSuspended (), true, "");
  NS_TEST_ASSERT_MSG_EQ (timer.GetState (), Timer::EXPIRED, "");
}

/**
 * \ingroup timer-tests
 *  
 * \brief Check that Timer template magic is working.
 */
class TimerTemplateTestCase : public TestCase
{
public:
  TimerTemplateTestCase ();
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /* *NS_CHECK_STYLE_OFF* */
  /// Member function with one int parameter.
  void bazi (int) {}
  /// Member function with two int parameters.
  void baz2i (int, int) {}
  /// Member function with three int parameters.
  void baz3i (int, int, int) {}
  /// Member function with four int parameters.
  void baz4i (int, int, int, int) {}
  /// Member function with five int parameters.
  void baz5i (int, int, int, int, int) {}
  /// Member function with six int parameters.
  void baz6i (int, int, int, int, int, int) {}
  /// Member function with one const int reference parameter.
  void bazcir (const int&) {}
  /// Member function with one int reference parameter.
  void bazir (int&) {}
  /// Member function with one int pointer parameter.
  void bazip (int *) {}
  /// Member function with one const int pointer parameter.
  void bazcip (const int *) {}
  /* *NS_CHECK_STYLE_ON* */
};

TimerTemplateTestCase::TimerTemplateTestCase ()
  : TestCase ("Check that template magic is working")
{}

void
TimerTemplateTestCase::DoRun (void)
{
  Timer timer = Timer (Timer::CANCEL_ON_DESTROY);

  int a = 0;
  int &b = a;
  const int &c = a;

  timer.SetFunction (&bari);
  timer.SetArguments (2);
  timer.SetArguments (a);
  timer.SetArguments (b);
  timer.SetArguments (c);
  timer.SetFunction (&barir);
  timer.SetArguments (2);
  timer.SetArguments (a);
  timer.SetArguments (b);
  timer.SetArguments (c);
  timer.SetFunction (&barcir);
  timer.SetArguments (2);
  timer.SetArguments (a);
  timer.SetArguments (b);
  timer.SetArguments (c);
  // the following call cannot possibly work and is flagged by
  // a runtime error.
  // timer.SetArguments (0.0);
  timer.SetDelay (Seconds (1.0));
  timer.Schedule ();

  timer.SetFunction (&TimerTemplateTestCase::bazi, this);
  timer.SetArguments (3);
  timer.SetFunction (&TimerTemplateTestCase::bazir, this);
  timer.SetArguments (3);
  timer.SetFunction (&TimerTemplateTestCase::bazcir, this);
  timer.SetArguments (3);

  timer.SetFunction (&bar2i);
  timer.SetArguments (1, 1);
  timer.SetFunction (&bar3i);
  timer.SetArguments (1, 1, 1);
  timer.SetFunction (&bar4i);
  timer.SetArguments (1, 1, 1, 1);
  timer.SetFunction (&bar5i);
  timer.SetArguments (1, 1, 1, 1, 1);
  // unsupported in simulator class
  // timer.SetFunction (&bar6i);
  // timer.SetArguments (1, 1, 1, 1, 1, 1);

  timer.SetFunction (&TimerTemplateTestCase::baz2i, this);
  timer.SetArguments (1, 1);
  timer.SetFunction (&TimerTemplateTestCase::baz3i, this);
  timer.SetArguments (1, 1, 1);
  timer.SetFunction (&TimerTemplateTestCase::baz4i, this);
  timer.SetArguments (1, 1, 1, 1);
  timer.SetFunction (&TimerTemplateTestCase::baz5i, this);
  timer.SetArguments (1, 1, 1, 1, 1);
  // unsupported in simulator class
  // timer.SetFunction (&TimerTemplateTestCase::baz6i, this);
  // timer.SetArguments (1, 1, 1, 1, 1, 1);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
TimerTemplateTestCase::DoTeardown (void)
{
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup timer-tests
 *  
 * \brief The timer Test Suite.
 */
class TimerTestSuite : public TestSuite
{
public:
  TimerTestSuite ()
    : TestSuite ("timer", UNIT)
  {
    AddTestCase (new TimerStateTestCase (), TestCase::QUICK);
    AddTestCase (new TimerTemplateTestCase (), TestCase::QUICK);
  }
};

static TimerTestSuite g_timerTestSuite; //!< Static variable for test initialization
