/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 */

#include "ns3/test.h"
#include "ns3/rtt-estimator.h"
#include "ns3/attribute.h"
#include "ns3/nstime.h"
#include "ns3/config.h"
#include "ns3/log.h"
#include "ns3/double.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RttEstimatorTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief RTT estimator Test
 */
class RttEstimatorTestCase : public TestCase
{
public:
  RttEstimatorTestCase ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /**
   * \brief Check RTT values.
   * \param rtt The RTT estimator.
   * \param m The measurement.
   * \param e The expected value.
   * \param v The expected variance.
   */
  void CheckValues (Ptr<RttEstimator> rtt, Time m, Time e, Time v);
  /**
   * \brief Check RTT values with a 1 nanosecond of tolerance.
   * \param rtt The RTT estimator.
   * \param m The measurement.
   * \param e The expected value.
   * \param v The expected variance.
   */
  void CheckValuesWithTolerance (Ptr<RttEstimator> rtt, Time m, Time e, Time v);
};

RttEstimatorTestCase::RttEstimatorTestCase ()
  : TestCase ("Rtt Estimator Test")
{
}

void
RttEstimatorTestCase::CheckValues (Ptr<RttEstimator> rtt, Time m, Time e, Time v)
{
  rtt->Measurement (m);
  NS_TEST_EXPECT_MSG_EQ (rtt->GetEstimate (), e, "Estimate not correct");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetVariation (), v, "Estimate not correct");
}

void
RttEstimatorTestCase::CheckValuesWithTolerance (Ptr<RttEstimator> rtt, Time m, Time e, Time v)
{
  rtt->Measurement (m);
  NS_TEST_EXPECT_MSG_EQ_TOL (rtt->GetEstimate (), e, Time (NanoSeconds (1)), "Estimate not correct");
  NS_TEST_EXPECT_MSG_EQ_TOL (rtt->GetVariation (), v, Time (NanoSeconds (1)), "Estimate not correct");
}


void
RttEstimatorTestCase::DoRun (void)
{
  // Set to a non-default value
  Config::SetDefault ("ns3::RttEstimator::InitialEstimation", TimeValue (MilliSeconds (500)));
  Config::SetDefault ("ns3::RttMeanDeviation::Alpha", DoubleValue (0.5));
  Config::SetDefault ("ns3::RttMeanDeviation::Beta", DoubleValue (0.6));

  Ptr<RttMeanDeviation> rtt = CreateObject<RttMeanDeviation> ();

  bool ok;
  TimeValue timeval;
  DoubleValue doubleval;
  ok = rtt->GetAttributeFailSafe ("InitialEstimation", timeval);
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be gettable");
  NS_TEST_EXPECT_MSG_EQ (timeval.Get (), MilliSeconds (500), "Initial estimate should match");
  ok = rtt->GetAttributeFailSafe ("Alpha", doubleval);
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be gettable");
  NS_TEST_ASSERT_MSG_EQ_TOL (doubleval.Get (), 0.5, 0.001, "Alpha not set");
  ok = rtt->GetAttributeFailSafe ("Beta", doubleval);
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be gettable");
  NS_TEST_ASSERT_MSG_EQ_TOL (doubleval.Get (), 0.6, 0.001, "Beta not set");

  // Reset to default values
  ok = rtt->SetAttributeFailSafe ("InitialEstimation", TimeValue (Seconds (1)));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  ok = rtt->SetAttributeFailSafe ("Alpha", DoubleValue (0.125));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  ok = rtt->SetAttributeFailSafe ("Beta", DoubleValue (0.25));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  rtt->Reset ();

  Time t (Seconds (1));
  Time t2 (MilliSeconds (125));
  NS_TEST_EXPECT_MSG_EQ (t2, Time::From (t.GetInteger () >> 3), "X");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetEstimate (), Time (Seconds (1)), "Incorrect initial estimate");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetVariation (), Time (Seconds (0)), "Incorrect initial variance");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetNSamples (), 0, "Incorrect initial estimate");

  // CheckValues (rtt, measurement, new estimate, new variance);
  // Initial value:  SRTT <- measurement; RTTVAR <- measurement/2
  CheckValues (rtt, Time (Seconds (1)), Time (Seconds (1)), Time (MilliSeconds (500)));
  // Subsequent values:  according to RFC 6298
  CheckValues (rtt, Time (MilliSeconds (1200)), Time (MilliSeconds (1025)), Time (MilliSeconds (425)));
  Ptr<RttEstimator> copy = rtt->Copy ();
  CheckValues (rtt, Time (MilliSeconds (900)), Time (MicroSeconds (1009375)), Time (MilliSeconds (350)));

  // Check behavior of copy; should have inherited state
  CheckValues (copy, Time (MilliSeconds (900)), Time (MicroSeconds (1009375)), Time (MilliSeconds (350)));

  // Floating point arithmetic due to alpha and beta settings
  rtt->Reset ();
  ok = rtt->SetAttributeFailSafe ("Alpha", DoubleValue (0.1));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  ok = rtt->SetAttributeFailSafe ("Beta", DoubleValue (0.1));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  CheckValuesWithTolerance (rtt, Time (Seconds (1.2)), Time (Seconds (1.2)), Time (Seconds (0.6)));
  CheckValuesWithTolerance (rtt, Time (MilliSeconds (950)), Time (MilliSeconds (1175)), Time (MilliSeconds (565)));
  CheckValuesWithTolerance (rtt, Time (MilliSeconds (1400)), Time (MicroSeconds (1197500)), Time (MilliSeconds (531)));

  // Check boundary values; 0 will not update, 1 will use most recent value
  rtt->Reset ();
  ok = rtt->SetAttributeFailSafe ("Alpha", DoubleValue (0));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  ok = rtt->SetAttributeFailSafe ("Beta", DoubleValue (0));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  CheckValues (rtt, Time (Seconds (1)), Time (Seconds (1)), Time (MilliSeconds (500)));
  CheckValues (rtt, Time (Seconds (2)), Time (Seconds (1)), Time (MilliSeconds (500)));
  CheckValues (rtt, Time (Seconds (3)), Time (Seconds (1)), Time (MilliSeconds (500)));
  rtt->Reset ();
  ok = rtt->SetAttributeFailSafe ("Alpha", DoubleValue (1));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  ok = rtt->SetAttributeFailSafe ("Beta", DoubleValue (1));
  NS_TEST_EXPECT_MSG_EQ (ok, true, "Attribute should be settable");
  CheckValues (rtt, Time (Seconds (1)), Time (Seconds (1)), Time (MilliSeconds (500)));
  CheckValues (rtt, Time (Seconds (2.5)), Time (Seconds (2.5)), Time (Seconds (1.5)));
  CheckValues (rtt, Time (Seconds (7)), Time (Seconds (7)), Time (Seconds (4.5)));
  
  // recheck initial values
  rtt->Reset ();
  NS_TEST_EXPECT_MSG_EQ (rtt->GetEstimate (), Time (Seconds (1)), "Incorrect initial estimate");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetVariation (), Time (Seconds (0)), "Incorrect initial variation");
  NS_TEST_EXPECT_MSG_EQ (rtt->GetNSamples (), 0, "Incorrect initial estimate");
}

void
RttEstimatorTestCase::DoTeardown (void)
{
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief RTT estimator TestSuite
 */
class RttEstimatorTestSuite : public TestSuite
{
public:
  RttEstimatorTestSuite ()
    : TestSuite ("rtt-estimator", UNIT)
  {
    AddTestCase (new RttEstimatorTestCase, TestCase::QUICK);
  }

};

static RttEstimatorTestSuite  g_rttEstimatorTestSuite; //!< Static variable for test initialization

