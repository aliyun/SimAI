/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington
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
 * Author: Mitch Watrous (watrous@u.washington.edu)
 */

#include <cmath>

#include "ns3/test.h"
#include "ns3/average.h"

using namespace ns3;

// Note, the rationale for this particular value of TOLERANCE is not
// documented.  Current value is sufficient for all test platforms.
const double TOLERANCE = 2e-14;

/**
 * \ingroup stats-tests
 *
 * \brief Average class - Test case for a single integer.
 */
class OneIntegerAverageTestCase : public TestCase
{
public:
  OneIntegerAverageTestCase ();
  virtual ~OneIntegerAverageTestCase ();

private:
  virtual void DoRun (void);
};

OneIntegerAverageTestCase::OneIntegerAverageTestCase ()
  : TestCase ("Average Object Test using One Integer")

{
}

OneIntegerAverageTestCase::~OneIntegerAverageTestCase ()
{
}

void
OneIntegerAverageTestCase::DoRun (void)
{
  Average<int> calculator;

  long count = 1;

  double sum = 0;
  double min;
  double max;
  double mean;
  double stddev;
  double variance;

  // Put all of the values into the calculator.
  int multiple = 5;
  int value;
  for (long i = 0; i < count; i++)
    {
      value = multiple * (i + 1);

      calculator.Update (value);

      sum    += value;
    }

  // Calculate the expected values for the statistical functions.
  min = multiple;
  max = multiple * count;
  mean = sum / count;
  variance = 0;
  stddev = std::sqrt (variance);

  // Test the calculator.
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Count (), count, TOLERANCE, "Count value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Count () - count);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Min (), min, TOLERANCE, "Min value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Min () - min);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Max (), max, TOLERANCE, "Max value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Max () - max);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Mean (), mean, TOLERANCE, "Mean value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Mean () - mean);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Stddev (), stddev, TOLERANCE, "Stddev value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Stddev () - stddev);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Var (), variance, TOLERANCE, "Variance value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Var () - variance);
}


/**
 * \ingroup stats-tests
 *
 * \brief Average class - Test case for five integers.
 */
class FiveIntegersAverageTestCase : public TestCase
{
public:
  FiveIntegersAverageTestCase ();
  virtual ~FiveIntegersAverageTestCase ();

private:
  virtual void DoRun (void);
};

FiveIntegersAverageTestCase::FiveIntegersAverageTestCase ()
  : TestCase ("Average Object Test using Five Integers")

{
}

FiveIntegersAverageTestCase::~FiveIntegersAverageTestCase ()
{
}

void
FiveIntegersAverageTestCase::DoRun (void)
{
  Average<int> calculator;

  long count = 5;

  double sum = 0;
  double sqrSum = 0;
  double min;
  double max;
  double mean;
  double stddev;
  double variance;

  // Put all of the values into the calculator.
  int multiple = 5;
  int value;
  for (long i = 0; i < count; i++)
    {
      value = multiple * (i + 1);

      calculator.Update (value);

      sum    += value;
      sqrSum += value * value;
    }

  // Calculate the expected values for the statistical functions.
  min = multiple;
  max = multiple * count;
  mean = sum / count;
  variance = (count * sqrSum - sum * sum) / (count * (count - 1));
  stddev = std::sqrt (variance);

  // Test the calculator.
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Count (), count, TOLERANCE, "Count value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Count () - count);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Min (), min, TOLERANCE, "Min value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Min () - min);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Max (), max, TOLERANCE, "Max value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Max () - max);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Mean (), mean, TOLERANCE, "Mean value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Mean () - mean);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Stddev (), stddev, TOLERANCE, "Stddev value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Stddev () - stddev);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Var (), variance, TOLERANCE, "Variance value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Var () - variance);
}


/**
 * \ingroup stats-tests
 *
 * \brief Average class - Test case for five double values.
 */
class FiveDoublesAverageTestCase : public TestCase
{
public:
  FiveDoublesAverageTestCase ();
  virtual ~FiveDoublesAverageTestCase ();

private:
  virtual void DoRun (void);
};

FiveDoublesAverageTestCase::FiveDoublesAverageTestCase ()
  : TestCase ("Average Object Test using Five Double Values")

{
}

FiveDoublesAverageTestCase::~FiveDoublesAverageTestCase ()
{
}

void
FiveDoublesAverageTestCase::DoRun (void)
{
  Average<double> calculator;

  long count = 5;

  double sum = 0;
  double sqrSum = 0;
  double min;
  double max;
  double mean;
  double stddev;
  double variance;

  // Put all of the values into the calculator.
  double multiple = 3.14;
  double value;
  for (long i = 0; i < count; i++)
    {
      value = multiple * (i + 1);

      calculator.Update (value);

      sum    += value;
      sqrSum += value * value;
    }

  // Calculate the expected values for the statistical functions.
  min = multiple;
  max = multiple * count;
  mean = sum / count;
  variance = (count * sqrSum - sum * sum) / (count * (count - 1));
  stddev = std::sqrt (variance);

  // Test the calculator.
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Count (), count, TOLERANCE, "Count value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Count () - count);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Min (), min, TOLERANCE, "Min value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Min () - min);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Max (), max, TOLERANCE, "Max value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Max () - max);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Mean (), mean, TOLERANCE, "Mean value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Mean () - mean);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Stddev (), stddev, TOLERANCE, "Stddev value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Stddev () - stddev);
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.Var (), variance, TOLERANCE, "Variance value outside of tolerance " << TOLERANCE << "; difference: " << calculator.Var () - variance);
}


/**
 * \ingroup stats-tests
 *
 * \brief Average class TestSuite
 */
class AverageTestSuite : public TestSuite
{
public:
  AverageTestSuite ();
};

AverageTestSuite::AverageTestSuite ()
  : TestSuite ("average", UNIT)
{
  AddTestCase (new OneIntegerAverageTestCase, TestCase::QUICK);
  AddTestCase (new FiveIntegersAverageTestCase, TestCase::QUICK);
  AddTestCase (new FiveDoublesAverageTestCase, TestCase::QUICK);
}

 /// Static variable for test initialization
static AverageTestSuite averageTestSuite;
