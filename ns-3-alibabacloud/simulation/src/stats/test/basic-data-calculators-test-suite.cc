/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 University of Washington
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
#include "ns3/basic-data-calculators.h"

using namespace ns3;

const double TOLERANCE = 1e-14;

/**
 * \ingroup stats-tests
 *
 * \brief MinMaxAvgTotalCalculator class - Test case for a single integer.
 */
class OneIntegerTestCase : public TestCase
{
public:
  OneIntegerTestCase ();
  virtual ~OneIntegerTestCase ();

private:
  virtual void DoRun (void);
};

OneIntegerTestCase::OneIntegerTestCase ()
  : TestCase ("Basic Statistical Functions using One Integer")

{
}

OneIntegerTestCase::~OneIntegerTestCase ()
{
}

void
OneIntegerTestCase::DoRun (void)
{
  MinMaxAvgTotalCalculator<int> calculator;

  long count = 1;

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
  variance = 0;
  stddev = std::sqrt (variance);

  // Test the calculator.
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getCount(),    count,    TOLERANCE, "Count value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSum(),      sum,      TOLERANCE, "Sum value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMin(),      min,      TOLERANCE, "Min value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMax(),      max,      TOLERANCE, "Max value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMean(),     mean,     TOLERANCE, "Mean value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getStddev(),   stddev,   TOLERANCE, "Stddev value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getVariance(), variance, TOLERANCE, "Variance value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSqrSum(),   sqrSum,   TOLERANCE, "SqrSum value wrong");
}


/**
 * \ingroup stats-tests
 *
 * \brief MinMaxAvgTotalCalculator class - Test case for five integers.
 */
class FiveIntegersTestCase : public TestCase
{
public:
  FiveIntegersTestCase ();
  virtual ~FiveIntegersTestCase ();

private:
  virtual void DoRun (void);
};

FiveIntegersTestCase::FiveIntegersTestCase ()
  : TestCase ("Basic Statistical Functions using Five Integers")

{
}

FiveIntegersTestCase::~FiveIntegersTestCase ()
{
}

void
FiveIntegersTestCase::DoRun (void)
{
  MinMaxAvgTotalCalculator<int> calculator;

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
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getCount(),    count,    TOLERANCE, "Count value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSum(),      sum,      TOLERANCE, "Sum value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMin(),      min,      TOLERANCE, "Min value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMax(),      max,      TOLERANCE, "Max value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMean(),     mean,     TOLERANCE, "Mean value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getStddev(),   stddev,   TOLERANCE, "Stddev value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getVariance(), variance, TOLERANCE, "Variance value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSqrSum(),   sqrSum,   TOLERANCE, "SqrSum value wrong");
}


/**
 * \ingroup stats-tests
 *
 * \brief MinMaxAvgTotalCalculator class - Test case for five double values.
 */
class FiveDoublesTestCase : public TestCase
{
public:
  FiveDoublesTestCase ();
  virtual ~FiveDoublesTestCase ();

private:
  virtual void DoRun (void);
};

FiveDoublesTestCase::FiveDoublesTestCase ()
  : TestCase ("Basic Statistical Functions using Five Double Values")

{
}

FiveDoublesTestCase::~FiveDoublesTestCase ()
{
}

void
FiveDoublesTestCase::DoRun (void)
{
  MinMaxAvgTotalCalculator<double> calculator;

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
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getCount(),    count,    TOLERANCE, "Count value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSum(),      sum,      TOLERANCE, "Sum value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMin(),      min,      TOLERANCE, "Min value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMax(),      max,      TOLERANCE, "Max value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getMean(),     mean,     TOLERANCE, "Mean value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getStddev(),   stddev,   TOLERANCE, "Stddev value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getVariance(), variance, TOLERANCE, "Variance value wrong");
  NS_TEST_ASSERT_MSG_EQ_TOL (calculator.getSqrSum(),   sqrSum,   TOLERANCE, "SqrSum value wrong");
}


/**
 * \ingroup stats-tests
 *
 * \brief MinMaxAvgTotalCalculator class TestSuite
 */
class BasicDataCalculatorsTestSuite : public TestSuite
{
public:
  BasicDataCalculatorsTestSuite ();
};

BasicDataCalculatorsTestSuite::BasicDataCalculatorsTestSuite ()
  : TestSuite ("basic-data-calculators", UNIT)
{
  AddTestCase (new OneIntegerTestCase, TestCase::QUICK);
  AddTestCase (new FiveIntegersTestCase, TestCase::QUICK);
  AddTestCase (new FiveDoublesTestCase, TestCase::QUICK);
}

 /// Static variable for test initialization
 static BasicDataCalculatorsTestSuite basicDataCalculatorsTestSuite;
