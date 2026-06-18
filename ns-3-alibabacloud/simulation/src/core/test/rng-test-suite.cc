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
 */


#include <cmath>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_histogram.h>
#include <ctime>
#include <fstream>

#include "ns3/test.h"
#include "ns3/double.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"

using namespace ns3;

/**
 * \file
 * \ingroup rng-tests
 * Random number generators tests.
 */

/**
 * \ingroup core-tests
 * \defgroup rng-tests Random number generators tests
 */

/**
 * \ingroup rng-tests
 * 
 * Fill an array with increasing values, in the [start, end] range.
 * \param array The array to fill.
 * \param n The size of the array.
 * \param start The start value.
 * \param end The end value.
 */
void
FillHistoRangeUniformly (double *array, uint32_t n, double start, double end)
{
  double increment = (end - start) / (n - 1.);
  double d = start;

  for (uint32_t i = 0; i < n; ++i)
    {
      array[i] = d;
      d += increment;
    }
}

/**
 * \ingroup core-tests
 * 
 * Test case for uniform distribution random number generator.
 */
class RngUniformTestCase : public TestCase
{
public:
  /// Number of runs.
  static const uint32_t N_RUNS = 5;
  /// Number of bins.
  static const uint32_t N_BINS = 50;
  /// Number of measurements.
  static const uint32_t N_MEASUREMENTS = 1000000;

  RngUniformTestCase ();
  virtual ~RngUniformTestCase ();

  /**
   * Run a chi-squared test on the results of the random number generator.
   * \param u The random number generaor.
   * \return the chi-squared test result.
   */
  double ChiSquaredTest (Ptr<UniformRandomVariable> u);

private:
  virtual void DoRun (void);
};

RngUniformTestCase::RngUniformTestCase ()
  : TestCase ("Uniform Random Number Generator")
{}

RngUniformTestCase::~RngUniformTestCase ()
{}

double
RngUniformTestCase::ChiSquaredTest (Ptr<UniformRandomVariable> u)
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  gsl_histogram_set_ranges_uniform (h, 0., 1.);

  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      gsl_histogram_increment (h, u->GetValue ());
    }

  double tmp[N_BINS];

  double expected = ((double)N_MEASUREMENTS / (double)N_BINS);

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      tmp[i] = gsl_histogram_get (h, i);
      tmp[i] -= expected;
      tmp[i] *= tmp[i];
      tmp[i] /= expected;
    }

  gsl_histogram_free (h);

  double chiSquared = 0;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      chiSquared += tmp[i];
    }

  return chiSquared;
}

void
RngUniformTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (static_cast<uint32_t> (time (0)));

  double sum = 0.;
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);

  for (uint32_t i = 0; i < N_RUNS; ++i)
    {
      Ptr<UniformRandomVariable> u = CreateObject<UniformRandomVariable> ();
      double result = ChiSquaredTest (u);
      sum += result;
    }

  sum /= (double)N_RUNS;

  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");
}

/**
 * \ingroup rng-tests
 * 
 * Test case for normal distribution random number generator.
 */
class RngNormalTestCase : public TestCase
{
public:
  /// Number of runs.
  static const uint32_t N_RUNS = 5;
  /// Number of bins.
  static const uint32_t N_BINS = 50;
  /// Number of measurements.
  static const uint32_t N_MEASUREMENTS = 1000000;

  RngNormalTestCase ();
  virtual ~RngNormalTestCase ();

  /**
   * Run a chi-squared test on the results of the random number generator.
   * \param n The random number generaor.
   * \return the chi-squared test result.
   */
  double ChiSquaredTest (Ptr<NormalRandomVariable> n);

private:
  virtual void DoRun (void);
};

RngNormalTestCase::RngNormalTestCase ()
  : TestCase ("Normal Random Number Generator")
{}

RngNormalTestCase::~RngNormalTestCase ()
{}

double
RngNormalTestCase::ChiSquaredTest (Ptr<NormalRandomVariable> n)
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);

  double range[N_BINS + 1];
  FillHistoRangeUniformly (range, N_BINS + 1, -4., 4.);
  range[0] = -std::numeric_limits<double>::max ();
  range[N_BINS] = std::numeric_limits<double>::max ();

  gsl_histogram_set_ranges (h, range, N_BINS + 1);

  double expected[N_BINS];

  double sigma = 1.;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gaussian_P (range[i + 1], sigma) - gsl_cdf_gaussian_P (range[i], sigma);
      expected[i] *= N_MEASUREMENTS;
    }

  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      gsl_histogram_increment (h, n->GetValue ());
    }

  double tmp[N_BINS];

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      tmp[i] = gsl_histogram_get (h, i);
      tmp[i] -= expected[i];
      tmp[i] *= tmp[i];
      tmp[i] /= expected[i];
    }

  gsl_histogram_free (h);

  double chiSquared = 0;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      chiSquared += tmp[i];
    }

  return chiSquared;
}

void
RngNormalTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (static_cast<uint32_t> (time (0)));

  double sum = 0.;
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);

  for (uint32_t i = 0; i < N_RUNS; ++i)
    {
      Ptr<NormalRandomVariable> n = CreateObject<NormalRandomVariable> ();
      double result = ChiSquaredTest (n);
      sum += result;
    }

  sum /= (double)N_RUNS;

  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");
}

/**
 * \ingroup rng-tests
 * 
 * Test case for exponential distribution random number generator.
 */
class RngExponentialTestCase : public TestCase
{
public:
  /// Number of runs.
  static const uint32_t N_RUNS = 5;
  /// Number of bins.
  static const uint32_t N_BINS = 50;
  /// Number of measurements.
  static const uint32_t N_MEASUREMENTS = 1000000;

  RngExponentialTestCase ();
  virtual ~RngExponentialTestCase ();

  /**
   * Run a chi-squared test on the results of the random number generator.
   * \param n The random number generaor.
   * \return the chi-squared test result.
   */
  double ChiSquaredTest (Ptr<ExponentialRandomVariable> n);

private:
  virtual void DoRun (void);
};

RngExponentialTestCase::RngExponentialTestCase ()
  : TestCase ("Exponential Random Number Generator")
{}

RngExponentialTestCase::~RngExponentialTestCase ()
{}

double
RngExponentialTestCase::ChiSquaredTest (Ptr<ExponentialRandomVariable> e)
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);

  double range[N_BINS + 1];
  FillHistoRangeUniformly (range, N_BINS + 1, 0., 10.);
  range[N_BINS] = std::numeric_limits<double>::max ();

  gsl_histogram_set_ranges (h, range, N_BINS + 1);

  double expected[N_BINS];

  double mu = 1.;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_exponential_P (range[i + 1], mu) - gsl_cdf_exponential_P (range[i], mu);
      expected[i] *= N_MEASUREMENTS;
    }

  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      gsl_histogram_increment (h, e->GetValue ());
    }

  double tmp[N_BINS];

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      tmp[i] = gsl_histogram_get (h, i);
      tmp[i] -= expected[i];
      tmp[i] *= tmp[i];
      tmp[i] /= expected[i];
    }

  gsl_histogram_free (h);

  double chiSquared = 0;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      chiSquared += tmp[i];
    }

  return chiSquared;
}

void
RngExponentialTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (static_cast<uint32_t> (time (0)));

  double sum = 0.;
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);

  for (uint32_t i = 0; i < N_RUNS; ++i)
    {
      Ptr<ExponentialRandomVariable> e = CreateObject<ExponentialRandomVariable> ();
      double result = ChiSquaredTest (e);
      sum += result;
    }

  sum /= (double)N_RUNS;

  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");
}

/**
 * \ingroup rng-tests
 * 
 * Test case for pareto distribution random number generator.
 */
class RngParetoTestCase : public TestCase
{
public:
  /// Number of runs.
  static const uint32_t N_RUNS = 5;
  /// Number of bins.
  static const uint32_t N_BINS = 50;
  /// Number of measurements.
  static const uint32_t N_MEASUREMENTS = 1000000;

  RngParetoTestCase ();
  virtual ~RngParetoTestCase ();

  /**
   * Run a chi-squared test on the results of the random number generator.
   * \param p The random number generaor.
   * \return the chi-squared test result.
   */
  double ChiSquaredTest (Ptr<ParetoRandomVariable> p);

private:
  virtual void DoRun (void);
};

RngParetoTestCase::RngParetoTestCase ()
  : TestCase ("Pareto Random Number Generator")
{}

RngParetoTestCase::~RngParetoTestCase ()
{}

double
RngParetoTestCase::ChiSquaredTest (Ptr<ParetoRandomVariable> p)
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);

  double range[N_BINS + 1];
  FillHistoRangeUniformly (range, N_BINS + 1, 1., 10.);
  range[N_BINS] = std::numeric_limits<double>::max ();

  gsl_histogram_set_ranges (h, range, N_BINS + 1);

  double expected[N_BINS];

  double a = 1.5;
  double b = 0.33333333;

  // mean is 1 with these values

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_pareto_P (range[i + 1], a, b) - gsl_cdf_pareto_P (range[i], a, b);
      expected[i] *= N_MEASUREMENTS;
    }

  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      gsl_histogram_increment (h, p->GetValue ());
    }

  double tmp[N_BINS];

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      tmp[i] = gsl_histogram_get (h, i);
      tmp[i] -= expected[i];
      tmp[i] *= tmp[i];
      tmp[i] /= expected[i];
    }

  gsl_histogram_free (h);

  double chiSquared = 0;

  for (uint32_t i = 0; i < N_BINS; ++i)
    {
      chiSquared += tmp[i];
    }

  return chiSquared;
}

void
RngParetoTestCase::DoRun (void)
{
  RngSeedManager::SetSeed (static_cast<uint32_t> (time (0)));

  double sum = 0.;
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);

  for (uint32_t i = 0; i < N_RUNS; ++i)
    {
      Ptr<ParetoRandomVariable> e = CreateObject<ParetoRandomVariable> ();
      e->SetAttribute ("Shape", DoubleValue (1.5));
      e->SetAttribute ("Scale", DoubleValue (0.33333333));
      double result = ChiSquaredTest (e);
      sum += result;
    }

  sum /= (double)N_RUNS;

  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");
}

/**
 * \ingroup rng-tests
 *  
 * \brief The random number generators Test Suite.
 */
class RngTestSuite : public TestSuite
{
public:
  RngTestSuite ();
};

RngTestSuite::RngTestSuite ()
  : TestSuite ("random-number-generators", UNIT)
{
  AddTestCase (new RngUniformTestCase, TestCase::QUICK);
  AddTestCase (new RngNormalTestCase, TestCase::QUICK);
  AddTestCase (new RngExponentialTestCase, TestCase::QUICK);
  AddTestCase (new RngParetoTestCase, TestCase::QUICK);
}

static RngTestSuite g_rngTestSuite; //!< Static variable for test initialization