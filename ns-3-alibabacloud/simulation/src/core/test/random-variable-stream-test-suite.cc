/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009-12 University of Washington
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
 * This file is based on rng-test-suite.cc.
 *
 * Modified by Mitch Watrous <watrous@u.washington.edu>
 *
 */


#include <gsl/gsl_cdf.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_sf_zeta.h>
#include <ctime>
#include <fstream>
#include <cmath>

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/integer.h"
#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/random-variable-stream.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("RandomVariableStreamGenerators");

namespace ns3 {

namespace test {

namespace RandomVariable {

/**
 * \file
 * \ingroup rng-tests
 * Random number generator streams tests.
 */

/**
 * \ingroup rng-tests
 * Base class for RandomVariableStream test suites.
 */
class TestCaseBase : public TestCase
{
public:
  /** Number of bins for sampling the distributions. */
  static const uint32_t N_BINS {50};
  /** Number of samples to draw when populating the distributions. */
  static const uint32_t N_MEASUREMENTS {1000000};
  /** Number of retry attempts to pass a chi-square test. */
  static const uint32_t N_RUNS {5};

  /**
   * Constructor
   * \param [in] name The test case name.
   */
  TestCaseBase (std::string name)
    : TestCase (name)
  {}

  /**
   * Configure a GSL histogram with uniform bins, with optional
   * under/over-flow bins.
   * \param [in,out] h The GSL histogram to configure.
   * \param [in] start The minimum value of the lowest bin.
   * \param [in] end The maximum value of the last bin.
   * \param [in] underflow If \c true the lowest bin should contain the underflow,
   * \param [in] overflow If \c ture the highest bin should contain the overflow.
   * \returns A vector of the bin edges, including the top of the highest bin.
   * This vector has one more entry than the number of bins in the histogram.
   */
  std::vector<double>
  UniformHistogramBins (gsl_histogram *h, double start, double end,
                      bool underflow = true, bool overflow = true) const
  {
    NS_LOG_FUNCTION (this << h << start << end);
    std::size_t nBins = gsl_histogram_bins (h);
    double increment = (end - start) / (nBins - 1.);
    double d = start;

    std::vector<double> range (nBins + 1);

    for (auto & r : range)
      {
        r = d;
        d += increment;
      }
    if (underflow)
      {
        range[0] = -std::numeric_limits<double>::max ();
      }
    if (overflow)
      {
        range[nBins] = std::numeric_limits<double>::max ();
      }

    gsl_histogram_set_ranges (h, range.data (), nBins + 1);
    return range;
  }

  /**
   * Compute the average of a random variable.
   * \param [in] rng The random variable to sample.
   * \returns The average of \c N_MEASUREMENTS samples.
   */
  double
  Average (Ptr<RandomVariableStream> rng) const
  {
    NS_LOG_FUNCTION (this << rng);
    double sum = 0.0;
    for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
      {
        double value = rng->GetValue ();
        sum += value;
      }
    double valueMean = sum / N_MEASUREMENTS;
    return valueMean;
  }

  /** A factory base class to create new instances of a random variable. */
  class RngGeneratorBase
  {
  public:
    /**
     * Create a new instance of a random variable stream
     * \returns The new random variable stream instance.
     */
    virtual Ptr<RandomVariableStream> Create (void) const = 0;
  };

  /**
   * Factory class to create new instances of a particular random variable stream.
   *
   * \tparam RNG The type of random variable generator to create.
   */
  template <typename RNG>
  class RngGenerator : public RngGeneratorBase
  {
  public:
    /**
     * Constructor.
     * \param [in] anti Create antithetic streams if \c true.
     */
    RngGenerator (bool anti = false)
      : m_anti (anti)
    {}

    // Inherited
    Ptr<RandomVariableStream>
    Create (void) const
    {
      auto rng = CreateObject<RNG> ();
      rng->SetAttribute ("Antithetic", BooleanValue (m_anti));
      return rng;
    }

  private:
    /** Whether to create antithetic random variable streams. */
    bool m_anti;
  };

  /**
   * Compute the chi squared value of a sampled distribution
   * compared to the expected distribution.
   *
   * This function captures the actual computation of the chi square,
   * given an expected distribution.
   *
   * The random variable is sampled \c N_MEASUREMENTS times, filling
   * a histogram. The chi square value is formed by comparing to the
   * expected distribution.
   * \param [in,out] h The histogram, which defines the binning for sampling.
   * \param [in] expected The expected distribution.
   * \param [in] rng The random variable to sample.
   * \returns The chi square value.
   */
  double
  ChiSquared (gsl_histogram * h,
                const std::vector<double> & expected,
                Ptr<RandomVariableStream> rng) const
  {
    NS_LOG_FUNCTION (this << h << expected.size () << rng);
    NS_ASSERT_MSG (gsl_histogram_bins (h) == expected.size (),
                   "Histogram and expected vector have different sizes.");

    // Sample the rng into the histogram
    for (std::size_t i = 0; i < N_MEASUREMENTS; ++i)
      {
        double value = rng->GetValue ();
        gsl_histogram_increment (h, value);
      }

    // Compute the chi square value
    double chiSquared = 0;
    std::size_t nBins = gsl_histogram_bins (h);
    for (std::size_t i = 0; i < nBins; ++i)
      {
        double hbin = gsl_histogram_get (h, i);
        double tmp = hbin - expected[i];
        tmp *= tmp;
        tmp /= expected[i];
        chiSquared += tmp;
      }

    return chiSquared;
  }

  /**
   * Compute the chi square value from a random variable.
   *
   * This function sets up the binning and expected distribution
   * needed to actually compute the chi squared value, which
   * should be done by a call to ChiSquared.
   *
   * This is the point of customization expected to be implemented
   * in derived classes with the appropriate histogram binning and
   * expected distribution.  For example
   *
   *    SomeRngTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
   *    {
   *      gsl_histogram * h = gsl_histogram_alloc (N_BINS);
   *      auto range = UniformHistogramBins (h, -4., 4.);
   *      std::vector<double> expected (N_BINS);
   *      // Populated expected
   *      for (std::size_t i = 0; i < N_BINS; ++i)
   *        {
   *          expected[i] = ...;
   *          expected[i] *= N_MEASUREMENTS;
   *        }
   *      double chiSquared = ChiSquared (h, expected, rng);
   *      gsl_histogram_free (h);
   *      return chiSquared;
   *    }
   *
   * \param [in] rng The random number generator to test.
   * \returns The chi squared value.
   */
  virtual double
  ChiSquaredTest (Ptr<RandomVariableStream> rng) const
  {
    return 0;
  }

  /**
   * Average the chi squared value over some number of runs,
   * each run with a new instance of the random number generator.
   * \param [in] generator The factory to create instances of the
   *             random number generator.
   * \param [in] nRuns The number of runs to average over.
   * \returns The average chi square over the number of runs.
   */
  double
  ChiSquaredsAverage (const RngGeneratorBase * generator,
                        std::size_t nRuns) const
  {
    NS_LOG_FUNCTION (this << generator << nRuns);
    
    double sum = 0.;
    for (std::size_t i = 0; i < nRuns; ++i)
      {
        auto rng = generator->Create ();
        double result = ChiSquaredTest (rng);
        sum += result;
      }
    sum /= (double)nRuns;
    return sum;
  }

  /**
   * Set the seed used for this test suite.
   *
   * This test suite is designed to be run with both deterministic and
   * random seed and run number values.  Deterministic values can be used
   * for basic regression testing; random values can be used to more
   * exhaustively test the generated streams, with the side effect of
   * occasional test failures.
   * 
   * By default, this test suite will use the default values of RngSeed = 1
   * and RngRun = 1.  Users can configure any other seed and run number
   * in the usual way, but the special value of RngRun = 0 results in
   * selecting a RngSeed value that corresponds to the seconds since epoch
   * (\c time (0) from \c ctime).  Note: this is not a recommended practice for
   * seeding normal simulations, as described in the ns-3 manual, but
   * allows the test to be exposed to a wider range of seeds.
   *
   * In either case, the values produced will be checked with a chi-squared
   * test.
   *
   * For example, this command will cause this test suite to use the
   * deterministic value of seed=3 and default run number=1 every time:
   *   NS_GLOBAL_VALUE="RngSeed=3" ./test.py -s random-variable-stream-generators
   * or equivalently (to see log output):
   *   NS_LOG="RandomVariableStreamGenerators" NS_GLOBAL_VALUE="RngSeed=3" ./ns3 run "test-runner --suite=random-variable-stream-generators"
   *
   * Conversely, this command will cause this test suite to use a seed
   * based on time-of-day, and run number=0: 
   *   NS_GLOBAL_VALUE="RngRun=0" ./test.py -s random-variable-stream-generators
   */
  void
  SetTestSuiteSeed (void)
  {
    if (m_seedSet == false)
      {
        uint32_t seed;
        if (RngSeedManager::GetRun () == 0)
          {
            seed = static_cast<uint32_t> (time (0));
            m_seedSet = true;
            NS_LOG_DEBUG ("Special run number value of zero; seeding with time of day: " << seed);

          }
        else
          {
            seed = RngSeedManager::GetSeed ();
            m_seedSet = true;
            NS_LOG_DEBUG ("Using the values seed: " <<
                          seed << " and run: " << RngSeedManager::GetRun ());
          }
        SeedManager::SetSeed (seed);
      }
  }

private:
  /** \c true if we've already set the seed the correctly. */
  bool m_seedSet = false;

}; // class TestCaseBase


/**
 * \ingroup rng-tests
 * Test case for uniform distribution random variable stream generator.
 */
class UniformTestCase : public TestCaseBase
{
public:

  // Constructor
  UniformTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);
};

UniformTestCase::UniformTestCase ()
  : TestCaseBase ("Uniform Random Variable Stream Generator")
{}

double
UniformTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);

  // Note that this assumes that the range for u is [0,1], which is
  // the default range for this distribution.
  gsl_histogram_set_ranges_uniform (h, 0., 1.);

  std::vector<double> expected (N_BINS, ((double)N_MEASUREMENTS / (double)N_BINS));

  double chiSquared = ChiSquared (h, expected, rng);
  gsl_histogram_free (h);
  return chiSquared;
}

void
UniformTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  double confidence = 0.99;
  double maxStatistic = gsl_cdf_chisq_Pinv (confidence, (N_BINS - 1));
  NS_LOG_DEBUG ("Chi square required at " << confidence << " confidence for " << N_BINS << " bins is " << maxStatistic);

  double result = maxStatistic;
  // If chi-squared test fails, re-try it up to N_RUNS times
  for (uint32_t i = 0; i < N_RUNS; ++i)
    {
      Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable> ();
      result = ChiSquaredTest (rng);
      NS_LOG_DEBUG ("Chi square result is " << result);
      if (result < maxStatistic)
        {
          break;
        }
    }

  NS_TEST_ASSERT_MSG_LT (result, maxStatistic, "Chi-squared statistic out of range");

  double min = 0.0;
  double max = 10.0;
  double value;

  // Create the RNG with the specified range.
  Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();

  x->SetAttribute ("Min", DoubleValue (min));
  x->SetAttribute ("Max", DoubleValue (max));

  // Test that values are always within the range:
  //
  //     [min, max)
  //
  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      value = x->GetValue ();
      NS_TEST_ASSERT_MSG_EQ ((value >= min), true, "Value less than minimum.");
      NS_TEST_ASSERT_MSG_LT (value, max, "Value greater than or equal to maximum.");
    }

  // Boundary checking on GetInteger; should be [min,max]; from bug 1964
  static const uint32_t UNIFORM_INTEGER_MIN {0};
  static const uint32_t UNIFORM_INTEGER_MAX {4294967295U};
  // [0,0] should return 0
  uint32_t intValue;
  intValue = x->GetInteger (UNIFORM_INTEGER_MIN, UNIFORM_INTEGER_MIN);
  NS_TEST_ASSERT_MSG_EQ (intValue, UNIFORM_INTEGER_MIN, "Uniform RV GetInteger boundary testing");
  // [UNIFORM_INTEGER_MAX, UNIFORM_INTEGER_MAX] should return UNIFORM_INTEGER_MAX
  intValue = x->GetInteger (UNIFORM_INTEGER_MAX, UNIFORM_INTEGER_MAX);
  NS_TEST_ASSERT_MSG_EQ (intValue, UNIFORM_INTEGER_MAX, "Uniform RV GetInteger boundary testing");
  // [0,1] should return mix of 0 or 1
  intValue = 0;
  for (int i = 0; i < 20; i++)
    {
      intValue += x->GetInteger (UNIFORM_INTEGER_MIN, UNIFORM_INTEGER_MIN + 1);
    }
  NS_TEST_ASSERT_MSG_GT (intValue, 0, "Uniform RV GetInteger boundary testing");
  NS_TEST_ASSERT_MSG_LT (intValue, 20, "Uniform RV GetInteger boundary testing");
  // [MAX-1,MAX] should return mix of MAX-1 or MAX
  uint32_t count = 0;
  for (int i = 0; i < 20; i++)
    {
      intValue = x->GetInteger (UNIFORM_INTEGER_MAX - 1, UNIFORM_INTEGER_MAX);
      if (intValue == UNIFORM_INTEGER_MAX)
        {
          count++;
        }
    }
  NS_TEST_ASSERT_MSG_GT (count, 0, "Uniform RV GetInteger boundary testing");
  NS_TEST_ASSERT_MSG_LT (count, 20, "Uniform RV GetInteger boundary testing");
  // multiple [0,UNIFORM_INTEGER_MAX] should return non-zero
  intValue = x->GetInteger (UNIFORM_INTEGER_MIN, UNIFORM_INTEGER_MAX);
  uint32_t intValue2 = x->GetInteger (UNIFORM_INTEGER_MIN, UNIFORM_INTEGER_MAX);
  NS_TEST_ASSERT_MSG_GT (intValue + intValue2, 0, "Uniform RV GetInteger boundary testing");

}

/**
 * \ingroup rng-tests
 * Test case for antithetic uniform distribution random variable stream generator
 */
class UniformAntitheticTestCase : public TestCaseBase
{
public:

  // Constructor
  UniformAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);
};

UniformAntitheticTestCase::UniformAntitheticTestCase ()
  : TestCaseBase ("Antithetic Uniform Random Variable Stream Generator")
{}

double
UniformAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);

  // Note that this assumes that the range for u is [0,1], which is
  // the default range for this distribution.
  gsl_histogram_set_ranges_uniform (h, 0., 1.);

  std::vector<double> expected (N_BINS, ((double)N_MEASUREMENTS / (double)N_BINS));

  double chiSquared = ChiSquared (h, expected, rng);
  gsl_histogram_free (h);
  return chiSquared;
}

void
UniformAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<UniformRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double min = 0.0;
  double max = 10.0;
  double value;

  // Create the RNG with the specified range.
  Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  x->SetAttribute ("Min", DoubleValue (min));
  x->SetAttribute ("Max", DoubleValue (max));

  // Test that values are always within the range:
  //
  //     [min, max)
  //
  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      value = x->GetValue ();
      NS_TEST_ASSERT_MSG_EQ ((value >= min), true, "Value less than minimum.");
      NS_TEST_ASSERT_MSG_LT (value, max, "Value greater than or equal to maximum.");
    }


}
/**
 * \ingroup rng-tests
 * Test case for constant random variable stream generator
 */
class ConstantTestCase : public TestCaseBase
{
public:
  // Constructor
  ConstantTestCase ();

private:
  // Inherited
  virtual void DoRun (void);

  /** Tolerance for testing rng values against expectation. */
  static constexpr double TOLERANCE {1e-8};
};

ConstantTestCase::ConstantTestCase ()
  : TestCaseBase ("Constant Random Variable Stream Generator")
{}

void
ConstantTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  Ptr<ConstantRandomVariable> c = CreateObject<ConstantRandomVariable> ();

  double constant;

  // Test that the constant value can be changed using its attribute.
  constant = 10.0;
  c->SetAttribute ("Constant", DoubleValue (constant));
  NS_TEST_ASSERT_MSG_EQ_TOL (c->GetValue (), constant, TOLERANCE, "Constant value changed");
  c->SetAttribute ("Constant", DoubleValue (20.0));
  NS_TEST_ASSERT_MSG_NE (c->GetValue (), constant, "Constant value not changed");

  // Test that the constant value does not change.
  constant = c->GetValue ();
  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ_TOL (c->GetValue (), constant, TOLERANCE, "Constant value changed in loop");
    }
}

/**
 * \ingroup rng-tests
 * Test case for sequential random variable stream generator
 */
class SequentialTestCase : public TestCaseBase
{
public:
  // Constructor
  SequentialTestCase ();

private:
  // Inherited
  virtual void DoRun (void);
  
  /** Tolerance for testing rng values against expectation. */
  static constexpr double TOLERANCE {1e-8};
};

SequentialTestCase::SequentialTestCase ()
  : TestCaseBase ("Sequential Random Variable Stream Generator")
{}

void
SequentialTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  Ptr<SequentialRandomVariable> s = CreateObject<SequentialRandomVariable> ();

  // The following four attributes should give the sequence
  //
  //    4, 4, 7, 7, 10, 10
  //
  s->SetAttribute ("Min", DoubleValue (4));
  s->SetAttribute ("Max", DoubleValue (11));
  s->SetAttribute ("Increment", StringValue ("ns3::UniformRandomVariable[Min=3.0|Max=3.0]"));
  s->SetAttribute ("Consecutive", IntegerValue (2));

  double value;

  // Test that the sequencet is correct.
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4, TOLERANCE, "Sequence value 1 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4, TOLERANCE, "Sequence value 2 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 7, TOLERANCE, "Sequence value 3 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 7, TOLERANCE, "Sequence value 4 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 10, TOLERANCE, "Sequence value 5 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 10, TOLERANCE, "Sequence value 6 wrong.");

}

/**
 * \ingroup rng-tests
 * Test case for normal distribution random variable stream generator
 */
class NormalTestCase : public TestCaseBase
{
public:
  // Constructor
  NormalTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /** Tolerance for testing rng values against expectation, in rms. */
  static constexpr double TOLERANCE {5};  
};

NormalTestCase::NormalTestCase ()
  : TestCaseBase ("Normal Random Variable Stream Generator")
{}

double
NormalTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, -4., 4.);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has mean equal to zero and standard
  // deviation equal to one, which are their default values for this
  // distribution.
  double sigma = 1.;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gaussian_P (range[i + 1], sigma) - gsl_cdf_gaussian_P (range[i], sigma);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);
  gsl_histogram_free (h);
  return chiSquared;
}

void
NormalTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<NormalRandomVariable> ();
  auto rng = generator.Create ();
  
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mean = 5.0;
  double variance = 2.0;

  // Create the RNG with the specified range.
  Ptr<NormalRandomVariable> x = CreateObject<NormalRandomVariable> ();
  x->SetAttribute ("Mean", DoubleValue (mean));
  x->SetAttribute ("Variance", DoubleValue (variance));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // normally distributed random variable is equal to mean.
  double expectedMean = mean;
  double expectedRms  = mean / std::sqrt (variance * N_MEASUREMENTS);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedRms * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic normal distribution random variable stream generator
 */
class NormalAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  NormalAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /** Tolerance for testing rng values against expectation, in rms. */
  static constexpr double TOLERANCE {5};  
};

NormalAntitheticTestCase::NormalAntitheticTestCase ()
  : TestCaseBase ("Antithetic Normal Random Variable Stream Generator")
{}

double
NormalAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, -4, 4);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has mean equal to zero and standard
  // deviation equal to one, which are their default values for this
  // distribution.
  double sigma = 1.;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gaussian_P (range[i + 1], sigma) - gsl_cdf_gaussian_P (range[i], sigma);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
NormalAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<NormalRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mean = 5.0;
  double variance = 2.0;

  // Create the RNG with the specified range.
  Ptr<NormalRandomVariable> x = CreateObject<NormalRandomVariable> ();
  x->SetAttribute ("Mean", DoubleValue (mean));
  x->SetAttribute ("Variance", DoubleValue (variance));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // normally distributed random variable is equal to mean.
  double expectedMean = mean;
  double expectedRms  = mean / std::sqrt (variance * N_MEASUREMENTS);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedRms * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for exponential distribution random variable stream generator
 */
class ExponentialTestCase : public TestCaseBase
{
public:
  // Constructor
  ExponentialTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /** Tolerance for testing rng values against expectation, in rms. */
  static constexpr double TOLERANCE {5};  
};

ExponentialTestCase::ExponentialTestCase ()
  : TestCaseBase ("Exponential Random Variable Stream Generator")
{}

double
ExponentialTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that e has mean equal to one, which is the
  // default value for this distribution.
  double mu = 1.;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_exponential_P (range[i + 1], mu) - gsl_cdf_exponential_P (range[i], mu);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ExponentialTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ExponentialRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mean = 3.14;
  double bound = 0.0;

  // Create the RNG with the specified range.
  Ptr<ExponentialRandomVariable> x = CreateObject<ExponentialRandomVariable> ();
  x->SetAttribute ("Mean", DoubleValue (mean));
  x->SetAttribute ("Bound", DoubleValue (bound));

  // Calculate the mean of these values.
  double valueMean = Average (x);
  double expectedMean = mean;
  double expectedRms  = std::sqrt (mean / N_MEASUREMENTS);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedRms * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic exponential distribution random variable stream generator
 */
class ExponentialAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  ExponentialAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /** Tolerance for testing rng values against expectation, in rms. */
  static constexpr double TOLERANCE {5};
};

ExponentialAntitheticTestCase::ExponentialAntitheticTestCase ()
  : TestCaseBase ("Antithetic Exponential Random Variable Stream Generator")
{}

double
ExponentialAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that e has mean equal to one, which is the
  // default value for this distribution.
  double mu = 1.;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_exponential_P (range[i + 1], mu) - gsl_cdf_exponential_P (range[i], mu);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ExponentialAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ExponentialRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mean = 3.14;
  double bound = 0.0;

  // Create the RNG with the specified range.
  Ptr<ExponentialRandomVariable> x = CreateObject<ExponentialRandomVariable> ();
  x->SetAttribute ("Mean", DoubleValue (mean));
  x->SetAttribute ("Bound", DoubleValue (bound));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);
  double expectedMean = mean;
  double expectedRms  = std::sqrt (mean / N_MEASUREMENTS);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedRms * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for Pareto distribution random variable stream generator
 */
class ParetoTestCase : public TestCaseBase
{
public:
  // Constructor
  ParetoTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ParetoTestCase::ParetoTestCase ()
  : TestCaseBase ("Pareto Random Variable Stream Generator")
{}

double
ParetoTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 1, 10, false);

  std::vector<double> expected (N_BINS);

  double shape = 2.0;
  double scale = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_pareto_P (range[i + 1], shape, scale) - gsl_cdf_pareto_P (range[i], shape, scale);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ParetoTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ParetoRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double shape = 2.0;
  double scale = 1.0;

  // Create the RNG with the specified range.
  Ptr<ParetoRandomVariable> x = CreateObject<ParetoRandomVariable> ();
  x->SetAttribute ("Shape", DoubleValue (shape));
  x->SetAttribute ("Scale", DoubleValue (scale));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean is given by
  //
  //                   shape * scale
  //     E[value]  =  ---------------  ,
  //                     shape - 1
  //
  // where
  //
  //     scale  =  mean * (shape - 1.0) / shape .
  double expectedMean = (shape * scale) / (shape - 1.0);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic Pareto distribution random variable stream generator
 */
class ParetoAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  ParetoAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ParetoAntitheticTestCase::ParetoAntitheticTestCase ()
  : TestCaseBase ("Antithetic Pareto Random Variable Stream Generator")
{}

double
ParetoAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 1, 10, false);

  std::vector<double> expected (N_BINS);

  double shape = 2.0;
  double scale = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_pareto_P (range[i + 1], shape, scale) - gsl_cdf_pareto_P (range[i], shape, scale);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ParetoAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ParetoRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double shape = 2.0;
  double scale = 1.0;

  // Create the RNG with the specified range.
  Ptr<ParetoRandomVariable> x = CreateObject<ParetoRandomVariable> ();
  x->SetAttribute ("Shape", DoubleValue (shape));
  x->SetAttribute ("Scale", DoubleValue (scale));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean is given by
  //
  //                   shape * scale
  //     E[value]  =  ---------------  ,
  //                     shape - 1
  //
  // where
  //
  //     scale  =  mean * (shape - 1.0) / shape .
  //
  double expectedMean = (shape * scale) / (shape - 1.0);

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for Weibull distribution random variable stream generator
 */
class WeibullTestCase : public TestCaseBase
{
public:
  // Constructor
  WeibullTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

WeibullTestCase::WeibullTestCase ()
  : TestCaseBase ("Weibull Random Variable Stream Generator")
{}

double
WeibullTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 1, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that p has shape equal to one and scale
  // equal to one, which are their default values for this
  // distribution.
  double a = 1.0;
  double b = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_weibull_P (range[i + 1], a, b) - gsl_cdf_weibull_P (range[i], a, b);
      expected[i] *= N_MEASUREMENTS;
      NS_LOG_INFO ("weibull: " << expected[i]);
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
WeibullTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<WeibullRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double scale = 5.0;
  double shape = 1.0;

  // Create the RNG with the specified range.
  Ptr<WeibullRandomVariable> x = CreateObject<WeibullRandomVariable> ();
  x->SetAttribute ("Scale", DoubleValue (scale));
  x->SetAttribute ("Shape", DoubleValue (shape));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Weibull distributed random variable is
  //
  //     E[value]  =  scale * Gamma(1 + 1 / shape)  ,
  //
  // where Gamma() is the Gamma function.  Note that
  //
  //     Gamma(n)  =  (n - 1)!
  //
  // if n is a positive integer.
  //
  // For this test,
  //
  //     Gamma(1 + 1 / shape)  =  Gamma(1 + 1 / 1)
  //                           =  Gamma(2)
  //                           =  (2 - 1)!
  //                           =  1
  //
  // which means
  //
  //     E[value]  =  scale  .
  //
  double expectedMean = scale;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic Weibull distribution random variable stream generator
 */
class WeibullAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  WeibullAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

WeibullAntitheticTestCase::WeibullAntitheticTestCase ()
  : TestCaseBase ("Antithetic Weibull Random Variable Stream Generator")
{}

double
WeibullAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 1, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that p has shape equal to one and scale
  // equal to one, which are their default values for this
  // distribution.
  double a = 1.0;
  double b = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_weibull_P (range[i + 1], a, b) - gsl_cdf_weibull_P (range[i], a, b);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
WeibullAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<WeibullRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double scale = 5.0;
  double shape = 1.0;

  // Create the RNG with the specified range.
  Ptr<WeibullRandomVariable> x = CreateObject<WeibullRandomVariable> ();
  x->SetAttribute ("Scale", DoubleValue (scale));
  x->SetAttribute ("Shape", DoubleValue (shape));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Weibull distributed random variable is
  //
  //     E[value]  =  scale * Gamma(1 + 1 / shape)  ,
  //
  // where Gamma() is the Gamma function.  Note that
  //
  //     Gamma(n)  =  (n - 1)!
  //
  // if n is a positive integer.
  //
  // For this test,
  //
  //     Gamma(1 + 1 / shape)  =  Gamma(1 + 1 / 1)
  //                           =  Gamma(2)
  //                           =  (2 - 1)!
  //                           =  1
  //
  // which means
  //
  //     E[value]  =  scale  .
  //
  double expectedMean = scale;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for log-normal distribution random variable stream generator
 */
class LogNormalTestCase : public TestCaseBase
{
public:
  // Constructor
  LogNormalTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {3e-2};  
};

LogNormalTestCase::LogNormalTestCase ()
  : TestCaseBase ("Log-Normal Random Variable Stream Generator")
{}

double
LogNormalTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has mu equal to zero and sigma
  // equal to one, which are their default values for this
  // distribution.
  double mu = 0.0;
  double sigma = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_lognormal_P (range[i + 1], mu, sigma) - gsl_cdf_lognormal_P (range[i], mu, sigma);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
LogNormalTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<LogNormalRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);

  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mu = 5.0;
  double sigma = 2.0;

  // Create the RNG with the specified range.
  Ptr<LogNormalRandomVariable> x = CreateObject<LogNormalRandomVariable> ();
  x->SetAttribute ("Mu", DoubleValue (mu));
  x->SetAttribute ("Sigma", DoubleValue (sigma));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // log-normally distributed random variable is equal to
  //
  //                             2
  //                   mu + sigma  / 2
  //     E[value]  =  e                 .
  //
  double expectedMean = std::exp (mu + sigma * sigma / 2.0);

  // Test that values have approximately the right mean value.
  //
  /**
   * \todo This test fails sometimes if the required tolerance is less
   * than 3%, which may be because there is a bug in the
   * implementation or that the mean of this distribution is more
   * sensitive to its parameters than the others are.
   */
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic log-normal distribution random variable stream generator
 */
class LogNormalAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  LogNormalAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {3e-2};  
};

LogNormalAntitheticTestCase::LogNormalAntitheticTestCase ()
  : TestCaseBase ("Antithetic Log-Normal Random Variable Stream Generator")
{}

double
LogNormalAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has mu equal to zero and sigma
  // equal to one, which are their default values for this
  // distribution.
  double mu = 0.0;
  double sigma = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_lognormal_P (range[i + 1], mu, sigma) - gsl_cdf_lognormal_P (range[i], mu, sigma);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
LogNormalAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<LogNormalRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double mu = 5.0;
  double sigma = 2.0;

  // Create the RNG with the specified range.
  Ptr<LogNormalRandomVariable> x = CreateObject<LogNormalRandomVariable> ();
  x->SetAttribute ("Mu", DoubleValue (mu));
  x->SetAttribute ("Sigma", DoubleValue (sigma));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // log-normally distributed random variable is equal to
  //
  //                             2
  //                   mu + sigma  / 2
  //     E[value]  =  e                 .
  //
  double expectedMean = std::exp (mu + sigma * sigma / 2.0);

  // Test that values have approximately the right mean value.
  //
  /**
   * \todo This test fails sometimes if the required tolerance is less
   * than 3%, which may be because there is a bug in the
   * implementation or that the mean of this distribution is more
   * sensitive to its parameters than the others are.
   */
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for gamma distribution random variable stream generator
 */
class GammaTestCase : public TestCaseBase
{
public:
  // Constructor
  GammaTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

GammaTestCase::GammaTestCase ()
  : TestCaseBase ("Gamma Random Variable Stream Generator")
{}

double
GammaTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has alpha equal to one and beta
  // equal to one, which are their default values for this
  // distribution.
  double alpha = 1.0;
  double beta = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gamma_P (range[i + 1], alpha, beta) - gsl_cdf_gamma_P (range[i], alpha, beta);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
GammaTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<GammaRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double alpha = 5.0;
  double beta = 2.0;

  // Create the RNG with the specified range.
  Ptr<GammaRandomVariable> x = CreateObject<GammaRandomVariable> ();
  x->SetAttribute ("Alpha", DoubleValue (alpha));
  x->SetAttribute ("Beta", DoubleValue (beta));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // gammaly distributed random variable is equal to
  //
  //     E[value]  =  alpha * beta  .
  //
  double expectedMean = alpha * beta;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic gamma distribution random variable stream generator
 */
class GammaAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  GammaAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

GammaAntitheticTestCase::GammaAntitheticTestCase ()
  : TestCaseBase ("Antithetic Gamma Random Variable Stream Generator")
{}

double
GammaAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has alpha equal to one and beta
  // equal to one, which are their default values for this
  // distribution.
  double alpha = 1.0;
  double beta = 1.0;

  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gamma_P (range[i + 1], alpha, beta) - gsl_cdf_gamma_P (range[i], alpha, beta);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
GammaAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<GammaRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  double alpha = 5.0;
  double beta = 2.0;

  // Create the RNG with the specified range.
  Ptr<GammaRandomVariable> x = CreateObject<GammaRandomVariable> ();

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  x->SetAttribute ("Alpha", DoubleValue (alpha));
  x->SetAttribute ("Beta", DoubleValue (beta));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // gammaly distributed random variable is equal to
  //
  //     E[value]  =  alpha * beta  .
  //
  double expectedMean = alpha * beta;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for Erlang distribution random variable stream generator
 */
class ErlangTestCase : public TestCaseBase
{
public:
  // Constructor
  ErlangTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ErlangTestCase::ErlangTestCase ()
  : TestCaseBase ("Erlang Random Variable Stream Generator")
{}

double
ErlangTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has k equal to one and lambda
  // equal to one, which are their default values for this
  // distribution.
  uint32_t k = 1;
  double lambda = 1.0;

  // Note that Erlang distribution is equal to the gamma distribution
  // when k is an iteger, which is why the gamma distribution's cdf
  // function can be used here.
  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gamma_P (range[i + 1], k, lambda) - gsl_cdf_gamma_P (range[i], k, lambda);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ErlangTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ErlangRandomVariable> ();
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  uint32_t k = 5;
  double lambda = 2.0;

  // Create the RNG with the specified range.
  Ptr<ErlangRandomVariable> x = CreateObject<ErlangRandomVariable> ();
  x->SetAttribute ("K", IntegerValue (k));
  x->SetAttribute ("Lambda", DoubleValue (lambda));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Erlangly distributed random variable is equal to
  //
  //     E[value]  =  k * lambda  .
  //
  double expectedMean = k * lambda;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic Erlang distribution random variable stream generator
 */
class ErlangAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  ErlangAntitheticTestCase ();

  // Inherited
  double ChiSquaredTest (Ptr<RandomVariableStream> rng) const;

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ErlangAntitheticTestCase::ErlangAntitheticTestCase ()
  : TestCaseBase ("Antithetic Erlang Random Variable Stream Generator")
{}

double
ErlangAntitheticTestCase::ChiSquaredTest (Ptr<RandomVariableStream> rng) const
{
  gsl_histogram * h = gsl_histogram_alloc (N_BINS);
  auto range = UniformHistogramBins (h, 0, 10, false);

  std::vector<double> expected (N_BINS);

  // Note that this assumes that n has k equal to one and lambda
  // equal to one, which are their default values for this
  // distribution.
  uint32_t k = 1;
  double lambda = 1.0;

  // Note that Erlang distribution is equal to the gamma distribution
  // when k is an iteger, which is why the gamma distribution's cdf
  // function can be used here.
  for (std::size_t i = 0; i < N_BINS; ++i)
    {
      expected[i] = gsl_cdf_gamma_P (range[i + 1], k, lambda) - gsl_cdf_gamma_P (range[i], k, lambda);
      expected[i] *= N_MEASUREMENTS;
    }

  double chiSquared = ChiSquared (h, expected, rng);

  gsl_histogram_free (h);
  return chiSquared;
}

void
ErlangAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  auto generator = RngGenerator<ErlangRandomVariable> (true);
  double sum = ChiSquaredsAverage (&generator, N_RUNS);
  double maxStatistic = gsl_cdf_chisq_Qinv (0.05, N_BINS);
  NS_TEST_ASSERT_MSG_LT (sum, maxStatistic, "Chi-squared statistic out of range");

  uint32_t k = 5;
  double lambda = 2.0;

  // Create the RNG with the specified range.
  Ptr<ErlangRandomVariable> x = CreateObject<ErlangRandomVariable> ();

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  x->SetAttribute ("K", IntegerValue (k));
  x->SetAttribute ("Lambda", DoubleValue (lambda));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Erlangly distributed random variable is equal to
  //
  //     E[value]  =  k * lambda  .
  //
  double expectedMean = k * lambda;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for Zipf distribution random variable stream generator
 */
class ZipfTestCase : public TestCaseBase
{
public:
  // Constructor
  ZipfTestCase ();

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ZipfTestCase::ZipfTestCase ()
  : TestCaseBase ("Zipf Random Variable Stream Generator")
{}

void
ZipfTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  uint32_t n = 1;
  double alpha = 2.0;

  // Create the RNG with the specified range.
  Ptr<ZipfRandomVariable> x = CreateObject<ZipfRandomVariable> ();
  x->SetAttribute ("N", IntegerValue (n));
  x->SetAttribute ("Alpha", DoubleValue (alpha));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Zipfly distributed random variable is equal to
  //
  //                   H
  //                    N, alpha - 1
  //     E[value]  =  ---------------
  //                     H
  //                      N, alpha
  //
  // where
  //
  //                    N
  //                   ---
  //                   \     -alpha
  //     H          =  /    m        .
  //      N, alpha     ---
  //                   m=1
  //
  // For this test,
  //
  //                      -(alpha - 1)
  //                     1
  //     E[value]  =  ---------------
  //                      -alpha
  //                     1
  //
  //               =  1  .
  //
  double expectedMean = 1.0;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic Zipf distribution random variable stream generator
 */
class ZipfAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  ZipfAntitheticTestCase ();

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ZipfAntitheticTestCase::ZipfAntitheticTestCase ()
  : TestCaseBase ("Antithetic Zipf Random Variable Stream Generator")
{}

void
ZipfAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  uint32_t n = 1;
  double alpha = 2.0;

  // Create the RNG with the specified range.
  Ptr<ZipfRandomVariable> x = CreateObject<ZipfRandomVariable> ();
  x->SetAttribute ("N", IntegerValue (n));
  x->SetAttribute ("Alpha", DoubleValue (alpha));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // Zipfly distributed random variable is equal to
  //
  //                   H
  //                    N, alpha - 1
  //     E[value]  =  ---------------
  //                     H
  //                      N, alpha
  //
  // where
  //
  //                    N
  //                   ---
  //                   \     -alpha
  //     H          =  /    m        .
  //      N, alpha     ---
  //                   m=1
  //
  // For this test,
  //
  //                      -(alpha - 1)
  //                     1
  //     E[value]  =  ---------------
  //                      -alpha
  //                     1
  //
  //               =  1  .
  //
  double expectedMean = 1.0;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for Zeta distribution random variable stream generator
 */
class ZetaTestCase : public TestCaseBase
{
public:
  // Constructor
  ZetaTestCase ();

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ZetaTestCase::ZetaTestCase ()
  : TestCaseBase ("Zeta Random Variable Stream Generator")
{}

void
ZetaTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  double alpha = 5.0;

  // Create the RNG with the specified range.
  Ptr<ZetaRandomVariable> x = CreateObject<ZetaRandomVariable> ();
  x->SetAttribute ("Alpha", DoubleValue (alpha));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // zetaly distributed random variable is equal to
  //
  //                   zeta(alpha - 1)
  //     E[value]  =  ---------------   for alpha > 2 ,
  //                     zeta(alpha)
  //
  // where zeta(alpha) is the Riemann zeta function.
  //
  // There are no simple analytic forms for the Riemann zeta function,
  // which is why the gsl library is used in this test to calculate
  // the known mean of the values.
  double expectedMean =
    gsl_sf_zeta_int (static_cast<int> (alpha - 1)) /
    gsl_sf_zeta_int (static_cast<int> (alpha)       );

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic Zeta distribution random variable stream generator
 */
class ZetaAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  ZetaAntitheticTestCase ();

private:
  // Inherited
  virtual void DoRun (void);

  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};  
};

ZetaAntitheticTestCase::ZetaAntitheticTestCase ()
  : TestCaseBase ("Antithetic Zeta Random Variable Stream Generator")
{}

void
ZetaAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  double alpha = 5.0;

  // Create the RNG with the specified range.
  Ptr<ZetaRandomVariable> x = CreateObject<ZetaRandomVariable> ();
  x->SetAttribute ("Alpha", DoubleValue (alpha));

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Calculate the mean of these values.
  double valueMean = Average (x);

  // The expected value for the mean of the values returned by a
  // zetaly distributed random variable is equal to
  //
  //                   zeta(alpha - 1)
  //     E[value]  =  ---------------   for alpha > 2 ,
  //                     zeta(alpha)
  //
  // where zeta(alpha) is the Riemann zeta function.
  //
  // There are no simple analytic forms for the Riemann zeta function,
  // which is why the gsl library is used in this test to calculate
  // the known mean of the values.
  double expectedMean =
    gsl_sf_zeta_int (static_cast<int> (alpha) - 1) /
    gsl_sf_zeta_int (static_cast<int> (alpha)       );

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for deterministic random variable stream generator
 */
class DeterministicTestCase : public TestCaseBase
{
public:
  // Constructor
  DeterministicTestCase ();

private:
  // Inherited
  virtual void DoRun (void);
  
  /** Tolerance for testing rng values against expectation. */
  static constexpr double TOLERANCE {1e-8};
};

DeterministicTestCase::DeterministicTestCase ()
  : TestCaseBase ("Deterministic Random Variable Stream Generator")
{}

void
DeterministicTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  Ptr<DeterministicRandomVariable> s = CreateObject<DeterministicRandomVariable> ();

  // The following array should give the sequence
  //
  //    4, 4, 7, 7, 10, 10 .
  //
  double array1 [] = { 4, 4, 7, 7, 10, 10};
  std::size_t count1 = 6;
  s->SetValueArray (array1, count1);

  double value;

  // Test that the first sequence is correct.
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4, TOLERANCE, "Sequence 1 value 1 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4, TOLERANCE, "Sequence 1 value 2 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 7, TOLERANCE, "Sequence 1 value 3 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 7, TOLERANCE, "Sequence 1 value 4 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 10, TOLERANCE, "Sequence 1 value 5 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 10, TOLERANCE, "Sequence 1 value 6 wrong.");

  // The following array should give the sequence
  //
  //    1000, 2000, 7, 7 .
  //
  double array2 [] = { 1000, 2000, 3000, 4000};
  std::size_t count2 = 4;
  s->SetValueArray (array2, count2);

  // Test that the second sequence is correct.
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 1000, TOLERANCE, "Sequence 2 value 1 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 2000, TOLERANCE, "Sequence 2 value 2 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 3000, TOLERANCE, "Sequence 2 value 3 wrong.");
  value = s->GetValue ();
  NS_TEST_ASSERT_MSG_EQ_TOL (value, 4000, TOLERANCE, "Sequence 2 value 4 wrong.");
  value = s->GetValue ();
}

/**
 * \ingroup rng-tests
 * Test case for empirical distribution random variable stream generator
 */
class EmpiricalTestCase : public TestCaseBase
{
public:
  // Constructor
  EmpiricalTestCase ();

private:
  // Inherited
  virtual void DoRun (void);
  
  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};
};

EmpiricalTestCase::EmpiricalTestCase ()
  : TestCaseBase ("Empirical Random Variable Stream Generator")
{}

void
EmpiricalTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  // Create the RNG with a uniform distribution between 0 and 10.
  Ptr<EmpiricalRandomVariable> x = CreateObject<EmpiricalRandomVariable> ();
  x->SetInterpolate (false);
  x->CDF ( 0.0,  0.0);
  x->CDF ( 5.0,  0.25);
  x->CDF (10.0,  1.0);

  // Check that only the correct values are returned
  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      double value = x->GetValue ();
      NS_TEST_EXPECT_MSG_EQ ( (value == 5) || (value == 10), true,
                              "Incorrect value returned, expected only 5 or 10.");
    }

  // Calculate the mean of the sampled values.
  double valueMean = Average (x);

  // The expected distribution with sampled values is
  //     Value     Probability
  //      5        25%
  //     10        75%
  //
  // The expected mean is
  //
  //     E[value]  =  5 * 25%  +  10 * 75%  =  8.75
  //
  // Test that values have approximately the right mean value.
  double expectedMean = 8.75;
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
 

  // Calculate the mean of the interpolated values.
  x->SetInterpolate (true);
  valueMean = Average (x);
 
  // The expected distribution (with interpolation) is
  //     Bin     Probability
  //     [0, 5)     25%
  //     [5, 10)    75%
  //
  // Each bin is uniformly sampled, so the average of the samples in the
  // bin is the center of the bin.
  //
  // The expected mean is
  //
  //     E[value]  =  2.5 * 25% + 7.5 * 75% = 6.25
  //
  expectedMean = 6.25;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
 
  // Bug 2082: Create the RNG with a uniform distribution between -1 and 1.
  Ptr<EmpiricalRandomVariable> y = CreateObject<EmpiricalRandomVariable> ();
  y->SetInterpolate (false);
  y->CDF (-1.0,  0.0);
  y->CDF (0.0,  0.5);
  y->CDF (1.0,  1.0);
  NS_TEST_ASSERT_MSG_LT (y->GetValue (), 2, "Empirical variable with negative domain");
}

/**
 * \ingroup rng-tests
 * Test case for antithetic empirical distribution random variable stream generator
 */
class EmpiricalAntitheticTestCase : public TestCaseBase
{
public:
  // Constructor
  EmpiricalAntitheticTestCase ();

private:
  // Inherited
  virtual void DoRun (void);
  
  /**
   * Tolerance for testing rng values against expectation,
   * as a fraction of mean value.
   */
  static constexpr double TOLERANCE {1e-2};
};

EmpiricalAntitheticTestCase::EmpiricalAntitheticTestCase ()
  : TestCaseBase ("EmpiricalAntithetic Random Variable Stream Generator")
{}

void
EmpiricalAntitheticTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  // Create the RNG with a uniform distribution between 0 and 10.
  Ptr<EmpiricalRandomVariable> x = CreateObject<EmpiricalRandomVariable> ();
  x->SetInterpolate (false);
  x->CDF ( 0.0,  0.0);
  x->CDF ( 5.0,  0.25);
  x->CDF (10.0,  1.0);

  // Make this generate antithetic values.
  x->SetAttribute ("Antithetic", BooleanValue (true));

  // Check that only the correct values are returned
  for (uint32_t i = 0; i < N_MEASUREMENTS; ++i)
    {
      double value = x->GetValue ();
      NS_TEST_EXPECT_MSG_EQ ( (value == 5) || (value == 10), true,
                              "Incorrect value returned, expected only 5 or 10.");
    }

  // Calculate the mean of these values.
  double valueMean = Average (x);
  // Expected
  //    E[value] = 5 * 25%  + 10 * 75%  = 8.75
  double expectedMean = 8.75;
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");

  // Check interpolated sampling
  x->SetInterpolate (true);
  valueMean = Average (x);
  
  // The expected value for the mean of the values returned by this
  // empirical distribution with interpolation is
  //
  //     E[value]  =  2.5 * 25% + 7.5 * 75% = 6.25
  //
  expectedMean = 6.25;

  // Test that values have approximately the right mean value.
  NS_TEST_ASSERT_MSG_EQ_TOL (valueMean, expectedMean, expectedMean * TOLERANCE, "Wrong mean value.");
}

/**
 * \ingroup rng-tests
 * Test case for caching of Normal RV parameters (see issue #302)
 */
class NormalCachingTestCase : public TestCaseBase
{
public:
  // Constructor
  NormalCachingTestCase ();

private:
  // Inherited
  virtual void DoRun (void);
};

NormalCachingTestCase::NormalCachingTestCase ()
  : TestCaseBase ("NormalRandomVariable caching of parameters")
{}

void
NormalCachingTestCase::DoRun (void)
{
  NS_LOG_FUNCTION (this);
  SetTestSuiteSeed ();

  Ptr<NormalRandomVariable> n = CreateObject<NormalRandomVariable> ();
  double v1 = n->GetValue (-10, 1, 10);  // Mean -10, variance 1, bounded to [-20,0]
  double v2 = n->GetValue (10, 1, 10);   // Mean 10, variance 1, bounded to [0,20]

  NS_TEST_ASSERT_MSG_LT (v1, 0, "Incorrect value returned, expected < 0");
  NS_TEST_ASSERT_MSG_GT (v2, 0, "Incorrect value returned, expected > 0");
}

/**
 * \ingroup rng-tests
 * RandomVariableStream test suite, covering all random number variable
 * stream generator types.
 */ 
class RandomVariableSuite : public TestSuite
{
public:
  // Constructor
  RandomVariableSuite ();
};

RandomVariableSuite::RandomVariableSuite ()
  : TestSuite ("random-variable-stream-generators", UNIT)
{
  AddTestCase (new UniformTestCase);
  AddTestCase (new UniformAntitheticTestCase);
  AddTestCase (new ConstantTestCase);
  AddTestCase (new SequentialTestCase);
  AddTestCase (new NormalTestCase);
  AddTestCase (new NormalAntitheticTestCase);
  AddTestCase (new ExponentialTestCase);
  AddTestCase (new ExponentialAntitheticTestCase);
  AddTestCase (new ParetoTestCase);
  AddTestCase (new ParetoAntitheticTestCase);
  AddTestCase (new WeibullTestCase);
  AddTestCase (new WeibullAntitheticTestCase);
  AddTestCase (new LogNormalTestCase);
  /// \todo This test is currently disabled because it fails sometimes.
  /// A possible reason for the failure is that the antithetic code is
  /// not implemented properly for this log-normal case.
  /*
  AddTestCase (new LogNormalAntitheticTestCase);
  */
  AddTestCase (new GammaTestCase);
  /// \todo This test is currently disabled because it fails sometimes.
  /// A possible reason for the failure is that the antithetic code is
  /// not implemented properly for this gamma case.
  /*
  AddTestCase (new GammaAntitheticTestCase);
  */
  AddTestCase (new ErlangTestCase);
  AddTestCase (new ErlangAntitheticTestCase);
  AddTestCase (new ZipfTestCase);
  AddTestCase (new ZipfAntitheticTestCase);
  AddTestCase (new ZetaTestCase);
  AddTestCase (new ZetaAntitheticTestCase);
  AddTestCase (new DeterministicTestCase);
  AddTestCase (new EmpiricalTestCase);
  AddTestCase (new EmpiricalAntitheticTestCase);
  /// Issue #302:  NormalRandomVariable produces stale values
  AddTestCase (new NormalCachingTestCase);
}

static RandomVariableSuite randomVariableSuite;  //!< Static variable for test initialization

}  // namespace RandomVariable

}  // namespace test

}  // namespace ns3

