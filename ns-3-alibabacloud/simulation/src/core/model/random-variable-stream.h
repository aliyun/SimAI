/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
 * Copyright (c) 2011 Mathieu Lacage
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
 * Authors: Rajib Bhattacharjea<raj.b@gatech.edu>
 *          Hadi Arbabi<marbabi@cs.odu.edu>
 *          Mathieu Lacage <mathieu.lacage@gmail.com>
 *
 * Modified by Mitch Watrous <watrous@u.washington.edu>
 *
 */
#ifndef RANDOM_VARIABLE_STREAM_H
#define RANDOM_VARIABLE_STREAM_H

#include "type-id.h"
#include "object.h"
#include "attribute-helper.h"
#include <stdint.h>

/**
 * \file
 * \ingroup randomvariable
 * ns3::RandomVariableStream declaration, and related classes.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup randomvariable Random Variables
 *
 * \brief ns-3 random numbers are provided via instances of
 * ns3::RandomVariableStream.
 *
 * - By default, ns-3 simulations use a fixed seed; if there is any
 *   randomness in the simulation, each run of the program will yield
 *   identical results unless the seed and/or run number is changed.
 * - In ns-3.3 and earlier, ns-3 simulations used a random seed by default;
 *   this marks a change in policy starting with ns-3.4.
 * - In ns-3.14 and earlier, ns-3 simulations used a different wrapper
 *   class called ns3::RandomVariable.  This implementation is documented
 *   above under Legacy Random Variables. As of ns-3.15, this class has
 *   been replaced by ns3::RandomVariableStream; the underlying
 *   pseudo-random number generator has not changed.
 * - To obtain randomness across multiple simulation runs, you must
 *   either set the seed differently or set the run number differently.
 *   To set a seed, call ns3::RngSeedManager::SetSeed() at the beginning
 *   of the program; to set a run number with the same seed, call
 *   ns3::RngSeedManager::SetRun() at the beginning of the program.
 * - Each RandomVariableStream used in ns-3 has a virtual random number
 *   generator associated with it; all random variables use either
 *   a fixed or random seed based on the use of the global seed.
 * - If you intend to perform multiple runs of the same scenario,
 *   with different random numbers, please be sure to read the manual
 *   section on how to perform independent replications.
 */

class RngStream;

/**
 * \ingroup randomvariable
 * \brief The basic uniform Random Number Generator (RNG).
 *
 * \note The underlying random number generation method used
 * by ns-3 is the RngStream code by Pierre L'Ecuyer at
 * the University of Montreal.
 *
 * ns-3 has a rich set of random number generators that allow stream
 * numbers to be set deterministically if desired.  Class
 * RandomVariableStream defines the base class functionality required
 * for all such random number generators.
 *
 * By default, the underlying generator is seeded all the time with
 * the same seed value and run number coming from the ns3::GlobalValue
 * \ref GlobalValueRngSeed "RngSeed" and \ref GlobalValueRngRun
 * "RngRun".  Also by default, the stream number value for the
 * underlying RngStream is automatically allocated.
 *
 * Instances can be configured to return "antithetic" values.
 * See the documentation for the specific distributions to see
 * how this modifies the returned values.
 */
class RandomVariableStream : public Object
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  /**
   * \brief Default constructor.
   */
  RandomVariableStream ();
  /**
   * \brief Destructor.
   */
  virtual ~RandomVariableStream ();

  /**
   * \brief Specifies the stream number for the RngStream.
   * \param [in] stream The stream number for the RngStream.
   * -1 means "allocate a stream number automatically".
   */
  void SetStream (int64_t stream);

  /**
   * \brief Returns the stream number for the RngStream.
   * \return The stream number for the RngStream.
   * -1 means this stream was allocated automatically.
   */
  int64_t GetStream (void) const;

  /**
   * \brief Specify whether antithetic values should be generated.
   * \param [in] isAntithetic If \c true antithetic value will be generated.
   */
  void SetAntithetic (bool isAntithetic);

  /**
   * \brief Check if antithetic values will be generated.
   * \return \c true if antithetic values will be generated.
   */
  bool IsAntithetic (void) const;

  /**
   * \brief Get the next random value as a double drawn from the distribution.
   * \return A floating point random value.
   */
  virtual double GetValue (void) = 0;

  /**
   * \brief Get the next random value as an integer drawn from the distribution.
   * \return  An integer random value.
   */
  virtual uint32_t GetInteger (void) = 0;

protected:
  /**
   * \brief Get the pointer to the underlying RngStream.
   * \return The underlying RngStream
   */
  RngStream * Peek (void) const;

private:
  /**
   * Copy constructor.  These objects are not copyable.
   *
   * \param [in] o The RandomVariableStream to copy in construction.
   * \internal
   * Theoretically, it is possible to give them good copy semantics
   * but not enough time to iron out the details.
   */
  RandomVariableStream (const RandomVariableStream &o);
  /**
   * Assignment operator.  These objects can't be copied by assignment.
   *
   * \param [in] o The RandomVariableStream to copy.
   * \return lvalue RandomVariableStream.
   *
   * \internal
   * Theoretically, it is possible to give them good copy semantics
   * but not enough time to iron out the details.
   */
  RandomVariableStream &operator = (const RandomVariableStream &o);

  /** Pointer to the underlying RngStream. */
  RngStream *m_rng;

  /** Indicates if antithetic values should be generated by this RNG stream. */
  bool m_isAntithetic;

  /** The stream number for the RngStream. */
  int64_t m_stream;

};  // class RandomVariableStream


/**
 * \ingroup randomvariable
 * \brief The uniform distribution Random Number Generator (RNG).
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed uniform distribution.  It also supports the generation of
 * single random numbers from various uniform distributions.
 *
 * The output range is \f$[min, max)\f$ for floating point values,
 * (\c max <i>excluded</i>), and \f$[min, max]\f$ (\c max <i>included</i>)
 * for integral values.
 *
 * \par Example
 *
 * Here is an example of how to use this class:
 * \code
 *   double min = 0.0;
 *   double max = 10.0;
 *
 *   Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
 *   x->SetAttribute ("Min", DoubleValue (min));
 *   x->SetAttribute ("Max", DoubleValue (max));
 *
 *   // The values returned by a uniformly distributed random
 *   // variable should always be within the range
 *   //
 *   //     [min, max)  .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 *
 * \par Antithetic Values.
 *
 * Normally this RNG returns values \f$x\f$ in the interval \f$[min,max)\f$.
 * If an instance of this RNG is configured to return antithetic values,
 * the actual value returned is calculated as follows:
 *
 *   - Compute the initial random value \f$x\f$ as normal.
 *   - Compute the distance from the maximum, \f$y = max - x\f$
 *   - Return \f$x' = min + y = min + (max - x)\f$:
 */
class UniformRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a uniform distribution RNG with the default range.
   */
  UniformRandomVariable ();

  /**
   * \brief Get the lower bound on randoms returned by GetValue(void).
   * \return The lower bound on values from GetValue(void).
   */
  double GetMin (void) const;

  /**
   * \brief Get the upper bound on values returned by GetValue(void).
   * \return The upper bound on values from GetValue(void).
   */
  double GetMax (void) const;

  /**
   * \brief Get the next random value, as a double in the specified range
   * \f$[min, max)\f$.
   *
   * \note The upper limit is excluded from the output range.
   *
   * \param [in] min Low end of the range (included).
   * \param [in] max High end of the range (excluded).
   * \return A floating point random value.
   */
  double GetValue (double min, double max);

  /**
   * \brief Get the next random value, as an unsigned integer in the
   * specified range \f$[min, max]\f$.
   *
   * \note The upper limit is included in the output range.
   *
   * \param [in] min Low end of the range.
   * \param [in] max High end of the range.
   * \return A random unsigned integer value.
   */
  uint32_t GetInteger (uint32_t min, uint32_t max);

  // Inherited from RandomVariableStream
  /**
   * \brief Get the next random value as a double drawn from the distribution.
   * \return A floating point random value.
   * \note The upper limit is excluded from the output range.
  */
  virtual double GetValue (void);
  /**
   * \brief Get the next random value as an integer drawn from the distribution.
   * \return  An integer random value.
   * \note The upper limit is included in the output range.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The lower bound on values that can be returned by this RNG stream. */
  double m_min;

  /** The upper bound on values that can be returned by this RNG stream. */
  double m_max;

};  // class UniformRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Random Number Generator (RNG) that returns a constant.
 *
 * This RNG returns the same value for every sample.
 *
 * \par Antithetic Values.
 *
 * This RNG ignores the antithetic setting.
 */
class ConstantRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a constant RNG with the default constant value.
   */
  ConstantRandomVariable ();

  /**
   * \brief Get the constant value returned by this RNG stream.
   * \return The constant value.
   */
  double GetConstant (void) const;

  /**
   * \brief Get the next random value, as a double equal to the argument.
   * \param [in] constant The value to return.
   * \return The floating point argument.
   */
  double GetValue (double constant);
  /**
   * \brief Get the next random value, as an integer equal to the argument.
   * \param [in] constant The value to return.
   * \return The integer argument.
   */
  uint32_t GetInteger (uint32_t constant);

  // Inherited from RandomVariableStream
  /* \note This RNG always returns the same value. */
  virtual double GetValue (void);
  /* \note This RNG always returns the same value. */
  virtual uint32_t GetInteger (void);

private:
  /** The constant value returned by this RNG stream. */
  double m_constant;

};  // class ConstantRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Random Number Generator (RNG) that returns a pattern of
 * sequential values.
 *
 * This RNG has four configuration attributes:
 *
 *  - An increment, \c Increment.
 *  - A consecutive repeat numer, \c Consecutive.
 *  - The minimum value, \c Min.
 *  - The maximum value, \c Max.
 *
 * The RNG starts at the \c Min value.  Each return value is
 * repeated \c Consecutive times, before advancing by the \c Increment.
 * When the \c Increment would cause the value to equal or exceed
 * \c Max it is reset to \c Min first.
 *
 * For example, if an instance is configured with:
 *
 *   Attribute   | Value
 *   :---------- | -----:
 *   Min         |    2
 *   Max         |   13
 *   Increment   |    4
 *   Consecutive |    3
 *
 * The sequence will repeat this pattern:  2 2 2 6 6 6 10 10 10.
 *
 * Notice that \c Max will be a strict upper bound on the values:
 * all values in the sequence will be less than \c Max.
 *
 * \par Antithetic Values.
 *
 * This RNG ignores the antithetic setting.
 */
class SequentialRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a sequential RNG with the default values
   * for the sequence parameters.
   */
  SequentialRandomVariable ();

  /**
   * \brief Get the first value of the sequence.
   * \return The first value of the sequence.
   */
  double GetMin (void) const;

  /**
   * \brief Get the limit of the sequence, which is (at least)
   * one more than the last value of the sequence.
   * \return The limit of the sequence.
   */
  double GetMax (void) const;

  /**
   * \brief Get the increment for the sequence.
   * \return The increment between distinct values for the sequence.
   */
  Ptr<RandomVariableStream> GetIncrement (void) const;

  /**
   * \brief Get the number of times each distinct value of the sequence
   * is repeated before incrementing to the next value.
   * \return The number of times each value is repeated.
   */
  uint32_t GetConsecutive (void) const;

  // Inherited from RandomVariableStream
  virtual double GetValue (void);
  virtual uint32_t GetInteger (void);

private:
  /** The first value of the sequence. */
  double m_min;

  /** Strict upper bound on the sequence. */
  double m_max;

  /** Increment between distinct values. */
  Ptr<RandomVariableStream> m_increment;

  /** The number of times each distinct value is repeated. */
  uint32_t m_consecutive;

  /** The current sequence value. */
  double m_current;

  /** The number of times the current distinct value has been repeated. */
  uint32_t m_currentConsecutive;

  /** Indicates if the current sequence value has been properly initialized. */
  bool m_isCurrentSet;

};  // class SequentialRandomVariable


/**
 * \ingroup randomvariable
 * \brief The exponential distribution Random Number Generator (RNG).
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed exponential distribution.  It also supports the generation of
 * single random numbers from various exponential distributions.
 *
 * The probability density function of an exponential variable
 * is defined as:
 *   \f[
 *       P(x) dx = \alpha  e^{-\alpha x} dx, \quad x \in [0, +\infty)
 *   \f]
 * over the interval \f$[0, +\infty)\f$, where \f$ \alpha = \frac{1}{Mean} \f$
 * and \c Mean is a configurable attribute.
 *
 * The normal RNG value \f$x\f$ is calculated by
 *
 *   \f[
 *       x = - 1/\alpha \log(u)
 *   \f]
 *
 * where \f$u\f$ is a uniform random variable on \f$[0,1)\f$.
 *
 * \par Bounded Distribution
 *
 * Since exponential distributions can theoretically return unbounded
 * values, it is sometimes useful to specify a fixed upper limit.  The
 * bounded version is defined over the interval \f$[0,b]\f$ as:
 *
 *   \f[
 *       P(x; b) dx = \alpha e^{-\alpha x} dx \quad x \in [0,b]
 *   \f]
 *
 * where the \c Bound \f$b\f$ is a configurable attribute.
 *
 * Note that in this case the true mean of the distribution is smaller
 * than the nominal mean value:
 *
 *   \f[
 *       <X: P(x; b)> = 1/\alpha - b/(e^{\alpha \, b} -1)
 *   \f]
 *
 * \par Example
 *
 * Here is an example of how to use this class:
 * \code
 *   double mean = 3.14;
 *   double bound = 0.0;
 *
 *   Ptr<ExponentialRandomVariable> x = CreateObject<ExponentialRandomVariable> ();
 *   x->SetAttribute ("Mean", DoubleValue (mean));
 *   x->SetAttribute ("Bound", DoubleValue (bound));
 *
 *   // The expected value for the mean of the values returned by an
 *   // exponentially distributed random variable is equal to mean.
 *   double value = x->GetValue ();
 * \endcode
 *
 * \par Antithetic Values.
 *
 * The antithetic value is calculated from
 *
 *   \f[
 *       x' = - mean * \log(1 - u),
 *   \f]
 *
 * where again \f$u\f$ is a uniform random variable on \f$[0,1)\f$.
 */
class ExponentialRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates an exponential distribution RNG with the default
   * values for the mean and upper bound.
   */
  ExponentialRandomVariable ();

  /**
   * \brief Get the configured mean value of this RNG.
   *
   * \note This will not be the actual mean if the distribution is
   * truncated by a bound.
   * \return The configured mean value.
   */
  double GetMean (void) const;

  /**
   * \brief Get the configured upper bound of this RNG.
   * \return The upper bound.
   */
  double GetBound (void) const;

  /**
   * \brief Get the next random value, as a double from
   * the exponential distribution with the specified mean and upper bound.
   * \param [in] mean Mean value of the unbounded exponential distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A floating point random value.
   */
  double GetValue (double mean, double bound);

  /**
   * \brief Get the next random value, as an unsigned integer from
   * the exponential distribution with the specified mean and upper bound.
   * \param [in] mean Mean value of the unbounded exponential distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A random unsigned integer value.
   */
  uint32_t GetInteger (uint32_t mean, uint32_t bound);

  // Inherited from RandomVariableStream
  virtual double GetValue (void);
  virtual uint32_t GetInteger (void);

private:
  /** The mean value of the unbounded exponential distribution. */
  double m_mean;

  /** The upper bound on values that can be returned by this RNG stream. */
  double m_bound;

};  // class ExponentialRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Pareto distribution Random Number Generator (RNG).
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed Pareto distribution.  It also supports the generation of
 * single random numbers from various Pareto distributions.
 *
 * The probability density function of a Pareto variable is defined
 * over the range [\f$x_m\f$,\f$+\infty\f$) as: \f$ k \frac{x_m^k}{x^{k+1}}\f$
 * where \f$x_m > 0\f$ is called the scale parameter and \f$ k > 0\f$
 * is called the Pareto index or shape.
 *
 * The parameter \f$ x_m \f$ can be inferred from the mean and the parameter \f$ k \f$
 * with the equation \f$ x_m = mean \frac{k-1}{k},  k > 1\f$.
 *
 * Since Pareto distributions can theoretically return unbounded values,
 * it is sometimes useful to specify a fixed upper limit.  Note however
 * when the upper limit is specified, the true mean of the distribution
 * is slightly smaller than the mean value specified.
 *
 * Here is an example of how to use this class:
 * \code
 *   double scale = 5.0;
 *   double shape = 2.0;
 *
 *   Ptr<ParetoRandomVariable> x = CreateObject<ParetoRandomVariable> ();
 *   x->SetAttribute ("Scale", DoubleValue (scale));
 *   x->SetAttribute ("Shape", DoubleValue (shape));
 *
 *   // The expected value for the mean of the values returned by a
 *   // Pareto distributed random variable is
 *   //
 *   //                   shape * scale
 *   //     E[value]  =  ---------------  ,
 *   //                     shape - 1
 *
 *   double value = x->GetValue ();
 * \endcode
 */
class ParetoRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a Pareto distribution RNG with the default
   * values for the mean, the shape, and upper bound.
   */
  ParetoRandomVariable ();

  /**
   * \brief Returns the scale parameter for the Pareto distribution returned by this RNG stream.
   * \return The scale parameter for the Pareto distribution returned by this RNG stream.
   */
  double GetScale (void) const;

  /**
   * \brief Returns the shape parameter for the Pareto distribution returned by this RNG stream.
   * \return The shape parameter for the Pareto distribution returned by this RNG stream.
   */
  double GetShape (void) const;

  /**
   * \brief Returns the upper bound on values that can be returned by this RNG stream.
   * \return The upper bound on values that can be returned by this RNG stream.
   */
  double GetBound (void) const;

  /**
   * \brief Returns a random double from a Pareto distribution with the specified scale, shape, and upper bound.
   * \param [in] scale Mean parameter for the Pareto distribution.
   * \param [in] shape Shape parameter for the Pareto distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = \frac{scale}{u^{\frac{1}{shape}}}
   *    \f]
   *
   * is a value that would be returned normally.
   *
   * The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *         x' = \frac{scale}{{(1 - u)}^{\frac{1}{shape}}} ,
   *    \f]
   *
   * which now involves the distance \f$u\f$ is from 1 in the denominator.
   */
  double GetValue (double scale, double shape, double bound);

  /**
   * \brief Returns a random unsigned integer from a Pareto distribution with the specified mean, shape, and upper bound.
   * \param [in] scale Scale parameter for the Pareto distribution.
   * \param [in] shape Shape parameter for the Pareto distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = \frac{scale}{u^{\frac{1}{shape}}}
   *    \f]
   *
   * is a value that would be returned normally.
   *
   * The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *         x' = \frac{scale}{{(1 - u)}^{\frac{1}{shape}}} ,
   *    \f]
   *
   * which now involves the distance \f$u\f$ is from 1 in the denominator.
   */
  uint32_t GetInteger (uint32_t scale, uint32_t shape, uint32_t bound);

  /**
   * \brief Returns a random double from a Pareto distribution with the current mean, shape, and upper bound.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = \frac{scale}{u^{\frac{1}{shape}}}
   *    \f]
   *
   * is a value that would be returned normally, where
   *
   *    \f[
   *         scale  =  mean * (shape - 1.0) / shape  .
   *    \f]
   *
   * The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *         x' = \frac{scale}{{(1 - u)}^{\frac{1}{shape}}} ,
   *    \f]
   *
   * which now involves the distance \f$u\f$ is from 1 in the denominator.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the three-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a Pareto distribution with the current mean, shape, and upper bound.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = \frac{scale}{u^{\frac{1}{shape}}}
   *    \f]
   *
   * is a value that would be returned normally.
   *
   * The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *         x' = \frac{scale}{{(1 - u)}^{\frac{1}{shape}}} ,
   *    \f]
   *
   * which now involves the distance \f$u\f$ is from 1 in the denominator.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The scale parameter for the Pareto distribution returned by this RNG stream. */
  double m_scale;

  /** The shape parameter for the Pareto distribution returned by this RNG stream. */
  double m_shape;

  /** The upper bound on values that can be returned by this RNG stream. */
  double m_bound;

};  // class ParetoRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Weibull distribution Random Number Generator (RNG) that allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed Weibull distribution.  It also supports the generation of
 * single random numbers from various Weibull distributions.
 *
 * The probability density function is defined over the interval [0, \f$+\infty\f$]
 * as: \f$ \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-\left(\frac{x}{\lambda}\right)^k} \f$
 * where \f$ k > 0\f$ is the shape parameter and \f$ \lambda > 0\f$  is the scale parameter. The
 * specified mean is related to the scale and shape parameters by the following relation:
 * \f$ mean = \lambda\Gamma\left(1+\frac{1}{k}\right) \f$ where \f$ \Gamma \f$ is the Gamma function.
 *
 * Since Weibull distributions can theoretically return unbounded values,
 * it is sometimes useful to specify a fixed upper limit.  Note however
 * when the upper limit is specified, the true mean of the distribution
 * is slightly smaller than the mean value specified.
 *
 * Here is an example of how to use this class:
 * \code
 *   double scale = 5.0;
 *   double shape = 1.0;
 *
 *   Ptr<WeibullRandomVariable> x = CreateObject<WeibullRandomVariable> ();
 *   x->SetAttribute ("Scale", DoubleValue (scale));
 *   x->SetAttribute ("Shape", DoubleValue (shape));
 *
 *   // The expected value for the mean of the values returned by a
 *   // Weibull distributed random variable is
 *   //
 *   //     E[value]  =  scale * Gamma(1 + 1 / shape)  ,
 *   //
 *   // where Gamma() is the Gamma function.  Note that
 *   //
 *   //     Gamma(n)  =  (n - 1)!
 *   //
 *   // if n is a positive integer.
 *   //
 *   // For this example,
 *   //
 *   //     Gamma(1 + 1 / shape)  =  Gamma(1 + 1 / 1)
 *   //                           =  Gamma(2)
 *   //                           =  (2 - 1)!
 *   //                           =  1
 *   //
 *   // which means
 *   //
 *   //     E[value]  =  scale  .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class WeibullRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a Weibull distribution RNG with the default
   * values for the scale, shape, and upper bound.
   */
  WeibullRandomVariable ();

  /**
   * \brief Returns the scale parameter for the Weibull distribution returned by this RNG stream.
   * \return The scale parameter for the Weibull distribution returned by this RNG stream.
   */
  double GetScale (void) const;

  /**
   * \brief Returns the shape parameter for the Weibull distribution returned by this RNG stream.
   * \return The shape parameter for the Weibull distribution returned by this RNG stream.
   */
  double GetShape (void) const;

  /**
   * \brief Returns the upper bound on values that can be returned by this RNG stream.
   * \return The upper bound on values that can be returned by this RNG stream.
   */
  double GetBound (void) const;

  /**
   * \brief Returns a random double from a Weibull distribution with the specified scale, shape, and upper bound.
   * \param [in] scale Scale parameter for the Weibull distribution.
   * \param [in] shape Shape parameter for the Weibull distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = scale * {(-\log(u))}^{\frac{1}{shape}}
   *    \f]
   *
   * is a value that would be returned normally, then \f$(1 - u\f$) is
   * the distance that \f$u\f$ would be from \f$1\f$.  The value
   * returned in the antithetic case, \f$x'\f$, is calculated as
   *
   *    \f[
   *         x' = scale * {(-\log(1 - u))}^{\frac{1}{shape}} ,
   *    \f]
   *
   * which now involves the log of the distance \f$u\f$ is from 1.
   */
  double GetValue (double scale, double shape, double bound);

  /**
   * \brief Returns a random unsigned integer from a Weibull distribution with the specified scale, shape, and upper bound.
   * \param [in] scale Scale parameter for the Weibull distribution.
   * \param [in] shape Shape parameter for the Weibull distribution.
   * \param [in] bound Upper bound on values returned.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = scale * {(-\log(u))}^{\frac{1}{shape}}
   *    \f]
   *
   * is a value that would be returned normally, then \f$(1 - u\f$) is
   * the distance that \f$u\f$ would be from \f$1\f$.  The value
   * returned in the antithetic case, \f$x'\f$, is calculated as
   *
   *    \f[
   *         x' = scale * {(-\log(1 - u))}^{\frac{1}{shape}} ,
   *    \f]
   *
   * which now involves the log of the distance \f$u\f$ is from 1.
   */
  uint32_t GetInteger (uint32_t scale, uint32_t shape, uint32_t bound);

  /**
   * \brief Returns a random double from a Weibull distribution with the current scale, shape, and upper bound.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = scale * {(-\log(u))}^{\frac{1}{shape}}
   *    \f]
   *
   * is a value that would be returned normally, then \f$(1 - u\f$) is
   * the distance that \f$u\f$ would be from \f$1\f$.  The value
   * returned in the antithetic case, \f$x'\f$, is calculated as
   *
   *    \f[
   *         x' = scale * {(-\log(1 - u))}^{\frac{1}{shape}} ,
   *    \f]
   *
   * which now involves the log of the distance \f$u\f$ is from 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the three-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a Weibull distribution with the current scale, shape, and upper bound.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = scale * {(-\log(u))}^{\frac{1}{shape}}
   *    \f]
   *
   * is a value that would be returned normally, then \f$(1 - u\f$) is
   * the distance that \f$u\f$ would be from \f$1\f$.  The value
   * returned in the antithetic case, \f$x'\f$, is calculated as
   *
   *    \f[
   *         x' = scale * {(-\log(1 - u))}^{\frac{1}{shape}} ,
   *    \f]
   *
   * which now involves the log of the distance \f$u\f$ is from 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The scale parameter for the Weibull distribution returned by this RNG stream. */
  double m_scale;

  /** The shape parameter for the Weibull distribution returned by this RNG stream. */
  double m_shape;

  /** The upper bound on values that can be returned by this RNG stream. */
  double m_bound;

};  // class WeibullRandomVariable


/**
 * \ingroup randomvariable
 * \brief The normal (Gaussian) distribution Random Number Generator
 * (RNG) that allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed normal distribution.  It also supports the generation of
 * single random numbers from various normal distributions.
 *
 * The density probability function is defined over the interval (\f$-\infty\f$,\f$+\infty\f$)
 * as: \f$ \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{s\sigma^2}}\f$
 * where \f$ mean = \mu \f$ and \f$ variance = \sigma^2 \f$
 *
 * Since normal distributions can theoretically return unbounded
 * values, it is sometimes useful to specify a fixed bound.  The
 * NormalRandomVariable is bounded symmetrically about the mean by
 * this bound, i.e. its values are confined to the interval
 * [\f$mean-bound\f$,\f$mean+bound\f$].
 *
 * Here is an example of how to use this class:
 * \code
 *   double mean = 5.0;
 *   double variance = 2.0;
 *
 *   Ptr<NormalRandomVariable> x = CreateObject<NormalRandomVariable> ();
 *   x->SetAttribute ("Mean", DoubleValue (mean));
 *   x->SetAttribute ("Variance", DoubleValue (variance));
 *
 *   // The expected value for the mean of the values returned by a
 *   // normally distributed random variable is equal to mean.
 *   double value = x->GetValue ();
 * \endcode
 */
class NormalRandomVariable : public RandomVariableStream
{
public:
  /** Large constant to bound the range. */
  static const double INFINITE_VALUE;

  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a normal distribution RNG with the default
   * values for the mean, variance, and bound.
   */
  NormalRandomVariable ();

  /**
   * \brief Returns the mean value for the normal distribution returned by this RNG stream.
   * \return The mean value for the normal distribution returned by this RNG stream.
   */
  double GetMean (void) const;

  /**
   * \brief Returns the variance value for the normal distribution returned by this RNG stream.
   * \return The variance value for the normal distribution returned by this RNG stream.
   */
  double GetVariance (void) const;

  /**
   * \brief Returns the bound on values that can be returned by this RNG stream.
   * \return The bound on values that can be returned by this RNG stream.
   */
  double GetBound (void) const;

  /**
   * \brief Returns a random double from a normal distribution with the specified mean, variance, and bound.
   * \param [in] mean Mean value for the normal distribution.
   * \param [in] variance Variance value for the normal distribution.
   * \param [in] bound Bound on values returned.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the values that would be returned normally, \f$x1\f$ and \f$x2\f$, are calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & 2 * u1 - 1     \\
   *         v2 & = & 2 * u2 - 1     \\
   *         w & = & v1 * v1 + v2 * v2     \\
   *         y & = & \sqrt{\frac{-2 * \log(w)}{w}}     \\
   *         x1 & = & mean + v1 * y * \sqrt{variance}     \\
   *         x2 & = & mean + v2 * y * \sqrt{variance}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic values returned, \f$x1'\f$ and \f$x2'\f$, are
   * calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & 2 * (1 - u1) - 1     \\
   *         v2' & = & 2 * (1 - u2) - 1     \\
   *         w' & = & v1' * v1' + v2' * v2'     \\
   *         y' & = & \sqrt{\frac{-2 * \log(w')}{w'}}     \\
   *         x1' & = & mean + v1' * y' * \sqrt{variance}     \\
   *         x2' & = & mean + v2' * y' * \sqrt{variance}  ,
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  double GetValue (double mean, double variance, double bound = NormalRandomVariable::INFINITE_VALUE);

  /**
   * \brief Returns a random unsigned integer from a normal distribution with the specified mean, variance, and bound.
   * \param [in] mean Mean value for the normal distribution.
   * \param [in] variance Variance value for the normal distribution.
   * \param [in] bound Bound on values returned.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the values that would be returned normally, \f$x1\f$ and \f$x2\f$, are calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & 2 * u1 - 1     \\
   *         v2 & = & 2 * u2 - 1     \\
   *         w & = & v1 * v1 + v2 * v2     \\
   *         y & = & \sqrt{\frac{-2 * \log(w)}{w}}     \\
   *         x1 & = & mean + v1 * y * \sqrt{variance}     \\
   *         x2 & = & mean + v2 * y * \sqrt{variance}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic values returned, \f$x1'\f$ and \f$x2'\f$, are
   * calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & 2 * (1 - u1) - 1     \\
   *         v2' & = & 2 * (1 - u2) - 1     \\
   *         w' & = & v1' * v1' + v2' * v2'     \\
   *         y' & = & \sqrt{\frac{-2 * \log(w')}{w'}}     \\
   *         x1' & = & mean + v1' * y' * \sqrt{variance}     \\
   *         x2' & = & mean + v2' * y' * \sqrt{variance}  ,
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  uint32_t GetInteger (uint32_t mean, uint32_t variance, uint32_t bound);

  /**
   * \brief Returns a random double from a normal distribution with the current mean, variance, and bound.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the values that would be returned normally, \f$x1\f$ and \f$x2\f$, are calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & 2 * u1 - 1     \\
   *         v2 & = & 2 * u2 - 1     \\
   *         w & = & v1 * v1 + v2 * v2     \\
   *         y & = & \sqrt{\frac{-2 * \log(w)}{w}}     \\
   *         x1 & = & mean + v1 * y * \sqrt{variance}     \\
   *         x2 & = & mean + v2 * y * \sqrt{variance}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic values returned, \f$x1'\f$ and \f$x2'\f$, are
   * calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & 2 * (1 - u1) - 1     \\
   *         v2' & = & 2 * (1 - u2) - 1     \\
   *         w' & = & v1' * v1' + v2' * v2'     \\
   *         y' & = & \sqrt{\frac{-2 * \log(w')}{w'}}     \\
   *         x1' & = & mean + v1' * y' * \sqrt{variance}     \\
   *         x2' & = & mean + v2' * y' * \sqrt{variance}  ,
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the three-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a normal distribution with the current mean, variance, and bound.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the values that would be returned normally, \f$x1\f$ and \f$x2\f$, are calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & 2 * u1 - 1     \\
   *         v2 & = & 2 * u2 - 1     \\
   *         w & = & v1 * v1 + v2 * v2     \\
   *         y & = & \sqrt{\frac{-2 * \log(w)}{w}}     \\
   *         x1 & = & mean + v1 * y * \sqrt{variance}     \\
   *         x2 & = & mean + v2 * y * \sqrt{variance}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic values returned, \f$x1'\f$ and \f$x2'\f$, are
   * calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & 2 * (1 - u1) - 1     \\
   *         v2' & = & 2 * (1 - u2) - 1     \\
   *         w' & = & v1' * v1' + v2' * v2'     \\
   *         y' & = & \sqrt{\frac{-2 * \log(w')}{w'}}     \\
   *         x1' & = & mean + v1' * y' * \sqrt{variance}     \\
   *         x2' & = & mean + v2' * y' * \sqrt{variance}  ,
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The mean value for the normal distribution returned by this RNG stream. */
  double m_mean;

  /** The variance value for the normal distribution returned by this RNG stream. */
  double m_variance;

  /** The bound on values that can be returned by this RNG stream. */
  double m_bound;

  /** True if the next value is valid. */
  bool m_nextValid;

  /** The algorithm produces two values at a time. Cache parameters for possible reuse.*/
  double m_v2;
  /** The algorithm produces two values at a time. Cache parameters for possible reuse.*/
  double m_y;

};  // class NormalRandomVariable


/**
 * \ingroup randomvariable
 * \brief The log-normal distribution Random Number Generator
 * (RNG) that allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed log-normal distribution.  It also supports the generation of
 * single random numbers from various log-normal distributions.
 *
 * LogNormalRandomVariable defines a random variable with a log-normal
 * distribution.  If one takes the natural logarithm of random
 * variable following the log-normal distribution, the obtained values
 * follow a normal distribution.
 *
 * The probability density function is defined over the interval [0,\f$+\infty\f$) as:
 * \f$ \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(ln(x) - \mu)^2}{2\sigma^2}}\f$
 * where \f$ mean = e^{\mu+\frac{\sigma^2}{2}} \f$ and
 * \f$ variance = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}\f$
 *
 * The \f$ \mu \f$ and \f$ \sigma \f$ parameters can be calculated instead if
 * the mean and variance are known with the following equations:
 * \f$ \mu = ln(mean) - \frac{1}{2}ln\left(1+\frac{variance}{mean^2}\right)\f$, and,
 * \f$ \sigma = \sqrt{ln\left(1+\frac{variance}{mean^2}\right)}\f$
 *
 * Here is an example of how to use this class:
 * \code
 *   double mu = 5.0;
 *   double sigma = 2.0;
 *
 *   Ptr<LogNormalRandomVariable> x = CreateObject<LogNormalRandomVariable> ();
 *   x->SetAttribute ("Mu", DoubleValue (mu));
 *   x->SetAttribute ("Sigma", DoubleValue (sigma));
 *
 *   // The expected value for the mean of the values returned by a
 *   // log-normally distributed random variable is equal to
 *   //
 *   //                             2
 *   //                   mu + sigma  / 2
 *   //     E[value]  =  e                 .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class LogNormalRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a log-normal distribution RNG with the default
   * values for mu and sigma.
   */
  LogNormalRandomVariable ();

  /**
   * \brief Returns the mu value for the log-normal distribution returned by this RNG stream.
   * \return The mu value for the log-normal distribution returned by this RNG stream.
   */
  double GetMu (void) const;

  /**
   * \brief Returns the sigma value for the log-normal distribution returned by this RNG stream.
   * \return The sigma value for the log-normal distribution returned by this RNG stream.
   */
  double GetSigma (void) const;

  /**
   * \brief Returns a random double from a log-normal distribution with the specified mu and sigma.
   * \param [in] mu Mu value for the log-normal distribution.
   * \param [in] sigma Sigma value for the log-normal distribution.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the value that would be returned normally, \f$x\f$, is calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & -1 + 2 * u1       \\
   *         v2 & = & -1 + 2 * u2       \\
   *         r2 & = & v1 * v1 + v2 * v2       \\
   *         normal & = & v1 * \sqrt{\frac{-2.0 * \log{r2}}{r2}}       \\
   *         x & = &  \exp{sigma * normal + mu}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic value returned, \f$x'\f$, is calculated as
   * follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & -1 + 2 * (1 - u1)       \\
   *         v2' & = & -1 + 2 * (1 - u2)       \\
   *         r2' & = & v1' * v1' + v2' * v2'       \\
   *         normal' & = & v1' * \sqrt{\frac{-2.0 * \log{r2'}}{r2'}}       \\
   *         x' & = &  \exp{sigma * normal' + mu}  .
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  double GetValue (double mu, double sigma);

  /**
   * \brief Returns a random unsigned integer from a log-normal distribution with the specified mu and sigma.
   * \param [in] mu Mu value for the log-normal distribution.
   * \param [in] sigma Sigma value for the log-normal distribution.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the value that would be returned normally, \f$x\f$, is calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & -1 + 2 * u1       \\
   *         v2 & = & -1 + 2 * u2       \\
   *         r2 & = & v1 * v1 + v2 * v2       \\
   *         normal & = & v1 * \sqrt{\frac{-2.0 * \log{r2}}{r2}}       \\
   *         x & = &  \exp{sigma * normal + mu}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic value returned, \f$x'\f$, is calculated as
   * follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & -1 + 2 * (1 - u1)       \\
   *         v2' & = & -1 + 2 * (1 - u2)       \\
   *         r2' & = & v1' * v1' + v2' * v2'       \\
   *         normal' & = & v1' * \sqrt{\frac{-2.0 * \log{r2'}}{r2'}}       \\
   *         x' & = &  \exp{sigma * normal' + mu}  .
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  uint32_t GetInteger (uint32_t mu, uint32_t sigma);

  /**
   * \brief Returns a random double from a log-normal distribution with the current mu and sigma.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the value that would be returned normally, \f$x\f$, is calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & -1 + 2 * u1       \\
   *         v2 & = & -1 + 2 * u2       \\
   *         r2 & = & v1 * v1 + v2 * v2       \\
   *         normal & = & v1 * \sqrt{\frac{-2.0 * \log{r2}}{r2}}       \\
   *         x & = &  \exp{sigma * normal + mu}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic value returned, \f$x'\f$, is calculated as
   * follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & -1 + 2 * (1 - u1)       \\
   *         v2' & = & -1 + 2 * (1 - u2)       \\
   *         r2' & = & v1' * v1' + v2' * v2'       \\
   *         normal' & = & v1' * \sqrt{\frac{-2.0 * \log{r2'}}{r2'}}       \\
   *         x' & = &  \exp{sigma * normal' + mu}  .
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the two-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a log-normal distribution with the current mu and sigma.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the value that would be returned normally, \f$x\f$, is calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & -1 + 2 * u1       \\
   *         v2 & = & -1 + 2 * u2       \\
   *         r2 & = & v1 * v1 + v2 * v2       \\
   *         normal & = & v1 * \sqrt{\frac{-2.0 * \log{r2}}{r2}}       \\
   *         x & = &  \exp{sigma * normal + mu}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic value returned, \f$x'\f$, is calculated as
   * follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & -1 + 2 * (1 - u1)       \\
   *         v2' & = & -1 + 2 * (1 - u2)       \\
   *         r2' & = & v1' * v1' + v2' * v2'       \\
   *         normal' & = & v1' * \sqrt{\frac{-2.0 * \log{r2'}}{r2'}}       \\
   *         x' & = &  \exp{sigma * normal' + mu}  .
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The mu value for the log-normal distribution returned by this RNG stream. */
  double m_mu;

  /** The sigma value for the log-normal distribution returned by this RNG stream. */
  double m_sigma;

};  // class LogNormalRandomVariable


/**
 * \ingroup randomvariable
 * \brief The gamma distribution Random Number Generator (RNG) that
 * allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed gamma distribution.  It also supports the generation of
 * single random numbers from various gamma distributions.
 *
 * The probability density function is defined over the interval [0,\f$+\infty\f$) as:
 * \f$ x^{\alpha-1} \frac{e^{-\frac{x}{\beta}}}{\beta^\alpha \Gamma(\alpha)}\f$
 * where \f$ mean = \alpha\beta \f$ and
 * \f$ variance = \alpha \beta^2\f$
 *
 * Here is an example of how to use this class:
 * \code
 *   double alpha = 5.0;
 *   double beta = 2.0;
 *
 *   Ptr<GammaRandomVariable> x = CreateObject<GammaRandomVariable> ();
 *   x->SetAttribute ("Alpha", DoubleValue (alpha));
 *   x->SetAttribute ("Beta", DoubleValue (beta));
 *
 *   // The expected value for the mean of the values returned by a
 *   // gammaly distributed random variable is equal to
 *   //
 *   //     E[value]  =  alpha * beta  .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class GammaRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a gamma distribution RNG with the default values
   * for alpha and beta.
   */
  GammaRandomVariable ();

  /**
   * \brief Returns the alpha value for the gamma distribution returned by this RNG stream.
   * \return The alpha value for the gamma distribution returned by this RNG stream.
   */
  double GetAlpha (void) const;

  /**
   * \brief Returns the beta value for the gamma distribution returned by this RNG stream.
   * \return The beta value for the gamma distribution returned by this RNG stream.
   */
  double GetBeta (void) const;

  /**
   * \brief Returns a random double from a gamma distribution with the specified alpha and beta.
   * \param [in] alpha Alpha value for the gamma distribution.
   * \param [in] beta Beta value for the gamma distribution.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  double GetValue (double alpha, double beta);

  /**
   * \brief Returns a random unsigned integer from a gamma distribution with the specified alpha and beta.
   * \param [in] alpha Alpha value for the gamma distribution.
   * \param [in] beta Beta value for the gamma distribution.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  uint32_t GetInteger (uint32_t alpha, uint32_t beta);

  /**
   * \brief Returns a random double from a gamma distribution with the current alpha and beta.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the two-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a gamma distribution with the current alpha and beta.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /**
   * \brief Returns a random double from a normal distribution with the specified mean, variance, and bound.
   * \param [in] mean Mean value for the normal distribution.
   * \param [in] variance Variance value for the normal distribution.
   * \param [in] bound Bound on values returned.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u1\f$ and \f$u2\f$ are uniform variables
   * over [0,1], then the values that would be returned normally, \f$x1\f$ and \f$x2\f$, are calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1 & = & 2 * u1 - 1     \\
   *         v2 & = & 2 * u2 - 1     \\
   *         w & = & v1 * v1 + v2 * v2     \\
   *         y & = & \sqrt{\frac{-2 * \log(w)}{w}}     \\
   *         x1 & = & mean + v1 * y * \sqrt{variance}     \\
   *         x2 & = & mean + v2 * y * \sqrt{variance}  .
   *    \f}
   *
   * For the antithetic case, \f$(1 - u1\f$) and \f$(1 - u2\f$) are
   * the distances that \f$u1\f$ and \f$u2\f$ would be from \f$1\f$.
   * The antithetic values returned, \f$x1'\f$ and \f$x2'\f$, are
   * calculated as follows:
   *
   *    \f{eqnarray*}{
   *         v1' & = & 2 * (1 - u1) - 1     \\
   *         v2' & = & 2 * (1 - u2) - 1     \\
   *         w' & = & v1' * v1' + v2' * v2'     \\
   *         y' & = & \sqrt{\frac{-2 * \log(w')}{w'}}     \\
   *         x1' & = & mean + v1' * y' * \sqrt{variance}     \\
   *         x2' & = & mean + v2' * y' * \sqrt{variance}  ,
   *    \f}
   *
   * which now involves the distances \f$u1\f$ and \f$u2\f$ are from 1.
   */
  double GetNormalValue (double mean, double variance, double bound);

  /** The alpha value for the gamma distribution returned by this RNG stream. */
  double m_alpha;

  /** The beta value for the gamma distribution returned by this RNG stream. */
  double m_beta;

  /** True if the next normal value is valid. */
  bool m_nextValid;

  /** The algorithm produces two values at a time. Cache parameters for possible reuse.*/
  double m_v2;
  /** The algorithm produces two values at a time. Cache parameters for possible reuse.*/
  double m_y;

};  // class GammaRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Erlang distribution Random Number Generator (RNG) that
 * allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed Erlang distribution.  It also supports the generation of
 * single random numbers from various Erlang distributions.
 *
 * The Erlang distribution is a special case of the Gamma distribution where k
 * (= alpha) is a non-negative integer. Erlang distributed variables can be
 * generated using a much faster algorithm than gamma variables.
 *
 * The probability density function is defined over the interval [0,\f$+\infty\f$) as:
 * \f$ \frac{x^{k-1} e^{-\frac{x}{\lambda}}}{\lambda^k (k-1)!}\f$
 * where \f$ mean = k \lambda \f$ and
 * \f$ variance = k \lambda^2\f$
 *
 * Here is an example of how to use this class:
 * \code
 *   uint32_t k = 5;
 *   double lambda = 2.0;
 *
 *   Ptr<ErlangRandomVariable> x = CreateObject<ErlangRandomVariable> ();
 *   x->SetAttribute ("K", IntegerValue (k));
 *   x->SetAttribute ("Lambda", DoubleValue (lambda));
 *
 *   // The expected value for the mean of the values returned by a
 *   // Erlangly distributed random variable is equal to
 *   //
 *   //     E[value]  =  k * lambda  .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class ErlangRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates an Erlang distribution RNG with the default values
   * for k and lambda.
   */
  ErlangRandomVariable ();

  /**
   * \brief Returns the k value for the Erlang distribution returned by this RNG stream.
   * \return The k value for the Erlang distribution returned by this RNG stream.
   */
  uint32_t GetK (void) const;

  /**
   * \brief Returns the lambda value for the Erlang distribution returned by this RNG stream.
   * \return The lambda value for the Erlang distribution returned by this RNG stream.
   */
  double GetLambda (void) const;

  /**
   * \brief Returns a random double from an Erlang distribution with the specified k and lambda.
   * \param [in] k K value for the Erlang distribution.
   * \param [in] lambda Lambda value for the Erlang distribution.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  double GetValue (uint32_t k, double lambda);

  /**
   * \brief Returns a random unsigned integer from an Erlang distribution with the specified k and lambda.
   * \param [in] k K value for the Erlang distribution.
   * \param [in] lambda Lambda value for the Erlang distribution.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  uint32_t GetInteger (uint32_t k, uint32_t lambda);

  /**
   * \brief Returns a random double from an Erlang distribution with the current k and lambda.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the two-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from an Erlang distribution with the current k and lambda.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /**
   * \brief Returns a random double from an exponential distribution with the specified mean and upper bound.
   * \param [in] mean Mean value of the random variables.
   * \param [in] bound Upper bound on values returned.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *         x = - mean * \log(u)
   *    \f]
   *
   * is a value that would be returned normally, then \f$(1 - u\f$) is
   * the distance that \f$u\f$ would be from \f$1\f$.  The value
   * returned in the antithetic case, \f$x'\f$, is calculated as
   *
   *    \f[
   *         x' = - mean * \log(1 - u),
   *    \f]
   *
   * which now involves the log of the distance \f$u\f$ is from the 1.
   */
  double GetExponentialValue (double mean, double bound);

  /** The k value for the Erlang distribution returned by this RNG stream. */
  uint32_t m_k;

  /** The lambda value for the Erlang distribution returned by this RNG stream. */
  double m_lambda;

};  // class ErlangRandomVariable


/**
 * \ingroup randomvariable
 * \brief The triangular distribution Random Number Generator (RNG) that
 * allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed triangular distribution.  It also supports the generation of
 * single random numbers from various triangular distributions.
 *
 * This distribution is a triangular distribution.  The probability density
 * is in the shape of a triangle.
 *
 * Here is an example of how to use this class:
 * \code
 *   double mean = 5.0;
 *   double min = 2.0;
 *   double max = 10.0;
 *
 *   Ptr<TriangularRandomVariable> x = CreateObject<TriangularRandomVariable> ();
 *   x->SetAttribute ("Mean", DoubleValue (mean));
 *   x->SetAttribute ("Min", DoubleValue (min));
 *   x->SetAttribute ("Max", DoubleValue (max));
 *
 *   // The expected value for the mean of the values returned by a
 *   // triangularly distributed random variable is equal to mean.
 *   double value = x->GetValue ();
 * \endcode
 */
class TriangularRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a triangular distribution RNG with the default
   * values for the mean, lower bound, and upper bound.
   */
  TriangularRandomVariable ();

  /**
   * \brief Returns the mean value for the triangular distribution returned by this RNG stream.
   * \return The mean value for the triangular distribution returned by this RNG stream.
   */
  double GetMean (void) const;

  /**
   * \brief Returns the lower bound for the triangular distribution returned by this RNG stream.
   * \return The lower bound for the triangular distribution returned by this RNG stream.
   */
  double GetMin (void) const;

  /**
   * \brief Returns the upper bound on values that can be returned by this RNG stream.
   * \return The upper bound on values that can be returned by this RNG stream.
   */
  double GetMax (void) const;

  /**
   * \brief Returns a random double from a triangular distribution with the specified mean, min, and max.
   * \param [in] mean Mean value for the triangular distribution.
   * \param [in] min Low end of the range.
   * \param [in] max High end of the range.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *        x = \left\{ \begin{array}{rl}
   *           min + \sqrt{u * (max - min) * (mode - min)} &\mbox{ if $u <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ (1 - u) * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *    \f]
   *
   * is a value that would be returned normally, where the mode or
   * peak of the triangle is calculated as
   *
   *    \f[
   *         mode =  3.0 * mean - min - max  .
   *    \f]
   *
   * Then, \f$(1 - u\f$) is the distance that \f$u\f$ would be from
   * \f$1\f$.  The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *        x' = \left\{ \begin{array}{rl}
   *           min + \sqrt{(1 - u) * (max - min) * (mode - min)} &\mbox{ if $(1 - u) <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ u * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *     \f]
   *
   * which now involves the distance \f$u\f$ is from the 1.
   */
  double GetValue (double mean, double min, double max);

  /**
   * \brief Returns a random unsigned integer from a triangular distribution with the specified mean, min, and max.
   * \param [in] mean Mean value for the triangular distribution.
   * \param [in] min Low end of the range.
   * \param [in] max High end of the range.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *        x = \left\{ \begin{array}{rl}
   *           min + \sqrt{u * (max - min) * (mode - min)} &\mbox{ if $u <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ (1 - u) * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *    \f]
   *
   * is a value that would be returned normally, where the mode or
   * peak of the triangle is calculated as
   *
   *    \f[
   *         mode =  3.0 * mean - min - max  .
   *    \f]
   *
   * Then, \f$(1 - u\f$) is the distance that \f$u\f$ would be from
   * \f$1\f$.  The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *        x' = \left\{ \begin{array}{rl}
   *           min + \sqrt{(1 - u) * (max - min) * (mode - min)} &\mbox{ if $(1 - u) <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ u * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *     \f]
   *
   * which now involves the distance \f$u\f$ is from the 1.
   */
  uint32_t GetInteger (uint32_t mean, uint32_t min, uint32_t max);

  /**
   * \brief Returns a random double from a triangular distribution with the current mean, min, and max.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *        x = \left\{ \begin{array}{rl}
   *           min + \sqrt{u * (max - min) * (mode - min)} &\mbox{ if $u <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ (1 - u) * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *    \f]
   *
   * is a value that would be returned normally, where the mode or
   * peak of the triangle is calculated as
   *
   *    \f[
   *         mode =  3.0 * mean - min - max  .
   *    \f]
   *
   * Then, \f$(1 - u\f$) is the distance that \f$u\f$ would be from
   * \f$1\f$.  The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *        x' = \left\{ \begin{array}{rl}
   *           min + \sqrt{(1 - u) * (max - min) * (mode - min)} &\mbox{ if $(1 - u) <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ u * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *     \f]
   *
   * which now involves the distance \f$u\f$ is from the 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the three-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a triangular distribution with the current mean, min, and max.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if
   * m_isAntithetic is equal to true.  If \f$u\f$ is a uniform variable
   * over [0,1] and
   *
   *    \f[
   *        x = \left\{ \begin{array}{rl}
   *           min + \sqrt{u * (max - min) * (mode - min)} &\mbox{ if $u <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ (1 - u) * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *    \f]
   *
   * is a value that would be returned normally, where the mode or
   * peak of the triangle is calculated as
   *
   *    \f[
   *         mode =  3.0 * mean - min - max  .
   *    \f]
   *
   * Then, \f$(1 - u\f$) is the distance that \f$u\f$ would be from
   * \f$1\f$.  The value returned in the antithetic case, \f$x'\f$, is
   * calculated as
   *
   *    \f[
   *        x' = \left\{ \begin{array}{rl}
   *           min + \sqrt{(1 - u) * (max - min) * (mode - min)} &\mbox{ if $(1 - u) <= (mode - min)/(max - min)$} \\
   *           max - \sqrt{ u * (max - min) * (max - mode) } &\mbox{ otherwise}
   *                 \end{array} \right.
   *     \f]
   *
   * which now involves the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The mean value for the triangular distribution returned by this RNG stream. */
  double m_mean;

  /** The lower bound on values that can be returned by this RNG stream. */
  double m_min;

  /** The upper bound on values that can be returned by this RNG stream. */
  double m_max;

};  // class TriangularRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Zipf distribution Random Number Generator (RNG) that
 * allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed Zipf distribution.  It also supports the generation of
 * single random numbers from various Zipf distributions.
 *
 * The Zipf's law states that given some corpus of natural language
 * utterances, the frequency of any word is inversely proportional
 * to its rank in the frequency table.
 *
 * Zipf's distribution has two parameters, alpha and N, where:
 * \f$ \alpha > 0 \f$ (real) and \f$ N \in \{1,2,3 \dots\}\f$ (integer).
 * Probability Mass Function is \f$ f(k; \alpha, N) = k^{-\alpha}/ H_{N,\alpha} \f$
 * where \f$ H_{N,\alpha} = \sum_{m=1}^N m^{-\alpha} \f$
 *
 * Here is an example of how to use this class:
 * \code
 *   uint32_t n = 1;
 *   double alpha = 2.0;
 *
 *   Ptr<ZipfRandomVariable> x = CreateObject<ZipfRandomVariable> ();
 *   x->SetAttribute ("N", IntegerValue (n));
 *   x->SetAttribute ("Alpha", DoubleValue (alpha));
 *
 *   // The expected value for the mean of the values returned by a
 *   // Zipfly distributed random variable is equal to
 *   //
 *   //                   H
 *   //                    N, alpha - 1
 *   //     E[value]  =  ---------------
 *   //                     H
 *   //                      N, alpha
 *   //
 *   // where
 *   //
 *   //                    N
 *   //                   ---
 *   //                   \     -alpha
 *   //     H          =  /    m        .
 *   //      N, alpha     ---
 *   //                   m=1
 *   //
 *   // For this test,
 *   //
 *   //                      -(alpha - 1)
 *   //                     1
 *   //     E[value]  =  ---------------
 *   //                      -alpha
 *   //                     1
 *   //
 *   //               =  1  .
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class ZipfRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a Zipf distribution RNG with the default values
   * for n and alpha.
   */
  ZipfRandomVariable ();

  /**
   * \brief Returns the n value for the Zipf distribution returned by this RNG stream.
   * \return The n value for the Zipf distribution returned by this RNG stream.
   */
  uint32_t GetN (void) const;

  /**
   * \brief Returns the alpha value for the Zipf distribution returned by this RNG stream.
   * \return The alpha value for the Zipf distribution returned by this RNG stream.
   */
  double GetAlpha (void) const;

  /**
   * \brief Returns a random double from a Zipf distribution with the specified n and alpha.
   * \param [in] n N value for the Zipf distribution.
   * \param [in] alpha Alpha value for the Zipf distribution.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  double GetValue (uint32_t n, double alpha);

  /**
   * \brief Returns a random unsigned integer from a Zipf distribution with the specified n and alpha.
   * \param [in] n N value for the Zipf distribution.
   * \param [in] alpha Alpha value for the Zipf distribution.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  uint32_t GetInteger (uint32_t n, uint32_t alpha);

  /**
   * \brief Returns a random double from a Zipf distribution with the current n and alpha.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the two-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a Zipf distribution with the current n and alpha.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The n value for the Zipf distribution returned by this RNG stream. */
  uint32_t m_n;

  /** The alpha value for the Zipf distribution returned by this RNG stream. */
  double m_alpha;

  /** The normalization constant. */
  double m_c;

};  // class ZipfRandomVariable


/**
 * \ingroup randomvariable
 * \brief The zeta distribution Random Number Generator (RNG) that
 * allows stream numbers to be set deterministically.
 *
 * This class supports the creation of objects that return random numbers
 * from a fixed zeta distribution.  It also supports the generation of
 * single random numbers from various zeta distributions.
 *
 * The Zeta distribution is closely related to Zipf distribution when
 * N goes to infinity.
 *
 * Zeta distribution has one parameter, alpha, \f$ \alpha > 1 \f$ (real).
 * Probability Mass Function is \f$ f(k; \alpha) = k^{-\alpha}/\zeta(\alpha) \f$
 * where \f$ \zeta(\alpha) \f$ is the Riemann zeta function ( \f$ \sum_{n=1}^\infty n^{-\alpha} ) \f$
 *
 * Here is an example of how to use this class:
 * \code
 *   double alpha = 2.0;
 *
 *   Ptr<ZetaRandomVariable> x = CreateObject<ZetaRandomVariable> ();
 *   x->SetAttribute ("Alpha", DoubleValue (alpha));
 *
 *   // The expected value for the mean of the values returned by a
 *   // zetaly distributed random variable is equal to
 *   //
 *   //                   zeta(alpha - 1)
 *   //     E[value]  =  ---------------   for alpha > 2 ,
 *   //                     zeta(alpha)
 *   //
 *   // where zeta(alpha) is the Riemann zeta function.
 *   //
 *   // There are no simple analytic forms for the Riemann zeta
 *   // function, which is the reason the known mean of the values
 *   // cannot be calculated in this example.
 *   //
 *   double value = x->GetValue ();
 * \endcode
 */
class ZetaRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a zeta distribution RNG with the default value for
   * alpha.
   */
  ZetaRandomVariable ();

  /**
   * \brief Returns the alpha value for the zeta distribution returned by this RNG stream.
   * \return The alpha value for the zeta distribution returned by this RNG stream.
   */
  double GetAlpha (void) const;

  /**
   * \brief Returns a random double from a zeta distribution with the specified alpha.
   * \param [in] alpha Alpha value for the zeta distribution.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  double GetValue (double alpha);

  /**
   * \brief Returns a random unsigned integer from a zeta distribution with the specified alpha.
   * \param [in] alpha Alpha value for the zeta distribution.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  uint32_t GetInteger (uint32_t alpha);

  /**
   * \brief Returns a random double from a zeta distribution with the current alpha.
   * \return A floating point random value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   *
   * Note that we have to re-implement this method here because the method is
   * overloaded above for the two-argument variant and the c++ name resolution
   * rules don't work well with overloads split between parent and child
   * classes.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns a random unsigned integer from a zeta distribution with the current alpha.
   * \return A random unsigned integer value.
   *
   * Note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

private:
  /** The alpha value for the zeta distribution returned by this RNG stream. */
  double m_alpha;

  /** Just for calculus simplifications. */
  double m_b;

};  // class ZetaRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Random Number Generator (RNG) that returns a predetermined sequence.
 *
 * Defines a random variable that has a specified, predetermined
 * sequence.  This would be useful when trying to force the RNG to
 * return a known sequence, perhaps to compare ns-3 to some other
 * simulator
 *
 * Creates a generator that returns successive elements of the values
 * array on successive calls to RandomVariableStream::GetValue.  Note
 * that the values in the array are copied and stored by the generator
 * (deep-copy).  Also note that the sequence repeats if more values
 * are requested than are present in the array.
 *
 * Here is an example of how to use this class:
 * \code
 *   Ptr<DeterministicRandomVariable> s = CreateObject<DeterministicRandomVariable> ();
 *
 *   // The following array should give the sequence
 *   //
 *   //    4, 4, 7, 7, 10, 10 .
 *   //
 *   double array [] = { 4, 4, 7, 7, 10, 10};
 *   uint64_t count = 6;
 *   s->SetValueArray (array, count);
 *
 *   double value = x->GetValue ();
 * \endcode
 */
class DeterministicRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates a deterministic RNG that will have a predetermined
   * sequence of values.
   */
  DeterministicRandomVariable ();
  virtual ~DeterministicRandomVariable ();

  /**
   * \brief Sets the array of values that holds the predetermined sequence.
   * \param [in] values Array of random values to return in sequence.
   * \param [in] length Number of values in the array.
   *
   * Note that the values in the array are copied and stored
   * (deep-copy).
   */
  void SetValueArray (double* values, std::size_t length);

  /**
   * \brief Returns the next value in the sequence.
   * \return The floating point next value in the sequence.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns the next value in the sequence.
   * \return The integer next value in the sequence.
   */
  virtual uint32_t GetInteger (void);

private:
  /** Size of the array of values. */
  std::size_t   m_count;

  /** Position of the next value in the array of values. */
  std::size_t   m_next;

  /** Array of values to return in sequence. */
  double* m_data;

};  // class DeterministicRandomVariable


/**
 * \ingroup randomvariable
 * \brief The Random Number Generator (RNG) that has a specified
 * empirical distribution.
 *
 * Defines a random variable  that has a specified, empirical
 * distribution.  The cumulative probability distribution function (CDF)
 * is specified by a
 * series of calls to the CDF() member function, specifying a
 * value and the probability that the distribution is less than
 * the specified value.  When random values are requested,
 * a uniform random variable `r` is used to select a probability,
 * and the return value is chosen from the largest input value with CDF
 * less than the random value. The method is known
 * as inverse transform sampling:
 * (http://en.wikipedia.org/wiki/Inverse_transform_sampling).
 *
 * This generator has two modes: *sampling* and *interpolating*.
 * In *sampling* mode this random variable generator
 * treats the CDF as an exact histogram and returns
 * one of the histogram inputs exactly.  This is appropriate
 * when the configured CDF represents the exact probability
 * distribution, for a categorical variable, for example.
 *
 * In *interpolating* mode this random variable generator linearly
 * interpolates between the CDF values defining the histogram bins.
 * This is appropriate when the configured CDF is an approximation
 * to a continuous underlying probability distribution.
 *
 * For historical reasons the default is interpolating.
 * To switch modes use the \c Interpolate Attribute, or call SetInterpolate().
 * You can change modes at any time.
 *
 * If you find yourself switching frequently it could be simpler to
 * set the mode to sampling, then use the GetValue() function for
 * sampled values, and Interpolate() function for interpolated values.
 *
 * The CDF need not start with a probability of zero,
 * nor end with a probability of 1.0.  If the selected uniform
 * random value `r` is less than the first CDF point, that point
 * is selected. If `r` is greater than the last CDF point the last
 * point is selected.  In either case the interpolating mode will *not*
 * interpolate (since there is no value beyond `r` to work with), but
 * simply return the extremal CDF value, as in sampling.
 * 
 * Here is an example of how to use this class:
 *
 *    // Create the RNG with a non-uniform distribution between 0 and 10.
 *    // in sampling mode.
 *    Ptr<EmpiricalRandomVariable> x = CreateObject<EmpiricalRandomVariable> ();
 *    x->SetInterpolate (false);
 *    x->CDF ( 0.0,  0.0);
 *    x->CDF ( 5.0,  0.25);
 *    x->CDF (10.0,  1.0);
 *
 *    double value = x->GetValue ();
 *
 * The expected values and probabilities returned by GetValue are
 *
 * Value | Probability
 * ----: | ----------:
 *  0.0  |  0
 *  5.0  | 25%
 * 10.0  | 75%
 *
 * The only two values ever returned are 5 and 10, with unequal probability.
 *
 * If instead you want linear interpolation between the points of the CDF
 * use the Interpolate() function:
 *
 *     double interp = x->Interpolate ();
 *
 * This will return continuous values on the range [0,1).
 *
 * See empirical-random-variable-example.cc for an example.
 */
class EmpiricalRandomVariable : public RandomVariableStream
{
public:
  /**
   * \brief Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Creates an empirical RNG that has a specified, empirical
   * distribution, and configured for interpolating mode.
   */
  EmpiricalRandomVariable (void);

  /**
   * \brief Specifies a point in the empirical distribution
   * \note These *MUST* be inserted in ascending order of \p c
   *
   * \param [in] v The function value for this point
   * \param [in] c Probability that the function is less than or equal to \p v
   */
  void CDF (double v, double c);  // Value, prob <= Value

  /**
   * \brief Returns the next value in the empirical distribution.
   * \return The floating point next value in the empirical distribution.
   *
   * Note that this does not interpolate the CDF, but treats it as a
   * stepwise continuous function.
   *
   * Also note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual double GetValue (void);

  /**
   * \brief Returns the next value in the empirical distribution.
   * \return The integer next value in the empirical distribution.
   *
   * Note that this does not interpolate the CDF, but treats it as a
   * stepwise continuous function.
   * Also note that antithetic values are being generated if m_isAntithetic
   * is equal to true.  If \f$u\f$ is a uniform variable over [0,1]
   * and \f$x\f$ is a value that would be returned normally, then
   * \f$(1 - u\f$) is the distance that \f$u\f$ would be from \f$1\f$.
   * The value returned in the antithetic case, \f$x'\f$, uses (1-u),
   * which is the distance \f$u\f$ is from the 1.
   */
  virtual uint32_t GetInteger (void);

  /**
   * \brief Returns the next value in the empirical distribution using
   * linear interpolation.
   * \return The floating point next value in the empirical distribution
   * using linear interpolation.
   */
    virtual double Interpolate (void);

  /**
   * \brief Switch the mode between sampling the CDF and interpolating.
   * The default mode is sampling.
   * \param [in] interpolate If \c true set to interpolation, otherwise sampling.
   * \returns The previous interpolate flag value.
   */
  bool SetInterpolate (bool interpolate);

private:
  /** \brief Helper to hold one point of the CDF. */
  class ValueCDF
  {
  public:
    /** \brief Constructor. */
    ValueCDF (void);
    /**
     * \brief Construct from values.
     *
     * \param [in] v The argument value.
     * \param [in] c The CDF at the argument value \pname{v}
     */
    ValueCDF (double v, double c);

    /** The argument value. */
    double value;
    /** The CDF at \pname{value}  */
    double    cdf;
  };  // class ValueCDF

  /**
   * \brief Check that the CDF is valid.
   *
   * A valid CDF has
   *
   * - Strictly increasing arguments, and
   * - Strictly increasing CDF.
   *
   * It is a fatal error to fail validation.
   */
  void Validate (void);
   /**
   * \brief Do the initial rng draw and check against the extrema.
   *
   * If the extrema apply, \c value will have the extremal value
   * and the return will be \c true.
   *
   * If the extrema do not apply \c value will have the URNG value
   * and the return will be \c false.
   *
   * \param [out] value The extremal value, or the URNG.
   * \returns \c true if \p value is the extremal result,
   *          or \c false if \p value is the URNG value.
   */
  bool PreSample (double & value);
  /**
   * \brief Sample the CDF as a histogram (without interpolation).
   * \param [in] r The CDF value at which to sample the CDF.
   * \return The bin value corresponding to \c r.
   */
  double DoSampleCDF (double r);
  /**
   * \brief Linear interpolation between two points on the CDF to estimate
   * the value at \p r.
   *
   * \param [in] r  The argument value to interpolate to.
   * \returns The interpolated CDF at \pname{r}
   */
  double DoInterpolate (double r);

  /**
   * \brief Comparison operator, for use by std::upper_bound
   * \param a [in] the first value
   * \param b [in] the second value
   * \returns \c true if \c a.cdf < \c b.cdf
   */
  friend
  bool operator < (ValueCDF a, ValueCDF b);

  /** \c true once the CDF has been validated. */
  bool m_validated;
  /** The vector of CDF points. */
  std::vector<ValueCDF> m_emp;
  /**
   * If \c true GetValue will interpolate,
   * otherwise treat CDF as normal histogram.
   */
  bool m_interpolate;

};  // class EmpiricalRandomVariable


} // namespace ns3

#endif /* RANDOM_VARIABLE_STREAM_H */
