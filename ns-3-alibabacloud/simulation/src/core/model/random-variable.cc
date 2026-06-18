/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Rajib Bhattacharjea<raj.b@gatech.edu>
// Author: Hadi Arbabi<marbabi@cs.odu.edu>
//

#include <iostream>
#include <cmath>
#include <cstdlib>
#if HAVE_SYS_TIME_H
#include <sys/time.h>                   // for gettimeofday
#endif
#if HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef WIN32
#include "winport.h"
#endif
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <vector>

#include "assert.h"
#include "random-variable.h"
#include "rng-seed-manager.h"
#include "rng-stream.h"
#include "fatal-error.h"
#include "log.h"

namespace ns3 {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// RandomVariableBase methods

NS_LOG_COMPONENT_DEFINE ("RandomVariable");

/** \ingroup legacyrandom */
class RandomVariableBase
{
public:
  RandomVariableBase ();
  RandomVariableBase (const RandomVariableBase &o);
  virtual ~RandomVariableBase ();
  virtual double  GetValue () = 0;
  virtual uint32_t GetInteger ();
  virtual RandomVariableBase*   Copy (void) const = 0;
  RngStream *GetStream(void);
private:
  RngStream* m_generator;  // underlying generator being wrapped
};

RandomVariableBase::RandomVariableBase ()
  : m_generator (0)
{
  NS_LOG_FUNCTION (this);
}

RandomVariableBase::RandomVariableBase (const RandomVariableBase& r)
  : m_generator (0)
{
  NS_LOG_FUNCTION (this << &r);
  if (r.m_generator != 0)
    {
      m_generator = new RngStream (RngSeedManager::GetSeed (),
                                   RngSeedManager::GetNextStreamIndex (),
                                   RngSeedManager::GetRun ());
    }
}

RandomVariableBase::~RandomVariableBase ()
{
  NS_LOG_FUNCTION (this);
  delete m_generator;
}

uint32_t RandomVariableBase::GetInteger ()
{
  NS_LOG_FUNCTION (this);
  return (uint32_t)GetValue ();
}

RngStream *
RandomVariableBase::GetStream (void)
{
  NS_LOG_FUNCTION (this);
  if (m_generator == 0)
    {
      m_generator = new RngStream (RngSeedManager::GetSeed (),
                                   RngSeedManager::GetNextStreamIndex (),
                                   RngSeedManager::GetRun ());
    }
  return m_generator;
}

// -------------------------------------------------------

RandomVariable::RandomVariable ()
  : m_variable (0)
{
  NS_LOG_FUNCTION (this);
}
RandomVariable::RandomVariable (const RandomVariable&o)
  : m_variable (o.m_variable->Copy ())
{
}
RandomVariable::RandomVariable (const RandomVariableBase &variable)
  : m_variable (variable.Copy ())
{
}
RandomVariable &
RandomVariable::operator = (const RandomVariable &o)
{
  if (&o == this)
    {
      return *this;
    }
  delete m_variable;
  m_variable = o.m_variable->Copy ();
  return *this;
}
RandomVariable::~RandomVariable ()
{
  NS_LOG_FUNCTION (this);
  delete m_variable;
}
double
RandomVariable::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_variable->GetValue ();
}

uint32_t
RandomVariable::GetInteger (void) const
{
  NS_LOG_FUNCTION (this);
  return m_variable->GetInteger ();
}

RandomVariableBase *
RandomVariable::Peek (void) const
{
  NS_LOG_FUNCTION (this);
  return m_variable;
}


ATTRIBUTE_VALUE_IMPLEMENT (RandomVariable);
ATTRIBUTE_CHECKER_IMPLEMENT (RandomVariable);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// UniformVariableImpl

/** \ingroup legacyrandom */
class UniformVariableImpl : public RandomVariableBase
{
public:
  /**
   * Creates a uniform random number generator in the
   * range [0.0 .. 1.0).
   */
  UniformVariableImpl ();

  /**
   * Creates a uniform random number generator with the specified range
   * \param s Low end of the range
   * \param l High end of the range
   */
  UniformVariableImpl (double s, double l);

  UniformVariableImpl (const UniformVariableImpl& c);

  double GetMin (void) const;
  double GetMax (void) const;

  /**
   * \return A value between low and high values specified by the constructor
   */
  virtual double GetValue ();

  /**
   * \return A value between low and high values specified by parameters
   */
  virtual double GetValue (double s, double l);

  virtual RandomVariableBase*  Copy (void) const;

private:
  double m_min;
  double m_max;
};

UniformVariableImpl::UniformVariableImpl ()
  : m_min (0),
    m_max (1.0)
{
  NS_LOG_FUNCTION (this);
}

UniformVariableImpl::UniformVariableImpl (double s, double l)
  : m_min (s),
    m_max (l)
{
  NS_LOG_FUNCTION (this << s << l);
}

UniformVariableImpl::UniformVariableImpl (const UniformVariableImpl& c)
  : RandomVariableBase (c),
    m_min (c.m_min),
    m_max (c.m_max)
{
  NS_LOG_FUNCTION (this << &c);
}

double
UniformVariableImpl::GetMin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_min;
}
double
UniformVariableImpl::GetMax (void) const
{
  NS_LOG_FUNCTION (this);
  return m_max;
}


double UniformVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  return m_min + generator->RandU01 () * (m_max - m_min);
}

double UniformVariableImpl::GetValue (double s, double l)
{
  NS_LOG_FUNCTION (this << s << l);
  RngStream *generator = GetStream ();
  return s + generator->RandU01 () * (l - s);
}

RandomVariableBase* UniformVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new UniformVariableImpl (*this);
}

UniformVariable::UniformVariable ()
  : RandomVariable (UniformVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
UniformVariable::UniformVariable (double s, double l)
  : RandomVariable (UniformVariableImpl (s, l))
{
  NS_LOG_FUNCTION (this << s << l);
}

double UniformVariable::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return this->RandomVariable::GetValue ();
}

double UniformVariable::GetValue (double s, double l)
{
  NS_LOG_FUNCTION (this << s << l);
  return ((UniformVariableImpl*)Peek ())->GetValue (s,l);
}

uint32_t UniformVariable::GetInteger (uint32_t s, uint32_t l)
{
  NS_LOG_FUNCTION (this << s << l);
  NS_ASSERT (s <= l);
  return static_cast<uint32_t> ( GetValue (s, l + 1) );
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ConstantVariableImpl methods

/** \ingroup legacyrandom */
class ConstantVariableImpl : public RandomVariableBase
{

public:
  /**
   * Construct a ConstantVariableImpl RNG that returns zero every sample
   */
  ConstantVariableImpl ();

  /**
   * Construct a ConstantVariableImpl RNG that returns the specified value
   * every sample.
   * \param c Unchanging value for this RNG.
   */
  ConstantVariableImpl (double c);


  ConstantVariableImpl (const ConstantVariableImpl& c);

  /**
   * \brief Specify a new constant RNG for this generator.
   * \param c New constant value for this RNG.
   */
  void    NewConstant (double c);

  /**
   * \return The constant value specified
   */
  virtual double  GetValue ();
  virtual uint32_t GetInteger ();
  virtual RandomVariableBase*   Copy (void) const;
private:
  double m_const;
};

ConstantVariableImpl::ConstantVariableImpl ()
  : m_const (0)
{
  NS_LOG_FUNCTION (this);
}

ConstantVariableImpl::ConstantVariableImpl (double c)
  : m_const (c)
{
  NS_LOG_FUNCTION (this << c);
}

ConstantVariableImpl::ConstantVariableImpl (const ConstantVariableImpl& c)
  : RandomVariableBase (c),
    m_const (c.m_const)
{
  NS_LOG_FUNCTION (this << &c);
}

void ConstantVariableImpl::NewConstant (double c)
{
  NS_LOG_FUNCTION (this << c);
  m_const = c;
}

double ConstantVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  return m_const;
}

uint32_t ConstantVariableImpl::GetInteger ()
{
  NS_LOG_FUNCTION (this);
  return (uint32_t)m_const;
}

RandomVariableBase* ConstantVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ConstantVariableImpl (*this);
}

ConstantVariable::ConstantVariable ()
  : RandomVariable (ConstantVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
ConstantVariable::ConstantVariable (double c)
  : RandomVariable (ConstantVariableImpl (c))
{
  NS_LOG_FUNCTION (this << c);
}
void
ConstantVariable::SetConstant (double c)
{
  NS_LOG_FUNCTION (this << c);
  *this = ConstantVariable (c);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// SequentialVariableImpl methods


/** \ingroup legacyrandom */
class SequentialVariableImpl : public RandomVariableBase
{

public:
  /**
   * \brief Constructor for the SequentialVariableImpl RNG.
   *
   * The four parameters define the sequence.  For example
   * SequentialVariableImpl(0,5,1,2) creates a RNG that has the sequence
   * 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0 ...
   * \param f First value of the sequence.
   * \param l One more than the last value of the sequence.
   * \param i Increment between sequence values
   * \param c Number of times each member of the sequence is repeated
   */
  SequentialVariableImpl (double f, double l, double i = 1, uint32_t c = 1);

  /**
   * \brief Constructor for the SequentialVariableImpl RNG.
   *
   * Differs from the first only in that the increment parameter is a
   * random variable
   * \param f First value of the sequence.
   * \param l One more than the last value of the sequence.
   * \param i Reference to a RandomVariableBase for the sequence increment
   * \param c Number of times each member of the sequence is repeated
   */
  SequentialVariableImpl (double f, double l, const RandomVariable& i, uint32_t c = 1);

  SequentialVariableImpl (const SequentialVariableImpl& c);

  ~SequentialVariableImpl ();
  /**
   * \return The next value in the Sequence
   */
  virtual double GetValue ();
  virtual RandomVariableBase*  Copy (void) const;
private:
  double m_min;
  double m_max;
  RandomVariable  m_increment;
  uint32_t  m_consecutive;
  double m_current;
  uint32_t  m_currentConsecutive;
};

SequentialVariableImpl::SequentialVariableImpl (double f, double l, double i, uint32_t c)
  : m_min (f),
    m_max (l),
    m_increment (ConstantVariable (i)),
    m_consecutive (c),
    m_current (f),
    m_currentConsecutive (0)
{
  NS_LOG_FUNCTION (this << f << l << i << c);
}

SequentialVariableImpl::SequentialVariableImpl (double f, double l, const RandomVariable& i, uint32_t c)
  : m_min (f),
    m_max (l),
    m_increment (i),
    m_consecutive (c),
    m_current (f),
    m_currentConsecutive (0)
{
  NS_LOG_FUNCTION (this << f << l << i << c);
}

SequentialVariableImpl::SequentialVariableImpl (const SequentialVariableImpl& c)
  : RandomVariableBase (c),
    m_min (c.m_min),
    m_max (c.m_max),
    m_increment (c.m_increment),
    m_consecutive (c.m_consecutive),
    m_current (c.m_current),
    m_currentConsecutive (c.m_currentConsecutive)
{
  NS_LOG_FUNCTION (this << &c);
}

SequentialVariableImpl::~SequentialVariableImpl ()
{
  NS_LOG_FUNCTION (this);
}

double SequentialVariableImpl::GetValue ()
{ // Return a sequential series of values
  NS_LOG_FUNCTION (this);
  double r = m_current;
  if (++m_currentConsecutive == m_consecutive)
    { // Time to advance to next
      m_currentConsecutive = 0;
      m_current += m_increment.GetValue ();
      if (m_current >= m_max)
        {
          m_current = m_min + (m_current - m_max);
        }
    }
  return r;
}

RandomVariableBase* SequentialVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new SequentialVariableImpl (*this);
}

SequentialVariable::SequentialVariable (double f, double l, double i, uint32_t c)
  : RandomVariable (SequentialVariableImpl (f, l, i, c))
{
  NS_LOG_FUNCTION (this << f << l << i << c);
}
SequentialVariable::SequentialVariable (double f, double l, const RandomVariable& i, uint32_t c)
  : RandomVariable (SequentialVariableImpl (f, l, i, c))
{
  NS_LOG_FUNCTION (this << f << l << i << c);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ExponentialVariableImpl methods

/** \ingroup legacyrandom */
class ExponentialVariableImpl : public RandomVariableBase
{
public:
  /**
   * Constructs an exponential random variable  with a mean
   * value of 1.0.
   */
  ExponentialVariableImpl ();

  /**
   * \brief Constructs an exponential random variable with a specified mean
   *
   * \param m Mean value for the random variable
   */
  explicit ExponentialVariableImpl (double m);

  /**
   * \brief Constructs an exponential random variable with specified
   * mean and upper limit.
   *
   * Since exponential distributions can theoretically return unbounded values,
   * it is sometimes useful to specify a fixed upper limit.  Note however when
   * the upper limit is specified, the true mean of the distribution is
   * slightly smaller than the mean value specified: \f$ m - b/(e^{b/m}-1) \f$.
   * \param m Mean value of the random variable
   * \param b Upper bound on returned values
   */
  ExponentialVariableImpl (double m, double b);

  ExponentialVariableImpl (const ExponentialVariableImpl& c);

  /**
   * \return A random value from this exponential distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

private:
  double m_mean;  // Mean value of RV
  double m_bound; // Upper bound on value (if non-zero)
};

ExponentialVariableImpl::ExponentialVariableImpl ()
  : m_mean (1.0),
    m_bound (0)
{
  NS_LOG_FUNCTION (this);
}

ExponentialVariableImpl::ExponentialVariableImpl (double m)
  : m_mean (m),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << m);
}

ExponentialVariableImpl::ExponentialVariableImpl (double m, double b)
  : m_mean (m),
    m_bound (b)
{
  NS_LOG_FUNCTION (this << m << b);
}

ExponentialVariableImpl::ExponentialVariableImpl (const ExponentialVariableImpl& c)
  : RandomVariableBase (c),
    m_mean (c.m_mean),
    m_bound (c.m_bound)
{
  NS_LOG_FUNCTION (this << &c);
}

double ExponentialVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  while (1)
    {
      double r = -m_mean*std::log (generator->RandU01 ());
      if (m_bound == 0 || r <= m_bound)
        {
          return r;
        }
      // otherwise, try again
    }
}

RandomVariableBase* ExponentialVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ExponentialVariableImpl (*this);
}

ExponentialVariable::ExponentialVariable ()
  : RandomVariable (ExponentialVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
ExponentialVariable::ExponentialVariable (double m)
  : RandomVariable (ExponentialVariableImpl (m))
{
  NS_LOG_FUNCTION (this << m);
}
ExponentialVariable::ExponentialVariable (double m, double b)
  : RandomVariable (ExponentialVariableImpl (m, b))
{
  NS_LOG_FUNCTION (this << m << b);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ParetoVariableImpl methods
/** \ingroup legacyrandom */
class ParetoVariableImpl : public RandomVariableBase
{
public:
  /**
   * Constructs a pareto random variable with a mean of 1 and a shape
   * parameter of 1.5
   */
  ParetoVariableImpl ();

  /**
   * \brief Constructs a pareto random variable with specified mean and shape
   * parameter of 1.5
   *
   * \param m Mean value of the distribution
   */
  explicit ParetoVariableImpl (double m);

  /**
   * \brief Constructs a pareto random variable with the specified mean
   * value and shape parameter. Beware, s must be strictly greater than 1.
   *
   * \param m Mean value of the distribution
   * \param s Shape parameter for the distribution
   */
  ParetoVariableImpl (double m, double s);

  /**
   * \brief Constructs a pareto random variable with the specified mean
   * value, shape (alpha), and upper bound. Beware, s must be strictly greater than 1.
   *
   * Since pareto distributions can theoretically return unbounded values,
   * it is sometimes useful to specify a fixed upper limit.  Note however
   * when the upper limit is specified, the true mean of the distribution
   * is slightly smaller than the mean value specified.
   * \param m Mean value
   * \param s Shape parameter
   * \param b Upper limit on returned values
   */
  ParetoVariableImpl (double m, double s, double b);

  /**
   * \brief Constructs a pareto random variable with the specified scale and shape
   * parameters.
   *
   * \param params the two parameters, respectively scale and shape, of the distribution
   */
  ParetoVariableImpl (std::pair<double, double> params);

  /**
   * \brief Constructs a pareto random variable with the specified
   * scale, shape (alpha), and upper bound.
   *
   * Since pareto distributions can theoretically return unbounded values,
   * it is sometimes useful to specify a fixed upper limit.  Note however
   * when the upper limit is specified, the true mean of the distribution
   * is slightly smaller than the mean value specified.
   *
   * \param params the two parameters, respectively scale and shape, of the distribution
   * \param b Upper limit on returned values
   */
  ParetoVariableImpl (std::pair<double, double> params, double b);

  ParetoVariableImpl (const ParetoVariableImpl& c);

  /**
   * \return A random value from this Pareto distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy () const;

private:
  double m_scale; // Scale value of RV
  double m_shape; // Shape parameter
  double m_bound; // Upper bound on value (if non-zero)
};

ParetoVariableImpl::ParetoVariableImpl ()
  : m_scale (0.5 / 1.5),
    m_shape (1.5),
    m_bound (0)
{
  NS_LOG_FUNCTION (this);
}

ParetoVariableImpl::ParetoVariableImpl (double m)
  : m_scale (m * 0.5 / 1.5),
    m_shape (1.5),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << m);
}

ParetoVariableImpl::ParetoVariableImpl (double m, double s)
  : m_scale (m * (s - 1.0) / s),
    m_shape (s),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << m << s);
}

ParetoVariableImpl::ParetoVariableImpl (double m, double s, double b)
  : m_scale (m * (s - 1.0) / s),
    m_shape (s),
    m_bound (b)
{
  NS_LOG_FUNCTION (this << m << s << b);
}

ParetoVariableImpl::ParetoVariableImpl (std::pair<double, double> params)
  : m_scale (params.first),
    m_shape (params.second),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << &params);
}

ParetoVariableImpl::ParetoVariableImpl (std::pair<double, double> params, double b)
  : m_scale (params.first),
    m_shape (params.second),
    m_bound (b)
{
  NS_LOG_FUNCTION (this << &params << b);
}

ParetoVariableImpl::ParetoVariableImpl (const ParetoVariableImpl& c)
  : RandomVariableBase (c),
    m_scale (c.m_scale),
    m_shape (c.m_shape),
    m_bound (c.m_bound)
{
  NS_LOG_FUNCTION (this << &c);
}

double ParetoVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  while (1)
    {
      double r = (m_scale * ( 1.0 / std::pow (generator->RandU01 (), 1.0 / m_shape)));
      if (m_bound == 0 || r <= m_bound)
        {
          return r;
        }
      // otherwise, try again
    }
}

RandomVariableBase* ParetoVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ParetoVariableImpl (*this);
}

ParetoVariable::ParetoVariable ()
  : RandomVariable (ParetoVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
ParetoVariable::ParetoVariable (double m)
  : RandomVariable (ParetoVariableImpl (m))
{
  NS_LOG_FUNCTION (this << m);
}
ParetoVariable::ParetoVariable (double m, double s)
  : RandomVariable (ParetoVariableImpl (m, s))
{
  NS_LOG_FUNCTION (this << m << s);
}
ParetoVariable::ParetoVariable (double m, double s, double b)
  : RandomVariable (ParetoVariableImpl (m, s, b))
{
  NS_LOG_FUNCTION (this << m << s << b);
}
ParetoVariable::ParetoVariable (std::pair<double, double> params)
  : RandomVariable (ParetoVariableImpl (params))
{
  NS_LOG_FUNCTION (this << &params);
}
ParetoVariable::ParetoVariable (std::pair<double, double> params, double b)
  : RandomVariable (ParetoVariableImpl (params, b))
{
  NS_LOG_FUNCTION (this << &params << b);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// WeibullVariableImpl methods

/** \ingroup legacyrandom */
class WeibullVariableImpl : public RandomVariableBase
{
public:
  /**
   * Constructs a weibull random variable  with a mean
   * value of 1.0 and a shape (alpha) parameter of 1
   */
  WeibullVariableImpl ();


  /**
   * Constructs a weibull random variable with the specified mean
   * value and a shape (alpha) parameter of 1.5.
   * \param m mean value of the distribution
   */
  WeibullVariableImpl (double m);

  /**
   * Constructs a weibull random variable with the specified mean
   * value and a shape (alpha).
   * \param m Mean value for the distribution.
   * \param s Shape (alpha) parameter for the distribution.
   */
  WeibullVariableImpl (double m, double s);

  /**
  * \brief Constructs a weibull random variable with the specified mean
  * \brief value, shape (alpha), and upper bound.
  * Since WeibullVariableImpl distributions can theoretically return unbounded values,
  * it is sometimes usefull to specify a fixed upper limit.  Note however
  * that when the upper limit is specified, the true mean of the distribution
  * is slightly smaller than the mean value specified.
  * \param m Mean value for the distribution.
  * \param s Shape (alpha) parameter for the distribution.
  * \param b Upper limit on returned values
  */
  WeibullVariableImpl (double m, double s, double b);

  WeibullVariableImpl (const WeibullVariableImpl& c);

  /**
   * \return A random value from this Weibull distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

private:
  double m_mean;  // Mean value of RV
  double m_alpha; // Shape parameter
  double m_bound; // Upper bound on value (if non-zero)
};

WeibullVariableImpl::WeibullVariableImpl () : m_mean (1.0),
                                              m_alpha (1),
                                              m_bound (0)
{
  NS_LOG_FUNCTION (this);
}
WeibullVariableImpl::WeibullVariableImpl (double m)
  : m_mean (m),
    m_alpha (1),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << m);
}
WeibullVariableImpl::WeibullVariableImpl (double m, double s)
  : m_mean (m),
    m_alpha (s),
    m_bound (0)
{
  NS_LOG_FUNCTION (this << m << s);
}
WeibullVariableImpl::WeibullVariableImpl (double m, double s, double b)
  : m_mean (m),
    m_alpha (s),
    m_bound (b)
{
  NS_LOG_FUNCTION (this << m << s << b);
}
WeibullVariableImpl::WeibullVariableImpl (const WeibullVariableImpl& c)
  : RandomVariableBase (c),
    m_mean (c.m_mean),
    m_alpha (c.m_alpha),
    m_bound (c.m_bound)
{
  NS_LOG_FUNCTION (this << &c);
}

double WeibullVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  double exponent = 1.0 / m_alpha;
  while (1)
    {
      double r = m_mean * std::pow ( -std::log (generator->RandU01 ()), exponent);
      if (m_bound == 0 || r <= m_bound)
        {
          return r;
        }
      // otherwise, try again
    }
}

RandomVariableBase* WeibullVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new WeibullVariableImpl (*this);
}

WeibullVariable::WeibullVariable ()
  : RandomVariable (WeibullVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
WeibullVariable::WeibullVariable (double m)
  : RandomVariable (WeibullVariableImpl (m))
{
  NS_LOG_FUNCTION (this << m);
}
WeibullVariable::WeibullVariable (double m, double s)
  : RandomVariable (WeibullVariableImpl (m, s))
{
  NS_LOG_FUNCTION (this << m << s);
}
WeibullVariable::WeibullVariable (double m, double s, double b)
  : RandomVariable (WeibullVariableImpl (m, s, b))
{
  NS_LOG_FUNCTION (this << m << s << b);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// NormalVariableImpl methods

/** \ingroup legacyrandom */
class NormalVariableImpl : public RandomVariableBase   // Normally Distributed random var

{
public:
  static const double INFINITE_VALUE;
  /**
   * Constructs an normal random variable  with a mean
   * value of 0 and variance of 1.
   */
  NormalVariableImpl ();

  /**
   * \brief Construct a normal random variable with specified mean and variance
   * \param m Mean value
   * \param v Variance
   * \param b Bound.  The NormalVariableImpl is bounded within +-bound of the mean.
   */
  NormalVariableImpl (double m, double v, double b = INFINITE_VALUE);

  NormalVariableImpl (const NormalVariableImpl& c);

  /**
   * \return A value from this normal distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

  double GetMean (void) const;
  double GetVariance (void) const;
  double GetBound (void) const;

private:
  double m_mean;      // Mean value of RV
  double m_variance;  // Mean value of RV
  double m_bound;     // Bound on value's difference from the mean (absolute value)
  bool   m_nextValid; // True if next valid
  double m_next;      // The algorithm produces two values at a time
};

const double NormalVariableImpl::INFINITE_VALUE = 1e307;

NormalVariableImpl::NormalVariableImpl ()
  : m_mean (0.0),
    m_variance (1.0),
    m_bound (INFINITE_VALUE),
    m_nextValid (false)
{
  NS_LOG_FUNCTION (this);
}

NormalVariableImpl::NormalVariableImpl (double m, double v, double b)
  : m_mean (m),
    m_variance (v),
    m_bound (b),
    m_nextValid (false)
{
  NS_LOG_FUNCTION (this << m << v << b);
}

NormalVariableImpl::NormalVariableImpl (const NormalVariableImpl& c)
  : RandomVariableBase (c),
    m_mean (c.m_mean),
    m_variance (c.m_variance),
    m_bound (c.m_bound),
    m_nextValid (false)
{
  NS_LOG_FUNCTION (this << &c);
}

double NormalVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  if (m_nextValid)
    { // use previously generated
      m_nextValid = false;
      return m_next;
    }
  while (1)
    { // See Simulation Modeling and Analysis p. 466 (Averill Law)
      // for algorithm; basically a Box-Muller transform:
      // http://en.wikipedia.org/wiki/Box-Muller_transform
      double u1 = generator->RandU01 ();
      double u2 = generator->RandU01 ();
      double v1 = 2 * u1 - 1;
      double v2 = 2 * u2 - 1;
      double w = v1 * v1 + v2 * v2;
      if (w <= 1.0)
        { // Got good pair
          double y = std::sqrt ((-2 * std::log (w)) / w);
          m_next = m_mean + v2 * y * std::sqrt (m_variance);
          // if next is in bounds, it is valid
          m_nextValid = std::fabs (m_next - m_mean) <= m_bound;
          double x1 = m_mean + v1 * y * std::sqrt (m_variance);
          // if x1 is in bounds, return it
          if (std::fabs (x1 - m_mean) <= m_bound)
            {
              return x1;
            }
          // otherwise try and return m_next if it is valid
          else if (m_nextValid)
            {
              m_nextValid = false;
              return m_next;
            }
          // otherwise, just run this loop again
        }
    }
}

RandomVariableBase* NormalVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new NormalVariableImpl (*this);
}

double
NormalVariableImpl::GetMean (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mean;
}

double
NormalVariableImpl::GetVariance (void) const
{
  NS_LOG_FUNCTION (this);
  return m_variance;
}

double
NormalVariableImpl::GetBound (void) const
{
  NS_LOG_FUNCTION (this);
  return m_bound;
}

NormalVariable::NormalVariable ()
  : RandomVariable (NormalVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
NormalVariable::NormalVariable (double m, double v)
  : RandomVariable (NormalVariableImpl (m, v))
{
  NS_LOG_FUNCTION (this << m << v);
}
NormalVariable::NormalVariable (double m, double v, double b)
  : RandomVariable (NormalVariableImpl (m, v, b))
{
  NS_LOG_FUNCTION (this << m << v << b);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/** \ingroup legacyrandom */
class EmpiricalVariableImpl : public RandomVariableBase
{
public:
  /**
   * Constructor for the EmpiricalVariableImpl random variables.
   */
  explicit EmpiricalVariableImpl ();

  virtual ~EmpiricalVariableImpl ();
  EmpiricalVariableImpl (const EmpiricalVariableImpl& c);
  /**
   * \return A value from this empirical distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;
  /**
   * \brief Specifies a point in the empirical distribution
   * \param v The function value for this point
   * \param c Probability that the function is less than or equal to v
   */
  virtual void CDF (double v, double c);  // Value, prob <= Value

private:
  class ValueCDF
  {
public:
    ValueCDF ();
    ValueCDF (double v, double c);
    ValueCDF (const ValueCDF& c);
    double value;
    double    cdf;
  };
  virtual void Validate ();  // Insure non-decreasing emiprical values
  virtual double Interpolate (double, double, double, double, double);
  bool validated; // True if non-decreasing validated
  std::vector<ValueCDF> emp;       // Empicical CDF
};


// ValueCDF methods
EmpiricalVariableImpl::ValueCDF::ValueCDF ()
  : value (0.0),
    cdf (0.0)
{
  NS_LOG_FUNCTION (this);
}
EmpiricalVariableImpl::ValueCDF::ValueCDF (double v, double c)
  : value (v),
    cdf (c)
{
  NS_LOG_FUNCTION (this << v << c);
}
EmpiricalVariableImpl::ValueCDF::ValueCDF (const ValueCDF& c)
  : value (c.value),
    cdf (c.cdf)
{
  NS_LOG_FUNCTION (this << &c);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// EmpiricalVariableImpl methods
EmpiricalVariableImpl::EmpiricalVariableImpl ()
  : validated (false)
{
  NS_LOG_FUNCTION (this);
}

EmpiricalVariableImpl::EmpiricalVariableImpl (const EmpiricalVariableImpl& c)
  : RandomVariableBase (c),
    validated (c.validated),
    emp (c.emp)
{
  NS_LOG_FUNCTION (this << &c);
}

EmpiricalVariableImpl::~EmpiricalVariableImpl ()
{
  NS_LOG_FUNCTION (this);
}

double EmpiricalVariableImpl::GetValue ()
{ // Return a value from the empirical distribution
  // This code based (loosely) on code by Bruce Mah (Thanks Bruce!)
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  if (emp.size () == 0)
    {
      return 0.0; // HuH? No empirical data
    }
  if (!validated)
    {
      Validate ();      // Insure in non-decreasing
    }
  double r = generator->RandU01 ();
  if (r <= emp.front ().cdf)
    {
      return emp.front ().value; // Less than first
    }
  if (r >= emp.back ().cdf)
    {
      return emp.back ().value;  // Greater than last
    }
  // Binary search
  std::vector<ValueCDF>::size_type bottom = 0;
  std::vector<ValueCDF>::size_type top = emp.size () - 1;
  while (1)
    {
      std::vector<ValueCDF>::size_type c = (top + bottom) / 2;
      if (r >= emp[c].cdf && r < emp[c + 1].cdf)
        { // Found it
          return Interpolate (emp[c].cdf, emp[c + 1].cdf,
                              emp[c].value, emp[c + 1].value,
                              r);
        }
      // Not here, adjust bounds
      if (r < emp[c].cdf)
        {
          top    = c - 1;
        }
      else
        {
          bottom = c + 1;
        }
    }
}

RandomVariableBase* EmpiricalVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new EmpiricalVariableImpl (*this);
}

void EmpiricalVariableImpl::CDF (double v, double c)
{ // Add a new empirical datapoint to the empirical cdf
  // NOTE.   These MUST be inserted in non-decreasing order
  NS_LOG_FUNCTION (this << v << c);
  emp.push_back (ValueCDF (v, c));
}

void EmpiricalVariableImpl::Validate ()
{
  NS_LOG_FUNCTION (this);
  ValueCDF prior;
  for (std::vector<ValueCDF>::size_type i = 0; i < emp.size (); ++i)
    {
      ValueCDF& current = emp[i];
      if (current.value < prior.value || current.cdf < prior.cdf)
        { // Error
          std::cerr << "Empirical Dist error,"
                    << " current value " << current.value
                    << " prior value "   << prior.value
                    << " current cdf "   << current.cdf
                    << " prior cdf "     << prior.cdf << std::endl;
          NS_FATAL_ERROR ("Empirical Dist error");
        }
      prior = current;
    }
  validated = true;
}

double EmpiricalVariableImpl::Interpolate (double c1, double c2,
                                           double v1, double v2, double r)
{ // Interpolate random value in range [v1..v2) based on [c1 .. r .. c2)
  NS_LOG_FUNCTION (this << c1 << c2 << v1 << v2 << r);
  return (v1 + ((v2 - v1) / (c2 - c1)) * (r - c1));
}

EmpiricalVariable::EmpiricalVariable ()
  : RandomVariable (EmpiricalVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
EmpiricalVariable::EmpiricalVariable (const RandomVariableBase &variable)
  : RandomVariable (variable)
{
  NS_LOG_FUNCTION (this << &variable);
}
void
EmpiricalVariable::CDF (double v, double c)
{
  NS_LOG_FUNCTION (this << v << c);
  EmpiricalVariableImpl *impl = dynamic_cast<EmpiricalVariableImpl *> (Peek ());
  NS_ASSERT (impl);
  impl->CDF (v, c);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// IntegerValue EmpiricalVariableImpl methods
/** \ingroup legacyrandom */
class IntEmpiricalVariableImpl : public EmpiricalVariableImpl
{
public:
  IntEmpiricalVariableImpl ();

  virtual RandomVariableBase* Copy (void) const;
  /**
   * \return An integer value from this empirical distribution
   */
  virtual uint32_t GetInteger ();
private:
  virtual double Interpolate (double, double, double, double, double);
};


IntEmpiricalVariableImpl::IntEmpiricalVariableImpl ()
{
  NS_LOG_FUNCTION (this);
}

uint32_t IntEmpiricalVariableImpl::GetInteger ()
{
  NS_LOG_FUNCTION (this);
  return (uint32_t)GetValue ();
}

RandomVariableBase* IntEmpiricalVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new IntEmpiricalVariableImpl (*this);
}

double IntEmpiricalVariableImpl::Interpolate (double c1, double c2,
                                              double v1, double v2, double r)
{ // Interpolate random value in range [v1..v2) based on [c1 .. r .. c2)
  NS_LOG_FUNCTION (this << c1 << c2 << v1 << v2 << r);
  return std::ceil (v1 + ((v2 - v1) / (c2 - c1)) * (r - c1));
}

IntEmpiricalVariable::IntEmpiricalVariable ()
  : EmpiricalVariable (IntEmpiricalVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// DeterministicVariableImpl
/** \ingroup legacyrandom */
class DeterministicVariableImpl : public RandomVariableBase
{

public:
  /**
   * \brief Constructor
   *
   * Creates a generator that returns successive elements of the d array
   * on successive calls to ::Value().  Note that the d pointer is copied
   * for use by the generator (shallow-copy), not its contents, so the
   * contents of the array d points to have to remain unchanged for the use
   * of DeterministicVariableImpl to be meaningful.
   * \param d Pointer to array of random values to return in sequence
   * \param c Number of values in the array
   */
  explicit DeterministicVariableImpl (double* d, uint32_t c);

  virtual ~DeterministicVariableImpl ();
  /**
   * \return The next value in the deterministic sequence
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;
private:
  uint32_t   count;
  uint32_t   next;
  double* data;
};

DeterministicVariableImpl::DeterministicVariableImpl (double* d, uint32_t c)
  : count (c),
    next (c),
    data (d)
{ // Nothing else needed
  NS_LOG_FUNCTION (this << d << c);
}

DeterministicVariableImpl::~DeterministicVariableImpl ()
{
  NS_LOG_FUNCTION (this);
}

double DeterministicVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  if (next == count)
    {
      next = 0;
    }
  return data[next++];
}

RandomVariableBase* DeterministicVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new DeterministicVariableImpl (*this);
}

DeterministicVariable::DeterministicVariable (double* d, uint32_t c)
  : RandomVariable (DeterministicVariableImpl (d, c))
{
  NS_LOG_FUNCTION (this << d << c);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// LogNormalVariableImpl
/** \ingroup legacyrandom */
class LogNormalVariableImpl : public RandomVariableBase
{
public:
  /**
   * \param mu mu parameter of the lognormal distribution
   * \param sigma sigma parameter of the lognormal distribution
   */
  LogNormalVariableImpl (double mu, double sigma);

  /**
   * \return A random value from this distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

private:
  double m_mu;
  double m_sigma;
};


RandomVariableBase* LogNormalVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new LogNormalVariableImpl (*this);
}

LogNormalVariableImpl::LogNormalVariableImpl (double mu, double sigma)
  : m_mu (mu),
    m_sigma (sigma)
{
  NS_LOG_FUNCTION (this << mu << sigma);
}

// The code from this function was adapted from the GNU Scientific
// Library 1.8:
/* randist/lognormal.c
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 James Theiler, Brian Gough
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
/* The lognormal distribution has the form

   p(x) dx = 1/(x * sqrt(2 pi sigma^2)) exp(-(ln(x) - zeta)^2/2 sigma^2) dx

   for x > 0. Lognormal random numbers are the exponentials of
   gaussian random numbers */
double
LogNormalVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  double u, v, r2, normal, z;

  do
    {
      /* choose x,y in uniform square (-1,-1) to (+1,+1) */

      u = -1 + 2 * generator->RandU01 ();
      v = -1 + 2 * generator->RandU01 ();

      /* see if it is in the unit circle */
      r2 = u * u + v * v;
    }
  while (r2 > 1.0 || r2 == 0);

  normal = u * std::sqrt (-2.0 * std::log (r2) / r2);

  z = std::exp (m_sigma * normal + m_mu);

  return z;
}

LogNormalVariable::LogNormalVariable (double mu, double sigma)
  : RandomVariable (LogNormalVariableImpl (mu, sigma))
{
  NS_LOG_FUNCTION (this << mu << sigma);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// GammaVariableImpl
/** \ingroup legacyrandom */
class GammaVariableImpl : public RandomVariableBase
{
public:
  /**
   * \param alpha alpha parameter of the gamma distribution
   * \param beta beta parameter of the gamma distribution
   */
  GammaVariableImpl (double alpha, double beta);

  /**
   * \return A random value from this distribution
   */
  virtual double GetValue ();

  /**
   * \return A random value from the gamma distribution with parameters alpha
   * and beta
   */
  double GetValue (double alpha, double beta);

  virtual RandomVariableBase* Copy (void) const;

private:
  double m_alpha;
  double m_beta;
  NormalVariable m_normal;
};


RandomVariableBase* GammaVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new GammaVariableImpl (m_alpha, m_beta);
}

GammaVariableImpl::GammaVariableImpl (double alpha, double beta)
  : m_alpha (alpha),
    m_beta (beta)
{
  NS_LOG_FUNCTION (this << alpha << beta);
}

double
GammaVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  return GetValue (m_alpha, m_beta);
}

/*
  The code for the following generator functions was adapted from ns-2
  tools/ranvar.cc

  Originally the algorithm was devised by Marsaglia in 2000:
  G. Marsaglia, W. W. Tsang: A simple method for gereating Gamma variables
  ACM Transactions on mathematical software, Vol. 26, No. 3, Sept. 2000

  The Gamma distribution density function has the form

                             x^(alpha-1) * exp(-x/beta)
        p(x; alpha, beta) = ----------------------------
                             beta^alpha * Gamma(alpha)

  for x > 0.
*/
double
GammaVariableImpl::GetValue (double alpha, double beta)
{
  NS_LOG_FUNCTION (this << alpha << beta);
  RngStream *generator = GetStream ();

  if (alpha < 1)
    {
      double u = generator->RandU01 ();
      return GetValue (1.0 + alpha, beta) * std::pow (u, 1.0 / alpha);
    }

  double x, v, u;
  double d = alpha - 1.0 / 3.0;
  double c = (1.0 / 3.0) / std::sqrt (d);

  while (1)
    {
      do
        {
          x = m_normal.GetValue ();
          v = 1.0 + c * x;
        }
      while (v <= 0);

      v = v * v * v;
      u = generator->RandU01 ();
      if (u < 1 - 0.0331 * x * x * x * x)
        {
          break;
        }
      if (std::log (u) < 0.5 * x * x + d * (1 - v + std::log (v)))
        {
          break;
        }
    }

  return beta * d * v;
}

GammaVariable::GammaVariable ()
  : RandomVariable (GammaVariableImpl (1.0, 1.0))
{
  NS_LOG_FUNCTION (this);
}

GammaVariable::GammaVariable (double alpha, double beta)
  : RandomVariable (GammaVariableImpl (alpha, beta))
{
  NS_LOG_FUNCTION (this << alpha << beta);
}

double GammaVariable::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return this->RandomVariable::GetValue ();
}

double GammaVariable::GetValue (double alpha, double beta) const
{
  NS_LOG_FUNCTION (this << alpha << beta);
  return ((GammaVariableImpl*)Peek ())->GetValue (alpha, beta);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ErlangVariableImpl

/** \ingroup legacyrandom */
class ErlangVariableImpl : public RandomVariableBase
{
public:
  /**
   * \param k k parameter of the Erlang distribution
   * \param lambda lambda parameter of the Erlang distribution
   */
  ErlangVariableImpl (unsigned int k, double lambda);

  /**
   * \return A random value from this distribution
   */
  virtual double GetValue ();

  /**
   * \return A random value from the Erlang distribution with parameters k and
   * lambda.
   */
  double GetValue (unsigned int k, double lambda);

  virtual RandomVariableBase* Copy (void) const;

private:
  unsigned int m_k;
  double m_lambda;
};


RandomVariableBase* ErlangVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ErlangVariableImpl (m_k, m_lambda);
}

ErlangVariableImpl::ErlangVariableImpl (unsigned int k, double lambda)
  : m_k (k),
    m_lambda (lambda)
{
  NS_LOG_FUNCTION (this << k << lambda);
}

double
ErlangVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  return GetValue (m_k, m_lambda);
}

/*
  The code for the following generator functions was adapted from ns-2
  tools/ranvar.cc

  The Erlang distribution density function has the form

                           x^(k-1) * exp(-x/lambda)
        p(x; k, lambda) = ---------------------------
                             lambda^k * (k-1)!

  for x > 0.
*/
double
ErlangVariableImpl::GetValue (unsigned int k, double lambda)
{
  /// \todo do not create a new 
  /// RNG stream every time the function is called !
  NS_LOG_FUNCTION (this << k << lambda);
  ExponentialVariable exponential (lambda);

  double result = 0;
  for (unsigned int i = 0; i < k; ++i)
    {
      result += exponential.GetValue ();
    }

  return result;
}

ErlangVariable::ErlangVariable ()
  : RandomVariable (ErlangVariableImpl (1, 1.0))
{
  NS_LOG_FUNCTION (this);
}

ErlangVariable::ErlangVariable (unsigned int k, double lambda)
  : RandomVariable (ErlangVariableImpl (k, lambda))
{
  NS_LOG_FUNCTION (this << k << lambda);
}

double ErlangVariable::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return this->RandomVariable::GetValue ();
}

double ErlangVariable::GetValue (unsigned int k, double lambda) const
{
  NS_LOG_FUNCTION (this << k << lambda);
  return ((ErlangVariableImpl*)Peek ())->GetValue (k, lambda);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// TriangularVariableImpl methods
/** \ingroup legacyrandom */
class TriangularVariableImpl : public RandomVariableBase
{
public:
  /**
   * Creates a triangle distribution random number generator in the
   * range [0.0 .. 1.0), with mean of 0.5
   */
  TriangularVariableImpl ();

  /**
   * Creates a triangle distribution random number generator with the specified
   * range
   * \param s Low end of the range
   * \param l High end of the range
   * \param mean mean of the distribution
   */
  TriangularVariableImpl (double s, double l, double mean);

  TriangularVariableImpl (const TriangularVariableImpl& c);

  /**
   * \return A value from this distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase*  Copy (void) const;

private:
  double m_min;
  double m_max;
  double m_mode;  // easier to work with the mode internally instead of the mean
                  // they are related by the simple: mean = (min+max+mode)/3
};

TriangularVariableImpl::TriangularVariableImpl ()
  : m_min (0),
    m_max (1),
    m_mode (0.5)
{
  NS_LOG_FUNCTION (this);
}

TriangularVariableImpl::TriangularVariableImpl (double s, double l, double mean)
  : m_min (s),
    m_max (l),
    m_mode (3.0 * mean - s - l)
{
  NS_LOG_FUNCTION (this << s << l << mean);
}

TriangularVariableImpl::TriangularVariableImpl (const TriangularVariableImpl& c)
  : RandomVariableBase (c),
    m_min (c.m_min),
    m_max (c.m_max),
    m_mode (c.m_mode)
{
  NS_LOG_FUNCTION (this << &c);
}

double TriangularVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();
  double u = generator->RandU01 ();
  if (u <= (m_mode - m_min) / (m_max - m_min) )
    {
      return m_min + std::sqrt (u * (m_max - m_min) * (m_mode - m_min) );
    }
  else
    {
      return m_max - std::sqrt ( (1 - u) * (m_max - m_min) * (m_max - m_mode) );
    }
}

RandomVariableBase* TriangularVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new TriangularVariableImpl (*this);
}

TriangularVariable::TriangularVariable ()
  : RandomVariable (TriangularVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}
TriangularVariable::TriangularVariable (double s, double l, double mean)
  : RandomVariable (TriangularVariableImpl (s,l,mean))
{
  NS_LOG_FUNCTION (this << s << l << mean);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ZipfVariableImpl
/** \ingroup legacyrandom */
class ZipfVariableImpl : public RandomVariableBase
{
public:
  /**
   * \param n the number of possible items
   * \param alpha the alpha parameter
   */
  ZipfVariableImpl (long n, double alpha);

  /**
   * \A zipf variable with N=1 and alpha=0
   */
  ZipfVariableImpl ();

  /**
   * \return A random value from this distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

private:
  long m_n;
  double m_alpha;
  double m_c; // the normalization constant
};


RandomVariableBase* ZipfVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ZipfVariableImpl (m_n, m_alpha);
}

ZipfVariableImpl::ZipfVariableImpl ()
  : m_n (1),
    m_alpha (0),
    m_c (1)
{
  NS_LOG_FUNCTION (this);
}


ZipfVariableImpl::ZipfVariableImpl (long n, double alpha)
  : m_n (n),
    m_alpha (alpha),
    m_c (0)
{
  // calculate the normalization constant c
  NS_LOG_FUNCTION (this << n << alpha);
  for (int i = 1; i <= n; i++)
    {
      m_c += (1.0 / std::pow ((double)i, alpha));
    }
  m_c = 1.0 / m_c;
}

double
ZipfVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();

  double u = generator->RandU01 ();
  double sum_prob = 0,zipf_value = 0;
  for (int i = 1; i <= m_n; i++)
    {
      sum_prob += m_c / std::pow ((double)i, m_alpha);
      if (sum_prob > u)
        {
          zipf_value = i;
          break;
        }
    }
  return zipf_value;
}

ZipfVariable::ZipfVariable ()
  : RandomVariable (ZipfVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}

ZipfVariable::ZipfVariable (long n, double alpha)
  : RandomVariable (ZipfVariableImpl (n, alpha))
{
  NS_LOG_FUNCTION (this << n << alpha);
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ZetaVariableImpl
/** \ingroup legacyrandom */
class ZetaVariableImpl : public RandomVariableBase
{
public:
  /**
   * \param alpha the alpha parameter
   */
  ZetaVariableImpl (double alpha);

  /**
   * \A zipf variable with alpha=1.1
   */
  ZetaVariableImpl ();

  /**
   * \return A random value from this distribution
   */
  virtual double GetValue ();
  virtual RandomVariableBase* Copy (void) const;

private:
  double m_alpha;
  double m_b; // just for calculus simplifications
};


RandomVariableBase* ZetaVariableImpl::Copy () const
{
  NS_LOG_FUNCTION (this);
  return new ZetaVariableImpl (m_alpha);
}

ZetaVariableImpl::ZetaVariableImpl ()
  : m_alpha (3.14),
    m_b (std::pow (2.0, 2.14))
{
  NS_LOG_FUNCTION (this);
}


ZetaVariableImpl::ZetaVariableImpl (double alpha)
  : m_alpha (alpha),
    m_b (std::pow (2.0, alpha - 1.0))
{
  NS_LOG_FUNCTION (this << alpha);
}

/*
 The code for the following generator functions is borrowed from:
 L. Devroye: Non-Uniform Random Variate Generation, Springer-Verlag, New York, 1986.
 page 551
 */
double
ZetaVariableImpl::GetValue ()
{
  NS_LOG_FUNCTION (this);
  RngStream *generator = GetStream ();

  double u, v;
  double X, T;
  double test;

  do
    {
      u = generator->RandU01 ();
      v = generator->RandU01 ();
      X = floor (std::pow (u, -1.0 / (m_alpha - 1.0)));
      T = std::pow (1.0 + 1.0 / X, m_alpha - 1.0);
      test = v * X * (T - 1.0) / (m_b - 1.0);
    }
  while ( test > (T / m_b) );

  return X;
}

ZetaVariable::ZetaVariable ()
  : RandomVariable (ZetaVariableImpl ())
{
  NS_LOG_FUNCTION (this);
}

ZetaVariable::ZetaVariable (double alpha)
  : RandomVariable (ZetaVariableImpl (alpha))
{
  NS_LOG_FUNCTION (this << alpha);
}


std::ostream & operator << (std::ostream &os, const RandomVariable &var)
{
  RandomVariableBase *base = var.Peek ();
  ConstantVariableImpl *constant = dynamic_cast<ConstantVariableImpl *> (base);
  if (constant != 0)
    {
      os << "Constant:" << constant->GetValue ();
      return os;
    }
  UniformVariableImpl *uniform = dynamic_cast<UniformVariableImpl *> (base);
  if (uniform != 0)
    {
      os << "Uniform:" << uniform->GetMin () << ":" << uniform->GetMax ();
      return os;
    }
  NormalVariableImpl *normal = dynamic_cast<NormalVariableImpl *> (base);
  if (normal != 0)
    {
      os << "Normal:" << normal->GetMean () << ":" << normal->GetVariance ();
      double bound = normal->GetBound ();
      if (bound != NormalVariableImpl::INFINITE_VALUE)
        {
          os << ":" << bound;
        }
      return os;
    }
  /// \todo support other distributions
  os.setstate (std::ios_base::badbit);
  return os;
}
std::istream & operator >> (std::istream &is, RandomVariable &var)
{
  std::string value;
  is >> value;
  std::string::size_type tmp;
  tmp = value.find (":");
  if (tmp == std::string::npos)
    {
      is.setstate (std::ios_base::badbit);
      return is;
    }
  std::string type = value.substr (0, tmp);
  value = value.substr (tmp + 1, value.npos);
  if (type == "Constant")
    {
      std::istringstream iss (value);
      double constant;
      iss >> constant;
      var = ConstantVariable (constant);
    }
  else if (type == "Uniform")
    {
      if (value.size () == 0)
        {
          var = UniformVariable ();
        }
      else
        {
          tmp = value.find (":");
          if (tmp == value.npos)
            {
              NS_FATAL_ERROR ("bad Uniform value: " << value);
            }
          std::istringstream issA (value.substr (0, tmp));
          std::istringstream issB (value.substr (tmp + 1, value.npos));
          double a, b;
          issA >> a;
          issB >> b;
          var = UniformVariable (a, b);
        }
    }
  else if (type == "Normal")
    {
      if (value.size () == 0)
        {
          var = NormalVariable ();
        }
      else
        {
          tmp = value.find (":");
          if (tmp == value.npos)
            {
              NS_FATAL_ERROR ("bad Normal value: " << value);
            }
          std::string::size_type tmp2;
          std::string sub = value.substr (tmp + 1, value.npos);
          tmp2 = sub.find (":");
          if (tmp2 == value.npos)
            {
              std::istringstream issA (value.substr (0, tmp));
              std::istringstream issB (sub);
              double a, b;
              issA >> a;
              issB >> b;
              var = NormalVariable (a, b);
            }
          else
            {
              std::istringstream issA (value.substr (0, tmp));
              std::istringstream issB (sub.substr (0, tmp2));
              std::istringstream issC (sub.substr (tmp2 + 1, value.npos));
              double a, b, c;
              issA >> a;
              issB >> b;
              issC >> c;
              var = NormalVariable (a, b, c);
            }
        }
    }
  else
    {
      NS_FATAL_ERROR ("RandomVariable deserialization not implemented for " << type);
      /// \todo support other distributions.
    }
  return is;
}

} // namespace ns3
