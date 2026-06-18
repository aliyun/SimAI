/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Pavel Boyko <boyko@iitp.ru>
 * Corrections and extensions: Timo Bingmann <tbns@idlebox.net>
 */

#ifndef AVERAGE_H
#define AVERAGE_H
#include <cmath>
#include <ostream>
#include <limits>
#include <stdint.h>
#include "ns3/basic-data-calculators.h"

namespace ns3 {

/**
 * \ingroup stats
 *
 * Simple average, min, max and std. deviation calculator
 *
 */

template <typename T = double>
class Average
{
public:
  Average ()
    : m_size (0), m_min (std::numeric_limits<T>::max ()), m_max (0)
  {
  }

  /**
   * Add new sample
   * \param x The sample
   */
  void Update (T const & x)
  {
    // Give the variance calculator the next value.
    m_varianceCalculator.Update (x);

    m_min = std::min (x, m_min);
    m_max = std::max (x, m_max);
    m_size++;
  }
  /// Reset statistics
  void Reset ()
  {
    m_varianceCalculator.Reset ();

    m_size = 0;
    m_min = std::numeric_limits<T>::max ();
    m_max = 0;
  }

  // Sample statistics
  /**
   * Sample size
   * \return the sample size
   */
  uint32_t Count   () const { return m_size; }
  /**
   * Sample minimum
   * \return the minimum of the sample
   */
  T        Min     () const { return m_min; }
  /**
   * Sample maximum
   * \return the maximum of the sample
   */
  T        Max     () const { return m_max; }
  /**
   * Sample average
   * \return the average of the sample
   */
  double   Avg     () const { return m_varianceCalculator.getMean ();}
  /**
   * Sample estimate of mean, alias to Avg
   * \return the average of the sample
   */
  double   Mean    () const { return Avg (); }
  /**
   * Sample unbiased nbiased estimate of variance
   * \return the unbiased nbiased estimate of variance
   */
  double   Var     () const { return m_varianceCalculator.getVariance ();}
  /**
   * Sample standard deviation
   * \return the standard deviation
   */
  double   Stddev  () const { return std::sqrt (Var ()); }

  /**
   * \name Error of the mean estimates
   *
   * @{
   */
  /**
   * \brief Margin of error of the mean for 90% confidence level
   *
   * Note that estimates are valid for 
   *   - uncorrelated measurements, 
   *   - normal distribution and
   *   - large enough sample size.
   *
   * \returns Margin of error of the mean for 90% confidence level
   */
  double   Error90 () const { return 1.645 * std::sqrt (Var () / Count ()); }
  /**
   * \brief Margin of error of the mean for 95% confidence level 
   *
   * Note that estimates are valid for 
   *   - uncorrelated measurements, 
   *   - normal distribution and
   *   - large enough sample size.
   *
   * \returns Margin of error of the mean for 95% confidence level
   */
  double   Error95 () const { return 1.960 * std::sqrt (Var () / Count ()); }
  /**
   * \brief Margin of error of the mean for 99% confidence level 
   *
   * Note that estimates are valid for 
   *   - uncorrelated measurements, 
   *   - normal distribution and
   *   - large enough sample size.
   *
   * \returns Margin of error of the mean for 99% confidence level
   *
   */
  double   Error99 () const { return 2.576 * std::sqrt (Var () / Count ()); }
  /**@}*/

private:
  uint32_t m_size; //!< Number of sampled data.
  T m_min; //!< Minimum value observed.
  T m_max; //!< Maximum value observed.
  MinMaxAvgTotalCalculator<double> m_varianceCalculator; //!< Variance calculator.
};

/**
 * Print avg (err) [min, max]
 * \param os The output stream
 * \param x The Average value to print
 * \return the ouput stream.
 */
template <typename T>
std::ostream & operator<< (std::ostream & os, Average<T> const & x)
{
  if (x.Count () != 0)
    os << x.Avg () << " (" << x.Stddev () << ") [" << x.Min () << ", " << x.Max () << "]";
  else
    os << "NA";  // not available
  return os;
}
}
#endif /* AVERAGE_H */
