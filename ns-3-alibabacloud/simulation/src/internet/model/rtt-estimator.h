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
//

// Georgia Tech Network Simulator - Round Trip Time Estimation Class
// George F. Riley.  Georgia Tech, Spring 2002


#ifndef RTT_ESTIMATOR_H
#define RTT_ESTIMATOR_H

#include "ns3/nstime.h"
#include "ns3/object.h"

namespace ns3 {

/**
 * \ingroup tcp
 *
 * \brief Base class for all RTT Estimators
 *
 * The RTT Estimator class computes an estimate of the round trip time
 * observed in a series of Time measurements.  The estimate is provided in
 * the form of an estimate and a sample variation.  Subclasses can implement
 * different algorithms to provide values for the estimate and variation.  
 */
class RttEstimator : public Object {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  RttEstimator();
  /**
   * \brief Copy constructor
   * \param r the object to copy
   */
  RttEstimator (const RttEstimator& r);

  virtual ~RttEstimator();

  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief Add a new measurement to the estimator. Pure virtual function.
   * \param t the new RTT measure.
   */
  virtual void  Measurement (Time t) = 0;

  /**
   * \brief Copy object (including current internal state)
   * \returns a copy of itself
   */
  virtual Ptr<RttEstimator> Copy () const = 0;

  /**
   * \brief Resets the estimation to its initial state.
   */
  virtual void Reset ();

  /**
   * \brief gets the RTT estimate.
   * \return The RTT estimate.
   */
  Time GetEstimate (void) const;

  /**
   * Note that this is not a formal statistical variance; it has the
   * the same units as the estimate.  Mean deviation or standard deviation 
   * are example quantities that could be provided here.
   *
   * \brief gets the RTT estimate variation.
   * \return The RTT estimate variation.
   */
  Time GetVariation (void) const;

  /**
   * \brief gets the number of samples used in the estimates
   * \return the number of samples used in the estimates
   */
  uint32_t GetNSamples (void) const;

private:
  Time m_initialEstimatedRtt; //!< Initial RTT estimation

protected:
  Time         m_estimatedRtt;            //!< Current estimate
  Time         m_estimatedVariation;   //!< Current estimate variation
  uint32_t     m_nSamples;                //!< Number of samples
};

/**
 * \ingroup tcp
 *
 * \brief The "Mean--Deviation" RTT estimator, as discussed by Van Jacobson
 *
 * This class implements the "Mean--Deviation" RTT estimator, as discussed
 * by Van Jacobson and Michael J. Karels, in
 * "Congestion Avoidance and Control", SIGCOMM 88, Appendix A
 *
 * The default values for the gain (alpha and beta) are set as documented
 * in RFC 6298.
 *
 */
class RttMeanDeviation : public RttEstimator {
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  RttMeanDeviation ();

  /**
   * \brief Copy constructor
   * \param r the object to copy
   */
  RttMeanDeviation (const RttMeanDeviation& r);

  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief Add a new measurement to the estimator.
   * \param measure the new RTT measure.
   */
  void Measurement (Time measure);

  Ptr<RttEstimator> Copy () const;

  /**
   * \brief Resets the estimator.
   */
  void Reset ();

private:
  /** 
   * Utility function to check for possible conversion
   * of a double value (0 < value < 1) to a reciprocal power of two
   *
   * Values of 1/32, 1/16, 1/8, 1/4, and 1/2 (i.e., within the possible
   * range of experimentation for this estimator) are supported.
   * 
   * \param val value to check 
   * \return log base 2 (1/val) if reciprocal power of 2, or zero if not
   */
  uint32_t CheckForReciprocalPowerOfTwo (double val) const;
  /**
   * Method to update the rtt and variation estimates using integer
   * arithmetic, used when the values of Alpha and Beta support the
   * integer conversion.
   *
   * \param m time measurement
   * \param rttShift value corresponding to log base 2 (1/alpha)
   * \param variationShift value corresponding to log base 2 (1/beta)
   */
  void IntegerUpdate (Time m, uint32_t rttShift, uint32_t variationShift);
  /**
   * Method to update the rtt and variation estimates using floating
   * point arithmetic, used when the values of Alpha and Beta are not
   * both a reciprocal power of two.
   *
   * \param m time measurement
   */
  void FloatingPointUpdate (Time m);
  double       m_alpha;       //!< Filter gain for average
  double       m_beta;        //!< Filter gain for variation

};

} // namespace ns3

#endif /* RTT_ESTIMATOR_H */
