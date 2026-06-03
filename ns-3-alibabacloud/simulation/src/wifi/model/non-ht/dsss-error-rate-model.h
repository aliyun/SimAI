/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 The Boeing Company
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
 * Author: Gary Pei <guangyu.pei@boeing.com>
 */

#ifndef DSSS_ERROR_RATE_MODEL_H
#define DSSS_ERROR_RATE_MODEL_H

#ifdef HAVE_GSL
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_bessel.h>
#endif

namespace ns3 {

#ifdef HAVE_GSL
/**
 * Structure for integral function parameters
 */
typedef struct FunctionParameterType
{
  double beta; ///< Beta parameter
  double n;    ///< n parameter
} FunctionParameters;

/**
 * Integral function using GSL library
 *
 * \param x the input x variable
 * \param params a pointer to FunctionParameters struct
 *
 * \return the integral function
 */
double IntegralFunction (double x, void *params);
#endif

/**
 * \brief an implementation of DSSS error rate model
 * \ingroup wifi
 *
 * The 802.11b modulations:
 *    - 1 Mbps mode is based on DBPSK. BER is from equation 5.2-69 from John G. Proakis
 *      Digital Communications, 2001 edition
 *    - 2 Mbps model is based on DQPSK. Equation 8 from "Tight bounds and accurate
 *      approximations for DQPSK transmission bit error rate", G. Ferrari and G.E. Corazza
 *      ELECTRONICS LETTERS, 40(20):1284-1285, September 2004
 *    - 5.5 Mbps and 11 Mbps are based on equations (18) and (17) from "Properties and
 *      performance of the IEEE 802.11b complementarycode-key signal sets",
 *      Michael B. Pursley and Thomas C. Royster. IEEE TRANSACTIONS ON COMMUNICATIONS,
 *      57(2):440-449, February 2009.
 *
 *  This model is designed to run with highest accuracy using the GNU
 *  Scientific Library (GSL), but if GSL is not installed on the platform,
 *  will fall back to (slightly less accurate) Matlab-derived models for
 *  the CCK modulation types.
 *
 *  More detailed description and validation can be found in
 *      http://www.nsnam.org/~pei/80211b.pdf
 */
class DsssErrorRateModel
{
public:
  /**
   * A function DQPSK
   *
   * \param x the input variable
   *
   * \return DQPSK (x)
   */
  static double DqpskFunction (double x);
  /**
   * Return the chunk success rate of the differential BPSK.
   *
   * \param sinr the SINR ratio (not dB) of the chunk
   * \param nbits the size of the chunk
   *
   * \return the chunk success rate of the differential BPSK
   */
  static double GetDsssDbpskSuccessRate (double sinr, uint64_t nbits);
  /**
   * Return the chunk success rate of the differential encoded QPSK.
   *
   * \param sinr the SINR ratio (not dB) of the chunk
   * \param nbits the size of the chunk
   *
   * \return the chunk success rate of the differential encoded QPSK.
   */
  static double GetDsssDqpskSuccessRate (double sinr,uint64_t nbits);
  /**
   * Return the chunk success rate of the differential encoded QPSK for
   * 5.5Mbps data rate.
   *
   * \param sinr the SINR ratio (not dB) of the chunk
   * \param nbits the size of the chunk
   *
   * \return the chunk success rate of the differential encoded QPSK for
   */
  static double GetDsssDqpskCck5_5SuccessRate (double sinr,uint64_t nbits);
  /**
   * Return the chunk success rate of the differential encoded QPSK for
   * 11Mbps data rate.
   *
   * \param sinr the SINR ratio (not dB) of the chunk
   * \param nbits the size of the chunk
   *
   * \return the chunk success rate of the differential encoded QPSK for
   */
  static double GetDsssDqpskCck11SuccessRate (double sinr,uint64_t nbits);
#ifdef HAVE_GSL
  static double SymbolErrorProb16Cck (double e2); /// equation (18) in Pursley's paper
  static double SymbolErrorProb256Cck (double e1); /// equation (17) in Pursley's paper
#else


protected:
  /// WLAN perfect
  static const double WLAN_SIR_PERFECT;
  /// WLAN impossible
  static const double WLAN_SIR_IMPOSSIBLE;
#endif
};

} //namespace ns3

#endif /* DSSS_ERROR_RATE_MODEL_H */
