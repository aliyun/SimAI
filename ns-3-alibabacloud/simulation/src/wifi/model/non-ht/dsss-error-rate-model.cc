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

#include "ns3/log.h"
#include "dsss-error-rate-model.h"
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsssErrorRateModel");

#ifndef HAVE_GSL
const double DsssErrorRateModel::WLAN_SIR_PERFECT = 10.0;
const double DsssErrorRateModel::WLAN_SIR_IMPOSSIBLE = 0.1;
#endif

double
DsssErrorRateModel::DqpskFunction (double x)
{
  NS_LOG_FUNCTION_NOARGS ();
  return ((std::sqrt (2.0) + 1.0) / std::sqrt (8.0 * M_PI * std::sqrt (2.0)))
         * (1.0 / std::sqrt (x)) * std::exp ( -(2.0 - std::sqrt (2.0)) * x);
}

double
DsssErrorRateModel::GetDsssDbpskSuccessRate (double sinr, uint64_t nbits)
{
  NS_LOG_FUNCTION_NOARGS ();
  double EbN0 = sinr * 22000000.0 / 1000000.0; //1 bit per symbol with 1 MSPS
  double ber = 0.5 * std::exp (-EbN0);
  return std::pow ((1.0 - ber), static_cast<double> (nbits));
}

double
DsssErrorRateModel::GetDsssDqpskSuccessRate (double sinr, uint64_t nbits)
{
  NS_LOG_FUNCTION_NOARGS ();
  double EbN0 = sinr * 22000000.0 / 1000000.0 / 2.0; //2 bits per symbol, 1 MSPS
  double ber = DqpskFunction (EbN0);
  return std::pow ((1.0 - ber), static_cast<double> (nbits));
}

double
DsssErrorRateModel::GetDsssDqpskCck5_5SuccessRate (double sinr, uint64_t nbits)
{
  NS_LOG_FUNCTION_NOARGS ();
#ifdef HAVE_GSL
  //symbol error probability
  double EbN0 = sinr * 22000000.0 / 1375000.0 / 4.0;
  double sep = SymbolErrorProb16Cck (4.0 * EbN0 / 2.0);
  return std::min (1.0, std::pow (1.0 - sep, nbits / 4.0));
#else
  NS_LOG_WARN ("Running a 802.11b CCK Matlab model less accurate than GSL model");
  //The Matlab model
  double ber;
  if (sinr > WLAN_SIR_PERFECT)
    {
      ber = 0.0;
    }
  else if (sinr < WLAN_SIR_IMPOSSIBLE)
    {
      ber = 0.5;
    }
  else
    {
      //fitprops.coeff from Matlab berfit
      double a1 = 5.3681634344056195e-001;
      double a2 = 3.3092430025608586e-003;
      double a3 = 4.1654372361004000e-001;
      double a4 = 1.0288981434358866e+000;
      ber = a1 * std::exp (-std::pow ((sinr - a2) / a3, a4));
    }
  return std::min (1.0, std::pow ((1.0 - ber), static_cast<double> (nbits)));
#endif
}

double
DsssErrorRateModel::GetDsssDqpskCck11SuccessRate (double sinr, uint64_t nbits)
{
  NS_LOG_FUNCTION_NOARGS ();
#ifdef HAVE_GSL
  NS_LOG_DEBUG ("GSL enabled ");
  //symbol error probability
  double EbN0 = sinr * 22000000.0 / 1375000.0 / 8.0;
  double sep = SymbolErrorProb256Cck (8.0 * EbN0 / 2.0);
  return std::min (1.0, std::pow (1.0 - sep, nbits / 8.0));
#else
  NS_LOG_WARN ("Running a 802.11b CCK Matlab model less accurate than GSL model");
  //The Matlab model
  double ber;
  if (sinr > WLAN_SIR_PERFECT)
    {
      ber = 0.0;
    }
  else if (sinr < WLAN_SIR_IMPOSSIBLE)
    {
      ber = 0.5;
    }
  else
    {
      //fitprops.coeff from Matlab berfit
      double a1 = 7.9056742265333456e-003;
      double a2 = -1.8397449399176360e-001;
      double a3 = 1.0740689468707241e+000;
      double a4 = 1.0523316904502553e+000;
      double a5 = 3.0552298746496687e-001;
      double a6 = 2.2032715128698435e+000;
      ber =  (a1 * sinr * sinr + a2 * sinr + a3) / (sinr * sinr * sinr + a4 * sinr * sinr + a5 * sinr + a6);
    }
  return std::min (1.0, std::pow ((1.0 - ber), static_cast<double> (nbits)));
#endif
}

#ifdef HAVE_GSL
double
IntegralFunction (double x, void *params)
{
  double beta = ((FunctionParameters *) params)->beta;
  double n = ((FunctionParameters *) params)->n;
  double IntegralFunction = std::pow (2 * gsl_cdf_ugaussian_P (x + beta) - 1, n - 1)
    * std::exp (-x * x / 2.0) / std::sqrt (2.0 * M_PI);
  return IntegralFunction;
}

double
DsssErrorRateModel::SymbolErrorProb16Cck (double e2)
{
  double sep;
  double error;

  FunctionParameters params;
  params.beta = std::sqrt (2.0 * e2);
  params.n = 8.0;

  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

  gsl_function F;
  F.function = &IntegralFunction;
  F.params = &params;

  gsl_integration_qagiu (&F,-params.beta, 0, 1e-7, 1000, w, &sep, &error);
  gsl_integration_workspace_free (w);
  if (error == 0.0)
    {
      sep = 1.0;
    }

  return 1.0 - sep;
}

double DsssErrorRateModel::SymbolErrorProb256Cck (double e1)
{
  return 1.0 - std::pow (1.0 - SymbolErrorProb16Cck (e1 / 2.0), 2.0);
}

#endif

} //namespace ns3
