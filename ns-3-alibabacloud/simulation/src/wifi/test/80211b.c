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

/*
 * This program is used to generate plots found in the paper
 * G. Pei and Tom Henderson, "Validation of ns-3 802.11b PHY model",
 * available online at http://www.nsnam.org/~pei/80211b.pdf
 *
 * It can be compiled as a C program and relies on a library installation of
 * the GNU Scientific Library (gsl).  To compile:
 *   gcc 80211b.c -o 80211b -lm -lgsl -lgslcblas
 *
 * The executable output should be redirected into a text file 80211b.txt
 *   ./80211b > 80211b.txt
 *
 * Then gnuplot can load the associated plot file which references 80211b.txt:
 *   gnuplot 80211b.plt
 */

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf_bessel.h>

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#define WLAN_SIR_perfect 10.0 // if SIR > 10dB, perfect reception
#define WLAN_SIR_impossible 0.1 // if SIR < -10dB, impossible to receive

/**
 * \ingroup wifi
 * \defgroup wifi-test wifi module tests
 */


/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief fn_parameter_t structure
 */
typedef struct fn_parameter_t
{
  double beta; ///< beta
  double n; ///< n
} fn_parameters;

double QFunction (double x)
{
  return 0.5 * erfc (x / sqrt (2.0));
}

double f (double x, void * params)
{
  double beta = ((fn_parameters *) params)->beta;
  double n = ((fn_parameters *) params)->n;
  double f = pow ( 2 * gsl_cdf_ugaussian_P (x + beta) - 1, n - 1) * exp (-x * x / 2.0) / sqrt (2.0 * M_PI);
  return f;
}

double p_e2 (double e2)
{
  double sep;
  double error;
  fn_parameters params;
  params.beta = sqrt (2.0 * e2);
  params.n = 8.0;
  gsl_integration_workspace* w = gsl_integration_workspace_alloc (1000);
  gsl_function F;
  F.function = &f;
  F.params = &params;
  gsl_integration_qagiu (&F, -params.beta, 0, 1e-7, 1000, w, &sep, &error);
  gsl_integration_workspace_free (w);
  if (error == 0.0)
    {
      sep = 1.0;
    }
  return 1.0 - sep;
}

double p_e1 (double e1)
{
  return 1.0 - pow ( 1.0 - p_e2 (e1 / 2.0), 2.0);
}

double DbToNoneDb (double x)
{
  return pow (10.0, x / 10.0);
}

double NoneDbToDb (double x)
{
  return 10.0 * log10 (x);
}

double DQPSKFunction (double x)
{
  double pi = acos (-1.0);
  return ( (sqrt (2.0) + 1.0) / sqrt (8.0 * pi * sqrt (2.0))) * (1.0 / sqrt (x)) * exp ( -(2.0 - sqrt (2.0)) * x);
}

double Get80211bDsssDbpskBerIeee (double EcNc)
{
  double ber;
  if (EcNc > WLAN_SIR_perfect)
    {
      ber = 0;
    }
  else if (EcNc < WLAN_SIR_impossible)
    {
      ber =  0.5;
    }
  else
    {
      ber = min (QFunction (sqrt (11.0 * EcNc)),0.5);
    }
  return ber;
}

double Get80211bDsssDbpskBer (double sinr)
{
  double EbN0 = sinr * 22000000.0 / 1000000.0;
  double ber = 0.5 * exp (-EbN0);
  return ber;
}

double Get80211bDsssDqpskBerIeee (double EcNc)
{
  double ber;
  if (EcNc > WLAN_SIR_perfect)
    {
      ber = 0;
    }
  else if (EcNc < WLAN_SIR_impossible)
    {
      ber = 0.5;
    }
  else
    {
      ber = min (QFunction (sqrt (5.5 * EcNc)),0.5);
    }
  return ber;
}

double Get80211bDsssDqpskBer (double sinr)
{
// 2 bits per symbol, 1 MSPS
  double EbN0 = sinr * 22000000.0 / 1000000.0 / 2.0;
  double ber = DQPSKFunction (EbN0);
  return ber;
}

double Get80211bDsssDqpskCCK5_5BerIeee (double EcNc)
{
  double ber;
  if (EcNc > WLAN_SIR_perfect)
    {
      ber = 0.0;
    }
  else if (EcNc < WLAN_SIR_impossible)
    {
      ber = 0.5;
    }
  else
    {
      double pew = 14.0 * QFunction (sqrt (EcNc * 8.0)) + QFunction (sqrt (EcNc * 16.0));
      pew = min (pew, 0.99999);
      ber = 8.0 / 15.0 * pew;
    }
  return ber;
}

double Get80211bDsssDqpskCCK11BerIeee (double EcNc)
{
  double ber;
  if (EcNc > WLAN_SIR_perfect)
    {
      ber = 0.0;
    }
  else if (EcNc < WLAN_SIR_impossible)
    {
      ber = 0.5;
    }
  else
    {
      double pew = 24.0 * QFunction (sqrt (EcNc * 4.0))  + 16.0 * QFunction (sqrt (EcNc * 6.0))  + 174.0 * QFunction (sqrt (EcNc * 8.0)) + 16.0 * QFunction (sqrt (EcNc * 10.0)) + 24.0 * QFunction (sqrt (EcNc * 12.0)) + QFunction (sqrt (EcNc * 16.0));
      pew = min (pew, 0.99999);
      ber = 128.0 / 255.0 * pew;
    }
  return ber;
}

int main (int argc, char * argv[])
{
  double rss, sinr;
  double totalPkt = 200.0;
//double noise = 1.552058;  // (dB) this noise figure value corresponds to
  // -99 dBm noise floor reported in CMU paper
  double noise = 7;  // (dB) this noise figure value corresponds to the
                     // default in YansWifiPhy, and matches CMU testbed results
  double EcNc, EbN01, EbN02, EbN05, EbN011;
  double ieee1,ieee2,ieee5,ieee11;
  double numBits = (1024. + 40. + 14.) * 8.;
  double dbpsk,dqpsk,cck16,cck256,sepcck16,sepcck256;
  noise = DbToNoneDb (noise) * 1.3803e-23 * 290.0 * 22000000;
  for (rss = -102.0; rss <= -80.0; rss += 0.1)
    {
      sinr   = DbToNoneDb (rss) / 1000.0 / noise;
      EcNc   = sinr * 22000000.0 / 11000000.0; // IEEE sir
      EbN01  = sinr * 22000000.0 / 1000000.0;
      // 2 bits per symbol, 1 MSPS
      EbN02  = sinr * 22000000.0 / 1000000.0 / 2.0;
      EbN05  = sinr * 22000000.0 / 1375000.0 / 4.0;
      EbN011 = sinr * 22000000.0 / 1375000.0 / 8.0;
      // 1=rss, 2=EcNc, 3=EbN01, 4=EbN02, 5=EBN05, 6=EbN011
      printf ("%g %g %g %g %g %g ", rss, NoneDbToDb (EcNc),
              NoneDbToDb (EbN01),NoneDbToDb (EbN02),
              NoneDbToDb (EbN05),NoneDbToDb (EbN011));
      ieee1  = Get80211bDsssDbpskBerIeee (EcNc);
      ieee2  = Get80211bDsssDqpskBerIeee (EcNc);
      ieee5  = Get80211bDsssDqpskCCK5_5BerIeee (EcNc);
      ieee11 = Get80211bDsssDqpskCCK11BerIeee (EcNc);
      // 7=ber_ieee1, 8=ber_ieee2, 9=ber_ieee5, 10=ber_ieee11
      printf (" %g %g %g %g ", ieee1, ieee2,ieee5,ieee11);
      ieee1  = totalPkt * pow (1 - ieee1,  numBits);
      ieee2  = totalPkt * pow (1 - ieee2,  numBits);
      ieee5  = totalPkt * pow (1 - ieee5,  numBits);
      ieee11 = totalPkt * pow (1 - ieee11, numBits);
      // 11=pkt_ieee1, 12=pkt_ieee2, 13=pkt_ieee5, 14=pkt_ieee11
      printf (" %g %g %g %g ", ieee1, ieee2,ieee5,ieee11);
      dbpsk  = Get80211bDsssDbpskBer (sinr);
      dqpsk  = Get80211bDsssDqpskBer (sinr);
      cck16  = max (0, 8.0 / 15.0 * p_e2 (4.0 * EbN05 / 2.0));
      cck256 = max (0, 128.0 / 255.0 * p_e1 (8.0 * EbN011 / 2.0));
      // 15=ber_dbpsk, 16=ber_dqpsk, 17=ber_cck16, 18=ber_cck256
      printf (" %g %g %g %g ", dbpsk, dqpsk,cck16,cck256);
      dbpsk     = totalPkt * pow (1 - dbpsk,numBits);
      dqpsk     = totalPkt * pow (1 - dqpsk,numBits);
      sepcck16  = p_e2 (4.0 * EbN05 / 2.0);
      sepcck256 = p_e1 (8.0 * EbN011 / 2.0);
      cck16     = totalPkt * pow (1.0 - sepcck16,numBits / 4.0);
      cck256    = totalPkt * pow (1.0 - sepcck256,numBits / 8.0);
      // 19=pkt_dbpsk, 20=pkt_dqpsk, 21=pkt_cck16, 22=pkt_cck256
      printf (" %g %g %g %g ", dbpsk, dqpsk,cck16,cck256);
      // 23=sinr
      printf (" %g \n",NoneDbToDb (sinr));
    }
  return 0;
}
