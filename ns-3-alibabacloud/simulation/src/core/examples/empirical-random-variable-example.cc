/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/command-line.h"
#include "ns3/random-variable-stream.h"
#include "ns3/histogram.h"
#include "ns3/ptr.h"

#include <iomanip>
#include <iostream>
#include <map>


/**
 * \file
 * \ingroup core-examples randomvariable
 * \defgroup empirical-rng-example Core example: Empirical random variables use.
 * 
 * Example program illustrating use of ns3::EmpiricalRandomVariable
 *
 * This example illustrates
 *
 *   * Creating an EmpiricalRandomVariable instance.
 *   * Switching the mode.
 *   * Using the sampling mode
 *   * Switching modes
 *   * Using the interpolating mode
 *
 * Consult the ns-3 manual for more information about the use of the
 * random number generator
 */


using namespace ns3;

/**
 * \ingroup empirical-rng-example
 * 
 * \brief Sample the random variable only once.
 * \param mode Rng mode (Normal or Antithetic).
 * \param erv The empirical random variable.
 */
void
RunSingleSample (std::string mode, Ptr<EmpiricalRandomVariable> erv)
{
  std::cout << "------------------------------" << std::endl;
  std::cout << "Sampling " << mode << std::endl;
  
  std::cout << std::endl;
  std::cout << "Binned sample" << std::endl;
  double value = erv->GetValue ();
  std::cout << "Binned sample: " << value << std::endl;
  std::cout << std::endl;

  std::cout << "Interpolated sample" << std::endl;
  erv->SetInterpolate (true);
  value = erv->GetValue ();
  std::cout << "Interpolated sample:" << value << std::endl;
  erv->SetInterpolate (false);
}

/**
 * \ingroup empirical-rng-example
 * 
 * \brief Prints a stat line.
 * \param value The value to print.
 * \param count The number of times that value has been sampled.
 * \param n The total number of random values sampled.
 */
void
PrintStatsLine (const double value, const long count, const long n)
{
  std::cout << std::fixed << std::setprecision (3)
            << std::setw (10) << std::right << value
            << std::setw (10) << std::right << count
            << std::setw (10) << std::right
            << count / static_cast<double> (n) * 100.0
            << std::endl;
}

/**
 * \ingroup empirical-rng-example
 * 
 * \brief Prints the summary.
 * \param sum The number of sampled values.
 * \param n The total number of random values to be drawn.
 * \param weighted The average of the sample.
 * \param expected The expected average of the sample.
 */
void
PrintSummary (long sum, long n, double weighted, double expected)
{
  std::cout << std::endl;
  std::cout << "                      --------" << std::endl;
  std::cout << "              Total "
            << std::setprecision (3) << std::fixed
            << std::setw (10) << std::right
            << sum / static_cast<double> (n) * 100.0
            << std::endl;
  std::cout << "            Average "
            << std::setprecision (3) << std::fixed
            << std::setw (6) << std::right << weighted / n
            << std::endl;
  std::cout << "           Expected "
            << std::setprecision (3) << std::fixed
            << std::setw (6) << std::right << expected
            << std::endl
            << std::endl;
}

/**
 * \ingroup empirical-rng-example
 * 
 * \brief Sample the random variable.
 * \param mode Rng mode (Normal or Antithetic).
 * \param erv The empirical random variable.
 * \param n Number of samples to draw.
 */
void
RunBothModes (std::string mode, Ptr<EmpiricalRandomVariable> erv, long n)
{
  std::cout << std::endl;
  std::cout << "Sampling " << mode << std::endl;
  std::map <double, int> counts;
  counts[0] = 0;
  for (long i = 0; i < n; ++i)
    {
      ++counts[erv->GetValue ()];
    }
  long sum = 0;
  double weighted = 0;
  std::cout << std::endl;
  std::cout << "     Value    Counts         %" << std::endl;
  std::cout << "----------  --------  --------" << std::endl;
  for (auto c : counts)
    {
      long count = c.second;
      double value = c.first;
      sum += count;
      weighted +=  value * count;
      PrintStatsLine (value, count, n);
    }
  PrintSummary (sum, n, weighted, 8.75);

  std::cout << "Interpolating " << mode << std::endl;
  erv->SetInterpolate (true);
  Histogram h (0.5);
  for (long i = 0; i < n; ++i)
    {
      h.AddValue (erv->GetValue ());
      // This could also be expressed as
      //   h.AddValue (erv->Interpolate ());
    }
  erv->SetInterpolate (false);
  sum = 0;
  weighted = 0;
  std::cout << std::endl;
  std::cout << " Bin Start    Counts         %" << std::endl;
  std::cout << "----------  --------  --------" << std::endl;
  for (uint32_t i = 0; i < h.GetNBins (); ++i)
    {
      long count = h.GetBinCount (i);
      double start  = h.GetBinStart (i);
      double value  = start + h.GetBinWidth (i) / 2.;
      sum += count;
      weighted += count * value;
      PrintStatsLine (start, count, n);
    }
  PrintSummary (sum, n, weighted, 6.25);
}


int main (int argc, char *argv[])
{
  long n = 1000000;
  bool disableAnti = false;
  bool single = false;
  CommandLine cmd;
  cmd.AddValue ("count", "how many draws to make from the rng", n);
  cmd.AddValue ("antithetic", "disable antithetic sampling", disableAnti);
  cmd.AddValue ("single", "sample a single time", single);
  cmd.Parse (argc, argv);
  std::cout << std::endl;
  std::cout << cmd.GetName () << std::endl;
  if (!single)
    {
      std::cout << "Sample count: " << n << std::endl;
    }
  else
    {
      std::cout << "Sampling a single time" << std::endl;
    }
  if (disableAnti)
    {
      std::cout << "Antithetic sampling disabled" << std::endl;
    }
  
  // Create the ERV in sampling mode
  Ptr<EmpiricalRandomVariable> erv = CreateObject<EmpiricalRandomVariable> ();
  erv->SetInterpolate (false);
  erv->CDF ( 0.0,  0.0);
  erv->CDF ( 5.0,  0.25);
  erv->CDF (10.0,  1.0);

  if (single)
    {
      RunSingleSample ("normal", erv);
      if (!disableAnti)
        {
          std::cout << std::endl;
          std::cout << "Antithetic" << std::endl;
          erv->SetAntithetic (true);
          RunSingleSample ("antithetic", erv);
          erv->SetAntithetic (false);
        }

      std::cout << std::endl;
      return 0;
    }
  
  RunBothModes ("normal", erv, n);

  if (!disableAnti)
    {
      erv->SetAntithetic (true);
      RunBothModes ("antithetic", erv, n);
      erv->SetAntithetic (false);
    }
  
  return 0;
}
