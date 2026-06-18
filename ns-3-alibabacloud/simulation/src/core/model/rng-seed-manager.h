/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Mathieu Lacage
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
 */

#ifndef RNG_SEED_MANAGER_H
#define RNG_SEED_MANAGER_H

#include <stdint.h>

/**
 * \file
 * \ingroup randomvariable
 * ns3::RngSeedManager declaration.
 */

namespace ns3 {

/**
 * \ingroup randomvariable
 *
 * Manage the seed number and run number of the underlying
 * random number generator, and automatic assignment of stream numbers.
 */
class RngSeedManager
{
public:
  /**
   * \brief Set the seed.
   *
   * This sets the global initial seed which will be used all
   * subsequently instantiated RandomVariableStream objects.
   *
   * \code
   *   RngSeedManger::SetSeed(15);
   *   UniformVariable x(2,3);     // These will give the same output every time
   *   ExponentialVariable y(120); // as long as the seed stays the same.
   * \endcode
   * \param [in] seed The seed value to use.
   *
   * \note While the underlying RNG takes six integer values as a seed;
   * it is sufficient to set these all to the same integer, so we provide
   * a simpler interface here that just takes one integer.
   */
  static void SetSeed (uint32_t seed);

  /**
   * \brief Get the current seed value which will be used by all
   * subsequently instantiated RandomVariableStream objects.
   *
   * \return The seed value.
   *
   * This returns the current seed value.
   */
  static uint32_t GetSeed (void);

  /**
   * \brief Set the run number of simulation.
   *
   * \code
   *   RngSeedManager::SetSeed(12);
   *   int N = atol(argv[1]);      // Read in run number from command line.
   *   RngSeedManager::SetRun(N);
   *   UniformVariable x(0,10);
   *   ExponentialVariable y(2902);
   * \endcode
   * In this example, \c N could successively be equal to 1,2,3, _etc._
   * and the user would continue to get independent runs out of the
   * single simulation.  For this simple example, the following might work:
   * \code
   *   ./simulation 0
   *   ...Results for run 0:...
   *
   *   ./simulation 1
   *   ...Results for run 1:...
   * \endcode
   *
   * \param [in] run The run number.
   */
  static void SetRun (uint64_t run);
  /**
   * \brief Get the current run number.
   * \returns The current run number
   * \see SetRun
   */
  static uint64_t GetRun (void);

  /**
   * Get the next automatically assigned stream index.
   * \returns The next stream index.
   */
  static uint64_t GetNextStreamIndex (void);

};

/** Alias for compatibility. */
typedef RngSeedManager SeedManager;

} // namespace ns3


#endif /* RNG_SEED_MANAGER_H */
