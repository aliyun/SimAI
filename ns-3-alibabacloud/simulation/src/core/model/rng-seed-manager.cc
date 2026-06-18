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

#include "rng-seed-manager.h"
#include "global-value.h"
#include "attribute-helper.h"
#include "uinteger.h"
#include "config.h"
#include "log.h"

/**
 * \file
 * \ingroup randomvariable
 * ns3::RngSeedManager implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("RngSeedManager");

/**
 * \relates RngSeedManager
 * The next random number generator stream number to use
 * for automatic assignment.
 */
static uint64_t g_nextStreamIndex = 0;
/**
 * \relates RngSeedManager
 * \anchor GlobalValueRngSeed
 * The random number generator seed number global value.  This is used to
 * generate an new master PRNG sequence.  It is typically not modified
 * by user programs; the variable RngRun is preferred for independent
 * replications.
 *
 * This is accessible as "--RngSeed" from CommandLine.
 */
static ns3::GlobalValue g_rngSeed ("RngSeed",
                                   "The global seed of all rng streams",
                                   ns3::UintegerValue (1),
                                   ns3::MakeUintegerChecker<uint32_t> ());
/**
 * \relates RngSeedManager
 * \anchor GlobalValueRngRun
 * The random number generator substream index.  This is used to generate
 * new PRNG sequences for all streams (random variables) in such a manner
 * that the streams remain uncorrelated.  Incrementing this variable can
 * be used for independent replications.
 *
 * This is accessible as "--RngRun" from CommandLine.
 */
static ns3::GlobalValue g_rngRun ("RngRun",
                                  "The substream index used for all streams",
                                  ns3::UintegerValue (1),
                                  ns3::MakeUintegerChecker<uint64_t> ());


uint32_t RngSeedManager::GetSeed (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  UintegerValue seedValue;
  g_rngSeed.GetValue (seedValue);
  return static_cast<uint32_t> (seedValue.Get ());
}
void
RngSeedManager::SetSeed (uint32_t seed)
{
  NS_LOG_FUNCTION (seed);
  Config::SetGlobal ("RngSeed", UintegerValue (seed));
}

void RngSeedManager::SetRun (uint64_t run)
{
  NS_LOG_FUNCTION (run);
  Config::SetGlobal ("RngRun", UintegerValue (run));
}

uint64_t RngSeedManager::GetRun ()
{
  NS_LOG_FUNCTION_NOARGS ();
  UintegerValue value;
  g_rngRun.GetValue (value);
  uint64_t run = value.Get ();
  return run;
}

uint64_t RngSeedManager::GetNextStreamIndex (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  uint64_t next = g_nextStreamIndex;
  g_nextStreamIndex++;
  return next;
}

} // namespace ns3
