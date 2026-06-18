/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 */
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/command-line.h"
#include "ns3/random-variable-stream.h"
#include <iostream>

/**
 * \file
 * \ingroup core-examples
 * \ingroup randomvariable
 * Example program illustrating use of ns3::RandomVariable
 */

using namespace ns3;

/**
 * This program can be run from ns3 such as
 * `./ns3 run sample-random-variable`
 *
 * This program is about as simple as possible to display the use of ns-3
 * to generate random numbers.  By default, the uniform random variate that
 * will be outputted is 0.816532.  Because ns-3 uses a fixed seed for the
 * pseudo-random number generator, this program should always output the
 * same number.  Likewise, ns-3 simulations using random variables will
 * behave deterministically unless the user changes the Run number or the
 * Seed.
 *
 * There are three primary mechanisms to change the seed or run numbers
 * from their default integer value of 1:
 *
 * 1. Through explicit call of SeedManager::SetSeed () and
 *    SeedManager::SetRun () (commented out below)
 * 2. Through the passing of command line arguments such as:
 *    ./ns3 run program-name --command-template="%s --RngRun=<value>"`
 * 3. Through the use of the NS_GLOBAL_VALUE environment variable, such as:
 *    `$ NS_GLOBAL_VALUE="RngRun=<value>" ./ns3 run program-name`
 *
 * For instance, setting the run number to 3 will change the program output to
 * 0.775417
 *
 * Consult the ns-3 manual for more information about the use of the
 * random number generator
 */

int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  // SeedManager::SetRun (3);

  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();

  std::cout << uv->GetValue () << std::endl;

}
