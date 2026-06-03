/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Lawrence Livermore National Laboratory
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
 * Author: Gustavo Carneiro <gjc@inescporto.pt>
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

/**
 * \file
 * \ingroup core-examples
 * \ingroup logging
 * Example program that demonstrates ShowProgress.
 */

/**
 * Example program that demonstrates ShowProgress.
 *
 */

#include "ns3/core-module.h"

#include <chrono>
#include <iomanip>
#include <thread>
#include <string>


using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SampleShowProgress");

namespace {

/**
 * Execute a function periodically,
 * which takes more or less time to run.
 *
 * Inspired by PHOLD.
 */
class Hold : public SimpleRefCount<Hold>
{
public:

  /**
   * Create a Hold with mean inter-event time \pname{wait},
   * changing workload every \pname{interval}.
   * \param wait The mean inter-event time.
   * \param interval How often to change work load.  This
   *                 should be an order of magnitude larger than \pname{wait}.
   */
  Hold (Time wait, Time interval)
  {
    m_wait = wait;
    m_interval = interval;

    m_rng = CreateObject<ExponentialRandomVariable> ();
    m_rng->SetAttribute ("Mean", DoubleValue (m_wait.GetSeconds ()));
  }

  /**
   * Create a hold with a specified random number generator for the
   * \pname{wait} time.  The RNG value will be interpreted as seconds.
   * \param rng The random variable generator to use for the inter-event time.
   */
  Hold (Ptr<RandomVariableStream> rng)
    : m_rng (rng)
  {}

  /** The Hold event. */
  void Event (void)
  {
    double delta = m_rng->GetValue ();
    Time delay = Seconds (delta);
    NS_LOG_LOGIC ("event delay: " << delay);

    Simulator::Schedule (delay, &Hold::Event, this);

    // Switch work load every 10 * m_interval of simulation time
    int64x64_t ratio = (Simulator::Now () / m_interval) / 10;
    bool even = (ratio.GetHigh () % 2);
    Time work = m_wait * (even ? 3 : 1);
    std::this_thread::sleep_for (std::chrono::nanoseconds (work.GetNanoSeconds ()));
  }

private:
  /** The random number generator for the interval between events. */
  Ptr<RandomVariableStream> m_rng;
  /** Mean inter-event time. */
  Time m_wait;
  /** Time between switching workloads. */
  Time m_interval;

};  // class HOLD

}  // unnamed namespace


int
main (int argc, char ** argv)
{
  Time stop = Seconds (100);
  Time interval = Seconds (10);
  Time wait = MilliSeconds (10);
  bool verbose = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("stop", "Simulation duration in virtual time.", stop);
  cmd.AddValue ("interval", "Approximate reporting interval, in wall clock time.", interval);
  cmd.AddValue ("wait", "Wallclock time to burn on each event.", wait);
  cmd.AddValue ("verbose", "Turn on verbose progress message.", verbose);
  cmd.Parse (argc, argv);

  std::cout << "\n"
            << cmd.GetName () << ":\n"
            << "\n"
            << "verbose progress message:  " << (verbose ? "on\n" : "off\n")
            << "target reporting interval: " << interval.As (Time::S) << "\n"
            << "average event sleep time:  " << wait.As (Time::MS)    << "\n"
            << "total simulation run time: " << stop.As (Time::S)
            << std::endl;

  Ptr<Hold> h = Create<Hold> (wait, interval);
  h->Event ();

  Simulator::Stop (stop);
  ShowProgress spinner (interval);
  spinner.SetVerbose (verbose);

  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
