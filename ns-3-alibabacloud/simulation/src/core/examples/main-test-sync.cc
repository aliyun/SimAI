/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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
 */

#include "ns3/command-line.h"
#include "ns3/simulator.h"
#include "ns3/realtime-simulator-impl.h"
#include "ns3/nstime.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/config.h"
#include "ns3/global-value.h"
#include "ns3/ptr.h"

#include <chrono>  // seconds, milliseconds
#include <thread>  // sleep_for

/**
 * \file
 * \ingroup core-examples
 * \ingroup scheduler
 * An example of scheduling events in a background thread.
 *
 * See \ref ns3::SimulatorImpl::ScheduleWithContext
 */

using namespace ns3;


NS_LOG_COMPONENT_DEFINE ("TestSync");

namespace {

/** Check that the event functions run in the intended order. */
bool gFirstRun = false;

/** An event method called many times from the background thread. */
void
inserted_function (void)
{
  NS_ASSERT (gFirstRun);
  NS_LOG_UNCOND ("inserted_function() called at " <<
                 Simulator::Now ().GetSeconds () << " s");
}

/** An event method called many times from the main thread. */
void
background_function (void)
{
  NS_ASSERT (gFirstRun);
  NS_LOG_UNCOND ("background_function() called at " <<
                 Simulator::Now ().GetSeconds () << " s");
}

/** An event method called once from the main thread. */
void
first_function (void)
{
  NS_LOG_UNCOND ("first_function() called at " <<
                 Simulator::Now ().GetSeconds () << " s");
  gFirstRun = true;
}

/** Example class with a method for the background task. */
class FakeNetDevice
{
public:
  /** Constructor. */
  FakeNetDevice ();
  /** The thread entry point. */
  void Doit3 (void);
};

FakeNetDevice::FakeNetDevice ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
FakeNetDevice::Doit3 (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  std::this_thread::sleep_for (std::chrono::seconds (1));

  for (uint32_t i = 0; i < 10000; ++i)
    {
      //
      // Exercise the realtime relative now path
      //
      Simulator::ScheduleWithContext (Simulator::NO_CONTEXT, Seconds (0.0), MakeEvent (&inserted_function));
      std::this_thread::sleep_for (std::chrono::milliseconds (1));
    }
}

/**
 * Example use of std::thread.
 *
 * This example is a complete simulation.
 * It schedules \c first_function and many executions of \c background_function
 * to execute in the main (foreground) thread.  It also launches a background
 * thread with an instance of FakeNetDevice, which schedules many instances of
 * \c inserted_function.
 */
void
test (void)
{
  GlobalValue::Bind ("SimulatorImplementationType",
                     StringValue ("ns3::RealtimeSimulatorImpl"));

  FakeNetDevice fnd;

  //
  // Make sure ScheduleNow works when the system isn't running
  //
  Simulator::ScheduleWithContext (0xffffffff, Seconds (0.0), MakeEvent (&first_function));

  //
  // drive the progression of m_currentTs at a ten millisecond rate from the main thread
  //
  for (double d = 0.; d < 14.999; d += 0.01)
    {
      Simulator::Schedule (Seconds (d), &background_function);
    }

  std::thread st3 = std::thread (&FakeNetDevice::Doit3, &fnd);

  Simulator::Stop (Seconds (15.0));
  Simulator::Run ();

  if (st3.joinable ())
    {
      st3.join ();
    }

  Simulator::Destroy ();
}

}  // unnamed namespace


int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  while (true)
    {
      test ();
    }
}

