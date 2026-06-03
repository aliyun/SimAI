/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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

//
// This example is designed to show the main features of an ns3::TimeProbe.
// A test object is used to emit values through a trace source.  The 
// example shows three ways to use a ns3::TimeProbe to hook the output
// of this trace source (in addition to hooking the raw trace source).
//
// It produces two types of output.  By default, it will generate a
// gnuplot of interarrival times.  If the '--verbose=1' argument is passed,
// it will also generate debugging output of the form (for example):
//
//     Emitting at 96.5378 seconds
//     context: raw trace source old 0.293343 new 0.00760254
//     context: probe1 old 0.293343 new 0.00760254
//     context: probe2 old 0.293343 new 0.00760254
//     context: probe3 old 0.293343 new 0.00760254
//
// The stopTime defaults to 100 seconds but can be changed by an argument.
//

#include <string>

#include "ns3/core-module.h"
#include "ns3/time-probe.h"
#include "ns3/gnuplot-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TimeProbeExample");

/**
 * This is our test object, an object that emits values according to
 * a Poisson arrival process.   It emits a traced Time value as a 
 * trace source; this takes the value of interarrival time
 */
class Emitter : public Object
{
public:
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  Emitter ();
private:
  void DoInitialize (void);
  /// Generate data.
  void Emit (void);

  TracedValue<Time> m_interval;  //!< Interarrival time between events.
  Time m_last;                   //!< Current interarrival time.
  Ptr<ExponentialRandomVariable> m_var; //!< Random number generator.
};

NS_OBJECT_ENSURE_REGISTERED (Emitter);

TypeId
Emitter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Emitter")
    .SetParent<Object> ()
    .SetGroupName ("Stats")
    .AddConstructor<Emitter> ()
    .AddTraceSource ("Interval",
                     "Trace source",
                     MakeTraceSourceAccessor (&Emitter::m_interval),
                     "ns3::TracedValueCallback::Time")
  ;
  return tid;
}

Emitter::Emitter (void)
  : m_interval (Seconds (0)),
    m_last (Seconds (0))
{
  m_var = CreateObject<ExponentialRandomVariable> ();
}

void
Emitter::DoInitialize (void)
{
  Simulator::Schedule (Seconds (m_var->GetValue ()), &Emitter::Emit, this);
}

void
Emitter::Emit (void)
{
  NS_LOG_DEBUG ("Emitting at " << Simulator::Now ().As (Time::S));
  m_interval = Simulator::Now () - m_last;
  m_last = Simulator::Now ();
  TimeProbe::SetValueByPath ("/Names/probe3", m_interval);
  Simulator::Schedule (Seconds (m_var->GetValue ()), &Emitter::Emit, this);
}

// This is a function to test hooking a raw function to the trace source
void
NotifyViaTraceSource (std::string context, Time oldVal, Time newVal)
{
  BooleanValue verbose;
  GlobalValue::GetValueByName ("verbose", verbose);
  if (verbose.Get ())
    {
      std::cout << "context: " << context << " old " << oldVal.As (Time::S) << " new " << newVal.As (Time::S) << std::endl;
    }
}

// This is a function to test hooking it to the probe output
void
NotifyViaProbe (std::string context, double oldVal, double newVal)
{
  BooleanValue verbose;
  GlobalValue::GetValueByName ("verbose", verbose);
  if (verbose.Get ())
    {
      std::cout << "context: " << context << " old " << oldVal << " new " << newVal << std::endl;
    }
}

static ns3::GlobalValue g_verbose ("verbose",
                                   "Whether to enable verbose output",
                                   ns3::BooleanValue (false),
                                   ns3::MakeBooleanChecker ());

int main (int argc, char *argv[])
{
  double stopTime = 100.0;
  bool verbose = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("stopTime", "Time (seconds) to terminate simulation", stopTime);
  cmd.AddValue ("verbose", "Whether to enable verbose output", verbose);
  cmd.Parse (argc, argv);
  bool connected;

  // Set a global value, so that the callbacks can access it
  if (verbose)
    {
      GlobalValue::Bind ("verbose", BooleanValue (true));
      LogComponentEnable ("TimeProbeExample", LOG_LEVEL_ALL);
    }

  Ptr<Emitter> emitter = CreateObject<Emitter> ();
  Names::Add ("/Names/Emitter", emitter);

  //
  // The below shows typical functionality without a probe
  // (connect a sink function to a trace source)
  //
  connected = emitter->TraceConnect ("Interval", "raw trace source", MakeCallback (&NotifyViaTraceSource));
  NS_ASSERT_MSG (connected, "Trace source not connected");

  //
  // Next, we'll show several use cases of using a Probe to access and
  // filter the values of the underlying trace source
  //

  //
  // Probe1 will be hooked directly to the Emitter trace source object
  //

  // probe1 will be hooked to the Emitter trace source
  Ptr<TimeProbe> probe1 = CreateObject<TimeProbe> ();
  // the probe's name can serve as its context in the tracing
  probe1->SetName ("probe1");

  // Connect the probe to the emitter's Interval
  connected = probe1->ConnectByObject ("Interval", emitter);
  NS_ASSERT_MSG (connected, "Trace source not connected to probe1");

  // The probe itself should generate output.  The context that we provide
  // to this probe (in this case, the probe name) will help to disambiguate
  // the source of the trace
  connected = probe1->TraceConnect ("Output", probe1->GetName (), MakeCallback (&NotifyViaProbe));
  NS_ASSERT_MSG (connected, "Trace source not connected to probe1 Output");

  //
  // Probe2 will be hooked to the Emitter trace source object by
  // accessing it by path name in the Config database
  //

  // Create another similar probe; this will hook up via a Config path
  Ptr<TimeProbe> probe2 = CreateObject<TimeProbe> ();
  probe2->SetName ("probe2");

  // Note, no return value is checked here
  probe2->ConnectByPath ("/Names/Emitter/Interval");

  // The probe itself should generate output.  The context that we provide
  // to this probe (in this case, the probe name) will help to disambiguate
  // the source of the trace
  connected = probe2->TraceConnect ("Output", "probe2", MakeCallback (&NotifyViaProbe));
  NS_ASSERT_MSG (connected, "Trace source not connected to probe2 Output");

  //
  // Probe3 will be called by the emitter directly through the
  // static method SetValueByPath().
  //
  Ptr<TimeProbe> probe3 = CreateObject<TimeProbe> ();
  probe3->SetName ("probe3");

  // By adding to the config database, we can access it later
  Names::Add ("/Names/probe3", probe3);

  // The probe itself should generate output.  The context that we provide
  // to this probe (in this case, the probe name) will help to disambiguate
  // the source of the trace
  connected = probe3->TraceConnect ("Output", "probe3", MakeCallback (&NotifyViaProbe));
  NS_ASSERT_MSG (connected, "Trace source not connected to probe3 Output");

  // Plot the interval values
  GnuplotHelper plotHelper;
  plotHelper.ConfigurePlot ("time-probe-example",
                            "Emitter interarrivals vs. Time",
                            "Simulation time (Seconds)",
                            "Interarrival time (Seconds)",
                            "png");

  // Helper creates a TimeProbe and hooks it to the /Names/Emitter/Interval
  // source.  Helper also takes the Output of the TimeProbe and plots it
  // as a dataset labeled 'Emitter Interarrival Time'
  plotHelper.PlotProbe ("ns3::TimeProbe",
                        "/Names/Emitter/Interval",
                        "Output",
                        "Emitter Interarrival Time",
                        GnuplotAggregator::KEY_INSIDE);

  // The Emitter object is not associated with an ns-3 node, so
  // it won't get started automatically, so we need to do this ourselves
  Simulator::Schedule (Seconds (0.0), &Emitter::Initialize, emitter);

  Simulator::Stop (Seconds (stopTime));
  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
