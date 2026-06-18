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

/**
 * \file
 * \ingroup core-examples
 * \ingroup logging
 * Example program that demonstrates how to replace the time printer.
 */

/**
 * Example program that demonstrates how to replace the time printer.
 *
 * This program creates a sample object and schedules some methods to
 * occur at future times.  When run with no arguments, it prints out
 * something like this:
 * \code
 * $ ./ns3 run sample-log-time-format
 * RandomVariableStream:RandomVariableStream(0x184e3a0)
 * RandomVariableStream:UniformRandomVariable(0x184e3a0)
 * RandomVariableStream:SetStream(0x184e3a0, -1)
 * RandomVariableStream:SetAntithetic(0x184e3a0, 0)
 * +0.000000001s RandomVariableStream:SetAntithetic(0x184e3a0, 0)
 * +0.000000123s RandomVariableStream:SetAntithetic(0x184e3a0, 0)
 * +0.000123456s RandomVariableStream:SetAntithetic(0x184e3a0, 0)
 * +0.123456789s RandomVariableStream:SetAntithetic(0x184e3a0, 0)
 * RandomVariableStream:~RandomVariableStream(0x184e3a0)
 * \endcode
 *
 * These statements are printed out because of these two program
 * statements:
 *
 * \code
 * LogComponentEnable ("RandomVariableStream", LOG_LEVEL_ALL);
 * LogComponentEnableAll (LOG_PREFIX_TIME);
 * \endcode
 *
 * The first statement enables logging on the methods found in
 * random-variable-stream.cc.  The second prepends a time prefix
 * to logging statements.  Note that only those logging statements
 * occurring after Simulator::Run () is called have a time prefix;
 * this is because the time printer is only hooked to the simulator
 * once the simulator object is instantiated (after Simulator::Run ()
 * is called).
 *
 * To change the format, one can schedule (at simulation time 0) a
 * replacement function for printing time.  This can be demonstrated
 * by setting the 'replace-time-printer' parameter to true:
 * \code
 * ./ns3 run 'sample-log-time-format --replaceTimePrinter=1'
 * RandomVariableStream:RandomVariableStream(0x15fb080)
 * RandomVariableStream:UniformRandomVariable(0x15fb080)
 * RandomVariableStream:SetStream(0x15fb080, -1)
 * RandomVariableStream:SetAntithetic(0x15fb080, 0)
 * Replacing time printer function after Simulator::Run ()
 * 1e-09s RandomVariableStream:SetAntithetic(0x15fb080, 0)
 * 1.23e-07s RandomVariableStream:SetAntithetic(0x15fb080, 0)
 * 0.000123456s RandomVariableStream:SetAntithetic(0x15fb080, 0)
 * 0.123457s RandomVariableStream:SetAntithetic(0x15fb080, 0)
 * RandomVariableStream:~RandomVariableStream(0x15fb080)
 * \endcode
 *
 * In the above, the default C++ iostream precision is instead used
 * (which was the default for ns-3 versions 3.26 and earlier).
 *
 * In addition, the 'resolution' program argument allows one to experiment
 * with changing ns-3's time resolution from its default of Time::NS, such
 * as Time::PS or Time::FS; the precision changes accordingly.
 *
 * The maximum useful precision is 20 decimal digits, since Time is
 * signed 64 bits.
 */

#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/command-line.h"
#include "ns3/log.h"
#include "ns3/random-variable-stream.h"

using namespace ns3;

namespace {

/**
 * Pre-ns-3.26 TimePrinter equivalent (was called LogTimePrinter).
 *
 * Prior to ns-3.26, the time printer used default C++ iostream precision
 * This function sets it back to the format used before ns-3.26
 *
 * \param [in] os The stream to print on.
 */
void
ReplacementTimePrinter (std::ostream &os)
{
  os << Simulator::Now ().GetSeconds () << "s";
}

/** Set ReplacementTimePrinter as the time printer for log messages. */
void
ReplaceTimePrinter (void)
{
  std::cout << "Replacing time printer function after Simulator::Run ()" << std::endl;
  LogSetTimePrinter (&ReplacementTimePrinter);
}


}  // unnamed namespace


int main (int argc, char *argv[])
{
  bool replaceTimePrinter = false;
  std::string resolution = "Time::NS";
  LogComponentEnable ("RandomVariableStream", LOG_LEVEL_ALL);
  LogComponentEnableAll (LOG_PREFIX_TIME);

  std::map<std::string, Time::Unit> resolutionMap = {{"Time::US", Time::US}, {"Time::NS", Time::NS}, {"Time::PS", Time::PS}, {"Time::FS", Time::FS}};

  CommandLine cmd (__FILE__);
  cmd.AddValue ("replaceTimePrinter", "replace time printing function", replaceTimePrinter);
  cmd.AddValue ("resolution", "time resolution", resolution);
  cmd.Parse (argc, argv);

  auto search = resolutionMap.find (resolution);
  if (search != resolutionMap.end ())
    {
      Time::SetResolution (search->second);
    }

  Ptr<UniformRandomVariable> uniformRv = CreateObject<UniformRandomVariable> ();

  if (replaceTimePrinter)
    {
      Simulator::Schedule (Seconds (0), &ReplaceTimePrinter);
    }

  Simulator::Schedule (NanoSeconds (1), &UniformRandomVariable::SetAntithetic, uniformRv, false);
  Simulator::Schedule (NanoSeconds (123), &UniformRandomVariable::SetAntithetic, uniformRv, false);
  Simulator::Schedule (NanoSeconds (123456), &UniformRandomVariable::SetAntithetic, uniformRv, false);
  Simulator::Schedule (NanoSeconds (123456789), &UniformRandomVariable::SetAntithetic, uniformRv, false);

  Simulator::Run ();
  Simulator::Destroy ();
}
