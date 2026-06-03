/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 LLNL
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


/**
 * @file
 * @ingroup simulator
 * ns3::DesMetrics implementation.
 */

#include "des-metrics.h"
#include "simulator.h"
#include "system-path.h"

#include <ctime>    // time_t, time()
#include <sstream>
#include <string>

namespace ns3 {

/* static */
std::string DesMetrics::m_outputDir; // = "";

void
DesMetrics::Initialize (std::vector<std::string> args, std::string outDir /* = "" */ )
{
  if (m_initialized)
    {
      // Running multiple tests, so close the previous output file
      Close ();
    }

  m_initialized = true;

  std::string model_name ("desTraceFile");
  if (args.size () > 0)
    {
      std::string arg0 = args[0];
      model_name = SystemPath::Split (arg0).back ();
    }
  std::string jsonFile = model_name + ".json";
  if (outDir != "")
    {
      DesMetrics::m_outputDir = outDir;
    }
  if (DesMetrics::m_outputDir != "")
    {
      jsonFile = SystemPath::Append (DesMetrics::m_outputDir, jsonFile);
    }

  time_t current_time;
  time (&current_time);
  const char * date = ctime (&current_time);
  std::string capture_date (date, 24);  // discard trailing newline from ctime

  m_os.open (jsonFile.c_str ());
  m_os << "{" << std::endl;
  m_os << " \"simulator_name\" : \"ns-3\"," << std::endl;
  m_os << " \"model_name\" : \"" << model_name << "\"," << std::endl;
  m_os << " \"capture_date\" : \"" << capture_date << "\"," << std::endl;
  m_os << " \"command_line_arguments\" : \"";
  if (args.size () == 0)
    {
      for (std::size_t i = 0; i < args.size (); ++i)
        {
          if (i > 0)
            {
              m_os << " ";
            }
          m_os << args[i];
        }
    }
  else
    {
      m_os << "[argv empty or not available]";
    }
  m_os << "\"," << std::endl;
  m_os << " \"events\" : [" << std::endl;

  m_separator = ' ';

}

void
DesMetrics::Trace (const Time & now, const Time & delay)
{
  TraceWithContext (Simulator::GetContext (), now, delay);
}

void
DesMetrics::TraceWithContext (uint32_t context, const Time & now, const Time & delay)
{
  if (!m_initialized)
    {
      std::vector<std::string> args;
      Initialize (args);
    }

  std::ostringstream ss;
  if (m_separator == ',')
    {
      ss << m_separator << std::endl;
    }

  uint32_t sendCtx = Simulator::GetContext ();
  // Force to signed so we can show NoContext as '-1'
  int32_t send = (sendCtx != Simulator::NO_CONTEXT) ? (int32_t)sendCtx : -1;
  int32_t recv = (context != Simulator::NO_CONTEXT) ? (int32_t)context : -1;

  ss <<                                 "  [\""
     << send                         << "\",\""
     << now.GetTimeStep ()           << "\",\""
     << recv                         << "\",\""
     << (now + delay).GetTimeStep () << "\"]";

  {
    std::unique_lock lock {m_mutex};
    m_os << ss.str ();
  }

  m_separator = ',';
}

DesMetrics::~DesMetrics (void)
{
  Close ();
}

void
DesMetrics::Close (void)
{
  m_os << std::endl;    // Finish the last event line

  m_os << " ]" << std::endl;
  m_os << "}" << std::endl;
  m_os.close ();

  m_initialized = false;
}



} // namespace ns3

