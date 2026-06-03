/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: L. Felipe Perrone (perrone@bucknell.edu)
 *          Tiago G. Rodrigues (tgr002@bucknell.edu)
 *
 * Modified by: Mitch Watrous (watrous@u.washington.edu)
 *
 */

#include "ns3/time-probe.h"
#include "ns3/object.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TimeProbe");

NS_OBJECT_ENSURE_REGISTERED (TimeProbe);

TypeId
TimeProbe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::TimeProbe")
    .SetParent<Probe> ()
    .SetGroupName ("Stats")
    .AddConstructor<TimeProbe> ()
    .AddTraceSource ("Output",
                     "The double valued (units of seconds) probe output",
                     MakeTraceSourceAccessor (&TimeProbe::m_output),
                     "ns3::TracedValueCallback::Double")
  ;
  return tid;
}

TimeProbe::TimeProbe ()
{
  NS_LOG_FUNCTION (this);
  m_output = 0;
}

TimeProbe::~TimeProbe ()
{
  NS_LOG_FUNCTION (this);
}

double
TimeProbe::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_output;
}
void
TimeProbe::SetValue (Time newVal)
{
  NS_LOG_FUNCTION (this << newVal.As (Time::S));
  m_output = newVal.GetSeconds ();
}

void
TimeProbe::SetValueByPath (std::string path, Time newVal)
{
  NS_LOG_FUNCTION (path << newVal.As (Time::S));
  Ptr<TimeProbe> probe = Names::Find<TimeProbe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (newVal);
}

bool
TimeProbe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of trace source (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::TimeProbe::TraceSink, this));
  return connected;
}

void
TimeProbe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of trace source to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::TimeProbe::TraceSink, this));
}

void
TimeProbe::TraceSink (Time oldData, Time newData)
{
  NS_LOG_FUNCTION (this << oldData.As (Time::S) << newData.As (Time::S));
  if (IsEnabled ())
    {
      m_output = newData.GetSeconds ();
    }
}

} // namespace ns3
