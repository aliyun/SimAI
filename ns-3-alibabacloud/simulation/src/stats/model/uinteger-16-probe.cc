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
 */

#include "ns3/uinteger-16-probe.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Uinteger16Probe");

NS_OBJECT_ENSURE_REGISTERED (Uinteger16Probe);

TypeId
Uinteger16Probe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Uinteger16Probe")
    .SetParent<Probe> ()
    .SetGroupName ("Stats")
    .AddConstructor<Uinteger16Probe> ()
    .AddTraceSource ( "Output",
                      "The uint16_t that serves as output for this probe",
                      MakeTraceSourceAccessor (&Uinteger16Probe::m_output),
                     "ns3::TracedValueCallback::Uint16")
  ;
  return tid;
}

Uinteger16Probe::Uinteger16Probe ()
{
  NS_LOG_FUNCTION (this);
  m_output = 0;
}

Uinteger16Probe::~Uinteger16Probe ()
{
  NS_LOG_FUNCTION (this);
}

uint16_t
Uinteger16Probe::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_output;
}
void
Uinteger16Probe::SetValue (uint16_t newVal)
{
  NS_LOG_FUNCTION (this << newVal);
  m_output = newVal;
}

void
Uinteger16Probe::SetValueByPath (std::string path, uint16_t newVal)
{
  NS_LOG_FUNCTION (path << newVal);
  Ptr<Uinteger16Probe> probe = Names::Find<Uinteger16Probe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (newVal);
}

bool
Uinteger16Probe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of probe (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::Uinteger16Probe::TraceSink, this));
  return connected;
}

void
Uinteger16Probe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of probe to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::Uinteger16Probe::TraceSink, this));
}

void
Uinteger16Probe::TraceSink (uint16_t oldData, uint16_t newData)
{
  NS_LOG_FUNCTION (this << oldData << newData);
  if (IsEnabled ())
    {
      m_output = newData;
    }
}

} // namespace ns3
