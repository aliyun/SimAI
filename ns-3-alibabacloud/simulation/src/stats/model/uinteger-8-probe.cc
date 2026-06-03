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

#include "ns3/uinteger-8-probe.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Uinteger8Probe");

NS_OBJECT_ENSURE_REGISTERED (Uinteger8Probe);

TypeId
Uinteger8Probe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Uinteger8Probe")
    .SetParent<Probe> ()
    .SetGroupName ("Stats")
    .AddConstructor<Uinteger8Probe> ()
    .AddTraceSource ( "Output",
                      "The uint8_t that serves as output for this probe",
                      MakeTraceSourceAccessor (&Uinteger8Probe::m_output),
                     "ns3::TracedValueCallback::Uint8")
  ;
  return tid;
}

Uinteger8Probe::Uinteger8Probe ()
{
  NS_LOG_FUNCTION (this);
  m_output = 0;
}

Uinteger8Probe::~Uinteger8Probe ()
{
  NS_LOG_FUNCTION (this);
}

uint8_t
Uinteger8Probe::GetValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_output;
}
void
Uinteger8Probe::SetValue (uint8_t newVal)
{
  NS_LOG_FUNCTION (this << newVal);
  m_output = newVal;
}

void
Uinteger8Probe::SetValueByPath (std::string path, uint8_t newVal)
{
  NS_LOG_FUNCTION (path << newVal);
  Ptr<Uinteger8Probe> probe = Names::Find<Uinteger8Probe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (newVal);
}

bool
Uinteger8Probe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of probe (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::Uinteger8Probe::TraceSink, this));
  return connected;
}

void
Uinteger8Probe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of probe to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::Uinteger8Probe::TraceSink, this));
}

void
Uinteger8Probe::TraceSink (uint8_t oldData, uint8_t newData)
{
  NS_LOG_FUNCTION (this << oldData << newData);
  if (IsEnabled ())
    {
      m_output = newData;
    }
}

} // namespace ns3
