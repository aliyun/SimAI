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

#include "synchronizer.h"
#include "log.h"

/**
 * \file
 * \ingroup realtime
 * ns3::Synchronizer implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Synchronizer");

NS_OBJECT_ENSURE_REGISTERED (Synchronizer);

TypeId
Synchronizer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Synchronizer")
    .SetParent<Object> ()
    .SetGroupName ("Core")
  ;
  return tid;
}

Synchronizer::Synchronizer ()
  : m_realtimeOriginNano (0),
    m_simOriginNano (0)
{
  NS_LOG_FUNCTION (this);
}

Synchronizer::~Synchronizer ()
{
  NS_LOG_FUNCTION (this);
}

bool
Synchronizer::Realtime (void)
{
  NS_LOG_FUNCTION (this);
  return DoRealtime ();
}

uint64_t
Synchronizer::GetCurrentRealtime (void)
{
  NS_LOG_FUNCTION (this);
  return NanosecondToTimeStep (DoGetCurrentRealtime ());
}

void
Synchronizer::SetOrigin (uint64_t ts)
{
  NS_LOG_FUNCTION (this << ts);
  m_simOriginNano = TimeStepToNanosecond (ts);
  DoSetOrigin (m_simOriginNano);
}

uint64_t
Synchronizer::GetOrigin (void)
{
  NS_LOG_FUNCTION (this);
  return NanosecondToTimeStep (m_simOriginNano);
}

int64_t
Synchronizer::GetDrift (uint64_t ts)
{
  NS_LOG_FUNCTION (this << ts);
  int64_t tDrift = DoGetDrift (TimeStepToNanosecond (ts));

  if (tDrift < 0)
    {
      return -static_cast<int64_t> (NanosecondToTimeStep (-tDrift));
    }
  else
    {
      return static_cast<int64_t> (NanosecondToTimeStep (tDrift));
    }
}

bool
Synchronizer::Synchronize (uint64_t tsCurrent, uint64_t tsDelay)
{
  NS_LOG_FUNCTION (this << tsCurrent << tsDelay);
  return DoSynchronize (TimeStepToNanosecond (tsCurrent),
                        TimeStepToNanosecond (tsDelay));
}

void
Synchronizer::Signal (void)
{
  NS_LOG_FUNCTION (this);
  DoSignal ();
}

void
Synchronizer::SetCondition (bool cond)
{
  NS_LOG_FUNCTION (this << cond);
  DoSetCondition (cond);
}

void
Synchronizer::EventStart (void)
{
  NS_LOG_FUNCTION (this);
  DoEventStart ();
}

uint64_t
Synchronizer::EventEnd (void)
{
  NS_LOG_FUNCTION (this);
  return NanosecondToTimeStep (DoEventEnd ());
}

uint64_t
Synchronizer::TimeStepToNanosecond (uint64_t ts)
{
  NS_LOG_FUNCTION (this << ts);
  return TimeStep (ts).GetNanoSeconds ();
}

uint64_t
Synchronizer::NanosecondToTimeStep (uint64_t ns)
{
  NS_LOG_FUNCTION (this << ns);
  return NanoSeconds (ns).GetTimeStep ();
}

} // namespace ns3


