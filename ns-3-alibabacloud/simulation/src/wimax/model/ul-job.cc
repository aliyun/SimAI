/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c)
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
 * Author:  Juliana Freitag Borin, Flavio Kubota and Nelson L.
 * S. da Fonseca - wimaxgroup@lrc.ic.unicamp.br
 */

#include <stdint.h>
#include "ul-job.h"

namespace ns3 {

UlJob::UlJob (void) : m_deadline (Seconds (0)), m_size (0)
{
}

UlJob::~UlJob (void)
{
}

SSRecord*
UlJob::GetSsRecord (void)
{
  return m_ssRecord;
}
void
UlJob::SetSsRecord (SSRecord* ssRecord)
{
  m_ssRecord = ssRecord;
}

enum
ServiceFlow::SchedulingType UlJob::GetSchedulingType (void)
{
  return m_schedulingType;
}

void
UlJob::SetSchedulingType (ServiceFlow::SchedulingType schedulingType)
{
  m_schedulingType = schedulingType;
}

ReqType
UlJob::GetType (void)
{
  return m_type;
}

void
UlJob::SetType (ReqType type)
{
  m_type = type;
}

ServiceFlow *
UlJob::GetServiceFlow (void)
{
  return m_serviceFlow;
}

void
UlJob::SetServiceFlow (ServiceFlow *serviceFlow)
{
  m_serviceFlow = serviceFlow;
}

Time
UlJob::GetReleaseTime (void)
{
  return m_releaseTime;
}

void
UlJob::SetReleaseTime (Time releaseTime)
{
  m_releaseTime = releaseTime;
}

Time
UlJob::GetPeriod (void)
{
  return m_period;
}
void
UlJob::SetPeriod (Time period)
{
  m_period = period;
}

Time
UlJob::GetDeadline (void)
{
  return m_deadline;
}
void
UlJob::SetDeadline (Time deadline)
{
  m_deadline = deadline;
}

uint32_t
UlJob::GetSize (void)
{
  return m_size;
}

void
UlJob::SetSize (uint32_t size)
{
  m_size = size;
}

/**
 * \brief equality operator
 * \param a first ULJob
 * \param b second ULJob
 * \returns true if equal
 */
bool operator == (const UlJob &a, const UlJob &b)
{
  UlJob A = a;
  UlJob B = b;

  if ((A.GetServiceFlow () == B.GetServiceFlow ()) && (A.GetSsRecord () == B.GetSsRecord ()))
    {
      return true;
    }
  return false;
}

PriorityUlJob::PriorityUlJob (void)
{
}

int
PriorityUlJob::GetPriority (void)
{
  return m_priority;
}

void
PriorityUlJob::SetPriority (int priority)
{
  m_priority = priority;
}

Ptr<UlJob>
PriorityUlJob::GetUlJob (void)
{
  return m_job;
}
void
PriorityUlJob::SetUlJob (Ptr<UlJob> job)
{
  m_job = job;
}

} // namespace ns3
