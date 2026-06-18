/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007, Emmanuelle Laprise
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
 * Author: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 */

#include "backoff.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Backoff");

Backoff::Backoff () 
{
  m_slotTime = MicroSeconds (1);
  m_minSlots = 1;
  m_maxSlots = 1000;
  m_ceiling = 10;
  m_maxRetries = 1000;
  m_numBackoffRetries = 0;
  m_rng = CreateObject<UniformRandomVariable> ();

  ResetBackoffTime ();
}

Backoff::Backoff(Time slotTime, uint32_t minSlots, uint32_t maxSlots, uint32_t ceiling, uint32_t maxRetries)
{
  m_slotTime = slotTime;
  m_minSlots = minSlots;
  m_maxSlots = maxSlots;
  m_ceiling = ceiling;
  m_maxRetries = maxRetries;
  m_numBackoffRetries = 0;
  m_rng = CreateObject<UniformRandomVariable> ();
}

Time
Backoff::GetBackoffTime (void)
{
  uint32_t ceiling;

  if ((m_ceiling > 0) &&(m_numBackoffRetries > m_ceiling))
    {
      ceiling = m_ceiling;
    }
  else
    {
      ceiling = m_numBackoffRetries;
    }

  uint32_t minSlot = m_minSlots;
  uint32_t maxSlot = (uint32_t)pow (2, ceiling) - 1;
  if (maxSlot > m_maxSlots)
    {
      maxSlot = m_maxSlots;
    }

  uint32_t backoffSlots = (uint32_t)m_rng->GetValue (minSlot, maxSlot);

  Time backoff = Time (backoffSlots * m_slotTime);
  return backoff;
}

void 
Backoff::ResetBackoffTime (void)
{
  m_numBackoffRetries = 0;
}

bool 
Backoff::MaxRetriesReached (void) 
{
  return (m_numBackoffRetries >= m_maxRetries);
}

void 
Backoff::IncrNumRetries (void) 
{
  m_numBackoffRetries++;
}

int64_t
Backoff::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_rng->SetStream (stream);
  return 1;
}

} // namespace ns3
