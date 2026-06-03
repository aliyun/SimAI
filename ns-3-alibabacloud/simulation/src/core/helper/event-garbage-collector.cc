/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INESC Porto
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
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */
#include "event-garbage-collector.h"

/**
 * \file
 * \ingroup core-helpers
 * \ingroup events
 * ns3::EventGarbageCollector implementation.
 */

namespace ns3 {


EventGarbageCollector::EventGarbageCollector ()
  : m_nextCleanupSize (CHUNK_INIT_SIZE),
    m_events ()
{}

void
EventGarbageCollector::Track (EventId event)
{
  m_events.insert (event);
  if (m_events.size () >= m_nextCleanupSize)
    {
      Cleanup ();
    }
}

void
EventGarbageCollector::Grow ()
{
  m_nextCleanupSize += (m_nextCleanupSize < CHUNK_MAX_SIZE ?
                        m_nextCleanupSize : CHUNK_MAX_SIZE);
}

void
EventGarbageCollector::Shrink ()
{
  while (m_nextCleanupSize > m_events.size ())
    {
      m_nextCleanupSize >>= 1;
    }
  Grow ();
}

// Called when a new event was added and
// the cleanup limit was exceeded in consequence.
void
EventGarbageCollector::Cleanup ()
{
  for (EventList::iterator iter = m_events.begin (); iter != m_events.end ();)
    {
      if ((*iter).IsExpired ())
        {
          m_events.erase (iter++);
        }
      else
        {
          break; // EventIds are sorted by timestamp => further events are not expired for sure
        }
    }

  // If after cleanup we are still over the limit, increase the limit.
  if (m_events.size () >= m_nextCleanupSize)
    {
      Grow ();
    }
  else
    {
      Shrink ();
    }
}


EventGarbageCollector::~EventGarbageCollector ()
{
  for (auto event : m_events)
    {
      Simulator::Cancel (event);
    }
}

} // namespace ns3


