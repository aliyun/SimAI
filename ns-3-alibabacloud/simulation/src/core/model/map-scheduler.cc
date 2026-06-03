/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * The idea to use a std c++ map came from GTNetS
 */

#include "map-scheduler.h"
#include "event-impl.h"
#include "assert.h"
#include "log.h"
#include <string>

/**
 * \file
 * \ingroup scheduler
 * ns3::MapScheduler implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MapScheduler");

NS_OBJECT_ENSURE_REGISTERED (MapScheduler);

TypeId
MapScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MapScheduler")
    .SetParent<Scheduler> ()
    .SetGroupName ("Core")
    .AddConstructor<MapScheduler> ()
  ;
  return tid;
}

MapScheduler::MapScheduler ()
{
  NS_LOG_FUNCTION (this);
}
MapScheduler::~MapScheduler ()
{
  NS_LOG_FUNCTION (this);
}

void
MapScheduler::Insert (const Event &ev)
{
  NS_LOG_FUNCTION (this << ev.impl << ev.key.m_ts << ev.key.m_uid);
  std::pair<EventMapI,bool> result;
  result = m_list.insert (std::make_pair (ev.key, ev.impl));
  NS_ASSERT (result.second);
}

bool
MapScheduler::IsEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_list.empty ();
}

Scheduler::Event
MapScheduler::PeekNext (void) const
{
  NS_LOG_FUNCTION (this);
  EventMapCI i = m_list.begin ();
  NS_ASSERT (i != m_list.end ());

  Event ev;
  ev.impl = i->second;
  ev.key = i->first;
  NS_LOG_DEBUG (this << ev.impl << ev.key.m_ts << ev.key.m_uid);
  return ev;
}
Scheduler::Event
MapScheduler::RemoveNext (void)
{
  NS_LOG_FUNCTION (this);
  EventMapI i = m_list.begin ();
  NS_ASSERT (i != m_list.end ());
  Event ev;
  ev.impl = i->second;
  ev.key = i->first;
  m_list.erase (i);
  NS_LOG_DEBUG (this << ev.impl << ev.key.m_ts << ev.key.m_uid);
  return ev;
}

void
MapScheduler::Remove (const Event &ev)
{
  NS_LOG_FUNCTION (this << ev.impl << ev.key.m_ts << ev.key.m_uid);
  EventMapI i = m_list.find (ev.key);
  NS_ASSERT (i->second == ev.impl);
  m_list.erase (i);
}

} // namespace ns3
