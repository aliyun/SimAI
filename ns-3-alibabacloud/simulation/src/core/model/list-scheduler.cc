/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 */

#include "list-scheduler.h"
#include "event-impl.h"
#include "log.h"
#include <utility>
#include <string>
#include "assert.h"

/**
 * \file
 * \ingroup scheduler
 * ns3::ListScheduler implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ListScheduler");

NS_OBJECT_ENSURE_REGISTERED (ListScheduler);

TypeId
ListScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ListScheduler")
    .SetParent<Scheduler> ()
    .SetGroupName ("Core")
    .AddConstructor<ListScheduler> ()
  ;
  return tid;
}

ListScheduler::ListScheduler ()
{
  NS_LOG_FUNCTION (this);
}
ListScheduler::~ListScheduler ()
{}

void
ListScheduler::Insert (const Event &ev)
{
  NS_LOG_FUNCTION (this << &ev);
  for (EventsI i = m_events.begin (); i != m_events.end (); i++)
    {
      if (ev.key < i->key)
        {
          m_events.insert (i, ev);
          return;
        }
    }
  m_events.push_back (ev);
}
bool
ListScheduler::IsEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_events.empty ();
}
Scheduler::Event
ListScheduler::PeekNext (void) const
{
  NS_LOG_FUNCTION (this);
  return m_events.front ();
}

Scheduler::Event
ListScheduler::RemoveNext (void)
{
  NS_LOG_FUNCTION (this);
  Event next = m_events.front ();
  m_events.pop_front ();
  return next;
}

void
ListScheduler::Remove (const Event &ev)
{
  NS_LOG_FUNCTION (this << &ev);
  for (EventsI i = m_events.begin (); i != m_events.end (); i++)
    {
      if (i->key.m_uid == ev.key.m_uid)
        {
          NS_ASSERT (ev.impl == i->impl);
          m_events.erase (i);
          return;
        }
    }
  NS_ASSERT (false);
}

} // namespace ns3
