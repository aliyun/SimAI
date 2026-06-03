/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 IITP
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 */

#include "priority-queue-scheduler.h"
#include "event-impl.h"
#include "assert.h"
#include "log-macros-disabled.h"
#include "log.h"
#include "scheduler.h"
#include <string>

/**
 * \file
 * \ingroup scheduler
 * Implementation of ns3::PriorityQueueScheduler class.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PriorityQueueScheduler");

NS_OBJECT_ENSURE_REGISTERED (PriorityQueueScheduler);

TypeId
PriorityQueueScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PriorityQueueScheduler")
    .SetParent<Scheduler> ()
    .SetGroupName ("Core")
    .AddConstructor<PriorityQueueScheduler> ()
  ;
  return tid;
}

PriorityQueueScheduler::PriorityQueueScheduler ()
{
  NS_LOG_FUNCTION (this);
}
PriorityQueueScheduler::~PriorityQueueScheduler ()
{
  NS_LOG_FUNCTION (this);
}

void
PriorityQueueScheduler::Insert (const Event &ev)
{
  NS_LOG_FUNCTION (this << ev.impl << ev.key.m_ts << ev.key.m_uid);
  m_queue.push (ev);
}

bool
PriorityQueueScheduler::IsEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return m_queue.empty ();
}

Scheduler::Event
PriorityQueueScheduler::PeekNext (void) const
{
  NS_LOG_FUNCTION (this);
  return m_queue.top ();
}

Scheduler::Event
PriorityQueueScheduler::RemoveNext (void)
{
  NS_LOG_FUNCTION (this);
  Scheduler::Event ev = m_queue.top ();
  m_queue.pop ();
  return ev;
}

bool
PriorityQueueScheduler::EventPriorityQueue::remove(const Scheduler::Event &ev)
{
  auto it = std::find(this->c.begin(), this->c.end(), ev);
  if (it != this->c.end())
    {
      this->c.erase(it);
      std::make_heap(this->c.begin(), this->c.end(), this->comp);
      return true;
    }
  else
    {
      return false;
    }
}

void
PriorityQueueScheduler::Remove (const Scheduler::Event &ev)
{
  NS_LOG_FUNCTION (this);
  m_queue.remove (ev);
}

} // namespace ns3
