/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
 * Copyright (c) 2005 Mathieu Lacage
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
 *
 */

#include "heap-scheduler.h"
#include "event-impl.h"
#include "assert.h"
#include "log.h"

/**
 * \file
 * \ingroup scheduler
 * Implementation of ns3::HeapScheduler class.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HeapScheduler");

NS_OBJECT_ENSURE_REGISTERED (HeapScheduler);

TypeId
HeapScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::HeapScheduler")
    .SetParent<Scheduler> ()
    .SetGroupName ("Core")
    .AddConstructor<HeapScheduler> ()
  ;
  return tid;
}

HeapScheduler::HeapScheduler ()
{
  NS_LOG_FUNCTION (this);
  // we purposely waste an item at the start of
  // the array to make sure the indexes in the
  // array start at one.
  Scheduler::Event empty = { 0,{ 0,0}};
  m_heap.push_back (empty);
}

HeapScheduler::~HeapScheduler ()
{
  NS_LOG_FUNCTION (this);
}

std::size_t
HeapScheduler::Parent (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return id / 2;
}
std::size_t
HeapScheduler::Sibling (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return id + 1;
}
std::size_t
HeapScheduler::LeftChild (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return id * 2;
}
std::size_t
HeapScheduler::RightChild (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return id * 2 + 1;
}

std::size_t
HeapScheduler::Root (void) const
{
  NS_LOG_FUNCTION (this);
  return 1;
}

bool
HeapScheduler::IsRoot (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return (id == Root ());
}

std::size_t
HeapScheduler::Last (void) const
{
  NS_LOG_FUNCTION (this);
  return m_heap.size () - 1;
}


bool
HeapScheduler::IsBottom (std::size_t id) const
{
  NS_LOG_FUNCTION (this << id);
  return (id >= m_heap.size ());
}

void
HeapScheduler::Exch (std::size_t a, std::size_t b)
{
  NS_LOG_FUNCTION (this << a << b);
  NS_ASSERT (b < m_heap.size () && a < m_heap.size ());
  NS_LOG_DEBUG ("Exch " << a << ", " << b);
  Event tmp (m_heap[a]);
  m_heap[a] = m_heap[b];
  m_heap[b] = tmp;
}

bool
HeapScheduler::IsLessStrictly (std::size_t a, std::size_t b) const
{
  NS_LOG_FUNCTION (this << a << b);
  return m_heap[a] < m_heap[b];
}

std::size_t
HeapScheduler::Smallest (std::size_t a, std::size_t b) const
{
  NS_LOG_FUNCTION (this << a << b);
  return IsLessStrictly (a,b) ? a : b;
}

bool
HeapScheduler::IsEmpty (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_heap.size () == 1);
}

void
HeapScheduler::BottomUp (void)
{
  NS_LOG_FUNCTION (this);
  std::size_t index = Last ();
  while (!IsRoot (index)
         && IsLessStrictly (index, Parent (index)))
    {
      Exch (index, Parent (index));
      index = Parent (index);
    }
}

void
HeapScheduler::TopDown (std::size_t start)
{
  NS_LOG_FUNCTION (this << start);
  std::size_t index = start;
  std::size_t right = RightChild (index);
  while (!IsBottom (right))
    {
      std::size_t left = LeftChild (index);
      std::size_t tmp = Smallest (left, right);
      if (IsLessStrictly (index, tmp))
        {
          return;
        }
      Exch (index, tmp);
      index = tmp;
      right = RightChild (index);
    }
  if (IsBottom (index))
    {
      return;
    }
  NS_ASSERT (!IsBottom (index));
  std::size_t left = LeftChild (index);
  if (IsBottom (left))
    {
      return;
    }
  if (IsLessStrictly (index, left))
    {
      return;
    }
  Exch (index, left);
}


void
HeapScheduler::Insert (const Event &ev)
{
  NS_LOG_FUNCTION (this << &ev);
  m_heap.push_back (ev);
  BottomUp ();
}

Scheduler::Event
HeapScheduler::PeekNext (void) const
{
  NS_LOG_FUNCTION (this);
  return m_heap[Root ()];
}
Scheduler::Event
HeapScheduler::RemoveNext (void)
{
  NS_LOG_FUNCTION (this);
  Event next = m_heap[Root ()];
  Exch (Root (), Last ());
  m_heap.pop_back ();
  TopDown (Root ());
  return next;
}


void
HeapScheduler::Remove (const Event &ev)
{
  NS_LOG_FUNCTION (this << &ev);
  std::size_t uid = ev.key.m_uid;
  for (std::size_t i = 1; i < m_heap.size (); i++)
    {
      if (uid == m_heap[i].key.m_uid)
        {
          NS_ASSERT (m_heap[i].impl == ev.impl);
          Exch (i, Last ());
          m_heap.pop_back ();
          TopDown (i);
          return;
        }
    }
  NS_ASSERT (false);
}

} // namespace ns3

