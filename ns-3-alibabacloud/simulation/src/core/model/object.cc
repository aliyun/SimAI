/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA, Gustavo Carneiro
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
 * Authors: Gustavo Carneiro <gjcarneiro@gmail.com>,
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "object.h"
#include "object-factory.h"
#include "assert.h"
#include "attribute.h"
#include "log.h"
#include "string.h"
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cstring>

/**
 * \file
 * \ingroup object
 * ns3::Object class implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Object");

/*********************************************************************
 *         The Object implementation
 *********************************************************************/

NS_OBJECT_ENSURE_REGISTERED (Object);

Object::AggregateIterator::AggregateIterator ()
  : m_object (0),
    m_current (0)
{
  NS_LOG_FUNCTION (this);
}

uint32_t Object::GetAggregatesNum() {
  return m_aggregates->n;
}

bool
Object::AggregateIterator::HasNext (void) const
{
  NS_LOG_FUNCTION (this);
  return m_current < m_object->m_aggregates->n;
}
Ptr<const Object>
Object::AggregateIterator::Next (void)
{
  NS_LOG_FUNCTION (this);
  Object *object = m_object->m_aggregates->buffer[m_current];
  m_current++;
  return object;
}
Object::AggregateIterator::AggregateIterator (Ptr<const Object> object)
  : m_object (object),
    m_current (0)
{
  NS_LOG_FUNCTION (this << object);
}


TypeId
Object::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tid;
}

TypeId
Object::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Object")
    .SetParent<ObjectBase> ()
    .SetGroupName ("Core")
  ;
  return tid;
}


Object::Object ()
  : m_tid (Object::GetTypeId ()),
    m_disposed (false),
    m_initialized (false),
    m_aggregates ((struct Aggregates *) std::malloc (sizeof (struct Aggregates))),
    m_getObjectCount (0)
{
  NS_LOG_FUNCTION (this);
  m_aggregates->n = 1;
  m_aggregates->buffer[0] = this;
}
Object::~Object ()
{
  // remove this object from the aggregate list
  NS_LOG_FUNCTION (this);
  uint32_t n = m_aggregates->n;
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (current == this)
        {
          std::memmove (&m_aggregates->buffer[i],
                        &m_aggregates->buffer[i + 1],
                        sizeof (Object *) * (m_aggregates->n - (i + 1)));
          m_aggregates->n--;
        }
    }
  // finally, if all objects have been removed from the list,
  // delete the aggregate list
  if (m_aggregates->n == 0)
    {
      std::free (m_aggregates);
    }
  m_aggregates = 0;
}
Object::Object (const Object &o)
  : m_tid (o.m_tid),
    m_disposed (false),
    m_initialized (false),
    m_aggregates ((struct Aggregates *) std::malloc (sizeof (struct Aggregates))),
    m_getObjectCount (0)
{
  m_aggregates->n = 1;
  m_aggregates->buffer[0] = this;
}
void
Object::Construct (const AttributeConstructionList &attributes)
{
  NS_LOG_FUNCTION (this << &attributes);
  ConstructSelf (attributes);
}

Ptr<Object>
Object::DoGetObject (TypeId tid) const
{
  NS_LOG_FUNCTION (this << tid);
  NS_ASSERT (CheckLoose ());

  uint32_t n = m_aggregates->n;
  TypeId objectTid = Object::GetTypeId ();
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      TypeId cur = current->GetInstanceTypeId ();
      while (cur != tid && cur != objectTid)
        {
          cur = cur.GetParent ();
        }
      if (cur == tid)
        {
#ifndef NS3_MTP
          // This is an attempt to 'cache' the result of this lookup.
          // the idea is that if we perform a lookup for a TypeId on this object,
          // we are likely to perform the same lookup later so, we make sure
          // that the aggregate array is sorted by the number of accesses
          // to each object.

          // first, increment the access count
          current->m_getObjectCount++;
          // then, update the sort
          UpdateSortedArray (m_aggregates, i);
#endif
          // finally, return the match
          return const_cast<Object *> (current);
        }
    }
  return 0;
}
void
Object::Initialize (void)
{
  /**
   * Note: the code here is a bit tricky because we need to protect ourselves from
   * modifications in the aggregate array while DoInitialize is called. The user's
   * implementation of the DoInitialize method could call GetObject (which could
   * reorder the array) and it could call AggregateObject which would add an
   * object at the end of the array. To be safe, we restart iteration over the
   * array whenever we call some user code, just in case.
   */
  NS_LOG_FUNCTION (this);
restart:
  uint32_t n = m_aggregates->n;
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (!current->m_initialized)
        {
          current->DoInitialize ();
          current->m_initialized = true;
          goto restart;
        }
    }
}
bool
Object::IsInitialized (void) const
{
  NS_LOG_FUNCTION (this);
  return m_initialized;
}
void
Object::Dispose (void)
{
  /**
   * Note: the code here is a bit tricky because we need to protect ourselves from
   * modifications in the aggregate array while DoDispose is called. The user's
   * DoDispose implementation could call GetObject (which could reorder the array)
   * and it could call AggregateObject which would add an object at the end of the array.
   * So, to be safe, we restart the iteration over the array whenever we call some
   * user code.
   */
  NS_LOG_FUNCTION (this);
restart:
  uint32_t n = m_aggregates->n;
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (!current->m_disposed)
        {
          current->DoDispose ();
          current->m_disposed = true;
          goto restart;
        }
    }
}
void
Object::UpdateSortedArray (struct Aggregates *aggregates, uint32_t j) const
{
  NS_LOG_FUNCTION (this << aggregates << j);
  while (j > 0
         && aggregates->buffer[j]->m_getObjectCount > aggregates->buffer[j - 1]->m_getObjectCount)
    {
      Object *tmp = aggregates->buffer[j - 1];
      aggregates->buffer[j - 1] = aggregates->buffer[j];
      aggregates->buffer[j] = tmp;
      j--;
    }
}
void
Object::AggregateObject (Ptr<Object> o)
{
  NS_LOG_FUNCTION (this << o);
  NS_ASSERT (!m_disposed);
  NS_ASSERT (!o->m_disposed);
  NS_ASSERT (CheckLoose ());
  NS_ASSERT (o->CheckLoose ());

  Object *other = PeekPointer (o);
  // first create the new aggregate buffer.
  uint32_t total = m_aggregates->n + other->m_aggregates->n;
  struct Aggregates *aggregates =
    (struct Aggregates *)std::malloc (sizeof(struct Aggregates) + (total - 1) * sizeof(Object*));
  aggregates->n = total;

  // copy our buffer to the new buffer
  std::memcpy (&aggregates->buffer[0],
               &m_aggregates->buffer[0],
               m_aggregates->n * sizeof(Object*));

  // append the other buffer into the new buffer too
  for (uint32_t i = 0; i < other->m_aggregates->n; i++)
    {
      aggregates->buffer[m_aggregates->n + i] = other->m_aggregates->buffer[i];
      const TypeId typeId = other->m_aggregates->buffer[i]->GetInstanceTypeId ();
      if (DoGetObject (typeId))
        {
          NS_FATAL_ERROR ("Object::AggregateObject(): "
                          "Multiple aggregation of objects of type " <<
                          other->GetInstanceTypeId () <<
                          " on objects of type " << typeId);
        }
      UpdateSortedArray (aggregates, m_aggregates->n + i);
    }

  // keep track of the old aggregate buffers for the iteration
  // of NotifyNewAggregates
  struct Aggregates *a = m_aggregates;
  struct Aggregates *b = other->m_aggregates;

  // Then, assign the new aggregation buffer to every object
  uint32_t n = aggregates->n;
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = aggregates->buffer[i];
      current->m_aggregates = aggregates;
    }

  // Finally, call NotifyNewAggregate on all the objects aggregates together.
  // We purposely use the old aggregate buffers to iterate over the objects
  // because this allows us to assume that they will not change from under
  // our feet, even if our users call AggregateObject from within their
  // NotifyNewAggregate method.
  for (uint32_t i = 0; i < a->n; i++)
    {
      Object *current = a->buffer[i];
      current->NotifyNewAggregate ();
    }
  for (uint32_t i = 0; i < b->n; i++)
    {
      Object *current = b->buffer[i];
      current->NotifyNewAggregate ();
    }

  // Now that we are done with them, we can free our old aggregate buffers
  std::free (a);
  std::free (b);
}
/**
 * This function must be implemented in the stack that needs to notify
 * other stacks connected to the node of their presence in the node.
 */
void
Object::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);
}

Object::AggregateIterator
Object::GetAggregateIterator (void) const
{
  NS_LOG_FUNCTION (this);
  return AggregateIterator (this);
}

void
Object::SetTypeId (TypeId tid)
{
  NS_LOG_FUNCTION (this << tid);
  NS_ASSERT (Check ());
  m_tid = tid;
}

void
Object::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (!m_disposed);
}

void
Object::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (!m_initialized);
}

bool
Object::Check (void) const
{
  NS_LOG_FUNCTION (this);
  return (GetReferenceCount () > 0);
}

/* In some cases, when an event is scheduled against a subclass of
 * Object, and if no one owns a reference directly to this object, the
 * object is alive, has a refcount of zero and the method ran when the
 * event expires runs against the raw pointer which means that we are
 * manipulating an object with a refcount of zero.  So, instead we
 * check the aggregate reference count.
 */
bool
Object::CheckLoose (void) const
{
  NS_LOG_FUNCTION (this);
  bool nonZeroRefCount = false;
  uint32_t n = m_aggregates->n;
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (current->GetReferenceCount ())
        {
          nonZeroRefCount = true;
          break;
        }
    }
  return nonZeroRefCount;
}
void
Object::DoDelete (void)
{
  // check if we really need to die
  NS_LOG_FUNCTION (this);
  for (uint32_t i = 0; i < m_aggregates->n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (current->GetReferenceCount () > 0)
        {
          return;
        }
    }

  // Now, we know that we are alone to use this aggregate so,
  // we can dispose and delete everything safely.

  uint32_t n = m_aggregates->n;
  // Ensure we are disposed.
  for (uint32_t i = 0; i < n; i++)
    {
      Object *current = m_aggregates->buffer[i];
      if (!current->m_disposed)
        {
          current->DoDispose ();
        }
    }

  // Now, actually delete all objects
  struct Aggregates *aggregates = m_aggregates;
  for (uint32_t i = 0; i < n; i++)
    {
      // There is a trick here: each time we call delete below,
      // the deleted object is removed from the aggregate buffer
      // in the destructor so, the index of the next element to
      // lookup is always zero
      Object *current = aggregates->buffer[0];
      delete current;
    }
}
} // namespace ns3

