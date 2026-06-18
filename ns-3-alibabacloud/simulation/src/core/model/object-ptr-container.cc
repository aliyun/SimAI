/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA, Mathieu Lacage
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
 * Authors: Mathieu Lacage <mathieu.lacage@gmail.com>
 */
#include "object-ptr-container.h"
#include "log.h"

/**
 * \file
 * \ingroup attribute_ObjectPtrContainer
 * ns3::ObjectPtrContainerValue attribute value implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ObjectPtrContainer");

ObjectPtrContainerValue::ObjectPtrContainerValue ()
{
  NS_LOG_FUNCTION (this);
}

ObjectPtrContainerValue::Iterator
ObjectPtrContainerValue::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.begin ();
}
ObjectPtrContainerValue::Iterator
ObjectPtrContainerValue::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.end ();
}
std::size_t
ObjectPtrContainerValue::GetN (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.size ();
}
Ptr<Object>
ObjectPtrContainerValue::Get (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Iterator it = m_objects.find (i);
  Ptr<Object> value = 0;
  if ( it != m_objects.end () )
    {
      value = m_objects.find (i)->second;
    }
  return value;
}

Ptr<AttributeValue>
ObjectPtrContainerValue::Copy (void) const
{
  NS_LOG_FUNCTION (this);
  return ns3::Create<ObjectPtrContainerValue> (*this);
}
std::string
ObjectPtrContainerValue::SerializeToString (Ptr<const AttributeChecker> checker) const
{
  NS_LOG_FUNCTION (this << checker);
  std::ostringstream oss;
  Iterator it;
  for (it = Begin (); it != End (); ++it)
    {
      oss << (*it).second;
      if (it != End ())
        {
          oss << " ";
        }
    }
  return oss.str ();
}
bool
ObjectPtrContainerValue::DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker)
{
  NS_LOG_FUNCTION (this << value << checker);
  NS_FATAL_ERROR ("cannot deserialize a set of object pointers.");
  return true;
}

bool
ObjectPtrContainerAccessor::Set (ObjectBase * object, const AttributeValue & value) const
{
  // not allowed.
  NS_LOG_FUNCTION (this << object << &value);
  return false;
}
bool
ObjectPtrContainerAccessor::Get (const ObjectBase * object, AttributeValue &value) const
{
  NS_LOG_FUNCTION (this << object << &value);
  ObjectPtrContainerValue *v = dynamic_cast<ObjectPtrContainerValue *> (&value);
  if (v == 0)
    {
      return false;
    }
  v->m_objects.clear ();
  std::size_t n;
  bool ok = DoGetN (object, &n);
  if (!ok)
    {
      return false;
    }
  for (std::size_t i = 0; i < n; i++)
    {
      std::size_t index;
      Ptr<Object> o = DoGet (object, i, &index);
      v->m_objects[index] = o;
    }
  return true;
}
bool
ObjectPtrContainerAccessor::HasGetter (void) const
{
  NS_LOG_FUNCTION (this);
  return true;
}
bool
ObjectPtrContainerAccessor::HasSetter (void) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

} // name
