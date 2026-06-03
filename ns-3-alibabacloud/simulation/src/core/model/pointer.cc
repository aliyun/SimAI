/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "pointer.h"
#include "object-factory.h"
#include "log.h"
#include <sstream>

/**
 * \file
 * \ingroup attribute_Pointer
 * ns3::PointerValue attribute value implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Pointer");

PointerValue::PointerValue ()
  : m_value ()
{
  NS_LOG_FUNCTION (this);
}

PointerValue::PointerValue (Ptr<Object> object)
  : m_value (object)
{
  NS_LOG_FUNCTION (object);
}

void
PointerValue::SetObject (Ptr<Object> object)
{
  NS_LOG_FUNCTION (object);
  m_value = object;
}

Ptr<Object>
PointerValue::GetObject (void) const
{
  NS_LOG_FUNCTION (this);
  return m_value;
}

Ptr<AttributeValue>
PointerValue::Copy (void) const
{
  NS_LOG_FUNCTION (this);
  return Create<PointerValue> (*this);
}
std::string
PointerValue::SerializeToString (Ptr<const AttributeChecker> checker) const
{
  NS_LOG_FUNCTION (this << checker);
  std::ostringstream oss;
  oss << m_value;
  return oss.str ();
}

bool
PointerValue::DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker)
{
  // We assume that the string you want to deserialize contains
  // a description for an ObjectFactory to create an object and then assign it to the
  // member variable.
  NS_LOG_FUNCTION (this << value << checker);
  ObjectFactory factory;
  std::istringstream iss;
  iss.str (value);
  iss >> factory;
  if (iss.fail ())
    {
      return false;
    }
  m_value = factory.Create<Object> ();
  return true;
}

} // namespace ns3
