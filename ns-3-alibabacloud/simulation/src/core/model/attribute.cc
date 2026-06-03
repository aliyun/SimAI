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
#include "attribute.h"
#include "log.h"
#include "fatal-error.h"
#include "string.h"
#include <sstream>

/**
 * \file
 * \ingroup attributes
 * ns3::AttributeValue, ns3::AttributeAccessor and
 * ns3::AttributeChecker implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AttributeValue");

AttributeValue::AttributeValue ()
{}
AttributeValue::~AttributeValue ()
{}


AttributeAccessor::AttributeAccessor ()
{}
AttributeAccessor::~AttributeAccessor ()
{}

AttributeChecker::AttributeChecker ()
{}
AttributeChecker::~AttributeChecker ()
{}

Ptr<AttributeValue>
AttributeChecker::CreateValidValue (const AttributeValue &value) const
{
  NS_LOG_FUNCTION (this << &value);
  if (Check (value))
    {
      return value.Copy ();
    }
  // attempt to convert to string.
  const StringValue *str = dynamic_cast<const StringValue *> (&value);
  if (str == 0)
    {
      return 0;
    }
  // attempt to convert back to value.
  Ptr<AttributeValue> v = Create ();
  bool ok = v->DeserializeFromString (str->Get (), this);
  if (!ok)
    {
      return 0;
    }
  ok = Check (*v);
  if (!ok)
    {
      return 0;
    }
  return v;
}

EmptyAttributeValue::EmptyAttributeValue ()
{
  NS_LOG_FUNCTION (this);
}
Ptr<AttributeValue>
EmptyAttributeValue::Copy (void) const
{
  NS_LOG_FUNCTION (this);
  return Create<EmptyAttributeValue> ();
}
std::string
EmptyAttributeValue::SerializeToString (Ptr<const AttributeChecker> checker) const
{
  NS_LOG_FUNCTION (this << checker);
  return "";
}
bool
EmptyAttributeValue::DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker)
{
  NS_LOG_FUNCTION (this << value << checker);
  return true;
}

EmptyAttributeAccessor::EmptyAttributeAccessor () : AttributeAccessor ()
{}

EmptyAttributeAccessor::~EmptyAttributeAccessor ()
{}

bool
EmptyAttributeAccessor::Set (ObjectBase * object, const AttributeValue &value) const
{
  (void) object;
  (void) value;
  return true;
}

bool
EmptyAttributeAccessor::Get (const ObjectBase * object, AttributeValue &attribute) const
{
  (void) object;
  (void) attribute;
  return true;
}

bool
EmptyAttributeAccessor::HasGetter (void) const
{
  return false;
}

bool
EmptyAttributeAccessor::HasSetter (void) const
{
  return false;
}

EmptyAttributeChecker::EmptyAttributeChecker () : AttributeChecker ()
{}

EmptyAttributeChecker::~EmptyAttributeChecker ()
{}

bool
EmptyAttributeChecker::Check (const AttributeValue &value) const
{
  (void) value;
  return true;
}

std::string
EmptyAttributeChecker::GetValueTypeName (void) const
{
  return "EmptyAttribute";
}

bool
EmptyAttributeChecker::HasUnderlyingTypeInformation (void) const
{
  return false;
}

std::string
EmptyAttributeChecker::GetUnderlyingTypeInformation (void) const
{
  return "";
}

Ptr<AttributeValue>
EmptyAttributeChecker::Create (void) const
{
  static EmptyAttributeValue t;
  return Ptr<AttributeValue> (&t, false);
}

bool
EmptyAttributeChecker::Copy (const AttributeValue &source, AttributeValue &destination) const
{
  (void) source;
  (void) destination;
  return true;
}


} // namespace ns3
