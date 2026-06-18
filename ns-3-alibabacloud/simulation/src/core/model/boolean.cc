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
#include "boolean.h"
#include "fatal-error.h"
#include "log.h"

/**
 * \file
 * \ingroup attribute_Boolean
 * ns3::BooleanValue attribute value implementaation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Boolean");

BooleanValue::BooleanValue ()
  : m_value (false)
{
  NS_LOG_FUNCTION (this);
}
BooleanValue::BooleanValue (bool value)
  : m_value (value)
{
  NS_LOG_FUNCTION (this << value);
}
void
BooleanValue::Set (bool value)
{
  NS_LOG_FUNCTION (this << value);
  m_value = value;
}
bool
BooleanValue::Get (void) const
{
  NS_LOG_FUNCTION (this);
  return m_value;
}
BooleanValue::operator bool () const
{
  return m_value;
}

std::ostream & operator << (std::ostream &os, const BooleanValue &value)
{
  if (value.Get ())
    {
      os << "true";
    }
  else
    {
      os << "false";
    }
  return os;
}

Ptr<AttributeValue>
BooleanValue::Copy (void) const
{
  NS_LOG_FUNCTION (this);

  return Create<BooleanValue> (*this);
}
std::string
BooleanValue::SerializeToString (Ptr<const AttributeChecker> checker) const
{
  NS_LOG_FUNCTION (this << checker);

  if (m_value)
    {
      return "true";
    }
  else
    {
      return "false";
    }
}
bool
BooleanValue::DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker)
{
  NS_LOG_FUNCTION (this << value << checker);

  if (value == "true"
      || value == "1"
      || value == "t")
    {
      m_value = true;
      return true;
    }
  else if (value == "false"
           || value == "0"
           || value == "f")
    {
      m_value = false;
      return true;
    }
  else
    {
      return false;
    }
}


ATTRIBUTE_CHECKER_IMPLEMENT_WITH_NAME (Boolean,"bool");

} // namespace ns3
