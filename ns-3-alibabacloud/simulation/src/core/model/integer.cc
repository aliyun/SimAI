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
#include "integer.h"
#include "fatal-error.h"
#include "log.h"
#include <sstream>

/**
 * \file
 * \ingroup attribute_Integer
 * ns3::MakeIntegerChecker implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Integer");

ATTRIBUTE_VALUE_IMPLEMENT_WITH_NAME (int64_t, Integer);

namespace internal {

/**
 * \ingroup attribute_Integer
 * Make an Integer attribute checker with embedded numeric type name.
 *
 * \param [in] min The minimum allowed value.
 * \param [in] max The maximum allowed value.
 * \param [in] name The original type name ("int8_t", "int16_t", _etc_.).
 * \returns The AttributeChecker.
 */
Ptr<const AttributeChecker>
MakeIntegerChecker (int64_t min, int64_t max, std::string name)
{
  NS_LOG_FUNCTION (min << max << name);
  struct IntegerChecker : public AttributeChecker
  {
    IntegerChecker (int64_t minValue, int64_t maxValue, std::string name)
      : m_minValue (minValue),
        m_maxValue (maxValue),
        m_name (name)
    {}
    virtual bool Check (const AttributeValue &value) const
    {
      NS_LOG_FUNCTION (&value);
      const IntegerValue *v = dynamic_cast<const IntegerValue *> (&value);
      if (v == 0)
        {
          return false;
        }
      return v->Get () >= m_minValue && v->Get () <= m_maxValue;
    }
    virtual std::string GetValueTypeName (void) const
    {
      NS_LOG_FUNCTION_NOARGS ();
      return "ns3::IntegerValue";
    }
    virtual bool HasUnderlyingTypeInformation (void) const
    {
      NS_LOG_FUNCTION_NOARGS ();
      return true;
    }
    virtual std::string GetUnderlyingTypeInformation (void) const
    {
      NS_LOG_FUNCTION_NOARGS ();
      std::ostringstream oss;
      oss << m_name << " " << m_minValue << ":" << m_maxValue;
      return oss.str ();
    }
    virtual Ptr<AttributeValue> Create (void) const
    {
      NS_LOG_FUNCTION_NOARGS ();
      return ns3::Create<IntegerValue> ();
    }
    virtual bool Copy (const AttributeValue &src, AttributeValue &dst) const
    {
      NS_LOG_FUNCTION (&src << &dst);
      const IntegerValue *source = dynamic_cast<const IntegerValue *> (&src);
      IntegerValue *destination = dynamic_cast<IntegerValue *> (&dst);
      if (source == 0 || destination == 0)
        {
          return false;
        }
      *destination = *source;
      return true;
    }
    int64_t m_minValue;
    int64_t m_maxValue;
    std::string m_name;
  } *checker = new IntegerChecker (min, max, name);
  return Ptr<AttributeChecker> (checker, false);
}

} // namespace internal

} // namespace ns3
