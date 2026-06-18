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
#include "global-value.h"
#include "fatal-error.h"
#include "attribute.h"
#include "string.h"
#include "uinteger.h"
#include "log.h"

#include "ns3/core-config.h"

#include <cstdlib>  // getenv
#include <cstring>  // strlen


/**
 * \file
 * \ingroup core
 * ns3::GlobalValue implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("GlobalValue");

GlobalValue::GlobalValue (std::string name, std::string help,
                          const AttributeValue &initialValue,
                          Ptr<const AttributeChecker> checker)
  : m_name (name),
    m_help (help),
    m_initialValue (0),
    m_currentValue (0),
    m_checker (checker)
{
  NS_LOG_FUNCTION (name << help << &initialValue << checker);
  if (m_checker == 0)
    {
      NS_FATAL_ERROR ("Checker should not be zero on " << name );
    }
  m_initialValue = m_checker->CreateValidValue (initialValue);
  m_currentValue = m_initialValue;
  if (m_initialValue == 0)
    {
      NS_FATAL_ERROR ("Value set by user on " << name << " is invalid.");
    }
  GetVector ()->push_back (this);
  InitializeFromEnv ();
}

void
GlobalValue::InitializeFromEnv (void)
{
  NS_LOG_FUNCTION (this);

  const char *envVar = getenv ("NS_GLOBAL_VALUE");
  if (envVar == 0 || std::strlen (envVar) == 0)
    {
      return;
    }
  std::string env = envVar;
  std::string::size_type cur = 0;
  std::string::size_type next = 0;
  while (next != std::string::npos)
    {
      next = env.find (";", cur);
      std::string tmp = std::string (env, cur, next - cur);
      std::string::size_type equal = tmp.find ("=");
      if (equal != std::string::npos)
        {
          std::string name = tmp.substr (0, equal);
          std::string value = tmp.substr (equal + 1, tmp.size () - equal - 1);
          if (name == m_name)
            {
              Ptr<AttributeValue> v = m_checker->CreateValidValue (StringValue (value));
              if (v != 0)
                {
                  m_initialValue = v;
                  m_currentValue = v;
                }
              return;
            }
        }
      cur = next + 1;
    }
}

std::string
GlobalValue::GetName (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_name;
}
std::string
GlobalValue::GetHelp (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return m_help;
}
void
GlobalValue::GetValue (AttributeValue &value) const
{
  NS_LOG_FUNCTION (&value);
  bool ok = m_checker->Copy (*m_currentValue, value);
  if (ok)
    {
      return;
    }
  StringValue *str = dynamic_cast<StringValue *> (&value);
  if (str == 0)
    {
      NS_FATAL_ERROR ("GlobalValue name=" << m_name << ": input value is not a string");
    }
  str->Set (m_currentValue->SerializeToString (m_checker));
}
Ptr<const AttributeChecker>
GlobalValue::GetChecker (void) const
{
  NS_LOG_FUNCTION (this);

  return m_checker;
}

bool
GlobalValue::SetValue (const AttributeValue &value)
{
  NS_LOG_FUNCTION (&value);

  Ptr<AttributeValue> v = m_checker->CreateValidValue (value);
  if (v == 0)
    {
      return 0;
    }
  m_currentValue = v;
  return true;
}

void
GlobalValue::Bind (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);

  for (Iterator i = Begin (); i != End (); i++)
    {
      if ((*i)->GetName () == name)
        {
          if (!(*i)->SetValue (value))
            {
              NS_FATAL_ERROR ("Invalid new value for global value: " << name);
            }
          return;
        }
    }
  NS_FATAL_ERROR ("Non-existant global value: " << name);
}
bool
GlobalValue::BindFailSafe (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);

  for (Iterator i = Begin (); i != End (); i++)
    {
      if ((*i)->GetName () == name)
        {
          return (*i)->SetValue (value);
        }
    }
  return false;
}
GlobalValue::Iterator
GlobalValue::Begin (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  return GetVector ()->begin ();
}
GlobalValue::Iterator
GlobalValue::End (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return GetVector ()->end ();
}

void
GlobalValue::ResetInitialValue (void)
{
  NS_LOG_FUNCTION (this);
  m_currentValue = m_initialValue;
}

bool
GlobalValue::GetValueByNameFailSafe (std::string name, AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);
  for (GlobalValue::Iterator gvit = GlobalValue::Begin (); gvit != GlobalValue::End (); ++gvit)
    {
      if ((*gvit)->GetName () == name)
        {
          (*gvit)->GetValue (value);
          return true;
        }
    }
  return false; // not found
}

void
GlobalValue::GetValueByName (std::string name, AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);
  if (!GetValueByNameFailSafe (name, value))
    {
      NS_FATAL_ERROR ("Could not find GlobalValue named \"" << name << "\"");
    }
}

GlobalValue::Vector *
GlobalValue::GetVector (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Vector vector;
  return &vector;
}

} // namespace ns3

