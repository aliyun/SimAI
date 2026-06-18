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
#include "object-factory.h"
#include "log.h"
#include <sstream>

/**
 * \file
 * \ingroup object
 * ns3::ObjectFactory class implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ObjectFactory");

ObjectFactory::ObjectFactory ()
{
  NS_LOG_FUNCTION (this);
}

void
ObjectFactory::SetTypeId (TypeId tid)
{
  NS_LOG_FUNCTION (this << tid.GetName ());
  m_tid = tid;
}
void
ObjectFactory::SetTypeId (std::string tid)
{
  NS_LOG_FUNCTION (this << tid);
  m_tid = TypeId::LookupByName (tid);
}
void
ObjectFactory::SetTypeId (const char *tid)
{
  NS_LOG_FUNCTION (this << tid);
  m_tid = TypeId::LookupByName (tid);
}
bool
ObjectFactory::IsTypeIdSet (void) const
{
  if (m_tid.GetUid () != 0)
    {
      return true;
    }
  return false;
}
void
ObjectFactory::DoSet (const std::string &name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (this << name << &value);
  if (name == "")
    {
      return;
    }

  struct TypeId::AttributeInformation info;
  if (!m_tid.LookupAttributeByName (name, &info))
    {
      NS_FATAL_ERROR ("Invalid attribute set (" << name << ") on " << m_tid.GetName ());
      return;
    }
  Ptr<AttributeValue> v = info.checker->CreateValidValue (value);
  if (v == 0)
    {
      NS_FATAL_ERROR ("Invalid value for attribute set (" << name << ") on " << m_tid.GetName ());
      return;
    }
  m_parameters.Add (name, info.checker, value.Copy ());
}

TypeId
ObjectFactory::GetTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tid;
}

Ptr<Object>
ObjectFactory::Create (void) const
{
  NS_LOG_FUNCTION (this);
  Callback<ObjectBase *> cb = m_tid.GetConstructor ();
  ObjectBase *base = cb ();
  Object *derived = dynamic_cast<Object *> (base);
  NS_ASSERT (derived != 0);
  derived->SetTypeId (m_tid);
  derived->Construct (m_parameters);
  Ptr<Object> object = Ptr<Object> (derived, false);
  return object;
}

std::ostream & operator << (std::ostream &os, const ObjectFactory &factory)
{
  os << factory.m_tid.GetName () << "[";
  bool first = true;
  for (AttributeConstructionList::CIterator i = factory.m_parameters.Begin (); i != factory.m_parameters.End (); ++i)
    {
      os << i->name << "=" << i->value->SerializeToString (i->checker);
      if (first)
        {
          os << "|";
        }
    }
  os << "]";
  return os;
}
std::istream & operator >> (std::istream &is, ObjectFactory &factory)
{
  std::string v;
  is >> v;
  std::string::size_type lbracket, rbracket;
  lbracket = v.find ("[");
  rbracket = v.find ("]");
  if (lbracket == std::string::npos && rbracket == std::string::npos)
    {
      factory.SetTypeId (v);
      return is;
    }
  if (lbracket == std::string::npos || rbracket == std::string::npos)
    {
      return is;
    }
  NS_ASSERT (lbracket != std::string::npos);
  NS_ASSERT (rbracket != std::string::npos);
  std::string tid = v.substr (0, lbracket);
  std::string parameters = v.substr (lbracket + 1,rbracket - (lbracket + 1));
  factory.SetTypeId (tid);
  std::string::size_type cur;
  cur = 0;
  while (cur != parameters.size ())
    {
      std::string::size_type equal = parameters.find ("=", cur);
      if (equal == std::string::npos)
        {
          is.setstate (std::ios_base::failbit);
          break;
        }
      else
        {
          std::string name = parameters.substr (cur, equal - cur);
          struct TypeId::AttributeInformation info;
          if (!factory.m_tid.LookupAttributeByName (name, &info))
            {
              is.setstate (std::ios_base::failbit);
              break;
            }
          else
            {
              std::string::size_type next = parameters.find ("|", cur);
              std::string value;
              if (next == std::string::npos)
                {
                  value = parameters.substr (equal + 1, parameters.size () - (equal + 1));
                  cur = parameters.size ();
                }
              else
                {
                  value = parameters.substr (equal + 1, next - (equal + 1));
                  cur = next + 1;
                }
              Ptr<AttributeValue> val = info.checker->Create ();
              bool ok = val->DeserializeFromString (value, info.checker);
              if (!ok)
                {
                  is.setstate (std::ios_base::failbit);
                  break;
                }
              else
                {
                  factory.m_parameters.Add (name, info.checker, val);
                }
            }
        }
    }
  NS_ABORT_MSG_IF (is.bad (), "Failure to parse " << parameters);
  return is;
}


ATTRIBUTE_HELPER_CPP (ObjectFactory);

} // namespace ns3
