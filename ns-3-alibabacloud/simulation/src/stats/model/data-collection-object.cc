/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: Tiago G. Rodrigues (tgr002@bucknell.edu)
 */

#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/boolean.h"

#include "data-collection-object.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DataCollectionObject");

NS_OBJECT_ENSURE_REGISTERED (DataCollectionObject);

TypeId
DataCollectionObject::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DataCollectionObject")
    .SetParent<Object> ()
    .SetGroupName ("Stats")
    .AddConstructor<DataCollectionObject> ()
    .AddAttribute ( "Name",
                    "Object's name",
                    StringValue ("unnamed"), MakeStringAccessor (&DataCollectionObject::GetName, &DataCollectionObject::SetName), MakeStringChecker ())
    .AddAttribute ( "Enabled",
                    "Object's enabled status",
                    BooleanValue (true), MakeBooleanAccessor (&DataCollectionObject::m_enabled), MakeBooleanChecker ())
  ;
  return tid;
}

DataCollectionObject::DataCollectionObject ()
{
}

DataCollectionObject::~DataCollectionObject ()
{
  NS_LOG_FUNCTION (this);
}

bool
DataCollectionObject::IsEnabled (void) const
{
  return m_enabled;
}

std::string
DataCollectionObject::GetName (void) const
{
  return m_name;
}

void
DataCollectionObject::SetName (std::string name)
{
  NS_LOG_FUNCTION (this << name);
  for (size_t pos = name.find (" "); pos != std::string::npos; pos = name.find (" ", pos + 1, 1))
    {
      name[pos] = '_';
    }

  m_name = name;
}

void
DataCollectionObject::Enable (void)
{
  NS_LOG_FUNCTION (this);
  m_enabled = true;
}

void
DataCollectionObject::Disable (void)
{
  NS_LOG_FUNCTION (this);
  m_enabled = false;
}

} // namespace ns3
