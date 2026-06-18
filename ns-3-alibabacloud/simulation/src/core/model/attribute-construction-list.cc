/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Mathieu Lacage
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
#include "attribute-construction-list.h"
#include "log.h"

/**
 * \file
 * \ingroup object
 * ns3::AttributeConstructionList implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AttributeConstructionList");

AttributeConstructionList::AttributeConstructionList ()
{
  NS_LOG_FUNCTION (this);
}

void
AttributeConstructionList::Add (std::string name, Ptr<const AttributeChecker> checker, Ptr<AttributeValue> value)
{
  // get rid of any previous value stored in this
  // vector of values.
  NS_LOG_FUNCTION (this << name << checker << value);

  for (std::list<struct Item>::iterator k = m_list.begin (); k != m_list.end (); k++)
    {
      if (k->checker == checker)
        {
          m_list.erase (k);
          break;
        }
    }
  // store the new value.
  struct Item attr;
  attr.checker = checker;
  attr.value = value;
  attr.name = name;
  m_list.push_back (attr);

}
Ptr<AttributeValue>
AttributeConstructionList::Find (Ptr<const AttributeChecker> checker) const
{
  NS_LOG_FUNCTION (this << checker);
  for (CIterator k = m_list.begin (); k != m_list.end (); k++)
    {
      NS_LOG_DEBUG ("Found " << k->name << " " << k->checker << " " << k->value);
      if (k->checker == checker)
        {
          return k->value;
        }
    }
  return 0;
}

AttributeConstructionList::CIterator
AttributeConstructionList::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_list.begin ();
}
AttributeConstructionList::CIterator
AttributeConstructionList::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_list.end ();
}

} // namespace ns3
