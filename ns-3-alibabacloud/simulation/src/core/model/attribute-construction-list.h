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
#ifndef ATTRIBUTE_CONSTRUCTION_LIST_H
#define ATTRIBUTE_CONSTRUCTION_LIST_H

#include "attribute.h"
#include <list>

/**
 * \file
 * \ingroup object
 * ns3::AttributeConstructionList declaration.
 */

namespace ns3 {

/**
 * \ingroup object
 * List of Attribute name, value and checker triples used
 * to construct Objects.
 */
class AttributeConstructionList
{
public:
  /** A single Attribute triple */
  struct Item
  {
    /** Checker used to validate serialized values. */
    Ptr<const AttributeChecker> checker;
    /** The value of the Attribute. */
    Ptr<AttributeValue> value;
    /** The name of the Attribute. */
    std::string name;
  };
  /** Iterator type. */
  typedef std::list<struct Item>::const_iterator CIterator;

  /** Constructor */
  AttributeConstructionList ();

  /**
   * Add an Attribute to the list.
   *
   * \param [in] name The Attribute name.
   * \param [in] checker The checker to use for this Attribute.
   * \param [in] value The AttributeValue to add.
   */
  void Add (std::string name, Ptr<const AttributeChecker> checker,
            Ptr<AttributeValue> value);

  /**
   * Find an Attribute in the list from its AttributeChecker.
   *
   * \param [in] checker The AttributeChecker to find.  Typically this is the
   *             AttributeChecker from TypeId::AttributeInformation.
   * \returns The AttributeValue.
   */
  Ptr<AttributeValue> Find (Ptr<const AttributeChecker> checker) const;

  /** \returns The first item in the list */
  CIterator Begin (void) const;
  /** \returns The end of the list (iterator to one past the last). */
  CIterator End (void) const;

private:

  /** The list of Items */
  std::list<struct Item> m_list;
};

} // namespace ns3

#endif /* ATTRIBUTE_CONSTRUCTION_LIST_H */
