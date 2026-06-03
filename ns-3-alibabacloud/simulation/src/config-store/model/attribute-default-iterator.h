/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  This program is free software; you can redistribute it and/or modify
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
 * Authors: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
 
#ifndef ATTRIBUTE_DEFAULT_ITERATOR_H
#define ATTRIBUTE_DEFAULT_ITERATOR_H

#include "ns3/type-id.h"
#include <string>

namespace ns3 {

/**
 * \ingroup configstore
 *
 * \brief Iterator to iterate on the default values of attributes of an ns3::Object
 */
class AttributeDefaultIterator
{
public:
  virtual ~AttributeDefaultIterator () = 0;
  /**
   * \brief This function will go through all the TypeIds and get only the attributes which are
   * explicit values (not vectors or pointer or arrays) and apply StartVisitTypeId
   * and VisitAttribute on the attributes in one TypeId. At the end of each TypeId
   * EndVisitTypeId is called.
   */
  void Iterate (void);
private:
  /**
   * \brief Begin the analysis of a TypeId
   * \param name TypeId name
   */
  virtual void StartVisitTypeId (std::string name);
  /**
   * \brief End the analysis of a TypeId
   */
  virtual void EndVisitTypeId (void);
  /**
   * \brief Visit an Attribute
   * \param tid the TypeId the attribute belongs to
   * \param name the Attribute name
   * \param defaultValue the attribute default value
   * \param index the index of the Attribute
   */
  virtual void VisitAttribute (TypeId tid, std::string name, std::string defaultValue, uint32_t index);
  /**
   * \brief Visit an Attribute
   * \param name the Attribute name
   * \param defaultValue the attribute default value
   */
  virtual void DoVisitAttribute (std::string name, std::string defaultValue);
};

} // namespace ns3

#endif /* ATTRIBUTE_DEFAULT_ITERATOR_H */
