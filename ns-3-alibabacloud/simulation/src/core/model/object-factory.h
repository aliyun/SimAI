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
#ifndef OBJECT_FACTORY_H
#define OBJECT_FACTORY_H

#include "attribute-construction-list.h"
#include "object.h"
#include "type-id.h"

/**
 * \file
 * \ingroup object
 * ns3::ObjectFactory class declaration.
 */

namespace ns3 {

class AttributeValue;

/**
 * \ingroup object
 *
 * \brief Instantiate subclasses of ns3::Object.
 *
 * This class can also hold a set of attributes to set
 * automatically during the object construction.
 *
 * \see attribute_ObjectFactory
 */
class ObjectFactory
{
public:
  /**
   * Default constructor.
   *
   * This factory is not capable of constructing a real Object
   * until it has at least a TypeId.
   */
  ObjectFactory ();
  /**
   * Construct a factory for a specific TypeId by name.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs
   * \param [in] typeId The name of the TypeId this factory should create.
   * \param [in] args A sequence of name-value pairs of additional attributes to set.
   *
   * The args sequence can be made of any number of pairs, each consisting of a
   * name (of std::string type) followed by a value (of const AttributeValue & type).
   */
  template <typename... Args>
  ObjectFactory (const std::string& typeId, Args&&... args);

  /**@{*/
  /**
   * Set the TypeId of the Objects to be created by this factory.
   *
   * \param [in] tid The TypeId of the object to instantiate.
   */
  void SetTypeId (TypeId tid);
  void SetTypeId (const char *tid);
  void SetTypeId (std::string tid);
  /**@}*/

  /**
   * Check if the ObjectFactory has been configured with a TypeId
   *
   * \return true if a TypeId has been configured to the ObjectFactory
   */
  bool IsTypeIdSet (void) const;

  /**
   * Set an attribute to be set during construction.
   *
   * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs
   * \param [in] name The name of the attribute to set.
   * \param [in] value The value of the attribute to set.
   * \param [in] args A sequence of name-value pairs of additional attributes to set.
   *
   * The args sequence can be made of any number of pairs, each consisting of a
   * name (of std::string type) followed by a value (of const AttributeValue & type).
   */
  template <typename... Args>
  void Set (const std::string &name, const AttributeValue &value, Args&&... args);

  /**
   * Base case to stop the recursion performed by the templated version of this
   * method.
   */
  void Set (void)
  { }

  /**
   * Get the TypeId which will be created by this ObjectFactory.
   * \returns The currently-selected TypeId.
   */
  TypeId GetTypeId (void) const;

  /**
   * Create an Object instance of the configured TypeId.
   *
   * \returns A new object instance.
   */
  Ptr<Object> Create (void) const;
  /**
   * Create an Object instance of the requested type.
   *
   * This method performs an extra call to ns3::Object::GetObject before
   * returning a pointer of the requested type to the user. This method
   * is really syntactical sugar.
   *
   * \tparam T \explicit The requested Object type.
   * \returns A new object instance.
   */
  template <typename T>
  Ptr<T> Create (void) const;

private:
  /**
   * Set an attribute to be set during construction.
   *
   * \param [in] name The name of the attribute to set.
   * \param [in] value The value of the attribute to set.
   */
  void DoSet (const std::string &name, const AttributeValue &value);
  /**
   * Print the factory configuration on an output stream.
   *
   * The configuration will be printed as a string with the form
   * "<TypeId-name>[<attribute-name>=<attribute-value>|...]"
   *
   * \param [in,out] os The stream.
   * \param [in] factory The ObjectFactory.
   * \returns The stream.
   */
  friend std::ostream & operator << (std::ostream &os, const ObjectFactory &factory);
  /**
   * Read a factory configuration from an input stream.
   *
   * The configuration should be in the form
   * "<TypeId-name>[<attribute-name>=<attribute-value>|...]"
   *
   * \param [in,out] is The input stream.
   * \param [out] factory The factory to configure as described by the stream.
   * \return The stream.
   */
  friend std::istream & operator >> (std::istream &is, ObjectFactory &factory);

  /** The TypeId this factory will create. */
  TypeId m_tid;
  /**
   * The list of attributes and values to be used in constructing
   * objects by this factory.
   */
  AttributeConstructionList m_parameters;
};

std::ostream & operator << (std::ostream &os, const ObjectFactory &factory);
std::istream & operator >> (std::istream &is, ObjectFactory &factory);


/**
 * \ingroup object
 * Allocate an Object on the heap and initialize with a set of attributes.
 *
 * \tparam T \explicit The requested Object type.
 * \tparam Args \deduced The type of the sequence of name-value pairs.
 * \param [in] args A sequence of name-value pairs of the attributes to set.
 * \returns A pointer to a newly allocated object.
 *
 * The args sequence can be made of any number of pairs, each consisting of a
 * name (of std::string type) followed by a value (of const AttributeValue & type).
 */
template <typename T, typename... Args>
Ptr<T>
CreateObjectWithAttributes (Args... args);


ATTRIBUTE_HELPER_HEADER (ObjectFactory);

} // namespace ns3


/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3 {

template <typename T>
Ptr<T>
ObjectFactory::Create (void) const
{
  Ptr<Object> object = Create ();
  return object->GetObject<T> ();
}

template <typename... Args>
ObjectFactory::ObjectFactory (const std::string& typeId, Args&&... args)
{
  SetTypeId (typeId);
  Set (args...);
}

template <typename... Args>
void
ObjectFactory::Set (const std::string &name, const AttributeValue & value, Args&&... args)
{
  DoSet (name, value);
  Set (args...);
}

template <typename T, typename... Args>
Ptr<T>
CreateObjectWithAttributes (Args... args)
{
  ObjectFactory factory;
  factory.SetTypeId (T::GetTypeId ());
  factory.Set (args...);
  return factory.Create<T> ();
}


} // namespace ns3

#endif /* OBJECT_FACTORY_H */
