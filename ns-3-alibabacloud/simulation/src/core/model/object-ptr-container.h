/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA, Mathieu Lacage
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
#ifndef OBJECT_PTR_CONTAINER_H
#define OBJECT_PTR_CONTAINER_H

#include <map>
#include "object.h"
#include "ptr.h"
#include "attribute.h"

/**
 * \file
 * \ingroup attribute_ObjectPtrContainer
 * ns3::ObjectPtrContainerValue attribute value declarations and template implementations.
 */


namespace ns3 {

/**
 * \ingroup attribute_ObjectPtrContainer
 *
 * \brief Container for a set of ns3::Object pointers.
 *
 * This class it used to get attribute access to an array of
 * ns3::Object pointers.
 */
class ObjectPtrContainerValue : public AttributeValue
{
public:
  /** Iterator type for traversing this container. */
  typedef std::map<std::size_t, Ptr<Object> >::const_iterator Iterator;

  /** Default constructor. */
  ObjectPtrContainerValue ();

  /**
   * Get an iterator to the first Object.
   *
   * \returns An iterator to the first Object.
   */
  Iterator Begin (void) const;
  /**
   * Get an iterator to the _past-the-end_ Object.
   *
   * \returns An iterator to the _past-the-end_ Object.
   */
  Iterator End (void) const;
  /**
   * Get the number of Objects.
   *
   * \returns The number of objects.
   */
  std::size_t GetN (void) const;
  /**
   * Get a specific Object.
   *
   * \param [in] i The index of the requested object.
   * \returns The requested object
   */
  Ptr<Object> Get (std::size_t i) const;

  /**
   * Get a copy of this container.
   *
   * \returns A copy of this container.
   */
  virtual Ptr<AttributeValue> Copy (void) const;
  /**
   * Serialize each of the Object pointers to a string.
   *
   * Note this serializes the Ptr values, not the Objects themselves.
   *
   * \param [in] checker The checker to use (currently not used.)
   * \returns The string form of the Objects.
   */
  virtual std::string SerializeToString (Ptr<const AttributeChecker> checker) const;
  /**
   * Deserialize from a string. (Not implemented; raises a fatal error.)
   *
   * \param [in] value The serialized string form.
   * \param [in] checker The checker to use.
   * \returns \c true.
   */
  virtual bool DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker);

private:
  /** ObjectPtrContainerAccessor::Get() needs access. */
  friend class ObjectPtrContainerAccessor;
  /** The container implementation. */
  std::map<std::size_t, Ptr<Object> > m_objects;
};

/**
 * \ingroup attribute_ObjectPtrContainer
 * Create an AttributeAccessor using a container class indexed get method.
 *
 * The two versions of this function differ only in argument order.
 *
 * \tparam T \deduced The container class type.
 * \tparam U \deduced The type of object the get method returns.
 * \tparam INDEX \deduced The type of the index variable.
 * \param [in] get The class method to get a specific instance
 *             from the container.
 * \param [in] getN The class method to return the number of objects
 *             in the container.
 * \return The AttributeAccessor.
 */
template <typename T, typename U, typename INDEX>
Ptr<const AttributeAccessor>
MakeObjectPtrContainerAccessor (Ptr<U> (T::*get)(INDEX) const,
                                INDEX (T::*getN)(void) const);

/**
 * \ingroup attribute_ObjectPtrContainer
 * Create an AttributeAccessor using a container class indexed get method.
 *
 * The two versions of this function differ only in argument order.
 *
 * \tparam T \deduced The container class type.
 * \tparam U \deduced The type of object the get method returns.
 * \tparam INDEX \deduced The type of the index variable.
 * \param [in] get The class method to get a specific instance
 *             from the container.
 * \param [in] getN The class method to return the number of objects
 *             in the container.
 * \return The AttributeAccessor.
 */
template <typename T, typename U, typename INDEX>
Ptr<const AttributeAccessor>
MakeObjectPtrContainerAccessor (INDEX (T::*getN)(void) const,
                                Ptr<U> (T::*get)(INDEX) const);

class ObjectPtrContainerChecker : public AttributeChecker
{
public:
  /**
   * Get the TypeId of the container class type.
   * \returns The class TypeId.
   */
  virtual TypeId GetItemTypeId (void) const = 0;
};

template <typename T>
Ptr<const AttributeChecker> MakeObjectPtrContainerChecker (void);

} // namespace ns3


/***************************************************************
 *        The implementation of the above functions.
 ***************************************************************/

namespace ns3 {

namespace internal {

/** ObjectPtrContainerChecker implementation class. */
template <typename T>
class ObjectPtrContainerChecker : public ns3::ObjectPtrContainerChecker
{
public:
  virtual TypeId GetItemTypeId (void) const
  {
    return T::GetTypeId ();
  }
  virtual bool Check (const AttributeValue &value) const
  {
    return dynamic_cast<const ObjectPtrContainerValue *> (&value) != 0;
  }
  virtual std::string GetValueTypeName (void) const
  {
    return "ns3::ObjectPtrContainerValue";
  }
  virtual bool HasUnderlyingTypeInformation (void) const
  {
    return true;
  }
  virtual std::string GetUnderlyingTypeInformation (void) const
  {
    return "ns3::Ptr< " + T::GetTypeId ().GetName () + " >";
  }
  virtual Ptr<AttributeValue> Create (void) const
  {
    return ns3::Create<ObjectPtrContainerValue> ();
  }
  virtual bool Copy (const AttributeValue &source, AttributeValue &destination) const
  {
    const ObjectPtrContainerValue *src = dynamic_cast<const ObjectPtrContainerValue *> (&source);
    ObjectPtrContainerValue *dst = dynamic_cast<ObjectPtrContainerValue *> (&destination);
    if (src == 0 || dst == 0)
      {
        return false;
      }
    *dst = *src;
    return true;
  }
};

} // namespace internal


/**
 * \ingroup attribute_ObjectPtrContainer
 * AttributeAccessor implementation for ObjectPtrContainerValue.
 */
class ObjectPtrContainerAccessor : public AttributeAccessor
{
public:
  virtual bool Set (ObjectBase * object, const AttributeValue &value) const;
  virtual bool Get (const ObjectBase * object, AttributeValue &value) const;
  virtual bool HasGetter (void) const;
  virtual bool HasSetter (void) const;

private:
  /**
   * Get the number of instances in the container.
   *
   * \param [in] object The container object.
   * \param [out] n The number of instances in the container.
   * \returns true if the value could be obtained successfully.
   */
  virtual bool DoGetN (const ObjectBase *object, std::size_t *n) const = 0;
  /**
   * Get an instance from the container, identified by index.
   *
   * \param [in] object The container object.
   * \param [in] i The desired instance index.
   * \param [out] index The index retrieved.
   * \returns The index requested.
   */
  virtual Ptr<Object> DoGet (const ObjectBase *object, std::size_t i, std::size_t *index) const = 0;
};

template <typename T, typename U, typename INDEX>
Ptr<const AttributeAccessor>
MakeObjectPtrContainerAccessor (Ptr<U> (T::*get)(INDEX) const,
                                INDEX (T::*getN)(void) const)
{
  struct MemberGetters : public ObjectPtrContainerAccessor
  {
    virtual bool DoGetN (const ObjectBase *object, std::size_t *n) const
    {
      const T *obj = dynamic_cast<const T *> (object);
      if (obj == 0)
        {
          return false;
        }
      *n = (obj->*m_getN)();
      return true;
    }
    virtual Ptr<Object> DoGet (const ObjectBase *object, std::size_t i, std::size_t *index) const
    {
      const T *obj = static_cast<const T *> (object);
      *index = i;
      return (obj->*m_get)(i);
    }
    Ptr<U> (T::*m_get)(INDEX) const;
    INDEX (T::*m_getN)(void) const;
  } *spec = new MemberGetters ();
  spec->m_get = get;
  spec->m_getN = getN;
  return Ptr<const AttributeAccessor> (spec, false);
}

template <typename T, typename U, typename INDEX>
Ptr<const AttributeAccessor>
MakeObjectPtrContainerAccessor (INDEX (T::*getN)(void) const,
                                Ptr<U> (T::*get)(INDEX) const)
{
  return MakeObjectPtrContainerAccessor (get, getN);
}

template <typename T>
Ptr<const AttributeChecker> MakeObjectPtrContainerChecker (void)
{
  return Create<internal::ObjectPtrContainerChecker<T> > ();
}


} // namespace ns3

#endif /* OBJECT_PTR_CONTAINER_H */
