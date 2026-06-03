/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Caliola Engineering, LLC.
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
 * Author: Jared Dulmage <jared.dulmage@caliola.com>
 */

#ifndef ATTRIBUTE_CONTAINER_H
#define ATTRIBUTE_CONTAINER_H

#include <ns3/attribute-helper.h>
#include <ns3/string.h>

#include <list>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <utility>

namespace ns3 {

class AttributeChecker;

// A = attribute value type, C = container type to return
/**
 * A container for one type of attribute.
 * 
 * The container uses \p A to parse items into elements.
 * Internally the container is always a list but an instance
 * can return the items in a container specified by \p C.
 * 
 * @tparam A AttributeValue type to be contained.
 * @tparam C Possibly templated container class returned by Get.
 */
template <class A, template<class...> class C=std::list>
class AttributeContainerValue : public AttributeValue
{
public:
  /** AttributeValue (element) type. */
  typedef A attribute_type;
  /** Type actually stored within the container. */
  typedef Ptr<A> value_type;
  /** Internal container type. */
  typedef std::list<value_type> container_type;
  /** stl-style Const iterator type. */
  typedef typename container_type::const_iterator const_iterator;
  /** stl-style Non-const iterator type. */
  typedef typename container_type::iterator iterator;
  /** Size type for container. */
  typedef typename container_type::size_type size_type;
  /** NS3 style iterator type. */
  typedef typename AttributeContainerValue::const_iterator Iterator; // NS3 type

  // use underlying AttributeValue to get return element type
  /** Item type of container returned by Get. */
  typedef typename std::invoke_result_t<decltype(&A::Get), A> item_type;
  /** Type of container returned. */
  typedef C<item_type> result_type;

  /**
   * Default constructor.
   * \param[in] sep Character separator between elements for parsing.
   */
  AttributeContainerValue (char sep = ',');

  /**
   * Construct from another container.
   * @tparam CONTAINER \deduced type of container passed for initialization.
   * \param c Instance of CONTAINER with which to initialize AttributeContainerValue.
   */
  template <class CONTAINER>
  AttributeContainerValue (const CONTAINER &c);

  /**
   * Construct from iterators.
   * @tparam ITER \deduced type of iterator.
   * \param[in] begin Iterator that points to first initialization item.
   * \param[in] end Iterator that points ones past last initialization item.
   */
  template <class ITER>
  AttributeContainerValue (const ITER begin, const ITER end);

  /** Destructor. */
  ~AttributeContainerValue ();

  // Inherited
  Ptr<AttributeValue> Copy (void) const;
  bool DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker);
  std::string SerializeToString (Ptr<const AttributeChecker> checker) const;

  // defacto pure virtuals to integrate with built-in accessor code
  /**
   * Return a container of items.
   * \return Container of items.
   */
  result_type Get (void) const;
  /**
   * Copy items from container c.
   * 
   * This method assumes \p c has stl-style begin and end methods.
   * The AttributeContainerValue value is cleared before copying from \p c.
   * @tparam T type of container.
   * \param c Container from which to copy items.
   */
  template <class T>
  void Set (const T &c);

  // NS3 interface
  /**
   * NS3-style Number of items.
   * \return Number of items in container.
   */
  size_type GetN (void) const;
  /**
   * NS3-style beginning of container.
   * \return Iterator pointing to first time in container.
   */
  Iterator Begin (void);
  /**
   * NS3-style ending of container.
   * \return Iterator pointing one past last item of container.
   */
  Iterator End (void);

  // STL-interface
  /**
   * STL-style number of items in container
   * \return number of items in container.
   */
  size_type size (void) const;
  /**
   * STL-style beginning of container.
   * \return Iterator pointing to first item in container.
   */
  iterator begin (void);
  /**
   * STL-style end of container.
   * \return Iterator pointing to one past last item in container.
   */
  iterator end (void);
  /**
   * STL-style const beginning of container.
   * \return Const iterator pointing to first item in container.
   */
  const_iterator begin (void) const;
  /**
   * STL-style const end of container.
   * \return Const iterator pointing to one past last item in container.
   */
  const_iterator end (void) const;

private:
  /**
   * Copy items from \ref begin to \ref end. 
   * 
   * The internal container is cleared before values are copied
   * using the push_back method.
   * @tparam ITER \deduced iterator type
   * \param[in] begin Points to first item to copy
   * \param[in] end Points to one after last item to copy
   * \return This object with items copied.
   */
  template <class ITER>
  Ptr<AttributeContainerValue<A, C> > CopyFrom (const ITER begin, const ITER end);

  char m_sep;                 //!< Item separator
  container_type m_container; //!< Internal container
};

class AttributeContainerChecker : public AttributeChecker
{
public:
  /**
   * Set the item checker
   * \param itemchecker The item checker
   */
  virtual void SetItemChecker (Ptr<const AttributeChecker> itemchecker) = 0;
  /**
   * Get the item checker
   * \return The item checker
   */
  virtual Ptr<const AttributeChecker> GetItemChecker (void) const = 0;
};

/**
 * Make AttributeContainerChecker from AttributeContainerValue.
 * @tparam A \deduced AttributeValue type in container.
 * @tparam C \deduced Container type returned by Get.
 * \param[in] value AttributeContainerValue from which to deduce types.
 * \return AttributeContainerChecker for value.
 */
template <class A, template <class...> class C>
Ptr<AttributeChecker>
MakeAttributeContainerChecker (const AttributeContainerValue<A, C> &value);

/**
 * Make AttributeContainerChecker using explicit types, initialize item checker.
 * @tparam A AttributeValue type in container.
 * @tparam C Container type returned by Get.
 * \param[in] itemchecker AttributeChecker used for each item in the container.
 * \return AttributeContainerChecker.
 */
template <class A, template <class...> class C=std::list>
Ptr<const AttributeChecker>
MakeAttributeContainerChecker (Ptr<const AttributeChecker> itemchecker);

/**
 * Make unitialized AttributeContainerChecker using explicit types.
 * @tparam A AttributeValue type in container.
 * @tparam C Container type returned by Get.
 * \return AttributeContainerChecker.
 */
template <class A, template <class...> class C=std::list>
Ptr<AttributeChecker> MakeAttributeContainerChecker (void);

/**
 * Make AttributeContainerAccessor  using explicit types.
 * @tparam A AttributeValue type in container.
 * @tparam C Container type returned by Get.
 * \param[in] a1 AttributeContainer to be used.
 * \return AttributeContainerAccessor.
 */
template <typename A, template <typename...> class C=std::list, typename T1>
Ptr<const AttributeAccessor> MakeAttributeContainerAccessor (T1 a1);

} // namespace ns3

/*****************************************************************************
 * Implementation below
 *****************************************************************************/

namespace ns3 {

namespace internal {

/**
 * \internal
 * Templated AttributeContainerChecker class that is instantiated
 * in MakeAttributeContainerChecker. The non-templated base ns3::AttributeContainerChecker
 * is returned from that function. This is the same pattern as ObjectPtrContainer.
 */
template <class A, template <class...> class C>
class AttributeContainerChecker : public ns3::AttributeContainerChecker
{
public:
  AttributeContainerChecker (void);
  /**
   * Explicit constructor
   * \param itemchecker The AttributeChecker.
   */
  explicit AttributeContainerChecker (Ptr<const AttributeChecker> itemchecker);
  void SetItemChecker (Ptr<const AttributeChecker> itemchecker);
  Ptr<const AttributeChecker> GetItemChecker (void) const;

private:
  Ptr<const AttributeChecker> m_itemchecker; //!< The AttributeChecker
};

template <class A, template <class...> class C>
AttributeContainerChecker<A, C>::AttributeContainerChecker (void)
  : m_itemchecker (0)
{}

template <class A, template <class...> class C>
AttributeContainerChecker<A, C>::AttributeContainerChecker (Ptr<const AttributeChecker> itemchecker)
  : m_itemchecker (itemchecker)
{}

template <class A, template <class...> class C>
void
AttributeContainerChecker<A, C>::SetItemChecker (Ptr<const AttributeChecker> itemchecker)
{
  m_itemchecker = itemchecker;
}

template <class A, template <class...> class C>
Ptr<const AttributeChecker>
AttributeContainerChecker<A, C>::GetItemChecker (void) const
{
  return m_itemchecker;
}

} // namespace internal

template <class A, template <class...> class C>
Ptr<AttributeChecker>
MakeAttributeContainerChecker (const AttributeContainerValue<A, C> &value)
{
  return MakeAttributeContainerChecker <A, C> ();
}

template <class A, template <class...> class C>
Ptr<const AttributeChecker>
MakeAttributeContainerChecker (Ptr<const AttributeChecker> itemchecker)
{
  auto checker = MakeAttributeContainerChecker <A, C> ();
  auto acchecker = DynamicCast<AttributeContainerChecker> (checker);
  acchecker->SetItemChecker (itemchecker);
  return checker;
}

template <class A, template <class...> class C>
Ptr<AttributeChecker>
MakeAttributeContainerChecker (void)
{
  std::string containerType;
  std::string underlyingType;
  typedef AttributeContainerValue<A, C> T;
  {
    std::ostringstream oss;
    oss << "ns3::AttributeContainerValue<" << typeid (typename T::attribute_type).name ()
        << ", " << typeid (typename T::container_type).name () << ">";
    containerType = oss.str ();
  }

  {
    std::ostringstream oss;
    oss << "ns3::Ptr<" << typeid (typename T::attribute_type).name () << ">";
    underlyingType = oss.str ();
  }

  return MakeSimpleAttributeChecker<T, internal::AttributeContainerChecker<A, C> > (containerType, underlyingType);
}

template <class A, template <class...> class C>
AttributeContainerValue<A, C>::AttributeContainerValue (char sep)
  : m_sep (sep)
{

}

template <class A, template <class...> class C>
template <class CONTAINER>
AttributeContainerValue<A, C>::AttributeContainerValue (const CONTAINER &c)
  : AttributeContainerValue<A, C> (c.begin (), c.end ())
{

}

template <class A, template <class...> class C>
template <class ITER>
AttributeContainerValue<A, C>::AttributeContainerValue (const ITER begin, const ITER end)
  : AttributeContainerValue ()
{
  CopyFrom (begin, end);
}

template <class A, template <class...> class C>
AttributeContainerValue<A, C>::~AttributeContainerValue ()
{
  m_container.clear ();
}

template <class A, template <class...> class C>
Ptr<AttributeValue>
AttributeContainerValue<A, C>::Copy (void) const
{
  auto c = Create<AttributeContainerValue<A, C> > ();
  c->m_sep = m_sep;
  c->m_container = m_container;
  return c;
}

template <class A, template <class...> class C>
bool
AttributeContainerValue<A, C>::DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker)
{
  auto acchecker = DynamicCast<const AttributeContainerChecker> (checker);
  if (!acchecker) return false;

  std::istringstream iss (value); // copies value
  while (std::getline (iss, value, m_sep))
    {
      auto avalue = acchecker->GetItemChecker ()->CreateValidValue (StringValue (value));
      if (!avalue) return false;

      auto attr = DynamicCast <A> (avalue);
      if (!attr) return false;

      // TODO(jared): make insertion more generic?
      m_container.push_back (attr);
    }
  return true;
}

template <class A, template <class...> class C>
std::string
AttributeContainerValue<A, C>::SerializeToString (Ptr<const AttributeChecker> checker) const
{
  std::ostringstream oss;
  bool first = true;
  for (auto attr: *this)
    {
      if (!first) oss << m_sep;
      oss << attr->SerializeToString (checker);
      first = false;
    }
  return oss.str ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::result_type
AttributeContainerValue<A, C>::Get (void) const
{
  result_type c;
  for (const value_type& a: *this)
    c.insert (c.end (), a->Get ());
  return c;
}

template <class A, template <class...> class C>
template <class T>
void
AttributeContainerValue<A, C>::Set (const T &c)
{
  m_container.clear ();
  CopyFrom (c.begin (), c.end ());
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::size_type
AttributeContainerValue<A, C>::GetN (void) const
{
  return size ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::Iterator
AttributeContainerValue<A, C>::Begin (void)
{
  return begin ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::Iterator
AttributeContainerValue<A, C>::End (void)
{
  return end ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::size_type
AttributeContainerValue<A, C>::size (void) const
{
  return m_container.size ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::iterator
AttributeContainerValue<A, C>::begin (void)
{
  return m_container.begin ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::iterator
AttributeContainerValue<A, C>::end (void)
{
  return m_container.end ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::const_iterator
AttributeContainerValue<A, C>::begin (void) const
{
  return m_container.cbegin ();
}

template <class A, template <class...> class C>
typename AttributeContainerValue<A, C>::const_iterator
AttributeContainerValue<A, C>::end (void) const
{
  return m_container.cend ();
}

template <class A, template <class...> class C>
template <class ITER>
Ptr<AttributeContainerValue<A, C> >
AttributeContainerValue<A, C>::CopyFrom (const ITER begin, const ITER end)
{
  for (ITER iter = begin; iter != end; ++iter)
    {
      m_container.push_back (Create<A> (*iter));
    }
  return this;
}

template <typename A, template <typename...> class C, typename T1>
Ptr<const AttributeAccessor> MakeAttributeContainerAccessor (T1 a1)
{
  return MakeAccessorHelper<AttributeContainerValue<A, C> > (a1);
}

} // namespace ns3

#endif // ATTRIBUTE_CONTAINER_H
