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

#ifndef ATTRIBUTE_CONTAINER_ACCESSOR_HELPER_H
#define ATTRIBUTE_CONTAINER_ACCESSOR_HELPER_H

#include "attribute-container.h"

#include <ns3/attribute-helper.h>


#include <type_traits>
#include <list>

namespace ns3 {

/**
 * SFINAE compile time check if type T has const iterator.
 */
template <typename T>
struct has_const_iterator
{
private:
  /// positive result
  typedef char yes;
  /// negative result
  typedef struct
  {
    char array[2]; //!< Check value, its size must be different than the "yes" size
  } no;

  /**
   * Test function, compiled if type has a const_iterator
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static yes test (typename C::const_iterator *);
  /**
   * Test function, compiled if type does not have a const_iterator
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static no test (...);

public:
  /// Value of the test - true if type has a const_iterator
  static const bool value = sizeof (test<T> (0)) == sizeof (yes);
  /// Equivalent name of the type T
  typedef T type;
};

/**
 * SFINAE compile time check if type T has begin() and end() methods.
 */
template <typename T>
struct has_begin_end
{
  /**
   * Compiled if type T has a begin() method.
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static char (&f (typename std::enable_if<
    std::is_same<decltype (static_cast<typename C::const_iterator (C::*) () const> (&C::begin)),
    typename C::const_iterator (C::*) () const>::value, void>::type *))[1];

  /** 
   * Compiled if type T does not have a begin() method.
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static char (&f (...))[2];

  /**
   * Compiled if type T has an end() method.
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static char (&g (typename std::enable_if<
    std::is_same<decltype (static_cast<typename C::const_iterator (C::*) () const> (&C::end)),
    typename C::const_iterator (C::*) () const>::value, void>::type *))[1];

  /**
   * Compiled if type T does not have an end() method.
   * \return A value indicating that this specialization has been compiled.
   */
  template <typename C> static char (&g (...))[2];

  /// True if type T has a begin() method.
  static bool const beg_value = sizeof (f<T> (0)) == 1;
  /// True if type T has an end() method.
  static bool const end_value = sizeof (g<T> (0)) == 1;
};

/**
 * Compile time check if type T is a container.
 *
 * Container here means has an iterator and supports begin() and end()
 * methods.
 * 
 * Can be used when defining specializations when a type T is an STL
 * like container.
 */
template<typename T> 
struct is_container : std::integral_constant<bool, has_const_iterator<T>::value && has_begin_end<T>::beg_value && has_begin_end<T>::end_value> 
{ };

/**
 * \ingroup attributeimpl
 *
 * DoMakeAccessorHelperOne specialization for member containers
 *
 * The template parameter list contains an extra parameter that is
 * intended to disambiguate an attribute container from any other
 * templated attribute, e.g Ptr or Callback.  Disambiguation is based
 * on begin/end and iterator.
 *
 * \tparam V  \explicit The specific AttributeValue type to use to represent
 *            the Attribute.
 * \tparam T  \deduced The class holding the data member.
 * \tparam U  \deduced The type of the container.
 * \tparam I  \deduced The type of item (s) within the container.
 * \param [in]  memberContainer  The address of the data member.
 * \returns The AttributeAccessor.
 */
template <typename V, typename T, template <typename...> class U, typename ...I,
          typename = typename std::enable_if< ( is_container< U<I...> >::value ), void>::type >
inline
Ptr<const AttributeAccessor>
DoMakeAccessorHelperOne (U<I...> T::*memberContainer)
{
  /* AttributeAcessor implementation for a class member variable. */
  class MemberContainer : public AccessorHelper<T,V>
  {
    public:
    /*
     * Construct from a class data member address.
     * \param [in] memberContainer The class data member address.
     */
    MemberContainer (U<I...> T::*memberContainer)
      : AccessorHelper<T,V> (),
        m_memberContainer (memberContainer)
    {}
    private:
    virtual bool DoSet (T *object, const V *v) const {
      // typename AccessorTrait<U<I>::value_type>::Result tmp;
      // bool ok = v->GetAccessor (tmp);
      // if (!ok)
      //   {
      //     return false;
      //   }
      auto src = v->Get ();
      (object->*m_memberContainer).clear ();
      std::copy (src.begin (), src.end (), std::inserter ((object->*m_memberContainer), (object->*m_memberContainer).end ()));
      return true;
    }
    virtual bool DoGet (const T *object, V *v) const {
      v->Set (object->*m_memberContainer);
      return true;
    }
    virtual bool HasGetter (void) const {
      return true;
    }
    virtual bool HasSetter (void) const {
      return true;
    }

    U<I...> T::*m_memberContainer;  // Address of the class data member.
  };
  return Ptr<const AttributeAccessor> (new MemberContainer (memberContainer), false);
}

} // namespace ns3

#endif  // ATTRIBUTE_CONTAINER_ACCESSOR_HELPER_H
