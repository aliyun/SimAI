/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef PTR_H
#define PTR_H

#include <iostream>
#include <stdint.h>
#include "assert.h"

/**
 * \file
 * \ingroup ptr
 * ns3::Ptr smart pointer declaration and implementation.
 */

namespace ns3 {

/**
 * \ingroup core
 * \defgroup ptr Smart Pointer
 * \brief Heap memory management.
 *
 * See \ref ns3::Ptr for implementation details.
 *
 * See \ref main-ptr.cc for example usage.
 */
/**
 * \ingroup ptr
 *
 * \brief Smart pointer class similar to \c boost::intrusive_ptr.
 *
 * This smart-pointer class assumes that the underlying
 * type provides a pair of \c Ref and \c Unref methods which are
 * expected to increment and decrement the internal reference count
 * of the object instance.  You can add \c Ref and \c Unref
 * to a class simply by inheriting from ns3::SimpleRefCount.
 *
 * This implementation allows you to manipulate the smart pointer
 * as if it was a normal pointer: you can compare it with zero,
 * compare it against other pointers, assign zero to it, etc.
 *
 * It is possible to extract the raw pointer from this
 * smart pointer with the GetPointer() and PeekPointer() methods.
 *
 * If you want to store a \c new object into a smart pointer,
 * we recommend you to use the Create() template functions
 * to create the object and store it in a smart pointer to avoid
 * memory leaks. These functions are really small convenience
 * functions and their goal is just is save you a small
 * bit of typing.
 *
 * \tparam T \explicit The type of the underlying object.
 */
template <typename T>
class Ptr
{
private:

  /** The pointer. */
  T *m_ptr;

  /** Helper to test for null pointer. */
  class Tester
  {
  private:
    /** Disable delete (by virtue that this is unimplemented). */
    void operator delete (void *);
  };

  /** Interoperate with const instances. */
  friend class Ptr<const T>;

  /**
   * Get a permanent pointer to the underlying object.
   *
   * The underlying refcount is incremented prior
   * to returning to the caller so the caller is
   * responsible for calling Unref himself.
   *
   * \tparam U \deduced The actual type of the argument and return pointer.
   * \param [in] p Smart pointer
   * \return The pointer managed by this smart pointer.
   */
  template <typename U>
  friend U * GetPointer (const Ptr<U> &p);
  /**
   * Get a temporary pointer to the underlying object.
   *
   * The underlying refcount is not incremented prior
   * to returning to the caller so the caller is not
   * responsible for calling Unref himself.
   *
   * \tparam U \deduced The actual type of the argument and return pointer.
   * \param [in] p Smart pointer
   * \return The pointer managed by this smart pointer.
   */
  template <typename U>
  friend U * PeekPointer (const Ptr<U> &p);

  /** Mark this as a a reference by incrementing the reference count. */
  inline void Acquire (void) const;

public:
  /** Create an empty smart pointer */
  Ptr ();
  /**
   * Create a smart pointer which points to the object pointed to by
   * the input raw pointer ptr. This method creates its own reference
   * to the pointed object. The caller is responsible for Unref()'ing
   * its own reference, and the smart pointer will eventually do the
   * same, so that object is deleted if no more references to it
   * remain.
   *
   * \param [in] ptr Raw pointer to manage
   */
  Ptr (T *ptr);
  /**
   * Create a smart pointer which points to the object pointed to by
   * the input raw pointer ptr.
   *
   * \param [in] ptr Raw pointer to manage
   * \param [in] ref if set to true, this method calls Ref, otherwise,
   *        it does not call Ref.
   */
  Ptr (T *ptr, bool ref);
  /**
   * Copy by referencing the same underlying object.
   *
   * \param [in] o The other Ptr instance.
   */
  Ptr (Ptr const&o);
  /**
   * Copy, removing \c const qualifier.
   *
   * \tparam U \deduced The type underlying the Ptr being copied.
   * \param [in] o The Ptr to copy.
   */
  template <typename U>
  Ptr (Ptr<U> const &o);
  /** Destructor. */
  ~Ptr ();
  /**
   * Assignment operator by referencing the same underlying object.
   *
   * \param [in] o The other Ptr instance.
   * \return A reference to self.
   */
  Ptr<T> &operator = (Ptr const& o);
  /**
   * An rvalue member access.
   * \returns A pointer to the underlying object.
   */
  T *operator -> () const;
  /**
   * An lvalue member access.
   * \returns A pointer to the underlying object.
   */
  T *operator -> ();
  /**
   * A \c const dereference.
   * \returns A pointer to the underlying object.
   */
  T &operator * () const;
  /**
   * A  dereference.
   * \returns A pointer to the underlying object.
   */
  T &operator * ();
  /**
   * Test for NULL pointer.
   *
   * This enables simple NULL pointer checks like
   * \code
   *   Ptr<..> p = ...;
   *   if (!p) ...
   * \endcode
   * \returns \c true if the underlying pointer is NULL.
   */
  bool operator! ();
  /**
   * Test for non-NULL pointer.
   *
   * This enables simple pointer checks like
   * \code
   *   Ptr<...> p = ...;
   *   if (p) ...
   * \endcode
   * This also disables deleting a Ptr
   */
  operator Tester * () const;
};

/**
 * \ingroup ptr
 * Create class instances by constructors with varying numbers
 * of arguments and return them by Ptr.
 *
 * This template work for any class \c T derived from ns3::SimpleRefCount
 *
 * \see CreateObject for methods to create derivatives of ns3::Object
 */
/** @{ */
/**
 * \tparam T  \explicit The type of class object to create.
 * \tparam Ts \deduced Types of the constructor arguments.
 * \param  [in] args Constructor arguments.
 * \return A Ptr to the newly created \c T.
 */
template <typename T,
          typename... Ts>
Ptr<T> Create (Ts&&... args);

/** @}*/

/**
 * \ingroup ptr
 * Output streamer.
 * \tparam T \deduced The type of the underlying Object.
 * \param [in,out] os The output stream.
 * \param [in] p The Ptr.
 * \returns The stream.
 */
template <typename T>
std::ostream &operator << (std::ostream &os, const Ptr<T> &p);

/**
 * \ingroup ptr
 * Equality operator.
 *
 * This enables code such as
 * \code
 *   Ptr<...> p = ...;
 *   Ptr<...> q = ...;
 *   if (p == q) ...
 * \endcode
 *
 * Note that either \c p or \c q could also be ordinary pointers
 * to the underlying object.
 *
 * \tparam T1 \deduced Type of the object on the lhs.
 * \tparam T2 \deduced Type of the object on the rhs.
 * \param [in] lhs The left operand.
 * \param [in] rhs The right operand.
 * \return \c true if the operands point to the same underlying object.
 */
/** @{ */
template <typename T1, typename T2>
bool operator == (Ptr<T1> const &lhs, T2 const *rhs);

template <typename T1, typename T2>
bool operator == (T1 const *lhs, Ptr<T2> &rhs);

template <typename T1, typename T2>
bool operator == (Ptr<T1> const &lhs, Ptr<T2> const &rhs);
/**@}*/

/**
 * \ingroup ptr
 * Inequality operator.
 *
 * This enables code such as
 * \code
 *   Ptr<...> p = ...;
 *   Ptr<...> q = ...;
 *   if (p != q) ...
 * \endcode
 *
 * Note that either \c p or \c q could also be ordinary pointers
 * to the underlying object.
 *
 * \tparam T1 \deduced Type of the object on the lhs.
 * \tparam T2 \deduced Type of the object on the rhs.
 * \param [in] lhs The left operand.
 * \param [in] rhs The right operand.
 * \return \c true if the operands point to the same underlying object.
 */
/** @{ */
template <typename T1, typename T2>
bool operator != (Ptr<T1> const &lhs, T2 const *rhs);

template <typename T1, typename T2>
bool operator != (T1 const *lhs, Ptr<T2> &rhs);

template <typename T1, typename T2>
bool operator != (Ptr<T1> const &lhs, Ptr<T2> const &rhs);
/**@}*/


/**
 * \ingroup ptr
 * Comparison operator applied to the underlying pointers.
 *
 * \tparam T \deduced The type of the operands.
 * \param [in] lhs The left operand.
 * \param [in] rhs The right operand.
 * \return The comparison on the underlying pointers.
 */
/** @{ */
template <typename T>
bool operator < (const Ptr<T> &lhs, const Ptr<T> &rhs);
template <typename T>
bool operator< (const Ptr<T> &lhs, const Ptr<const T> &rhs);
template <typename T>
bool operator< (const Ptr<const T> &lhs, const Ptr<T> &rhs);
template <typename T>
bool operator<= (const Ptr<T> &lhs, const Ptr<T> &rhs);
template <typename T>
bool operator > (const Ptr<T> &lhs, const Ptr<T> &rhs);
template <typename T>
bool operator >= (const Ptr<T> &lhs, const Ptr<T> &rhs);
/** @} */

/**
 * Return a copy of \c p with its stored pointer const casted from
 * \c T2 to \c T1.
 *
 * \tparam T1 \deduced The type to return in a Ptr.
 * \tparam T2 \deduced The type of the underlying object.
 * \param [in] p The original \c const Ptr.
 * \return A non-const Ptr.
 */
template <typename T1, typename T2>
Ptr<T1> const_pointer_cast (Ptr<T2> const&p);

// Duplicate of struct CallbackTraits<T> as defined in callback.h
template <typename T>
struct CallbackTraits;

/**
 * \ingroup callbackimpl
 *
 * Trait class to convert a pointer into a reference,
 * used by MemPtrCallBackImpl.
 *
 * This is the specialization for Ptr types.
 *
 * \tparam T \deduced The type of the underlying object.
 */
template <typename T>
struct CallbackTraits<Ptr<T> >
{
  /**
   * \param [in] p Object pointer
   * \return A reference to the object pointed to by p
   */
  static T & GetReference (Ptr<T> const p)
  {
    return *PeekPointer (p);
  }
};

// Duplicate of struct EventMemberImplObjTraits<T> as defined in make-event.h
// We repeat it here to declare a specialization on Ptr<T>
// without making this header dependent on make-event.h
template <typename T>
struct EventMemberImplObjTraits;

/**
 * \ingroup makeeventmemptr
 * Helper for the MakeEvent functions which take a class method.
 *
 * This is the specialization for Ptr types.
 *
 * \tparam T \deduced The type of the underlying object.
 */
template <typename T>
struct EventMemberImplObjTraits<Ptr<T> >
{
  /**
   * \param [in] p Object pointer
   * \return A reference to the object pointed to by p
   */
  static T & GetReference (Ptr<T> p)
  {
    return *PeekPointer (p);
  }
};



} // namespace ns3


namespace ns3 {

/*************************************************
 *  friend non-member function implementations
 ************************************************/

template <typename T, typename... Ts>
Ptr<T> Create (Ts&&... args)
{
  return Ptr<T> (new T (std::forward<Ts> (args)...), false);
}

template <typename U>
U * PeekPointer (const Ptr<U> &p)
{
  return p.m_ptr;
}

template <typename U>
U * GetPointer (const Ptr<U> &p)
{
  p.Acquire ();
  return p.m_ptr;
}

template <typename T>
std::ostream &operator << (std::ostream &os, const Ptr<T> &p)
{
  os << PeekPointer (p);
  return os;
}

template <typename T1, typename T2>
bool
operator == (Ptr<T1> const &lhs, T2 const *rhs)
{
  return PeekPointer (lhs) == rhs;
}

template <typename T1, typename T2>
bool
operator == (T1 const *lhs, Ptr<T2> &rhs)
{
  return lhs == PeekPointer (rhs);
}

template <typename T1, typename T2>
bool
operator != (Ptr<T1> const &lhs, T2 const *rhs)
{
  return PeekPointer (lhs) != rhs;
}

template <typename T1, typename T2>
bool
operator != (T1 const *lhs, Ptr<T2> &rhs)
{
  return lhs != PeekPointer (rhs);
}

template <typename T1, typename T2>
bool
operator == (Ptr<T1> const &lhs, Ptr<T2> const &rhs)
{
  return PeekPointer (lhs) == PeekPointer (rhs);
}

template <typename T1, typename T2>
bool
operator != (Ptr<T1> const &lhs, Ptr<T2> const &rhs)
{
  return PeekPointer (lhs) != PeekPointer (rhs);
}

template <typename T>
bool operator < (const Ptr<T> &lhs, const Ptr<T> &rhs)
{
  return PeekPointer<T> (lhs) < PeekPointer<T> (rhs);
}

template <typename T>
bool
operator< (const Ptr<T> &lhs, const Ptr<const T> &rhs)
{
  return PeekPointer<T> (lhs) < PeekPointer<const T> (rhs);
}

template <typename T>
bool
operator< (const Ptr<const T> &lhs, const Ptr<T> &rhs)
{
  return PeekPointer<const T> (lhs) < PeekPointer<T> (rhs);
}

template <typename T>
bool
operator<= (const Ptr<T> &lhs, const Ptr<T> &rhs)
{
  return PeekPointer<T> (lhs) <= PeekPointer<T> (rhs);
}

template <typename T>
bool operator > (const Ptr<T> &lhs, const Ptr<T> &rhs)
{
  return PeekPointer<T> (lhs) > PeekPointer<T> (rhs);
}

template <typename T>
bool operator >= (const Ptr<T> &lhs, const Ptr<T> &rhs)
{
  return PeekPointer<T> (lhs) >= PeekPointer<T> (rhs);
}

/**
 * Cast a Ptr.
 *
 * \tparam T1 \deduced The desired type to cast to.
 * \tparam T2 \deduced The type of the original Ptr.
 * \param [in] p The original Ptr.
 * \return The result of the cast.
 */
/** @{ */
template <typename T1, typename T2>
Ptr<T1>
ConstCast (Ptr<T2> const&p)
{
  return Ptr<T1> (const_cast<T1 *> (PeekPointer (p)));
}

template <typename T1, typename T2>
Ptr<T1>
DynamicCast (Ptr<T2> const&p)
{
  return Ptr<T1> (dynamic_cast<T1 *> (PeekPointer (p)));
}

template <typename T1, typename T2>
Ptr<T1>
StaticCast (Ptr<T2> const&p)
{
  return Ptr<T1> (static_cast<T1 *> (PeekPointer (p)));
}
/** @} */

/**
 * Return a deep copy of a Ptr.
 *
 * \tparam T \deduced The type of the underlying object.
 * \param [in] object The object Ptr to copy.
 * \returns The copy.
 */
/** @{ */
template <typename T>
Ptr<T> Copy (Ptr<T> object)
{
  Ptr<T> p = Ptr<T> (new T (*PeekPointer (object)), false);
  return p;
}

template <typename T>
Ptr<T> Copy (Ptr<const T> object)
{
  Ptr<T> p = Ptr<T> (new T (*PeekPointer (object)), false);
  return p;
}
/** @} */

/****************************************************
 *      Member method implementations.
 ***************************************************/

template <typename T>
void
Ptr<T>::Acquire (void) const
{
  if (m_ptr != 0)
    {
      m_ptr->Ref ();
    }
}

template <typename T>
Ptr<T>::Ptr ()
  : m_ptr (0)
{}

template <typename T>
Ptr<T>::Ptr (T *ptr)
  : m_ptr (ptr)
{
  Acquire ();
}

template <typename T>
Ptr<T>::Ptr (T *ptr, bool ref)
  : m_ptr (ptr)
{
  if (ref)
    {
      Acquire ();
    }
}

template <typename T>
Ptr<T>::Ptr (Ptr const&o)
  : m_ptr (PeekPointer (o))
{
  Acquire ();
}
template <typename T>
template <typename U>
Ptr<T>::Ptr (Ptr<U> const &o)
  : m_ptr (PeekPointer (o))
{
  Acquire ();
}

template <typename T>
Ptr<T>::~Ptr ()
{
  if (m_ptr != 0)
    {
      m_ptr->Unref ();
    }
}

template <typename T>
Ptr<T> &
Ptr<T>::operator = (Ptr const& o)
{
  if (&o == this)
    {
      return *this;
    }
  if (m_ptr != 0)
    {
      m_ptr->Unref ();
    }
  m_ptr = o.m_ptr;
  Acquire ();
  return *this;
}

template <typename T>
T *
Ptr<T>::operator -> ()
{
  NS_ASSERT_MSG (m_ptr, "Attempted to dereference zero pointer");
  return m_ptr;
}

template <typename T>
T *
Ptr<T>::operator -> () const
{
  NS_ASSERT_MSG (m_ptr, "Attempted to dereference zero pointer");
  return m_ptr;
}

template <typename T>
T &
Ptr<T>::operator * () const
{
  NS_ASSERT_MSG (m_ptr, "Attempted to dereference zero pointer");
  return *m_ptr;
}

template <typename T>
T &
Ptr<T>::operator * ()
{
  NS_ASSERT_MSG (m_ptr, "Attempted to dereference zero pointer");
  return *m_ptr;
}

template <typename T>
bool
Ptr<T>::operator! ()
{
  return m_ptr == 0;
}

template <typename T>
Ptr<T>::operator Tester * () const
{
  if (m_ptr == 0)
    {
      return 0;
    }
  static Tester test;
  return &test;
}


} // namespace ns3


/****************************************************
 *      Global Functions (outside namespace ns3)
 ***************************************************/

/**
 * \ingroup ptr
 * Hashing functor taking a `Ptr` and returning a @c std::size_t.
 * For use with `unordered_map` and `unordered_set`.
 *
 * \note When a `Ptr` is used in a container the lifetime of the underlying
 * object is at least as long as the container.  In other words,
 * you need to remove the object from the container when you are done with
 * it, otherwise the object will persist until the container itself is
 * deleted.
 *
 * \tparam T \deduced The type held by the `Ptr`
 */
template<class T>
struct
std::hash<ns3::Ptr<T>>
{
  /**
   * The functor.
   * \param p The `Ptr` value to hash.
   * \return the hash
   */
  std::size_t
  operator () (ns3::Ptr<T> p) const
  {
    return std::hash<const T *> () (ns3::PeekPointer (p));
  }
};


#endif /* PTR_H */
