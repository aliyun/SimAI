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
 */

#ifndef MAKE_EVENT_H
#define MAKE_EVENT_H

/**
 * \file
 * \ingroup events
 * ns3::MakeEvent function declarations and template implementation.
 */

namespace ns3 {

class EventImpl;

/**
 * \ingroup events
 * \defgroup makeeventmemptr MakeEvent from Member Function Pointer.
 *
 * Create EventImpl instances from class member functions which take
 * varying numbers of arguments.
 */
/**
 * \ingroup makeeventmemptr
 * @{
 */
/**
 * Make an EventImpl from class method members which take
 * varying numbers of arguments.
 *
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \tparam T1 \deduced Type of the argument to the underlying function.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \param [in] a1 Argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \tparam T1 \deduced Type of the first argument to the underlying function.
 * \tparam T2 \deduced Type of the second argument to the underlying function.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \param [in] a1 First argument value to be bound to the underlying function.
 * \param [in] a2 Second argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1, typename T2>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \tparam T1 \deduced Type of the first argument to the underlying function.
 * \tparam T2 \deduced Type of the second argument to the underlying function.
 * \tparam T3 \deduced Type of the third argument to the underlying function.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \param [in] a1 First argument value to be bound to the underlying function.
 * \param [in] a2 Second argument value to be bound to the underlying function.
 * \param [in] a3 Third argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2, T3 a3);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \tparam T1 \deduced Type of the first argument to the underlying function.
 * \tparam T2 \deduced Type of the second argument to the underlying function.
 * \tparam T3 \deduced Type of the third argument to the underlying function.
 * \tparam T4 \deduced Type of the fourth argument to the underlying function.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \param [in] a1 First argument value to be bound to the underlying function.
 * \param [in] a2 Second argument value to be bound to the underlying function.
 * \param [in] a3 Third argument value to be bound to the underlying function.
 * \param [in] a4 Fourth argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2, T3 a3, T4 a4);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM \deduced The class method function signature.
 * \tparam OBJ \deduced The class type holding the method.
 * \tparam T1 \deduced Type of the first argument to the underlying function.
 * \tparam T2 \deduced Type of the second argument to the underlying function.
 * \tparam T3 \deduced Type of the third argument to the underlying function.
 * \tparam T4 \deduced Type of the fourth argument to the underlying function.
 * \tparam T5 \deduced Type of the fifth argument to the underlying function.
 * \param [in] mem_ptr Class method member function pointer
 * \param [in] obj Class instance.
 * \param [in] a1 First argument value to be bound to the underlying function.
 * \param [in] a2 Second argument value to be bound to the underlying function.
 * \param [in] a3 Third argument value to be bound to the underlying function.
 * \param [in] a4 Fourth argument value to be bound to the underlying function.
 * \param [in] a5 Fifh argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4, typename T5>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj,
                       T1 a1, T2 a2, T3 a3, T4 a4, T5 a5);

/**
 * \copybrief MakeEvent(MEM,OBJ)
 * \tparam MEM The class method function signature.
 * \tparam OBJ The class type holding the method.
 * \tparam T1 Type of the first argument to the underlying function.
 * \tparam T2 Type of the second argument to the underlying function.
 * \tparam T3 Type of the third argument to the underlying function.
 * \tparam T4 Type of the fourth argument to the underlying function.
 * \tparam T5 Type of the fifth argument to the underlying function.
 * \tparam T6 Type of the sixth argument to the underlying function.
 * \param mem_ptr Class method member function pointer
 * \param obj Class instance.
 * \param a1 First argument value to be bound to the underlying function.
 * \param a2 Second argument value to be bound to the underlying function.
 * \param a3 Third argument value to be bound to the underlying function.
 * \param a4 Fourth argument value to be bound to the underlying function.
 * \param a5 Fifth argument value to be bound to the underlying function.
 * \param a6 Sixth argument value to be bound to the underlying function.
 * \returns The constructed EventImpl.
 */
template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj,
                       T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6);

/**@}*/

/**
 * \ingroup events
 * \defgroup makeeventfnptr MakeEvent from Function Pointers and Lambdas.
 *
 * Create EventImpl instances from function pointers or lambdas which take
 * varying numbers of arguments.
 *
 * @{
 */
/**
 * Make an EventImpl from a function pointer taking varying numbers
 * of arguments.
 *
 * \param [in] f The function pointer.
 * \returns The constructed EventImpl.
 */
EventImpl * MakeEvent (void (*f)(void));

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 \deduced Formal type of the argument to the function.
 * \tparam T1 \deduced Actual type of the argument to the function.
 * \param [in] f The function pointer.
 * \param [in] a1 Argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1,
          typename T1>
EventImpl * MakeEvent (void (*f)(U1), T1 a1);

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 \deduced Formal type of the first argument to the function.
 * \tparam U2 \deduced Formal type of the second argument to the function.
 * \tparam T1 \deduced Actual type of the first argument to the function.
 * \tparam T2 \deduced Actual type of the second argument to the function.
 * \param [in] f The function pointer.
 * \param [in] a1 First argument to be bound to the function.
 * \param [in] a2 Second argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1, typename U2,
          typename T1, typename T2>
EventImpl * MakeEvent (void (*f)(U1,U2), T1 a1, T2 a2);

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 \deduced Formal type of the first argument to the function.
 * \tparam U2 \deduced Formal type of the second argument to the function.
 * \tparam U3 \deduced Formal type of the third argument to the function.
 * \tparam T1 \deduced Actual type of the first argument to the function.
 * \tparam T2 \deduced Actual type of the second argument to the function.
 * \tparam T3 \deduced Actual type of the third argument to the function.
 * \param [in] f The function pointer.
 * \param [in] a1 First argument to be bound to the function.
 * \param [in] a2 Second argument to be bound to the function.
 * \param [in] a3 Third argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1, typename U2, typename U3,
          typename T1, typename T2, typename T3>
EventImpl * MakeEvent (void (*f)(U1,U2,U3), T1 a1, T2 a2, T3 a3);

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 \deduced Formal type of the first argument to the function.
 * \tparam U2 \deduced Formal type of the second argument to the function.
 * \tparam U3 \deduced Formal type of the third argument to the function.
 * \tparam U4 \deduced Formal type of the fourth argument to the function.
 * \tparam T1 \deduced Actual type of the first argument to the function.
 * \tparam T2 \deduced Actual type of the second argument to the function.
 * \tparam T3 \deduced Actual type of the third argument to the function.
 * \tparam T4 \deduced Actual type of the fourth argument to the function.
 * \param [in] f The function pointer.
 * \param [in] a1 First argument to be bound to the function.
 * \param [in] a2 Second argument to be bound to the function.
 * \param [in] a3 Third argument to be bound to the function.
 * \param [in] a4 Fourth argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1, typename U2, typename U3, typename U4,
          typename T1, typename T2, typename T3, typename T4>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4), T1 a1, T2 a2, T3 a3, T4 a4);

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 \deduced Formal type of the first argument to the function.
 * \tparam U2 \deduced Formal type of the second argument to the function.
 * \tparam U3 \deduced Formal type of the third argument to the function.
 * \tparam U4 \deduced Formal type of the fourth argument to the function.
 * \tparam U5 \deduced Formal type of the fifth argument to the function.
 * \tparam T1 \deduced Actual type of the first argument to the function.
 * \tparam T2 \deduced Actual type of the second argument to the function.
 * \tparam T3 \deduced Actual type of the third argument to the function.
 * \tparam T4 \deduced Actual type of the fourth argument to the function.
 * \tparam T5 \deduced Actual type of the fifth argument to the function.
 * \param [in] f The function pointer.
 * \param [in] a1 First argument to be bound to the function.
 * \param [in] a2 Second argument to be bound to the function.
 * \param [in] a3 Third argument to be bound to the function.
 * \param [in] a4 Fourth argument to be bound to the function.
 * \param [in] a5 Fifth argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1, typename U2, typename U3, typename U4, typename U5,
          typename T1, typename T2, typename T3, typename T4, typename T5>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4,U5), T1 a1, T2 a2, T3 a3, T4 a4, T5 a5);

/**
 * \copybrief MakeEvent(void(*f)(void))
 * \tparam U1 Formal type of the first argument to the function.
 * \tparam U2 Formal type of the second argument to the function.
 * \tparam U3 Formal type of the third argument to the function.
 * \tparam U4 Formal type of the fourth argument to the function.
 * \tparam U5 Formal type of the fifth argument to the function.
 * \tparam U6 Formal type of the sixth argument to the function.
 * \tparam T1 Actual type of the first argument to the function.
 * \tparam T2 Actual type of the second argument to the function.
 * \tparam T3 Actual type of the third argument to the function.
 * \tparam T4 Actual type of the fourth argument to the function.
 * \tparam T5 Actual type of the fifth argument to the function.
 * \tparam T6 Actual type of the sixth argument to the function.
 * \param f The function pointer.
 * \param a1 First argument to be bound to the function.
 * \param a2 Second argument to be bound to the function.
 * \param a3 Third argument to be bound to the function.
 * \param a4 Fourth argument to be bound to the function.
 * \param a5 Fifth argument to be bound to the function.
 * \param a6 Sixth argument to be bound to the function.
 * \returns The constructed EventImpl.
 */
template <typename U1, typename U2, typename U3, typename U4, typename U5, typename U6,
          typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4,U5,U6), T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6);

/**
 * Make an EventImpl from a lambda.
 *
 * \param [in] function The lambda
 * \returns The constructed EventImpl.
 */
template <typename T>
EventImpl * MakeEvent (T function);


/**@}*/

} // namespace ns3

/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

#include "event-impl.h"
#include "type-traits.h"

namespace ns3 {

/**
 * \ingroup makeeventmemptr
 * Helper for the MakeEvent functions which take a class method.
 *
 * This helper converts a pointer to a reference.
 *
 * This is the generic template declaration (with empty body).
 *
 * \tparam T \explicit The class type.
 */
template <typename T>
struct EventMemberImplObjTraits;

/**
 * \ingroup makeeventmemptr
 * Helper for the MakeEvent functions which take a class method.
 *
 * This helper converts a pointer to a reference.
 *
 * This is the specialization for pointer types.
 *
 * \tparam T \explicit The class type.
 */
template <typename T>
struct EventMemberImplObjTraits<T *>
{
  /**
   * \param [in] p Object pointer.
   * \return A reference to the object pointed to by p.
   */
  static T & GetReference (T *p)
  {
    return *p;
  }
};

template <typename MEM, typename OBJ>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj)
{
  // zero argument version
  class EventMemberImpl0 : public EventImpl
  {
  public:
    EventMemberImpl0 (OBJ obj, MEM function)
      : m_obj (obj),
        m_function (function)
    {}
    virtual ~EventMemberImpl0 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)();
    }
    OBJ m_obj;
    MEM m_function;
  } *ev = new EventMemberImpl0 (obj, mem_ptr);
  return ev;
}


template <typename MEM, typename OBJ,
          typename T1>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1)
{
  // one argument version
  class EventMemberImpl1 : public EventImpl
  {
  public:
    EventMemberImpl1 (OBJ obj, MEM function, T1 a1)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1)
    {}

  protected:
    virtual ~EventMemberImpl1 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
  } *ev = new EventMemberImpl1 (obj, mem_ptr, a1);
  return ev;
}

template <typename MEM, typename OBJ,
          typename T1, typename T2>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2)
{
  // two argument version
  class EventMemberImpl2 : public EventImpl
  {
  public:
    EventMemberImpl2 (OBJ obj, MEM function, T1 a1, T2 a2)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1),
        m_a2 (a2)
    {}

  protected:
    virtual ~EventMemberImpl2 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1, m_a2);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
  } *ev = new EventMemberImpl2 (obj, mem_ptr, a1, a2);
  return ev;
}

template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2, T3 a3)
{
  // three argument version
  class EventMemberImpl3 : public EventImpl
  {
  public:
    EventMemberImpl3 (OBJ obj, MEM function, T1 a1, T2 a2, T3 a3)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3)
    {}

  protected:
    virtual ~EventMemberImpl3 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1, m_a2, m_a3);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
  } *ev = new EventMemberImpl3 (obj, mem_ptr, a1, a2, a3);
  return ev;
}

template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj, T1 a1, T2 a2, T3 a3, T4 a4)
{
  // four argument version
  class EventMemberImpl4 : public EventImpl
  {
  public:
    EventMemberImpl4 (OBJ obj, MEM function, T1 a1, T2 a2, T3 a3, T4 a4)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4)
    {}

  protected:
    virtual ~EventMemberImpl4 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1, m_a2, m_a3, m_a4);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
  } *ev = new EventMemberImpl4 (obj, mem_ptr, a1, a2, a3, a4);
  return ev;
}

template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4, typename T5>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj,
                       T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
{
  // five argument version
  class EventMemberImpl5 : public EventImpl
  {
  public:
    EventMemberImpl5 (OBJ obj, MEM function, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4),
        m_a5 (a5)
    {}

  protected:
    virtual ~EventMemberImpl5 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
    typename TypeTraits<T5>::ReferencedType m_a5;
  } *ev = new EventMemberImpl5 (obj, mem_ptr, a1, a2, a3, a4, a5);
  return ev;
}

template <typename MEM, typename OBJ,
          typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
EventImpl * MakeEvent (MEM mem_ptr, OBJ obj,
                       T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
{
  // six argument version
  class EventMemberImpl6 : public EventImpl
  {
  public:
    EventMemberImpl6 (OBJ obj, MEM function, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
      : m_obj (obj),
        m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4),
        m_a5 (a5),
        m_a6 (a6)
    {}

  protected:
    virtual ~EventMemberImpl6 ()
    {}

  private:
    virtual void Notify (void)
    {
      (EventMemberImplObjTraits<OBJ>::GetReference (m_obj).*m_function)(m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    OBJ m_obj;
    MEM m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
    typename TypeTraits<T5>::ReferencedType m_a5;
    typename TypeTraits<T6>::ReferencedType m_a6;
  } *ev = new EventMemberImpl6 (obj, mem_ptr, a1, a2, a3, a4, a5, a6);
  return ev;
}

template <typename U1, typename T1>
EventImpl * MakeEvent (void (*f)(U1), T1 a1)
{
  // one arg version
  class EventFunctionImpl1 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1);

    EventFunctionImpl1 (F function, T1 a1)
      : m_function (function),
        m_a1 (a1)
    {}

  protected:
    virtual ~EventFunctionImpl1 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
  } *ev = new EventFunctionImpl1 (f, a1);
  return ev;
}

template <typename U1, typename U2, typename T1, typename T2>
EventImpl * MakeEvent (void (*f)(U1,U2), T1 a1, T2 a2)
{
  // two arg version
  class EventFunctionImpl2 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1, U2);

    EventFunctionImpl2 (F function, T1 a1, T2 a2)
      : m_function (function),
        m_a1 (a1),
        m_a2 (a2)
    {}

  protected:
    virtual ~EventFunctionImpl2 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1, m_a2);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
  } *ev = new EventFunctionImpl2 (f, a1, a2);
  return ev;
}

template <typename U1, typename U2, typename U3,
          typename T1, typename T2, typename T3>
EventImpl * MakeEvent (void (*f)(U1,U2,U3), T1 a1, T2 a2, T3 a3)
{
  // three arg version
  class EventFunctionImpl3 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1, U2, U3);

    EventFunctionImpl3 (F function, T1 a1, T2 a2, T3 a3)
      : m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3)
    {}

  protected:
    virtual ~EventFunctionImpl3 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1, m_a2, m_a3);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
  } *ev = new EventFunctionImpl3 (f, a1, a2, a3);
  return ev;
}

template <typename U1, typename U2, typename U3, typename U4,
          typename T1, typename T2, typename T3, typename T4>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4), T1 a1, T2 a2, T3 a3, T4 a4)
{
  // four arg version
  class EventFunctionImpl4 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1, U2, U3, U4);

    EventFunctionImpl4 (F function, T1 a1, T2 a2, T3 a3, T4 a4)
      : m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4)
    {}

  protected:
    virtual ~EventFunctionImpl4 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1, m_a2, m_a3, m_a4);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
  } *ev = new EventFunctionImpl4 (f, a1, a2, a3, a4);
  return ev;
}

template <typename U1, typename U2, typename U3, typename U4, typename U5,
          typename T1, typename T2, typename T3, typename T4, typename T5>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4,U5), T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
{
  // five arg version
  class EventFunctionImpl5 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1,U2,U3,U4,U5);

    EventFunctionImpl5 (F function, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
      : m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4),
        m_a5 (a5)
    {}

  protected:
    virtual ~EventFunctionImpl5 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
    typename TypeTraits<T5>::ReferencedType m_a5;
  } *ev = new EventFunctionImpl5 (f, a1, a2, a3, a4, a5);
  return ev;
}

template <typename U1, typename U2, typename U3, typename U4, typename U5, typename U6,
          typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
EventImpl * MakeEvent (void (*f)(U1,U2,U3,U4,U5,U6), T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
{
  // six arg version
  class EventFunctionImpl6 : public EventImpl
  {
  public:
    [[maybe_unused]]
    typedef void (*F)(U1,U2,U3,U4,U5,U6);

    EventFunctionImpl6 (F function, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
      : m_function (function),
        m_a1 (a1),
        m_a2 (a2),
        m_a3 (a3),
        m_a4 (a4),
        m_a5 (a5),
        m_a6 (a6)
    {}

  protected:
    virtual ~EventFunctionImpl6 ()
    {}

  private:
    virtual void Notify (void)
    {
      (*m_function)(m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    F m_function;
    typename TypeTraits<T1>::ReferencedType m_a1;
    typename TypeTraits<T2>::ReferencedType m_a2;
    typename TypeTraits<T3>::ReferencedType m_a3;
    typename TypeTraits<T4>::ReferencedType m_a4;
    typename TypeTraits<T5>::ReferencedType m_a5;
    typename TypeTraits<T6>::ReferencedType m_a6;
  } *ev = new EventFunctionImpl6 (f, a1, a2, a3, a4, a5, a6);
  return ev;
}

template <typename T>
EventImpl * MakeEvent (T function)
{
  class EventImplFunctional : public EventImpl
  {
public:
    EventImplFunctional (T function)
      : m_function (function)
    {
    }
    virtual ~EventImplFunctional ()
    {
    }
private:
    virtual void Notify (void)
    {
      m_function();
    }
    T m_function;
  } *ev = new EventImplFunctional (function);
  return ev;
}


} // namespace ns3

#endif /* MAKE_EVENT_H */
