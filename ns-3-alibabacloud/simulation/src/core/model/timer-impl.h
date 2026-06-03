/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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

#ifndef TIMER_IMPL_H
#define TIMER_IMPL_H

#include "simulator.h"
#include "type-traits.h"
#include "fatal-error.h"
#include "int-to-type.h"

/**
 * \file
 * \ingroup timer
 * \ingroup timerimpl
 * ns3::TimerImpl declaration and implementation.
 */

namespace ns3 {

/**
 * \ingroup timer
 * The timer implementation underlying Timer and Watchdog.
 */
class TimerImpl
{
public:
  /** Destructor. */
  virtual ~TimerImpl ()
  {}

  /**
   * Set the arguments to be used when invoking the expire function.
   */
  /**@{*/
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \param [in] a1 The first argument
   */
  template <typename T1>
  void SetArgs (T1 a1);
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \tparam T2 \deduced Type of the second argument.
   * \param [in] a1 the first argument
   * \param [in] a2 the second argument
   */
  template <typename T1, typename T2>
  void SetArgs (T1 a1, T2 a2);
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \tparam T2 \deduced Type of the second argument.
   * \tparam T3 \deduced Type of the third argument.
   * \param [in] a1 the first argument
   * \param [in] a2 the second argument
   * \param [in] a3 the third argument
   */
  template <typename T1, typename T2, typename T3>
  void SetArgs (T1 a1, T2 a2, T3 a3);
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \tparam T2 \deduced Type of the second argument.
   * \tparam T3 \deduced Type of the third argument.
   * \tparam T4 \deduced Type of the fourth argument.
   * \param [in] a1 the first argument
   * \param [in] a2 the second argument
   * \param [in] a3 the third argument
   * \param [in] a4 the fourth argument
   */
  template <typename T1, typename T2, typename T3,
            typename T4>
  void SetArgs (T1 a1, T2 a2, T3 a3, T4 a4);
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \tparam T2 \deduced Type of the second argument.
   * \tparam T3 \deduced Type of the third argument.
   * \tparam T4 \deduced Type of the fourth argument.
   * \tparam T5 \deduced Type of the fifth argument.
   * \param [in] a1 the first argument
   * \param [in] a2 the second argument
   * \param [in] a3 the third argument
   * \param [in] a4 the fourth argument
   * \param [in] a5 the fifth argument
   */
  template <typename T1, typename T2, typename T3,
            typename T4, typename T5>
  void SetArgs (T1 a1, T2 a2, T3 a3, T4 a4, T5 a5);
  /**
   * \tparam T1 \deduced Type of the first argument.
   * \tparam T2 \deduced Type of the second argument.
   * \tparam T3 \deduced Type of the third argument.
   * \tparam T4 \deduced Type of the fourth argument.
   * \tparam T5 \deduced Type of the fifth argument.
   * \tparam T6 \deduced Type of the sixth argument.
   * \param [in] a1 the first argument
   * \param [in] a2 the second argument
   * \param [in] a3 the third argument
   * \param [in] a4 the fourth argument
   * \param [in] a5 the fifth argument
   * \param [in] a6 the sixth argument
   */
  template <typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6>
  void SetArgs (T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6);
  /**@}*/

  /**
   * Schedule the callback for a future time.
   *
   * \param [in] delay The amount of time until the timer expires.
   * \returns The scheduled EventId.
   */
  virtual EventId Schedule (const Time &delay) = 0;
  /** Invoke the expire function. */
  virtual void Invoke (void) = 0;
};

} // namespace ns3


/********************************************************************
 *  Implementation of TimerImpl implementation functions.
 ********************************************************************/

namespace ns3 {

/**
 * \ingroup timer
 * \defgroup timerimpl TimerImpl Implementation
 * @{
 */

/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1>
struct TimerImplOne : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   */
  virtual void SetArguments (T1 a1) = 0;
};
/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1, typename T2>
struct TimerImplTwo : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   * \param [in] a2 The second argument.
   */
  virtual void SetArguments (T1 a1,T2 a2) = 0;
};
/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1, typename T2, typename T3>
struct TimerImplThree : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   * \param [in] a2 The second argument.
   * \param [in] a3 The third argument.
   */
  virtual void SetArguments (T1 a1,T2 a2,T3 a3) = 0;
};
/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1, typename T2, typename T3, typename T4>
struct TimerImplFour : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   * \param [in] a2 The second argument.
   * \param [in] a3 The third argument.
   * \param [in] a4 The fourth argument.
   */
  virtual void SetArguments (T1 a1,T2 a2,T3 a3, T4 a4) = 0;
};
/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct TimerImplFive : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   * \param [in] a2 The second argument.
   * \param [in] a3 The third argument.
   * \param [in] a4 The fourth argument.
   * \param [in] a5 The fifth argument.
   */
  virtual void SetArguments (T1 a1,T2 a2,T3 a3, T4 a4, T5 a5) = 0;
};
/** TimerImpl specialization class for varying numbers of arguments. */
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
struct TimerImplSix : public TimerImpl
{
  /**
   * Bind the arguments to be used when the callback function is invoked.
   *
   * \param [in] a1 The first argument.
   * \param [in] a2 The second argument.
   * \param [in] a3 The third argument.
   * \param [in] a4 The fourth argument.
   * \param [in] a5 The fifth argument.
   * \param [in] a6 The sixth argument.
   */
  virtual void SetArguments (T1 a1,T2 a2,T3 a3, T4 a4, T5 a5, T6 a6) = 0;
};


/** Type and reference traits for TimerImpl arguments. */
template <typename T>
struct TimerTraits
{
  /** Storage type for an argument. */
  typedef typename TypeTraits<typename TypeTraits<T>::ReferencedType>::NonConstType StoredType;
  /** Parameter type for an argument. */
  typedef const StoredType &ParameterType;
};

/**
 * Make a TimerImpl from a function pointer taking varying numbers of arguments.
 *
 * \tparam FN \deduced Function signature type of the callback function.
 * \param [in] fn The function pointer to invoke when the timer expires.
 * \returns The TimerImpl.
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (FN fn)
{
  NS_ASSERT (TypeTraits<FN>::IsFunctionPointer);
  return MakeTimerImpl (IntToType<TypeTraits<FN>::FunctionPointerTraits::nArgs> (), fn);
}

/**
 * Make a TimerImpl from a function pointer taking zero arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<0>, FN fn)
{
  struct FnTimerImplZero : public TimerImpl
  {
    FnTimerImplZero (FN fn)
      : m_fn (fn)
    {}
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn);
    }
    virtual void Invoke (void)
    {
      m_fn ();
    }
    FN m_fn;
  } *function = new FnTimerImplZero (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking one argument.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<1>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;

  struct FnTimerImplOne : public TimerImplOne<T1Parameter>
  {
    FnTimerImplOne (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1)
    {
      m_a1 = a1;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1);
    }
    FN m_fn;
    T1Stored m_a1;
  } *function = new FnTimerImplOne (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking two arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<2>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;

  struct FnTimerImplTwo : public TimerImplTwo<T1Parameter,T2Parameter>
  {
    FnTimerImplTwo (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2)
    {
      m_a1 = a1;
      m_a2 = a2;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1, m_a2);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1, m_a2);
    }
    FN m_fn;
    T1Stored m_a1;
    T2Stored m_a2;
  } *function = new FnTimerImplTwo (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking three arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<3>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;

  struct FnTimerImplThree : public TimerImplThree<T1Parameter,T2Parameter,T3Parameter>
  {
    FnTimerImplThree (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1, m_a2, m_a3);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1, m_a2, m_a3);
    }
    FN m_fn;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
  } *function = new FnTimerImplThree (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking four arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<4>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;

  struct FnTimerImplFour : public TimerImplFour<T1Parameter,T2Parameter,T3Parameter,T4Parameter>
  {
    FnTimerImplFour (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1, m_a2, m_a3, m_a4);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1, m_a2, m_a3, m_a4);
    }
    FN m_fn;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
  } *function = new FnTimerImplFour (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking five arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<5>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg5Type T5;
  typedef typename TimerTraits<T5>::ParameterType T5Parameter;
  typedef typename TimerTraits<T5>::StoredType T5Stored;

  struct FnTimerImplFive : public TimerImplFive<T1Parameter,T2Parameter,T3Parameter,T4Parameter,T5Parameter>
  {
    FnTimerImplFive (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4, T5Parameter a5)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
      m_a5 = a5;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    FN m_fn;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
    T5Stored m_a5;
  } *function = new FnTimerImplFive (fn);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking six arguments.
 * \copydetails MakeTimerImpl(FN)
 */
template <typename FN>
TimerImpl *
MakeTimerImpl (IntToType<6>, FN fn)
{
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg5Type T5;
  typedef typename TimerTraits<T5>::ParameterType T5Parameter;
  typedef typename TimerTraits<T5>::StoredType T5Stored;
  typedef typename TypeTraits<FN>::FunctionPointerTraits::Arg6Type T6;
  typedef typename TimerTraits<T6>::ParameterType T6Parameter;
  typedef typename TimerTraits<T6>::StoredType T6Stored;

  struct FnTimerImplSix : public TimerImplSix<T1Parameter,T2Parameter,T3Parameter,T4Parameter,T5Parameter,T6Parameter>
  {
    FnTimerImplSix (FN fn)
      : m_fn (fn)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4, T5Parameter a5, T6Parameter a6)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
      m_a5 = a5;
      m_a6 = a6;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_fn, m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    virtual void Invoke (void)
    {
      m_fn (m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    FN m_fn;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
    T5Stored m_a5;
    T6Stored m_a6;
  } *function = new FnTimerImplSix (fn);
  return function;
}


/**
 * Helper for the MakeTimerImpl functions which take a class method.
 *
 * This helper converts a pointer to a reference.
 *
 * This is the generic template declaration (with empty body).
 *
 * \tparam T \explicit The object type.
 */
template <typename T>
struct TimerImplMemberTraits;


/**
 * Helper for the MakeTimerImpl functions which take a class method.
 *
 * This helper converts a pointer to a reference.
 *
 * This is the specialization for pointer to \c T.
 *
 * \tparam T \explicit The object type.
 */
template <typename T>
struct TimerImplMemberTraits<T *>
{
  /**
   * Convert a pointer type to a reference.
   *
   * \param [in] p The pointer.
   * \returns A reference to the object pointed to by \c p.
   */
  static T & GetReference (T *p)
  {
    return *p;
  }
};

/**
 * Make a TimerImpl from a class method pointer taking
 * a varying number of arguments.
 *
 * \tparam MEM_PTR \deduced Class method function signature type.
 * \tparam OBJ_PTR \deduced Class type.
 * \param [in] memPtr Class method to invoke when the timer expires.
 * \param [in] objPtr Object instance pointer.
 * \returns The TimerImpl.
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (MEM_PTR memPtr, OBJ_PTR objPtr)
{
  NS_ASSERT (TypeTraits<MEM_PTR>::IsPointerToMember);
  return MakeTimerImpl (IntToType<TypeTraits<MEM_PTR>::PointerToMemberTraits::nArgs> (), memPtr, objPtr);
}

/**
 * Make a TimerImpl from a function pointer taking zero arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<0>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  struct MemFnTimerImplZero : public TimerImpl
  {
    MemFnTimerImplZero (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)();
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
  } *function = new MemFnTimerImplZero (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking one argument.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<1>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;

  struct MemFnTimerImplOne : public TimerImplOne<T1Parameter>
  {
    MemFnTimerImplOne (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1)
    {
      m_a1 = a1;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
  } *function = new MemFnTimerImplOne (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking two arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<2>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;

  struct MemFnTimerImplTwo : public TimerImplTwo<T1Parameter,T2Parameter>
  {
    MemFnTimerImplTwo (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2)
    {
      m_a1 = a1;
      m_a2 = a2;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1, m_a2);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1, m_a2);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
    T2Stored m_a2;
  } *function = new MemFnTimerImplTwo (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking three arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<3>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;

  struct MemFnTimerImplThree : public TimerImplThree<T1Parameter,T2Parameter,T3Parameter>
  {
    MemFnTimerImplThree (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1, m_a2, m_a3);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1, m_a2, m_a3);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
  } *function = new MemFnTimerImplThree (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking four arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<4>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;

  struct MemFnTimerImplFour : public TimerImplFour<T1Parameter,T2Parameter,T3Parameter,T4Parameter>
  {
    MemFnTimerImplFour (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1, m_a2, m_a3, m_a4);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1, m_a2, m_a3, m_a4);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
  } *function = new MemFnTimerImplFour (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking five arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<5>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg5Type T5;
  typedef typename TimerTraits<T5>::ParameterType T5Parameter;
  typedef typename TimerTraits<T5>::StoredType T5Stored;

  struct MemFnTimerImplFive : public TimerImplFive<T1Parameter,T2Parameter,T3Parameter,T4Parameter,T5Parameter>
  {
    MemFnTimerImplFive (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4,T5Parameter a5)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
      m_a5 = a5;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1, m_a2, m_a3, m_a4, m_a5);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
    T5Stored m_a5;
  } *function = new MemFnTimerImplFive (memPtr, objPtr);
  return function;
}

/**
 * Make a TimerImpl from a function pointer taking six arguments.
 * \copydetails MakeTimerImpl(MEM_PTR,OBJ_PTR)
 */
template <typename MEM_PTR, typename OBJ_PTR>
TimerImpl *
MakeTimerImpl (IntToType<6>, MEM_PTR memPtr, OBJ_PTR objPtr)
{
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg1Type T1;
  typedef typename TimerTraits<T1>::ParameterType T1Parameter;
  typedef typename TimerTraits<T1>::StoredType T1Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg2Type T2;
  typedef typename TimerTraits<T2>::ParameterType T2Parameter;
  typedef typename TimerTraits<T2>::StoredType T2Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg3Type T3;
  typedef typename TimerTraits<T3>::ParameterType T3Parameter;
  typedef typename TimerTraits<T3>::StoredType T3Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg4Type T4;
  typedef typename TimerTraits<T4>::ParameterType T4Parameter;
  typedef typename TimerTraits<T4>::StoredType T4Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg5Type T5;
  typedef typename TimerTraits<T5>::ParameterType T5Parameter;
  typedef typename TimerTraits<T5>::StoredType T5Stored;
  typedef typename TypeTraits<MEM_PTR>::PointerToMemberTraits::Arg6Type T6;
  typedef typename TimerTraits<T6>::ParameterType T6Parameter;
  typedef typename TimerTraits<T6>::StoredType T6Stored;

  struct MemFnTimerImplSix : public TimerImplSix<T1Parameter,T2Parameter,T3Parameter,T4Parameter,T5Parameter,T6Parameter>
  {
    MemFnTimerImplSix (MEM_PTR memPtr, OBJ_PTR objPtr)
      : m_memPtr (memPtr),
        m_objPtr (objPtr)
    {}
    virtual void SetArguments (T1Parameter a1, T2Parameter a2, T3Parameter a3, T4Parameter a4,T5Parameter a5,T6Parameter a6)
    {
      m_a1 = a1;
      m_a2 = a2;
      m_a3 = a3;
      m_a4 = a4;
      m_a5 = a5;
      m_a6 = a6;
    }
    virtual EventId Schedule (const Time &delay)
    {
      return Simulator::Schedule (delay, m_memPtr, m_objPtr, m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    virtual void Invoke (void)
    {
      (TimerImplMemberTraits<OBJ_PTR>::GetReference (m_objPtr).*m_memPtr)(m_a1, m_a2, m_a3, m_a4, m_a5, m_a6);
    }
    MEM_PTR m_memPtr;
    OBJ_PTR m_objPtr;
    T1Stored m_a1;
    T2Stored m_a2;
    T3Stored m_a3;
    T4Stored m_a4;
    T5Stored m_a5;
    T6Stored m_a6;
  } *function = new MemFnTimerImplSix (memPtr, objPtr);
  return function;
}

/**@}*/  // \ingroup timer


/********************************************************************
 *  Implementation of TimerImpl itself.
 ********************************************************************/

template <typename T1>
void
TimerImpl::SetArgs (T1 a1)
{
  typedef struct TimerImplOne<
      typename TimerTraits<T1>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1);
}

template <typename T1, typename T2>
void
TimerImpl::SetArgs (T1 a1, T2 a2)
{
  typedef struct TimerImplTwo<
      typename TimerTraits<T1>::ParameterType,
      typename TimerTraits<T2>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1, a2);
}

template <typename T1, typename T2, typename T3>
void
TimerImpl::SetArgs (T1 a1, T2 a2, T3 a3)
{
  typedef struct TimerImplThree<
      typename TimerTraits<T1>::ParameterType,
      typename TimerTraits<T2>::ParameterType,
      typename TimerTraits<T3>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1, a2, a3);
}

template <typename T1, typename T2, typename T3, typename T4>
void
TimerImpl::SetArgs (T1 a1, T2 a2, T3 a3, T4 a4)
{
  typedef struct TimerImplFour<
      typename TimerTraits<T1>::ParameterType,
      typename TimerTraits<T2>::ParameterType,
      typename TimerTraits<T3>::ParameterType,
      typename TimerTraits<T4>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1, a2, a3, a4);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void
TimerImpl::SetArgs (T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
{
  typedef struct TimerImplFive<
      typename TimerTraits<T1>::ParameterType,
      typename TimerTraits<T2>::ParameterType,
      typename TimerTraits<T3>::ParameterType,
      typename TimerTraits<T4>::ParameterType,
      typename TimerTraits<T5>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1, a2, a3, a4, a5);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void
TimerImpl::SetArgs (T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
{
  typedef struct TimerImplSix<
      typename TimerTraits<T1>::ParameterType,
      typename TimerTraits<T2>::ParameterType,
      typename TimerTraits<T3>::ParameterType,
      typename TimerTraits<T4>::ParameterType,
      typename TimerTraits<T5>::ParameterType,
      typename TimerTraits<T6>::ParameterType
      > TimerImplBase;
  TimerImplBase *impl = dynamic_cast<TimerImplBase *> (this);
  if (impl == 0)
    {
      NS_FATAL_ERROR ("You tried to set Timer arguments incompatible with its function.");
      return;
    }
  impl->SetArguments (a1, a2, a3, a4, a5, a6);
}

} // namespace ns3

#endif /* TIMER_IMPL_H */
