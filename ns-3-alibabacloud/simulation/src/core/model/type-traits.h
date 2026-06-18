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
 */

#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include "ptr.h"

/**
 *  \file
 *  \ingroup object
 *  ns3::TypeTraits introspection declaration and template implementation.
 */

namespace ns3 {

/**
 * \ingroup object
 *  Inspect a type to deduce its features.
 *  \tparam T \deduced The type to inspect.
 */
template <typename T>
struct TypeTraits
{
private:
  /** Null value type traits. */
  struct NullType {};
  /**
   *  Not a const type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct UnConst
  {
    typedef U Result;                             /**< Non-const type.       */
  };
  /**
   *  Const type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct UnConst<const U>
  {
    typedef U Result;                             /**< Non-const type.       */
  };
  /**
   *  Not a reference type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct ReferenceTraits
  {
    /** Value. */
    enum
    {
      IsReference = 0                             /**< Not a reference type. */
    };
    typedef U ReferencedType;                     /**< Base type.            */
  };
  /**
   *  Reference type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct ReferenceTraits<U&>
  {
    /** Value. */
    enum
    {
      IsReference = 1                             /**< Reference type.       */
    };
    typedef U ReferencedType;                     /**< Base type.            */
  };
  /**
   *  Not a pointer type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct PointerTraits
  {
    /** Value. */
    enum
    {
      IsPointer = 0,                              /**< Not a pointer type.   */
      IsPtr = 0                                   /**< Not a Ptr type.       */
    };
    typedef U PointeeType;                        /**< Base type.            */
  };
  /**
   *  Pointer type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct PointerTraits<U *>
  {
    /** Value. */
    enum
    {
      IsPointer = 1,                              /**< Pointer type.         */
      IsPtr = 0                                   /**< Not a Ptr type.       */
    };
    typedef U PointeeType;                        /**< Pointee type.         */
  };
  /**
   *  Ptr type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct PointerTraits<ns3::Ptr<U> >
  {
    /** Value. */
    enum
    {
      IsPointer = 0,                              /**< Not a pointer type.   */
      IsPtr = 1                                   /**< Ptr type.             */
    };
    typedef U PointeeType;                        /**< Pointee type.         */
  };
  /**
   * Base type, after removing \c &, \c * and \c const.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct Base
  {
    typedef U Type;                               /**< Base type.            */
  };
  /**
   * Base type, after removing \c &.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct Base <U &>
  {
    typedef typename Base<U>::Type Type;          /**< Base type.            */
  };
  /**
   * Base type, after removing \c *.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct Base <U *>
  {
    typedef typename Base<U>::Type  Type;         /**< Base type.            */
  };
  /**
   * Base type, after removing \c const.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct Base <const U >
  {
    typedef typename Base<U>::Type  Type;         /**< Base type.            */
  };
  /**
   * Base type of a Ptr.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct PtrBase
  {
    typedef U Type;                               /**< Base type.            */
  };
  /**
   * Base type of a Ptr.
   * \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct PtrBase <ns3::Ptr<U> >
  {
    typedef U Type;                               /**< Base type.            */
  };
  /**
   *  Not a function pointer type.
   *  \tparam U \deduced The type being inspected.
   */
  template <typename U>
  struct FunctionPtrTraits
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 0                       /**< Not a function pointer. */
    };
    typedef NullType ReturnType;                  /**< Return type.            */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   */
  template <typename U>
  struct FunctionPtrTraits <U (*)(void)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 0                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   */
  template <typename U,
            typename V1>
  struct FunctionPtrTraits <U (*)(V1)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 1                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   *  \tparam V2 \deduced Argument type.
   */
  template <typename U,
            typename V1, typename V2>
  struct FunctionPtrTraits <U (*)(V1,V2)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 2                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
    typedef V2 Arg2Type;                          /**< Second argument type. */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   *  \tparam V2 \deduced Argument type.
   *  \tparam V3 \deduced Argument type.
   */
  template <typename U,
            typename V1, typename V2, typename V3>
  struct FunctionPtrTraits <U (*)(V1,V2,V3)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 3                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
    typedef V2 Arg2Type;                          /**< Second argument type. */
    typedef V3 Arg3Type;                          /**< Third argument type.  */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   *  \tparam V2 \deduced Argument type.
   *  \tparam V3 \deduced Argument type.
   *  \tparam V4 \deduced Argument type.
   */
  template <typename U,
            typename V1, typename V2, typename V3,
            typename V4>
  struct FunctionPtrTraits <U (*)(V1,V2,V3,V4)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 4                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
    typedef V2 Arg2Type;                          /**< Second argument type. */
    typedef V3 Arg3Type;                          /**< Third argument type.  */
    typedef V4 Arg4Type;                          /**< Fourth argument type. */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   *  \tparam V2 \deduced Argument type.
   *  \tparam V3 \deduced Argument type.
   *  \tparam V4 \deduced Argument type.
   *  \tparam V5 \deduced Argument type.
   */
  template <typename U,
            typename V1, typename V2, typename V3,
            typename V4, typename V5>
  struct FunctionPtrTraits <U (*)(V1,V2,V3,V4,V5)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 5                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
    typedef V2 Arg2Type;                          /**< Second argument type. */
    typedef V3 Arg3Type;                          /**< Third argument type.  */
    typedef V4 Arg4Type;                          /**< Fourth argument type. */
    typedef V5 Arg5Type;                          /**< Fifth argument type.  */
  };
  /**
   *  Function pointer type.
   *  \tparam U \deduced Return type.
   *  \tparam V1 \deduced Argument type.
   *  \tparam V2 \deduced Argument type.
   *  \tparam V3 \deduced Argument type.
   *  \tparam V4 \deduced Argument type.
   *  \tparam V5 \deduced Argument type.
   *  \tparam V6 \deduced Argument type.
   */
  template <typename U,
            typename V1, typename V2, typename V3,
            typename V4, typename V5, typename V6>
  struct FunctionPtrTraits <U (*)(V1,V2,V3,V4,V5,V6)>
  {
    /** Value. */
    enum
    {
      IsFunctionPointer = 1                       /**< Function pointer.     */
    };
    /** Value. */ enum
    {
      nArgs = 6                                   /**< Number of arguments.  */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef V1 Arg1Type;                          /**< First argument type.  */
    typedef V2 Arg2Type;                          /**< Second argument type. */
    typedef V3 Arg3Type;                          /**< Third argument type.  */
    typedef V4 Arg4Type;                          /**< Fourth argument type. */
    typedef V5 Arg5Type;                          /**< Fifth argument type.  */
    typedef V6 Arg6Type;                          /**< Sixth argument type.  */
  };
  /**
   *  Not a pointer to member type.
   *  \tparam U \deduced Return type.
   */
  template <typename U>
  struct PtrToMemberTraits
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 0                       /**< Not a pointer to member.    */
    };
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   */
  template <typename U,  typename V>
  struct PtrToMemberTraits <U (V::*)(void)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 0                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.                */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   */
  template <typename U,  typename V>
  struct PtrToMemberTraits <U (V::*)(void) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 0                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.                */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1>
  struct PtrToMemberTraits <U (V::*)(W1)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 1                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1>
  struct PtrToMemberTraits <U (V::*)(W1) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 1                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2>
  struct PtrToMemberTraits <U (V::*)(W1,W2)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 2                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2>
  struct PtrToMemberTraits <U (V::*)(W1,W2) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 2                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 3                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 3                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 4                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 4                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   *  \tparam W5 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4, typename W5>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4,W5)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 5                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
    typedef W5 Arg5Type;                          /**< Fifth argument type.  */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   *  \tparam W5 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4, typename W5>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4,W5) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 5                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
    typedef W5 Arg5Type;                          /**< Fifth argument type.  */
  };
  /**
   *  Pointer to member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   *  \tparam W5 \deduced Argument type.
   *  \tparam W6 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4, typename W5, typename W6>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4,W5,W6)>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 6                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
    typedef W5 Arg5Type;                          /**< Fifth argument type.  */
    typedef W6 Arg6Type;                          /**< Sixth argument type.  */
  };
  /**
   *  Pointer to const member function.
   *  \tparam U \deduced Return type.
   *  \tparam V \deduced Class type.
   *  \tparam W1 \deduced Argument type.
   *  \tparam W2 \deduced Argument type.
   *  \tparam W3 \deduced Argument type.
   *  \tparam W4 \deduced Argument type.
   *  \tparam W5 \deduced Argument type.
   *  \tparam W6 \deduced Argument type.
   */
  template <typename U,  typename V,
            typename W1, typename W2, typename W3,
            typename W4, typename W5, typename W6>
  struct PtrToMemberTraits <U (V::*)(W1,W2,W3,W4,W5,W6) const>
  {
    /** Value. */
    enum
    {
      IsPointerToMember = 1                       /**< Pointer to member function. */
    };
    /** Value. */ enum
    {
      nArgs = 6                                   /**< Number of arguments.        */
    };
    typedef U ReturnType;                         /**< Return type.          */
    typedef W1 Arg1Type;                          /**< First argument type.  */
    typedef W2 Arg2Type;                          /**< Second argument type. */
    typedef W3 Arg3Type;                          /**< Third argument type.  */
    typedef W4 Arg4Type;                          /**< Fourth argument type. */
    typedef W5 Arg5Type;                          /**< Fifth argument type.  */
    typedef W6 Arg6Type;                          /**< Sixth argument type.  */
  };

public:
  /** Not a const type. */
  typedef typename UnConst<T>::Result NonConstType;
  /** Referenced type. */
  typedef typename ReferenceTraits<T>::ReferencedType ReferencedType;
  /** Pointee type. */
  typedef typename PointerTraits<T>::PointeeType PointeeType;
  /** Base type, after removing \c &, \c * and \c const. */
  typedef typename Base<T>::Type BaseType;
  /** Ptr base type. */
  typedef typename PtrBase<T>::Type PtrBaseType;
  /** Predicates. */
  enum
  {
    /** Pointer to member predicate. */
    IsPointerToMember = PtrToMemberTraits<T>::IsPointerToMember,
    /** Pointer predicate. */
    IsPointer = PointerTraits<T>::IsPointer,
    /** Ptr predicate. */
    IsPtr = PointerTraits<T>::IsPtr,
    /** Reference predicate. */
    IsReference = ReferenceTraits<T>::IsReference,
    /** Function pointer predicate. */
    IsFunctionPointer = FunctionPtrTraits<T>::IsFunctionPointer
  };
  /** Pointer to member traits type. */
  typedef PtrToMemberTraits<T> PointerToMemberTraits;
  /** Function pointer traits. */
  typedef FunctionPtrTraits<T> FunctionPointerTraits;
};

}  // namespace ns3

#endif /* TYPE_TRAITS_H */
