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
#ifndef ENUM_VALUE_H
#define ENUM_VALUE_H

#include "attribute.h"
#include "attribute-accessor-helper.h"
#include <list>

/**
 * \file
 * \ingroup attribute_Enum
 * ns3::EnumValue attribute value declarations.
 */

namespace ns3 {

//  Additional docs for class EnumValue:
/**
 * Hold variables of type \c enum
 *
 * This class can be used to hold variables of any kind
 * of enum.
 *
 * This is often used with ObjectFactory and Config to bind
 * the value of a particular enum to an Attribute or Config name.
 * For example,
 * \code
 *   Ptr<RateErrorModel> model = CreateObjectWithAttributes<RateErrorModel> (
 *     "ErrorRate", DoubleValue (0.05),
 *     "ErrorUnit", EnumValue (RateErrorModel::ERROR_UNIT_PACKET));
 *
 *   Config::SetDefault ("ns3::RipNg::SplitHorizon",
 *                       EnumValue (RipNg::NO_SPLIT_HORIZON));
 * \endcode
 */
class EnumValue : public AttributeValue
{
public:
  EnumValue ();
  /**
   * Construct from an explicit value.
   *
   * \param [in] value The value to begin with.
   */
  EnumValue (int value);
  void Set (int  value);
  int Get (void) const;
  template <typename T>
  bool GetAccessor (T & value) const;

  virtual Ptr<AttributeValue> Copy (void) const;
  virtual std::string SerializeToString (Ptr<const AttributeChecker> checker) const;
  virtual bool DeserializeFromString (std::string value, Ptr<const AttributeChecker> checker);

private:
  int m_value;  //!< The stored integer value.
};

template <typename T>
bool
EnumValue::GetAccessor (T &value) const
{
  value = T (m_value);
  return true;
}

class EnumChecker : public AttributeChecker
{
public:
  EnumChecker ();

  /**
   * Add a default value.
   * \param [in] value The value.
   * \param [in] name Then enum symbol name.
   */
  void AddDefault (int value, std::string name);
  /**
   * Add a new value.
   * \param [in] value The value.
   * \param [in] name The enum symbol name.
   */
  void Add (int value, std::string name);

  /**
   * Get the enum symbol name by value.
   * \param [in] value The value.
   * \return The enum symbol name.
   */
  std::string GetName (int value) const;

  /**
   * Get the enum value by name.
   * \param [in] name Then enum symbol name.
   * \returns The enum value.
   */
  int GetValue (const std::string name) const;

  // Inherited
  virtual bool Check (const AttributeValue &value) const;
  virtual std::string GetValueTypeName (void) const;
  virtual bool HasUnderlyingTypeInformation (void) const;
  virtual std::string GetUnderlyingTypeInformation (void) const;
  virtual Ptr<AttributeValue> Create (void) const;
  virtual bool Copy (const AttributeValue &src, AttributeValue &dst) const;

private:
  /** Type for the pair value, name */
  typedef std::pair<int,std::string> Value;
  /** Type of container for storing Enum values and symbol names. */
  typedef std::list<Value> ValueSet;
  /** The stored Enum values and symbol names. */
  ValueSet m_valueSet;
};

template <typename T1>
Ptr<const AttributeAccessor> MakeEnumAccessor (T1 a1);

template <typename T1, typename T2>
Ptr<const AttributeAccessor> MakeEnumAccessor (T1 a1, T2 a2);

/**
 * Make an EnumChecker pre-configured with a set of allowed
 * values by name.
 *
 * Values are normally given as fully qualified enum symbols
 * with matching names.  For example,
 * \c MakeEnumChecker (RipNg::SPLIT_HORIZON, "ns3::RipNg::SplitHorizon");
 *
 * As many additional enum value, name pairs as desired can be passed
 * as arguments.
 *
 * \see AttributeChecker
 *
 * \tparam Ts The type list of additional parameters. Additional parameters
 *            should be int, string pairs.
 * \returns The AttributeChecker
 * \param [in] v  The default enum value.
 * \param [in] n  The corresponding name.
 * \param [in] args Any additional arguments.
 */
template <typename... Ts>
Ptr<const AttributeChecker>
MakeEnumChecker (int v, std::string n, Ts... args)
{
  Ptr<EnumChecker> checker = Create<EnumChecker> ();
  checker->AddDefault (v, n);
  return MakeEnumChecker (checker, args...);
}

/**
 * Handler for enum value, name pairs other than the default.
 *
 * \tparam Ts The type list of additional parameters. Additional parameters
 *            should be in int, string pairs.
 * \returns The AttributeChecker
 * \param [in] checker The AttributeChecker.
 * \param [in] v  The next enum value.
 * \param [in] n  The corresponding name.
 * \param [in] args Any additional arguments.
 */

template <typename... Ts>
Ptr<const AttributeChecker>
MakeEnumChecker (Ptr<EnumChecker> checker, int v, std::string n, Ts... args)
{
  checker->Add (v, n);
  return MakeEnumChecker (checker, args...);
}

/**
 * Terminate the recursion of variadic arguments.
 *
 * \returns The \p checker
 * \param [in] checker The AttributeChecker.
 */
// inline to allow tail call optimization
inline
Ptr<const AttributeChecker>
MakeEnumChecker (Ptr<EnumChecker> checker)
{
  return checker;
}


template <typename T1>
Ptr<const AttributeAccessor> MakeEnumAccessor (T1 a1)
{
  return MakeAccessorHelper<EnumValue> (a1);
}

template <typename T1, typename T2>
Ptr<const AttributeAccessor> MakeEnumAccessor (T1 a1, T2 a2)
{
  return MakeAccessorHelper<EnumValue> (a1, a2);
}

} // namespace ns3

#endif /* ENUM_VALUE_H */
