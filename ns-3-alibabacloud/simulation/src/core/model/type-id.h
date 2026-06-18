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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef TYPE_ID_H
#define TYPE_ID_H

#include "attribute.h"
#include "attribute-accessor-helper.h"
#include "trace-source-accessor.h"
#include "attribute-helper.h"
#include "callback.h"
#include "deprecated.h"
#include "hash.h"
#include <string>
#include <stdint.h>

/**
 * \file
 * \ingroup object
 * ns3::TypeId declaration; inline and template implementations.
 */

namespace ns3 {

class ObjectBase;

/**
 * \ingroup object
 * \brief a unique identifier for an interface.
 *
 * This class records a lot of meta-information about a
 * subclass of the Object base class:
 *  - the base class of the subclass
 *  - the set of accessible constructors in the subclass
 *  - the set of 'attributes' accessible in the subclass
 *
 * \see attribute_TypeId
 *
 * \internal
 *  See the discussion in IidManager about hash chaining of TypeId's.
 */
class TypeId
{
public:
  /** Flags describing when a given attribute can be read or written. */
  enum AttributeFlag
  {
    ATTR_GET = 1 << 0, /**< The attribute can be read */
    ATTR_SET = 1 << 1, /**< The attribute can be written */
    ATTR_CONSTRUCT = 1 << 2, /**< The attribute can be written at construction-time */
    ATTR_SGC = ATTR_GET | ATTR_SET | ATTR_CONSTRUCT, /**< The attribute can be read, and written at any time */
  };
  /** The level of support or deprecation for attributes or trace sources. */
  enum SupportLevel
  {
    SUPPORTED,   /**< Attribute or trace source is currently used. */
    DEPRECATED,  /**< Attribute or trace source is deprecated; user is warned. */
    OBSOLETE     /**< Attribute or trace source is not used anymore; simulation fails. */
  };
  /** Attribute implementation. */
  struct AttributeInformation
  {
    /** Attribute name. */
    std::string name;
    /** Attribute help string. */
    std::string help;
    /** AttributeFlags value. */
    uint32_t flags;
    /** Default initial value. */
    Ptr<const AttributeValue> originalInitialValue;
    /** Configured initial value. */
    Ptr<const AttributeValue> initialValue;
    /** Accessor object. */
    Ptr<const AttributeAccessor> accessor;
    /** Checker object. */
    Ptr<const AttributeChecker> checker;
    /** Support level/deprecation. */
    TypeId::SupportLevel supportLevel;
    /** Support message. */
    std::string supportMsg;
  };
  /** TraceSource implementation. */
  struct TraceSourceInformation
  {
    /** Trace name. */
    std::string name;
    /** Trace help string. */
    std::string help;
    /** Callback function signature type. */
    std::string callback;
    /** Trace accessor. */
    Ptr<const TraceSourceAccessor> accessor;
    /** Support level/deprecation. */
    TypeId::SupportLevel supportLevel;
    /** Support message. */
    std::string supportMsg;
  };

  /** Type of hash values. */
  typedef uint32_t hash_t;

  /**
   * Get a TypeId by name.
   *
   * \param [in] name The name of the requested TypeId
   * \returns The unique id associated with the requested name.
   *
   * This method cannot fail: it will crash if the input
   * name is not a valid TypeId name.
   */
  static TypeId LookupByName (std::string name);
  /**
   * Get a TypeId by name.
   *
   * \param [in]  name The name of the requested TypeId
   * \param [out] tid A pointer to the TypeId instance where the
   *              result of this function should be stored.
   * \returns \c true if the requested name was found.
   */
  static bool LookupByNameFailSafe (std::string name, TypeId *tid);
  /**
   * Get a TypeId by hash.
   *
   * \param [in] hash The hash to lookup
   * \returns The unique id associated with the requested hash.
   *
   * This method cannot fail: it will crash if the input
   * hash does not match an existing TypeId.
   */
  static TypeId LookupByHash (hash_t hash);
  /**
   * Get a TypeId by hash.
   *
   * \param [in]  hash The hash of the requested TypeId
   * \param [out] tid A pointer to the TypeId instance where the
   *              result of this function should be stored.
   * \returns \c true if the requested hash was found.
   */
  static bool LookupByHashFailSafe (hash_t hash, TypeId *tid);

  /**
   * Get the number of registered TypeIds.
   *
   * \returns The number of TypeId instances registered.
   */
  static uint16_t GetRegisteredN (void);
  /**
   * Get a TypeId by index.
   *
   * \param [in] i Index of the TypeId.
   * \returns The TypeId instance whose index is \c i.
   */
  static TypeId GetRegistered (uint16_t i);

  /**
   * Constructor.
   *
   * \param [in] name The name of the interface to construct.
   *
   * No two instances can share the same name. The name is expected to be
   * the full c++ typename of associated c++ object.
   */
  explicit TypeId (const std::string &name);

  /**
   * Get the parent of this TypeId.
   *
   * \returns The parent of this TypeId
   *
   * This method cannot fail. It will return itself
   * if this TypeId has no parent. i.e., it is at the top
   * of the TypeId hierarchy. Currently, this is the
   * case for the TypeId associated to the ns3::ObjectBase class
   * only.
   */
  TypeId GetParent (void) const;

  /**
   * Check if this TypeId has a parent.
   *
   * \return \c true if this TypeId has a parent.
   */
  bool HasParent (void) const;

  /**
   * Check if this TypeId is a child of another.
   *
   * \param [in] other A parent TypeId
   * \returns \c true if the input TypeId is really a parent of this TypeId.
   *
   * Calling this method is roughly similar to calling dynamic_cast
   * except that you do not need object instances: you can do the check
   * with TypeId instances instead.
   */
  bool IsChildOf (TypeId other) const;

  /**
   * Get the group name.
   *
   * \returns The name of the group associated to this TypeId.
   */
  std::string GetGroupName (void) const;

  /**
   * Get the name.
   *
   * \returns The name of this interface.
   */
  std::string GetName (void) const;

  /**
   * Get the hash.
   *
   * \returns The hash of this interface.
   */
  hash_t GetHash (void) const;

  /**
   * Get the size of this object.
   *
   * \returns The size of this interface.
   */
  std::size_t GetSize (void) const;

  /**
   * Check if this TypeId has a constructor.
   *
   * \returns \c true if this TypeId has a constructor
   */
  bool HasConstructor (void) const;

  /**
   * Get the number of attributes.
   *
   * \returns The number of attributes associated to this TypeId
   */
  std::size_t GetAttributeN (void) const;
  /**
   * Get Attribute information by index.
   *
   * \param [in] i Index into attribute array
   * \returns The information associated to attribute whose index is \pname{i}.
   */
  struct TypeId::AttributeInformation GetAttribute (std::size_t i) const;
  /**
   * Get the Attribute name by index.
   *
   * \param [in] i Index into attribute array
   * \returns The full name associated to the attribute whose index is \pname{i}.
   */
  std::string GetAttributeFullName (std::size_t i) const;

  /**
   * Get the constructor callback.
   *
   * \returns A callback which can be used to instantiate an object
   *          of this type.
   */
  Callback<ObjectBase *> GetConstructor (void) const;

  /**
   * Check if this TypeId should not be listed in documentation.
   *
   * \returns \c true if this TypeId should be hidden from the user.
   */
  bool MustHideFromDocumentation (void) const;


  /**
   * Get the number of Trace sources.
   *
   * \returns The number of trace sources defined in this TypeId.
   */
  std::size_t GetTraceSourceN (void) const;
  /**
   * Get the trace source by index.
   *
   * \param [in] i Index into trace source array.
   * \returns Detailed information about the requested trace source.
   */
  struct TypeId::TraceSourceInformation GetTraceSource (std::size_t i) const;

  /**
   * Set the parent TypeId.
   *
   * \param [in] tid The TypeId of the base class.
   * \return This TypeId instance.
   *
   * Record in this TypeId which TypeId is the TypeId
   * of the base class of the subclass.
   */
  TypeId SetParent (TypeId tid);
  /**
   * Set the parent TypeId.
   *
   * \tparam T \explicit The parent TypeID type.
   * \return This TypeId instance.
   *
   * Record in this TypeId which TypeId is the TypeId
   * of the base class of the subclass.
   */
  template <typename T>
  TypeId SetParent (void);

  /**
   * Set the group name.
   *
   * \param [in] groupName The name of the group this TypeId belongs to.
   * \returns This TypeId instance.
   *
   * The group name is purely an advisory information used to
   * group together types according to a user-specific grouping
   * scheme.
   */
  TypeId SetGroupName (std::string groupName);

  /**
   * Set the size of this type.
   *
   * Call this way:
   * \code
   *   SetSize (sizeof (<typename>));
   * \endcode
   * This is done automatically by NS_LOG_ENSURE_REGISTERED()
   * A ridiculously large reported size is a symptom that the
   * type hasn't been registered.
   *
   * \param [in] size The size of the object, in bytes.
   * \returns This TypeId instance.
   */
  TypeId SetSize (std::size_t size);

  /**
   * Record in this TypeId the fact that the default constructor
   * is accessible.
   *
   * \tparam T \explicit The class name represented by this TypeId.
   * \returns This TypeId instance
   */
  template <typename T>
  TypeId AddConstructor (void);

  /**
   * Record in this TypeId the fact that a new attribute exists.
   *
   * \param [in] name The name of the new attribute
   * \param [in] help Some help text which describes the purpose of this
   *             attribute.
   * \param [in] initialValue The initial value for this attribute.
   * \param [in] accessor An instance of the associated AttributeAccessor
   *             subclass.
   * \param [in] checker An instance of the associated AttributeChecker
   *             subclass.
   * \param [in] supportLevel Support/deprecation status of the attribute.
   * \param [in] supportMsg Upgrade hint if this attribute is no longer
   *             supported.  If the attribute is \c DEPRECATED the attribute
   *             behavior still exists, but user code should be updated
   *             following guidance in the hint.
   *             If the attribute is \c OBSOLETE, the hint should indicate
   *             which class the attribute functional has been moved to,
   *             or that the functionality is no longer supported.
   *             See test file type-id-test-suite.cc for examples.
   * \returns This TypeId instance
   */
  TypeId AddAttribute (std::string name,
                       std::string help,
                       const AttributeValue &initialValue,
                       Ptr<const AttributeAccessor> accessor,
                       Ptr<const AttributeChecker> checker,
                       SupportLevel supportLevel = SUPPORTED,
                       const std::string &supportMsg = "");

  /**
   * Set the initial value of an Attribute.
   *
   * \param [in] i The attribute to manipulate
   * \param [in] initialValue The new initial value to use for this attribute.
   * \returns \c true if the call was successfully.
   */
  bool SetAttributeInitialValue (std::size_t i,
                                 Ptr<const AttributeValue> initialValue);

  /**
   * Record in this TypeId the fact that a new attribute exists.
   *
   * \param [in] name The name of the new attribute
   * \param [in] help Some help text which describes the purpose of this
   *        attribute
   * \param [in] flags Flags which describe how this attribute can be read and/or written.
   * \param [in] initialValue The initial value for this attribute.
   * \param [in] accessor An instance of the associated AttributeAccessor
   *             subclass.
   * \param [in] checker An instance of the associated AttributeChecker
   *             subclass.
   * \param [in] supportLevel Support/deprecation status of the attribute.
   * \param [in] supportMsg Upgrade hint if this attribute is no longer
   *             supported.  If the attribute is \c DEPRECATED the attribute
   *             behavior still exists, but user code should be updated
   *             following guidance in the hint..
   *             If the attribute is \c OBSOLETE, the hint should indicate
   *             which class the attribute functional has been moved to,
   *             or that the functionality is no longer supported.
   * \returns This TypeId instance
   */
  TypeId AddAttribute (std::string name,
                       std::string help,
                       uint32_t flags,
                       const AttributeValue &initialValue,
                       Ptr<const AttributeAccessor> accessor,
                       Ptr<const AttributeChecker> checker,
                       SupportLevel supportLevel = SUPPORTED,
                       const std::string &supportMsg = "");

  /**
   * Record a new TraceSource.
   *
   * \param [in] name The name of the new trace source
   * \param [in] help Some help text which describes the purpose of this
   *             trace source.
   * \param [in] accessor A pointer to a TraceSourceAccessor which can be
   *             used to connect/disconnect sinks to this trace source.
   * \param [in] callback Fully qualified typedef name for the callback
   *             signature.  Generally this should begin with the
   *             "ns3::" namespace qualifier.
   * \param [in] supportLevel Support/deprecation status of the attribute.
   * \param [in] supportMsg Upgrade hint if this attribute is no longer
   *             supported.  If the attribute is \c DEPRECATED the attribute
   *             behavior still exists, but user code should be updated
   *             following guidance in the hint..
   *             If the attribute is \c OBSOLETE, the hint should indicate
   *             which class the attribute functional has been moved to,
   *             or that the functionality is no longer supported.
   *             See test file type-id-test-suite.cc for examples.
   * \returns This TypeId instance.
   */
  TypeId AddTraceSource (std::string name,
                         std::string help,
                         Ptr<const TraceSourceAccessor> accessor,
                         std::string callback,
                         SupportLevel supportLevel = SUPPORTED,
                         const std::string &supportMsg = "");

  /**
   * Hide this TypeId from documentation.
   * \returns This TypeId instance.
   */
  TypeId HideFromDocumentation (void);

  /**
   * Find an Attribute by name, retrieving the associated AttributeInformation.
   *
   * \param [in]  name The name of the requested attribute
   * \param [in,out] info A pointer to the TypeId::AttributeInformation
   *              data structure where the result value of this method
   *              will be stored.
   * \returns \c true if the requested attribute could be found.
   */
  bool LookupAttributeByName (std::string name, struct AttributeInformation *info) const;
  /**
   * Find a TraceSource by name.
   *
   * If no matching trace source is found, this method returns zero.
   *
   * \param [in] name The name of the requested trace source
   * \return The trace source accessor which can be used to connect
   *  and disconnect trace sinks with the requested trace source on
   *  an object instance.
   */
  Ptr<const TraceSourceAccessor> LookupTraceSourceByName (std::string name) const;
  /**
   * Find a TraceSource by name, retrieving the associated TraceSourceInformation.
   *
   * \param [in]  name The name of the requested trace source.
   * \param [out] info A pointer to the TypeId::TraceSourceInformation
   *              data structure where the result value of this method
   *              will be stored.
   * \return The trace source accessor which can be used to connect
   *  and disconnect trace sinks with the requested trace source on
   *  an object instance.
   */
  Ptr<const TraceSourceAccessor> LookupTraceSourceByName (std::string name, struct TraceSourceInformation *info) const;

  /**
   * Get the internal id of this TypeId.
   *
   * \returns The internal integer which uniquely identifies this TypeId.
   *
   * This is really an internal method which users are not expected
   * to use.
   */
  uint16_t GetUid (void) const;
  /**
   * Set the internal id of this TypeId.
   *
   * \param [in] uid The internal integer which uniquely identifies
   *            this TypeId.  This TypeId should already have been registered.
   *
   * Typically this is used in serialization/deserialization.
   *
   * This method is even more internal than GetUid(). Use
   * at your own risk and don't be surprised that it eats raw
   * babies on full-moon nights.
   */
  void SetUid (uint16_t uid);

  /** Default constructor.  This produces an invalid TypeId. */
  inline TypeId ();
  /**
   * Copy constructor.
   * \param [in] o The other TypeId.
   */
  inline TypeId (const TypeId &o);
  /**
   * Assignment.
   * \param [in] o The other TypeId.
   * \returns The copied TypeId.
   */
  inline TypeId &operator = (const TypeId &o);
  /** Destructor. */
  inline ~TypeId ();

private:
  /**
   * \name Comparison operators.
   * Standard comparison operators.
   * @{
   */
  friend inline bool operator == (TypeId a, TypeId b);
  friend inline bool operator != (TypeId a, TypeId b);
  friend  bool operator <  (TypeId a, TypeId b);
  /**@}*/

  /**
   * Construct from an integer value.
   * \param [in] tid The TypeId value as an integer.
   */
  explicit TypeId (uint16_t tid);
  /**
   * Implementation for AddConstructor().
   *
   * \param [in] callback Callback which constructs an instance of this TypeId.
   */
  void DoAddConstructor (Callback<ObjectBase *> callback);

  /** The TypeId value. */
  uint16_t m_tid;
};

/**
 * \relates TypeId
 * Output streamer.
 *
 * \param [in,out] os The output stream.
 * \param [in] tid The TypeId.
 * \returns The output stream.
 */
std::ostream & operator << (std::ostream &os, TypeId tid);
/**
 * \relates TypeId
 * Input Streamer.
 * \param [in,out] is The input stream.
 * \param [out] tid The TypeId to set from the stream.
 * \returns The input stream.
 */
std::istream & operator >> (std::istream &is, TypeId &tid);

/**
  * Comparison operator.
  * \param [in] a One value.
  * \param [in] b The other value.
  * \returns The result of the comparison.
  * @{
  */
inline bool operator == (TypeId a, TypeId b);
inline bool operator != (TypeId a, TypeId b);
bool operator <  (TypeId a, TypeId b);
/** @} */

ATTRIBUTE_HELPER_HEADER (TypeId);

} // namespace ns3

namespace ns3 {

TypeId::TypeId ()
  : m_tid (0)
{}
TypeId::TypeId (const TypeId &o)
  : m_tid (o.m_tid)
{}
TypeId &TypeId::operator = (const TypeId &o)
{
  m_tid = o.m_tid;
  return *this;
}
TypeId::~TypeId ()
{}
inline bool operator == (TypeId a, TypeId b)
{
  return a.m_tid == b.m_tid;
}

inline bool operator != (TypeId a, TypeId b)
{
  return a.m_tid != b.m_tid;
}


/*************************************************************************
 *   The TypeId implementation which depends on templates
 *************************************************************************/

template <typename T>
TypeId
TypeId::SetParent (void)
{
  return SetParent (T::GetTypeId ());
}

template <typename T>
TypeId
TypeId::AddConstructor (void)
{
  struct Maker
  {
    static ObjectBase * Create ()
    {
      ObjectBase * base = new T ();
      return base;
    }
  };
  Callback<ObjectBase *> cb = MakeCallback (&Maker::Create);
  DoAddConstructor (cb);
  return *this;
}

} // namespace ns3

#endif /* TYPE_ID_H */
