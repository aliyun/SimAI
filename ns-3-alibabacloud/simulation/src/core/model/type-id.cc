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
#include "log.h"  // NS_ASSERT and NS_LOG
#include "hash.h"
#include "type-id.h"
#include "singleton.h"
#include "trace-source-accessor.h"

#include <map>
#include <vector>
#include <sstream>
#include <iomanip>

/**
 * \file
 * \ingroup object
 * ns3::TypeId and ns3::IidManager implementations.
 */


/*********************************************************************
 *         Helper code
 *********************************************************************/

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TypeId");

// IidManager needs to be in ns3 namespace for NS_ASSERT and NS_LOG
// to find g_log

/**
 * \ingroup object
 * \brief TypeId information manager
 *
 * Information records are stored in a vector.  Name and hash lookup
 * are performed by maps to the vector index.
 *
 * \internal
 * <b>Hash Chaining</b>
 *
 * We require all types to produce distinct hashes. What if we encounter
 * two types that produce the same hash value?  As we move to a
 * federated distribution model (the App store), it becomes increasingly
 * likely that the core ns3 team *won't* discover this in test builds.
 * Therefore, we need to handle this case explicitly.
 *
 * Note, we expect this to be *extremely* rare.  As of this writing we
 * have ~400 < 2^9 types, so the probability of getting a collision
 * when we introduce a new type is ~2^9/2^31 = 2^-22, assuming we
 * reserve 31 bits for the hash, and one bit for chaining.  Even with
 * double the number of types the probability of having a collision
 * is only 2 x 10^-4.  The probability for a three-fold collision is
 * 1 x 10^-10.
 *
 * Therefore, we'll handle one collision explicitly by reserving
 * the high order bit of the hash value, and assert on higher level
 * collisions.  The three-fold collision probability should be an
 * acceptablly small error rate.
 */
class IidManager : public Singleton<IidManager>
{
public:
  /**
   * Create a new unique type id.
   * \param [in] name The name of this type id.
   * \returns The id.
   */
  uint16_t AllocateUid (std::string name);
  /**
   * Set the parent of a type id.
   * \param [in] uid The id.
   * \param [in] parent The id of the parent.
   */
  void SetParent (uint16_t uid, uint16_t parent);
  /**
   * Set the group name of a type id.
   * \param [in] uid The id.
   * \param [in] groupName The group name.
   */
  void SetGroupName (uint16_t uid, std::string groupName);
  /**
   * Set the size of the object class referred to by this id.
   * \param [in] uid The id.
   * \param [in] size The object size.
   */
  void SetSize (uint16_t uid, std::size_t size);
  /**
   * Add a constructor Callback to this type id.
   * \param [in] uid The id.
   * \param [in] callback The Callback for the constructor.
   */
  void AddConstructor (uint16_t uid, Callback<ObjectBase *> callback);
  /**
   * Mark this type id to be excluded from documentation.
   * \param [in] uid The id.
   */
  void HideFromDocumentation (uint16_t uid);
  /**
   * Get a type id by name.
   * \param [in] name The type id to find.
   * \returns The type id.  A type id of 0 means \pname{name} wasn't found.
   */
  uint16_t GetUid (std::string name) const;
  /**
   * Get a type id by hash value.
   * \param [in] hash The type id to find.
   * \returns The type id.  A type id of 0 means \pname{hash} wasn't found.
   */
  uint16_t GetUid (TypeId::hash_t hash) const;
  /**
   * Get the name of a type id.
   * \param [in] uid The id.
   * \returns The name of the type id.
   */
  std::string GetName (uint16_t uid) const;
  /**
   * Get the hash of a type id.
   * \param [in] uid The id.
   * \returns The hash of the type id.
   */
  TypeId::hash_t GetHash (uint16_t uid) const;
  /**
   * Get the parent of a type id.
   * \param [in] uid The id.
   * \returns The parent type id of the type id.
   */
  uint16_t GetParent (uint16_t uid) const;
  /**
   * Get the group name of a type id.
   * \param [in] uid The id.
   * \returns The group name of the type id.
   */
  std::string GetGroupName (uint16_t uid) const;
  /**
   * Get the size of a type id.
   * \param [in] uid The id.
   * \returns The size of the type id.
   */
  std::size_t GetSize (uint16_t uid) const;
  /**
   * Get the constructor Callback of a type id.
   * \param [in] uid The id.
   * \returns The constructor Callback of the type id.
   */
  Callback<ObjectBase *> GetConstructor (uint16_t uid) const;
  /**
   * Check if a type id has a constructor Callback.
   * \param [in] uid The id.
   * \returns \c true if the type id has a constructor Callback.
   */
  bool HasConstructor (uint16_t uid) const;
  /**
   * Get the total number of type ids.
   * \returns The total number.
   */
  uint16_t GetRegisteredN (void) const;
  /**
   * Get a type id by index.
   *
   * The type id value 0 indicates not registered, so there is an offset
   * of 1 between the index and the type id value.  This function converts
   * from an index to the type id value.
   * \param [in] i The index.
   * \returns The type id.
   */
  uint16_t GetRegistered (uint16_t i) const;
  /**
   * Record a new attribute in a type id.
   * \param [in] uid The id.
   * \param [in] name The name of the new attribute
   * \param [in] help Some help text which describes the purpose of this
   *             attribute.
   * \param [in] flags Flags which describe how this attribute can be
   *             read and/or written.
   * \param [in] initialValue The initial value for this attribute.
   * \param [in] accessor An instance of the associated AttributeAccessor
   *             subclass.
   * \param [in] checker An instance of the associated AttributeChecker
   *             subclass.
   * \param [in] supportLevel The support/deprecation status for this attribute.
   * \param [in] supportMsg Upgrade hint if this attribute is no longer supported.
   */
  void AddAttribute (uint16_t uid,
                     std::string name,
                     std::string help,
                     uint32_t flags,
                     Ptr<const AttributeValue> initialValue,
                     Ptr<const AttributeAccessor> accessor,
                     Ptr<const AttributeChecker> checker,
                     TypeId::SupportLevel supportLevel = TypeId::SUPPORTED,
                     const std::string &supportMsg = "");
  /**
   * Set the initial value of an Attribute.
   * \param [in] uid The id.
   * \param [in] i The attribute to manipulate
   * \param [in] initialValue The new initial value to use for this attribute.
   */
  void SetAttributeInitialValue (uint16_t uid,
                                 std::size_t i,
                                 Ptr<const AttributeValue> initialValue);
  /**
   * Get the number of attributes.
   * \param [in] uid The id.
   * \returns The number of attributes associated to this TypeId
   */
  std::size_t GetAttributeN (uint16_t uid) const;
  /**
   * Get Attribute information by index.
   * \param [in] uid The id.
   * \param [in] i Index into attribute array
   * \returns The information associated to attribute whose index is \pname{i}.
   */
  struct TypeId::AttributeInformation GetAttribute (uint16_t uid, std::size_t i) const;
  /**
   * Record a new TraceSource.
   * \param [in] uid The id.
   * \param [in] name The name of the new trace source
   * \param [in] help Some help text which describes the purpose of this
   *             trace source.
   * \param [in] accessor A pointer to a TraceSourceAccessor which can be
   *             used to connect/disconnect sinks to this trace source.
   * \param [in] callback Fully qualified typedef name for the callback
   *             signature.  Generally this should begin with the
   *             "ns3::" namespace qualifier.
   * \param [in] supportLevel The support/deprecation status for this attribute.
   * \param [in] supportMsg Upgrade hint if this attribute is no longer supported.
   */
  void AddTraceSource (uint16_t uid,
                       std::string name,
                       std::string help,
                       Ptr<const TraceSourceAccessor> accessor,
                       std::string callback,
                       TypeId::SupportLevel supportLevel = TypeId::SUPPORTED,
                       const std::string &supportMsg = "");
  /**
   * Get the number of Trace sources.
   * \param [in] uid The id.
   * \returns The number of trace sources defined in this TypeId.
   */
  std::size_t GetTraceSourceN (uint16_t uid) const;
  /**
   * Get the trace source by index.
   * \param [in] uid The id.
   * \param [in] i Index into trace source array.
   * \returns Detailed information about the requested trace source.
   */
  struct TypeId::TraceSourceInformation GetTraceSource (uint16_t uid, std::size_t i) const;
  /**
   * Check if this TypeId should not be listed in documentation.
   * \param [in] uid The id.
   * \returns \c true if this TypeId should be hidden from the user.
   */
  bool MustHideFromDocumentation (uint16_t uid) const;

private:
  /**
   * Check if a type id has a given TraceSource.
   * \param [in] uid The id.
   * \param [in] name The TraceSource name.
   * \returns \c true if \pname{uid} has the TraceSource \pname{name}.
   */
  bool HasTraceSource (uint16_t uid, std::string name);
  /**
   * Check if a type id has a given Attribute.
   * \param [in] uid The id.
   * \param [in] name The Attribute name.
   * \returns \c true if \pname{uid} has the Attribute \pname{name}.
   */
  bool HasAttribute (uint16_t uid, std::string name);
  /**
   * Hashing function.
   * \param [in] name The type id name.
   * \returns The hashed value of \pname{name}.
   */
  static TypeId::hash_t Hasher (const std::string name);

  /** The information record about a single type id. */
  struct IidInformation
  {
    /** The type id name. */
    std::string name;
    /** The type id hash value. */
    TypeId::hash_t hash;
    /** The parent type id. */
    uint16_t parent;
    /** The group name. */
    std::string groupName;
    /** The size of the object represented by this type id. */
    std::size_t size;
    /** \c true if a constructor Callback has been registered. */
    bool hasConstructor;
    /** The constructor Callback. */
    Callback<ObjectBase *> constructor;
    /** \c true if this type should be omitted from documentation. */
    bool mustHideFromDocumentation;
    /** The container of Attributes. */
    std::vector<struct TypeId::AttributeInformation> attributes;
    /** The container of TraceSources. */
    std::vector<struct TypeId::TraceSourceInformation> traceSources;
    /** Support level/deprecation. */
    TypeId::SupportLevel supportLevel;
    /** Support message. */
    std::string supportMsg;
  };
  /** Iterator type. */
  typedef std::vector<struct IidInformation>::const_iterator Iterator;

  /**
   * Retrieve the information record for a type.
   * \param [in] uid The id.
   * \returns The information record.
   */
  struct IidManager::IidInformation * LookupInformation (uint16_t uid) const;

  /** The container of all type id records. */
  std::vector<struct IidInformation> m_information;

  /** Type of the by-name index. */
  typedef std::map<std::string, uint16_t> namemap_t;
  /** The by-name index. */
  namemap_t m_namemap;

  /** Type of the by-hash index. */
  typedef std::map<TypeId::hash_t, uint16_t> hashmap_t;
  /** The by-hash index. */
  hashmap_t m_hashmap;


  /** IidManager constants. */
  enum
  {
    /**
     * Hash chaining flag.
     *
     * To handle the first collision, we reserve the high bit as a
     * chain flag.
     */
    HashChainFlag = 0x80000000
  };
};


//static
TypeId::hash_t
IidManager::Hasher (const std::string name)
{
  static ns3::Hasher hasher ( Create<Hash::Function::Murmur3> () );
  return hasher.clear ().GetHash32 (name);
}

/**
 * \ingroup object
 * \internal
 * IidManager shorthand for use in NS_LOG
 */
#define IID "IidManager"
/**
 * \ingroup object
 * \internal
 * IidManager shorthand for use in NS_LOG
 */
#define IIDL IID << ": "

uint16_t
IidManager::AllocateUid (std::string name)
{
  NS_LOG_FUNCTION (IID << name);
  // Type names are definitive: equal names are equal types
  NS_ASSERT_MSG (m_namemap.count (name) == 0,
                 "Trying to allocate twice the same uid: " << name);

  TypeId::hash_t hash = Hasher (name) & (~HashChainFlag);
  if (m_hashmap.count (hash) == 1)
    {
      NS_LOG_ERROR ("Hash chaining TypeId for '" << name << "'.  "
                                                 << "This is not a bug, but is extremely unlikely.  "
                                                 << "Please contact the ns3 developers.");
      // ns3 developer contacted about this message:
      // You have four options (in order of difficulty):
      //   1. Let it ride, and play the odds that a third collision
      //        never appears.
      //   2. Change the name of the new (or old) tag, even trivially, to
      //        remove the collision.
      //   3. Switch to 64-bit hashes.
      //   4. Implement 2-bit (or higher) chaining.
      //
      //  Oh, by the way, I owe you a beer, since I bet Mathieu that
      //  this would never happen..  -- Peter Barnes, LLNL

      NS_ASSERT_MSG (m_hashmap.count (hash | HashChainFlag) == 0,
                     "Triplicate hash detected while chaining TypeId for '"
                     << name
                     << "'. Please contact the ns3 developers for assistance.");
      // ns3 developer contacted about this message:
      // You have three options: #2-4 above.
      //
      // Oh, by the way, I have no idea how this crazy hashing idea got
      // into ns3.  -- Peter Barnes, LLNL

      // Alphabetize the two types, so it's deterministic
      struct IidInformation * hinfo = LookupInformation (GetUid (hash));
      if (name > hinfo->name)
        { // new type gets chained
          NS_LOG_LOGIC (IIDL << "New TypeId '" << name << "' getting chained.");
          hash = hash | HashChainFlag;
        }
      else
        { // chain old type
          NS_LOG_LOGIC (IIDL << "Old TypeId '" << hinfo->name << "' getting chained.");
          uint16_t oldUid = GetUid (hinfo->hash);
          m_hashmap.erase (m_hashmap.find (hinfo->hash));
          hinfo->hash = hash | HashChainFlag;
          m_hashmap.insert (std::make_pair (hinfo->hash, oldUid));
          // leave new hash unchained
        }
    }

  struct IidInformation information;
  information.name = name;
  information.hash = hash;
  information.parent = 0;
  information.groupName = "";
  information.size = (std::size_t)(-1);
  information.hasConstructor = false;
  information.mustHideFromDocumentation = false;
  information.supportLevel = TypeId::SUPPORTED;
  m_information.push_back (information);
  std::size_t tuid = m_information.size ();
  NS_ASSERT (tuid <= 0xffff);
  uint16_t uid = static_cast<uint16_t> (tuid);

  // Add to both maps:
  m_namemap.insert (std::make_pair (name, uid));
  m_hashmap.insert (std::make_pair (hash, uid));
  NS_LOG_LOGIC (IIDL << uid);
  return uid;
}

struct IidManager::IidInformation *
IidManager::LookupInformation (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  NS_ASSERT (uid <= m_information.size () && uid != 0);
  NS_LOG_LOGIC (IIDL << m_information[uid - 1].name);
  return const_cast<struct IidInformation *> (&m_information[uid - 1]);
}

void
IidManager::SetParent (uint16_t uid, uint16_t parent)
{
  NS_LOG_FUNCTION (IID << uid << parent);
  NS_ASSERT (parent <= m_information.size ());
  struct IidInformation *information = LookupInformation (uid);
  information->parent = parent;
}
void
IidManager::SetGroupName (uint16_t uid, std::string groupName)
{
  NS_LOG_FUNCTION (IID << uid << groupName);
  struct IidInformation *information = LookupInformation (uid);
  information->groupName = groupName;
}
void
IidManager::SetSize (uint16_t uid, std::size_t size)
{
  NS_LOG_FUNCTION (IID << uid << size);
  struct IidInformation *information = LookupInformation (uid);
  information->size = size;
}
void
IidManager::HideFromDocumentation (uint16_t uid)
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  information->mustHideFromDocumentation = true;
}

void
IidManager::AddConstructor (uint16_t uid, Callback<ObjectBase *> callback)
{
  NS_LOG_FUNCTION (IID << uid << &callback);
  struct IidInformation *information = LookupInformation (uid);
  if (information->hasConstructor)
    {
      NS_FATAL_ERROR (information->name << " already has a constructor.");
    }
  information->hasConstructor = true;
  information->constructor = callback;
}

uint16_t
IidManager::GetUid (std::string name) const
{
  NS_LOG_FUNCTION (IID << name);
  uint16_t uid = 0;
  namemap_t::const_iterator it = m_namemap.find (name);
  if (it != m_namemap.end ())
    {
      uid = it->second;
    }
  NS_LOG_LOGIC (IIDL << uid);
  return uid;
}
uint16_t
IidManager::GetUid (TypeId::hash_t hash) const
{
  NS_LOG_FUNCTION (IID << hash);
  hashmap_t::const_iterator it = m_hashmap.find (hash);
  uint16_t uid = 0;
  if (it != m_hashmap.end ())
    {
      uid = it->second;
    }
  NS_LOG_LOGIC (IIDL << uid);
  return uid;
}
std::string
IidManager::GetName (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  NS_LOG_LOGIC (IIDL << information->name);
  return information->name;
}
TypeId::hash_t
IidManager::GetHash (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  TypeId::hash_t hash = information->hash;
  NS_LOG_LOGIC (IIDL << hash);
  return hash;
}
uint16_t
IidManager::GetParent (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  uint16_t pid = information->parent;
  NS_LOG_LOGIC (IIDL << pid);
  return pid;
}
std::string
IidManager::GetGroupName (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  NS_LOG_LOGIC (IIDL << information->groupName);
  return information->groupName;
}
std::size_t
IidManager::GetSize (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  std::size_t size = information->size;
  NS_LOG_LOGIC (IIDL << size);
  return size;
}

Callback<ObjectBase *>
IidManager::GetConstructor (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  if (!information->hasConstructor)
    {
      NS_FATAL_ERROR ("Requested constructor for " << information->name << " but it does not have one.");
    }
  return information->constructor;
}

bool
IidManager::HasConstructor (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  bool hasC = information->hasConstructor;
  NS_LOG_LOGIC (IIDL << hasC);
  return hasC;
}

uint16_t
IidManager::GetRegisteredN (void) const
{
  NS_LOG_FUNCTION (IID << m_information.size ());
  return static_cast<uint16_t> (m_information.size ());
}
uint16_t
IidManager::GetRegistered (uint16_t i) const
{
  NS_LOG_FUNCTION (IID << i);
  return i + 1;
}

bool
IidManager::HasAttribute (uint16_t uid,
                          std::string name)
{
  NS_LOG_FUNCTION (IID << uid << name);
  struct IidInformation *information  = LookupInformation (uid);
  while (true)
    {
      for (std::vector<struct TypeId::AttributeInformation>::const_iterator i = information->attributes.begin ();
           i != information->attributes.end (); ++i)
        {
          if (i->name == name)
            {
              NS_LOG_LOGIC (IIDL << true);
              return true;
            }
        }
      struct IidInformation *parent = LookupInformation (information->parent);
      if (parent == information)
        {
          // top of inheritance tree
          NS_LOG_LOGIC (IIDL << false);
          return false;
        }
      // check parent
      information = parent;
    }
  NS_LOG_LOGIC (IIDL << false);
  return false;
}

void
IidManager::AddAttribute (uint16_t uid,
                          std::string name,
                          std::string help,
                          uint32_t flags,
                          Ptr<const AttributeValue> initialValue,
                          Ptr<const AttributeAccessor> accessor,
                          Ptr<const AttributeChecker> checker,
                          TypeId::SupportLevel supportLevel,
                          const std::string &supportMsg)
{
  NS_LOG_FUNCTION (IID << uid << name << help << flags <<
                   initialValue << accessor << checker <<
                   supportLevel << supportMsg);
  struct IidInformation *information = LookupInformation (uid);
  if (name.find (' ') != std::string::npos)
    {
      NS_FATAL_ERROR ("Attribute name \"" << name << "\" may not contain spaces ' ', " <<
                      "encountered when registering TypeId \"" <<
                      information->name << "\"");
    }
  if (HasAttribute (uid, name))
    {
      NS_FATAL_ERROR ("Attribute \"" << name <<
                      "\" already registered on tid=\"" <<
                      information->name << "\"");
    }
  struct TypeId::AttributeInformation info;
  info.name = name;
  info.help = help;
  info.flags = flags;
  info.initialValue = initialValue;
  info.originalInitialValue = initialValue;
  info.accessor = accessor;
  info.checker = checker;
  info.supportLevel = supportLevel;
  info.supportMsg = supportMsg;
  information->attributes.push_back (info);
  NS_LOG_LOGIC (IIDL << information->attributes.size () - 1);
}
void
IidManager::SetAttributeInitialValue (uint16_t uid,
                                      std::size_t i,
                                      Ptr<const AttributeValue> initialValue)
{
  NS_LOG_FUNCTION (IID << uid << i << initialValue);
  struct IidInformation *information = LookupInformation (uid);
  NS_ASSERT (i < information->attributes.size ());
  information->attributes[i].initialValue = initialValue;
}



std::size_t
IidManager::GetAttributeN (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  std::size_t size = information->attributes.size ();
  NS_LOG_LOGIC (IIDL << size);
  return size;
}
struct TypeId::AttributeInformation
IidManager::GetAttribute (uint16_t uid, std::size_t i) const
{
  NS_LOG_FUNCTION (IID << uid << i);
  struct IidInformation *information = LookupInformation (uid);
  NS_ASSERT (i < information->attributes.size ());
  NS_LOG_LOGIC (IIDL << information->name);
  return information->attributes[i];
}

bool
IidManager::HasTraceSource (uint16_t uid,
                            std::string name)
{
  NS_LOG_FUNCTION (IID << uid << name);
  struct IidInformation *information  = LookupInformation (uid);
  while (true)
    {
      for (std::vector<struct TypeId::TraceSourceInformation>::const_iterator i = information->traceSources.begin ();
           i != information->traceSources.end (); ++i)
        {
          if (i->name == name)
            {
              NS_LOG_LOGIC (IIDL << true);
              return true;
            }
        }
      struct IidInformation *parent = LookupInformation (information->parent);
      if (parent == information)
        {
          // top of inheritance tree
          NS_LOG_LOGIC (IIDL << false);
          return false;
        }
      // check parent
      information = parent;
    }
  NS_LOG_LOGIC (IIDL << false);
  return false;
}

void
IidManager::AddTraceSource (uint16_t uid,
                            std::string name,
                            std::string help,
                            Ptr<const TraceSourceAccessor> accessor,
                            std::string callback,
                            TypeId::SupportLevel supportLevel,
                            const std::string &supportMsg)
{
  NS_LOG_FUNCTION (IID << uid << name << help <<
                   accessor << callback <<
                   supportLevel << supportMsg);
  struct IidInformation *information  = LookupInformation (uid);
  if (HasTraceSource (uid, name))
    {
      NS_FATAL_ERROR ("Trace source \"" << name <<
                      "\" already registered on tid=\"" <<
                      information->name << "\"");
    }
  struct TypeId::TraceSourceInformation source;
  source.name = name;
  source.help = help;
  source.accessor = accessor;
  source.callback = callback;
  source.supportLevel = supportLevel;
  source.supportMsg = supportMsg;
  information->traceSources.push_back (source);
  NS_LOG_LOGIC (IIDL << information->traceSources.size () - 1);
}
std::size_t
IidManager::GetTraceSourceN (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  std::size_t size = information->traceSources.size ();
  NS_LOG_LOGIC (IIDL << size);
  return size;
}
struct TypeId::TraceSourceInformation
IidManager::GetTraceSource (uint16_t uid, std::size_t i) const
{
  NS_LOG_FUNCTION (IID << uid << i);
  struct IidInformation *information = LookupInformation (uid);
  NS_ASSERT (i < information->traceSources.size ());
  NS_LOG_LOGIC (IIDL << information->name);
  return information->traceSources[i];
}
bool
IidManager::MustHideFromDocumentation (uint16_t uid) const
{
  NS_LOG_FUNCTION (IID << uid);
  struct IidInformation *information = LookupInformation (uid);
  bool hide = information->mustHideFromDocumentation;
  NS_LOG_LOGIC (IIDL << hide);
  return hide;
}

} // namespace ns3

namespace ns3 {

/*********************************************************************
 *         The TypeId class
 *********************************************************************/

TypeId::TypeId (const std::string &name)
{
  NS_LOG_FUNCTION (this << name);
  uint16_t uid = IidManager::Get ()->AllocateUid (name);
  NS_LOG_LOGIC (uid);
  NS_ASSERT (uid != 0);
  m_tid = uid;
}


TypeId::TypeId (uint16_t tid)
  : m_tid (tid)
{
  NS_LOG_FUNCTION (this << tid);
}
TypeId
TypeId::LookupByName (std::string name)
{
  NS_LOG_FUNCTION (name);
  uint16_t uid = IidManager::Get ()->GetUid (name);
  NS_ASSERT_MSG (uid != 0, "Assert in TypeId::LookupByName: " << name << " not found");
  return TypeId (uid);
}
bool
TypeId::LookupByNameFailSafe (std::string name, TypeId *tid)
{
  NS_LOG_FUNCTION (name << tid->GetUid ());
  uint16_t uid = IidManager::Get ()->GetUid (name);
  if (uid == 0)
    {
      return false;
    }
  *tid = TypeId (uid);
  return true;
}
TypeId
TypeId::LookupByHash (hash_t hash)
{
  uint16_t uid = IidManager::Get ()->GetUid (hash);
  NS_ASSERT_MSG (uid != 0, "Assert in TypeId::LookupByHash: 0x"
                 << std::hex << hash << std::dec << " not found");
  return TypeId (uid);
}
bool
TypeId::LookupByHashFailSafe (hash_t hash, TypeId *tid)
{
  uint16_t uid = IidManager::Get ()->GetUid (hash);
  if (uid == 0)
    {
      return false;
    }
  *tid = TypeId (uid);
  return true;
}

uint16_t
TypeId::GetRegisteredN (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return IidManager::Get ()->GetRegisteredN ();
}
TypeId
TypeId::GetRegistered (uint16_t i)
{
  NS_LOG_FUNCTION (i);
  return TypeId (IidManager::Get ()->GetRegistered (i));
}

bool
TypeId::LookupAttributeByName (std::string name, struct TypeId::AttributeInformation *info) const
{
  NS_LOG_FUNCTION (this << name << info);
  TypeId tid;
  TypeId nextTid = *this;
  do
    {
      tid = nextTid;
      for (std::size_t i = 0; i < tid.GetAttributeN (); i++)
        {
          struct TypeId::AttributeInformation tmp = tid.GetAttribute (i);
          if (tmp.name == name)
            {
              if (tmp.supportLevel == TypeId::SUPPORTED)
                {
                  *info = tmp;
                  return true;
                }
              else if (tmp.supportLevel == TypeId::DEPRECATED)
                {
                  std::cerr << "Attribute '" << name << "' is deprecated: "
                            << tmp.supportMsg << std::endl;
                  *info = tmp;
                  return true;
                }
              else if (tmp.supportLevel == TypeId::OBSOLETE)
                {
                  NS_FATAL_ERROR ("Attribute '" << name <<
                                  "' is obsolete, with no fallback: " <<
                                  tmp.supportMsg);
                }
            }
        }
      nextTid = tid.GetParent ();
    }
  while (nextTid != tid);
  return false;
}

TypeId
TypeId::SetParent (TypeId tid)
{
  NS_LOG_FUNCTION (this << tid.GetUid ());
  IidManager::Get ()->SetParent (m_tid, tid.m_tid);
  return *this;
}
TypeId
TypeId::SetGroupName (std::string groupName)
{
  NS_LOG_FUNCTION (this << groupName);
  IidManager::Get ()->SetGroupName (m_tid, groupName);
  return *this;
}
TypeId
TypeId::SetSize (std::size_t size)
{
  NS_LOG_FUNCTION (this << size);
  IidManager::Get ()->SetSize (m_tid, size);
  return *this;
}
TypeId
TypeId::GetParent (void) const
{
  NS_LOG_FUNCTION (this);
  uint16_t parent = IidManager::Get ()->GetParent (m_tid);
  return TypeId (parent);
}
bool
TypeId::HasParent (void) const
{
  NS_LOG_FUNCTION (this);
  uint16_t parent = IidManager::Get ()->GetParent (m_tid);
  return parent != m_tid;
}
bool
TypeId::IsChildOf (TypeId other) const
{
  NS_LOG_FUNCTION (this << other.GetUid ());
  TypeId tmp = *this;
  while (tmp != other && tmp != tmp.GetParent ())
    {
      tmp = tmp.GetParent ();
    }
  return tmp == other && *this != other;
}
std::string
TypeId::GetGroupName (void) const
{
  NS_LOG_FUNCTION (this);
  std::string groupName = IidManager::Get ()->GetGroupName (m_tid);
  return groupName;
}

std::string
TypeId::GetName (void) const
{
  NS_LOG_FUNCTION (this);
  std::string name = IidManager::Get ()->GetName (m_tid);
  return name;
}

TypeId::hash_t
TypeId::GetHash (void) const
{
  hash_t hash = IidManager::Get ()->GetHash (m_tid);
  return hash;
}
std::size_t
TypeId::GetSize (void) const
{
  NS_LOG_FUNCTION (this);
  std::size_t size = IidManager::Get ()->GetSize (m_tid);
  return size;
}

bool
TypeId::HasConstructor (void) const
{
  NS_LOG_FUNCTION (this);
  bool hasConstructor = IidManager::Get ()->HasConstructor (m_tid);
  return hasConstructor;
}

void
TypeId::DoAddConstructor (Callback<ObjectBase *> cb)
{
  NS_LOG_FUNCTION (this << &cb);
  IidManager::Get ()->AddConstructor (m_tid, cb);
}

TypeId
TypeId::AddAttribute (std::string name,
                      std::string help,
                      const AttributeValue &initialValue,
                      Ptr<const AttributeAccessor> accessor,
                      Ptr<const AttributeChecker> checker,
                      SupportLevel supportLevel,
                      const std::string &supportMsg)
{
  NS_LOG_FUNCTION (this << name << help <<
                   &initialValue << accessor << checker <<
                   supportLevel << supportMsg);
  IidManager::Get ()->AddAttribute (m_tid, name, help, ATTR_SGC,
                                    initialValue.Copy (), accessor, checker,
                                    supportLevel, supportMsg);
  return *this;
}

TypeId
TypeId::AddAttribute (std::string name,
                      std::string help,
                      uint32_t flags,
                      const AttributeValue &initialValue,
                      Ptr<const AttributeAccessor> accessor,
                      Ptr<const AttributeChecker> checker,
                      SupportLevel supportLevel,
                      const std::string &supportMsg)
{
  NS_LOG_FUNCTION (this << name << help << flags <<
                   &initialValue << accessor << checker <<
                   supportLevel << supportMsg);
  IidManager::Get ()->AddAttribute (m_tid, name, help, flags,
                                    initialValue.Copy (), accessor, checker,
                                    supportLevel, supportMsg);
  return *this;
}

bool
TypeId::SetAttributeInitialValue (std::size_t i,
                                  Ptr<const AttributeValue> initialValue)
{
  NS_LOG_FUNCTION (this << i << initialValue);
  IidManager::Get ()->SetAttributeInitialValue (m_tid, i, initialValue);
  return true;
}


Callback<ObjectBase *>
TypeId::GetConstructor (void) const
{
  NS_LOG_FUNCTION (this);
  Callback<ObjectBase *>  cb = IidManager::Get ()->GetConstructor (m_tid);
  return cb;
}

bool
TypeId::MustHideFromDocumentation (void) const
{
  NS_LOG_FUNCTION (this);
  bool mustHide = IidManager::Get ()->MustHideFromDocumentation (m_tid);
  return mustHide;
}

std::size_t
TypeId::GetAttributeN (void) const
{
  NS_LOG_FUNCTION (this);
  std::size_t n = IidManager::Get ()->GetAttributeN (m_tid);
  return n;
}
struct TypeId::AttributeInformation
TypeId::GetAttribute (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return IidManager::Get ()->GetAttribute (m_tid, i);
}
std::string
TypeId::GetAttributeFullName (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  struct TypeId::AttributeInformation info = GetAttribute (i);
  return GetName () + "::" + info.name;
}

std::size_t
TypeId::GetTraceSourceN (void) const
{
  NS_LOG_FUNCTION (this);
  return IidManager::Get ()->GetTraceSourceN (m_tid);
}
struct TypeId::TraceSourceInformation
TypeId::GetTraceSource (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return IidManager::Get ()->GetTraceSource (m_tid, i);
}

TypeId
TypeId::AddTraceSource (std::string name,
                        std::string help,
                        Ptr<const TraceSourceAccessor> accessor,
                        std::string callback,
                        SupportLevel supportLevel,
                        const std::string &supportMsg)
{
  NS_LOG_FUNCTION (this << name << help <<
                   accessor << callback <<
                   supportLevel << supportMsg);
  IidManager::Get ()->AddTraceSource (m_tid, name, help,
                                      accessor, callback,
                                      supportLevel, supportMsg);
  return *this;
}

TypeId
TypeId::HideFromDocumentation (void)
{
  NS_LOG_FUNCTION (this);
  IidManager::Get ()->HideFromDocumentation (m_tid);
  return *this;
}

Ptr<const TraceSourceAccessor>
TypeId::LookupTraceSourceByName (std::string name,
                                 struct TraceSourceInformation *info) const
{
  NS_LOG_FUNCTION (this << name);
  TypeId tid;
  TypeId nextTid = *this;
  struct TypeId::TraceSourceInformation tmp;
  do
    {
      tid = nextTid;
      for (std::size_t i = 0; i < tid.GetTraceSourceN (); i++)
        {
          tmp = tid.GetTraceSource (i);
          if (tmp.name == name)
            {
              if (tmp.supportLevel == TypeId::SUPPORTED)
                {
                  *info = tmp;
                  return tmp.accessor;
                }
              else if (tmp.supportLevel == TypeId::DEPRECATED)
                {
                  std::cerr << "TraceSource '" << name << "' is deprecated: "
                            << tmp.supportMsg << std::endl;
                  *info = tmp;
                  return tmp.accessor;
                }
              else if (tmp.supportLevel == TypeId::OBSOLETE)
                {
                  NS_FATAL_ERROR ("TraceSource '" << name <<
                                  "' is obsolete, with no fallback: " <<
                                  tmp.supportMsg);
                }
            }
        }
      nextTid = tid.GetParent ();
    }
  while (nextTid != tid);
  return 0;
}

Ptr<const TraceSourceAccessor>
TypeId::LookupTraceSourceByName (std::string name) const
{
  struct TraceSourceInformation info;
  return LookupTraceSourceByName (name, &info);
}

uint16_t
TypeId::GetUid (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tid;
}
void
TypeId::SetUid (uint16_t uid)
{
  NS_LOG_FUNCTION (this << uid);
  m_tid = uid;
}

/**
 *  \brief Insertion operator for TypeId
 *  \param [in] os the output stream
 *  \param [in] tid the TypeId
 *  \returns the updated output stream.
 */
std::ostream & operator << (std::ostream &os, TypeId tid)
{
  os << tid.GetName ();
  return os;
}
/**
 *  \brief Extraction operator for TypeId
 *  \param [in] is the input stream
 *  \param [out] tid the TypeId value
 *  \returns the updated input stream.
 */
std::istream & operator >> (std::istream &is, TypeId &tid)
{
  std::string tidString;
  is >> tidString;
  bool ok = TypeId::LookupByNameFailSafe (tidString, &tid);
  if (!ok)
    {
      is.setstate (std::ios_base::badbit);
    }
  return is;
}


ATTRIBUTE_HELPER_CPP (TypeId);

bool operator < (TypeId a, TypeId b)
{
  return a.m_tid < b.m_tid;
}

} // namespace ns3
