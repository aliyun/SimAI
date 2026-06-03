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
#include "config.h"
#include "singleton.h"
#include "object.h"
#include "global-value.h"
#include "object-ptr-container.h"
#include "names.h"
#include "pointer.h"
#include "log.h"

#include <sstream>

/**
 * \file
 * \ingroup config
 * ns3::Config implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Config");

namespace Config {

MatchContainer::MatchContainer ()
{
  NS_LOG_FUNCTION (this);
}
MatchContainer::MatchContainer (const std::vector<Ptr<Object> > &objects,
                                const std::vector<std::string> &contexts,
                                std::string path)
  : m_objects (objects),
  m_contexts (contexts),
  m_path (path)
{
  NS_LOG_FUNCTION (this << &objects << &contexts << path);
}
MatchContainer::Iterator
MatchContainer::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.begin ();
}
MatchContainer::Iterator
MatchContainer::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.end ();
}
std::size_t
MatchContainer::GetN (void) const
{
  NS_LOG_FUNCTION (this);
  return m_objects.size ();
}
Ptr<Object>
MatchContainer::Get (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_objects[i];
}
std::string
MatchContainer::GetMatchedPath (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_contexts[i];
}
std::string
MatchContainer::GetPath (void) const
{
  NS_LOG_FUNCTION (this);
  return m_path;
}

void
MatchContainer::Set (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (this << name << &value);
  for (Iterator tmp = Begin (); tmp != End (); ++tmp)
    {
      Ptr<Object> object = *tmp;
      // Let ObjectBase::SetAttribute raise any errors
      object->SetAttribute (name, value);
    }
}
bool
MatchContainer::SetFailSafe (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (this << name << &value);
  bool ok = false;
  for (Iterator tmp = Begin (); tmp != End (); ++tmp)
    {
      Ptr<Object> object = *tmp;
      ok |= object->SetAttributeFailSafe (name, value);
    }
  return ok;
}
void
MatchContainer::Connect (std::string name, const CallbackBase &cb)
{
  if (!ConnectFailSafe (name, cb))
    {
      NS_FATAL_ERROR ("Cound not connect callback to " << name);
    }
}
bool
MatchContainer::ConnectFailSafe (std::string name, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << name << &cb);
  NS_ASSERT (m_objects.size () == m_contexts.size ());
  bool ok = false;
  for (uint32_t i = 0; i < m_objects.size (); ++i)
    {
      Ptr<Object> object = m_objects[i];
      std::string ctx = m_contexts[i] + name;
      ok |= object->TraceConnect (name, ctx, cb);
    }
  return ok;
}
void
MatchContainer::ConnectWithoutContext (std::string name, const CallbackBase &cb)
{
  if (!ConnectWithoutContextFailSafe (name, cb))
    {
      NS_FATAL_ERROR ("Could not connect callback to " << name);
    }
}
bool
MatchContainer::ConnectWithoutContextFailSafe (std::string name, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << name << &cb);
  bool ok = false;
  for (Iterator tmp = Begin (); tmp != End (); ++tmp)
    {
      Ptr<Object> object = *tmp;
      ok |= object->TraceConnectWithoutContext (name, cb);
    }
  return ok;
}
void
MatchContainer::Disconnect (std::string name, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << name << &cb);
  NS_ASSERT (m_objects.size () == m_contexts.size ());
  for (uint32_t i = 0; i < m_objects.size (); ++i)
    {
      Ptr<Object> object = m_objects[i];
      std::string ctx = m_contexts[i] + name;
      object->TraceDisconnect (name, ctx, cb);
    }
}
void
MatchContainer::DisconnectWithoutContext (std::string name, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << name << &cb);
  for (Iterator tmp = Begin (); tmp != End (); ++tmp)
    {
      Ptr<Object> object = *tmp;
      object->TraceDisconnectWithoutContext (name, cb);
    }
}


/**
 * \ingroup config-impl
 * Helper to test if an array entry matches a config path specification.
 */
class ArrayMatcher
{
public:
  /**
   * Construct from a Config path specification.
   *
   * \param [in] element The Config path specification.
   */
  ArrayMatcher (std::string element);
  /**
   * Test if a specific index matches the Config Path.
   *
   * \param [in] i The index.
   * \returns \c true if the index matches the Config Path.
   */
  bool Matches (std::size_t i) const;

private:
  /**
   * Convert a string to an \c uint32_t.
   *
   * \param [in] str The string.
   * \param [in] value The location to store the \c uint32_t.
   * \returns \c true if the string could be converted.
   */
  bool StringToUint32 (std::string str, uint32_t *value) const;
  /** The Config path element. */
  std::string m_element;

};  // class ArrayMatcher


ArrayMatcher::ArrayMatcher (std::string element)
  : m_element (element)
{
  NS_LOG_FUNCTION (this << element);
}
bool
ArrayMatcher::Matches (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  if (m_element == "*")
    {
      NS_LOG_DEBUG ("Array " << i << " matches *");
      return true;
    }
  std::string::size_type tmp;
  tmp = m_element.find ("|");
  if (tmp != std::string::npos)
    {
      std::string left = m_element.substr (0, tmp - 0);
      std::string right = m_element.substr (tmp + 1, m_element.size () - (tmp + 1));
      ArrayMatcher matcher = ArrayMatcher (left);
      if (matcher.Matches (i))
        {
          NS_LOG_DEBUG ("Array " << i << " matches " << left);
          return true;
        }
      matcher = ArrayMatcher (right);
      if (matcher.Matches (i))
        {
          NS_LOG_DEBUG ("Array " << i << " matches " << right);
          return true;
        }
      NS_LOG_DEBUG ("Array " << i << " does not match " << m_element);
      return false;
    }
  std::string::size_type leftBracket = m_element.find ("[");
  std::string::size_type rightBracket = m_element.find ("]");
  std::string::size_type dash = m_element.find ("-");
  if (leftBracket == 0 && rightBracket == m_element.size () - 1
      && dash > leftBracket && dash < rightBracket)
    {
      std::string lowerBound = m_element.substr (leftBracket + 1, dash - (leftBracket + 1));
      std::string upperBound = m_element.substr (dash + 1, rightBracket - (dash + 1));
      uint32_t min;
      uint32_t max;
      if (StringToUint32 (lowerBound, &min)
          && StringToUint32 (upperBound, &max)
          && i >= min && i <= max)
        {
          NS_LOG_DEBUG ("Array " << i << " matches " << m_element);
          return true;
        }
      else
        {
          NS_LOG_DEBUG ("Array " << i << " does not " << m_element);
          return false;
        }
    }
  uint32_t value;
  if (StringToUint32 (m_element, &value)
      && i == value)
    {
      NS_LOG_DEBUG ("Array " << i << " matches " << m_element);
      return true;
    }
  NS_LOG_DEBUG ("Array " << i << " does not match " << m_element);
  return false;
}

bool
ArrayMatcher::StringToUint32 (std::string str, uint32_t *value) const
{
  NS_LOG_FUNCTION (this << str << value);
  std::istringstream iss;
  iss.str (str);
  iss >> (*value);
  return !iss.bad () && !iss.fail ();
}

/**
 * \ingroup config-impl
 * Abstract class to parse Config paths into object references.
 */
class Resolver
{
public:
  /**
   * Construct from a base Config path.
   *
   * \param [in] path The Config path.
   */
  Resolver (std::string path);
  /** Destructor. */
  virtual ~Resolver ();

  /**
   * Parse the stored Config path into an object reference,
   * beginning at the indicated root object.
   *
   * \param [in] root The object corresponding to the current position in
   *                  in the Config path.
   */
  void Resolve (Ptr<Object> root);

private:
  /** Ensure the Config path starts and ends with a '/'. */
  void Canonicalize (void);
  /**
   * Parse the next element in the Config path.
   *
   * \param [in] path The remaining portion of the Config path.
   * \param [in] root The object corresponding to the current position
   *                  in the Config path.
   */
  void DoResolve (std::string path, Ptr<Object> root);
  /**
   * Parse an index on the Config path.
   *
   * \param [in] path The remaining Config path.
   * \param [in,out] vector The resulting list of matching objects.
   */
  void DoArrayResolve (std::string path, const ObjectPtrContainerValue &vector);
  /**
   * Handle one object found on the path.
   *
   * \param [in] object The current object on the Config path.
   */
  void DoResolveOne (Ptr<Object> object);
  /**
   * Get the current Config path.
   *
   * \returns The current Config path.
   */
  std::string GetResolvedPath (void) const;
  /**
   * Handle one found object.
   *
   * \param [in] object The found object.
   * \param [in] path The matching Config path context.
   */
  virtual void DoOne (Ptr<Object> object, std::string path) = 0;

  /** Current list of path tokens. */
  std::vector<std::string> m_workStack;
  /** The Config path. */
  std::string m_path;

};  // class Resolver

Resolver::Resolver (std::string path)
  : m_path (path)
{
  NS_LOG_FUNCTION (this << path);
  Canonicalize ();
}
Resolver::~Resolver ()
{
  NS_LOG_FUNCTION (this);
}
void
Resolver::Canonicalize (void)
{
  NS_LOG_FUNCTION (this);

  // ensure that we start and end with a '/'
  std::string::size_type tmp = m_path.find ("/");
  if (tmp != 0)
    {
      // no slash at start
      m_path = "/" + m_path;
    }
  tmp = m_path.find_last_of ("/");
  if (tmp != (m_path.size () - 1))
    {
      // no slash at end
      m_path = m_path + "/";
    }
}

void
Resolver::Resolve (Ptr<Object> root)
{
  NS_LOG_FUNCTION (this << root);

  DoResolve (m_path, root);
}

std::string
Resolver::GetResolvedPath (void) const
{
  NS_LOG_FUNCTION (this);

  std::string fullPath = "/";
  for (std::vector<std::string>::const_iterator i = m_workStack.begin (); i != m_workStack.end (); i++)
    {
      fullPath += *i + "/";
    }
  return fullPath;
}

void
Resolver::DoResolveOne (Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << object);

  NS_LOG_DEBUG ("resolved=" << GetResolvedPath ());
  DoOne (object, GetResolvedPath ());
}

void
Resolver::DoResolve (std::string path, Ptr<Object> root)
{
  NS_LOG_FUNCTION (this << path << root);
  NS_ASSERT ((path.find ("/")) == 0);
  std::string::size_type next = path.find ("/", 1);

  if (next == std::string::npos)
    {
      //
      // If root is zero, we're beginning to see if we can use the object name
      // service to resolve this path.  It is impossible to have a object name
      // associated with the root of the object name service since that root
      // is not an object.  This path must be referring to something in another
      // namespace and it will have been found already since the name service
      // is always consulted last.
      //
      if (root)
        {
          DoResolveOne (root);
        }
      return;
    }
  std::string item = path.substr (1, next - 1);
  std::string pathLeft = path.substr (next, path.size () - next);

  //
  // If root is zero, we're beginning to see if we can use the object name
  // service to resolve this path.  In this case, we must see the name space
  // "/Names" on the front of this path.  There is no object associated with
  // the root of the "/Names" namespace, so we just ignore it and move on to
  // the next segment.
  //
  if (root == 0)
    {
      std::string::size_type offset = path.find ("/Names");
      if (offset == 0)
        {
          m_workStack.push_back (item);
          DoResolve (pathLeft, root);
          m_workStack.pop_back ();
          return;
        }
    }

  //
  // We have an item (possibly a segment of a namespace path.  Check to see if
  // we can determine that this segment refers to a named object.  If root is
  // zero, this means to look in the root of the "/Names" name space, otherwise
  // it refers to a name space context (level).
  //
  Ptr<Object> namedObject = Names::Find<Object> (root, item);
  if (namedObject)
    {
      NS_LOG_DEBUG ("Name system resolved item = " << item << " to " << namedObject);
      m_workStack.push_back (item);
      DoResolve (pathLeft, namedObject);
      m_workStack.pop_back ();
      return;
    }

  //
  // We're done with the object name service hooks, so proceed down the path
  // of types and attributes; but only if root is nonzero.  If root is zero
  // and we find ourselves here, we are trying to check in the namespace for
  // a path that is not in the "/Names" namespace.  We will have previously
  // found any matches, so we just bail out.
  //
  if (root == 0)
    {
      return;
    }
  std::string::size_type dollarPos = item.find ("$");
  if (dollarPos == 0)
    {
      // This is a call to GetObject
      std::string tidString = item.substr (1, item.size () - 1);
      NS_LOG_DEBUG ("GetObject=" << tidString << " on path=" << GetResolvedPath ());
      TypeId tid = TypeId::LookupByName (tidString);
      Ptr<Object> object = root->GetObject<Object> (tid);
      if (object == 0)
        {
          NS_LOG_DEBUG ("GetObject (" << tidString << ") failed on path=" << GetResolvedPath ());
          return;
        }
      m_workStack.push_back (item);
      DoResolve (pathLeft, object);
      m_workStack.pop_back ();
    }
  else
    {
      // this is a normal attribute.
      TypeId tid;
      TypeId nextTid = root->GetInstanceTypeId ();
      bool foundMatch = false;

      do
        {
          tid = nextTid;

          for (uint32_t i = 0; i < tid.GetAttributeN (); i++)
            {
              struct TypeId::AttributeInformation info;
              info = tid.GetAttribute (i);
              if (info.name != item && item != "*")
                {
                  continue;
                }
              // attempt to cast to a pointer checker.
              const PointerChecker *pChecker = dynamic_cast<const PointerChecker *> (PeekPointer (info.checker));
              if (pChecker != 0)
                {
                  NS_LOG_DEBUG ("GetAttribute(ptr)=" << info.name << " on path=" << GetResolvedPath ());
                  PointerValue pValue;
                  root->GetAttribute (info.name, pValue);
                  Ptr<Object> object = pValue.Get<Object> ();
                  if (object == 0)
                    {
                      NS_LOG_ERROR ("Requested object name=\"" << item <<
                                    "\" exists on path=\"" << GetResolvedPath () << "\""
                                    " but is null.");
                      continue;
                    }
                  foundMatch = true;
                  m_workStack.push_back (info.name);
                  DoResolve (pathLeft, object);
                  m_workStack.pop_back ();
                }
              // attempt to cast to an object vector.
              const ObjectPtrContainerChecker *vectorChecker =
                dynamic_cast<const ObjectPtrContainerChecker *> (PeekPointer (info.checker));
              if (vectorChecker != 0)
                {
                  NS_LOG_DEBUG ("GetAttribute(vector)=" << info.name << " on path=" << GetResolvedPath () << pathLeft);
                  foundMatch = true;
                  ObjectPtrContainerValue vector;
                  root->GetAttribute (info.name, vector);
                  m_workStack.push_back (info.name);
                  DoArrayResolve (pathLeft, vector);
                  m_workStack.pop_back ();
                }
              // this could be anything else and we don't know what to do with it.
              // So, we just ignore it.
            }

          nextTid = tid.GetParent ();
        }
      while (nextTid != tid);

      if (!foundMatch)
        {
          NS_LOG_DEBUG ("Requested item=" << item << " does not exist on path=" << GetResolvedPath ());
          return;
        }
    }
}

void
Resolver::DoArrayResolve (std::string path, const ObjectPtrContainerValue &container)
{
  NS_LOG_FUNCTION (this << path << &container);
  NS_ASSERT (path != "");
  NS_ASSERT ((path.find ("/")) == 0);
  std::string::size_type next = path.find ("/", 1);
  if (next == std::string::npos)
    {
      return;
    }
  std::string item = path.substr (1, next - 1);
  std::string pathLeft = path.substr (next, path.size () - next);

  ArrayMatcher matcher = ArrayMatcher (item);
  ObjectPtrContainerValue::Iterator it;
  for (it = container.Begin (); it != container.End (); ++it)
    {
      if (matcher.Matches ((*it).first))
        {
          std::ostringstream oss;
          oss << (*it).first;
          m_workStack.push_back (oss.str ());
          DoResolve (pathLeft, (*it).second);
          m_workStack.pop_back ();
        }
    }
}

/**
 * \ingroup config-impl
 * Config system implementation class.
 */
class ConfigImpl : public Singleton<ConfigImpl>
{
public:
  // Keep Set and SetFailSafe since their errors are triggered
  // by the underlying ObjecBase functions.
  /** \copydoc ns3::Config::Set() */
  void Set (std::string path, const AttributeValue &value);
  /** \copydoc ns3::Config::SetFailSafe() */
  bool SetFailSafe (std::string path, const AttributeValue &value);
  /** \copydoc ns3::Config::ConnectWithoutContextFailSafe() */
  bool ConnectWithoutContextFailSafe (std::string path, const CallbackBase &cb);
  /** \copydoc ns3::Config::ConnectFailSafe() */
  bool ConnectFailSafe (std::string path, const CallbackBase &cb);
  /** \copydoc ns3::Config::DisconnectWithoutContext() */
  void DisconnectWithoutContext (std::string path, const CallbackBase &cb);
  /** \copydoc ns3::Config::Disconnect() */
  void Disconnect (std::string path, const CallbackBase &cb);
  /** \copydoc ns3::Config::LookupMatches() */
  MatchContainer LookupMatches (std::string path);

  /** \copydoc ns3::Config::RegisterRootNamespaceObject() */
  void RegisterRootNamespaceObject (Ptr<Object> obj);
  /** \copydoc ns3::Config::UnregisterRootNamespaceObject() */
  void UnregisterRootNamespaceObject (Ptr<Object> obj);

  /** \copydoc ns3::Config::GetRootNamespaceObjectN() */
  std::size_t GetRootNamespaceObjectN (void) const;
  /** \copydoc ns3::Config::GetRootNamespaceObject() */
  Ptr<Object> GetRootNamespaceObject (std::size_t i) const;

private:
  /**
   * Break a Config path into the leading path and the last leaf token.
   * \param [in] path The Config path.
   * \param [in,out] root The leading part of the \pname{path},
   *   up to the final slash.
   * \param [in,out] leaf The trailing part of the \pname{path}.
   */
  void ParsePath (std::string path, std::string *root, std::string *leaf) const;

  /** Container type to hold the root Config path tokens. */
  typedef std::vector<Ptr<Object> > Roots;

  /** The list of Config path roots. */
  Roots m_roots;

};  // class ConfigImpl

void
ConfigImpl::ParsePath (std::string path, std::string *root, std::string *leaf) const
{
  NS_LOG_FUNCTION (this << path << root << leaf);

  std::string::size_type slash = path.find_last_of ("/");
  NS_ASSERT (slash != std::string::npos);
  *root = path.substr (0, slash);
  *leaf = path.substr (slash + 1, path.size () - (slash + 1));
  NS_LOG_FUNCTION (path << *root << *leaf);
}

void
ConfigImpl::Set (std::string path, const AttributeValue &value)
{
  NS_LOG_FUNCTION (this << path << &value);

  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  container.Set (leaf, value);
}
bool
ConfigImpl::SetFailSafe (std::string path, const AttributeValue &value)
{
  NS_LOG_FUNCTION (this << path << &value);

  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  return container.SetFailSafe (leaf, value);
}
bool
ConfigImpl::ConnectWithoutContextFailSafe (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << path << &cb);
  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  return container.ConnectWithoutContextFailSafe (leaf, cb);
}
void
ConfigImpl::DisconnectWithoutContext (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << path << &cb);
  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  if (container.GetN () == 0)
    {
      std::size_t lastFwdSlash = root.rfind ("/");
      NS_LOG_WARN ("Failed to disconnect " << leaf
                                           << ", the Requested object name = " << root.substr (lastFwdSlash + 1)
                                           << " does not exits on path " << root.substr (0, lastFwdSlash));
    }
  container.DisconnectWithoutContext (leaf, cb);
}
bool
ConfigImpl::ConnectFailSafe (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << path << &cb);

  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  return container.ConnectFailSafe (leaf, cb);
}
void
ConfigImpl::Disconnect (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (this << path << &cb);

  std::string root, leaf;
  ParsePath (path, &root, &leaf);
  MatchContainer container = LookupMatches (root);
  if (container.GetN () == 0)
    {
      std::size_t lastFwdSlash = root.rfind ("/");
      NS_LOG_WARN ("Failed to disconnect " << leaf
                                           << ", the Requested object name = " << root.substr (lastFwdSlash + 1)
                                           << " does not exits on path " << root.substr (0, lastFwdSlash));
    }
  container.Disconnect (leaf, cb);
}

MatchContainer
ConfigImpl::LookupMatches (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  class LookupMatchesResolver : public Resolver
  {
public:
    LookupMatchesResolver (std::string path)
      : Resolver (path)
    {
    }
    virtual void DoOne (Ptr<Object> object, std::string path)
    {
      m_objects.push_back (object);
      m_contexts.push_back (path);
    }
    std::vector<Ptr<Object> > m_objects;
    std::vector<std::string> m_contexts;
  } resolver = LookupMatchesResolver (path);
  for (Roots::const_iterator i = m_roots.begin (); i != m_roots.end (); i++)
    {
      resolver.Resolve (*i);
    }

  //
  // See if we can do something with the object name service.  Starting with
  // the root pointer zeroed indicates to the resolver that it should start
  // looking at the root of the "/Names" namespace during this go.
  //
  resolver.Resolve (0);

  return MatchContainer (resolver.m_objects, resolver.m_contexts, path);
}

void
ConfigImpl::RegisterRootNamespaceObject (Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << obj);
  m_roots.push_back (obj);
}

void
ConfigImpl::UnregisterRootNamespaceObject (Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << obj);

  for (std::vector<Ptr<Object> >::iterator i = m_roots.begin (); i != m_roots.end (); i++)
    {
      if (*i == obj)
        {
          m_roots.erase (i);
          return;
        }
    }
}

std::size_t
ConfigImpl::GetRootNamespaceObjectN (void) const
{
  NS_LOG_FUNCTION (this);
  return m_roots.size ();
}
Ptr<Object>
ConfigImpl::GetRootNamespaceObject (std::size_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_roots[i];
}


void Reset (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  // First, let's reset the initial value of every attribute
  for (uint16_t i = 0; i < TypeId::GetRegisteredN (); i++)
    {
      TypeId tid = TypeId::GetRegistered (i);
      for (uint32_t j = 0; j < tid.GetAttributeN (); j++)
        {
          struct TypeId::AttributeInformation info = tid.GetAttribute (j);
          tid.SetAttributeInitialValue (j, info.originalInitialValue);
        }
    }
  // now, let's reset the initial value of every global value.
  for (GlobalValue::Iterator i = GlobalValue::Begin (); i != GlobalValue::End (); ++i)
    {
      (*i)->ResetInitialValue ();
    }
}
void Set (std::string path, const AttributeValue &value)
{
  NS_LOG_FUNCTION (path << &value);
  ConfigImpl::Get ()->Set (path, value);
}
bool SetFailSafe (std::string path, const AttributeValue &value)
{
  NS_LOG_FUNCTION (path << &value);
  return ConfigImpl::Get ()->SetFailSafe (path, value);
}
void SetDefault (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);
  if (!SetDefaultFailSafe (name, value))
    {
      NS_FATAL_ERROR ("Could not set default value for " << name);
    }
}
bool SetDefaultFailSafe (std::string fullName, const AttributeValue &value)
{
  NS_LOG_FUNCTION (fullName << &value);
  std::string::size_type pos = fullName.rfind ("::");
  if (pos == std::string::npos)
    {
      return false;
    }
  std::string tidName = fullName.substr (0, pos);
  std::string paramName = fullName.substr (pos + 2, fullName.size () - (pos + 2));
  TypeId tid;
  bool ok = TypeId::LookupByNameFailSafe (tidName, &tid);
  if (!ok)
    {
      return false;
    }
  struct TypeId::AttributeInformation info;
  tid.LookupAttributeByName(paramName, &info);
  for (uint32_t j = 0; j < tid.GetAttributeN (); j++)
    {
      struct TypeId::AttributeInformation tmp = tid.GetAttribute (j);
      if (tmp.name == paramName)
        {
          Ptr<AttributeValue> v = tmp.checker->CreateValidValue (value);
          if (v == 0)
            {
              return false;
            }
          tid.SetAttributeInitialValue (j, v);
          return true;
        }
    }
  return false;
}
void SetGlobal (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);
  GlobalValue::Bind (name, value);
}
bool SetGlobalFailSafe (std::string name, const AttributeValue &value)
{
  NS_LOG_FUNCTION (name << &value);
  return GlobalValue::BindFailSafe (name, value);
}
void ConnectWithoutContext (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  if (!ConnectWithoutContextFailSafe (path, cb))
    {
      NS_FATAL_ERROR ("Could not connect callback to " << path);
    }
}
bool ConnectWithoutContextFailSafe (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  return ConfigImpl::Get ()->ConnectWithoutContextFailSafe (path, cb);
}
void DisconnectWithoutContext (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  ConfigImpl::Get ()->DisconnectWithoutContext (path, cb);
}
void
Connect (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  if (!ConnectFailSafe (path, cb))
    {
      NS_FATAL_ERROR ("Could not connect callback to " << path);
    }
}
bool
ConnectFailSafe (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  return ConfigImpl::Get ()->ConnectFailSafe (path, cb);
}
void
Disconnect (std::string path, const CallbackBase &cb)
{
  NS_LOG_FUNCTION (path << &cb);
  ConfigImpl::Get ()->Disconnect (path, cb);
}
MatchContainer LookupMatches (std::string path)
{
  NS_LOG_FUNCTION (path);
  return ConfigImpl::Get ()->LookupMatches (path);
}

void RegisterRootNamespaceObject (Ptr<Object> obj)
{
  NS_LOG_FUNCTION (obj);
  ConfigImpl::Get ()->RegisterRootNamespaceObject (obj);
}

void UnregisterRootNamespaceObject (Ptr<Object> obj)
{
  NS_LOG_FUNCTION (obj);
  ConfigImpl::Get ()->UnregisterRootNamespaceObject (obj);
}

std::size_t GetRootNamespaceObjectN (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return ConfigImpl::Get ()->GetRootNamespaceObjectN ();
}

Ptr<Object> GetRootNamespaceObject (uint32_t i)
{
  NS_LOG_FUNCTION (i);
  return ConfigImpl::Get ()->GetRootNamespaceObject (i);
}

} // namespace Config

} // namespace ns3
