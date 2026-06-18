/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include <map>
#include "object.h"
#include "log.h"
#include "assert.h"
#include "abort.h"
#include "names.h"
#include "singleton.h"

/**
 * \file
 * \ingroup config
 * ns3::Names, ns3::NamesNode and ns3::NamePriv implementations.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Names");

/**
 * \ingroup config
 *  Node in the naming tree.
 */
class NameNode
{
public:
  /** Default constructor. */
  NameNode ();
  /**
   * Copy constructor.
   *
   * \param [in] nameNode The NameNode to copy from.
   */
  NameNode (const NameNode &nameNode);
  /**
   * Constructor.
   *
   * \param [in] parent The parent NameNode.
   * \param [in] name The name of this NameNode
   * \param [in] object The object corresponding to this NameNode.
   */
  NameNode (NameNode *parent, std::string name, Ptr<Object> object);
  /**
   * Assignment operator.
   *
   * \param [in] rhs The NameNode to copy from.
   * \returns The lhs NameNode.
   */
  NameNode &operator = (const NameNode &rhs);

  /** Destructor. */
  ~NameNode ();

  /** The parent NameNode. */
  NameNode *m_parent;
  /** The name of this NameNode. */
  std::string m_name;
  /** The object corresponding to this NameNode. */
  Ptr<Object> m_object;

  /** Children of this NameNode. */
  std::map<std::string, NameNode *> m_nameMap;
};

NameNode::NameNode ()
  : m_parent (0), m_name (""), m_object (0)
{}

NameNode::NameNode (const NameNode &nameNode)
{
  m_parent = nameNode.m_parent;
  m_name = nameNode.m_name;
  m_object = nameNode.m_object;
  m_nameMap = nameNode.m_nameMap;
}

NameNode &
NameNode::operator = (const NameNode &rhs)
{
  m_parent = rhs.m_parent;
  m_name = rhs.m_name;
  m_object = rhs.m_object;
  m_nameMap = rhs.m_nameMap;
  return *this;
}

NameNode::NameNode (NameNode *parent, std::string name, Ptr<Object> object)
  : m_parent (parent), m_name (name), m_object (object)
{
  NS_LOG_FUNCTION (this << parent << name << object);
}

NameNode::~NameNode ()
{
  NS_LOG_FUNCTION (this);
}

/**
 * \ingroup config
 * The singleton root Names object.
 */
class NamesPriv : public Singleton<NamesPriv>
{
public:
  /** Constructor. */
  NamesPriv ();
  /** Destructor. */
  ~NamesPriv ();

  // Doxygen \copydoc bug: won't copy these docs, so we repeat them.

  /**
   * Internal implementation for Names::Add(std::string,Ptr<Object>)
   *
   * \param [in] name The name of the object you want to associate;
   *             which may be prepended with a path to that object.
   * \param [in] object A smart pointer to the object itself.
   * \return \c true if the object was named successfully.
   */
  bool Add (std::string name, Ptr<Object> object);
  /**
   * Internal implementation for Names::Add(std::string,std::string,Ptr<Object>)
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want this new name to be defined.
   * \param [in] name The name of the object you want to associate.
   * \param [in] object A smart pointer to the object itself.
   * \return \c true if the object was named successfully.
   */
  bool Add (std::string path, std::string name, Ptr<Object> object);
  /**
   * Internal implementation for Names::Add(Ptr<Object>,std::string,Ptr<Object>)
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this new
   *             name to be defined.
   * \param [in] name The name of the object you want to associate.
   * \param [in] object A smart pointer to the object itself.
   * \return \c true if the object was named successfully.
   */
  bool Add (Ptr<Object> context, std::string name, Ptr<Object> object);

  /**
   * Internal implementation for Names::Rename(std::string,std::string)
   *
   * \param [in] oldpath The current path name to the object you want
   *             to change.
   * \param [in] newname The new name of the object you want to change.
   * \return \c true if the object was renamed successfully.
   */
  bool Rename (std::string oldpath, std::string newname);
  /**
   * Internal implementation for
   * Names::Rename(std::string,std::string,std::string)
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want this name change to occur
   *             (cf. directory).
   * \param [in] oldname The currently defined name of the object.
   * \param [in] newname The new name you want the object to have.
   * \return \c true if the object was renamed successfully.
   */
  bool Rename (std::string path, std::string oldname, std::string newname);
  /**
   * Internal implementation for
   * Names::Rename(Ptr<Object>,std::string,std::string)
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this
   *             new name to be defined.
   * \param [in] oldname The current shortname of the object you want
   *             to change.
   * \param [in] newname The new shortname of the object you want
   *             to change.
   * \return \c true if the object was renamed successfully.
   */
  bool Rename (Ptr<Object> context, std::string oldname, std::string newname);

  /**
   * Internal implementation for Names::FindName()
   *
   * \param [in] object A smart pointer to an object for which you want
   *             to find its name.
   * \returns A string containing the name of the object if found,
   *          otherwise the empty string.
   */
  std::string FindName (Ptr<Object> object);
  /**
   * Internal implementation of Names::FindPath()
   *
   * \param [in] object A smart pointer to an object for which you
   *             want to find its fullname.
   * \returns A string containing the name path of the object,
   *          otherwise the empty string.
   */
  std::string FindPath (Ptr<Object> object);

  /**
   * Internal implementation for Names::Clear()
   */
  void Clear (void);

  /**
   * Internal implementation for ns3::Names::Find(std::string)
   *
   * \param [in] path A string containing a name space path used
   *             to locate the object.
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  Ptr<Object> Find (std::string path);
  /**
   * Internal implementation for ns3::Names::Find(std::string,std::string)
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want to look for the specified name.
   * \param [in] name A string containing a name to search for.
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  Ptr<Object> Find (std::string path, std::string name);
  /**
   * Internal implementation for ns3::Names::Find(Ptr<Object>,std::string)
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this
   *             new name to be defined.
   * \param [in] name A string containing a name to search for.
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  Ptr<Object> Find (Ptr<Object> context, std::string name);

private:
  
  /**
   * Check if an object has a name.
   *
   * \param [in] object The object to check.
   * \returns The corresponding NameNode, if it exists.
   */
  NameNode * IsNamed (Ptr<Object> object);
  /**
   * Check if a name already exists as a child of a NameNode.
   *
   * \param [in] node The node to search.
   * \param [in] name The name to search for.
   * \returns \c true if \c name already exists as a child of \c node.
   */
  bool IsDuplicateName (NameNode *node, std::string name);

  /** The root NameNode. */
  NameNode m_root;

  /** Map from object pointers to their NameNodes. */
  std::map<Ptr<Object>, NameNode *> m_objectMap;
};

NamesPriv::NamesPriv ()
{
  NS_LOG_FUNCTION (this);

  m_root.m_parent = 0;
  m_root.m_name = "Names";
  m_root.m_object = 0;
}

NamesPriv::~NamesPriv ()
{
  NS_LOG_FUNCTION (this);
  Clear ();
  m_root.m_name = "";
}

void
NamesPriv::Clear (void)
{
  NS_LOG_FUNCTION (this);
  //
  // Every name is associated with an object in the object map, so freeing the
  // NameNodes in this map will free all of the memory allocated for the NameNodes
  //
  for (std::map<Ptr<Object>, NameNode *>::iterator i = m_objectMap.begin (); i != m_objectMap.end (); ++i)
    {
      delete i->second;
      i->second = 0;
    }

  m_objectMap.clear ();

  m_root.m_parent = 0;
  m_root.m_name = "Names";
  m_root.m_object = 0;
  m_root.m_nameMap.clear ();
}

bool
NamesPriv::Add (std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << name << object);
  //
  // This is the simple, easy to use version of Add, so we want it to be flexible.
  // We don't want to force a user to always type the fully qualified namespace
  // name, so we allow the namespace name to be omitted.  For example, calling
  // Add ("Client/ath0", obj) should result in exactly the same behavior as
  // Add ("/Names/Client/ath0", obj).  Calling Add ("Client", obj) should have
  // the same effect as Add ("Names/Client", obj)
  //
  // The first thing to do, then, is to "canonicalize" the input string to always
  // be a fully qualified name.
  //
  // If we are given a name that begins with "/Names/" we assume that this is a
  // fully qualified path name to the object we want to create.  We split the name
  // into a path string and a final segment (name) and then call the "Real" Add.
  //
  std::string namespaceName = "/Names";
  std::string::size_type offset = name.find (namespaceName);
  if (offset != 0)
    {
      //
      // This must be a name that has the "/Names" namespace prefix omitted.
      // Do some reasonableness checking on the rest of the name.
      //
      offset = name.find ("/");
      if (offset == 0)
        {
          NS_ASSERT_MSG (false, "NamesPriv::Add(): Name begins with '/' but not \"/Names\"");
          return false;
        }

      name = "/Names/" + name;
    }

  //
  // There must now be a fully qualified path in the string.  All fully
  // qualified names begin with "/Names".  We have to split off the final
  // segment which will become the name of the object.  A '/' that
  // separates the path from the final segment had better be there since
  // we just made sure that at least the namespace name was there.
  //
  std::string::size_type i = name.rfind ("/");
  NS_ASSERT_MSG (i != std::string::npos, "NamesPriv::Add(): Internal error.  Can't find '/' in name");

  //
  // The slash we found cannot be the slash at the start of the namespaceName.
  // This would indicate there is no name in the path at all.  It can be
  // any other index.
  //
  NS_ASSERT_MSG (i != 0, "NamesPriv::Add(): Can't find a name in the path string");

  //
  // We now know where the path string starts and ends, and where the
  // name starts and ends.  All we have to do is to call our available
  // function for adding a name under a path string.
  //
  return Add (name.substr (0, i), name.substr (i + 1), object);
}

bool
NamesPriv::Add (std::string path, std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << path << name << object);
  if (path == "/Names")
    {
      return Add (Ptr<Object> (0, false), name, object);
    }
  return Add (Find (path), name, object);
}

bool
NamesPriv::Add (Ptr<Object> context, std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << context << name << object);

  if (IsNamed (object))
    {
      NS_LOG_LOGIC ("Object is already named");
      return false;
    }

  NameNode *node = 0;
  if (context)
    {
      node = IsNamed (context);
      NS_ASSERT_MSG (node, "NamesPriv::Name(): context must point to a previously named node");
    }
  else
    {
      node = &m_root;
    }

  if (IsDuplicateName (node, name))
    {
      NS_LOG_LOGIC ("Name is already taken");
      return false;
    }

  NameNode *newNode = new NameNode (node, name, object);
  node->m_nameMap[name] = newNode;
  m_objectMap[object] = newNode;

  return true;
}

bool
NamesPriv::Rename (std::string oldpath, std::string newname)
{
  NS_LOG_FUNCTION (this << oldpath << newname);
  //
  // This is the simple, easy to use version of Rename, so we want it to be
  // flexible.   We don't want to force a user to always type the fully
  // qualified namespace name, so we allow the namespace name to be omitted.
  // For example, calling Rename ("Client/ath0", "eth0") should result in
  // exactly the same behavior as Rename ("/Names/Client/ath0", "eth0").
  // Calling Rename ("Client", "Router") should have the same effect as
  // Rename ("Names/Client", "Router")
  //
  // The first thing to do, then, is to "canonicalize" the input string to always
  // be a fully qualified path.
  //
  // If we are given a name that begins with "/Names/" we assume that this is a
  // fully qualified path to the object we want to change.  We split the path into
  // path string (cf directory) and a final segment (cf filename) and then call
  // the "Real" Rename.
  //
  std::string namespaceName = "/Names";
  std::string::size_type offset = oldpath.find (namespaceName);
  if (offset != 0)
    {
      //
      // This must be a name that has the "/Names" namespace prefix omitted.
      // Do some reasonableness checking on the rest of the name.
      //
      offset = oldpath.find ("/");
      if (offset == 0)
        {
          NS_ASSERT_MSG (false, "NamesPriv::Add(): Name begins with '/' but not \"/Names\"");
          return false;
        }

      oldpath = "/Names/" + oldpath;
    }

  //
  // There must now be a fully qualified path in the oldpath string.  All
  // fully qualified names begin with "/Names".  We have to split off the final
  // segment which will become the name we want to rename.  A '/' that
  // separates the path from the final segment (name) had better be there since
  // we just made sure that at least the namespace name was there.
  //
  std::string::size_type i = oldpath.rfind ("/");
  NS_ASSERT_MSG (i != std::string::npos, "NamesPriv::Add(): Internal error.  Can't find '/' in name");

  //
  // The slash we found cannot be the slash at the start of the namespaceName.
  // This would indicate there is no name in the path at all.  It can be
  // any other index.
  //
  NS_ASSERT_MSG (i != 0, "NamesPriv::Add(): Can't find a name in the path string");

  //
  // We now know where the path part of the string starts and ends, and where the
  // name part starts and ends.  All we have to do is to call our available
  // function for creating adding a name under a path string.
  //
  return Rename (oldpath.substr (0, i), oldpath.substr (i + 1), newname);
}

bool
NamesPriv::Rename (std::string path, std::string oldname, std::string newname)
{
  NS_LOG_FUNCTION (this << path << oldname << newname);
  if (path == "/Names")
    {
      return Rename (Ptr<Object> (0, false), oldname, newname);
    }
  return Rename (Find (path), oldname, newname);
}

bool
NamesPriv::Rename (Ptr<Object> context, std::string oldname, std::string newname)
{
  NS_LOG_FUNCTION (this << context << oldname << newname);

  NameNode *node = 0;
  if (context)
    {
      node = IsNamed (context);
      NS_ASSERT_MSG (node, "NamesPriv::Name(): context must point to a previously named node");
    }
  else
    {
      node = &m_root;
    }

  if (IsDuplicateName (node, newname))
    {
      NS_LOG_LOGIC ("New name is already taken");
      return false;
    }

  std::map<std::string, NameNode *>::iterator i = node->m_nameMap.find (oldname);
  if (i == node->m_nameMap.end ())
    {
      NS_LOG_LOGIC ("Old name does not exist in name map");
      return false;
    }
  else
    {
      NS_LOG_LOGIC ("Old name exists in name map");

      //
      // The rename process consists of:
      // 1.  Getting the pointer to the name node from the map and remembering it;
      // 2.  Removing the map entry corresponding to oldname from the map;
      // 3.  Changing the name string in the name node;
      // 4.  Adding the name node back in the map under the newname.
      //
      NameNode *changeNode = i->second;
      node->m_nameMap.erase (i);
      changeNode->m_name = newname;
      node->m_nameMap[newname] = changeNode;
      return true;
    }
}

std::string
NamesPriv::FindName (Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << object);

  std::map<Ptr<Object>, NameNode *>::iterator i = m_objectMap.find (object);
  if (i == m_objectMap.end ())
    {
      NS_LOG_LOGIC ("Object does not exist in object map");
      return "";
    }
  else
    {
      NS_LOG_LOGIC ("Object exists in object map");
      return i->second->m_name;
    }
}

std::string
NamesPriv::FindPath (Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << object);

  std::map<Ptr<Object>, NameNode *>::iterator i = m_objectMap.find (object);
  if (i == m_objectMap.end ())
    {
      NS_LOG_LOGIC ("Object does not exist in object map");
      return "";
    }

  NameNode *p = i->second;
  NS_ASSERT_MSG (p, "NamesPriv::FindFullName(): Internal error: Invalid NameNode pointer from map");

  std::string path;

  do
    {
      path = "/" + p->m_name + path;
      NS_LOG_LOGIC ("path is " << path);
    }
  while ((p = p->m_parent) != 0);

  return path;
}


Ptr<Object>
NamesPriv::Find (std::string path)
{
  //
  // This is hooked in from simple, easy to use version of Find, so we want it
  // to be flexible.
  //
  // If we are provided a path that doesn't begin with "/Names", we assume
  // that the caller has simply given us a path starting with a name that
  // is in the root namespace.  This allows people to omit the "/Names" prefix.
  // and simply do a Find ("Client/eth0") instead of having to always do a
  // Find ("/Names/Client/eth0");
  //
  // So, if we are given a name that begins with "/Names/" the upshot is that we
  // just remove that prefix and treat the rest of the string as starting with a
  // name in the root namespace.
  //

  NS_LOG_FUNCTION (this << path);
  std::string namespaceName = "/Names/";
  std::string remaining;

  std::string::size_type offset = path.find (namespaceName);
  if (offset == 0)
    {
      NS_LOG_LOGIC (path << " is a fully qualified name");
      remaining = path.substr (namespaceName.size ());
    }
  else
    {
      NS_LOG_LOGIC (path << " begins with a relative name");
      remaining = path;
    }

  NameNode *node = &m_root;

  //
  // The string <remaining> is now composed entirely of path segments in
  // the /Names name space and we have eaten the leading slash. e.g.,
  // remaining = "ClientNode/eth0"
  //
  // The start of the search is always at the root of the name space.
  //
  for (;;)
    {
      NS_LOG_LOGIC ("Looking for the object of name " << remaining);
      offset = remaining.find ("/");
      if (offset == std::string::npos)
        {
          //
          // There are no remaining slashes so this is the last segment of the
          // specified name.  We're done when we find it
          //
          std::map<std::string, NameNode *>::iterator i = node->m_nameMap.find (remaining);
          if (i == node->m_nameMap.end ())
            {
              NS_LOG_LOGIC ("Name does not exist in name map");
              return 0;
            }
          else
            {
              NS_LOG_LOGIC ("Name parsed, found object");
              return i->second->m_object;
            }
        }
      else
        {
          //
          // There are more slashes so this is an intermediate segment of the
          // specified name.  We need to "recurse" when we find this segment.
          //
          offset = remaining.find ("/");
          std::string segment = remaining.substr (0, offset);

          std::map<std::string, NameNode *>::iterator i = node->m_nameMap.find (segment);
          if (i == node->m_nameMap.end ())
            {
              NS_LOG_LOGIC ("Name does not exist in name map");
              return 0;
            }
          else
            {
              node = i->second;
              remaining = remaining.substr (offset + 1);
              NS_LOG_LOGIC ("Intermediate segment parsed");
              continue;
            }
        }
    }

  NS_ASSERT_MSG (node, "NamesPriv::Find(): Internal error:  this can't happen");
  return 0;
}

Ptr<Object>
NamesPriv::Find (std::string path, std::string name)
{
  NS_LOG_FUNCTION (this << path << name);

  if (path == "/Names")
    {
      return Find (Ptr<Object> (0, false), name);
    }
  return Find (Find (path), name);
}

Ptr<Object>
NamesPriv::Find (Ptr<Object> context, std::string name)
{
  NS_LOG_FUNCTION (this << context << name);

  NameNode *node = 0;

  if (context == 0)
    {
      NS_LOG_LOGIC ("Zero context implies root NameNode");
      node = &m_root;
    }
  else
    {
      node = IsNamed (context);
      if (node == 0)
        {
          NS_LOG_LOGIC ("Context does not point to a previously named node");
          return 0;
        }
    }

  std::map<std::string, NameNode *>::iterator i = node->m_nameMap.find (name);
  if (i == node->m_nameMap.end ())
    {
      NS_LOG_LOGIC ("Name does not exist in name map");
      return 0;
    }
  else
    {
      NS_LOG_LOGIC ("Name exists in name map");
      return i->second->m_object;
    }
}

NameNode *
NamesPriv::IsNamed (Ptr<Object> object)
{
  NS_LOG_FUNCTION (this << object);

  std::map<Ptr<Object>, NameNode *>::iterator i = m_objectMap.find (object);
  if (i == m_objectMap.end ())
    {
      NS_LOG_LOGIC ("Object does not exist in object map, returning NameNode 0");
      return 0;
    }
  else
    {
      NS_LOG_LOGIC ("Object exists in object map, returning NameNode " << &i->second);
      return i->second;
    }
}

bool
NamesPriv::IsDuplicateName (NameNode *node, std::string name)
{
  NS_LOG_FUNCTION (this << node << name);

  std::map<std::string, NameNode *>::iterator i = node->m_nameMap.find (name);
  if (i == node->m_nameMap.end ())
    {
      NS_LOG_LOGIC ("Name does not exist in name map");
      return false;
    }
  else
    {
      NS_LOG_LOGIC ("Name exists in name map");
      return true;
    }
}

void
Names::Add (std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (name << object);
  bool result = NamesPriv::Get ()->Add (name, object);
  NS_ABORT_MSG_UNLESS (result, "Names::Add(): Error adding name " << name);
}

void
Names::Rename (std::string oldpath, std::string newname)
{
  NS_LOG_FUNCTION (oldpath << newname);
  bool result = NamesPriv::Get ()->Rename (oldpath, newname);
  NS_ABORT_MSG_UNLESS (result, "Names::Rename(): Error renaming " << oldpath << " to " << newname);
}

void
Names::Add (std::string path, std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (path << name << object);
  bool result = NamesPriv::Get ()->Add (path, name, object);
  NS_ABORT_MSG_UNLESS (result, "Names::Add(): Error adding " << path << " " << name);
}

void
Names::Rename (std::string path, std::string oldname, std::string newname)
{
  NS_LOG_FUNCTION (path << oldname << newname);
  bool result = NamesPriv::Get ()->Rename (path, oldname, newname);
  NS_ABORT_MSG_UNLESS (result, "Names::Rename (): Error renaming " << path << " " << oldname << " to " << newname);
}

void
Names::Add (Ptr<Object> context, std::string name, Ptr<Object> object)
{
  NS_LOG_FUNCTION (context << name << object);
  bool result = NamesPriv::Get ()->Add (context, name, object);
  NS_ABORT_MSG_UNLESS (result, "Names::Add(): Error adding name " << name << " under context " << &context);
}

void
Names::Rename (Ptr<Object> context, std::string oldname, std::string newname)
{
  NS_LOG_FUNCTION (context << oldname << newname);
  bool result = NamesPriv::Get ()->Rename (context, oldname, newname);
  NS_ABORT_MSG_UNLESS (result, "Names::Rename (): Error renaming " << oldname << " to " << newname << " under context " <<
                       &context);
}

std::string
Names::FindName (Ptr<Object> object)
{
  NS_LOG_FUNCTION (object);
  return NamesPriv::Get ()->FindName (object);
}

std::string
Names::FindPath (Ptr<Object> object)
{
  NS_LOG_FUNCTION (object);
  return NamesPriv::Get ()->FindPath (object);
}

void
Names::Clear (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return NamesPriv::Get ()->Clear ();
}

Ptr<Object>
Names::FindInternal (std::string name)
{
  NS_LOG_FUNCTION (name);
  return NamesPriv::Get ()->Find (name);
}

Ptr<Object>
Names::FindInternal (std::string path, std::string name)
{
  NS_LOG_FUNCTION (path << name);
  return NamesPriv::Get ()->Find (path, name);
}

Ptr<Object>
Names::FindInternal (Ptr<Object> context, std::string name)
{
  NS_LOG_FUNCTION (context << name);
  return NamesPriv::Get ()->Find (context, name);
}

} // namespace ns3
