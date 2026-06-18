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

#ifndef OBJECT_NAMES_H
#define OBJECT_NAMES_H

#include "ptr.h"
#include "object.h"

/**
 * \file
 * \ingroup config
 * Declaration of class ns3::Names.
 */

namespace ns3 {

/**
 * \ingroup config
 * \brief A directory of name and Ptr<Object> associations that allows
 * us to give any ns3 Object a name.
 */
class Names
{
public:

  /**
   * \brief Add the association between the string "name" and the
   * Ptr<Object> obj.
   *
   * The name may begin either with "/Names" to explicitly call out
   * the fact that the name provided is installed under the root of
   * the name space, or it may begin with the name of the first object
   * in the path.  For example, Names::Add ("/Names/client", obj) and
   * Names::Add ("client", obj) accomplish exactly the same thing.  A
   * name at a given level in the name space path must be unique.  In
   * the case of the example above, it would be illegal to try and
   * associate a different object with the same name: "client" at the
   * same level ("/Names") in the path.
   *
   * As well as specifying a name at the root of the "/Names"
   * namespace, the name parameter can contain a path that fully
   * qualifies the name to be added.  For example, if you previously
   * have named an object "client" in the root namespace as above, you
   * could name an object "under" that name by making a call like
   * Names::Add ("/Names/client/eth0", obj).  This will define the
   * name "eth0" and make it reachable using the path specified.  Note
   * that Names::Add ("client/eth0", obj) would accomplish exactly the
   * same thing.
   *
   * Duplicate names are not allowed at the same level in a path,
   * however you may associate similar names with different paths.
   * For example, if you define "/Names/Client", you may not define
   * another "/Names/Client" just as you may not have two files with
   * the same name in a classical filesystem.  However, you may have
   * "/Names/Client/eth0" and "/Names/Server/eth0" defined at the same
   * time just as you might have different files of the same name
   * under different directories.
   *
   * \param [in] name The name of the object you want to associate; which may be
   *             prepended with a path to that object.
   * \param [in] object A smart pointer to the object itself.
   */
  static void Add (std::string name, Ptr<Object> object);

  /**
   * \brief An intermediate form of Names::Add allowing you to provide
   * a path to the parent object (under which you want this name to be
   * defined) in the form of a name path string.
   *
   * In some cases, it is desirable to break up the path used to
   * describe an item in the names namespace into a path and a name.
   * This is analogous to a file system operation in which you provide
   * a directory name and a file name.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server".  If you further want to create an
   * association for between a Ptr<Object> object that you want to
   * live "under" the server in the name space -- perhaps "eth0" --
   * you could do this in two ways, depending on which was more
   * convenient: Names::Add ("/Names/server/eth0", object) or, using
   * the split path and name approach, Names::Add ("/Names/server",
   * "eth0", object).
   *
   * Duplicate names are not allowed at the same level in a path,
   * however you may associate similar names with different paths.
   * For example, if you define "/Names/Client", you may not define
   * another "/Names/Client" just as you may not have two files with
   * the same name in a classical filesystem.  However, you may have
   * "/Names/Client/eth0" and "/Names/Server/eth0" defined at the same
   * time just as you might have different files of the same name
   * under different directories.
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want this new name to be defined.
   * \param [in] name The name of the object you want to associate.
   * \param [in] object A smart pointer to the object itself.
   *
   * \see Names::Add (Ptr<Object>,std::string,Ptr<Object>);
   */
  static void Add (std::string path, std::string name, Ptr<Object> object);

  /**
   * \brief A low-level form of Names::Add allowing you to specify the
   * path to the parent object (under which you want this name to be
   * defined) in the form of a previously named object.
   *
   * In some use cases, it is desirable to break up the path in the
   * names name space into a path and a name.  This is analogous to a
   * file system operation in which you provide a directory name and a
   * file name.  Recall that the path string actually refers to a
   * previously named object, "under" which you want to accomplish
   * some naming action.
   *
   * However, the path is sometimes not available, and you only have
   * the object that is represented by the path in the names name
   * space.  To support this use-case in a reasonably high-performance
   * way, the path string is can be replaced by the object pointer to
   * which that path would refer.  In the spirit of the Config code
   * where this use-case is most prominent, we refer to this object as
   * the "context" for the names operation.
   *
   * You can think of the context roughly as the inode number of a
   * directory file in Unix.  The inode number can be used to look up
   * the directory file which contains the list of file names defined
   * at that directory level.  Similarly the context is used to look
   * up an internal name service entry which contains the names
   * defined for that context.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server".  If you further want to create an
   * association for between a Ptr<Object> object that you want to
   * live "under" the server in the name space -- perhaps "eth0" --
   * you could do this by providing a complete path to the new name:
   * Names::Add ("/Names/server/eth0", object).  If, however,
   * somewhere in your code you only had a pointer to the server, say
   * Ptr<Node> node, and not a handy path string, you could also
   * accomplish this by Names::Add (node, "eth0", object).
   *
   * Duplicate names are not allowed at the same level in a path.  In
   * the case of this method, the context object gives the same
   * information as a path string.  You may associate similar names
   * with different paths.  For example, if you define"/Names/Client",
   * you may not define another "/Names/Client" just as you may not
   * have two files with the same name in a classical filesystem.
   * However, you may have "/Names/Client/eth0" and
   * "/Names/Server/eth0" defined at the same time just as you might
   * have different files of the same name under different
   * directories.
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this new
   *             name to be defined.
   * \param [in] name The name of the object you want to associate.
   * \param [in] object A smart pointer to the object itself.
   */
  static void Add (Ptr<Object> context, std::string name, Ptr<Object> object);

  /**
   * \brief Rename a previously associated name.
   *
   * The name may begin either with "/Names" to explicitly call out
   * the fact that the name provided is installed under the root of
   * the name space, or it may begin with the name of the first object
   * in the path.  For example, Names::Rename ("/Names/client",
   * "server") and Names::Rename ("client", "server") accomplish
   * exactly the same thing.  Names at a given level in the name space
   * path must be unique. In the case of the example above, it would
   * be illegal to try and rename a different object to the same name:
   * "server" at the same level ("/Names") in the path.
   *
   * As well as specifying a name at the root of the "/Names"
   * namespace, the name parameter can contain a path that fully
   * qualifies the name to be changed.  For example, if you previously
   * have (re)named an object "server" in the root namespace as above,
   * you could then rename an object "under" that name by making a
   * call like Names::Rename ("/Names/server/csma", "eth0").  This
   * will rename the object previously associated with
   * "/Names/server/csma" to "eth0" and make leave it reachable using
   * the path "/Names/server/eth0".  Note that Names::Rename
   * ("server/csma", "eth0") would accomplish exactly the same thing.
   *
   * \param [in] oldpath The current path name to the object you want
   *             to change.
   * \param [in] newname The new name of the object you want to change.
   *
   * \see Names::Add (std::string name, Ptr<Object> obj)
   */
  static void Rename (std::string oldpath, std::string newname);

  /**
   * \brief An intermediate form of Names::Rename allowing you to
   * provide a path to the parent object (under which you want this
   * name to be changed) in the form of a name path string.
   *
   * In some cases, it is desirable to break up the path used to
   * describe an item in the names namespace into a path and a name.
   * This is analogous to a file system operation in which you provide
   * a directory name and a file name.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server/csma".  If you want to change the name
   * "csma" to "eth0", you could do this in two ways, depending on
   * which was more convenient: Names::Rename ("/Names/server/csma",
   * "eth0") or, using the split path and name approach, Names::Rename
   * ("/Names/server", "csma", "eth0").
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want this name change to occur
   *             (cf. directory).
   * \param [in] oldname The currently defined name of the object.
   * \param [in] newname The new name you want the object to have.
   */
  static void Rename (std::string path, std::string oldname, std::string newname);

  /**
   * \brief A low-level form of Names::Rename allowing you to specify
   * the path to the parent object (under which you want this name to
   * be changed) in the form of a previously named object.
   *
   * In some use cases, it is desirable to break up the path in the
   * names name space into a path and a name.  This is analogous to a
   * file system operation in which you provide a directory name and a
   * file name.  Recall that the path string actually refers to a
   * previously named object, "under" which you want to accomplish
   * some naming action.
   *
   * However, the path is sometimes not available, and you only have
   * the object that is represented by the path in the names name
   * space.  To support this use-case in a reasonably high-performance
   * way, the path string is can be replaced by the object pointer to
   * which that path would refer.  In the spirit of the Config code
   * where this use-case is most prominent, we refer to this object as
   * the "context" for the names operation.
   *
   * You can think of the context roughly as the inode number of a
   * directory file in Unix.  The inode number can be used to look up
   * the directory file which contains the list of file names defined
   * at that directory level.  Similarly the context is used to look
   * up an internal name service entry which contains the names
   * defined for that context.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server/csma".  If you later decide to rename
   * the csma object to say "eth0" -- you could do this by providing a
   * complete path as in Names::Rename ("/Names/server/csma", "eth0").
   * If, however, somewhere in your code you only had a pointer to the
   * server, and not a handy path string, say Ptr<Node> node, you
   * could also accomplish this by Names::Rename (node, "csma",
   * "eth0").
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this
   *             new name to be defined.
   * \param [in] oldname The current shortname of the object you want
   *             to change.
   * \param [in] newname The new shortname of the object you want
   *             to change.
   */
  static void Rename (Ptr<Object> context, std::string oldname, std::string newname);

  /**
   * \brief Given a pointer to an object, look to see if that object
   * has a name associated with it and, if so, return the name of the
   * object otherwise return an empty string.
   *
   * An object can be referred to in two ways.  Either you can talk
   * about it using its fully qualified path name, for example,
   * "/Names/client/eth0" or you can refer to it by its name, in this
   * case "eth0".
   *
   * This method returns the name of the object, e.g., "eth0".
   *
   * \param [in] object A smart pointer to an object for which you want
   *             to find its name.
   *
   * \returns A string containing the name of the object if found,
   *          otherwise the empty string.
   */
  static std::string FindName (Ptr<Object> object);

  /**
   * \brief Given a pointer to an object, look to see if that object
   * has a name associated with it and return the fully qualified name
   * path of the object otherwise return an empty string.
   *
   * An object can be referred to in two ways.  Either you can talk
   * about it using its fully qualified path name, for example,
   * "/Names/client/eth0" or you can refer to it by its name, in this
   * case "eth0".
   *
   * This method returns the name path of the object, e.g.,
   * "Names/client/eth0".
   *
   * \param [in] object A smart pointer to an object for which you
   *             want to find its fullname.
   *
   * \returns A string containing the name path of the object,
   *          otherwise the empty string.
   */
  static std::string FindPath (Ptr<Object> object);

  /**
   * \brief Clear the list of objects associated with names.
   */

  static void Clear (void);

  /**
   * \brief Given a name path string, look to see if there's an object
   * in the system with that associated to it.  If there is, do a
   * GetObject on the resulting object to convert it to the requested
   * typename and return it.
   *
   * An object can be referred to in two ways.  Either you can talk
   * about it using its fully qualified path name, for example,
   * "/Names/client/eth0" or you can refer to it by its name, in this
   * case "eth0".
   *
   * This method requires that the name path of the object be
   * provided, e.g., "Names/client/eth0".
   *
   * \param [in] path A string containing a name space path used
   *             to locate the object.
   *
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  template <typename T>
  static Ptr<T> Find (std::string path);

  /**
   * \brief Given a path to an object and an object name, look through
   * the names defined under the path to see if there's an object
   * there with the given name.
   *
   * In some cases, it is desirable to break up the path used to
   * describe an item in the names namespace into a path and a name.
   * This is analogous to a file system operation in which you provide
   * a directory name and a file name.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server/eth0".  If you want to discover the
   * object which you associated with this path, you could do this in
   * two ways, depending on which was more convenient: Names::Find
   * ("/Names/server/eth0") or, using the split path and name
   * approach, Names::Find ("/Names/server", "eth0").
   *
   * \param [in] path A path name describing a previously named object
   *             under which you want to look for the specified name.
   * \param [in] name A string containing a name to search for.
   *
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  template <typename T>
  static Ptr<T> Find (std::string path, std::string name);

  /**
   * \brief Given a path to an object and an object name, look through
   * the names defined under the path to see if there's an object
   * there with the given name.
   *
   * In some cases, it is desirable to break up the path used to
   * describe an item in the names namespace into a path and a name.
   * This is analogous to a file system operation in which you provide
   * a directory name and a file name.
   *
   * For example, consider a situation where you have previously named
   * an object "/Names/server/eth0".  If you want to discover the
   * object which you associated with this path, you could do this in
   * two ways, depending on which was more convenient: Names::Find
   * ("/Names/server/eth0") or, using the split path and name
   * approach, Names::Find ("/Names/server", "eth0").
   *
   * However, the path is sometimes not available, and you only have
   * the object that is represented by the path in the names name
   * space.  To support this use-case in a reasonably high-performance
   * way, the path string is can be replaced by the object pointer to
   * which that path would refer.  In the spirit of the Config code
   * where this use-case is most prominent, we refer to this object as
   * the "context" for the names operation.
   *
   * You can think of the context roughly as the inode number of a
   * directory file in Unix.  The inode number can be used to look up
   * the directory file which contains the list of file names defined
   * at that directory level.  Similarly the context is used to look
   * up an internal name service entry which contains the names
   * defined for that context.
   *
   * \param [in] context A smart pointer to an object that is used
   *             in place of the path under which you want this
   *             new name to be defined.
   * \param [in] name A string containing a name to search for.
   *
   * \returns A smart pointer to the named object converted to
   *          the requested type.
   */
  template <typename T>
  static Ptr<T> Find (Ptr<Object> context, std::string name);

private:
  /**
   * \brief Non-templated internal version of Names::Find
   *
   * \param [in] path A string containing the path of the object
   *             to look for.
   *
   * \returns A smart pointer to the named object.
   */
  static Ptr<Object> FindInternal (std::string path);

  /**
   * \brief Non-templated internal version of Names::Find
   *
   * \param [in] path A string containing the path to search
   *             for the object in.
   * \param [in] name A string containing the name of the object
   *             to look for.
   *
   * \returns A smart pointer to the named object.
   */
  static Ptr<Object> FindInternal (std::string path, std::string name);

  /**
   * \brief Non-templated internal version of Names::Find
   *
   * \param [in] context A smart pointer to an object under which
   *             you want to look for the provided name.
   * \param [in] name A string containing the name to look for.
   *
   * \returns A smart pointer to the named object.
   */
  static Ptr<Object> FindInternal (Ptr<Object> context, std::string name);
};


template <typename T>
/* static */
Ptr<T>
Names::Find (std::string path)
{
  Ptr<Object> obj = FindInternal (path);
  if (obj)
    {
      return obj->GetObject<T> ();
    }
  else
    {
      return 0;
    }
}

template <typename T>
/* static */
Ptr<T>
Names::Find (std::string path, std::string name)
{
  Ptr<Object> obj = FindInternal (path, name);
  if (obj)
    {
      return obj->GetObject<T> ();
    }
  else
    {
      return 0;
    }
}

template <typename T>
/* static */
Ptr<T>
Names::Find (Ptr<Object> context, std::string name)
{
  Ptr<Object> obj = FindInternal (context, name);
  if (obj)
    {
      return obj->GetObject<T> ();
    }
  else
    {
      return 0;
    }
}

} // namespace ns3

#endif /* OBJECT_NAMES_H */
