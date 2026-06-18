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
#ifndef CONFIG_H
#define CONFIG_H

#include "ptr.h"
#include <string>
#include <vector>

/**
 * \file
 * \ingroup config
 * Declaration of the various ns3::Config functions and classes.
 */

namespace ns3 {

class AttributeValue;
class Object;
class CallbackBase;

/**
 * \ingroup core
 * \defgroup config Configuration
 * \brief Configuration of simulation parameters and tracing.
 */

/**
 * \ingroup config
 * Namespace for the various functions implementing the Config system.
 */
namespace Config {

/**
 * \ingroup config
 * Reset the initial value of every attribute as well as the value of every
 * global to what they were before any call to SetDefault and SetGlobal.
 */
void Reset (void);

/**
 * \ingroup config
 * \param [in] path A path to match attributes.
 * \param [in] value The value to set in all matching attributes.
 *
 * This function will attempt to find attributes which
 * match the input path and will then set their value to the input
 * value.  If no such attributes are found, the function will throw
 * a fatal error; use SetFailSafe if the lack of a match is to be permitted.
 */
void Set (std::string path, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] path A path to match attributes.
 * \param [in] value The value to set in all matching attributes.
 *
 * This function will attempt to find attributes which
 * match the input path and will then set their value to the input
 * value, and will return true if at least one such attribute is found.
 * \return \c true if any matching attributes could be set.
 */
bool SetFailSafe (std::string path, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] name The full name of the attribute
 * \param [in] value The value to set.
 *
 * This method overrides the initial value of the
 * matching attribute. This method cannot fail: it will
 * crash if the input attribute name or value is invalid.
 */
void SetDefault (std::string name, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] name The full name of the attribute
 * \param [in] value The value to set.
 * \returns \c true if the value was set successfully, false otherwise.
 *
 * This method overrides the initial value of the
 * matching attribute.
 */
bool SetDefaultFailSafe (std::string name, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] name The name of the requested GlobalValue.
 * \param [in] value The value to set
 *
 * This method is equivalent to GlobalValue::Bind
 */
void SetGlobal (std::string name, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] name The name of the requested GlobalValue.
 * \param [in] value The value to set
 * \return \c true if the GlobalValue could be set.
 *
 * This method is equivalent to GlobalValue::BindFailSafe
 */
bool SetGlobalFailSafe (std::string name, const AttributeValue &value);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to connect to the matching trace sources.
 *
 * This function will attempt to find all trace sources which
 * match the input path and will then connect the input callback
 * to them.  If no matching trace sources are found, this method will 
 * throw a fatal error.  Use ConnectWithoutContextFailSafe if the absence
 * of matching trace sources should not be fatal.
 */
void ConnectWithoutContext (std::string path, const CallbackBase &cb);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to connect to the matching trace sources.
 *
 * This function will attempt to find all trace sources which
 * match the input path and will then connect the input callback
 * to them.  If no matching trace sources are found, this method will 
 * return false; otherwise true.
 * \returns \c true if any trace sources could be connected.
 */
bool ConnectWithoutContextFailSafe (std::string path, const CallbackBase &cb);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to disconnect to the matching trace sources.
 *
 * This function undoes the work of Config::Connect.
 */
void DisconnectWithoutContext (std::string path, const CallbackBase &cb);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to connect to the matching trace sources.
 *
 * This function will attempt to find all trace sources which
 * match the input path and will then connect the input callback
 * to them in such a way that the callback will receive an extra
 * context string upon trace event notification.
 * If no matching trace sources are found, this method will 
 * throw a fatal error.  Use ConnectFailSafe if the absence
 * of matching trace sources should not be fatal.
 */
void Connect (std::string path, const CallbackBase &cb);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to connect to the matching trace sources.
 *
 * This function will attempt to find all trace sources which
 * match the input path and will then connect the input callback
 * to them in such a way that the callback will receive an extra
 * context string upon trace event notification.
 * \returns \c true if any trace sources could be connected.
 */
bool ConnectFailSafe (std::string path, const CallbackBase &cb);
/**
 * \ingroup config
 * \param [in] path A path to match trace sources.
 * \param [in] cb The callback to connect to the matching trace sources.
 *
 * This function undoes the work of Config::ConnectWithContext.
 */
void Disconnect (std::string path, const CallbackBase &cb);

/**
 * \ingroup config
 * \brief hold a set of objects which match a specific search string.
 *
 * This class also allows you to perform a set of configuration operations
 * on the set of matching objects stored in the container. Specifically,
 * it is possible to perform bulk Connects and Sets.
 */
class MatchContainer
{
public:
  /** Const iterator over the objects in this container. */
  typedef std::vector<Ptr<Object> >::const_iterator Iterator;
  MatchContainer ();
  /**
   * Constructor used only by implementation.
   *
   * \param [in] objects The vector of objects to store in this container.
   * \param [in] contexts The corresponding contexts.
   * \param [in] path The path used for object matching.
   */
  MatchContainer (const std::vector<Ptr<Object> > &objects,
                  const std::vector<std::string> &contexts,
                  std::string path);

  /**
   * \returns An iterator which points to the first item in the container
   * @{
   */
  MatchContainer::Iterator Begin (void) const;
  MatchContainer::Iterator begin (void) const { return Begin (); }
  /** @} */
  /**
   * \returns An iterator which points to the last item in the container
   * @{
   */
  MatchContainer::Iterator End (void) const;
  MatchContainer::Iterator end (void) const { return End (); }
  /** @} */
  /**
   * \returns The number of items in the container
   * @{
   */
  std::size_t GetN (void) const;
  std::size_t size (void) const { return GetN (); }
  /** @} */
  /**
   * \param [in] i Index of item to lookup ([0,n[)
   * \returns The item requested.
   */
  Ptr<Object> Get (std::size_t i) const;
  /**
   * \param [in] i Index of item to lookup ([0,n[)
   * \returns The fully-qualified matching path associated
   *          to the requested item.
   *
   * The matching patch uniquely identifies the requested object.
   */
  std::string GetMatchedPath (uint32_t i) const;
  /**
   * \returns The path used to perform the object matching.
   */
  std::string GetPath (void) const;

  /**
   * \param [in] name Name of attribute to set
   * \param [in] value Value to set to the attribute
   *
   * Set the specified attribute value to all the objects stored in this
   * container.  This method will raise a fatal error if no such attribute
   * exists; use SetFailSafe if the absence of the attribute is to be
   * permitted.  
   * \sa ns3::Config::Set
   */
  void Set (std::string name, const AttributeValue &value);
  /**
   * \param [in] name Name of attribute to set
   * \param [in] value Value to set to the attribute
   *
   * Set the specified attribute value to all the objects stored in this
   * container.  This method will return true if any attributes could be
   * set, and false otherwise.
   * \returns \c true if any attributes could be set.
   */
  bool SetFailSafe (std::string name, const AttributeValue &value);
  /**
   * \param [in] name The name of the trace source to connect to
   * \param [in] cb The sink to connect to the trace source
   *
   * Connect the specified sink to all the objects stored in this
   * container.  This method will raise a fatal error if no objects could
   * be connected; use ConnectFailSafe if no connections is a valid possible
   * outcome.
   * \sa ns3::Config::Connect
   */
  void Connect (std::string name, const CallbackBase &cb);
  /**
   * \param [in] name The name of the trace source to connect to
   * \param [in] cb The sink to connect to the trace source
   *
   * Connect the specified sink to all the objects stored in this
   * container.  This method will return true if any trace sources could be
   * connected, and false otherwise.
   * \returns \c true if any trace sources could be connected.
   */
  bool ConnectFailSafe (std::string name, const CallbackBase &cb);
  /**
   * \param [in] name The name of the trace source to connect to
   * \param [in] cb The sink to connect to the trace source
   *
   * Connect the specified sink to all the objects stored in this
   * container.  This method will raise a fatal error if no objects could
   * be connected; use ConnectWithoutContextFailSafe if no connections is
   * a valid possible outcome.
   * \sa ns3::Config::ConnectWithoutContext
   */
  void ConnectWithoutContext (std::string name, const CallbackBase &cb);
  /**
   * \param [in] name The name of the trace source to connect to
   * \param [in] cb The sink to connect to the trace source
   *
   * Connect the specified sink to all the objects stored in this
   * container.  This method will return true if any trace sources could be
   * connected, and false otherwise.
   * \returns \c true if any trace sources could be connected.
   */
  bool ConnectWithoutContextFailSafe (std::string name, const CallbackBase &cb);
  /**
   * \param [in] name The name of the trace source to disconnect from
   * \param [in] cb The sink to disconnect from the trace source
   *
   * Disconnect the specified sink from all the objects stored in this
   * container.
   * \sa ns3::Config::Disconnect
   */
  void Disconnect (std::string name, const CallbackBase &cb);
  /**
   * \param [in] name The name of the trace source to disconnect from
   * \param [in] cb The sink to disconnect from the trace source
   *
   * Disconnect the specified sink from all the objects stored in this
   * container.
   * \sa ns3::Config::DisconnectWithoutContext
   */
  void DisconnectWithoutContext (std::string name, const CallbackBase &cb);

private:
  /** The list of objects in this container. */
  std::vector<Ptr<Object> > m_objects;
  /** The context for each object. */
  std::vector<std::string> m_contexts;
  /** The path used to perform the object matching. */
  std::string m_path;
};

/**
 * \ingroup config
 * \param [in] path The path to perform a match against
 * \returns A container which contains all the objects which match the input
 *          path.
 */
MatchContainer LookupMatches (std::string path);

/**
 * \ingroup config
 * \param [in] obj A new root object
 *
 * Each root object is used during path matching as
 * the root of the path by Config::Connect, and Config::Set.
 */
void RegisterRootNamespaceObject (Ptr<Object> obj);
/**
 * \ingroup config
 * \param [in] obj A new root object
 *
 * This function undoes the work of Config::RegisterRootNamespaceObject.
 */
void UnregisterRootNamespaceObject (Ptr<Object> obj);

/**
 * \ingroup config
 * \returns The number of registered root namespace objects.
 */
std::size_t GetRootNamespaceObjectN (void);

/**
 * \ingroup config
 * \param [in] i The index of the requested object.
 * \returns The requested root namespace object
 */
Ptr<Object> GetRootNamespaceObject (uint32_t i);

} // namespace Config

} // namespace ns3

#endif /* CONFIG_H */
