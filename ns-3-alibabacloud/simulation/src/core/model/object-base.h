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
#ifndef OBJECT_BASE_H
#define OBJECT_BASE_H

#include "type-id.h"
#include "callback.h"
#include <string>
#include <list>

/**
 * \file
 * \ingroup object
 * ns3::ObjectBase declaration and
 * NS_OBJECT_ENSURE_REGISTERED() madro definition.
 */

/**
 * \ingroup object
 * \brief Register an Object subclass with the TypeId system.
 *
 * This macro should be invoked once for every class which
 * defines a new GetTypeId method.
 *
 * If the class is in a namespace, then the macro call should also be
 * in the namespace.
 */
#define NS_OBJECT_ENSURE_REGISTERED(type)               \
  static struct Object ## type ## RegistrationClass     \
  {                                                     \
    Object ## type ## RegistrationClass () {            \
      ns3::TypeId tid = type::GetTypeId ();             \
      tid.SetSize (sizeof (type));                      \
      tid.GetParent ();                                 \
    }                                                   \
  } Object ## type ## RegistrationVariable


/**
 * \ingroup object
 * \brief Explicitly instantiate a template class and register the resulting
 *        instance with the TypeId system.
 *
 * This macro should be invoked once for every required instance of a template
 * class which derives from the Object class and defines a new GetTypeId method.
 *
 * If the template class is in a namespace, then the macro call should also be
 * in the namespace.
 */
#define NS_OBJECT_TEMPLATE_CLASS_DEFINE(type,param)                    \
  template class type<param>;                                          \
  template <> std::string DoGetTypeParamName<type<param> > ()          \
  {                                                                    \
    return #param;                                                     \
  }                                                                    \
  static struct Object ## type ## param ## RegistrationClass           \
  {                                                                    \
    Object ## type ## param ## RegistrationClass () {                  \
      ns3::TypeId tid = type<param>::GetTypeId ();                     \
      tid.SetSize (sizeof (type<param>));                              \
      tid.GetParent ();                                                \
    }                                                                  \
  } Object ## type ## param ## RegistrationVariable


namespace ns3 {

/**
 * \brief Helper function to get the name (as a string) of the type parameter
 *        of a template class
 * \return the name of the type parameter as a string
 *
 * A specialization of this function is defined by the
 * NS_OBJECT_TEMPLATE_CLASS_DEFINE macro.
 */
template <typename T>
std::string DoGetTypeParamName (void);

/**
 * \brief Helper function to get the name (as a string) of the type parameter
 *        of a template class
 * \return the name of the type parameter as a string
 */
template <typename T>
std::string GetTypeParamName (void)
{
  return DoGetTypeParamName<T> ();
}

class AttributeConstructionList;

/**
 * \ingroup object
 *
 * \brief Anchor the ns-3 type and attribute system.
 *
 * Every class which wants to integrate in the ns-3 type and attribute
 * system should derive from this base class. This base class provides:
 * - A way to associate an ns3::TypeId to each object instance.
 * - A way to set and get the attributes registered in the ns3::TypeId.
 */
class ObjectBase
{
public:
  /**
   * Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  /**
   * Virtual destructor.
   */
  virtual ~ObjectBase ();

  /**
   * Get the most derived TypeId for this Object.
   *
   * This method is typically implemented by ns3::Object::GetInstanceTypeId
   * but some classes which derive from ns3::ObjectBase directly
   * have to implement it themselves.
   *
   * \return The TypeId associated to the most-derived type
   *          of this instance.
   */
  virtual TypeId GetInstanceTypeId (void) const = 0;

  /**
   *
   * Set a single attribute, raising fatal errors if unsuccessful.
   *
   * This will either succeed at setting the attribute
   * or it will raise NS_FATAL_ERROR() on these conditions:
   *
   *   - The attribute doesn't exist in this Object.
   *   - The attribute can't be set (no Setter).
   *   - The attribute couldn't be deserialized from the AttributeValue.
   *
   * \param [in] name The name of the attribute to set.
   * \param [in] value The name of the attribute to set.
   */
  void SetAttribute (std::string name, const AttributeValue &value);
  /**
   * Set a single attribute without raising errors.
   *
   * If the attribute could not be set this will return \c false,
   * but not raise any errors.
   *
   * \param [in] name The name of the attribute to set.
   * \param [in] value The value to set it to.
   * \return \c true if the requested attribute exists and could be set,
   *         \c false otherwise.
   */
  bool SetAttributeFailSafe (std::string name, const AttributeValue &value);
  /**
   * Get the value of an attribute, raising fatal errors if unsuccessful.
   *
   * This will either succeed at setting the attribute
   * or it will raise NS_FATAL_ERROR() on these conditions:
   *
   *   - The attribute doesn't exist in this Object.
   *   - The attribute can't be read (no Getter).
   *   - The attribute doesn't support string formatting.
   *   - The attribute couldn't be serialized into the AttributeValue.
   *
   * \param [in]  name The name of the attribute to read.
   * \param [out] value Where the result should be stored.
   */
  void GetAttribute (std::string name, AttributeValue &value) const;
  /**
   * Get the value of an attribute without raising erros.
   *
   * If the attribute could not be read this will return \c false,
   * but not raise any errors.
   *
   * \param [in]  name The name of the attribute to read.
   * \param [out] value Where the result value should be stored.
   * \return \c true if the requested attribute was found, \c false otherwise.
   */
  bool GetAttributeFailSafe (std::string name, AttributeValue &value) const;

  /**
   * Connect a TraceSource to a Callback with a context.
   *
   * The target trace source should be registered with TypeId::AddTraceSource.
   *
   * \param [in] name The name of the target trace source.
   * \param [in] context The trace context associated to the callback.
   * \param [in] cb The callback to connect to the trace source.
   * \returns \c true on success, \c false if TraceSource was not found.
   */
  bool TraceConnect (std::string name, std::string context, const CallbackBase &cb);
  /**
   * Connect a TraceSource to a Callback without a context.
   *
   * The target trace source should be registered with TypeId::AddTraceSource.
   *
   * \param [in] name The name of the target trace source.
   * \param [in] cb The callback to connect to the trace source.
   * \returns \c true on success, \c false if TraceSource was not found.
   */
  bool TraceConnectWithoutContext (std::string name, const CallbackBase &cb);
  /**
   * Disconnect from a TraceSource a Callback previously connected
   * with a context.
   *
   * The target trace source should be registered with TypeId::AddTraceSource.
   *
   * \param [in] name The name of the target trace source.
   * \param [in] context The trace context associated to the callback.
   * \param [in] cb The callback to disconnect from the trace source.
   * \returns \c true on success, \c false if TraceSource was not found.
   */
  bool TraceDisconnect (std::string name, std::string context, const CallbackBase &cb);
  /**
   * Disconnect from a TraceSource a Callback previously connected
   * without a context.
   *
   * The target trace source should be registered with TypeId::AddTraceSource.
   *
   * \param [in] name The name of the target trace source.
   * \param [in] cb The callback to disconnect from the trace source.
   * \returns \c true on success, \c false if TraceSource was not found.
   */
  bool TraceDisconnectWithoutContext (std::string name, const CallbackBase &cb);

protected:
  /**
   * Notifier called once the ObjectBase is fully constructed.
   *
   * This method is invoked once all member attributes have been
   * initialized. Subclasses can override this method to be notified
   * of this event but if they do this, they must chain up to their
   * parent's NotifyConstructionCompleted method.
   */
  virtual void NotifyConstructionCompleted (void);
  /**
   * Complete construction of ObjectBase; invoked by derived classes.
   *
   * Invoked from subclasses to initialize all of their
   * attribute members. This method will typically be invoked
   * automatically from ns3::CreateObject if your class derives
   * from ns3::Object. If you derive from ns3::ObjectBase directly,
   * you should make sure that you invoke this method from
   * your most-derived constructor.
   *
   * \param [in] attributes The attribute values used to initialize
   *        the member variables of this object's instance.
   */
  void ConstructSelf (const AttributeConstructionList &attributes);

private:
  /**
   * Attempt to set the value referenced by the accessor \pname{spec}
   * to a valid value according to the \c checker, based on \pname{value}.
   *
   * \param [in] spec The accessor for the storage location.
   * \param [in] checker The checker to use in validating the value.
   * \param [in] value The value to attempt to store.
   * \returns \c true if the \c value could be validated by the \pname{checker}
   *          and written to the storage location.
   */
  bool DoSet (Ptr<const AttributeAccessor> spec,
              Ptr<const AttributeChecker> checker,
              const AttributeValue &value);

};

} // namespace ns3

#endif /* OBJECT_BASE_H */
