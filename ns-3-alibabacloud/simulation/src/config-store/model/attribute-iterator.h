/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  This program is free software; you can redistribute it and/or modify
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
 
#ifndef ATTRIBUTE_ITERATOR_H
#define ATTRIBUTE_ITERATOR_H

#include "ns3/ptr.h"
#include "ns3/object.h"
#include "ns3/object-ptr-container.h"
#include <vector>

namespace ns3 {

/**
 * \ingroup configstore
 *
 * \brief Iterator to iterate on the values of attributes of an ns3::Object 
 * \note This class is used internally by ConfigStore and GtkConfigStore.
 */
class AttributeIterator
{
public:
  AttributeIterator ();
  virtual ~AttributeIterator ();

  /**
   * Start the process of iterating all objects from the root namespace object
   */
  void Iterate (void);
protected:
  /**
   * Get the current attribute path
   * \returns the current path string
   */
  std::string GetCurrentPath (void) const;
private:
  /**
   * This method visits and performs a config-store action (such as saving
   * to a text file) on the attribute values corresponding to the input 
   * object pointer and attribute name.
   *
   * \param object the object visited
   * \param name the attribute name
   */
  virtual void DoVisitAttribute (Ptr<Object> object, std::string name) = 0;
  /**
   * This method is called to start the process of visiting the input object
   * \param object the object visited
   */
  virtual void DoStartVisitObject (Ptr<Object> object);
  /**
   * This method is called to end the process of visiting the currently
   * visited object.
   */
  virtual void DoEndVisitObject (void);
  /**
   * Visit the attribute of type ns3::PointerValue, with the provided name,
   * found on the object pointed to by the first argument.
   *
   * \param object the object on which the attribute of type PointerValue resides
   * \param name the attribute name provided
   * \param [in] value Ptr to the ns3::Object pointed to by the attribute
   */
  virtual void DoStartVisitPointerAttribute (Ptr<Object> object, std::string name, Ptr<Object> value);
  /**
   * End the visit to the attribute of type ns3::PointerValue.
   */
  virtual void DoEndVisitPointerAttribute (void);
  /**
   * Visit the attribute of type ns3::ObjectVectorValue, with the
   * provided name, found on the object pointed to by the first argument.
   *
   * \note type name ObjectVectorValue is an alias for ObjectPtrContainerValue
   *
   * \param object the object on which the attribute of type ObjectVectorValue resides
   * \param name the attribute name provided
   * \param [in] vector the ObjectPtrContainerValue corresponding to the named attribute 
   */
  virtual void DoStartVisitArrayAttribute (Ptr<Object> object, std::string name, const ObjectPtrContainerValue &vector);
  /**
   * End the visit to the attribute of type ns3::ObjectVectorValue.
   */
  virtual void DoEndVisitArrayAttribute (void);
  /**
   * Start to visit the object found in the input array at the provided index
   * \param vector the array
   * \param index the index into the array
   * \param [in] item the array item to visit
   */
  virtual void DoStartVisitArrayItem (const ObjectPtrContainerValue &vector, uint32_t index, Ptr<Object> item);
  /**
   * End the visit to the array item
   */
  virtual void DoEndVisitArrayItem (void);

  /**
   * Perform the iteration
   * \param object the object visited
   */
  void DoIterate (Ptr<Object> object);
  /**
   * Check if this object has already been examined
   * \param object the object to check
   * \returns true if object has been examined
   */
  bool IsExamined (Ptr<const Object> object);
  /**
   * Get current attribute path
   * \param attr the current attribute string
   * \returns the current path string
   */
  std::string GetCurrentPath (std::string attr) const;

  /**
   * Visit attribute to perform a config store operation on it
   * \param object the current object
   * \param name the attribute name
   */
  void VisitAttribute (Ptr<Object> object, std::string name);
  /**
   * Start to visit an object to visit its attributes
   * \param object the current object
   */
  void StartVisitObject (Ptr<Object> object);
  /**
   * End the visit to the object
   */
  void EndVisitObject (void);
  /**
   * Visit the attribute of type ns3::PointerValue, with the provided name,
   * found on the object pointed to by the first argument.
   *
   * \param object the object on which the attribute of type PointerValue resides
   * \param name the attribute name provided
   * \param [in] value Ptr to the ns3::Object pointed to by the attribute
   */
  void StartVisitPointerAttribute (Ptr<Object> object, std::string name, Ptr<Object> value);
  /**
   * End the visit to the attribute of type ns3::PointerValue.
   */
  void EndVisitPointerAttribute (void);
  /**
   * Visit the attribute of type ns3::ObjectVectorValue, with the
   * provided name, found on the object pointed to by the first argument.
   *
   * \note type name ObjectVectorValue is an alias for ObjectPtrContainerValue
   *
   * \param object the object on which the attribute of type ObjectVectorValue resides
   * \param name the attribute name provided
   * \param [in] vector the ObjectPtrContainerValue corresponding to the named attribute 
   */
  void StartVisitArrayAttribute (Ptr<Object> object, std::string name, const ObjectPtrContainerValue &vector);
  /**
   * End the visit to the attribute of type ns3::ObjectVectorValue.
   */
  void EndVisitArrayAttribute (void);
  /**
   * Start to visit the object found in the input array at the provided index
   * \param vector the array
   * \param index the index into the array
   * \param [in] item the array item to visit
   */
  void StartVisitArrayItem (const ObjectPtrContainerValue &vector, uint32_t index, Ptr<Object> item);
  /**
   * End the visit to the array item
   */
  void EndVisitArrayItem (void);


  std::vector<Ptr<Object> > m_examined; ///< list of attributes examined
  std::vector<std::string> m_currentPath; ///< current attribute path
};

} // namespace ns3

#endif /* ATTRIBUTE_ITERATOR_H */
