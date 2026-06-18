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
 
#include "attribute-iterator.h"
#include "ns3/config.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/object-ptr-container.h"
#include "ns3/string.h"
#include <fstream>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AttributeIterator");

AttributeIterator::AttributeIterator ()
{
}

AttributeIterator::~AttributeIterator ()
{
}

void 
AttributeIterator::Iterate (void)
{
  for (uint32_t i = 0; i < Config::GetRootNamespaceObjectN (); ++i)
    {
      Ptr<Object> object = Config::GetRootNamespaceObject (i);
      StartVisitObject (object);
      DoIterate (object);
      EndVisitObject ();
    }
  NS_ASSERT (m_currentPath.empty ());
  NS_ASSERT (m_examined.empty ());
}

bool
AttributeIterator::IsExamined (Ptr<const Object> object)
{
  for (uint32_t i = 0; i < m_examined.size (); ++i)
    {
      if (object == m_examined[i])
        {
          return true;
        }
    }
  return false;
}


std::string
AttributeIterator::GetCurrentPath (std::string attr) const
{
  std::ostringstream oss;
  for (uint32_t i = 0; i < m_currentPath.size (); ++i)
    {
      oss << "/" << m_currentPath[i];
    }
  if (attr != "")
    {
      oss << "/" << attr;
    }
  return oss.str ();
}

std::string
AttributeIterator::GetCurrentPath (void) const
{
  std::ostringstream oss;
  for (uint32_t i = 0; i < m_currentPath.size (); ++i)
    {
      oss << "/" << m_currentPath[i];
    }
  return oss.str ();
}

void 
AttributeIterator::DoStartVisitObject (Ptr<Object> object)
{
}
void 
AttributeIterator::DoEndVisitObject (void)
{
}
void 
AttributeIterator::DoStartVisitPointerAttribute (Ptr<Object> object, std::string name, Ptr<Object> item)
{
}
void 
AttributeIterator::DoEndVisitPointerAttribute (void)
{
}
void 
AttributeIterator::DoStartVisitArrayAttribute (Ptr<Object> object, std::string name, const ObjectPtrContainerValue &vector)
{
}
void 
AttributeIterator::DoEndVisitArrayAttribute (void)
{
}
void 
AttributeIterator::DoStartVisitArrayItem (const ObjectPtrContainerValue &vector, uint32_t index, Ptr<Object> item)
{
}
void 
AttributeIterator::DoEndVisitArrayItem (void)
{
}

void 
AttributeIterator::VisitAttribute (Ptr<Object> object, std::string name)
{
  m_currentPath.push_back (name);
  DoVisitAttribute (object, name);
  m_currentPath.pop_back ();
}

void 
AttributeIterator::StartVisitObject (Ptr<Object> object)
{
  m_currentPath.push_back ("$" + object->GetInstanceTypeId ().GetName ());
  DoStartVisitObject (object);
}
void 
AttributeIterator::EndVisitObject (void)
{
  m_currentPath.pop_back ();
  DoEndVisitObject ();
}
void 
AttributeIterator::StartVisitPointerAttribute (Ptr<Object> object, std::string name, Ptr<Object> value)
{
  m_currentPath.push_back (name);
  m_currentPath.push_back ("$" + value->GetInstanceTypeId ().GetName ());
  DoStartVisitPointerAttribute (object, name, value);
}
void 
AttributeIterator::EndVisitPointerAttribute (void)
{
  m_currentPath.pop_back ();
  m_currentPath.pop_back ();
  DoEndVisitPointerAttribute ();
}
void 
AttributeIterator::StartVisitArrayAttribute (Ptr<Object> object, std::string name, const ObjectPtrContainerValue &vector)
{
  m_currentPath.push_back (name);
  DoStartVisitArrayAttribute (object, name, vector);
}
void 
AttributeIterator::EndVisitArrayAttribute (void)
{
  m_currentPath.pop_back ();
  DoEndVisitArrayAttribute ();
}

void 
AttributeIterator::StartVisitArrayItem (const ObjectPtrContainerValue &vector, uint32_t index, Ptr<Object> item)
{
  std::ostringstream oss;
  oss << index;
  m_currentPath.push_back (oss.str ());
  m_currentPath.push_back ("$" + item->GetInstanceTypeId ().GetName ());
  DoStartVisitArrayItem (vector, index, item);
}
void 
AttributeIterator::EndVisitArrayItem (void)
{
  m_currentPath.pop_back ();
  m_currentPath.pop_back ();
  DoEndVisitArrayItem ();
}


void
AttributeIterator::DoIterate (Ptr<Object> object)
{
  if (IsExamined (object))
    {
      return;
    }
  TypeId tid;
  for (tid = object->GetInstanceTypeId (); tid.HasParent (); tid = tid.GetParent ())
    {
      NS_LOG_DEBUG ("store " << tid.GetName ());
      for (uint32_t i = 0; i < tid.GetAttributeN (); ++i)
        {
          struct TypeId::AttributeInformation info = tid.GetAttribute(i);
          const PointerChecker *ptrChecker = dynamic_cast<const PointerChecker *> (PeekPointer (info.checker));
          if (ptrChecker != 0)
            {
              NS_LOG_DEBUG ("pointer attribute " << info.name);
              PointerValue ptr;
              object->GetAttribute (info.name, ptr);
              Ptr<Object> tmp = ptr.Get<Object> ();
              if (tmp != 0)
                {
                  StartVisitPointerAttribute (object, info.name,
                                              tmp);
                  m_examined.push_back (object);
                  DoIterate (tmp);
                  m_examined.pop_back ();
                  EndVisitPointerAttribute ();
                }
              continue;
            }
          // attempt to cast to an object container
          const ObjectPtrContainerChecker *vectorChecker = dynamic_cast<const ObjectPtrContainerChecker *> (PeekPointer (info.checker));
          if (vectorChecker != 0)
            {
              NS_LOG_DEBUG ("ObjectPtrContainer attribute " << info.name);
              ObjectPtrContainerValue vector;
              object->GetAttribute (info.name, vector);
              StartVisitArrayAttribute (object, info.name, vector);
              ObjectPtrContainerValue::Iterator it;
              for (it = vector.Begin (); it != vector.End (); ++it)
                {
                  uint32_t j = (*it).first;
                  NS_LOG_DEBUG ("ObjectPtrContainer attribute item " << j);
                  Ptr<Object> tmp = (*it).second;
                  if (tmp)
                    {
                      StartVisitArrayItem (vector, j, tmp);
                      m_examined.push_back (object);
                      DoIterate (tmp);
                      m_examined.pop_back ();
                      EndVisitArrayItem ();
                    }
                }
              EndVisitArrayAttribute ();
              continue;
            }
          if ((info.flags & TypeId::ATTR_GET) && info.accessor->HasGetter () && 
              (info.flags & TypeId::ATTR_SET) && info.accessor->HasSetter ())
            {
              VisitAttribute (object, info.name);
            }
          else
            {
              NS_LOG_DEBUG ("could not store " << info.name);
            }
        }
    }
  Object::AggregateIterator iter = object->GetAggregateIterator ();
  bool recursiveAggregate = false;
  while (iter.HasNext ())
    {
      Ptr<const Object> tmp = iter.Next ();
      if (IsExamined (tmp))
        {
          recursiveAggregate = true;
        }
    }
  if (!recursiveAggregate)
    {
      iter = object->GetAggregateIterator ();
      while (iter.HasNext ())
        {
          Ptr<Object> tmp = const_cast<Object *> (PeekPointer (iter.Next ()));
          StartVisitObject (tmp);
          m_examined.push_back (object);
          DoIterate (tmp);
          m_examined.pop_back ();
          EndVisitObject ();
        }
    }
}


} // namespace ns3
