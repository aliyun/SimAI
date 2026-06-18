/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Pavel Boyko <boyko@iitp.ru>
 */

#include "wifi-information-element-vector.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (WifiInformationElementVector);

WifiInformationElementVector::WifiInformationElementVector ()
  : m_maxSize (1500)
{
}

WifiInformationElementVector::~WifiInformationElementVector ()
{
  for (IE_VECTOR::iterator i = m_elements.begin (); i != m_elements.end (); i++)
    {
      *i = 0;
    }
  m_elements.clear ();
}

TypeId
WifiInformationElementVector::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::WifiInformationElementVector")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiInformationElementVector> ();
  return tid;
}

TypeId
WifiInformationElementVector::GetInstanceTypeId () const
{
  return GetTypeId ();
}

uint32_t
WifiInformationElementVector::GetSerializedSize () const
{
  return GetSize ();
}

void
WifiInformationElementVector::Serialize (Buffer::Iterator start) const
{
  for (IE_VECTOR::const_iterator i = m_elements.begin (); i != m_elements.end (); i++)
    {
      start = (*i)->Serialize (start);
    }
}

uint32_t
WifiInformationElementVector::Deserialize (Buffer::Iterator start)
{
  NS_FATAL_ERROR ("This variant should not be called on a variable-sized header");
  return 0;
}

uint32_t
WifiInformationElementVector::Deserialize (Buffer::Iterator start, Buffer::Iterator end)
{
  uint32_t size = start.GetDistanceFrom (end);
  uint32_t remaining = size;
  while (remaining > 0)
    {
      uint32_t deserialized = DeserializeSingleIe (start);
      start.Next (deserialized);
      NS_ASSERT (deserialized <= remaining);
      remaining -= deserialized;
    }
  NS_ASSERT_MSG (remaining == 0, "Error in deserialization");
  return size;
}

uint32_t
WifiInformationElementVector::DeserializeSingleIe (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t id = i.ReadU8 ();
  //unused: uint8_t length = i.ReadU8 ();
  //but need side effects of read:
  i.ReadU8 ();
  Ptr<WifiInformationElement> newElement;
  switch (id)
    {
    case 0: // eliminate compiler warning
    default:
      NS_FATAL_ERROR ("Information element " << +id << " is not implemented");
      return 0;
    }
  /*  unreachable:  b/c switch is guaranteed to return from this function
  if (GetSize () + length > m_maxSize)
    {
      NS_FATAL_ERROR ("Check max size for information element!");
    }
  newElement->DeserializeInformationField (i, length);
  i.Next (length);
  m_elements.push_back (newElement);
  return i.GetDistanceFrom (start);
  */
}

void
WifiInformationElementVector::Print (std::ostream & os) const
{
  for (IE_VECTOR::const_iterator i = m_elements.begin (); i != m_elements.end (); i++)
    {
      os << "(";
      (*i)->Print (os);
      os << ")";
    }
}

WifiInformationElementVector::Iterator
WifiInformationElementVector::Begin ()
{
  return m_elements.begin ();
}

WifiInformationElementVector::Iterator
WifiInformationElementVector::End ()
{
  return m_elements.end ();
}

bool
WifiInformationElementVector::AddInformationElement (Ptr<WifiInformationElement> element)
{
  if (element->GetInformationFieldSize () + 2 + GetSize () > m_maxSize)
    {
      return false;
    }
  m_elements.push_back (element);
  return true;
}

Ptr<WifiInformationElement>
WifiInformationElementVector::FindFirst (WifiInformationElementId id) const
{
  for (IE_VECTOR::const_iterator i = m_elements.begin (); i != m_elements.end (); i++)
    {
      if ((*i)->ElementId () == id)
        {
          return (*i);
        }
    }
  return 0;
}

uint32_t
WifiInformationElementVector::GetSize () const
{
  uint32_t size = 0;
  for (IE_VECTOR::const_iterator i = m_elements.begin (); i != m_elements.end (); i++)
    {
      size += ((*i)->GetInformationFieldSize () + 2);
    }
  return size;
}

bool
WifiInformationElementVector::operator== (const WifiInformationElementVector & a) const
{
  if (m_elements.size () != a.m_elements.size ())
    {
      NS_ASSERT (false);
      return false;
    }
  //In principle we could bypass some of the faffing about (and speed
  //the comparison) by simply serialising each IE vector into a
  //buffer and memcmp'ing the two.
  //
  //I'm leaving it like this, however, so that there is the option of
  //having individual Information Elements implement slightly more
  //flexible equality operators.
  WifiInformationElementVector::IE_VECTOR::const_iterator j = a.m_elements.begin ();
  for (WifiInformationElementVector::IE_VECTOR::const_iterator i = m_elements.begin (); i
       != m_elements.end (); i++, j++)
    {
      if (!(*(*i) == *(*j)))
        {
          return false;
        }
    }

  return true;
}

} //namespace ns3
