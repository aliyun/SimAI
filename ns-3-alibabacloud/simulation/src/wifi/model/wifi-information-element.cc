/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Dean Armstrong
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
 * Author: Dean Armstrong <deanarm@gmail.com>
 */

#include "wifi-information-element.h"

namespace ns3 {

WifiInformationElement::~WifiInformationElement ()
{
}

void
WifiInformationElement::Print (std::ostream &os) const
{
}

uint16_t
WifiInformationElement::GetSerializedSize () const
{
  return (2 + GetInformationFieldSize ());
}

WifiInformationElementId
WifiInformationElement::ElementIdExt () const
{
  return 0;
}

Buffer::Iterator
WifiInformationElement::Serialize (Buffer::Iterator i) const
{
  i.WriteU8 (ElementId ());
  i.WriteU8 (GetInformationFieldSize ());
  if (ElementId () == IE_EXTENSION)
    {
      i.WriteU8 (ElementIdExt ());
      SerializeInformationField (i);
      i.Next (GetInformationFieldSize () - 1);
    }
  else
    {
      SerializeInformationField (i);
      i.Next (GetInformationFieldSize ());
    }
  return i;
}

Buffer::Iterator
WifiInformationElement::Deserialize (Buffer::Iterator i)
{
  Buffer::Iterator start = i;
  i = DeserializeIfPresent (i);
  //This IE was not optional, so confirm that we did actually
  //deserialise something.
  NS_ASSERT (i.GetDistanceFrom (start) != 0);
  return i;
}

Buffer::Iterator
WifiInformationElement::DeserializeIfPresent (Buffer::Iterator i)
{
  if (i.IsEnd ())
    {
      return i;
    }
  Buffer::Iterator start = i;
  uint8_t elementId = i.ReadU8 ();

  //If the element here isn't the one we're after then we immediately
  //return the iterator we were passed indicating that we haven't
  //taken anything from the buffer.
  if (elementId != ElementId ())
    {
      return start;
    }

  uint8_t length = i.ReadU8 ();
  if (ElementId () == IE_EXTENSION)
  {
    uint8_t elementIdExt = i.ReadU8 ();
    //If the element here isn't the one we're after then we immediately
    //return the iterator we were passed indicating that we haven't
    //taken anything from the buffer.
    if (elementIdExt != ElementIdExt ())
      {
        return start;
      }
    DeserializeInformationField (i, length - 1);
    i.Next (length - 1);
  }
  else
  {
    DeserializeInformationField (i, length);
    i.Next (length);
  }
  return i;
}

bool
WifiInformationElement::operator== (WifiInformationElement const & a) const
{
  if (ElementId () != a.ElementId ())
    {
      return false;
    }

  if (GetInformationFieldSize () != a.GetInformationFieldSize ())
    {
      return false;
    }

  if (ElementIdExt () != a.ElementIdExt ())
    {
      return false;
    }

  uint32_t ieSize = GetInformationFieldSize ();

  Buffer myIe, hisIe;
  myIe.AddAtEnd (ieSize);
  hisIe.AddAtEnd (ieSize);

  SerializeInformationField (myIe.Begin ());
  a.SerializeInformationField (hisIe.Begin ());

  return (memcmp (myIe.PeekData (), hisIe.PeekData (), ieSize) == 0);
}

} //namespace ns3
