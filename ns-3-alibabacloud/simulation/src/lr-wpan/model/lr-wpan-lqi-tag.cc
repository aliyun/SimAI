/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Fraunhofer FKIE
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
 * Author:
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */
#include "lr-wpan-lqi-tag.h"
#include <ns3/integer.h>

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (LrWpanLqiTag);

TypeId
LrWpanLqiTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanLqiTag")
    .SetParent<Tag> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanLqiTag> ()
    .AddAttribute ("Lqi", "The lqi of the last packet received",
                   IntegerValue (0),
                   MakeIntegerAccessor (&LrWpanLqiTag::Get),
                   MakeIntegerChecker<uint8_t> ())
  ;
  return tid;
}

TypeId
LrWpanLqiTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

LrWpanLqiTag::LrWpanLqiTag (void)
  : m_lqi (0)
{
}

LrWpanLqiTag::LrWpanLqiTag (uint8_t lqi)
  : m_lqi (lqi)
{
}

uint32_t
LrWpanLqiTag::GetSerializedSize (void) const
{
  return sizeof (uint8_t);
}

void
LrWpanLqiTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_lqi);
}

void
LrWpanLqiTag::Deserialize (TagBuffer i)
{
  m_lqi = i.ReadU8 ();
}

void
LrWpanLqiTag::Print (std::ostream &os) const
{
  os << "Lqi = " << m_lqi;
}

void
LrWpanLqiTag::Set (uint8_t lqi)
{
  m_lqi = lqi;
}

uint8_t
LrWpanLqiTag::Get (void) const
{
  return m_lqi;
}

}
