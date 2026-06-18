/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013
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
 * Authors: Ghada Badawy <gbadawy@gmail.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ampdu-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (AmpduTag);

TypeId
AmpduTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AmpduTag")
    .SetParent<Tag> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AmpduTag> ()
  ;
  return tid;
}

TypeId
AmpduTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

AmpduTag::AmpduTag ()
  : m_nbOfMpdus (0),
    m_duration (Seconds(0))
{
}

void
AmpduTag::SetRemainingNbOfMpdus (uint8_t nbOfMpdus)
{
  m_nbOfMpdus = nbOfMpdus;
}

void
AmpduTag::SetRemainingAmpduDuration (Time duration)
{
  NS_ASSERT (m_duration <= MilliSeconds(10));
  m_duration = duration;
}

uint32_t
AmpduTag::GetSerializedSize (void) const
{
  return (1 + sizeof (Time));
}

void
AmpduTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_nbOfMpdus);
  int64_t duration = m_duration.GetTimeStep ();
  i.Write ((const uint8_t *)&duration, sizeof(int64_t));
}

void
AmpduTag::Deserialize (TagBuffer i)
{
  m_nbOfMpdus = i.ReadU8 ();
  int64_t duration;
  i.Read ((uint8_t *)&duration, sizeof(int64_t));
  m_duration = Time (duration);
}

uint8_t
AmpduTag::GetRemainingNbOfMpdus () const
{
  return m_nbOfMpdus;
}

Time
AmpduTag::GetRemainingAmpduDuration () const
{
  return m_duration;
}

void
AmpduTag::Print (std::ostream &os) const
{
  os << "Remaining number of MPDUs=" << m_nbOfMpdus
     << " Remaining A-MPDU duration=" << m_duration;
}

} //namespace ns3
