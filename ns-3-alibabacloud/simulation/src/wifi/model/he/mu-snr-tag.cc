/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "mu-snr-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (MuSnrTag);

TypeId
MuSnrTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MuSnrTag")
    .SetParent<Tag> ()
    .SetGroupName ("Wifi")
    .AddConstructor<MuSnrTag> ()
  ;
  return tid;
}

TypeId
MuSnrTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

MuSnrTag::MuSnrTag ()
{
}

void
MuSnrTag::Reset (void)
{
  m_snrMap.clear ();
}

void
MuSnrTag::Set (uint16_t staId, double snr)
{
  m_snrMap[staId] = snr;
}

bool
MuSnrTag::IsPresent (uint16_t staId) const
{
  return (m_snrMap.find (staId) != m_snrMap.end ());
}

double
MuSnrTag::Get (uint16_t staId) const
{
  NS_ASSERT (IsPresent (staId));
  return m_snrMap.at (staId);
}

uint32_t
MuSnrTag::GetSerializedSize (void) const
{
  return (sizeof (uint16_t) + sizeof (double)) * m_snrMap.size () + 1;
}

void
MuSnrTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_snrMap.size ());

  for (const auto& staIdSnrPair : m_snrMap)
    {
      i.WriteU16 (staIdSnrPair.first);
      i.WriteDouble (staIdSnrPair.second);
    }
}

void
MuSnrTag::Deserialize (TagBuffer i)
{
  uint8_t n = i.ReadU8 ();
  for (uint8_t j = 0; j < n; j++)
    {
      uint16_t staId = i.ReadU16 ();
      double snr = i.ReadDouble ();
      m_snrMap.insert ({staId, snr});
    }
}

void
MuSnrTag::Print (std::ostream &os) const
{
  for (const auto& staIdSnrPair : m_snrMap)
    {
      os << "{STA-ID=" << staIdSnrPair.first << " Snr=" << staIdSnrPair.second << "} ";
    }
  os << std::endl;
}

}
