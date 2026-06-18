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
 * Author: Ghada Badawy <gbadawy@gmail.com>
 */

#include "ampdu-subframe-header.h"
#include <iomanip>

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (AmpduSubframeHeader);

TypeId
AmpduSubframeHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::AmpduSubframeHeader")
    .SetParent<Header> ()
    .SetGroupName ("Wifi")
    .AddConstructor<AmpduSubframeHeader> ()
  ;
  return tid;
}

TypeId
AmpduSubframeHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

AmpduSubframeHeader::AmpduSubframeHeader ()
  : m_length (0),
    m_eof (0),
    m_signature (0x4E) // Per 802.11 standard, the unique pattern is set to the value 0x4E.
{
}

AmpduSubframeHeader::~AmpduSubframeHeader ()
{
}

uint32_t
AmpduSubframeHeader::GetSerializedSize () const
{
  return (2 + 1 + 1);
}

void
AmpduSubframeHeader::Serialize (Buffer::Iterator i) const
{
  i.WriteHtolsbU16 ((m_eof << 15) | m_length);
  i.WriteU8 (1); //not used, CRC always set to 1
  i.WriteU8 (m_signature);
}

uint32_t
AmpduSubframeHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint16_t field = i.ReadLsbtohU16 ();
  m_eof = (field & 0x8000) >> 15;
  m_length = (field & 0x3fff);
  i.ReadU8 (); //CRC
  m_signature = i.ReadU8 (); //SIG
  return i.GetDistanceFrom (start);
}

void
AmpduSubframeHeader::Print (std::ostream &os) const
{
  os << "EOF = " << m_eof << ", length = " << m_length
     << ", signature = 0x" << std::hex << m_signature;
}

void
AmpduSubframeHeader::SetLength (uint16_t length)
{
  m_length = length;
}

void
AmpduSubframeHeader::SetEof (bool eof)
{
  m_eof = eof;
}

uint16_t
AmpduSubframeHeader::GetLength (void) const
{
  return m_length;
}

bool
AmpduSubframeHeader::GetEof (void) const
{
  return m_eof;
}

bool
AmpduSubframeHeader::IsSignatureValid (void) const
{
  return m_signature == 0x4E;
}

} //namespace ns3
