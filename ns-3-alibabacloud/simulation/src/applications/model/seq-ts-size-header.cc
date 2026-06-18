/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA
 * Copyright (c) 2018 Natale Patriciello <natale.patriciello@gmail.com>
 *                    (added timestamp and size fields)
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
 */

#include "ns3/log.h"
#include "seq-ts-size-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SeqTsSizeHeader");

NS_OBJECT_ENSURE_REGISTERED (SeqTsSizeHeader);

SeqTsSizeHeader::SeqTsSizeHeader ()
  : SeqTsHeader ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
SeqTsSizeHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SeqTsSizeHeader")
    .SetParent<SeqTsHeader> ()
    .SetGroupName ("Applications")
    .AddConstructor<SeqTsSizeHeader> ()
  ;
  return tid;
}

TypeId
SeqTsSizeHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
SeqTsSizeHeader::SetSize (uint64_t size)
{
  m_size = size;
}

uint64_t
SeqTsSizeHeader::GetSize (void) const
{
  return m_size;
}

void
SeqTsSizeHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "(size=" << m_size << ") AND ";
  SeqTsHeader::Print (os);
}

uint32_t
SeqTsSizeHeader::GetSerializedSize (void) const
{
  return SeqTsHeader::GetSerializedSize () + 8;
}

void
SeqTsSizeHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.WriteHtonU64 (m_size);
  SeqTsHeader::Serialize (i);
}

uint32_t
SeqTsSizeHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  m_size = i.ReadNtohU64 ();
  SeqTsHeader::Deserialize (i);
  return GetSerializedSize ();
}

} // namespace ns3
