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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "ns3/assert.h"
#include "ns3/address-utils.h"
#include "dot11s-mac-header.h"
#include "ns3/packet.h"

namespace ns3 {
namespace dot11s {
/***********************************************************
 *  Here Mesh Mac Header functionality is defined.
 ***********************************************************/
TypeId
MeshHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dot11s::MeshHeader")
    .SetParent<Header> ()
    .SetGroupName ("Mesh")
    .AddConstructor<MeshHeader> ();
  return tid;
}
MeshHeader::MeshHeader () :
  m_meshFlags (0), m_meshTtl (0), m_meshSeqno (0), m_addr4 (Mac48Address ()), m_addr5 (Mac48Address ()),
  m_addr6 (Mac48Address ())
{
}
MeshHeader::~MeshHeader ()
{
}
TypeId
MeshHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}
void
MeshHeader::SetAddr4 (Mac48Address address)
{
  m_addr4 = address;
}
void
MeshHeader::SetAddr5 (Mac48Address address)
{
  m_addr5 = address;
}
void
MeshHeader::SetAddr6 (Mac48Address address)
{
  m_addr6 = address;
}
Mac48Address
MeshHeader::GetAddr4 () const
{
  return m_addr4;
}
Mac48Address
MeshHeader::GetAddr5 () const
{
  return m_addr5;
}
Mac48Address
MeshHeader::GetAddr6 () const
{
  return m_addr6;
}
void
MeshHeader::SetMeshSeqno (uint32_t seqno)
{
  m_meshSeqno = seqno;
}
uint32_t
MeshHeader::GetMeshSeqno () const
{
  return m_meshSeqno;
}
void
MeshHeader::SetMeshTtl (uint8_t TTL)
{
  m_meshTtl = TTL;
}
uint8_t
MeshHeader::GetMeshTtl () const
{
  return m_meshTtl;
}
void
MeshHeader::SetAddressExt (uint8_t value)
{
  NS_ASSERT (value <= 3);
  m_meshFlags |= 0x03 & value;
}
uint8_t
MeshHeader::GetAddressExt () const
{
  return (0x03 & m_meshFlags);
}
uint32_t
MeshHeader::GetSerializedSize () const
{
  return 6 + GetAddressExt () * 6;
}
void
MeshHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_meshFlags);
  i.WriteU8 (m_meshTtl);
  i.WriteHtolsbU32 (m_meshSeqno);
  uint8_t addresses_to_add = GetAddressExt ();
  //Writing Address extensions:
  if ((addresses_to_add == 1) || (addresses_to_add == 3))
    {
      WriteTo (i, m_addr4);
    }
  if (addresses_to_add > 1)
    {
      WriteTo (i, m_addr5);
    }
  if (addresses_to_add > 1)
    {
      WriteTo (i, m_addr6);
    }
}
uint32_t
MeshHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t addresses_to_read = 0;
  m_meshFlags = i.ReadU8 ();
  m_meshTtl = i.ReadU8 ();
  m_meshSeqno = i.ReadLsbtohU32 ();
  addresses_to_read = m_meshFlags & 0x03;
  if ((addresses_to_read == 1) || (addresses_to_read == 3))
    {
      ReadFrom (i, m_addr4);
    }
  if (addresses_to_read > 1)
    {
      ReadFrom (i, m_addr5);
    }
  if (addresses_to_read > 1)
    {
      ReadFrom (i, m_addr6);
    }
  return i.GetDistanceFrom (start);
}
void
MeshHeader::Print (std::ostream &os) const
{
  os << "flags=" << (uint16_t) m_meshFlags << ", ttl=" << (uint16_t) m_meshTtl 
     << ", seqno=" << m_meshSeqno << ", addr4=" << m_addr4 
     << ", addr5=" << m_addr5 << ", addr6=" << m_addr6;
}
bool
operator== (const MeshHeader & a, const MeshHeader & b)
{
  return ((a.m_meshFlags == b.m_meshFlags) && (a.m_meshTtl == b.m_meshTtl)
          && (a.m_meshSeqno == b.m_meshSeqno) && (a.m_addr4 == b.m_addr4) && (a.m_addr5 == b.m_addr5)
          && (a.m_addr6 == b.m_addr6));
}
} // namespace dot11s
} // namespace ns3
