/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ns3/simulator.h"
#include "simple-seq-ts-header.h"

NS_LOG_COMPONENT_DEFINE ("SimpleSeqTsHeader");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (SimpleSeqTsHeader);

SimpleSeqTsHeader::SimpleSeqTsHeader ()
  : m_seq (0)
{
	if (IntHeader::mode == 1)
		ih.ts = Simulator::Now().GetTimeStep();
}

void
SimpleSeqTsHeader::SetSeq (uint64_t seq)
{
  m_seq = seq;
}
uint64_t
SimpleSeqTsHeader::GetSeq (void) const
{
  return m_seq;
}

void
SimpleSeqTsHeader::SetPG (uint16_t pg)
{
	m_pg = pg;
}
uint16_t
SimpleSeqTsHeader::GetPG (void) const
{
	return m_pg;
}

Time
SimpleSeqTsHeader::GetTs (void) const
{
	NS_ASSERT_MSG(IntHeader::mode == 1, "SimpleSeqTsHeader cannot GetTs when IntHeader::mode != 1");
	return TimeStep (ih.ts);
}

TypeId
SimpleSeqTsHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SimpleSeqTsHeader")
    .SetParent<Header> ()
    .AddConstructor<SimpleSeqTsHeader> ()
  ;
  return tid;
}
TypeId
SimpleSeqTsHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
void
SimpleSeqTsHeader::Print (std::ostream &os) const
{
  //os << "(seq=" << m_seq << " time=" << TimeStep (m_ts).GetSeconds () << ")";
	//os << m_seq << " " << TimeStep (m_ts).GetSeconds () << " " << m_pg;
	os << m_seq << " " << m_pg;
}
uint32_t
SimpleSeqTsHeader::GetSerializedSize (void) const
{
	return GetHeaderSize();
}
uint32_t SimpleSeqTsHeader::GetHeaderSize(void){
	return sizeof(m_seq) + sizeof(m_pg) + IntHeader::GetStaticSize();
}

void
SimpleSeqTsHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteHtonU64 (m_seq);
  i.WriteHtonU16 (m_pg);

  // write IntHeader
  ih.Serialize(i);
}
uint32_t
SimpleSeqTsHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_seq = i.ReadNtohU64 ();
  m_pg =  i.ReadNtohU16 ();

  // read IntHeader
  ih.Deserialize(i);
  return GetSerializedSize ();
}

} // namespace ns3
