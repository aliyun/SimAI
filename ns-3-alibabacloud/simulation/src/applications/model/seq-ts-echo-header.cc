/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 INRIA
 * Copyright (c) 2016 Universita' di Firenze (added echo fields)
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ns3/simulator.h"
#include "seq-ts-echo-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SeqTsEchoHeader");

NS_OBJECT_ENSURE_REGISTERED (SeqTsEchoHeader);

SeqTsEchoHeader::SeqTsEchoHeader ()
  : m_seq (0),
    m_tsValue (Simulator::Now ()),
    m_tsEchoReply (Seconds (0))
{
  NS_LOG_FUNCTION (this);
}

void
SeqTsEchoHeader::SetSeq (uint32_t seq)
{
  NS_LOG_FUNCTION (this << seq);
  m_seq = seq;
}

uint32_t
SeqTsEchoHeader::GetSeq (void) const
{
  NS_LOG_FUNCTION (this);
  return m_seq;
}

void
SeqTsEchoHeader::SetTsValue (Time ts)
{
  NS_LOG_FUNCTION (this << ts);
  m_tsValue = ts;
}

Time
SeqTsEchoHeader::GetTsValue (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tsValue;
}

void
SeqTsEchoHeader::SetTsEchoReply (Time ts)
{
  NS_LOG_FUNCTION (this << ts);
  m_tsEchoReply = ts;
}

Time
SeqTsEchoHeader::GetTsEchoReply (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tsEchoReply;
}

TypeId
SeqTsEchoHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SeqTsEchoHeader")
    .SetParent<Header> ()
    .SetGroupName ("Applications")
    .AddConstructor<SeqTsEchoHeader> ()
  ;
  return tid;
}

TypeId
SeqTsEchoHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
SeqTsEchoHeader::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "(seq=" << m_seq << " Tx time=" << m_tsValue.As (Time::S) << " Rx time=" << m_tsEchoReply.As (Time::S) << ")";
}

uint32_t
SeqTsEchoHeader::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 4+8+8;
}

void
SeqTsEchoHeader::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.WriteHtonU32 (m_seq);
  i.WriteHtonU64 (m_tsValue.GetTimeStep ());
  i.WriteHtonU64 (m_tsEchoReply.GetTimeStep ());
}

uint32_t
SeqTsEchoHeader::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  m_seq = i.ReadNtohU32 ();
  m_tsValue = TimeStep (i.ReadNtohU64 ());
  m_tsEchoReply = TimeStep (i.ReadNtohU64 ());
  return GetSerializedSize ();
}

} // namespace ns3
