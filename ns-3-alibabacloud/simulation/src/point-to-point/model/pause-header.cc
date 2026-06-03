/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 New York University
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
 * Author: Adrian S.-W. Tam <adrian.sw.tam@gmail.com>
 */

#include <stdint.h>
#include <iostream>
#include "pause-header.h"
#include "ns3/buffer.h"
#include "ns3/address-utils.h"
#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE ("PauseHeader");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (PauseHeader);

PauseHeader::PauseHeader (uint32_t time, uint32_t qlen, uint8_t qindex)
  : m_time(time), m_qlen(qlen), m_qindex(qindex)
{
}

PauseHeader::PauseHeader ()
  : m_time(0)
{}

PauseHeader::~PauseHeader ()
{}

void PauseHeader::SetTime (uint32_t time)
{
  m_time = time;
}

uint32_t PauseHeader::GetTime () const
{
  return m_time;
}

void PauseHeader::SetQLen (uint32_t qlen)
{
  m_qlen = qlen;
}

uint32_t PauseHeader::GetQLen () const
{
  return m_qlen;
}

void PauseHeader::SetQIndex (uint8_t qindex)
{
  m_qindex = qindex;
}

uint8_t PauseHeader::GetQIndex () const
{
  return m_qindex;
}

TypeId
PauseHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PauseHeader")
    .SetParent<Header> ()
    .AddConstructor<PauseHeader> ()
    ;
  return tid;
}
TypeId
PauseHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
void PauseHeader::Print (std::ostream &os) const
{
  os << "pause=" << m_time << "us, qlen=" << m_qlen <<", qidx=" << int(m_qindex);
}
uint32_t PauseHeader::GetSerializedSize (void)  const
{
  return 9;
}
void PauseHeader::Serialize (Buffer::Iterator start)  const
{
  start.WriteU32 (m_time);
  start.WriteU32 (m_qlen);
  start.WriteU8 (m_qindex);
}

uint32_t PauseHeader::Deserialize (Buffer::Iterator start)
{
  m_time = start.ReadU32 ();
  m_qlen = start.ReadU32 ();
  m_qindex = start.ReadU8 ();
  return GetSerializedSize ();
}


}; // namespace ns3
