/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Adrian Sai-wah Tam
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
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */

#include "tcp-option-ts.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpOptionTS");

NS_OBJECT_ENSURE_REGISTERED (TcpOptionTS);

TcpOptionTS::TcpOptionTS ()
  : TcpOption (),
    m_timestamp (0),
    m_echo (0)
{
}

TcpOptionTS::~TcpOptionTS ()
{
}

TypeId
TcpOptionTS::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpOptionTS")
    .SetParent<TcpOption> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpOptionTS> ()
  ;
  return tid;
}

TypeId
TcpOptionTS::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
TcpOptionTS::Print (std::ostream &os) const
{
  os << m_timestamp << ";" << m_echo;
}

uint32_t
TcpOptionTS::GetSerializedSize (void) const
{
  return 10;
}

void
TcpOptionTS::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (GetKind ()); // Kind
  i.WriteU8 (10); // Length
  i.WriteHtonU32 (m_timestamp); // Local timestamp
  i.WriteHtonU32 (m_echo); // Echo timestamp
}

uint32_t
TcpOptionTS::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  uint8_t readKind = i.ReadU8 ();
  if (readKind != GetKind ())
    {
      NS_LOG_WARN ("Malformed Timestamp option");
      return 0;
    }

  uint8_t size = i.ReadU8 ();
  if (size != 10)
    {
      NS_LOG_WARN ("Malformed Timestamp option");
      return 0;
    }
  m_timestamp = i.ReadNtohU32 ();
  m_echo = i.ReadNtohU32 ();
  return GetSerializedSize ();
}

uint8_t
TcpOptionTS::GetKind (void) const
{
  return TcpOption::TS;
}

uint32_t
TcpOptionTS::GetTimestamp (void) const
{
  return m_timestamp;
}

uint32_t
TcpOptionTS::GetEcho (void) const
{
  return m_echo;
}

void
TcpOptionTS::SetTimestamp (uint32_t ts)
{
  m_timestamp = ts;
}

void
TcpOptionTS::SetEcho (uint32_t ts)
{
  m_echo = ts;
}

uint32_t
TcpOptionTS::NowToTsValue ()
{
  uint64_t now = (uint64_t) Simulator::Now ().GetMilliSeconds ();

  // high: (now & 0xFFFFFFFF00000000ULL) >> 32;
  // low: now & 0xFFFFFFFF
  return (now & 0xFFFFFFFF);
}

Time
TcpOptionTS::ElapsedTimeFromTsValue (uint32_t echoTime)
{
  uint64_t now64 = (uint64_t) Simulator::Now ().GetMilliSeconds ();
  uint32_t now32 = now64 & 0xFFFFFFFF;

  Time ret = Seconds (0.0);
  if (now32 > echoTime)
    {
      ret = MilliSeconds (now32 - echoTime);
    }

  return ret;
}

} // namespace ns3
