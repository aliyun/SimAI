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
 * Documentation, test cases: Natale Patriciello <natale.patriciello@gmail.com>
 */

#include "tcp-option-winscale.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpOptionWinScale");

NS_OBJECT_ENSURE_REGISTERED (TcpOptionWinScale);

TcpOptionWinScale::TcpOptionWinScale ()
  : TcpOption (),
    m_scale (0)
{
}

TcpOptionWinScale::~TcpOptionWinScale ()
{
}

TypeId
TcpOptionWinScale::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpOptionWinScale")
    .SetParent<TcpOption> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpOptionWinScale> ()
  ;
  return tid;
}

TypeId
TcpOptionWinScale::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
TcpOptionWinScale::Print (std::ostream &os) const
{
  os << static_cast<int> (m_scale);
}

uint32_t
TcpOptionWinScale::GetSerializedSize (void) const
{
  return 3;
}

void
TcpOptionWinScale::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (GetKind ()); // Kind
  i.WriteU8 (3); // Length
  i.WriteU8 (m_scale); // Max segment size
}

uint32_t
TcpOptionWinScale::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  uint8_t readKind = i.ReadU8 ();
  if (readKind != GetKind ())
    {
      NS_LOG_WARN ("Malformed Window Scale option");
      return 0;
    }
  uint8_t size = i.ReadU8 ();
  if (size != 3)
    {
      NS_LOG_WARN ("Malformed Window Scale option");
      return 0;
    }
  m_scale = i.ReadU8 ();
  return GetSerializedSize ();
}

uint8_t
TcpOptionWinScale::GetKind (void) const
{
  return TcpOption::WINSCALE;
}

uint8_t
TcpOptionWinScale::GetScale (void) const
{
  NS_ASSERT (m_scale <= 14);

  return m_scale;
}

void
TcpOptionWinScale::SetScale (uint8_t scale)
{
  NS_ASSERT (scale <= 14);

  m_scale = scale;
}

} // namespace ns3
