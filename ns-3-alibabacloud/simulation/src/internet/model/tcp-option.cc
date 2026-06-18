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

#include "tcp-option.h"
#include "tcp-option-rfc793.h"
#include "tcp-option-winscale.h"
#include "tcp-option-ts.h"
#include "tcp-option-sack-permitted.h"
#include "tcp-option-sack.h"

#include "ns3/type-id.h"
#include "ns3/log.h"

#include <vector>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpOption");

NS_OBJECT_ENSURE_REGISTERED (TcpOption);


TcpOption::TcpOption ()
{
}

TcpOption::~TcpOption ()
{
}

TypeId
TcpOption::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpOption")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId
TcpOption::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

Ptr<TcpOption>
TcpOption::CreateOption (uint8_t kind)
{
  struct kindToTid
  {
    TcpOption::Kind kind;
    TypeId tid;
  };

  static kindToTid toTid[] =
  {
    { TcpOption::END,           TcpOptionEnd::GetTypeId () },
    { TcpOption::MSS,           TcpOptionMSS::GetTypeId () },
    { TcpOption::NOP,           TcpOptionNOP::GetTypeId () },
    { TcpOption::TS,            TcpOptionTS::GetTypeId () },
    { TcpOption::WINSCALE,      TcpOptionWinScale::GetTypeId () },
    { TcpOption::SACKPERMITTED, TcpOptionSackPermitted::GetTypeId () },
    { TcpOption::SACK,          TcpOptionSack::GetTypeId () },
    { TcpOption::UNKNOWN,  TcpOptionUnknown::GetTypeId () }
  };

  for (unsigned int i = 0; i < sizeof (toTid) / sizeof (kindToTid); ++i)
    {
      if (toTid[i].kind == kind)
        {
          ObjectFactory objectFactory;
          objectFactory.SetTypeId (toTid[i].tid);
          return objectFactory.Create<TcpOption> ();
        }
    }

  return CreateObject<TcpOptionUnknown> ();
}

bool
TcpOption::IsKindKnown (uint8_t kind)
{
  switch (kind)
    {
    case END:
    case NOP:
    case MSS:
    case WINSCALE:
    case SACKPERMITTED:
    case SACK:
    case TS:
      // Do not add UNKNOWN here
      return true;
    }

  return false;
}

NS_OBJECT_ENSURE_REGISTERED (TcpOptionUnknown);

TcpOptionUnknown::TcpOptionUnknown ()
  : TcpOption ()
{
  m_kind = 0xFF;
  m_size = 0;
}

TcpOptionUnknown::~TcpOptionUnknown ()
{
}

TypeId
TcpOptionUnknown::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpOptionUnknown")
    .SetParent<TcpOption> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpOptionUnknown> ()
  ;
  return tid;
}

TypeId
TcpOptionUnknown::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
TcpOptionUnknown::Print (std::ostream &os) const
{
  os << "Unknown option";
}

uint32_t
TcpOptionUnknown::GetSerializedSize (void) const
{
  return m_size;
}

void
TcpOptionUnknown::Serialize (Buffer::Iterator i) const
{
  if (m_size == 0)
    {
      NS_LOG_WARN ("Can't Serialize an Unknown Tcp Option");
      return;
    }

  i.WriteU8 (GetKind ());
  i.WriteU8 (static_cast<uint8_t> (GetSerializedSize ()));
  i.Write (m_content, m_size - 2);
}

uint32_t
TcpOptionUnknown::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_kind = i.ReadU8 ();
  NS_LOG_WARN ("Trying to Deserialize an Unknown Option of Kind " << int (m_kind));

  m_size = i.ReadU8 ();
  if (m_size < 2 || m_size > 40)
    {
      NS_LOG_WARN ("Unable to parse an unknown option of kind " << int (m_kind) << " with apparent size " << int (m_size));
      return 0;
    }

  i.Read (m_content, m_size - 2);

  return m_size;
}

uint8_t
TcpOptionUnknown::GetKind (void) const
{
  return m_kind;
}

} // namespace ns3
