/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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

#include "status-code.h"

namespace ns3 {

StatusCode::StatusCode ()
{
}

void
StatusCode::SetSuccess (void)
{
  m_code = 0;
}

void
StatusCode::SetFailure (void)
{
  m_code = 1;
}

bool
StatusCode::IsSuccess (void) const
{
  return (m_code == 0);
}

uint32_t
StatusCode::GetSerializedSize (void) const
{
  return 2;
}

Buffer::Iterator
StatusCode::Serialize (Buffer::Iterator start) const
{
  start.WriteHtolsbU16 (m_code);
  return start;
}

Buffer::Iterator
StatusCode::Deserialize (Buffer::Iterator start)
{
  m_code = start.ReadLsbtohU16 ();
  return start;
}

std::ostream &
operator << (std::ostream &os, const StatusCode &code)
{
  if (code.IsSuccess ())
    {
      os << "success";
    }
  else
    {
      os << "failure";
    }
  return os;
}

} //namespace ns3
