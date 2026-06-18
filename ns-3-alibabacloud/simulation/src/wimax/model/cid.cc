/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#include "cid.h"

// 0 will match IR CID, -1 will match broadcast CID 0xFFFF, hence 60000
#define CID_UNINITIALIZED 60000

namespace ns3 {

Cid::Cid (void)
{
  m_identifier = CID_UNINITIALIZED;
}

Cid::Cid (uint16_t identifier)
{
  m_identifier = identifier;
}

Cid::~Cid (void)
{
}

uint16_t
Cid::GetIdentifier (void) const
{
  return m_identifier;
}

bool
Cid::IsMulticast (void) const
{
  return m_identifier >= 0xff00 && m_identifier <= 0xfffd;
}
bool
Cid::IsBroadcast (void) const
{
  return *this == Broadcast ();
}
bool
Cid::IsPadding (void) const
{
  return *this == Padding ();
}
bool
Cid::IsInitialRanging (void) const
{
  return *this == InitialRanging ();
}

Cid
Cid::Broadcast (void)
{
  return 0xffff;
}
Cid
Cid::Padding (void)
{
  return 0xfffe;
}
Cid
Cid::InitialRanging (void)
{
  return 0;
}

/**
 * \brief equality operator
 * \param lhs left hand side
 * \param rhs right hand side
 * \returns true if equal
 */
bool operator == (const Cid &lhs,
                  const Cid &rhs)
{
  return lhs.m_identifier == rhs.m_identifier;
}

/**
 * \brief inequality operator
 * \param lhs left hand side
 * \param rhs right hand side
 * \returns true if not equal
 */
bool operator != (const Cid &lhs,
                  const Cid &rhs)
{
  return !(lhs == rhs);
}

/**
 * \brief output stream output opertor
 * \param os output stream
 * \param cid CID
 * \returns output stream
 */
std::ostream & operator << (std::ostream &os, const Cid &cid)
{
  os << cid.GetIdentifier ();
  return os;
}

} // namespace ns3
