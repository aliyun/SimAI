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


#include <stdint.h>
#include "cid-factory.h"
#include "ns3/uinteger.h"

namespace ns3 {

CidFactory::CidFactory (void)
  : m_m (0x5500),
    // this is an arbitrary default
    m_basicIdentifier (1),
    m_primaryIdentifier (m_m + 1),
    m_transportOrSecondaryIdentifier (2 * m_m + 1),
    m_multicastPollingIdentifier (0xff00)
{
}

Cid
CidFactory::AllocateBasic (void)
{
  NS_ASSERT (m_basicIdentifier < m_m);
  m_basicIdentifier++;
  return Cid (m_basicIdentifier);
}

Cid
CidFactory::AllocatePrimary (void)
{
  NS_ASSERT (m_primaryIdentifier < 2 * m_m);
  m_primaryIdentifier++;
  return Cid (m_primaryIdentifier);
}

Cid
CidFactory::AllocateTransportOrSecondary (void)
{
  NS_ASSERT (m_transportOrSecondaryIdentifier < 0xfefe);
  m_transportOrSecondaryIdentifier++;
  return Cid (m_transportOrSecondaryIdentifier);
}

Cid
CidFactory::AllocateMulticast (void)
{
  NS_ASSERT (m_multicastPollingIdentifier < 0xfffd);
  m_multicastPollingIdentifier++;
  return Cid (m_multicastPollingIdentifier);
}


Cid
CidFactory::Allocate (enum Cid::Type type)
{
  switch (type)
    {
    case Cid::BROADCAST:
      return Cid::Broadcast ();
    case Cid::INITIAL_RANGING:
      return Cid::InitialRanging ();
    case Cid::BASIC:
      return AllocateBasic ();
    case Cid::PRIMARY:
      return AllocatePrimary ();
    case Cid::TRANSPORT:
      return AllocateTransportOrSecondary ();
    case Cid::MULTICAST:
      return AllocateMulticast ();
    case Cid::PADDING:
      return Cid::Padding ();
    default:
      NS_FATAL_ERROR ("Cannot be reached");
      return 0; // quiet compiler
    }
}

bool
CidFactory::IsTransport (Cid cid) const
{
  int id = cid.m_identifier;
  return (id - 2 * m_m > 0) && (id <= 0xfefe);
}
bool
CidFactory::IsPrimary (Cid cid) const
{
  int id = cid.m_identifier;
  return (id - m_m > 0) && (id <= 2 * m_m);
}
bool
CidFactory::IsBasic (Cid cid) const
{
  uint16_t id = cid.m_identifier;
  return id >= 1 && id <= m_m;
}


void
CidFactory::FreeCid (Cid cid)
{
  /// \todo We need to update the cid bitmap properly here.
  NS_FATAL_ERROR ("TODO: Update the cid bitmap properly here-- please implement and contribute a patch");
}

} // namespace ns3


