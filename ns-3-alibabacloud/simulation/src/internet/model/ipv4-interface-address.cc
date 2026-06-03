/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#include "ns3/log.h"
#include "ns3/assert.h"
#include "ipv4-interface-address.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4InterfaceAddress");

Ipv4InterfaceAddress::Ipv4InterfaceAddress ()
  : m_scope (GLOBAL), 
    m_secondary (false)
{
  NS_LOG_FUNCTION (this);
}

Ipv4InterfaceAddress::Ipv4InterfaceAddress (Ipv4Address local, Ipv4Mask mask)
  : m_scope (GLOBAL), 
    m_secondary (false)
{
  NS_LOG_FUNCTION (this << local << mask);
  m_local = local;
  if (m_local == Ipv4Address::GetLoopback ())
    {
      m_scope = HOST;
    }
  m_mask = mask;
  m_broadcast = Ipv4Address (local.Get () | (~mask.Get ()));
}

Ipv4InterfaceAddress::Ipv4InterfaceAddress (const Ipv4InterfaceAddress &o)
  : m_local (o.m_local),
    m_mask (o.m_mask),
    m_broadcast (o.m_broadcast),
    m_scope (o.m_scope),
    m_secondary (o.m_secondary)
{
  NS_LOG_FUNCTION (this << &o);
}

void
Ipv4InterfaceAddress::SetLocal (Ipv4Address local)
{
  NS_LOG_FUNCTION (this << local);
  m_local = local;
}

void
Ipv4InterfaceAddress::SetAddress (Ipv4Address address)
{
  SetLocal (address);
}

Ipv4Address
Ipv4InterfaceAddress::GetLocal (void) const
{
  NS_LOG_FUNCTION (this);
  return m_local;
}

Ipv4Address
Ipv4InterfaceAddress::GetAddress (void) const
{
  return GetLocal ();
}

void 
Ipv4InterfaceAddress::SetMask (Ipv4Mask mask) 
{
  NS_LOG_FUNCTION (this << mask);
  m_mask = mask;
}

Ipv4Mask 
Ipv4InterfaceAddress::GetMask (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mask;
}

void 
Ipv4InterfaceAddress::SetBroadcast (Ipv4Address broadcast)
{
  NS_LOG_FUNCTION (this << broadcast);
  m_broadcast = broadcast;
}

Ipv4Address 
Ipv4InterfaceAddress::GetBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  return m_broadcast;
}

void 
Ipv4InterfaceAddress::SetScope (Ipv4InterfaceAddress::InterfaceAddressScope_e scope)
{
  NS_LOG_FUNCTION (this << scope);
  m_scope = scope;
}

Ipv4InterfaceAddress::InterfaceAddressScope_e 
Ipv4InterfaceAddress::GetScope (void) const
{
  NS_LOG_FUNCTION (this);
  return m_scope;
}

bool Ipv4InterfaceAddress::IsInSameSubnet (const Ipv4Address b) const
{
  Ipv4Address aAddr = m_local;
  aAddr = aAddr.CombineMask(m_mask);
  Ipv4Address bAddr = b;
  bAddr = bAddr.CombineMask(m_mask);

  return (aAddr == bAddr);
}

bool 
Ipv4InterfaceAddress::IsSecondary (void) const
{
  NS_LOG_FUNCTION (this);
  return m_secondary;
}

void 
Ipv4InterfaceAddress::SetSecondary (void)
{
  NS_LOG_FUNCTION (this);
  m_secondary = true;
}

void 
Ipv4InterfaceAddress::SetPrimary (void)
{
  NS_LOG_FUNCTION (this);
  m_secondary = false;
}

std::ostream& operator<< (std::ostream& os, const Ipv4InterfaceAddress &addr)
{ 
  os << "m_local=" << addr.GetLocal () << "; m_mask=" <<
  addr.GetMask () << "; m_broadcast=" << addr.GetBroadcast () << "; m_scope=" << addr.GetScope () <<
  "; m_secondary=" << addr.IsSecondary ();
  return os;
} 

} // namespace ns3
