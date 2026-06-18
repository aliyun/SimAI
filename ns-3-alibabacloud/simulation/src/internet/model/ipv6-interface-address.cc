/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include <iostream>

#include "ns3/log.h"
#include "ns3/assert.h"

#include "ipv6-interface-address.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("Ipv6InterfaceAddress");

Ipv6InterfaceAddress::Ipv6InterfaceAddress () 
  : m_address (Ipv6Address ()),
    m_prefix (Ipv6Prefix ()),
    m_state (TENTATIVE_OPTIMISTIC),
    m_scope (HOST),
    m_onLink (true),
    m_nsDadUid (0)
{
  NS_LOG_FUNCTION (this);
}

Ipv6InterfaceAddress::Ipv6InterfaceAddress (Ipv6Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_prefix = Ipv6Prefix (64);
  SetAddress (address);
  SetState (TENTATIVE_OPTIMISTIC);
  m_onLink = true;
  m_nsDadUid = 0;
}

Ipv6InterfaceAddress::Ipv6InterfaceAddress (Ipv6Address address, Ipv6Prefix prefix)
{
  NS_LOG_FUNCTION (this << address << prefix);
  m_prefix = prefix;
  SetAddress (address);
  SetState (TENTATIVE_OPTIMISTIC);
  m_onLink = true;
  m_nsDadUid = 0;
}

Ipv6InterfaceAddress::Ipv6InterfaceAddress (Ipv6Address address, Ipv6Prefix prefix, bool onLink)
{
  NS_LOG_FUNCTION (this << address << prefix << onLink);
  m_prefix = prefix;
  SetAddress (address);
  SetState (TENTATIVE_OPTIMISTIC);
  m_onLink = onLink;
  m_nsDadUid = 0;
}

Ipv6InterfaceAddress::Ipv6InterfaceAddress (const Ipv6InterfaceAddress& o) 
  : m_address (o.m_address),
    m_prefix (o.m_prefix),
    m_state (o.m_state),
    m_scope (o.m_scope),
    m_onLink (o.m_onLink),
    m_nsDadUid (o.m_nsDadUid)
{
}

Ipv6InterfaceAddress::~Ipv6InterfaceAddress ()
{
  NS_LOG_FUNCTION (this);
}

Ipv6Address Ipv6InterfaceAddress::GetAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}

void Ipv6InterfaceAddress::SetAddress (Ipv6Address address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;

  if (address.IsLocalhost ())
    {
      m_scope = HOST;
      /* localhost address is always /128 prefix */
      m_prefix = Ipv6Prefix (128);
    }
  else if (address.IsLinkLocal ())
    {
      m_scope = LINKLOCAL;
      /* link-local address is always /64 prefix */
      m_prefix = Ipv6Prefix (64);
    }
  else if (address.IsLinkLocalMulticast ())
    {
      m_scope = LINKLOCAL;
      /* link-local multicast address is always /16 prefix */
      m_prefix = Ipv6Prefix (16);
    }
  else
    {
      m_scope = GLOBAL;
    }
}

Ipv6Prefix Ipv6InterfaceAddress::GetPrefix () const
{
  NS_LOG_FUNCTION (this);
  return m_prefix;
}

void Ipv6InterfaceAddress::SetState (Ipv6InterfaceAddress::State_e state)
{
  NS_LOG_FUNCTION (this << state);
  m_state = state;
}

Ipv6InterfaceAddress::State_e Ipv6InterfaceAddress::GetState () const
{
  NS_LOG_FUNCTION (this);
  return m_state;
}

void Ipv6InterfaceAddress::SetScope (Ipv6InterfaceAddress::Scope_e scope)
{
  NS_LOG_FUNCTION (this << scope);
  m_scope = scope;
}

Ipv6InterfaceAddress::Scope_e Ipv6InterfaceAddress::GetScope () const
{
  NS_LOG_FUNCTION (this);
  return m_scope;
}

bool Ipv6InterfaceAddress::IsInSameSubnet (Ipv6Address b) const
{
  NS_LOG_FUNCTION (this);

  Ipv6Address aAddr = m_address;
  aAddr = aAddr.CombinePrefix (m_prefix);
  Ipv6Address bAddr = b;
  bAddr = bAddr.CombinePrefix (m_prefix);

  if (aAddr == bAddr)
    {
      return true;
    }

  if ((bAddr.IsLinkLocalMulticast () && aAddr.IsLinkLocal ()) ||
      (aAddr.IsLinkLocalMulticast () && bAddr.IsLinkLocal ()))
    {
      return true;
    }

  return false;
}

std::ostream& operator<< (std::ostream& os, const Ipv6InterfaceAddress &addr)
{
  os << "address: " << addr.GetAddress () << addr.GetPrefix () << "; scope: ";
  switch (addr.GetScope ())
  {
    case Ipv6InterfaceAddress::HOST:
      os << "HOST";
      break;
    case Ipv6InterfaceAddress::LINKLOCAL:
      os << "LINK-LOCAL";
      break;
    case Ipv6InterfaceAddress::GLOBAL:
      os << "GLOBAL";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

uint32_t Ipv6InterfaceAddress::GetNsDadUid () const
{
  NS_LOG_FUNCTION (this);
  return m_nsDadUid;
}

void Ipv6InterfaceAddress::SetNsDadUid (uint32_t nsDadUid)
{
  NS_LOG_FUNCTION (this << nsDadUid);
  m_nsDadUid = nsDadUid;
}

bool Ipv6InterfaceAddress::GetOnLink () const
{
  NS_LOG_FUNCTION (this);
  return m_onLink;
}

void Ipv6InterfaceAddress::SetOnLink (bool onLink)
{
  NS_LOG_FUNCTION (this << onLink);
  m_onLink = onLink;
}

#if 0
void Ipv6InterfaceAddress::StartDadTimer (Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << interface);
  m_dadTimer.SetFunction (&Icmpv6L4Protocol::FunctionDadTimeout);
  m_dadTimer.SetArguments (interface, m_address);
  m_dadTimer.Schedule (Seconds (1));
  m_dadId = Simulator::Schedule (Seconds (1.), &Icmpv6L4Protocol::FunctionDadTimeout, interface, m_address);
}

void Ipv6InterfaceAddress::StopDadTimer ()
{
  NS_LOG_FUNCTION (this);
  m_dadTimer.Cancel ();
  Simulator::Cancel (m_dadId);
}
#endif

} /* namespace ns3 */

