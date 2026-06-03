/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Telecom Bretagne 
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
 * Author: Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 */

#include "ns3/log.h"
#include "ns3/node.h"

#include "ipv6-l3-protocol.h"
#include "ipv6-autoconfigured-prefix.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("Ipv6AutoconfiguredPrefix");

uint32_t Ipv6AutoconfiguredPrefix::m_prefixId = 0;

Ipv6AutoconfiguredPrefix::Ipv6AutoconfiguredPrefix (Ptr<Node> node, uint32_t interface, Ipv6Address prefix, Ipv6Prefix mask, uint32_t preferredLifeTime, uint32_t validLifeTime, Ipv6Address router)
{
  if (preferredLifeTime+validLifeTime == 0)
    {
      NS_LOG_WARN ("Ipv6AutoconfiguredPrefix: Preferred and Valid Lifetimes are zero, the address will be removed immediately.");
    }
  m_node = node;
  m_interface = interface;
  m_validLifeTime = validLifeTime;
  m_preferredLifeTime = preferredLifeTime;
  m_id = m_prefixId;
  m_prefixId++;
  m_preferred = false;
  m_valid = false;
  m_prefix = prefix;
  m_mask = mask;
  m_defaultGatewayRouter = router;
}

Ipv6AutoconfiguredPrefix::~Ipv6AutoconfiguredPrefix ()
{
}

void Ipv6AutoconfiguredPrefix::SetDefaultGatewayRouter (Ipv6Address router)
{
  m_defaultGatewayRouter = router;
}

Ipv6Address Ipv6AutoconfiguredPrefix::GetDefaultGatewayRouter () const
{
  return m_defaultGatewayRouter;
}

void Ipv6AutoconfiguredPrefix::SetInterface (uint32_t interface)
{
  m_interface = interface;
}

uint32_t Ipv6AutoconfiguredPrefix::GetInterface () const
{
  return m_interface;
}

void Ipv6AutoconfiguredPrefix::SetPreferredLifeTime (uint32_t t)
{
  m_preferredLifeTime = t;
  if (m_preferredLifeTime+m_validLifeTime == 0)
    {
      NS_LOG_WARN ("Ipv6AutoconfiguredPrefix: Preferred and Valid Lifetimes are zero, the address will be removed immediately.");
    }
}

uint32_t Ipv6AutoconfiguredPrefix::GetPreferredLifeTime () const
{
  return m_preferredLifeTime;
}

void Ipv6AutoconfiguredPrefix::SetValidLifeTime (uint32_t t)
{
  m_validLifeTime = t;
  if (m_preferredLifeTime+m_validLifeTime == 0)
    {
      NS_LOG_WARN ("Ipv6AutoconfiguredPrefix: Preferred and Valid Lifetimes are zero, the address will be removed immediately.");
    }
}

uint32_t Ipv6AutoconfiguredPrefix::GetValidLifeTime () const
{
  return m_validLifeTime;
}

void Ipv6AutoconfiguredPrefix::MarkPreferredTime ()
{
  m_preferred = true;
}

void Ipv6AutoconfiguredPrefix::MarkValidTime ()
{
  m_preferred = false;
  m_valid = true;
}

void Ipv6AutoconfiguredPrefix::FunctionPreferredTimeout ()
{
  NS_LOG_INFO ("Preferred Time expired for " << m_prefix);
  m_preferred = false;
  MarkValidTime ();
  StartValidTimer ();
}

void Ipv6AutoconfiguredPrefix::FunctionValidTimeout ()
{
  NS_LOG_INFO ("Valid Time expired for " << m_prefix);
  m_valid = false;
  RemoveMe ();
}

void Ipv6AutoconfiguredPrefix::StartPreferredTimer ()
{
  if (m_preferredLifeTime != 0xffffffff)
    {
      NS_LOG_INFO ("Start PreferredTimer for " << m_prefix);
      m_preferredTimer.SetFunction (&Ipv6AutoconfiguredPrefix::FunctionPreferredTimeout, this);
      m_preferredTimer.SetDelay (Seconds (m_preferredLifeTime));
      m_preferredTimer.Schedule ();
    }
}

void Ipv6AutoconfiguredPrefix::StartValidTimer ()
{
  if (m_validLifeTime != 0xffffffff)
    {
      NS_LOG_INFO ("Start ValidTimer for " << m_prefix);
      m_validTimer.SetFunction (&Ipv6AutoconfiguredPrefix::FunctionValidTimeout, this);
      m_validTimer.SetDelay (Seconds (m_validLifeTime - m_preferredLifeTime));
      m_validTimer.Schedule ();
    }
}

void Ipv6AutoconfiguredPrefix::StopPreferredTimer ()
{
  if (m_preferredTimer.IsRunning())
    {
      NS_LOG_INFO ("Stop PreferredTimer for " << m_prefix);
      m_preferredTimer.Cancel ();
    }
}

void Ipv6AutoconfiguredPrefix::StopValidTimer ()
{
  if (m_validTimer.IsRunning())
    {
      NS_LOG_INFO ("Stop ValidTimer for " << m_prefix);
      m_validTimer.Cancel ();
    }
}

void Ipv6AutoconfiguredPrefix::RemoveMe ()
{
  NS_LOG_INFO ("The prefix " << m_prefix << " will be removed on interface " << m_interface);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  ipv6->RemoveAutoconfiguredAddress (m_interface, m_prefix, m_mask, m_defaultGatewayRouter);
}

void Ipv6AutoconfiguredPrefix::SetPreferred ()
{
  m_preferred = true;
}

void Ipv6AutoconfiguredPrefix::SetValid ()
{
  m_preferred = false;
  m_valid = true;
}

uint32_t Ipv6AutoconfiguredPrefix::GetId () const
{ 
  return m_id;
}

bool Ipv6AutoconfiguredPrefix::IsPreferred () const
{
  return m_preferred;
}

bool Ipv6AutoconfiguredPrefix::IsValid () const
{
  return m_valid;
}

Ipv6Address Ipv6AutoconfiguredPrefix::GetPrefix () const
{
  return m_prefix;
}

void Ipv6AutoconfiguredPrefix::SetPrefix (Ipv6Address prefix)
{
  m_prefix = prefix;
}

Ipv6Prefix Ipv6AutoconfiguredPrefix::GetMask () const
{
  return m_mask;
}

void Ipv6AutoconfiguredPrefix::SetMask (Ipv6Prefix mask)
{
  m_mask = mask;
}

} /* namespace ns3 */

