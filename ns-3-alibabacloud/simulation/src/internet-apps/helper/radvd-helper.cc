/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "radvd-helper.h"
#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/radvd.h"
#include "ns3/radvd-interface.h"
#include "ns3/radvd-prefix.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("RadvdHelper");

RadvdHelper::RadvdHelper ()
{
  m_factory.SetTypeId (Radvd::GetTypeId ());
}

void RadvdHelper::AddAnnouncedPrefix (uint32_t interface, Ipv6Address prefix, uint32_t prefixLength)
{
  NS_LOG_FUNCTION(this << int(interface) << prefix << int(prefixLength));
  if (prefixLength != 64)
    {
      NS_LOG_WARN("Adding a non-64 prefix is generally a bad idea. Autoconfiguration might not work.");
    }

  bool prefixFound = false;
  if (m_radvdInterfaces.find(interface) == m_radvdInterfaces.end())
    {
      m_radvdInterfaces[interface] = Create<RadvdInterface> (interface);
    }
  else
    {
      RadvdInterface::RadvdPrefixList prefixList = m_radvdInterfaces[interface]->GetPrefixes();
      RadvdInterface::RadvdPrefixListCI iter;
      for (iter=prefixList.begin(); iter!=prefixList.end(); iter++)
        {
          if ((*iter)->GetNetwork() == prefix)
            {
              NS_LOG_LOGIC("Not adding the same prefix twice, skipping " << prefix << " " << int(prefixLength));
              prefixFound = true;
              break;
            }
        }
    }
  if (!prefixFound)
    {
      Ptr<RadvdPrefix> routerPrefix = Create<RadvdPrefix> (prefix, prefixLength);
      m_radvdInterfaces[interface]->AddPrefix(routerPrefix);
    }
}

void RadvdHelper::EnableDefaultRouterForInterface (uint32_t interface)
{
  if (m_radvdInterfaces.find(interface) == m_radvdInterfaces.end())
    {
      m_radvdInterfaces[interface] = Create<RadvdInterface> (interface);
    }
  uint32_t maxRtrAdvInterval = m_radvdInterfaces[interface]->GetMaxRtrAdvInterval();
  m_radvdInterfaces[interface]->SetDefaultLifeTime(3*maxRtrAdvInterval/1000);
}

void RadvdHelper::DisableDefaultRouterForInterface (uint32_t interface)
{
  if (m_radvdInterfaces.find(interface) == m_radvdInterfaces.end())
    {
      m_radvdInterfaces[interface] = Create<RadvdInterface> (interface);
    }
  m_radvdInterfaces[interface]->SetDefaultLifeTime(0);
}

Ptr<RadvdInterface> RadvdHelper::GetRadvdInterface (uint32_t interface)
{
  if (m_radvdInterfaces.find(interface) == m_radvdInterfaces.end())
    {
      m_radvdInterfaces[interface] = Create<RadvdInterface> (interface);
    }
  return m_radvdInterfaces[interface];
}

void RadvdHelper::ClearPrefixes()
{
  m_radvdInterfaces.clear();
}

void RadvdHelper::SetAttribute (std::string name, const AttributeValue& value)
{
  m_factory.Set (name, value);
}

ApplicationContainer RadvdHelper::Install (Ptr<Node> node)
{
  ApplicationContainer apps;
  Ptr<Radvd> radvd = m_factory.Create<Radvd> ();
  for (RadvdInterfaceMapI iter = m_radvdInterfaces.begin(); iter != m_radvdInterfaces.end(); iter ++)
    {
      if (!iter->second->GetPrefixes().empty())
        {
          radvd->AddConfiguration(iter->second);
        }
    }
  node->AddApplication (radvd);
  apps.Add (radvd);
  return apps;
}

} /* namespace ns3 */

