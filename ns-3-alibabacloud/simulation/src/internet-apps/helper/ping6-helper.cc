/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Strasbourg University
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

#include "ping6-helper.h"
#include "ns3/ping6.h"
#include "ns3/uinteger.h"


namespace ns3
{

Ping6Helper::Ping6Helper ()
  : m_ifIndex (0)
{
  m_factory.SetTypeId (Ping6::GetTypeId ());
}

void Ping6Helper::SetLocal (Ipv6Address ip)
{
  m_localIp = ip;
}

void Ping6Helper::SetRemote (Ipv6Address ip)
{
  m_remoteIp = ip;
}

void Ping6Helper::SetAttribute (std::string name, const AttributeValue& value)
{
  m_factory.Set (name, value);
}

ApplicationContainer Ping6Helper::Install (NodeContainer c)
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Ptr<Node> node = *i;
      Ptr<Ping6> client = m_factory.Create<Ping6> ();
      client->SetLocal (m_localIp);
      client->SetRemote (m_remoteIp);
      client->SetIfIndex (m_ifIndex);
      client->SetRouters (m_routers);
      node->AddApplication (client);
      apps.Add (client);
    }
  return apps;
}

void Ping6Helper::SetIfIndex (uint32_t ifIndex)
{
  m_ifIndex = ifIndex;
}

void Ping6Helper::SetRoutersAddress (std::vector<Ipv6Address> routers)
{
  m_routers = routers;
}

} /* namespace ns3 */

