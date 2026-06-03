/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2013 Magister Solutions
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
 * Original work author (from packet-sink-helper.cc):
 * - Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *
 * Converted to 3GPP HTTP web browsing traffic models by:
 * - Budiarto Herman <budiarto.herman@magister.fi>
 *
 */

#include <ns3/names.h>
#include "three-gpp-http-helper.h"

namespace ns3 {


// 3GPP HTTP CLIENT HELPER /////////////////////////////////////////////////////////

ThreeGppHttpClientHelper::ThreeGppHttpClientHelper (const Address &address)
{
  m_factory.SetTypeId ("ns3::ThreeGppHttpClient");
  m_factory.Set ("RemoteServerAddress", AddressValue (address));
}

void
ThreeGppHttpClientHelper::SetAttribute (const std::string &name,
                                        const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
ThreeGppHttpClientHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
ThreeGppHttpClientHelper::Install (const std::string &nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
ThreeGppHttpClientHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
ThreeGppHttpClientHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Application> app = m_factory.Create<Application> ();
  node->AddApplication (app);

  return app;
}


// HTTP SERVER HELPER /////////////////////////////////////////////////////////

ThreeGppHttpServerHelper::ThreeGppHttpServerHelper (const Address &address)
{
  m_factory.SetTypeId ("ns3::ThreeGppHttpServer");
  m_factory.Set ("LocalAddress", AddressValue (address));
}

void
ThreeGppHttpServerHelper::SetAttribute (const std::string &name,
                                        const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
ThreeGppHttpServerHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
ThreeGppHttpServerHelper::Install (const std::string &nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
ThreeGppHttpServerHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
ThreeGppHttpServerHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<Application> app = m_factory.Create<Application> ();
  node->AddApplication (app);

  return app;
}


} // end of `namespace ns3`
