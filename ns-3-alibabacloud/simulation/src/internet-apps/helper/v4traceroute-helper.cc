/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Ritsumeikan University, Shiga, Japan
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
 *
 * Author: Alberto Gallegos Ramonet <ramonet@fc.ritsumei.ac.jp>
 *
 */

#include "v4traceroute-helper.h"
#include "ns3/v4traceroute.h"
#include "ns3/names.h"


namespace ns3 {

V4TraceRouteHelper::V4TraceRouteHelper (Ipv4Address remote)
{
  m_factory.SetTypeId ("ns3::V4TraceRoute");
  m_factory.Set ("Remote", Ipv4AddressValue (remote));
}

void
V4TraceRouteHelper::SetAttribute (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}

ApplicationContainer
V4TraceRouteHelper::Install (Ptr<Node> node) const
{
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
V4TraceRouteHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return ApplicationContainer (InstallPriv (node));
}

ApplicationContainer
V4TraceRouteHelper::Install (NodeContainer c) const
{
  ApplicationContainer apps;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      apps.Add (InstallPriv (*i));
    }

  return apps;
}

Ptr<Application>
V4TraceRouteHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<V4TraceRoute> app = m_factory.Create<V4TraceRoute> ();
  node->AddApplication (app);

  return app;
}

void
V4TraceRouteHelper::PrintTraceRouteAt (Ptr<Node> node, Ptr<OutputStreamWrapper> stream)
{
  Ptr<V4TraceRoute> trace;

  for (uint32_t i = 0; i < node->GetNApplications (); ++i)
    {
      trace = node->GetApplication (i)->GetObject <V4TraceRoute> ();
      if (trace != NULL)
        {
          *stream->GetStream () << "Tracing Route from Node " << node->GetId () << "\n";
          trace->Print (stream);
          return;
        }
    }
  NS_ASSERT_MSG (false, "No V4TraceRoute application found in node " << node->GetId ());
}

} // namespace ns3
