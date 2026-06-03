/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' di Firenze, Italy
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

#include "ns3/node.h"
#include "ns3/node-list.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/rip.h"
#include "rip-helper.h"

namespace ns3 {

RipHelper::RipHelper ()
{
  m_factory.SetTypeId ("ns3::Rip");
}

RipHelper::RipHelper (const RipHelper &o)
  : m_factory (o.m_factory)
{
  m_interfaceExclusions = o.m_interfaceExclusions;
  m_interfaceMetrics = o.m_interfaceMetrics;
}

RipHelper::~RipHelper ()
{
  m_interfaceExclusions.clear ();
  m_interfaceMetrics.clear ();
}

RipHelper*
RipHelper::Copy (void) const
{
  return new RipHelper (*this);
}

Ptr<Ipv4RoutingProtocol>
RipHelper::Create (Ptr<Node> node) const
{
  Ptr<Rip> rip = m_factory.Create<Rip> ();

  std::map<Ptr<Node>, std::set<uint32_t> >::const_iterator it = m_interfaceExclusions.find (node);

  if(it != m_interfaceExclusions.end ())
    {
      rip->SetInterfaceExclusions (it->second);
    }

  std::map< Ptr<Node>, std::map<uint32_t, uint8_t> >::const_iterator iter = m_interfaceMetrics.find (node);

  if(iter != m_interfaceMetrics.end ())
    {
      std::map<uint32_t, uint8_t>::const_iterator subiter;
      for (subiter = iter->second.begin (); subiter != iter->second.end (); subiter++)
        {
          rip->SetInterfaceMetric (subiter->first, subiter->second);
        }
    }

  node->AggregateObject (rip);
  return rip;
}

void
RipHelper::Set (std::string name, const AttributeValue &value)
{
  m_factory.Set (name, value);
}


int64_t
RipHelper::AssignStreams (NodeContainer c, int64_t stream)
{
  int64_t currentStream = stream;
  Ptr<Node> node;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      node = (*i);
      Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
      NS_ASSERT_MSG (ipv4, "Ipv4 not installed on node");
      Ptr<Ipv4RoutingProtocol> proto = ipv4->GetRoutingProtocol ();
      NS_ASSERT_MSG (proto, "Ipv4 routing not installed on node");
      Ptr<Rip> rip = DynamicCast<Rip> (proto);
      if (rip)
        {
          currentStream += rip->AssignStreams (currentStream);
          continue;
        }
      // RIP may also be in a list
      Ptr<Ipv4ListRouting> list = DynamicCast<Ipv4ListRouting> (proto);
      if (list)
        {
          int16_t priority;
          Ptr<Ipv4RoutingProtocol> listProto;
          Ptr<Rip> listRip;
          for (uint32_t i = 0; i < list->GetNRoutingProtocols (); i++)
            {
              listProto = list->GetRoutingProtocol (i, priority);
              listRip = DynamicCast<Rip> (listProto);
              if (listRip)
                {
                  currentStream += listRip->AssignStreams (currentStream);
                  break;
                }
            }
        }
    }
  return (currentStream - stream);
}

void RipHelper::SetDefaultRouter (Ptr<Node> node, Ipv4Address nextHop, uint32_t interface)
{
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  NS_ASSERT_MSG (ipv4, "Ipv4 not installed on node");
  Ptr<Ipv4RoutingProtocol> proto = ipv4->GetRoutingProtocol ();
  NS_ASSERT_MSG (proto, "Ipv4 routing not installed on node");
  Ptr<Rip> rip = DynamicCast<Rip> (proto);
  if (rip)
    {
      rip->AddDefaultRouteTo (nextHop, interface);
    }
  // RIP may also be in a list
  Ptr<Ipv4ListRouting> list = DynamicCast<Ipv4ListRouting> (proto);
  if (list)
    {
      int16_t priority;
      Ptr<Ipv4RoutingProtocol> listProto;
      Ptr<Rip> listRip;
      for (uint32_t i = 0; i < list->GetNRoutingProtocols (); i++)
        {
          listProto = list->GetRoutingProtocol (i, priority);
          listRip = DynamicCast<Rip> (listProto);
          if (listRip)
            {
              listRip->AddDefaultRouteTo (nextHop, interface);
              break;
            }
        }
    }
}

void
RipHelper::ExcludeInterface (Ptr<Node> node, uint32_t interface)
{
  std::map< Ptr<Node>, std::set<uint32_t> >::iterator it = m_interfaceExclusions.find (node);

  if (it == m_interfaceExclusions.end ())
    {
      std::set<uint32_t> interfaces;
      interfaces.insert (interface);

      m_interfaceExclusions.insert (std::make_pair (node, interfaces));
    }
  else
    {
      it->second.insert (interface);
    }
}

void RipHelper::SetInterfaceMetric (Ptr<Node> node, uint32_t interface, uint8_t metric)
{
  m_interfaceMetrics[node][interface] = metric;
}

}

