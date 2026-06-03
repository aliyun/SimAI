/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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

#include "ns3/node.h"
#include "ns3/node-list.h"
#include "ns3/simulator.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/ipv6-interface.h"
#include "ns3/ndisc-cache.h"
#include "ns3/names.h"
#include "ipv6-routing-helper.h"

namespace ns3 {

Ipv6RoutingHelper::~Ipv6RoutingHelper ()
{
}

void
Ipv6RoutingHelper::PrintRoutingTableAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printTime, &Ipv6RoutingHelper::Print, node, stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintRoutingTableAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintEvery, printInterval, node, stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintRoutingTableAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printTime, &Ipv6RoutingHelper::Print, node, stream, unit);
}

void
Ipv6RoutingHelper::PrintRoutingTableEvery (Time printInterval,Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintEvery, printInterval, node, stream, unit);
}

void
Ipv6RoutingHelper::Print (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv6> ipv6 = node->GetObject<Ipv6> ();
  if (ipv6)
    {
      Ptr<Ipv6RoutingProtocol> rp = ipv6->GetRoutingProtocol ();
      NS_ASSERT (rp);
      rp->PrintRoutingTable (stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv6> ipv6 = node->GetObject<Ipv6> ();
  if (ipv6)
    {
      Ptr<Ipv6RoutingProtocol> rp = ipv6->GetRoutingProtocol ();
      NS_ASSERT (rp);
      rp->PrintRoutingTable (stream, unit);
      Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintEvery, printInterval, node, stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintNeighborCacheAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printTime, &Ipv6RoutingHelper::PrintNdiscCache, node, stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintNeighborCacheAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintNdiscCacheEvery, printInterval, node, stream, unit);
    }
}

void
Ipv6RoutingHelper::PrintNeighborCacheAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  Simulator::Schedule (printTime, &Ipv6RoutingHelper::PrintNdiscCache, node, stream, unit);
}

void
Ipv6RoutingHelper::PrintNeighborCacheEvery (Time printInterval,Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintNdiscCacheEvery, printInterval, node, stream, unit);
}

void
Ipv6RoutingHelper::PrintNdiscCache (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  Ptr<Ipv6L3Protocol> ipv6 = node->GetObject<Ipv6L3Protocol> ();
  if (ipv6)
    {
      std::ostream* os = stream->GetStream ();

      *os << "NDISC Cache of node ";
      std::string found = Names::FindName (node);
      if (Names::FindName (node) != "")
        {
          *os << found;
        }
      else
        {
          *os << static_cast<int> (node->GetId ());
        }
      *os << " at time " << Simulator::Now ().As (unit) << "\n";

      for (uint32_t i=0; i<ipv6->GetNInterfaces(); i++)
        {
          Ptr<NdiscCache> ndiscCache = ipv6->GetInterface (i)->GetNdiscCache ();
          if (ndiscCache)
            {
              ndiscCache->PrintNdiscCache (stream);
            }
        }
    }
}

void
Ipv6RoutingHelper::PrintNdiscCacheEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  Ptr<Ipv6L3Protocol> ipv6 = node->GetObject<Ipv6L3Protocol> ();
  if (ipv6)
    {
      std::ostream* os = stream->GetStream ();

      *os << "NDISC Cache of node ";
      std::string found = Names::FindName (node);
      if (Names::FindName (node) != "")
        {
          *os << found;
        }
      else
        {
          *os << static_cast<int> (node->GetId ());
        }
      *os << " at time " << Simulator::Now ().As (unit) << "\n";

      for (uint32_t i=0; i<ipv6->GetNInterfaces(); i++)
        {
          Ptr<NdiscCache> ndiscCache = ipv6->GetInterface (i)->GetNdiscCache ();
          if (ndiscCache)
            {
              ndiscCache->PrintNdiscCache (stream);
            }
        }
      Simulator::Schedule (printInterval, &Ipv6RoutingHelper::PrintNdiscCacheEvery, printInterval, node, stream, unit);
    }
}

} // namespace ns3
