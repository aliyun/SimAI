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
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-interface.h"
#include "ns3/arp-cache.h"
#include "ns3/names.h"
#include "ipv4-routing-helper.h"

namespace ns3 {

Ipv4RoutingHelper::~Ipv4RoutingHelper ()
{
}

void
Ipv4RoutingHelper::PrintRoutingTableAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printTime, &Ipv4RoutingHelper::Print, node, stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintRoutingTableAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintEvery, printInterval, node, stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintRoutingTableAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printTime, &Ipv4RoutingHelper::Print, node, stream, unit);
}

void
Ipv4RoutingHelper::PrintRoutingTableEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintEvery, printInterval, node, stream, unit);
}

void
Ipv4RoutingHelper::Print (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  if (ipv4)
    {
      Ptr<Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol ();
      NS_ASSERT (rp);
      rp->PrintRoutingTable (stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  if (ipv4)
    {
      Ptr<Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol ();
      NS_ASSERT (rp);
      rp->PrintRoutingTable (stream, unit);
      Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintEvery, printInterval, node, stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintNeighborCacheAllAt (Time printTime, Ptr<OutputStreamWrapper> stream, Time::Unit unit /* = Time::S */)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printTime, &Ipv4RoutingHelper::PrintArpCache, node, stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintNeighborCacheAllEvery (Time printInterval, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  for (uint32_t i = 0; i < NodeList::GetNNodes (); i++)
    {
      Ptr<Node> node = NodeList::GetNode (i);
      Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintArpCacheEvery, printInterval, node, stream, unit);
    }
}

void
Ipv4RoutingHelper::PrintNeighborCacheAt (Time printTime, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printTime, &Ipv4RoutingHelper::PrintArpCache, node, stream, unit);
}

void
Ipv4RoutingHelper::PrintNeighborCacheEvery (Time printInterval,Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintArpCacheEvery, printInterval, node, stream, unit);
}

void
Ipv4RoutingHelper::PrintArpCache (Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv4L3Protocol> ipv4 = node->GetObject<Ipv4L3Protocol> ();
  if (ipv4)
    {
      std::ostream* os = stream->GetStream ();

      *os << "ARP Cache of node ";
      std::string found = Names::FindName (node);
      if (Names::FindName (node) != "")
        {
          *os << found;
        }
      else
        {
          *os << static_cast<int> (node->GetId ());
        }
      *os << " at time " << Simulator::Now ().GetSeconds () << "\n";

      for (uint32_t i=0; i<ipv4->GetNInterfaces(); i++)
        {
          Ptr<ArpCache> arpCache = ipv4->GetInterface (i)->GetArpCache ();
          if (arpCache)
            {
              arpCache->PrintArpCache (stream);
            }
        }
    }
}

void
Ipv4RoutingHelper::PrintArpCacheEvery (Time printInterval, Ptr<Node> node, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<Ipv4L3Protocol> ipv4 = node->GetObject<Ipv4L3Protocol> ();
  if (ipv4)
    {
      std::ostream* os = stream->GetStream ();

      *os << "ARP Cache of node ";
      std::string found = Names::FindName (node);
      if (Names::FindName (node) != "")
        {
          *os << found;
        }
      else
        {
          *os << static_cast<int> (node->GetId ());
        }
      *os << " at time " << Simulator::Now ().GetSeconds () << "\n";

      for (uint32_t i=0; i<ipv4->GetNInterfaces(); i++)
        {
          Ptr<ArpCache> arpCache = ipv4->GetInterface (i)->GetArpCache ();
          if (arpCache)
            {
              arpCache->PrintArpCache (stream);
            }
        }
      Simulator::Schedule (printInterval, &Ipv4RoutingHelper::PrintArpCacheEvery, printInterval, node, stream, unit);
    }
}

} // namespace ns3
