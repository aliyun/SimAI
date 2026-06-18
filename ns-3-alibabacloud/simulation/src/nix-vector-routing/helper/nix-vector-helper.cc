/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Georgia Institute of Technology
 * Copyright (c) 2021 NITK Surathkal
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
 * This file is adapted from the old ipv4-nix-vector-helper.cc.
 *
 * Authors: Josh Pelkey <jpelkey@gatech.edu>
 * 
 * Modified by: Ameya Deshpande <ameyanrd@outlook.com>
 */

#include "nix-vector-helper.h"
#include "ns3/nix-vector-routing.h"
#include "ns3/assert.h"

namespace ns3 {

template <typename T>
NixVectorHelper<T>::NixVectorHelper ()
{
  std::string name = (IsIpv4::value ? "Ipv4" : "Ipv6");
  m_agentFactory.SetTypeId ("ns3::" + name + "NixVectorRouting");
}

template <typename T>
NixVectorHelper<T>::NixVectorHelper (const NixVectorHelper<T> &o)
  : m_agentFactory (o.m_agentFactory)
{
  // Check if the T is Ipv4RoutingHelper or Ipv6RoutingHelper.
  NS_ASSERT_MSG ((IsIpv4::value || std::is_same <Ipv6RoutingHelper, T>::value),
                  "Template parameter is not Ipv4RoutingHelper or Ipv6Routing Helper");
}

template <typename T>
NixVectorHelper<T>*
NixVectorHelper<T>::Copy (void) const
{
  return new NixVectorHelper<T> (*this);
}

template <typename T>
Ptr<typename NixVectorHelper<T>::IpRoutingProtocol>
NixVectorHelper<T>::Create (Ptr<Node> node) const
{
  Ptr<NixVectorRouting<IpRoutingProtocol>> agent = m_agentFactory.Create<NixVectorRouting<IpRoutingProtocol>> ();
  agent->SetNode (node);
  node->AggregateObject (agent);
  return agent;
}

template <typename T>
void
NixVectorHelper<T>::PrintRoutingPathAt (Time printTime, Ptr<Node> source, IpAddress dest, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Simulator::Schedule (printTime, &NixVectorHelper<T>::PrintRoute, source, dest, stream, unit);
}

template <typename T>
void
NixVectorHelper<T>::PrintRoute (Ptr<Node> source, IpAddress dest, Ptr<OutputStreamWrapper> stream, Time::Unit unit)
{
  Ptr<NixVectorRouting<IpRoutingProtocol>> rp = T::template GetRouting <NixVectorRouting<IpRoutingProtocol>> (source->GetObject<Ip> ()->GetRoutingProtocol ());
  NS_ASSERT (rp);
  rp->PrintRoutingPath (source, dest, stream, unit);
}

template class NixVectorHelper<Ipv4RoutingHelper>;
template class NixVectorHelper<Ipv6RoutingHelper>;

} // namespace ns3
