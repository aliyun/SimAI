/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include <vector>

#include "ns3/log.h"
#include "ns3/ptr.h"
#include "ns3/names.h"
#include "ns3/node.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv6-list-routing.h"
#include "ns3/assert.h"
#include "ns3/ipv6-address.h"
#include "ns3/ipv6-routing-protocol.h"

#include "ipv6-static-routing-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6StaticRoutingHelper");

Ipv6StaticRoutingHelper::Ipv6StaticRoutingHelper ()
{
}

Ipv6StaticRoutingHelper::Ipv6StaticRoutingHelper (const Ipv6StaticRoutingHelper &o)
{
}

Ipv6StaticRoutingHelper*
Ipv6StaticRoutingHelper::Copy (void) const
{
  return new Ipv6StaticRoutingHelper (*this);
}

Ptr<Ipv6RoutingProtocol>
Ipv6StaticRoutingHelper::Create (Ptr<Node> node) const
{
  return CreateObject<Ipv6StaticRouting> ();
}

Ptr<Ipv6StaticRouting>
Ipv6StaticRoutingHelper::GetStaticRouting (Ptr<Ipv6> ipv6) const
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv6RoutingProtocol> ipv6rp = ipv6->GetRoutingProtocol ();
  NS_ASSERT_MSG (ipv6rp, "No routing protocol associated with Ipv6");
  if (DynamicCast<Ipv6StaticRouting> (ipv6rp))
    {
      NS_LOG_LOGIC ("Static routing found as the main IPv4 routing protocol.");
      return DynamicCast<Ipv6StaticRouting> (ipv6rp); 
    } 
  if (DynamicCast<Ipv6ListRouting> (ipv6rp))
    {
      Ptr<Ipv6ListRouting> lrp = DynamicCast<Ipv6ListRouting> (ipv6rp);
      int16_t priority;
      for (uint32_t i = 0; i < lrp->GetNRoutingProtocols ();  i++)
        {
          NS_LOG_LOGIC ("Searching for static routing in list");
          Ptr<Ipv6RoutingProtocol> temp = lrp->GetRoutingProtocol (i, priority);
          if (DynamicCast<Ipv6StaticRouting> (temp))
            {
              NS_LOG_LOGIC ("Found static routing in list");
              return DynamicCast<Ipv6StaticRouting> (temp);
            }
        }
    }
  NS_LOG_LOGIC ("Static routing not found");
  return 0;
}

void
Ipv6StaticRoutingHelper::AddMulticastRoute (
  Ptr<Node> n,
  Ipv6Address source, 
  Ipv6Address group,
  Ptr<NetDevice> input, 
  NetDeviceContainer output)
{
  Ptr<Ipv6> ipv6 = n->GetObject<Ipv6> ();

  // We need to convert the NetDeviceContainer to an array of interface 
  // numbers
  std::vector<uint32_t> outputInterfaces;
  for (NetDeviceContainer::Iterator i = output.Begin (); i != output.End (); ++i)
    {
      Ptr<NetDevice> nd = *i;
      int32_t interface = ipv6->GetInterfaceForDevice (nd);
      NS_ASSERT_MSG (interface >= 0, 
                     "Ipv6StaticRoutingHelper::AddMulticastRoute (): "
                     "Expected an interface associated with the device nd");
      outputInterfaces.push_back (interface);
    }

  int32_t inputInterface = ipv6->GetInterfaceForDevice (input);
  NS_ASSERT_MSG (inputInterface >= 0, 
                 "Ipv6StaticRoutingHelper::AddMulticastRoute (): "
                 "Expected an interface associated with the device input");
  Ipv6StaticRoutingHelper helper;
  Ptr<Ipv6StaticRouting> ipv6StaticRouting = helper.GetStaticRouting (ipv6);
  if (!ipv6StaticRouting)
    {
      NS_ASSERT_MSG (ipv6StaticRouting,
                     "Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (): "
                     "Expected an Ipv6StaticRouting associated with this node");
    }
  ipv6StaticRouting->AddMulticastRoute (source, group, inputInterface, outputInterfaces);
}

void
Ipv6StaticRoutingHelper::AddMulticastRoute (
  Ptr<Node> n,
  Ipv6Address source, 
  Ipv6Address group,
  std::string inputName, 
  NetDeviceContainer output)
{
  Ptr<NetDevice> input = Names::Find<NetDevice> (inputName);
  AddMulticastRoute (n, source, group, input, output);
}

void
Ipv6StaticRoutingHelper::AddMulticastRoute (
  std::string nName,
  Ipv6Address source, 
  Ipv6Address group,
  Ptr<NetDevice> input, 
  NetDeviceContainer output)
{
  Ptr<Node> n = Names::Find<Node> (nName);
  AddMulticastRoute (n, source, group, input, output);
}

void
Ipv6StaticRoutingHelper::AddMulticastRoute (
  std::string nName,
  Ipv6Address source, 
  Ipv6Address group,
  std::string inputName, 
  NetDeviceContainer output)
{
  Ptr<NetDevice> input = Names::Find<NetDevice> (inputName);
  Ptr<Node> n = Names::Find<Node> (nName);
  AddMulticastRoute (n, source, group, input, output);
}

#if 0
void
Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (
  Ptr<Node> n, 
  Ptr<NetDevice> nd)
{
  Ptr<Ipv6> ipv6 = n->GetObject<Ipv6> ();
  int32_t interfaceSrc = ipv6->GetInterfaceForDevice (nd);
  NS_ASSERT_MSG (interfaceSrc >= 0, 
                 "Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (): "
                 "Expected an interface associated with the device");
  Ipv6StaticRoutingHelper helper;
  Ptr<Ipv6StaticRouting> ipv6StaticRouting = helper.GetStaticRouting (ipv6);
  if (!ipv6StaticRouting)
    {
      NS_ASSERT_MSG (ipv6StaticRouting, 
                     "Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (): "
                     "Expected an Ipv6StaticRouting associated with this node");
    }
  ipv6StaticRouting->SetDefaultMulticastRoute (interfaceSrc);
}

void
Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (
  Ptr<Node> n, 
  std::string ndName)
{
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  SetDefaultMulticastRoute (n, nd);
}

void
Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (
  std::string nName, 
  Ptr<NetDevice> nd)
{
  Ptr<Node> n = Names::Find<Node> (nName);
  SetDefaultMulticastRoute (n, nd);
}

void
Ipv6StaticRoutingHelper::SetDefaultMulticastRoute (
  std::string nName, 
  std::string ndName)
{
  Ptr<Node> n = Names::Find<Node> (nName);
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  SetDefaultMulticastRoute (n, nd);
}
#endif

} // namespace ns3
