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

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/enum.h"
#include "ns3/tap-bridge.h"
#include "ns3/names.h"
#include "tap-bridge-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TapBridgeHelper");

TapBridgeHelper::TapBridgeHelper ()
{
  NS_LOG_FUNCTION_NOARGS ();
  m_deviceFactory.SetTypeId ("ns3::TapBridge");
}

TapBridgeHelper::TapBridgeHelper (Ipv4Address gateway)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_deviceFactory.SetTypeId ("ns3::TapBridge");
  SetAttribute ("Gateway", Ipv4AddressValue (gateway));
  SetAttribute ("Mode", EnumValue (TapBridge::CONFIGURE_LOCAL));
}

void 
TapBridgeHelper::SetAttribute (std::string n1, const AttributeValue &v1)
{
  NS_LOG_FUNCTION (n1 << &v1);
  m_deviceFactory.Set (n1, v1);
}


Ptr<NetDevice>
TapBridgeHelper::Install (Ptr<Node> node, Ptr<NetDevice> nd, const AttributeValue &v1)
{
  NS_LOG_FUNCTION (node << nd << &v1);
  m_deviceFactory.Set ("DeviceName", v1);
  return Install (node, nd);
}

Ptr<NetDevice>
TapBridgeHelper::Install (Ptr<Node> node, Ptr<NetDevice> nd)
{
  NS_LOG_FUNCTION (node << nd);
  NS_LOG_LOGIC ("Install TapBridge on node " << node->GetId () << " bridging net device " << nd);

  Ptr<TapBridge> bridge = m_deviceFactory.Create<TapBridge> ();
  node->AddDevice (bridge);
  bridge->SetBridgedNetDevice (nd);

  return bridge;
}

Ptr<NetDevice>
TapBridgeHelper::Install (std::string nodeName, Ptr<NetDevice> nd)
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return Install (node, nd);
}

Ptr<NetDevice>
TapBridgeHelper::Install (Ptr<Node> node, std::string ndName)
{
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  return Install (node, nd);
}

Ptr<NetDevice>
TapBridgeHelper::Install (std::string nodeName, std::string ndName)
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  Ptr<NetDevice> nd = Names::Find<NetDevice> (ndName);
  return Install (node, nd);
}

} // namespace ns3
