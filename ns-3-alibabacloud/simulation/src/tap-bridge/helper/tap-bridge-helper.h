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

#ifndef TAP_BRIDGE_HELPER_H
#define TAP_BRIDGE_HELPER_H

#include "ns3/net-device-container.h"
#include "ns3/object-factory.h"
#include "ns3/tap-bridge.h"
#include <string>

namespace ns3 {

class Node;
class AttributeValue;

/**
 * \brief build TapBridge to allow ns-3 simulations to interact with Linux 
 * tap devices and processes on the Linux host.
 */
class TapBridgeHelper
{
public:
  /**
   * Construct a TapBridgeHelper to make life easier for people wanting to 
   * have their simulations interact with Linux tap devices and processes
   * on the Linux host.
   */
  TapBridgeHelper ();

  /**
   * Construct a TapBridgeHelper to make life easier for people wanting to 
   * have their simulations interact with Linux tap devices and processes
   * on the Linux host.
   *
   * \param gateway An Ipv4Address to be used as the default gateway for 
   * the created bridges,
   */
  TapBridgeHelper (Ipv4Address gateway);

  /**
   * Set an attribute in the underlying TapBridge net device when these
   * devices are automatically created.
   *
   * \param n1 the name of the attribute to set
   * \param v1 the value of the attribute to set
   */
  void SetAttribute (std::string n1, const AttributeValue &v1);

  /**
   * This method installs a TapBridge on the specified Node and forms the 
   * bridge with the NetDevice specified.  The Node is specified using
   * a Ptr<Node> and the NetDevice is specified using a Ptr<NetDevice>
   *
   * \param node The Ptr<Node> to install the TapBridge in
   * \param nd The Ptr<NetDevice> to attach to the bridge.
   * \returns A pointer to the new TapBridge NetDevice.
   */
  Ptr<NetDevice> Install (Ptr<Node> node, Ptr<NetDevice> nd);

  /**
   * This method installs a TapBridge on the specified Node and forms the 
   * bridge with the NetDevice specified.  The node is specified by a 
   * name string that has previously been associated with the Node using
   * the Object Name Service.  The NetDevice is specified by a Ptr<NetDevice>.
   *
   * \param nodeName The name of the Node to install the TapBridge in
   * \param nd The Ptr<NetDevice> to attach to the bridge.
   * \returns A pointer to the new TapBridge NetDevice.
   */
  Ptr<NetDevice> Install (std::string nodeName, Ptr<NetDevice> nd);

  /**
   * This method installs a TapBridge on the specified Node and forms the 
   * bridge with the NetDevice specified.  The NetDevice is specified by a 
   * name string that has previously been associated with the NetDevice
   * using the Object Name Service.
   *
   * \param node The Ptr<Node> to install the TapBridge in
   * \param ndName The name of the NetDevice to attach to the bridge.
   * \returns A pointer to the new TapBridge NetDevice.
   */
  Ptr<NetDevice> Install (Ptr<Node> node, std::string ndName);

  /**
   * This method installs a TapBridge on the specified Node and forms the 
   * bridge with the NetDevice specified.  The node is specified by a 
   * name string that has previously been associated with the Node using
   * the Object Name Service.  The NetDevice is specified by a name 
   * string that has previously been associated with the Object Name 
   * Service.
   *
   * \param nodeName The name of the Node to install the TapBridge in
   * \param ndName The name of the NetDevice to attach to the bridge.
   * \returns A pointer to the new TapBridge NetDevice.
   */
  Ptr<NetDevice> Install (std::string nodeName, std::string ndName);

  /**
   * This method installs a TapBridge on the specified Node and forms the 
   * bridge with the NetDevice specified.  The Node is specified using
   * a Ptr<Node> and the NetDevice is specified using a Ptr<NetDevice>.
   * The type of the actual Bridge device is specified with the
   * provided AttributeValue (typically "ns3::TapBridge").
   *
   * \param node The Ptr<Node> to install the TapBridge in
   * \param nd The Ptr<NetDevice> to attach to the bridge.
   * \param bridgeType The TypeId of the bridge that will be automatically 
   *                   created.
   * \returns A pointer to the new TapBridge NetDevice.
   */
  Ptr<NetDevice> Install (Ptr<Node> node, Ptr<NetDevice> nd, const AttributeValue &bridgeType);
private:
  ObjectFactory m_deviceFactory; //!< Object factory
};

} // namespace ns3


#endif /* TAP_BRIDGE_HELPER_H */
