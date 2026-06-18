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

#ifndef IPV4_STATIC_ROUTING_HELPER_H
#define IPV4_STATIC_ROUTING_HELPER_H

#include "ns3/ipv4.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/ipv4-routing-helper.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"

namespace ns3 {

/**
 * \ingroup ipv4Helpers
 *
 * \brief Helper class that adds ns3::Ipv4StaticRouting objects
 *
 * This class is expected to be used in conjunction with 
 * ns3::InternetStackHelper::SetRoutingHelper
 */
class Ipv4StaticRoutingHelper : public Ipv4RoutingHelper
{
public:
  /*
   * Construct an Ipv4StaticRoutingHelper object, used to make configuration
   * of static routing easier.
   */
  Ipv4StaticRoutingHelper ();

  /**
   * \brief Construct an Ipv4StaticRoutingHelper from another previously 
   * initialized instance (Copy Constructor).
   * \param o object to be copied
   */
  Ipv4StaticRoutingHelper (const Ipv4StaticRoutingHelper &o);

  /**
   * \returns pointer to clone of this Ipv4StaticRoutingHelper
   *
   * This method is mainly for internal use by the other helpers;
   * clients are expected to free the dynamic memory allocated by this method
   */
  Ipv4StaticRoutingHelper* Copy (void) const;

  /**
   * \param node the node on which the routing protocol will run
   * \returns a newly-created routing protocol
   *
   * This method will be called by ns3::InternetStackHelper::Install
   */
  virtual Ptr<Ipv4RoutingProtocol> Create (Ptr<Node> node) const;

  /**
   * Try and find the static routing protocol as either the main routing
   * protocol or in the list of routing protocols associated with the 
   * Ipv4 provided.
   *
   * \param ipv4 the Ptr<Ipv4> to search for the static routing protocol
   * \returns Ipv4StaticRouting pointer or 0 if not found
   */
  Ptr<Ipv4StaticRouting> GetStaticRouting (Ptr<Ipv4> ipv4) const;

  /**
   * \brief Add a multicast route to a node and net device using explicit 
   * Ptr<Node> and Ptr<NetDevice>
   *
   * \param n The node.
   * \param source Source address.
   * \param group Multicast group.
   * \param input Input NetDevice.
   * \param output Output NetDevices.
   */
  void AddMulticastRoute (Ptr<Node> n, Ipv4Address source, Ipv4Address group,
                          Ptr<NetDevice> input, NetDeviceContainer output);

  /**
   * \brief Add a multicast route to a node and device using a name string 
   * previously associated to the node using the Object Name Service and a
   * Ptr<NetDevice>
   *
   * \param n The node.
   * \param source Source address.
   * \param group Multicast group.
   * \param input Input NetDevice.
   * \param output Output NetDevices.
   */
  void AddMulticastRoute (std::string n, Ipv4Address source, Ipv4Address group,
                          Ptr<NetDevice> input, NetDeviceContainer output);

  /**
   * \brief Add a multicast route to a node and device using a Ptr<Node> and a 
   * name string previously associated to the device using the Object Name Service.
   *
   * \param n The node.
   * \param source Source address.
   * \param group Multicast group.
   * \param inputName Input NetDevice.
   * \param output Output NetDevices.
   */
  void AddMulticastRoute (Ptr<Node> n, Ipv4Address source, Ipv4Address group,
                          std::string inputName, NetDeviceContainer output);

  /**
   * \brief Add a multicast route to a node and device using name strings
   * previously associated to both the node and device using the Object Name 
   * Service.
   *
   * \param nName The node.
   * \param source Source address.
   * \param group Multicast group.
   * \param inputName Input NetDevice.
   * \param output Output NetDevices.
   */
  void AddMulticastRoute (std::string nName, Ipv4Address source, Ipv4Address group,
                          std::string inputName, NetDeviceContainer output);

  /**
   * \brief Add a default route to the static routing protocol to forward
   *        packets out a particular interface
   *
   * Functionally equivalent to:
   * route add 224.0.0.0 netmask 240.0.0.0 dev nd
   * \param n node
   * \param nd device of the node to add default route
   */
  void SetDefaultMulticastRoute (Ptr<Node> n, Ptr<NetDevice> nd);

  /**
   * \brief Add a default route to the static routing protocol to forward
   *        packets out a particular interface
   *
   * Functionally equivalent to:
   * route add 224.0.0.0 netmask 240.0.0.0 dev nd
   * \param n node
   * \param ndName string with name previously associated to device using the 
   *        Object Name Service
   */
  void SetDefaultMulticastRoute (Ptr<Node> n, std::string ndName);

  /**
   * \brief Add a default route to the static routing protocol to forward
   *        packets out a particular interface
   *
   * Functionally equivalent to:
   * route add 224.0.0.0 netmask 240.0.0.0 dev nd
   * \param nName string with name previously associated to node using the 
   *        Object Name Service
   * \param nd device of the node to add default route
   */
  void SetDefaultMulticastRoute (std::string nName, Ptr<NetDevice> nd);

  /**
   * \brief Add a default route to the static routing protocol to forward
   *        packets out a particular interface
   *
   * Functionally equivalent to:
   * route add 224.0.0.0 netmask 240.0.0.0 dev nd
   * \param nName string with name previously associated to node using the 
   *        Object Name Service
   * \param ndName string with name previously associated to device using the 
   *        Object Name Service
   */
  void SetDefaultMulticastRoute (std::string nName, std::string ndName);
private:
  /**
   * \brief Assignment operator declared private and not implemented to disallow
   * assignment and prevent the compiler from happily inserting its own.
   * \returns
   */
  Ipv4StaticRoutingHelper &operator = (const Ipv4StaticRoutingHelper &);
};

} // namespace ns3

#endif /* IPV4_STATIC_ROUTING_HELPER_H */
