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
 * Author: Lalith Suresh  <suresh.lalith@gmail.com>
 */

#ifdef NS3_CLICK

#ifndef CLICK_INTERNET_STACK_HELPER_H
#define CLICK_INTERNET_STACK_HELPER_H

#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/object-factory.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/internet-trace-helper.h"
#include <map>

namespace ns3 {

class Node;
class Ipv4RoutingHelper;

/**
 * \brief aggregate Click/IP/TCP/UDP functionality to existing Nodes.
 *
 * This helper has been adapted from the InternetStackHelper class and
 * nodes will not be able to use Ipv6 functionalities.
 *
 */
class ClickInternetStackHelper : public PcapHelperForIpv4,
                                 public AsciiTraceHelperForIpv4
{
public:
  /**
   * Create a new ClickInternetStackHelper which uses Ipv4ClickRouting for routing
   */
  ClickInternetStackHelper (void);

  /**
   * Destroy the ClickInternetStackHelper
   */
  virtual ~ClickInternetStackHelper (void);
  ClickInternetStackHelper (const ClickInternetStackHelper &);
  ClickInternetStackHelper &operator = (const ClickInternetStackHelper &o);

  /**
   * Return helper internal state to that of a newly constructed one
   */
  void Reset (void);

  /**
   * Aggregate implementations of the ns3::Ipv4L3ClickProtocol, ns3::ArpL3Protocol,
   * ns3::Udp, and ns3::Tcp classes onto the provided node.  This method will
   * assert if called on a node that already has an Ipv4 object aggregated to it.
   *
   * \param nodeName The name of the node on which to install the stack.
   */
  void Install (std::string nodeName) const;

  /**
   * Aggregate implementations of the ns3::Ipv4L3ClickProtocol, ns3::ArpL3Protocol,
   * ns3::Udp, and ns3::Tcp classes onto the provided node.  This method will
   * assert if called on a node that already has an Ipv4 object aggregated to it.
   *
   * \param node The node on which to install the stack.
   */
  void Install (Ptr<Node> node) const;

  /**
   * For each node in the input container, aggregate implementations of the
   * ns3::Ipv4L3ClickProtocol, ns3::ArpL3Protocol, ns3::Udp, and, ns3::Tcp classes.
   * The program will assert if this method is called on a container with a
   * node that already has an Ipv4 object aggregated to it.
   *
   * \param c NodeContainer that holds the set of nodes on which to install the
   * new stacks.
   */
  void Install (NodeContainer c) const;

  /**
   * Aggregate IPv4, UDP, and TCP stacks to all nodes in the simulation
   */
  void InstallAll (void) const;

  /**
   * \brief set the Tcp stack which will not need any other parameter.
   *
   * This function sets up the tcp stack to the given TypeId. It should not be
   * used for NSC stack setup because the nsc stack needs the Library attribute
   * to be setup, please use instead the version that requires an attribute
   * and a value. If you choose to use this function anyways to set nsc stack
   * the default value for the linux library will be used: "liblinux2.6.26.so".
   *
   * \param tid the type id, typically it is set to  "ns3::TcpL4Protocol"
   */
  void SetTcp (std::string tid);

  /**
   * \brief This function is used to setup the Network Simulation Cradle stack with library value.
   *
   * Give the NSC stack a shared library file name to use when creating the
   * stack implementation.  The attr string is actually the attribute name to
   * be setup and val is its value. The attribute is the stack implementation
   * to be used and the value is the shared library name.
   *
   * \param tid The type id, for the case of nsc it would be "ns3::NscTcpL4Protocol"
   * \param attr The attribute name that must be setup, for example "Library"
   * \param val The attribute value, which will be in fact the shared library name (example:"liblinux2.6.26.so")
   */
  void SetTcp (std::string tid, std::string attr, const AttributeValue &val);

  /**
   * \brief Set a Click file to be used for a group of nodes.
   * \param c NodeContainer of nodes
   * \param clickfile Click file to be used
   */
  void SetClickFile (NodeContainer c, std::string clickfile);

  /**
   * \brief Set a Click file to be used for a node.
   * \param node Node for which Click file is to be set
   * \param clickfile Click file to be used
   */
  void SetClickFile (Ptr<Node> node, std::string clickfile);

  /**
   * \brief Set defines to be used for a group of nodes.
   * \param c NodeContainer of nodes
   * \param defines Defines mapping to be used
   */
  void SetDefines (NodeContainer c, std::map<std::string, std::string> defines);

  /**
   * \brief Set defines to be used for a node.
   * \param node Node for which the defines are to be set
   * \param defines Defines mapping to be used
   */
  void SetDefines (Ptr<Node> node, std::map<std::string, std::string> defines);

  /**
   * \brief Set a Click routing table element for a group of nodes.
   * \param c NodeContainer of nodes
   * \param rt Click Routing Table element name
   */
  void SetRoutingTableElement (NodeContainer c, std::string rt);

  /**
   * \brief Set a Click routing table element for a node.
   * \param node Node for which Click file is to be set
   * \param rt Click Routing Table element name
   */
  void SetRoutingTableElement (Ptr<Node> node, std::string rt);
private:
  /**
   * @brief Enable pcap output the indicated Ipv4 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv4 Ptr to the Ipv4 interface on which you want to enable tracing.
   * @param interface Interface ID on the Ipv4 on which you want to enable tracing.
   */
  virtual void EnablePcapIpv4Internal (std::string prefix,
                                       Ptr<Ipv4> ipv4,
                                       uint32_t interface,
                                       bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the indicated Ipv4 and interface pair.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param ipv4 Ptr to the Ipv4 interface on which you want to enable tracing.
   * @param interface Interface ID on the Ipv4 on which you want to enable tracing.
   */
  virtual void EnableAsciiIpv4Internal (Ptr<OutputStreamWrapper> stream,
                                        std::string prefix,
                                        Ptr<Ipv4> ipv4,
                                        uint32_t interface,
                                        bool explicitFilename);

  void Initialize (void);
  ObjectFactory m_tcpFactory;

  static void CreateAndAggregateObjectFromTypeId (Ptr<Node> node, const std::string typeId);

  static void Cleanup (void);

  bool PcapHooked (Ptr<Ipv4> ipv4);

  bool AsciiHooked (Ptr<Ipv4> ipv4);

  /**
   * \brief IPv4 install state (enabled/disabled) ?
   */
  bool m_ipv4Enabled;

  /**
   * \brief Node to Click file mapping
   */
  std::map < Ptr<Node>, std::string  > m_nodeToClickFileMap;

  /**
   * \brief Node to Click defines mapping
   */
  std::map < Ptr<Node>, std::map<std::string, std::string>  > m_nodeToDefinesMap;

  /**
   * \brief Node to Routing Table Element mapping
   */
  std::map < Ptr<Node>, std::string  > m_nodeToRoutingTableElementMap;
};

} // namespace ns3

#endif /* CLICK_INTERNET_STACK_HELPER_H */

#endif /* NS3_CLICK */
