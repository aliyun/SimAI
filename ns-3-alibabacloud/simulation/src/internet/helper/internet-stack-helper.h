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

#ifndef INTERNET_STACK_HELPER_H
#define INTERNET_STACK_HELPER_H

#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/object-factory.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6-l3-protocol.h"
#include "internet-trace-helper.h"

namespace ns3 {

class Node;
class Ipv4RoutingHelper;
class Ipv6RoutingHelper;

/**
 * \defgroup internet Internet
 *
 * This section documents the API of the ns-3 internet module. For a generic functional description, please refer to the ns-3 manual.
 */

/**
 * \ingroup internet
 * \defgroup ipv4Helpers IPv4 Helper classes
 */

/**
 * \ingroup internet
 * \defgroup ipv6Helpers IPv6 Helper classes
 */

/**
 * \ingroup internet
 *
 * \brief aggregate IP/TCP/UDP functionality to existing Nodes.
 *
 * This helper enables pcap and ascii tracing of events in the internet stack
 * associated with a node.  This is substantially similar to the tracing
 * that happens in device helpers, but the important difference is that, well,
 * there is no device.  This means that the creation of output file names will
 * change, and also the user-visible methods will not reference devices and
 * therefore the number of trace enable methods is reduced.
 *
 * Normally we avoid multiple inheritance in ns-3, however, the classes 
 * PcapUserHelperForIpv4 and AsciiTraceUserHelperForIpv4 are
 * treated as "mixins".  A mixin is a self-contained class that
 * encapsulates a general attribute or a set of functionality that
 * may be of interest to many other classes.
 *
 * This class aggregates instances of these objects, by default, to each node:
 *  - ns3::ArpL3Protocol
 *  - ns3::Ipv4L3Protocol
 *  - ns3::Icmpv4L4Protocol
 *  - ns3::Ipv6L3Protocol
 *  - ns3::Icmpv6L4Protocol
 *  - ns3::UdpL4Protocol
 *  - ns3::TrafficControlLayer
 *  - a TCP based on the TCP factory provided
 *  - a PacketSocketFactory
 *  - Ipv4 routing (a list routing object, a global routing object, and a static routing object)
 *  - Ipv6 routing (a static routing object)
 */
class InternetStackHelper : public PcapHelperForIpv4, public PcapHelperForIpv6, 
                            public AsciiTraceHelperForIpv4, public AsciiTraceHelperForIpv6
{
public:
  /**
   * Create a new InternetStackHelper which uses a mix of static routing
   * and global routing by default. The static routing protocol 
   * (ns3::Ipv4StaticRouting) and the global routing protocol are
   * stored in an ns3::Ipv4ListRouting protocol with priorities 0, and -10
   * by default. If you wish to use different priorites and different
   * routing protocols, you need to use an adhoc ns3::Ipv4RoutingHelper, 
   * such as ns3::OlsrHelper
   */
  InternetStackHelper(void);

  /**
   * Destroy the InternetStackHelper
   */
  virtual ~InternetStackHelper(void);

  /**
   * \brief Copy constructor
   * \param o Object to copy from.
   */
  InternetStackHelper (const InternetStackHelper &o);

  /**
   * \brief Copy constructor
   * \param o Object to copy from.
   * \returns A copy of the InternetStackHelper.
   */
  InternetStackHelper &operator = (const InternetStackHelper &o);

  /**
   * Return helper internal state to that of a newly constructed one
   */
  void Reset (void);

  /**
   * \param routing a new routing helper
   *
   * Set the routing helper to use during Install. The routing
   * helper is really an object factory which is used to create 
   * an object of type ns3::Ipv4RoutingProtocol per node. This routing
   * object is then associated to a single ns3::Ipv4 object through its 
   * ns3::Ipv4::SetRoutingProtocol.
   */
  void SetRoutingHelper (const Ipv4RoutingHelper &routing);

  /**
   * \brief Set IPv6 routing helper.
   * \param routing IPv6 routing helper
   */
  void SetRoutingHelper (const Ipv6RoutingHelper &routing);

  /**
   * Aggregate implementations of the ns3::Ipv4, ns3::Ipv6, ns3::Udp, and ns3::Tcp classes
   * onto the provided node.  This method will assert if called on a node that 
   * already has an Ipv4 object aggregated to it.
   * 
   * \param nodeName The name of the node on which to install the stack.
   */
  void Install (std::string nodeName) const;

  /**
   * Aggregate implementations of the ns3::Ipv4, ns3::Ipv6, ns3::Udp, and ns3::Tcp classes
   * onto the provided node.  This method will assert if called on a node that 
   * already has an Ipv4 object aggregated to it.
   * 
   * \param node The node on which to install the stack.
   */
  void Install (Ptr<Node> node) const;

  /**
   * For each node in the input container, aggregate implementations of the 
   * ns3::Ipv4, ns3::Ipv6, ns3::Udp, and, ns3::Tcp classes.  The program will assert 
   * if this method is called on a container with a node that already has
   * an Ipv4 object aggregated to it.
   * 
   * \param c NodeContainer that holds the set of nodes on which to install the
   * new stacks.
   */
  void Install (NodeContainer c) const;

  /**
   * Aggregate IPv4, IPv6, UDP, and TCP stacks to all nodes in the simulation
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
   * \brief Enable/disable IPv4 stack install.
   * \param enable enable state
   */
  void SetIpv4StackInstall (bool enable);

  /**
   * \brief Enable/disable IPv6 stack install.
   * \param enable enable state
   */
  void SetIpv6StackInstall (bool enable);

  /**
   * \brief Enable/disable IPv4 ARP Jitter.
   * \param enable enable state
   */
  void SetIpv4ArpJitter (bool enable);

  /**
   * \brief Enable/disable IPv6 NS and RS Jitter.
   * \param enable enable state
   */
  void SetIpv6NsRsJitter (bool enable);

  /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.  The Install() method should have previously been
  * called by the user.
  *
  * \param stream first stream index to use
  * \param c NodeContainer of the set of nodes for which the internet models
  *          should be modified to use a fixed stream
  * \return the number of stream indices assigned by this helper
  */
  int64_t AssignStreams (NodeContainer c, int64_t stream);

private:
  /**
   * @brief Enable pcap output the indicated Ipv4 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv4 Ptr to the Ipv4 interface on which you want to enable tracing.
   * @param interface Interface ID on the Ipv4 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
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
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnableAsciiIpv4Internal (Ptr<OutputStreamWrapper> stream, 
                                        std::string prefix, 
                                        Ptr<Ipv4> ipv4, 
                                        uint32_t interface,
                                        bool explicitFilename);

  /**
   * @brief Enable pcap output the indicated Ipv6 and interface pair.
   *
   * @param prefix Filename prefix to use for pcap files.
   * @param ipv6 Ptr to the Ipv6 interface on which you want to enable tracing.
   * @param interface Interface ID on the Ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnablePcapIpv6Internal (std::string prefix, 
                                       Ptr<Ipv6> ipv6, 
                                       uint32_t interface,
                                       bool explicitFilename);

  /**
   * @brief Enable ascii trace output on the indicated Ipv6 and interface pair.
   *
   * @param stream An OutputStreamWrapper representing an existing file to use
   *               when writing trace data.
   * @param prefix Filename prefix to use for ascii trace files.
   * @param ipv6 Ptr to the Ipv6 interface on which you want to enable tracing.
   * @param interface Interface ID on the Ipv6 on which you want to enable tracing.
   * @param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnableAsciiIpv6Internal (Ptr<OutputStreamWrapper> stream, 
                                        std::string prefix, 
                                        Ptr<Ipv6> ipv6, 
                                        uint32_t interface,
                                        bool explicitFilename);

  /**
   * \brief Initialize the helper to its default values
   */
  void Initialize (void);

  /**
   * \brief TCP objects factory
   */
  ObjectFactory m_tcpFactory;

  /**
   * \brief IPv4 routing helper.
   */
  const Ipv4RoutingHelper *m_routing;

  /**
   * \brief IPv6 routing helper.
   */
  const Ipv6RoutingHelper *m_routingv6;

  /**
   * \brief create an object from its TypeId and aggregates it to the node
   * \param node the node
   * \param typeId the object TypeId
   */
  static void CreateAndAggregateObjectFromTypeId (Ptr<Node> node, const std::string typeId);

  /**
   * \brief checks if there is an hook to a Pcap wrapper
   * \param ipv4 pointer to the IPv4 object
   * \returns true if a hook is found
   */
  bool PcapHooked (Ptr<Ipv4> ipv4);

  /**
   * \brief checks if there is an hook to an ascii output stream
   * \param ipv4 pointer to the IPv4 object
   * \returns true if a hook is found
   */
  bool AsciiHooked (Ptr<Ipv4> ipv4);

  /**
   * \brief checks if there is an hook to a Pcap wrapper
   * \param ipv6 pointer to the IPv6 object
   * \returns true if a hook is found
   */
  bool PcapHooked (Ptr<Ipv6> ipv6);

  /**
   * \brief checks if there is an hook to an ascii output stream
   * \param ipv6 pointer to the IPv6 object
   * \returns true if a hook is found
   */
  bool AsciiHooked (Ptr<Ipv6> ipv6);

  /**
   * \brief IPv4 install state (enabled/disabled) ?
   */
  bool m_ipv4Enabled;

  /**
   * \brief IPv6 install state (enabled/disabled) ?
   */
  bool m_ipv6Enabled;

  /**
   * \brief IPv4 ARP Jitter state (enabled/disabled) ?
   */
  bool m_ipv4ArpJitterEnabled;

  /**
   * \brief IPv6 IPv6 NS and RS Jitter state (enabled/disabled) ?
   */
  bool m_ipv6NsRsJitterEnabled;
};

} // namespace ns3

#endif /* INTERNET_STACK_HELPER_H */
