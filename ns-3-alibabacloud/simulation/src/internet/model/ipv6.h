/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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

/* taken from src/node/ipv4.h and adapted to IPv6 */

#ifndef IPV6_H
#define IPV6_H

#include <stdint.h>

#include "ns3/object.h"
#include "ns3/socket.h"
#include "ns3/callback.h"

#include "ns3/ipv6-address.h"
#include "ipv6-interface-address.h"

namespace ns3 {

class Node;
class NetDevice;
class Packet;
class Ipv6RoutingProtocol;
class IpL4Protocol;
class Ipv6Route;

/**
 * \ingroup internet
 * \defgroup ipv6 IPv6 classes and sub-modules
 */

/**
 * \ingroup ipv6
 * \brief Access to the IPv6 forwarding table, interfaces, and configuration
 *
 * This class defines the API to manipulate the following aspects of
 * the IPv6 implementation:
 * -# set/get an Ipv6RoutingProtocol 
 * -# register a NetDevice for use by the IPv6 layer (basically, to
 * create IPv6-related state such as addressing and neighbor cache that 
 * is associated with a NetDevice)
 * -# manipulate the status of the NetDevice from the IPv6 perspective, 
 * such as marking it as Up or Down, 
 * -# adding, deleting, and getting addresses associated to the IPv6 
 * interfaces.
 * -# exporting IPv6 configuration attributes
 * 
 * Each NetDevice has conceptually a single IPv6 interface associated
 * with it (the corresponding structure in the Linux IPv6 implementation
 * is struct in_device).  Each interface may have one or more IPv6
 * addresses associated with it.  Each IPv6 address may have different
 * subnet mask, scope, etc., so all of this per-address information 
 * is stored in an Ipv6InterfaceAddress class (the corresponding 
 * structure in Linux is struct in6_ifaddr)
 *
 * IPv6 attributes such as whether IP forwarding is enabled and disabled
 * are also stored in this class
 *
 * TO DO:  Add API to allow access to the IPv6 neighbor table
 *
 * \see Ipv6RoutingProtocol
 * \see Ipv6InterfaceAddress
 */
class Ipv6 : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Constructor.
   */
  Ipv6 ();

  /**
   * \brief Destructor.
   */
  virtual ~Ipv6 ();

  /**
   * \brief Register a new routing protocol to be used by this IPv6 stack
   *
   * This call will replace any routing protocol that has been previously 
   * registered.  If you want to add multiple routing protocols, you must
   * add them to a Ipv6ListRoutingProtocol directly.
   * 
   * \param routingProtocol smart pointer to Ipv6RoutingProtocol object
   */
  virtual void SetRoutingProtocol (Ptr<Ipv6RoutingProtocol> routingProtocol) = 0;

  /**
   * \brief Get the routing protocol to be used by this IPv6 stack
   * 
   * \returns smart pointer to Ipv6RoutingProtocol object, or null pointer if none
   */
  virtual Ptr<Ipv6RoutingProtocol> GetRoutingProtocol (void) const = 0;

  /**
   * \brief Add a NetDevice interface.
   *
   * Once a device has been added, it can never be removed: if you want
   * to disable it, you can invoke Ipv6::SetDown which will
   * make sure that it is never used during packet forwarding.
   * \param device device to add to the list of IPv6 interfaces
   *        which can be used as output interfaces during packet forwarding.
   * \returns the index of the IPv6 interface added.
   */
  virtual uint32_t AddInterface (Ptr<NetDevice> device) = 0;

  /**
   * \brief Get number of interfaces.
   * \returns the number of interfaces added by the user.
   */
  virtual uint32_t GetNInterfaces (void) const = 0;

  /**
   * \brief Return the interface number of the interface that has been
   *        assigned the specified IP address.
   *
   * \param address The IP address being searched for
   * \returns The interface number of the IPv6 interface with the given 
   *          address or -1 if not found.
   *
   * Each IP interface has one or more IP addresses associated with it.
   * This method searches the list of interfaces for one that holds a
   * particular address.  This call takes an IP address as a parameter and
   * returns the interface number of the first interface that has been assigned
   * that address, or -1 if not found.  There must be an exact match.
   */
  virtual int32_t GetInterfaceForAddress (Ipv6Address address) const = 0;

  /**
   * \brief Return the interface number of first interface found that 
   *  has an IPv6 address within the prefix specified by the input
   *  address and mask parameters
   *
   * \param address The IP address assigned to the interface of interest.
   * \param mask The IP prefix to use in the mask
   * \returns The interface number of the IPv6 interface with the given 
   *          address or -1 if not found.
   *
   * Each IP interface has one or more IP addresses associated with it.
   * This method searches the list of interfaces for the first one found
   * that holds an address that is included within the prefix 
   * formed by the input address and mask parameters.  The value -1 is
   * returned if no match is found.
   */
  virtual int32_t GetInterfaceForPrefix (Ipv6Address address,
                                         Ipv6Prefix mask) const = 0;

  /**
   * \brief Get the NetDevice of the specified interface number.
   * \param interface The interface number of an IPv6 interface.
   * \returns The NetDevice associated with the IPv6 interface number.
   */
  virtual Ptr<NetDevice> GetNetDevice (uint32_t interface) = 0;

  /**
   * \brief Get the interface index of the specified NetDevice.
   * \param device The NetDevice for an Ipv6Interface
   * \returns The interface number of an IPv6 interface or -1 if not found.
   */
  virtual int32_t GetInterfaceForDevice (Ptr<const NetDevice> device) const = 0;

  /**
   * \brief Add an address on the specified IPv6 interface.
   * \param interface Interface number of an IPv6 interface
   * \param address Ipv6InterfaceAddress address to associate with the underlying IPv6 interface
   * \param addOnLinkRoute add on-link route to the network (default true)
   * \returns true if the operation succeeded
   */
  virtual bool AddAddress (uint32_t interface, Ipv6InterfaceAddress address, bool addOnLinkRoute = true) = 0;

  /**
   * \brief Get number of addresses on specified IPv6 interface.
   * \param interface Interface number of an IPv6 interface
   * \returns the number of Ipv6InterfaceAddress entries for the interface.
   */
  virtual uint32_t GetNAddresses (uint32_t interface) const = 0;

  /**
   * \brief Get IPv6 address on specified IPv6 interface.
   *
   * Because addresses can be removed, the addressIndex is not guaranteed
   * to be static across calls to this method.
   * 
   * \param interface Interface number of an IPv6 interface
   * \param addressIndex index of Ipv6InterfaceAddress 
   * \returns the Ipv6InterfaceAddress associated to the interface and addressIndex
   */
  virtual Ipv6InterfaceAddress GetAddress (uint32_t interface, uint32_t addressIndex) const = 0;

  /**
   * \brief Remove an address on specified IPv6 interface.
   *
   * Remove the address at addressIndex on named interface.  The addressIndex
   * for all higher indices will decrement by one after this method is called;
   * so, for example, to remove 5 addresses from an interface i, one could
   * call RemoveAddress (i, 0); 5 times.
   * 
   * \param interface Interface number of an IPv6 interface
   * \param addressIndex index of Ipv6InterfaceAddress to remove 
   * \returns true if the operation succeeded
   */
  virtual bool RemoveAddress (uint32_t interface, uint32_t addressIndex) = 0;

  /**
   * \brief Remove the given address on named Ipv6 interface
   *
   * \param interface Interface number of an IPv6 interface
   * \param address the address to remove
   * \returns true if the operation succeeded
   */
  virtual bool RemoveAddress (uint32_t interface, Ipv6Address address) = 0;

  /**
   * \brief Set metric on specified Ipv6 interface.
   *
   * \param interface The interface number of an IPv6 interface
   * \param metric routing metric (cost) associated to the underlying 
   *          IPv6 interface
   */
  virtual void SetMetric (uint32_t interface, uint16_t metric) = 0;

  /**
   * \brief Get metric for the specified IPv6 interface.
   *
   * \param interface The interface number of an IPv6 interface
   * \returns routing metric (cost) associated to the underlying 
   *          IPv6 interface
   */
  virtual uint16_t GetMetric (uint32_t interface) const = 0;

  /**
   * \brief Get MTU for the specified IPv6 interface.
   * \param interface Interface number of IPv6 interface
   * \returns the Maximum Transmission Unit (in bytes) associated
   *          to the underlying IPv6 interface
   */
  virtual uint16_t GetMtu (uint32_t interface) const = 0;

  /**
   * \brief Set the Path MTU for the specified IPv6 destination address.
   * \param dst Ipv6 destination address
   * \param pmtu the Path MTU
   */
  virtual void SetPmtu (Ipv6Address dst, uint32_t pmtu) = 0;

  /**
   * \brief If the specified interface index is in "up" state.
   * \param interface Interface number of IPv6 interface
   * \returns true if the underlying interface is in the "up" state,
   *          false otherwise.
   */
  virtual bool IsUp (uint32_t interface) const = 0;

  /**
   * \brief Set the interface into the "up" state. 
   *
   * In this state, it is considered valid during IPv6 forwarding.
   * \param interface Interface number of IPv6 interface
   */
  virtual void SetUp (uint32_t interface) = 0;

  /**
   * \brief Set the interface into the "down" state.
   *
   * In this state, it is ignored during IPv6 forwarding.
   * \param interface Interface number of IPv6 interface
   */
  virtual void SetDown (uint32_t interface) = 0;

  /**
   * \brief If the specified IPv6 interface has forwarding enabled.
   * \param interface Interface number of IPv6 interface
   * \returns true if IPv6 forwarding enabled for input datagrams on this device
   */
  virtual bool IsForwarding (uint32_t interface) const = 0;

  /**
   * \brief Set forwarding on specified IPv6 interface.
   * \param interface Interface number of IPv6 interface
   * \param val Value to set the forwarding flag
   * 
   * If set to true, IPv6 forwarding is enabled for input datagrams on this device
   */
  virtual void SetForwarding (uint32_t interface, bool val) = 0;

  /**
   * \brief Choose the source address to use with destination address.
   * \param interface interface index
   * \param dest IPv6 destination address
   * \return IPv6 source address to use
   */
  virtual Ipv6Address SourceAddressSelection (uint32_t interface, Ipv6Address dest) = 0;

  /**
   * \brief Add a L4 protocol.
   * \param protocol L4 protocol
   */
  virtual void Insert (Ptr<IpL4Protocol> protocol) = 0;

  /**
   * \brief Add a L4 protocol to a specific interface.
   *
   * This may be called multiple times for multiple interfaces for the same
   * protocol.  To insert for all interfaces, use the separate
   * Insert (Ptr<IpL4Protocol> protocol) method.
   *
   * Setting a protocol on a specific interface will overwrite the
   * previously bound protocol.
   *
   * \param protocol L4 protocol.
   * \param interfaceIndex interface index.
   */
  virtual void Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex) = 0;

  /**
   * \brief Remove a L4 protocol.
   * \param protocol L4 protocol to remove.
   */
  virtual void Remove (Ptr<IpL4Protocol> protocol) = 0;

  /**
   * \brief Remove a L4 protocol from a specific interface.
   * \param protocol L4 protocol to remove.
   * \param interfaceIndex interface index.
   */
  virtual void Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex) = 0;

  /**
   * \brief Get L4 protocol by protocol number.
   * \param protocolNumber protocol number
   * \return corresponding IpL4Protocol or 0 if not found
   */
  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber) const = 0;

  /**
   * \brief Get L4 protocol by protocol number for the specified interface.
   * \param protocolNumber protocol number
   * \param interfaceIndex interface index, -1 means "any" interface.
   * \return corresponding IpL4Protocol or 0 if not found
   */
  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber, int32_t interfaceIndex) const = 0;

  /**
   * \brief Higher-level layers call this method to send a packet
   * down the stack to the MAC and PHY layers. All PMTU values are
   * stored at this level, so packet size calculations should be
   * done mathematically at higher levels.
   *
   * \param packet packet to send
   * \param source source address of packet
   * \param destination address of packet
   * \param protocol number of packet
   * \param route route to take
   */
  virtual void Send (Ptr<Packet> packet, Ipv6Address source, Ipv6Address destination, uint8_t protocol, Ptr<Ipv6Route> route) = 0;

  /**
   * \brief Register the IPv6 Extensions.
   */
  virtual void RegisterExtensions () = 0;

  /**
   * \brief Register the IPv6 Options.
   */
  virtual void RegisterOptions () = 0;

  /**
   * \brief Any interface magic number.
   */
  static const uint32_t IF_ANY = 0xffffffff;

private:
  // Indirect the IPv6 attributes through private pure virtual methods
  /**
   * \brief Set IPv6 forwarding state.
   * \param forward IPv6 forwarding enabled or not
   */
  virtual void SetIpForward (bool forward) = 0;

  /**
   * \brief Get IPv6 forwarding state.
   * \return forwarding state (enabled or not)
   */
  virtual bool GetIpForward (void) const = 0;

  /**
   * \brief Set IPv6 MTU discover state.
   * \param mtuDiscover IPv6 MTU discover enabled or not
   */
  virtual void SetMtuDiscover (bool mtuDiscover) = 0;

  /**
   * \brief Get IPv6 MTU discover state.
   * \return MTU discover state (enabled or not)
   */
  virtual bool GetMtuDiscover (void) const = 0;
};

} // namespace ns3 

#endif /* IPV6_H */

