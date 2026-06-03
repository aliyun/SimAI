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
#ifndef IPV4_H
#define IPV4_H

#include <stdint.h>
#include "ns3/object.h"
#include "ns3/socket.h"
#include "ns3/callback.h"
#include "ns3/ipv4-address.h"
#include "ipv4-route.h"
#include "ipv4-interface-address.h"

namespace ns3 {

class Node;
class NetDevice;
class Packet;
class Ipv4RoutingProtocol;
class IpL4Protocol;
class Ipv4Header;

/**
 * \ingroup internet
 * \defgroup ipv4 IPv4 classes and sub-modules
 */
/**
 * \ingroup ipv4
 * \brief Access to the IPv4 forwarding table, interfaces, and configuration
 *
 * This class defines the API to manipulate the following aspects of
 * the IPv4 implementation:
 * -# set/get an Ipv4RoutingProtocol 
 * -# register a NetDevice for use by the Ipv4 layer (basically, to
 * create Ipv4-related state such as addressing and neighbor cache that 
 * is associated with a NetDevice)
 * -# manipulate the status of the NetDevice from the Ipv4 perspective, 
 * such as marking it as Up or Down, 
 * -# adding, deleting, and getting addresses associated to the Ipv4 
 * interfaces.
 * -# exporting Ipv4 configuration attributes
 * 
 * Each NetDevice has conceptually a single Ipv4 interface associated
 * with it (the corresponding structure in the Linux Ipv4 implementation
 * is struct in_device).  Each interface may have one or more Ipv4
 * addresses associated with it.  Each Ipv4 address may have different
 * subnet mask, scope, etc., so all of this per-address information 
 * is stored in an Ipv4InterfaceAddress class (the corresponding 
 * structure in Linux is struct in_ifaddr)
 *
 * Ipv4 attributes such as whether IP forwarding is enabled and disabled
 * are also stored in this class
 *
 * TO DO:  Add API to allow access to the Ipv4 neighbor table
 *
 * \see Ipv4RoutingProtocol
 * \see Ipv4InterfaceAddress
 */
class Ipv4 : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  Ipv4 ();
  virtual ~Ipv4 ();

  /**
   * \brief Register a new routing protocol to be used by this Ipv4 stack
   *
   * This call will replace any routing protocol that has been previously 
   * registered.  If you want to add multiple routing protocols, you must
   * add them to a Ipv4ListRoutingProtocol directly.
   * 
   * \param routingProtocol smart pointer to Ipv4RoutingProtocol object
   */
  virtual void SetRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol) = 0;

  /**
   * \brief Get the routing protocol to be used by this Ipv4 stack
   * 
   * \returns smart pointer to Ipv4RoutingProtocol object, or null pointer if none
   */
  virtual Ptr<Ipv4RoutingProtocol> GetRoutingProtocol (void) const = 0;

  /**
   * \param device device to add to the list of Ipv4 interfaces
   *        which can be used as output interfaces during packet forwarding.
   * \returns the index of the Ipv4 interface added.
   *
   * Once a device has been added, it can never be removed: if you want
   * to disable it, you can invoke Ipv4::SetDown which will
   * make sure that it is never used during packet forwarding.
   */
  virtual uint32_t AddInterface (Ptr<NetDevice> device) = 0;

  /**
   * \returns the number of interfaces added by the user.
   */
  virtual uint32_t GetNInterfaces (void) const = 0;

  /**
   * \brief Return the interface number of the interface that has been
   *        assigned the specified IP address.
   *
   * \param address The IP address being searched for
   * \returns The interface number of the Ipv4 interface with the given 
   *          address or -1 if not found.
   *
   * Each IP interface has one or more IP addresses associated with it.
   * This method searches the list of interfaces for one that holds a
   * particular address.  This call takes an IP address as a parameter and
   * returns the interface number of the first interface that has been assigned
   * that address, or -1 if not found.  There must be an exact match; this
   * method will not match broadcast or multicast addresses.
   */
  virtual int32_t GetInterfaceForAddress (Ipv4Address address) const = 0;

  /**
   * \param packet packet to send
   * \param source source address of packet
   * \param destination address of packet
   * \param protocol number of packet
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet
   * down the stack to the MAC and PHY layers.
   */
  virtual void Send (Ptr<Packet> packet, Ipv4Address source,
                     Ipv4Address destination, uint8_t protocol, Ptr<Ipv4Route> route) = 0;

  /**
   * \param packet packet to send
   * \param ipHeader IP Header
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet with IPv4 Header
   * (Intend to be used with IpHeaderInclude attribute.)
   */
  virtual void SendWithHeader (Ptr<Packet> packet, Ipv4Header ipHeader, Ptr<Ipv4Route> route) = 0;

  /**
   * \param protocol a template for the protocol to add to this L4 Demux.
   * \returns the L4Protocol effectively added.
   *
   * Invoke Copy on the input template to get a copy of the input
   * protocol which can be used on the Node on which this L4 Demux
   * is running. The new L4Protocol is registered internally as
   * a working L4 Protocol and returned from this method.
   * The caller does not get ownership of the returned pointer.
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
   * \param protocol protocol to remove from this demux.
   *
   * The input value to this method should be the value
   * returned from the Ipv4L4Protocol::Insert method.
   */
  virtual void Remove (Ptr<IpL4Protocol> protocol) = 0;

  /**
   * \brief Remove a L4 protocol from a specific interface.
   * \param protocol L4 protocol to remove.
   * \param interfaceIndex interface index.
   */
  virtual void Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex) = 0;

  /**
   * \brief Determine whether address and interface corresponding to
   *        received packet can be accepted for local delivery
   *
   * \param address The IP address being considered
   * \param iif The incoming Ipv4 interface index
   * \returns true if the address is associated with the interface index
   *
   * This method can be used to determine whether a received packet has
   * an acceptable address for local delivery on the host.  The address
   * may be a unicast, multicast, or broadcast address.  This method will
   * return true if address is an exact match of a unicast address on
   * one of the host's interfaces (see below), if address corresponds to 
   * a multicast group that the host has joined (and the incoming device
   * is acceptable), or if address corresponds to a broadcast address.
   *
   * If the Ipv4 attribute WeakEsModel is true, the unicast address may
   * match any of the Ipv4 addresses on any interface.  If the attribute is
   * false, the address must match one assigned to the incoming device.
   */
  virtual bool IsDestinationAddress (Ipv4Address address, uint32_t iif) const = 0;

  /**
   * \brief Return the interface number of first interface found that 
   *  has an Ipv4 address within the prefix specified by the input
   *  address and mask parameters
   *
   * \param address The IP address assigned to the interface of interest.
   * \param mask The IP prefix to use in the mask
   * \returns The interface number of the Ipv4 interface with the given 
   *          address or -1 if not found.
   *
   * Each IP interface has one or more IP addresses associated with it.
   * This method searches the list of interfaces for the first one found
   * that holds an address that is included within the prefix 
   * formed by the input address and mask parameters.  The value -1 is
   * returned if no match is found.
   */
  virtual int32_t GetInterfaceForPrefix (Ipv4Address address,
                                         Ipv4Mask mask) const = 0;

  /**
   * \param interface The interface number of an Ipv4 interface.
   * \returns The NetDevice associated with the Ipv4 interface number.
   */
  virtual Ptr<NetDevice> GetNetDevice (uint32_t interface) = 0;

  /**
   * \param device The NetDevice for an Ipv4Interface
   * \returns The interface number of an Ipv4 interface or -1 if not found.
   */
  virtual int32_t GetInterfaceForDevice (Ptr<const NetDevice> device) const = 0;

  /**
   * \param interface Interface number of an Ipv4 interface
   * \param address Ipv4InterfaceAddress address to associate with the underlying Ipv4 interface
   * \returns true if the operation succeeded
   */
  virtual bool AddAddress (uint32_t interface, Ipv4InterfaceAddress address) = 0;

  /**
   * \param interface Interface number of an Ipv4 interface
   * \returns the number of Ipv4InterfaceAddress entries for the interface.
   */
  virtual uint32_t GetNAddresses (uint32_t interface) const = 0;

  /**
   * Because addresses can be removed, the addressIndex is not guaranteed
   * to be static across calls to this method.
   * 
   * \param interface Interface number of an Ipv4 interface
   * \param addressIndex index of Ipv4InterfaceAddress 
   * \returns the Ipv4InterfaceAddress associated to the interface and addressIndex
   */
  virtual Ipv4InterfaceAddress GetAddress (uint32_t interface, uint32_t addressIndex) const = 0;

  /**
   * Remove the address at addressIndex on named interface.  The addressIndex
   * for all higher indices will decrement by one after this method is called;
   * so, for example, to remove 5 addresses from an interface i, one could
   * call RemoveAddress (i, 0); 5 times.
   * 
   * \param interface Interface number of an Ipv4 interface
   * \param addressIndex index of Ipv4InterfaceAddress to remove 
   * \returns true if the operation succeeded
   */
  virtual bool RemoveAddress (uint32_t interface, uint32_t addressIndex) = 0;

   /**
   * \brief Remove the given address on named Ipv4 interface
   *
   * \param interface Interface number of an Ipv4 interface
   * \param address The address to remove
   * \returns true if the operation succeeded
   */
  virtual bool RemoveAddress (uint32_t interface, Ipv4Address address) = 0;

  /**
   * \brief Return the first primary source address with scope less than 
   * or equal to the requested scope, to use in sending a packet to 
   * destination dst out of the specified device.
   *
   * This method mirrors the behavior of Linux inet_select_addr() and is
   * provided because interfaces may have multiple IP addresses configured
   * on them with different scopes, and with a primary and secondary status.
   * Secondary addresses are never returned.
   * \see Ipv4InterfaceAddress
   *
   * If a non-zero device pointer is provided, the method first tries to
   * return a primary address that is configured on that device, and whose
   * subnet matches that of dst and whose scope is less than or equal to
   * the requested scope.  If a primary address does not match the
   * subnet of dst but otherwise matches the scope, it is returned.
   * If no such address on the device is found, the other devices are 
   * searched in order of their interface index, but not considering dst
   * as a factor in the search.  Because a loopback interface is typically 
   * the first one configured on a node, it will be the first alternate 
   * device to be tried.  Addresses scoped at LINK scope are not returned
   * in this phase.
   * 
   * If no device pointer is provided, the same logic as above applies, only
   * that there is no preferred device that is consulted first.  This means
   * that if the device pointer is null, input parameter dst will be ignored.
   * 
   * If there are no possible addresses to return, a warning log message 
   * is issued and the all-zeroes address is returned.
   *
   * \param device output NetDevice (optionally provided, only to constrain the search)
   * \param dst Destination address to match, if device is provided 
   * \param scope Scope of returned address must be less than or equal to this
   * \returns the first primary Ipv4Address that meets the search criteria
   */
  virtual Ipv4Address SelectSourceAddress (Ptr<const NetDevice> device, 
                                           Ipv4Address dst, Ipv4InterfaceAddress::InterfaceAddressScope_e scope) = 0;

  /**
   * \param interface The interface number of an Ipv4 interface
   * \param metric routing metric (cost) associated to the underlying 
   *          Ipv4 interface
   */
  virtual void SetMetric (uint32_t interface, uint16_t metric) = 0;

  /**
   * \param interface The interface number of an Ipv4 interface
   * \returns routing metric (cost) associated to the underlying 
   *          Ipv4 interface
   */
  virtual uint16_t GetMetric (uint32_t interface) const = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   * \returns the Maximum Transmission Unit (in bytes) associated
   *          to the underlying Ipv4 interface
   */
  virtual uint16_t GetMtu (uint32_t interface) const = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   * \returns true if the underlying interface is in the "up" state,
   *          false otherwise.
   */
  virtual bool IsUp (uint32_t interface) const = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   * 
   * Set the interface into the "up" state. In this state, it is
   * considered valid during Ipv4 forwarding.
   */
  virtual void SetUp (uint32_t interface) = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   *
   * Set the interface into the "down" state. In this state, it is
   * ignored during Ipv4 forwarding.
   */
  virtual void SetDown (uint32_t interface) = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   * \returns true if IP forwarding enabled for input datagrams on this device
   */
  virtual bool IsForwarding (uint32_t interface) const = 0;

  /**
   * \param interface Interface number of Ipv4 interface
   * \param val Value to set the forwarding flag
   * 
   * If set to true, IP forwarding is enabled for input datagrams on this device
   */
  virtual void SetForwarding (uint32_t interface, bool val) = 0;

  /**
   * \brief Choose the source address to use with destination address.
   * \param interface interface index
   * \param dest IPv4 destination address
   * \return IPv4 source address to use
   */
  virtual Ipv4Address SourceAddressSelection (uint32_t interface, Ipv4Address dest) = 0;

  /**
   * \param protocolNumber number of protocol to lookup
   *        in this L4 Demux
   * \returns a matching L4 Protocol
   *
   * This method is typically called by lower layers
   * to forward packets up the stack to the right protocol.
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
   * \brief Creates a raw socket
   *
   * \returns a smart pointer to the instantiated raw socket
   */
  virtual Ptr<Socket> CreateRawSocket (void) = 0;

  /**
   * \brief Deletes a particular raw socket
   *
   * \param socket Smart pointer to the raw socket to be deleted
   */
  virtual void DeleteRawSocket (Ptr<Socket> socket) = 0;


  static const uint32_t IF_ANY = 0xffffffff; //!< interface wildcard, meaning any interface

private:
  // Indirect the Ipv4 attributes through private pure virtual methods

  /**
   * \brief Set or unset the IP forwarding state
   * \param forward the forwarding state
   */
  virtual void SetIpForward (bool forward) = 0;
  /**
   * \brief Get the IP forwarding state
   * \returns true if IP is in forwarding state
   */
  virtual bool GetIpForward (void) const = 0;

  /**
   * \brief Set or unset the Weak Es Model
   *
   * RFC1122 term for whether host accepts datagram with a dest. address on another interface
   * \param model true for Weak Es Model
   */
  virtual void SetWeakEsModel (bool model) = 0;
  /**
   * \brief Get the Weak Es Model status
   *
   * RFC1122 term for whether host accepts datagram with a dest. address on another interface
   * \returns true for Weak Es Model activated
   */
  virtual bool GetWeakEsModel (void) const = 0;
};

} // namespace ns3 

#endif /* IPV4_H */
