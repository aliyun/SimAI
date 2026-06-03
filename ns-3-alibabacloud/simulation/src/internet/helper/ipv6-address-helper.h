/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 * modified by Tom Henderson for ns-3.14 release
 */

#ifndef IPV6_ADDRESS_HELPER_H
#define IPV6_ADDRESS_HELPER_H

#include <vector>

#include "ns3/ipv6-address.h"
#include "ns3/net-device-container.h"
#include "ipv6-interface-container.h"

namespace ns3 {

/**
 * \ingroup ipv6Helpers
 *
 * \brief Helper class to auto-assign global IPv6 unicast addresses
 *
 * Assign global unicast IPv6 addresses based on \RFC{4291} definition.
 *
 *   |         n bits          | 64-n bits |       64 bits              |
 *   +-------------------------+-----------+----------------------------+
 *   | global routing prefix   | subnet ID |       interface ID         |
 *   +-------------------------+-----------+----------------------------+
 *   <-------------network---------------->
 *
 *   11111111111111111111111111000000000000
 *   <---prefix of length n--->
 * 
 * This class handles the following quantities.
 * 1) The "network" which covers the 64 bit union of the global routing
 *    prefix and the subnet ID
 * 2) the "prefix" length "n" which demarcates the global routing prefix
 *    and the subnet ID
 * 3) the "base" which is the initial 64-bit interface ID.
 *
 * The names "network", "prefix" and "base" are chosen to be consistent 
 * with a similar address helper for IPv4. 
 *
 * This helper class allows users to set or reset the network and
 * interface components, and call "NewAddress ()" to sequentially increment
 * the interface ID, and call "NewNetwork ()" to allocate a new subnet
 * (until the subnet ID quantity rolls over).  A call to NewNetwork ()
 * that causes the subnet ID to roll over will trigger an assertion.  
 * 
 * By default, the prefix is 32 bits and the network is '2001:db8::/32'
 * (\RFC{5156} section 2.6 Documentation prefix).  The prefix may range
 * from length 0 to 64, with the value 64 having a special meaning that
 * no subnet ID boundary is enforced (equivalent to value 0).
 * 
 * There are two variants of interface ID supported (\RFC{4291}, Sec. 2.5.1)
 * The default is a "local" scope, but a "universal" scoped ID may be
 * formed by calling "NewAddress (Address addr)" with a 48-bit MAC address.
 * If this method is called, the addressed returned will include a
 * modified EUI-64-format identifier created from the MAC address as 
 * specified in \RFC{4291}.
 *
 * BEWARE: the underlying implementation acts as a Singleton.
 * In other terms, two different instances of Ipv6AddressHelper will
 * pick IPv6 numbers from the same pool. Changing the network in one of them
 * will also change the network in the other instances.
 */
class Ipv6AddressHelper
{
public:
  /**
   * \brief Constructor.
   */
  Ipv6AddressHelper ();

  /**
   * \brief Constructor.
   * \param network The IPv6 network
   * \param prefix The prefix
   * \param base The base interface ID
   */
  Ipv6AddressHelper (Ipv6Address network, Ipv6Prefix prefix,  
                     Ipv6Address base = Ipv6Address ("::1"));

  /**
   * \brief Set the base network number, network prefix, and base interface ID
   *
   * \param network The IPv6 network
   * \param prefix The prefix
   * \param base The base interface ID
   */
  void SetBase (Ipv6Address network, Ipv6Prefix prefix,
                Ipv6Address base = Ipv6Address ("::1"));

  /**
   * \brief Allocate a new network.
   *
   * This method will cause the subnet prefix to increment, for future
   * network IDs, and resets the interface ID to the previously used
   * base.
   */
  void NewNetwork (void);

  /**
   * \brief Allocate a new Ipv6Address.
   *
   * If a Mac48Address is passed in, an Ipv6 autoconfigured address
   * according to the current subnet prefix is returned.  If something
   * other than Mac48 address is passed in, the program will terminate.
   *
   * \param addr address used to generate the interface ID of the IPv6 address
   * \return newly created Ipv6Address
   */
  Ipv6Address NewAddress (Address addr);

  /**
   * \brief Allocate a new Ipv6Address with interface ID equal to the
   * next one in the underlying generator.
   *
   * \return newly created Ipv6Address
   */
  Ipv6Address NewAddress (void);

  /**
   * \brief Allocate an Ipv6InterfaceContainer with auto-assigned addresses.
   * \param c netdevice container
   * \return newly created Ipv6InterfaceContainer
   */
  Ipv6InterfaceContainer Assign (const NetDeviceContainer &c);

  /**
   * \brief Allocate an Ipv6InterfaceContainer, and control whether the 
   *        interfaces have addresses auto-assigned to them
   *
   * \param c netdevice container
   * \param withConfiguration a vector of values for which, for a 
   *        given device, true : interface automatically addressed, 
   *        false : no automatic address
   * \return newly created Ipv6InterfaceContainer
   */
  Ipv6InterfaceContainer Assign (const NetDeviceContainer &c, std::vector<bool> withConfiguration);

  /**
   * \brief Allocate an Ipv6InterfaceContainer, and control whether the
   *        interfaces have addresses auto-assigned to them
   *
   * \param c netdevice container
   * \param withConfiguration a vector of values for which, for a
   *        given device, true : interface automatically addressed,
   *        false : no automatic address
   * \param onLink a vector of values for which, for a
   *        given device, true : on-link property added,
   *        false : network is not on-link
   *        Meaningful only if the address is automatically assigned.
   * \return newly created Ipv6InterfaceContainer
   */
  Ipv6InterfaceContainer Assign (const NetDeviceContainer &c, std::vector<bool> withConfiguration, std::vector<bool> onLink);

  /**
   * \brief Allocate an Ipv6InterfaceContainer but do not assign any IPv6 addresses
   *
   * This method is used when IPv6 address assignment may occur later
   * (such as dynamic address assignment)
   *
   * Equivalent to AssignWithoutAddress (c, std::vector<bool> of false);
   *
   * \param c netdevice container
   * \return newly created Ipv6InterfaceContainer
   */
  Ipv6InterfaceContainer AssignWithoutAddress (const NetDeviceContainer &c);

  /**
   * \brief Allocate an Ipv6InterfaceContainer with auto-assigned addresses,
   * but do not set the on-link property for the network.
   *
   * This method will assign a valid global address to the interface but
   * the routing will not consider the network as "on-link".
   *
   * \param c netdevice container
   * \return newly created Ipv6InterfaceContainer
   */
  Ipv6InterfaceContainer AssignWithoutOnLink (const NetDeviceContainer &c);

private:
  Ipv6Address m_network; //!< network address
  Ipv6Prefix m_prefix;   //!< prefix length
  Ipv6Address m_address; //!< host address
  Ipv6Address m_base;    //!< host base address
};

} /* namespace ns3 */

#endif /* IPV6_ADDRESS_STATIC_H */

