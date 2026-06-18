/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
 * Copyright (c) 2011 Atishay Jain
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

#ifndef IPV6_ADDRESS_GENERATOR_H
#define IPV6_ADDRESS_GENERATOR_H

#include "ns3/ipv6-address.h"

namespace ns3 {

/**
 * \ingroup address
 * \ingroup ipv6
 *
 * \brief This generator assigns addresses sequentially from a provided
 * network address; used in topology code. It also keeps track of all
 * addresses assigned to perform duplicate detection.
 *
 * Global unicast IPv6 addresses based on \RFC{4291} definition:
 *
 *     |         n bits          |   m bits  |       128-n-m bits         |
 *     +-------------------------+-----------+----------------------------+
 *     | global routing prefix   | subnet ID |       interface ID         |
 *     +-------------------------+-----------+----------------------------+
 *
 * In this class, the first two quantities (n + m) are what is called the
 * 'net', and the 'prefix' defines the length in bits of (n + m).
 *
 * The way this is expected to be used is that, after initializing the
 * network and interfaceId to a number, a user can call NextAddress ()
 * repeatedly to obtain new interface IDs with the current network (for
 * multiple addresses on the link) and can call NextNetwork () to increment
 * the subnet ID.
 *
 * The interface ID is often an EUI-64 address derived from the MAC address,
 * but can also be a pseudo-random value (\RFC{3041}).  This implementation
 * does not generate EUI-64-based interface IDs.
 *
 * \note BEWARE: this class acts as a Singleton.
 * In other terms, two different instances of Ipv6AddressGenerator will
 * pick IPv6 numbers from the same pool. Changing the network in one of them
 * will also change the network in the other instances.
 *
 */
class Ipv6AddressGenerator
{
public:
  /**
   * \brief Initialise the base network and interfaceId for the generator
   *
   * The first call to NextAddress() or GetAddress() will return the
   * value passed in.
   *
   * \param net The network for the base Ipv6Address
   * \param prefix The prefix of the base Ipv6Address
   * \param interfaceId The base interface ID used for initialization
   */
  static void Init (const Ipv6Address net, const Ipv6Prefix prefix,
                    const Ipv6Address interfaceId = "::1");

  /**
   * \brief Get the next network according to the given Ipv6Prefix
   *
   * This operation is a pre-increment, meaning that the internal state
   * is changed before returning the new network address.
   *
   * This also resets the interface ID to the base interface ID that was
   * used for initialization.
   *
   * \param prefix The Ipv6Prefix used to set the next network
   * \returns the IPv6 address of the next network
   */
  static Ipv6Address NextNetwork (const Ipv6Prefix prefix);

  /**
   * \brief Get the current network of the given Ipv6Prefix
   *
   * Does not change the internal state; this just peeks at the current
   * network
   *
   * \param prefix The Ipv6Prefix for the current network
   * \returns the IPv6 address of the current network
   */
  static Ipv6Address GetNetwork (const Ipv6Prefix prefix);

  /**
   * \brief Set the interfaceId for the given Ipv6Prefix
   *
   * \param interfaceId The interfaceId to set for the current Ipv6Prefix
   * \param prefix The Ipv6Prefix whose address is to be set
   */
  static void InitAddress (const Ipv6Address interfaceId, const Ipv6Prefix prefix);

  /**
   * \brief Allocate the next Ipv6Address for the configured network and prefix
   *
   * This operation is a post-increment, meaning that the first address
   * allocated will be the one that was initially configured.
   *
   * \param prefix The Ipv6Prefix for the current network
   * \returns the IPv6 address
   */
  static Ipv6Address NextAddress (const Ipv6Prefix prefix);

  /**
   * \brief Get the Ipv6Address that will be allocated upon NextAddress ()
   *
   * Does not change the internal state; just is used to peek the next
   * address that will be allocated upon NextAddress ()
   *
   * \param prefix The Ipv6Prefix for the current network
   * \returns the IPv6 address
   */
  static Ipv6Address GetAddress (const Ipv6Prefix prefix);

  /**
   * \brief Reset the networks and Ipv6Address to zero
   */
  static void Reset (void);

  /**
   * \brief Add the Ipv6Address to the list of IPv6 entries
   *
   * Typically, this is used by external address allocators that want
   * to make use of this class's ability to track duplicates.  AddAllocated
   * is always called internally for any address generated by NextAddress ()
   *
   * \param addr The Ipv6Address to be added to the list of Ipv6 entries
   * \returns true on success
   */
  static bool AddAllocated (const Ipv6Address addr);

  /**
   * \brief Check the Ipv6Address allocation in the list of IPv6 entries
   *
   * \param addr The Ipv6Address to be checked in the list of Ipv4 entries
   * \returns true if the address is already allocated
   */
  static bool IsAddressAllocated (const Ipv6Address addr);

  /**
   * \brief Check if a network has already allocated addresses
   *
   * \param addr The Ipv6 network to be checked
   * \param prefix The Ipv6 network prefix
   * \returns true if the network is already allocated
   */
  static bool IsNetworkAllocated (const Ipv6Address addr, const Ipv6Prefix prefix);

  /**
   * \brief Used to turn off fatal errors and assertions, for testing
   */
  static void TestMode (void);
};

}; // namespace ns3

#endif /* IPV6_ADDRESS_GENERATOR_H */
