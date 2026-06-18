/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 University of Washington
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

#ifndef IPV4_ADDRESS_HELPER_H
#define IPV4_ADDRESS_HELPER_H

#include "ns3/ipv4-address.h"
#include "ns3/net-device-container.h"
#include "ipv4-interface-container.h"

namespace ns3 {

/**
 * \ingroup ipv4Helpers
 *
 * @brief A helper class to make life easier while doing simple IPv4 address
 * assignment in scripts.
 *
 * This class is a very simple IPv4 address generator.  You can think of it
 * as a simple local number incrementer.  It has no notion that IP addresses
 * are part of a global address space.  If you have a complicated address 
 * assignment situation you may want to look at the Ipv4AddressGenerator which
 * does recognize that IP address and network number generation is part of a
 * global problem.  Ipv4AddressHelper is a simple class to make simple problems
 * easy to handle.
 *
 * We do call into the global address generator to make sure that there are
 * no duplicate addresses generated.
 *
 * @see Ipv4AddressGenerator
 */
class Ipv4AddressHelper
{
public:
  /**
   * @brief Construct a helper class to make life easier while doing simple IPv4
   * address assignment in scripts.
   */
  Ipv4AddressHelper ();

  /**
   * @brief Construct a helper class to make life easier while doing simple IPv4
   * address assignment in scripts.  This version sets the base and mask
   * in the constructor
   * @param network the network part
   * @param mask the address mask
   * @param base the host part to start from
   */
  Ipv4AddressHelper (Ipv4Address network, Ipv4Mask mask, 
                     Ipv4Address base = "0.0.0.1");

  /**
   * @brief Set the base network number, network mask and base address.
   *
   * The address helper allocates IP addresses based on a given network number
   * and mask combination along with an initial IP address.
   *
   * For example, if you want to use a /24 prefix with an initial network number
   * of 192.168.1 (corresponding to a mask of 255.255.255.0) and you want to
   * start allocating IP addresses out of that network beginning at 192.168.1.3,
   * you would call
   *
   *   SetBase ("192.168.1.0", "255.255.255.0", "0.0.0.3");
   *
   * If you don't care about the initial address it defaults to "0.0.0.1" in
   * which case you can simply use,
   *
   *   SetBase ("192.168.1.0", "255.255.255.0");
   *
   * and the first address generated will be 192.168.1.1.
   *
   * @param network The Ipv4Address containing the initial network number to
   * use during allocation.  The bits outside the network mask are not used.
   * @param mask The Ipv4Mask containing one bits in each bit position of the
   * network number.
   * @param base An optional Ipv4Address containing the initial address used for
   * IP address allocation.  Will be combined (ORed) with the network number to
   * generate the first IP address.  Defaults to 0.0.0.1.
   */
  void SetBase (Ipv4Address network, Ipv4Mask mask, 
                Ipv4Address base = "0.0.0.1");

  /**
   * @brief Increment the network number and reset the IP address counter to
   * the base value provided in the SetBase method.
   *
   * The address helper allocates IP addresses based on a given network number
   * and initial IP address.  In order to separate the network number and IP
   * address parts, SetBase was given an initial network number value, a network
   * mask and an initial address base.
   *
   * This method increments the network number and resets the IP address
   * counter to the last base value used.  For example, if the network number was
   * set to 192.168.0.0 with a mask of 255.255.255.0 and a base address of
   * 0.0.0.3 in the SetBase call; a call to NewNetwork will increment the
   * network number counter resulting in network numbers incrementing as
   * 192.168.1.0, 192.168.2.0, etc.  After each network number increment, the
   * IP address counter is reset to the initial value specified in SetBase.  In
   * this case, that would be 0.0.0.3. so if you were to call NewAddress after
   * the increment that resulted in a network number of 192.168.2.0, the
   * allocated addresses returned by NewAddress would be 192.168.2.3,
   * 192.168.2.4, etc.
   *
   * @returns The value of the incremented network number that will be used in
   * following address allocations.
   * @see SetBase
   * @see NewAddress
   */
  Ipv4Address NewNetwork (void);

  /**
   * @brief Increment the IP address counter used to allocate IP addresses
   *
   * The address helper allocates IP addresses based on a given network number
   * and initial IP address.  In order to separate the network number and IP
   * address parts, SetBase was given an initial network number value, a network
   * mask and an initial address base.
   *
   * This method increments IP address counter.  A check is made to ensure that
   * the address returned will not overflow the number of bits allocated to IP
   * addresses in SetBase (the number of address bits is defined by the number
   * of mask bits that are not '1').
   *
   * For example, if the network number was set to 192.168.0.0 with a mask of
   * 255.255.255.0 and a base address of 0.0.0.3 in SetBase, the next call to
   * NewAddress will return 192.168.1.3.  The NewAddress method
   * has post-increment semantics.  A following NewAddress would return
   * 192.168.0.4, etc., until the 253rd call which would assert due to an address
   * overflow.
   *
   * @returns The value of the newly allocated IP address.
   * @see SetBase
   * @see NewNetwork
   */
  Ipv4Address NewAddress (void);

  /**
   * @brief Assign IP addresses to the net devices specified in the container
   * based on the current network prefix and address base.
   *
   * The address helper allocates IP addresses based on a given network number
   * and initial IP address.  In order to separate the network number and IP
   * address parts, SetBase was given an initial value and a network mask.
   * The one bits of this mask define the prefix category from which the helper
   * will allocate new network numbers.  An initial value for the network
   * numbers was provided in the base parameter of the SetBase method in the
   * bits corresponding to positions in the mask that were 1.  An initial value
   * for the IP address counter was also provided in the base parameter in the
   * bits corresponding to positions in the mask that were 0.
   *
   * This method gets new addresses for each net device in the container.  For
   * each net device in the container, the helper finds the associated node and
   * looks up the Ipv4 interface corresponding to the net device.  It then sets
   * the Ipv4Address and mask in the interface to the appropriate values.  If
   * the addresses overflow the number of bits allocated for them by the network
   * mask in the SetBase method, the system will NS_ASSERT and halt.
   *
   * @param c The NetDeviceContainer holding the collection of net devices we
   * are asked to assign Ipv4 addresses to.
   *
   * @returns A container holding the added NetDevices
   * @see SetBase
   * @see NewNetwork
   */
  Ipv4InterfaceContainer Assign (const NetDeviceContainer &c);

private:
  /**
   * \brief Returns the number of address bits (hostpart) for a given netmask
   * \param maskbits the netmask
   * \returns the number of bits in the hostpart
   */
  uint32_t NumAddressBits (uint32_t maskbits) const;

  uint32_t m_network; //!< network address
  uint32_t m_mask;    //!< network mask
  uint32_t m_address; //!< address
  uint32_t m_base;    //!< base address
  uint32_t m_shift;   //!< shift, equivalent to the number of bits in the hostpart
  uint32_t m_max;     //!< maximum allowed address
};

} // namespace ns3

#endif /* IPV4_ADDRESS_HELPER_H */
