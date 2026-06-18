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
#ifndef PACKET_SOCKET_ADDRESS_H
#define PACKET_SOCKET_ADDRESS_H

#include "ns3/ptr.h"
#include "ns3/address.h"
#include "mac48-address.h"
#include "mac64-address.h"
#include "ns3/net-device.h"

namespace ns3 {

class NetDevice;

/**
 * \ingroup address
 *
 * \brief an address for a packet socket
 */
class PacketSocketAddress
{
public:
  PacketSocketAddress ();

  /**
   * \brief Set the protocol
   * \param protocol the protocol
   */
  void SetProtocol (uint16_t protocol);

  /**
   * \brief Set the address to match all the outgoing NetDevice
   */
  void SetAllDevices (void);

  /**
   * \brief Set the address to match only a specified NetDevice
   * \param device the NetDevice index
   */
  void SetSingleDevice (uint32_t device);

  /**
   * \brief Set the destination address
   * \param address the destination address
   */
  void SetPhysicalAddress (const Address address);

  /**
   * \brief Get the protocol
   * \return the protocol
   */
  uint16_t GetProtocol (void) const;

  /**
   * \brief Get the device this address is bound to
   * \return the device index
   */
  uint32_t GetSingleDevice (void) const;

  /**
   * \brief Checks if the address is bound to a specified NetDevice
   * \return true if the address is bound to a NetDevice
   */
  bool IsSingleDevice (void) const;

  /**
   * \brief Get the destination address
   * \returns The destination address
   */
  Address GetPhysicalAddress (void) const;

  /**
   * \returns a new Address instance
   *
   * Convert an instance of this class to a polymorphic Address instance.
   */
  operator Address () const;

  /**
   * \param address a polymorphic address
   * \returns an Address
   * Convert a polymorphic address to an Mac48Address instance.
   * The conversion performs a type check.
   */
  static PacketSocketAddress ConvertFrom (const Address &address);

  /**
   * \param address address to test
   * \returns true if the address matches, false otherwise.
   */
  static bool IsMatchingType (const Address &address);
private:

  /**
   * \brief Return the Type of address.
   * \return type of address
   */
  static uint8_t GetType (void);

  /**
   * \brief Convert an instance of this class to a polymorphic Address instance.
   * \returns a new Address instance
   */
  Address ConvertTo (void) const;

  uint16_t m_protocol;    //!< Protocol
  bool m_isSingleDevice;  //!< True if directed to a specific outgoing NetDevice
  uint32_t m_device;      //!< Outgoing NetDevice index
  Address m_address;      //!< Destination address
};


} // namespace ns3

#endif /* PACKET_SOCKET_ADDRESS_H */
