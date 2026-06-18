/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#ifndef INET_SOCKET_ADDRESS_H
#define INET_SOCKET_ADDRESS_H

#include "ns3/address.h"
#include "ipv4-address.h"
#include <stdint.h>

namespace ns3 {


/**
 * \ingroup address
 *
 * \brief an Inet address class
 *
 * This class is similar to inet_sockaddr in the BSD socket
 * API. i.e., this class holds an Ipv4Address and a port number
 * to form an ipv4 transport endpoint.
 */
class InetSocketAddress
{
public:
  /**
   * \param ipv4 the ipv4 address
   * \param port the port number
   */
  InetSocketAddress (Ipv4Address ipv4, uint16_t port);
  /**
   * \param ipv4 the ipv4 address
   *
   * The port number is set to zero by default.
   */
  InetSocketAddress (Ipv4Address ipv4);
  /**
   * \param port the port number
   *
   * The ipv4 address is set to the "Any" address by default.
   */
  InetSocketAddress (uint16_t port);
  /**
   * \param ipv4 string which represents an ipv4 address
   * \param port the port number
   */
  InetSocketAddress (const char *ipv4, uint16_t port);
  /**
   * \param ipv4 string which represents an ipv4 address
   *
   * The port number is set to zero.
   */
  InetSocketAddress (const char *ipv4);
  /**
   * \returns the port number
   */
  uint16_t GetPort (void) const;
  /**
   * \returns the ipv4 address
   */
  Ipv4Address GetIpv4 (void) const;
  /**
   * \returns the ToS
   */
  uint8_t GetTos (void) const;

  /**
   * \param port the new port number.
   */
  void SetPort (uint16_t port);
  /**
   * \param address the new ipv4 address
   */
  void SetIpv4 (Ipv4Address address);
  /**
   * \param tos the new ToS.
   */
  void SetTos (uint8_t tos);

  /**
   * \param address address to test
   * \returns true if the address matches, false otherwise.
   */
  static bool IsMatchingType (const Address &address);

  /**
   * \returns an Address instance which represents this
   * InetSocketAddress instance.
   */
  operator Address () const;

  /**
   * \brief Returns an InetSocketAddress which corresponds to the input
   * Address.
   *
   * \param address the Address instance to convert from.
   * \returns an InetSocketAddress
   */
  static InetSocketAddress ConvertFrom (const Address &address);
private:
  /**
   * \brief Convert to an Address type
   * \return the Address corresponding to this object.
   */
  Address ConvertTo (void) const;

  /**
   * \brief Get the underlying address type (automatically assigned).
   *
   * \returns the address type
   */
  static uint8_t GetType (void);
  Ipv4Address m_ipv4; //!< the IPv4 address
  uint16_t m_port;    //!< the port
  uint8_t m_tos;      //!< the ToS
};

} // namespace ns3


#endif /* INET_SOCKET_ADDRESS_H */
