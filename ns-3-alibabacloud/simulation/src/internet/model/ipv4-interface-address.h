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
 *
 */

#ifndef IPV4_INTERFACE_ADDRESS_H
#define IPV4_INTERFACE_ADDRESS_H

#include <stdint.h>
#include <ostream>
#include "ns3/ipv4-address.h"

namespace ns3 {

/**
 * \ingroup address
 * \ingroup ipv4
 *
 * \brief a class to store IPv4 address information on an interface
 *
 * Corresponds to Linux struct in_ifaddr.  A list of these addresses
 * is stored in Ipv4Interface.  This class is modelled after how current
 * Linux handles IP aliasing for IPv4.  Notably, aliasing of IPv4 
 * interfaces (e.g., "eth0:1") is not used, and instead an interface
 * is assigned possibly multiple addresses, with each address being
 * classified as being primary and secondary.  See the iproute2 
 * documentation for this distinction.
 */
class Ipv4InterfaceAddress 
{
public:
  /**
   * \enum InterfaceAddressScope_e
   * \brief Address scope.
   */
  enum InterfaceAddressScope_e {
    HOST,
    LINK,
    GLOBAL
  };

  Ipv4InterfaceAddress ();

  /**
   * \brief Configure local address, mask and broadcast address
   * \param local the local address
   * \param mask the network mask
   */
  Ipv4InterfaceAddress (Ipv4Address local, Ipv4Mask mask);
  /**
   * Copy constructor
   * \param o the object to copy
   */
  Ipv4InterfaceAddress (const Ipv4InterfaceAddress &o);

  /**
   * \brief Set local address
   * \param local the address
   *
   * \note Functionally identical to `Ipv4InterfaceAddress::SetAddress`.
   *       The method corresponds to the linux variable in_ifaddr.ifa_local
   *       `Ipv4InterfaceAddress::SetAddress` is to be preferred.
   */
  void SetLocal (Ipv4Address local);

  /**
   * \brief Set local address
   * \param address the address
   *
   * \note Functially identical to `Ipv4InterfaceAddress::SetLocal`.
   *       This function is consistent with `Ipv6InterfaceAddress::SetAddress`.
   */
  void SetAddress (Ipv4Address address);

  /**
   * \brief Get the local address
   * \returns the local address
   *
   * \note Functionally identical to `Ipv4InterfaceAddress::GetAddress`.
   *       The method corresponds to the linux variable in_ifaddr.ifa_local
   *       `Ipv4InterfaceAddress::GetAddress` is to be preferred.
   */
  Ipv4Address GetLocal (void) const;

  /**
   * \brief Get the local address
   * \returns the local address
   *
   * \note Functially identical to `Ipv4InterfaceAddress::GetLocal`.
   *       This function is consistent with `Ipv6InterfaceAddress::GetAddress`.
   */
  Ipv4Address GetAddress (void) const;

  /**
   * \brief Set the network mask
   * \param mask the network mask
   */
  void SetMask (Ipv4Mask mask);
  /**
   * \brief Get the network mask
   * \returns the network mask
   */
  Ipv4Mask GetMask (void) const;
  /**
   * \brief Set the broadcast address
   * \param broadcast the broadcast address
   */
  void SetBroadcast (Ipv4Address broadcast);
  /**
   * \brief Get the broadcast address
   * \returns the broadcast address
   */
  Ipv4Address GetBroadcast (void) const;
 
  /**
   * \brief Set the scope.
   * \param scope the scope of address
   */
  void SetScope (Ipv4InterfaceAddress::InterfaceAddressScope_e scope);

  /**
   * \brief Get address scope.
   * \return scope
   */
  Ipv4InterfaceAddress::InterfaceAddressScope_e GetScope (void) const;

  /**
   * \brief Checks if the address is in the same subnet.
   * \param b the address to check
   * \return true if the address is in the same subnet.
   */
  bool IsInSameSubnet (const Ipv4Address b) const;

  /**
   * \brief Check if the address is a secondary address
   *
   * Secondary address is used for multihoming
   * \returns true if the address is secondary
   */
  bool IsSecondary (void) const;

  /**
   * \brief Make the address secondary (used for multihoming)
   */
  void SetSecondary (void);
  /**
   * \brief Make the address primary
   */
  void SetPrimary (void);

private:

  Ipv4Address m_local;     //!< Interface address
  // Note:  m_peer may be added in future when necessary
  // Ipv4Address m_peer;   // Peer destination address (in Linux:  m_address)
  Ipv4Mask m_mask;         //!< Network mask
  Ipv4Address m_broadcast; //!< Broadcast address

  InterfaceAddressScope_e m_scope; //!< Address scope
  bool m_secondary;        //!< For use in multihoming

  /**
   * \brief Equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are equal
   */
  friend bool operator == (Ipv4InterfaceAddress const &a, Ipv4InterfaceAddress const &b);

  /**
   * \brief Not equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are not equal
   */
  friend bool operator != (Ipv4InterfaceAddress const &a, Ipv4InterfaceAddress const &b);
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param addr the Ipv4InterfaceAddress
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, const Ipv4InterfaceAddress &addr);

inline bool operator == (const Ipv4InterfaceAddress &a, const Ipv4InterfaceAddress &b)
{
  return (a.m_local == b.m_local && a.m_mask == b.m_mask &&
          a.m_broadcast == b.m_broadcast && a.m_scope == b.m_scope && a.m_secondary == b.m_secondary);
}
inline bool operator != (const Ipv4InterfaceAddress &a, const Ipv4InterfaceAddress &b)
{
  return (a.m_local != b.m_local || a.m_mask != b.m_mask ||
          a.m_broadcast != b.m_broadcast || a.m_scope != b.m_scope || a.m_secondary != b.m_secondary);
}


} // namespace ns3

#endif /* IPV4_ADDRESS_H */
