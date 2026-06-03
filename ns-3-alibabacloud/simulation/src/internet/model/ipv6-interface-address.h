/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 */

#ifndef IPV6_INTERFACE_ADDRESS_H
#define IPV6_INTERFACE_ADDRESS_H 

#include <stdint.h>

#include "ns3/ipv6-address.h"

namespace ns3
{

/**
 * \ingroup address
 * \ingroup ipv6
 *
 * \brief IPv6 address associated with an interface.
 */
class Ipv6InterfaceAddress
{
public:
  /**
   * \enum State_e
   * \brief State of an address associated with an interface.
   */
  enum State_e
  {
    TENTATIVE, /**< Address is tentative, no packet can be sent unless DAD finished */
    DEPRECATED, /**< Address is deprecated and should not be used */
    PREFERRED, /**< Preferred address */
    PERMANENT, /**< Permanent address */
    HOMEADDRESS, /**< Address is a HomeAddress */
    TENTATIVE_OPTIMISTIC, /**< Address is tentative but we are optimistic so we can send packet even if DAD is not yet finished */
    INVALID, /**< Invalid state (after a DAD failed) */
  };

  /**
   * \enum Scope_e
   * \brief Address scope.
   */
  enum Scope_e
  {
    HOST, /**< Localhost (::1/128) */
    LINKLOCAL, /**< Link-local address (fe80::/64) */
    GLOBAL, /**< Global address (2000::/3) */
  };

  /**
   * \brief Default constructor.
   */
  Ipv6InterfaceAddress ();

  /**
   * \brief Constructor. Prefix is 64 by default.
   * \param address the IPv6 address to set
   */
  Ipv6InterfaceAddress (Ipv6Address address);

  /**
   * \brief Constructor.
   * \param address IPv6 address to set
   * \param prefix IPv6 prefix
   */
  Ipv6InterfaceAddress (Ipv6Address address, Ipv6Prefix prefix);

  /**
   * \brief Constructor.
   * \param address IPv6 address to set
   * \param prefix IPv6 prefix
   * \param onLink on-link property
   */
  Ipv6InterfaceAddress (Ipv6Address address, Ipv6Prefix prefix, bool onLink);

  /**
   * \brief Copy constructor.
   * \param o object to copy
   */
  Ipv6InterfaceAddress (const Ipv6InterfaceAddress& o);

  /**
   * \brief Destructor.
   */
  ~Ipv6InterfaceAddress ();

  /**
   * \brief Set IPv6 address (and scope).
   * \param address IPv6 address to set
   */
  void SetAddress (Ipv6Address address);

  /**
   * \brief Get the IPv6 address.
   * \return IPv6 address
   */
  Ipv6Address GetAddress () const;

  /**
   * \brief Get the IPv6 prefix.
   * \return IPv6 prefix
   */
  Ipv6Prefix GetPrefix () const;

  /**
   * \brief Set the state.
   * \param state the state
   */
  void SetState (Ipv6InterfaceAddress::State_e state);

  /**
   * \brief Get the address state.
   * \return address state
   */
  Ipv6InterfaceAddress::State_e GetState () const;

  /**
   * \brief Set the scope.
   * \param scope the scope of address
   */
  void SetScope (Ipv6InterfaceAddress::Scope_e scope);

  /**
   * \brief Get address scope.
   * \return scope
   */
  Ipv6InterfaceAddress::Scope_e GetScope () const;

  /**
   * \brief Checks if the address is in the same subnet.
   * \param b the address to check
   * \return true if the address is in the same subnet.
   */
  bool IsInSameSubnet (Ipv6Address b) const;

  /**
   * \brief Set the latest DAD probe packet UID.
   * \param uid packet uid
   */
  void SetNsDadUid (uint32_t uid);

  /**
   * \brief Get the latest DAD probe packet UID.
   * \return uid
   */
  uint32_t GetNsDadUid () const;

  /**
   * \brief Get the on-link property.
   * \param onLink the on-link property
   */
  void SetOnLink (bool onLink);

  /**
   * \brief Get the on-link property.
   * \return on-link flag
   */
  bool GetOnLink () const;

#if 0
  /**
   * \brief Start the DAD timer.
   * \param interface interface
   */
  void StartDadTimer (Ptr<Ipv6Interface> interface);

  /**
   * \brief Stop the DAD timer.
   */
  void StopDadTimer ();
#endif

private:
  /**
   * \brief The IPv6 address.
   */
  Ipv6Address m_address;

  /**
   * \brief The IPv6 prefix.
   */
  Ipv6Prefix m_prefix;

  /**
   * \brief State of the address.
   */
  State_e m_state;

  /**
   * \brief Scope of the address.
   */
  Scope_e m_scope;

  /**
   * \brief The address belongs to an on-link network.
   */
  bool m_onLink;

  /**
   * \brief Equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are equal
   */
  friend bool operator == (Ipv6InterfaceAddress const& a, Ipv6InterfaceAddress const& b);

  /**
   * \brief Not equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are not equal
   */
  friend bool operator != (Ipv6InterfaceAddress const& a, Ipv6InterfaceAddress const& b);

  /**
   * \brief Last DAD probe packet UID.
   */
  uint32_t m_nsDadUid;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param addr the Ipv6InterfaceAddress
 * \returns the reference to the output stream
 */
std::ostream& operator<< (std::ostream& os, const Ipv6InterfaceAddress &addr);

/* follow Ipv4InterfaceAddress way, maybe not inline them */
inline bool operator == (const Ipv6InterfaceAddress& a, const Ipv6InterfaceAddress& b)
{
  return (a.m_address == b.m_address && a.m_prefix == b.m_prefix &&
          a.m_state == b.m_state && a.m_scope == b.m_scope);
}

inline bool operator != (const Ipv6InterfaceAddress& a, const Ipv6InterfaceAddress& b)
{
  return (a.m_address != b.m_address || a.m_prefix != b.m_prefix ||
          a.m_state != b.m_state || a.m_scope != b.m_scope);
}

} /* namespace ns3 */

#endif /* IPV6_INTERFACE_ADDRESS_H */

