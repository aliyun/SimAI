/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2008 Louis Pasteur University
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

#ifndef IPV6_ADDRESS_H
#define IPV6_ADDRESS_H

#include <stdint.h>
#include <cstring>

#include <ostream>

#include "ns3/attribute-helper.h"
#include "ns3/address.h"
#include "ns3/ipv4-address.h"
#include "ns3/deprecated.h"
#include "mac8-address.h"

namespace ns3 { 

class Ipv6Prefix;
class Mac16Address;
class Mac48Address;
class Mac64Address;

/**
 * \ingroup address
 * \class Ipv6Address
 * \brief Describes an IPv6 address.
 * \see Ipv6Prefix
 * \see attribute_Ipv6Address
 */
class Ipv6Address
{
public:
  /**
   * \brief Default constructor.
   */
  Ipv6Address ();

  /**
   * \brief Constructs an Ipv6Address by parsing the input C-string.
   * \param address the C-string containing the IPv6 address (e.g. 2001:660:4701::1).
   */
  Ipv6Address (char const* address);

  /**
   * \brief Constructs an Ipv6Address by using the input 16 bytes.
   * \param address the 128-bit address
   * \warning the parameter must point on a 16 bytes integer array!
   */
  Ipv6Address (uint8_t address[16]);

  /**
   * \brief Copy constructor.
   * \param addr Ipv6Address object
   */
  Ipv6Address (Ipv6Address const & addr);

  /**
   * \brief Copy constructor.
   * \param addr Ipv6Address pointer
   */
  Ipv6Address (Ipv6Address const* addr);

  /**
   * \brief Destructor.
   */
  ~Ipv6Address ();

  /**
   * \brief Sets an Ipv6Address by parsing the input C-string.
   * \param address the C-string containing the IPv6 address (e.g. 2001:660:4701::1).
   */
  void Set (char const* address);

  /**
   * \brief Set an Ipv6Address by using the input 16 bytes.
   *
   * \param address the 128-bit address
   * \warning the parameter must point on a 16 bytes integer array!
   */
  void Set (uint8_t address[16]);

  /**
   * \brief Serialize this address to a 16-byte buffer.
   * \param buf the output buffer to which this address gets overwritten with this
   * Ipv6Address
   */
  void Serialize (uint8_t buf[16]) const;

  /**
   * \brief Deserialize this address.
   * \param buf buffer to read address from
   * \return an Ipv6Address
   */
  static Ipv6Address Deserialize (const uint8_t buf[16]);

  /**
   * \brief Make the solicited IPv6 address.
   * \param addr the IPv6 address
   * \return Solicited IPv6 address
   */
  static Ipv6Address MakeSolicitedAddress (Ipv6Address addr);

  /**
   * \brief Make the Ipv4-mapped IPv6 address.
   * \param addr the IPv4 address
   * \return Ipv4-mapped IPv6 address
   */
  static Ipv6Address MakeIpv4MappedAddress (Ipv4Address addr);

  /**
   * \brief Return the Ipv4 address.
   * \return Ipv4 address
   */
  Ipv4Address GetIpv4MappedAddress () const;

  /**
   * \brief Make the autoconfigured IPv6 address from a Mac address.
   *
   * Actually the MAC supported are: Mac8, Mac16, Mac48, and Mac64.
   *
   * \param addr the MAC address.
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredAddress (Address addr, Ipv6Address prefix);

  /**
   * \brief Make the autoconfigured IPv6 address from a Mac address.
   *
   * Actually the MAC supported are: Mac8, Mac16, Mac48, and Mac64.
   *
   * \param addr the MAC address.
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */

  static Ipv6Address MakeAutoconfiguredAddress (Address addr, Ipv6Prefix prefix);

  /**
   * \brief Make the autoconfigured IPv6 address with Mac16Address.
   *
   * The EUI-64 scheme used is based on the \RFC{4944}.
   *
   * \param addr the MAC address (16 bits).
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredAddress (Mac16Address addr, Ipv6Address prefix);

  /**
   * \brief Make the autoconfigured IPv6 address with Mac48Address.
   *
   * The EUI-64 scheme used is based on \RFC{2464}.
   *
   * \param addr the MAC address (48 bits).
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredAddress (Mac48Address addr, Ipv6Address prefix);

  /**
   * \brief Make the autoconfigured IPv6 address with Mac64Address.
   * \param addr the MAC address (64 bits).
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredAddress (Mac64Address addr, Ipv6Address prefix);

  /**
   * \brief Make the autoconfigured IPv6 address with Mac8Address.
   *
   * The EUI-64 scheme used is loosely based on the \RFC{2464}.
   *
   * \param addr the Mac8Address address (8 bits).
   * \param prefix the IPv6 prefix
   * \return autoconfigured IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredAddress (Mac8Address addr, Ipv6Address prefix);

  /**
   * \brief Make the autoconfigured link-local IPv6 address from a Mac address.
   *
   * Actually the MAC supported are: Mac8, Mac16, Mac48, and Mac64.
   *
   * \param mac the MAC address.
   * \return autoconfigured link-local IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredLinkLocalAddress (Address mac);

  /**
   * \brief Make the autoconfigured link-local IPv6 address with Mac16Address.
   *
   * The EUI-64 scheme used is based on the \RFC{4944}.
   *
   * \param mac the MAC address (16 bits).
   * \return autoconfigured link-local IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredLinkLocalAddress (Mac16Address mac);

  /**
   * \brief Make the autoconfigured link-local IPv6 address with Mac48Address.
   *
   * The EUI-64 scheme used is based on \RFC{2464}.
   *
   * \param mac the MAC address (48 bits).
   * \return autoconfigured link-local IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredLinkLocalAddress (Mac48Address mac);

  /**
   * \brief Make the autoconfigured link-local IPv6 address with Mac64Address.
   * \param mac the MAC address (64 bits).
   * \return autoconfigured link-local IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredLinkLocalAddress (Mac64Address mac);

  /**
   * \brief Make the autoconfigured link-local IPv6 address with Mac8Address.
   *
   * The EUI-64 scheme used is loosely based on the \RFC{2464}.
   *
   * \param mac the MAC address (8 bits).
   * \return autoconfigured link-local IPv6 address
   */
  static Ipv6Address MakeAutoconfiguredLinkLocalAddress (Mac8Address mac);

  /**
   * \brief Print this address to the given output stream.
   *
   * The print format is in the typical "2001:660:4701::1".
   * \param os the output stream to which this Ipv6Address is printed
   */
  void Print (std::ostream& os) const;

  /**
   * \brief If the IPv6 address is localhost (::1).
   * \return true if localhost, false otherwise
   */
  bool IsLocalhost () const;

  /**
   * \brief If the IPv6 address is multicast (ff00::/8).
   * \return true if multicast, false otherwise
   */
  bool IsMulticast () const;

  /**
   * \brief If the IPv6 address is link-local multicast (ff02::/16).
   * \return true if link-local multicast, false otherwise
   */
  bool IsLinkLocalMulticast () const;

  /**
   * \brief If the IPv6 address is "all nodes multicast" (ff02::1/8).
   * \return true if "all nodes multicast", false otherwise
   */
  bool IsAllNodesMulticast () const;

  /**
   * \brief If the IPv6 address is "all routers multicast" (ff02::2/8).
   * \return true if "all routers multicast", false otherwise
   */
  bool IsAllRoutersMulticast () const;

  /**
   * \brief If the IPv6 address is a link-local address (fe80::/64).
   * \return true if the address is link-local, false otherwise
   */
  bool IsLinkLocal () const;

  /**
   * \brief If the IPv6 address is a Solicited multicast address.
   * \return true if it is, false otherwise
   */
  bool IsSolicitedMulticast () const;

  /**
   * \brief If the IPv6 address is the "Any" address.
   * \return true if it is, false otherwise
   */
  bool IsAny () const;

  /**
   * \brief If the IPv6 address is a documentation address (2001:DB8::/32).
   * \return true if the address is documentation, false otherwise
   */
  bool IsDocumentation () const;

  /**
   * \brief Compares an address and a prefix.
   * \param prefix the prefix to compare with
   * \return true if the address has the given prefix
   */
  bool HasPrefix (Ipv6Prefix const& prefix) const;

  /**
   * \brief Combine this address with a prefix.
   * \param prefix a IPv6 prefix
   * \return an IPv6 address that is this address combined
   * (bitwise AND) with a prefix, yielding an IPv6 network address.
   */
  Ipv6Address CombinePrefix (Ipv6Prefix const & prefix) const;

  /**
   * \brief If the Address matches the type.
   * \param address other address
   * \return true if the type matches, false otherwise
   */
  static bool IsMatchingType (const Address& address);

  /**
   * \brief If the address is an IPv4-mapped address
   * \return true if address is an IPv4-mapped address, otherwise false.
   */
  bool IsIpv4MappedAddress() const;

  /**
   * \brief Convert to Address object
   */
  operator Address () const;

  /**
   * \brief Convert the Address object into an Ipv6Address ones.
   * \param address address to convert
   * \return an Ipv6Address
   */
  static Ipv6Address ConvertFrom (const Address& address);

  /**
   * \return true if address is initialized (i.e., set to something), false otherwise
   */
  bool IsInitialized (void) const;

  /**
   * \brief Get the 0 (::) Ipv6Address.
   * \return the :: Ipv6Address representation
   */
  static Ipv6Address GetZero ();

  /**
   * \brief Get the "any" (::) Ipv6Address.
   * \return the "any" (::) Ipv6Address
   */
  static Ipv6Address GetAny ();

  /**
   * \brief Get the "all nodes multicast" address.
   * \return the "ff02::2/8" Ipv6Address representation
   */
  static Ipv6Address GetAllNodesMulticast ();

  /**
   * \brief Get the "all routers multicast" address.
   * \return the "ff02::2/8" Ipv6Address representation
   */
  static Ipv6Address GetAllRoutersMulticast ();

  /**
   * \brief Get the "all hosts multicast" address.
   * \return the "ff02::3/8" Ipv6Address representation
   */
  static Ipv6Address GetAllHostsMulticast ();

  /**
   * \brief Get the loopback address.
   * \return the "::1/128" Ipv6Address representation.
   */
  static Ipv6Address GetLoopback ();

  /**
   * \brief Get the "all-1" IPv6 address (ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff).
   * \return all-1 Ipv6Address representation
   */
  static Ipv6Address GetOnes ();

  /**
   * \brief Get the bytes corresponding to the address.
   * \param buf buffer to store the data
   */
  void GetBytes (uint8_t buf[16]) const;

private:
  /**
   * \brief convert the IPv6Address object to an Address object.
   * \return the Address object corresponding to this object.
   */
  Address ConvertTo (void) const;

  /**
   * \brief Return the Type of address.
   * \return type of address
   */
  static uint8_t GetType (void);

  /**
   * \brief The address representation on 128 bits (16 bytes).
   */
  uint8_t m_address[16];
  bool m_initialized; //!< IPv6 address has been explicitly initialized to a valid value.

  /**
   * \brief Equal to operator.
   *
   * \param a the first operand.
   * \param b the first operand.
   * \returns true if the operands are equal.
   */
  friend bool operator == (Ipv6Address const &a, Ipv6Address const &b);

  /**
   * \brief Not equal to operator.
   *
   * \param a the first operand.
   * \param b the first operand.
   * \returns true if the operands are not equal.
   */
  friend bool operator != (Ipv6Address const &a, Ipv6Address const &b);

  /**
   * \brief Less than to operator.
   *
   * \param a the first operand.
   * \param b the first operand.
   * \returns true if the first operand is less than the second.
   */
  friend bool operator < (Ipv6Address const &a, Ipv6Address const &b);
};

/**
 * \ingroup address
 * \class Ipv6Prefix
 * \brief Describes an IPv6 prefix. It is just a bitmask like Ipv4Mask.
 * \see Ipv6Address
 * \see attribute_Ipv6Prefix
 */
class Ipv6Prefix
{
public:
  /**
   * \brief Default constructor.
   */
  Ipv6Prefix ();

  /**
   * \brief Constructs an Ipv6Prefix by using the input 16 bytes.
   *
   * The prefix length is calculated as the minimum prefix length, i.e.,
   * 2001:db8:cafe:: will have a 47 bit prefix length.
   *
   * \param prefix the 128-bit prefix
   */
  Ipv6Prefix (uint8_t prefix[16]);

  /**
   * \brief Constructs an Ipv6Prefix by using the input string.
   *
   * The prefix length is calculated as the minimum prefix length, i.e.,
   * 2001:db8:cafe:: will have a 47 bit prefix length.
   *
   * \param prefix the 128-bit prefix
   */
  Ipv6Prefix (char const* prefix);

  /**
   * \brief Constructs an Ipv6Prefix by using the input 16 bytes.
   * \param prefix the 128-bit prefix
   * \param prefixLength the prefix length
   */
  Ipv6Prefix (uint8_t prefix[16], uint8_t prefixLength);

  /**
   * \brief Constructs an Ipv6Prefix by using the input string.
   * \param prefix the 128-bit prefix
   * \param prefixLength the prefix length
   */
  Ipv6Prefix (char const* prefix, uint8_t prefixLength);

  /**
   * \brief Constructs an Ipv6Prefix by using the input number of bits.
   * \param prefix number of bits of the prefix (0 - 128)
   * \note A valid number of bits is between 0 and 128).
   */
  Ipv6Prefix (uint8_t prefix);

  /**
   * \brief Copy constructor.
   * \param prefix Ipv6Prefix object
   */
  Ipv6Prefix (Ipv6Prefix const& prefix);

  /**
   * \brief Copy constructor.
   * \param prefix Ipv6Prefix pointer
   */
  Ipv6Prefix (Ipv6Prefix const* prefix);

  /**
   * \brief Destructor.
   */
  ~Ipv6Prefix ();

  /**
   * \brief If the Address match the type.
   * \param a a first address
   * \param b a second address
   * \return true if the type match, false otherwise
   */
  bool IsMatch (Ipv6Address a, Ipv6Address b) const;

  /**
   * \brief Get the bytes corresponding to the prefix.
   * \param buf buffer to store the data
   */
  void GetBytes (uint8_t buf[16]) const;

  /**
   * \brief Convert the Prefix into an IPv6 Address.
   * \return an IPv6 address representing the prefix
   */
  Ipv6Address ConvertToIpv6Address () const;

  /**
   * \brief Get prefix length.
   * \return prefix length
   */
  uint8_t GetPrefixLength () const;

  /**
   * \brief Set prefix length.
   * \param prefixLength the prefix length
   */
  void SetPrefixLength (uint8_t prefixLength);

   /**
    * \brief Get the minimum prefix length, i.e., 128 - the length of the largest sequence trailing zeroes.
    * \return minimum prefix length
    */
  uint8_t GetMinimumPrefixLength () const;

  /**
   * \brief Print this address to the given output stream.
   *
   * The print format is in the typical "2001:660:4701::1".
   * \param os the output stream to which this Ipv6Address is printed
   */
  void Print (std::ostream &os) const;

  /**
   * \brief Get the loopback prefix ( /128).
   * \return a Ipv6Prefix corresponding to loopback prefix
   */
  static Ipv6Prefix GetLoopback ();

  /**
   * \brief Get the "all-1" IPv6 mask (ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff).
   * \return /128 Ipv6Prefix representation
   */
  static Ipv6Prefix GetOnes ();

  /**
   * \brief Get the zero prefix ( /0).
   * \return an Ipv6Prefix
   */
  static Ipv6Prefix GetZero ();

private:
  /**
   * \brief The prefix representation.
   */
  uint8_t m_prefix[16];

  /**
   * \brief The prefix length.
   */
  uint8_t m_prefixLength;

  /**
   * \brief Equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are equal
   */
  friend bool operator == (Ipv6Prefix const &a, Ipv6Prefix const &b);

  /**
   * \brief Not equal to operator.
   *
   * \param a the first operand
   * \param b the first operand
   * \returns true if the operands are not equal
   */
  friend bool operator != (Ipv6Prefix const &a, Ipv6Prefix const &b);
};

ATTRIBUTE_HELPER_HEADER (Ipv6Address);
ATTRIBUTE_HELPER_HEADER (Ipv6Prefix);

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param address the Ipv6Address
 * \returns the reference to the output stream
 */
std::ostream & operator << (std::ostream& os, Ipv6Address const& address);

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param prefix the Ipv6Prefix
 * \returns the reference to the output stream
 */
std::ostream & operator << (std::ostream& os, Ipv6Prefix const& prefix);

/**
 * \brief Stream extraction operator.
 *
 * \param is the reference to the input stream
 * \param address the Ipv6Address
 * \returns the reference to the input stream
 */
std::istream & operator >> (std::istream &is, Ipv6Address &address);

/**
 * \brief Stream extraction operator.
 *
 * \param is the reference to the input stream
 * \param prefix the Ipv6Preofix
 * \returns the reference to the input stream
 */
std::istream & operator >> (std::istream &is, Ipv6Prefix &prefix);

inline bool operator == (const Ipv6Address& a, const Ipv6Address& b)
{
  return (!std::memcmp (a.m_address, b.m_address, 16));
}

inline bool operator != (const Ipv6Address& a, const Ipv6Address& b)
{
  return std::memcmp (a.m_address, b.m_address, 16);
}

inline bool operator < (const Ipv6Address& a, const Ipv6Address& b)
{
  return (std::memcmp (a.m_address, b.m_address, 16) < 0);
}

inline bool operator == (const Ipv6Prefix& a, const Ipv6Prefix& b)
{
  return (!std::memcmp (a.m_prefix, b.m_prefix, 16));
}

inline bool operator != (const Ipv6Prefix& a, const Ipv6Prefix& b)
{
  return std::memcmp (a.m_prefix, b.m_prefix, 16);
}

/**
 * \class Ipv6AddressHash
 * \brief Hash function class for IPv6 addresses.
 */
class Ipv6AddressHash
{
public:
  /**
   * \brief Returns the hash of an IPv6 address.
   * \param x IPv6 address to hash
   * \returns the hash of the address
   */
  size_t operator () (Ipv6Address const &x) const;
};

} /* namespace ns3 */

#endif /* IPV6_ADDRESS_H */

