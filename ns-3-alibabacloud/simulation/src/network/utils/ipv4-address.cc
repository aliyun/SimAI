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

#include <cstdlib>
#include "ns3/log.h"
#include "ipv4-address.h"
#include "ns3/assert.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4Address");

#define ASCII_DOT (0x2e)
#define ASCII_ZERO (0x30)
#define ASCII_SLASH (0x2f)

/**
 * \brief Converts a string representing an IP address into the address
 * \param address the address string
 * \returns the address
 */
static uint32_t 
AsciiToIpv4Host (char const *address)
{
  NS_LOG_FUNCTION (&address);
  uint32_t host = 0;
  uint8_t numberOfDots = 0;
  char const *ptr = address;

  NS_ASSERT_MSG (*ptr != ASCII_DOT, "Error, can not build an IPv4 address from an invalid string: " << address);
  while (true) 
    {
      uint8_t byte = 0;
      while (*ptr != ASCII_DOT && *ptr != 0)
        {
          byte *= 10;
          byte += *ptr - ASCII_ZERO;
          ptr++;
        }
      host <<= 8;
      host |= byte;
      if (*ptr == 0)
        {
          break;
        }
      ptr++;
      NS_ASSERT_MSG (*ptr != ASCII_DOT, "Error, can not build an IPv4 address from an invalid string: " << address);
      numberOfDots ++;
    }
  NS_ASSERT_MSG (*(ptr-1) != ASCII_DOT, "Error, can not build an IPv4 address from an invalid string: " << address);
  NS_ASSERT_MSG (numberOfDots == 3, "Error, can not build an IPv4 address from an invalid string: " << address);

  return host;
}

} // namespace ns3

namespace ns3 {


Ipv4Mask::Ipv4Mask ()
  : m_mask (0x66666666)
{
  NS_LOG_FUNCTION (this);
}

Ipv4Mask::Ipv4Mask (uint32_t mask)
  : m_mask (mask)
{
  NS_LOG_FUNCTION (this << mask);
}

Ipv4Mask::Ipv4Mask (char const *mask)
{
  NS_LOG_FUNCTION (this << mask);
  if (*mask == ASCII_SLASH)
    {
      uint32_t plen = static_cast<uint32_t> (std::atoi (++mask));
      NS_ASSERT (plen <= 32);
      if (plen > 0)
        {
          m_mask = 0xffffffff << (32 - plen);
        }
      else
        {
          m_mask = 0;
        }
    }
  else
    {
      m_mask = AsciiToIpv4Host (mask);
    }
}

bool 
Ipv4Mask::IsMatch (Ipv4Address a, Ipv4Address b) const
{
  NS_LOG_FUNCTION (this << a << b);
  if ((a.Get () & m_mask) == (b.Get () & m_mask)) {
      return true;
    } else {
      return false;
    }
}

uint32_t 
Ipv4Mask::Get (void) const
{
  NS_LOG_FUNCTION (this);
  return m_mask;
}
void 
Ipv4Mask::Set (uint32_t mask)
{
  NS_LOG_FUNCTION (this << mask);
  m_mask = mask;
}
uint32_t 
Ipv4Mask::GetInverse (void) const
{
  NS_LOG_FUNCTION (this);
  return ~m_mask;
}

void 
Ipv4Mask::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << ((m_mask >> 24) & 0xff) << "."
     << ((m_mask >> 16) & 0xff) << "."
     << ((m_mask >> 8) & 0xff) << "."
     << ((m_mask >> 0) & 0xff);
}


Ipv4Mask
Ipv4Mask::GetLoopback (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Mask loopback = Ipv4Mask ("255.0.0.0");
  return loopback;
}
Ipv4Mask
Ipv4Mask::GetZero (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Mask zero = Ipv4Mask ("0.0.0.0");
  return zero;
}
Ipv4Mask
Ipv4Mask::GetOnes (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Mask ones = Ipv4Mask ("255.255.255.255");
  return ones;
}

uint16_t
Ipv4Mask::GetPrefixLength (void) const
{
  NS_LOG_FUNCTION (this);
  uint16_t tmp = 0;
  uint32_t mask = m_mask;
  while (mask != 0 ) 
    {
      mask = mask << 1;
      tmp++;
    }
  return tmp; 
}

/**
 *  Value of a not-yet-initialized IPv4 address, corresponding to 102.102.102.102.
 *  This is totally arbitrary.
 */
static constexpr uint32_t UNINITIALIZED = 0x66666666U;

Ipv4Address::Ipv4Address ()
  : m_address (UNINITIALIZED), m_initialized (false)
{
  NS_LOG_FUNCTION (this);
}
Ipv4Address::Ipv4Address (uint32_t address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;
  m_initialized = true;
}
Ipv4Address::Ipv4Address (char const *address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = AsciiToIpv4Host (address);
  m_initialized = true;
}

uint32_t
Ipv4Address::Get (void) const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}
void
Ipv4Address::Set (uint32_t address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = address;
  m_initialized = true;
}
void
Ipv4Address::Set (char const *address)
{
  NS_LOG_FUNCTION (this << address);
  m_address = AsciiToIpv4Host (address);
  m_initialized = true;
}

Ipv4Address
Ipv4Address::CombineMask (Ipv4Mask const &mask) const
{
  NS_LOG_FUNCTION (this << mask);
  return Ipv4Address (Get () & mask.Get ());
}

Ipv4Address 
Ipv4Address::GetSubnetDirectedBroadcast (Ipv4Mask const &mask) const
{
  NS_LOG_FUNCTION (this << mask);
  if (mask == Ipv4Mask::GetOnes ())
    {
      NS_ASSERT_MSG (false, "Trying to get subnet-directed broadcast address with an all-ones netmask");
    }
  return Ipv4Address (Get () | mask.GetInverse ());
}

bool
Ipv4Address::IsSubnetDirectedBroadcast (Ipv4Mask const &mask) const
{
  NS_LOG_FUNCTION (this << mask);
  if (mask == Ipv4Mask::GetOnes ())
    {
      // If the mask is 255.255.255.255, there is no subnet directed
      // broadcast for this address.
      return false;
    }
  return ( (Get () | mask.GetInverse ()) == Get () );
}

bool
Ipv4Address::IsInitialized (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_initialized);
}

bool
Ipv4Address::IsAny (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_address == 0x00000000U);
}

bool
Ipv4Address::IsLocalhost (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_address == 0x7f000001U);
}

bool
Ipv4Address::IsBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_address == 0xffffffffU);
}

bool 
Ipv4Address::IsMulticast (void) const
{
//
// Multicast addresses are defined as ranging from 224.0.0.0 through 
// 239.255.255.255 (which is E0000000 through EFFFFFFF in hex).
//
  NS_LOG_FUNCTION (this);
  return (m_address >= 0xe0000000 && m_address <= 0xefffffff);
}

bool 
Ipv4Address::IsLocalMulticast (void) const
{
  NS_LOG_FUNCTION (this);
  // Link-Local multicast address is 224.0.0.0/24
  return (m_address & 0xffffff00) == 0xe0000000;
}

void
Ipv4Address::Serialize (uint8_t buf[4]) const
{
  NS_LOG_FUNCTION (this << &buf);
  buf[0] = (m_address >> 24) & 0xff;
  buf[1] = (m_address >> 16) & 0xff;
  buf[2] = (m_address >> 8) & 0xff;
  buf[3] = (m_address >> 0) & 0xff;
}
Ipv4Address 
Ipv4Address::Deserialize (const uint8_t buf[4])
{
  NS_LOG_FUNCTION (&buf);
  Ipv4Address ipv4;
  ipv4.m_address = 0;
  ipv4.m_address |= buf[0];
  ipv4.m_address <<= 8;
  ipv4.m_address |= buf[1];
  ipv4.m_address <<= 8;
  ipv4.m_address |= buf[2];
  ipv4.m_address <<= 8;
  ipv4.m_address |= buf[3];
  ipv4.m_initialized = true;

  return ipv4;
}

void 
Ipv4Address::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this);
  os << ((m_address >> 24) & 0xff) << "."
     << ((m_address >> 16) & 0xff) << "."
     << ((m_address >> 8) & 0xff) << "."
     << ((m_address >> 0) & 0xff);
}

bool 
Ipv4Address::IsMatchingType (const Address &address)
{
  NS_LOG_FUNCTION (&address);
  return address.CheckCompatible (GetType (), 4);
}
Ipv4Address::operator Address () const
{
  return ConvertTo ();
}

Address 
Ipv4Address::ConvertTo (void) const
{
  NS_LOG_FUNCTION (this);
  uint8_t buf[4];
  Serialize (buf);
  return Address (GetType (), buf, 4);
}

Ipv4Address
Ipv4Address::ConvertFrom (const Address &address)
{
  NS_LOG_FUNCTION (&address);
  NS_ASSERT (address.CheckCompatible (GetType (), 4));
  uint8_t buf[4];
  address.CopyTo (buf);
  return Deserialize (buf);
}

uint8_t 
Ipv4Address::GetType (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint8_t type = Address::Register ();
  return type;
}

Ipv4Address 
Ipv4Address::GetZero (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Address zero ("0.0.0.0");
  return zero;
}
Ipv4Address 
Ipv4Address::GetAny (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Address any ("0.0.0.0");
  return any;
}
Ipv4Address 
Ipv4Address::GetBroadcast (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Ipv4Address broadcast ("255.255.255.255");
  return broadcast;
}
Ipv4Address 
Ipv4Address::GetLoopback (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  Ipv4Address loopback ("127.0.0.1");
  return loopback;
}

size_t Ipv4AddressHash::operator() (Ipv4Address const &x) const
{ 
  return std::hash<uint32_t>()(x.Get ());
}

std::ostream& operator<< (std::ostream& os, Ipv4Address const& address)
{
  address.Print (os);
  return os;
}
std::ostream& operator<< (std::ostream& os, Ipv4Mask const& mask)
{
  mask.Print (os);
  return os;
}
std::istream & operator >> (std::istream &is, Ipv4Address &address)
{
  std::string str;
  is >> str;
  address = Ipv4Address (str.c_str ());
  return is;
}
std::istream & operator >> (std::istream &is, Ipv4Mask &mask)
{
  std::string str;
  is >> str;
  mask = Ipv4Mask (str.c_str ());
  return is;
}

ATTRIBUTE_HELPER_CPP (Ipv4Address);
ATTRIBUTE_HELPER_CPP (Ipv4Mask);

} // namespace ns3
