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
#include "mac48-address.h"
#include "ns3/address.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include <iomanip>
#include <iostream>
#include <cstring>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Mac48Address");

ATTRIBUTE_HELPER_CPP (Mac48Address);

#define ASCII_a (0x41)
#define ASCII_z (0x5a)
#define ASCII_A (0x61)
#define ASCII_Z (0x7a)
#define ASCII_COLON (0x3a)
#define ASCII_ZERO (0x30)

/**
 * Converts a char to lower case.
 * \param c the char
 * \returns the lower case
 */
static char
AsciiToLowCase (char c)
{
  NS_LOG_FUNCTION (c);
  if (c >= ASCII_a && c <= ASCII_z) {
      return c;
    } else if (c >= ASCII_A && c <= ASCII_Z) {
      return c + (ASCII_a - ASCII_A);
    } else {
      return c;
    }
}


Mac48Address::Mac48Address ()
{
  NS_LOG_FUNCTION (this);
  std::memset (m_address, 0, 6);
}
Mac48Address::Mac48Address (const char *str)
{
  NS_LOG_FUNCTION (this << str);
  int i = 0;
  while (*str != 0 && i < 6) 
    {
      uint8_t byte = 0;
      while (*str != ASCII_COLON && *str != 0) 
        {
          byte <<= 4;
          char low = AsciiToLowCase (*str);
          if (low >= ASCII_a)
            {
              byte |= low - ASCII_a + 10;
            }
          else
            {
              byte |= low - ASCII_ZERO;
            }
          str++;
        }
      m_address[i] = byte;
      i++;
      if (*str == 0) 
        {
          break;
        }
      str++;
    }
  NS_ASSERT (i == 6);
}
void 
Mac48Address::CopyFrom (const uint8_t buffer[6])
{
  NS_LOG_FUNCTION (this << &buffer);
  std::memcpy (m_address, buffer, 6);
}
void 
Mac48Address::CopyTo (uint8_t buffer[6]) const
{
  NS_LOG_FUNCTION (this << &buffer);
  std::memcpy (buffer, m_address, 6);
}

bool 
Mac48Address::IsMatchingType (const Address &address)
{
  NS_LOG_FUNCTION (&address);  
  return address.CheckCompatible (GetType (), 6);
}
Mac48Address::operator Address () const
{
  return ConvertTo ();
}
Address 
Mac48Address::ConvertTo (void) const
{
  NS_LOG_FUNCTION (this);
  return Address (GetType (), m_address, 6);
}
Mac48Address 
Mac48Address::ConvertFrom (const Address &address)
{
  NS_LOG_FUNCTION (&address);
  NS_ASSERT (address.CheckCompatible (GetType (), 6));
  Mac48Address retval;
  address.CopyTo (retval.m_address);
  return retval;
}
Mac48Address 
Mac48Address::Allocate (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint64_t id = 0;
  id++;
  Mac48Address address;
  address.m_address[0] = (id >> 40) & 0xff;
  address.m_address[1] = (id >> 32) & 0xff;
  address.m_address[2] = (id >> 24) & 0xff;
  address.m_address[3] = (id >> 16) & 0xff;
  address.m_address[4] = (id >> 8) & 0xff;
  address.m_address[5] = (id >> 0) & 0xff;
  return address;
}
uint8_t 
Mac48Address::GetType (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint8_t type = Address::Register ();
  return type;
}

bool
Mac48Address::IsBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  return *this == GetBroadcast ();
}
bool 
Mac48Address::IsGroup (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_address[0] & 0x01) == 0x01;
}
Mac48Address
Mac48Address::GetBroadcast (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Mac48Address broadcast = Mac48Address ("ff:ff:ff:ff:ff:ff");
  return broadcast;
}
Mac48Address 
Mac48Address::GetMulticastPrefix (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Mac48Address multicast = Mac48Address ("01:00:5e:00:00:00");
  return multicast;
}
Mac48Address
Mac48Address::GetMulticast6Prefix (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static Mac48Address multicast = Mac48Address ("33:33:00:00:00:00");
  return multicast;
}
Mac48Address 
Mac48Address::GetMulticast (Ipv4Address multicastGroup)
{
  NS_LOG_FUNCTION (multicastGroup);
  Mac48Address etherAddr = Mac48Address::GetMulticastPrefix ();
  //
  // We now have the multicast address in an abstract 48-bit container.  We 
  // need to pull it out so we can play with it.  When we're done, we have the 
  // high order bits in etherBuffer[0], etc.
  //
  uint8_t etherBuffer[6];
  etherAddr.CopyTo (etherBuffer);

  //
  // Now we need to pull the raw bits out of the Ipv4 destination address.
  //
  uint8_t ipBuffer[4];
  multicastGroup.Serialize (ipBuffer);

  //
  // RFC 1112 says that an Ipv4 host group address is mapped to an EUI-48
  // multicast address by placing the low-order 23-bits of the IP address into 
  // the low-order 23 bits of the Ethernet multicast address 
  // 01-00-5E-00-00-00 (hex). 
  //
  etherBuffer[3] |= ipBuffer[1] & 0x7f;
  etherBuffer[4] = ipBuffer[2];
  etherBuffer[5] = ipBuffer[3];

  //
  // Now, etherBuffer has the desired ethernet multicast address.  We have to
  // suck these bits back into the Mac48Address,
  //
  Mac48Address result;
  result.CopyFrom (etherBuffer);
  return result;
}
Mac48Address Mac48Address::GetMulticast (Ipv6Address addr)
{
  NS_LOG_FUNCTION (addr);
  Mac48Address etherAddr = Mac48Address::GetMulticast6Prefix ();
  uint8_t etherBuffer[6];
  uint8_t ipBuffer[16];

  /* a MAC multicast IPv6 address is like 33:33 and the four low bytes */
  /* for 2001:db8::2fff:fe11:ac10 => 33:33:FE:11:AC:10 */
  etherAddr.CopyTo (etherBuffer);
  addr.Serialize (ipBuffer);

  etherBuffer[2] = ipBuffer[12];
  etherBuffer[3] = ipBuffer[13];
  etherBuffer[4] = ipBuffer[14];
  etherBuffer[5] = ipBuffer[15];

  etherAddr.CopyFrom (etherBuffer);

  return etherAddr;
}

std::ostream& operator<< (std::ostream& os, const Mac48Address & address)
{
  uint8_t ad[6];
  address.CopyTo (ad);

  os.setf (std::ios::hex, std::ios::basefield);
  os.fill ('0');
  for (uint8_t i=0; i < 5; i++) 
    {
      os << std::setw (2) << (uint32_t)ad[i] << ":";
    }
  // Final byte not suffixed by ":"
  os << std::setw (2) << (uint32_t)ad[5];
  os.setf (std::ios::dec, std::ios::basefield);
  os.fill (' ');
  return os;
}

std::istream& operator>> (std::istream& is, Mac48Address & address)
{
  std::string v;
  is >> v;

  std::string::size_type col = 0;
  for (uint8_t i = 0; i < 6; ++i)
    {
      std::string tmp;
      std::string::size_type next;
      next = v.find (":", col);
      if (next == std::string::npos)
        {
          tmp = v.substr (col, v.size ()-col);
          address.m_address[i] = strtoul (tmp.c_str(), 0, 16);
          break;
        }
      else
        {
          tmp = v.substr (col, next-col);
          address.m_address[i] = strtoul (tmp.c_str(), 0, 16);
          col = next + 1;
        }
    }
  return is;
}


} // namespace ns3
