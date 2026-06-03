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

#include "ns3/assert.h"
#include "ns3/log.h"
#include "address.h"
#include <cstring>
#include <iostream>
#include <iomanip>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Address");

Address::Address ()
  : m_type (0),
    m_len (0)
{
  // Buffer left uninitialized
  NS_LOG_FUNCTION (this);

}

Address::Address (uint8_t type, const uint8_t *buffer, uint8_t len)
  : m_type (type),
    m_len (len)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type) << &buffer << static_cast<uint32_t> (len));
  NS_ASSERT (m_len <= MAX_SIZE);
  std::memcpy (m_data, buffer, m_len);
}
Address::Address (const Address & address)
  : m_type (address.m_type),
    m_len (address.m_len)
{
  NS_ASSERT (m_len <= MAX_SIZE);
  std::memcpy (m_data, address.m_data, m_len);
}
Address &
Address::operator = (const Address &address)
{
  NS_ASSERT (m_len <= MAX_SIZE);
  m_type = address.m_type;
  m_len = address.m_len;
  NS_ASSERT (m_len <= MAX_SIZE);
  std::memcpy (m_data, address.m_data, m_len);
  return *this;
}

bool
Address::IsInvalid (void) const
{
  NS_LOG_FUNCTION (this);
  return m_len == 0 && m_type == 0;
}

uint8_t 
Address::GetLength (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_len <= MAX_SIZE);
  return m_len;
}
uint32_t
Address::CopyTo (uint8_t buffer[MAX_SIZE]) const
{
  NS_LOG_FUNCTION (this << &buffer);
  NS_ASSERT (m_len <= MAX_SIZE);
  std::memcpy (buffer, m_data, m_len);
  return m_len;
}
uint32_t
Address::CopyAllTo (uint8_t *buffer, uint8_t len) const
{
  NS_LOG_FUNCTION (this << &buffer << static_cast<uint32_t> (len));
  NS_ASSERT (len - m_len > 1);
  buffer[0] = m_type;
  buffer[1] = m_len;
  std::memcpy (buffer + 2, m_data, m_len);
  return m_len + 2;
}

uint32_t
Address::CopyFrom (const uint8_t *buffer, uint8_t len)
{
  NS_LOG_FUNCTION (this << &buffer << static_cast<uint32_t> (len));
  NS_ASSERT (len <= MAX_SIZE);
  std::memcpy (m_data, buffer, len);
  m_len = len;
  return m_len;
}
uint32_t
Address::CopyAllFrom (const uint8_t *buffer, uint8_t len)
{
  NS_LOG_FUNCTION (this << &buffer << static_cast<uint32_t> (len));
  NS_ASSERT (len >= 2);
  m_type = buffer[0];
  m_len = buffer[1];

  NS_ASSERT (len - m_len > 1);
  std::memcpy (m_data, buffer + 2, m_len);
  return m_len + 2;
}
bool 
Address::CheckCompatible (uint8_t type, uint8_t len) const
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type) << static_cast<uint32_t> (len));
  NS_ASSERT (len <= MAX_SIZE);
  /// \internal
  /// Mac address type/length detection is discussed in \bugid{1568}
  return (m_len == len && m_type == type) || (m_len >= len && m_type == 0);
}
bool 
Address::IsMatchingType (uint8_t type) const
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type));
  return m_type == type;
}

uint8_t 
Address::Register (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  static uint8_t type = 1;
  type++;
  return type;
}

uint32_t
Address::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 1 + 1 + m_len;
}

void
Address::Serialize (TagBuffer buffer) const
{
  NS_LOG_FUNCTION (this << &buffer);
  buffer.WriteU8 (m_type);
  buffer.WriteU8 (m_len);
  buffer.Write (m_data,  m_len);
}

void
Address::Deserialize (TagBuffer buffer)
{
  NS_LOG_FUNCTION (this << &buffer);
  m_type = buffer.ReadU8 ();
  m_len = buffer.ReadU8 ();
  NS_ASSERT (m_len <= MAX_SIZE);
  buffer.Read (m_data, m_len);
}

ATTRIBUTE_HELPER_CPP (Address);


bool operator == (const Address &a, const Address &b)
{
  /* Two addresses can be equal even if their types are 
   * different if one of the two types is zero. a type of 
   * zero identifies an Address which might contain meaningful 
   * payload but for which the type field could not be set because
   * we did not know it. This can typically happen in the ARP
   * layer where we receive an address from an ArpHeader but
   * we do not know its type: we really want to be able to
   * compare addresses without knowing their real type.
   */
  if (a.m_type != b.m_type &&
      a.m_type != 0 && 
      b.m_type != 0)
    {
      return false;
    }
  if (a.m_len != b.m_len)
    {
      return false;
    }
  return std::memcmp (a.m_data, b.m_data, a.m_len) == 0;
}
bool operator != (const Address &a, const Address &b)
{
  return !(a == b);
}
bool operator < (const Address &a, const Address &b)
{
  if (a.m_type < b.m_type)
    {
      return true;
    }
  else if (a.m_type > b.m_type)
    {
      return false;
    }
  if (a.m_len < b.m_len)
    {
      return true;
    }
  else if (a.m_len > b.m_len)
    {
      return false;
    }
  NS_ASSERT (a.GetLength () == b.GetLength ());
  for (uint8_t i = 0; i < a.GetLength (); i++)
    {
      if (a.m_data[i] < b.m_data[i]) 
        {
          return true;
        }
      else if (a.m_data[i] > b.m_data[i]) 
        {
          return false;
        }
    }
  return false;
}

std::ostream& operator<< (std::ostream& os, const Address & address)
{
  os.setf (std::ios::hex, std::ios::basefield);
  os.fill ('0');
  os << std::setw (2) << (uint32_t) address.m_type << "-" << std::setw (2) << (uint32_t) address.m_len << "-";
  for (uint8_t i = 0; i < (address.m_len-1); ++i)
    {
      os << std::setw (2) << (uint32_t)address.m_data[i] << ":";
    }
  // Final byte not suffixed by ":"
  os << std::setw (2) << (uint32_t) address.m_data[address.m_len-1];
  os.setf (std::ios::dec, std::ios::basefield);
  os.fill (' ');
  return os;
}

std::istream& operator>> (std::istream& is, Address & address)
{
  std::string v;
  is >> v;
  std::string::size_type firstDash, secondDash;
  firstDash = v.find ("-");
  secondDash = v.find ("-", firstDash+1);
  std::string type = v.substr (0, firstDash-0);
  std::string len = v.substr (firstDash+1, secondDash-(firstDash+1));

  address.m_type = strtoul (type.c_str(), 0, 16);
  address.m_len = strtoul (len.c_str(), 0, 16);
  NS_ASSERT (address.m_len <= Address::MAX_SIZE);

  std::string::size_type col = secondDash + 1;
  for (uint8_t i = 0; i < address.m_len; ++i)
    {
      std::string tmp;
      std::string::size_type next;
      next = v.find (":", col);
      if (next == std::string::npos)
        {
          tmp = v.substr (col, v.size ()-col);
          address.m_data[i] = strtoul (tmp.c_str(), 0, 16);
          break;
        }
      else
        {
          tmp = v.substr (col, next-col);
          address.m_data[i] = strtoul (tmp.c_str(), 0, 16);
          col = next + 1;
        }
    }
  return is;
}



} // namespace ns3
