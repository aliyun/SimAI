/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "mac8-address.h"
#include "ns3/address.h"

namespace ns3 {

Mac8Address::Mac8Address ()
{
  m_address = 255;
}

Mac8Address::Mac8Address (uint8_t addr)
  : m_address (addr)
{
}

Mac8Address::~Mac8Address ()
{
}

uint8_t
Mac8Address::GetType (void)
{
  static uint8_t type = Address::Register ();
  return type;
}

Address
Mac8Address::ConvertTo (void) const
{
  return Address (GetType (), &m_address, 1);
}

Mac8Address
Mac8Address::ConvertFrom (const Address &address)
{
  NS_ASSERT (IsMatchingType (address));
  Mac8Address uAddr;
  address.CopyTo (&uAddr.m_address);
  return uAddr;
}

bool
Mac8Address::IsMatchingType (const Address &address)
{
  return address.CheckCompatible (GetType (), 1);
}

Mac8Address::operator Address () const
{
  return ConvertTo ();
}

void
Mac8Address::CopyFrom (const uint8_t *pBuffer)
{
  m_address = *pBuffer;
}

void
Mac8Address::CopyTo (uint8_t *pBuffer) const
{
  *pBuffer = m_address;

}

Mac8Address
Mac8Address::GetBroadcast ()
{
  return Mac8Address (255);
}
Mac8Address
Mac8Address::Allocate ()
{
  static uint8_t nextAllocated = 0;

  uint8_t address = nextAllocated++;
  if (nextAllocated == 255)
    {
      nextAllocated = 0;
    }

  return Mac8Address (address);
}

bool
operator < (const Mac8Address &a, const Mac8Address &b)
{
  return a.m_address < b.m_address;
}

bool
operator == (const Mac8Address &a, const Mac8Address &b)
{
  return a.m_address == b.m_address;
}

bool
operator != (const Mac8Address &a, const Mac8Address &b)
{
  return !(a == b);
}

std::ostream&
operator<< (std::ostream& os, const Mac8Address & address)
{
  os << (int) address.m_address;
  return os;
}
std::istream&
operator>> (std::istream& is, Mac8Address & address)
{
  uint8_t x;
  is >> x;
  NS_ASSERT (0 <= x);
  NS_ASSERT (x <= 255);
  address.m_address = x;
  return is;
}

} // namespace ns3
