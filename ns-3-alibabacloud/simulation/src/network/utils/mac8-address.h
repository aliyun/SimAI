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

#ifndef MAC8_ADDRESS_H
#define MAC8_ADDRESS_H

#include "ns3/address.h"
#include <iostream>

namespace ns3 {

class Address;

/**
 * \ingroup network
 *
 * A class used for addressing MAC8 MAC's.
 *
 * This implementation uses a simple 8 bit flat addressing scheme.
 * It is unlikely that perceived underwater networks will soon
 * exceed 200 nodes (or the overlapping of two underwater networks
 * - the ocean is big), so this should provide adequate addressing
 * for most applications.
 */
class Mac8Address
{
public:
  /** Constructor */
  Mac8Address ();
  /**
   * Create Mac8Address object with address addr.
   *
   * \param addr Byte address to assign to this address.
   */
  Mac8Address (uint8_t addr);
  /** Destructor */
  virtual ~Mac8Address ();

  /**
   * Convert a generic address to a Mac8Address.
   *
   * \param address  Address to convert to Mac8Address address.
   * \return Mac8Address from Address.
   */
  static Mac8Address ConvertFrom (const Address &address);

  /**
   * Check that a generic Address is compatible with Mac8Address.
   *
   * \param address  Address to test.
   * \return True if address given is consistent with Mac8Address.
   */
  static bool IsMatchingType  (const Address &address);

  /**
   * Create a generic Address.
   *
   * \return The Address.
   */
  operator Address () const;

  /**
   * Sets address to address stored in parameter.
   *
   * \param pBuffer Buffer to extract address from.
   */
  void CopyFrom (const uint8_t *pBuffer);

  /**
   * Writes address to buffer parameter.
   *
   * \param pBuffer
   */
  void CopyTo (uint8_t *pBuffer) const;

  /**
   * Get the broadcast address (255).
   *
   * \return Broadcast address.
   */
  static Mac8Address GetBroadcast (void);

  /**
   * Allocates Mac8Address from 0-254
   *
   * Will wrap back to 0 if more than 254 are allocated.
   * Excludes the broadcast address.
   *
   * \return The next sequential Mac8Address.
   */
  static Mac8Address Allocate ();


private:
  uint8_t m_address;  //!< The address.

  /**
   * Get the Mac8Address type.
   *
   * \return The type value.
   */
  static uint8_t GetType (void);
  /**
   * Convert to a generic Address.
   *
   * \return The Address value.
   */
  Address ConvertTo (void) const;

  friend bool operator <  (const Mac8Address &a, const Mac8Address &b);
  friend bool operator == (const Mac8Address &a, const Mac8Address &b);
  friend bool operator != (const Mac8Address &a, const Mac8Address &b);
  friend std::ostream& operator<< (std::ostream& os, const Mac8Address & address);
  friend std::istream& operator>> (std::istream& is, Mac8Address & address);

};  // class Mac8Address


/**
 * Address comparison, less than.
 *
 * \param a First address to compare.
 * \param b Second address to compare.
 * \return True if a < b.
 */
bool operator < (const Mac8Address &a, const Mac8Address &b);

/**
 * Address comparison, equalit.
 *
 * \param a First address to compare.
 * \param b Second address to compare.
 * \return True if a == b.
 */
bool operator == (const Mac8Address &a, const Mac8Address &b);

/**
 * Address comparison, unequal.
 *
 * \param a First address to compare.
 * \param b Second address to compare.
 * \return True if a != b.
 */
bool operator != (const Mac8Address &a, const Mac8Address &b);

/**
 * Write \pname{address} to stream \pname{os} as 8 bit integer.
 *
 * \param os The output stream.
 * \param address The address
 * \return The output stream.
 */
std::ostream& operator<< (std::ostream& os, const Mac8Address & address);

/**
 * Read \pname{address} from stream \pname{is} as 8 bit integer.
 *
 * \param is The input stream.
 * \param address The address variable to set.
 * \return The input stream.
 */
std::istream& operator>> (std::istream& is, Mac8Address & address);

} // namespace ns3

#endif /* MAC8_ADDRESS_H */
