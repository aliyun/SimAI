/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Rajib Bhattacharjea<raj.b@gatech.edu>
//

#ifndef DATA_RATE_H
#define DATA_RATE_H

#include <string>
#include <iostream>
#include <stdint.h>
#include "ns3/nstime.h"
#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/deprecated.h"

namespace ns3 {

/**
 * \ingroup network
 * \defgroup datarate Data Rate
 */
/**
 * \ingroup datarate
 * \brief Class for representing data rates
 *
 * Allows for natural and familiar use of data rates.  Allows construction
 * from strings, natural multiplication e.g.:
 * \code
 *   DataRate x("56kbps");
 *   double nBits = x*ns3::Seconds (19.2);
 *   uint32_t nBytes = 20;
 *   Time txtime = x.CalculateBytesTxTime (nBytes);
 * \endcode
 * This class also supports the regular comparison operators \c <, \c >,
 * \c <=, \c >=, \c ==, and \c !=
 *
 * Data rate specifiers consist of
 * * A numeric value,
 * * An optional multiplier prefix and
 * * A unit.
 *
 * Whitespace is allowed but not required between the numeric value and
 * multipler or unit.
 *
 * Supported multiplier prefixes:
 *
 * | Prefix   | Value       |
 * | :------- | ----------: |
 * | "k", "K" | 1000        |
 * | "Ki"     | 1024        |
 * | "M"      | 1000000     |
 * | "Mi"     | 1024 Ki     |
 * | "G"      | 10^9        |
 * | "Gi "    | 1024 Mi     |
 *
 * Supported unit strings:
 *
 * | Symbol   | Meaning     |
 * | :------- | :---------- |
 * | "b"      | bits        |
 * | "B"      | 8-bit bytes |
 * | "s", "/s"| per second  |
 *
 * Examples:
 * * "56kbps" = 56,000 bits/s
 * * "128 kb/s" = 128,000 bits/s
 * * "8Kib/s" = 1 KiB/s = 8192 bits/s
 * * "1kB/s" = 8000 bits/s 
 *
 * \see attribute_DataRate
 */
class DataRate
{
public:
  DataRate ();
  /**
   * \brief Integer constructor
   *
   * Construct a data rate from an integer.  This class only supports positive
   * integer data rates in units of bits/s, meaning 1bit/s is the smallest 
   * non-trivial bitrate available.
   * \param bps bit/s value
   */
  DataRate (uint64_t bps);
  /**
   * \brief String constructor
   *
   * Construct a data rate from a string.  Many different unit strings are supported
   * Supported unit strings:
   * bps, b/s, Bps, B/s \n
   * kbps, kb/s, Kbps, Kb/s, kBps, kB/s, KBps, KB/s, Kib/s, KiB/s \n
   * Mbps, Mb/s, MBps, MB/s, Mib/s, MiB/s \n
   * Gbps, Gb/s, GBps, GB/s, Gib/s, GiB/s \n
   * 
   * Examples:
   * "56kbps" = 56,000 bits/s \n
   * "128 kb/s" = 128,000 bits/s \n
   * "8Kib/s" = 1 KiB/s = 8192 bits/s \n
   * "1kB/s" = 8000 bits/s 
   *
   * \param rate string representing the desired rate
   */
  DataRate (std::string rate);

  /**
   * \return the DataRate representing the sum of this object with rhs
   * 
   * \param rhs the DataRate to add to this DataRate
   */
  DataRate operator + (DataRate rhs);

  /**
   * \return the DataRate representing the sum of this object with rhs
   * 
   * \param rhs the DataRate to add to this DataRate
   */
  DataRate& operator += (DataRate rhs);

  /**
   * \return the DataRate representing the difference of this object with rhs
   * 
   * \param rhs the DataRate to subtract from this DataRate
   */
  DataRate operator - (DataRate rhs);

  /**
   * \return the DataRate representing the difference of this object with rhs
   * 
   * \param rhs the DataRate to subtract from this DataRate
   */
  DataRate& operator -= (DataRate rhs);

  /**
   * \brief Scales the DataRate
   * 
   * Multiplies with double and is re-casted to an int
   * 
   * \return DataRate object representing the product of this object with rhs 
   * 
   * \param rhs the double to multiply to this datarate
   */
  DataRate operator * (double rhs);

  /**
   * \brief Scales the DataRate
   * 
   * Multiplies with double and is re-casted to an int 
   * 
   * \return DataRate object representing the product of this object with rhs
   * 
   * \param rhs the double to multipy to this datarate
   */
  DataRate& operator *= (double rhs);

  /**
   * \brief Scales the DataRate
   * 
   * \return DataRate object representing the product of this object with rhs
   * 
   * \param rhs the uint64_t to multipy to this datarate
   */
  DataRate operator * (uint64_t rhs);

  /**
   * \brief Scales the DataRate
   * 
   * \return DataRate object representing the product of this object with rhs
   * 
   * \param rhs the uint64_t to multipy to this datarate
   */
  DataRate& operator *= (uint64_t rhs);


  /**
   * \return true if this rate is less than rhs
   *
   * \param rhs the datarate to compare to this datarate
   */  
  bool operator <  (const DataRate& rhs) const;

  /**
   * \return true if this rate is less than or equal to rhs
   *
   * \param rhs the datarate to compare to this datarate
   */ 
  bool operator <= (const DataRate& rhs) const;
  
  /**
   * \return true if this rate is greater than rhs
   *
   * \param rhs the datarate to compare to this datarate
   */   
  bool operator >  (const DataRate& rhs) const;
  
  /**
   * \return true if this rate is greater than or equal to rhs
   *
   * \param rhs the datarate to compare to this datarate
   */   
  bool operator >= (const DataRate& rhs) const;
  
  /**
   * \return true if this rate is equal to rhs
   *
   * \param rhs the datarate to compare to this datarate
   */   
  bool operator == (const DataRate& rhs) const;
  
  /**
   * \return true if this rate is not equal to rhs
   *
   * \param rhs the datarate to compare to this datarate
   */   
  bool operator != (const DataRate& rhs) const;

  /**
   * \brief Calculate transmission time
   *
   * Calculates the transmission time at this data rate
   * \param bytes The number of bytes (not bits) for which to calculate
   * \return The transmission time for the number of bytes specified
   */
  Time CalculateBytesTxTime (uint32_t bytes) const;

  /**
   * \brief Calculate transmission time
   *
   * Calculates the transmission time at this data rate
   * \param bits The number of bits (not bytes) for which to calculate
   * \return The transmission time for the number of bits specified
   */
  Time CalculateBitsTxTime (uint32_t bits) const;

  /**
   * Get the underlying bitrate
   * \return The underlying bitrate in bits per second
   */
  uint64_t GetBitRate () const;

private:

  /**
   * \brief Parse a string representing a DataRate into an uint64_t
   *
   * Allowed unit representations include all combinations of
   *
   * * An SI prefix: k, K, M, G
   * * Decimal or kibibit (as in "Kibps", meaning 1024 bps)
   * * Bits or bytes (8 bits)
   * * "bps" or "/s"
   *
   * \param [in] s The string representation, including unit
   * \param [in,out] v The location to put the value, in bits/sec.
   * \return true if parsing was successful.
   */
  static bool DoParse (const std::string s, uint64_t *v);

  // Uses DoParse
  friend std::istream &operator >> (std::istream &is, DataRate &rate);
  
  uint64_t m_bps; //!< data rate [bps]
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param rate the data rate
 * \returns a reference to the stream
 */
std::ostream &operator << (std::ostream &os, const DataRate &rate);

/**
 * \brief Stream extraction operator.
 *
 * \param is the stream
 * \param rate the data rate
 * \returns a reference to the stream
 */
std::istream &operator >> (std::istream &is, DataRate &rate);

ATTRIBUTE_HELPER_HEADER (DataRate);


/**
 * \brief Multiply datarate by a time value
 *
 * Calculates the number of bits that have been transmitted over a period of time
 * \param lhs rate
 * \param rhs time
 * \return the number of bits over the period of time
 */
double operator* (const DataRate& lhs, const Time& rhs);
/**
 * \brief Multiply time value by a data rate
 *
 * Calculates the number of bits that have been transmitted over a period of time
 * \param lhs time
 * \param rhs rate
 * \return the number of bits over the period of time
 */
double operator* (const Time& lhs, const DataRate& rhs);

DataRate operator* (const double& c, const DataRate& d);
DataRate operator* (const DataRate& d, const double& c);

DataRate operator/ (const DataRate& d, const double& c);
double   operator/ (const DataRate& lhs, const DataRate& rhs);

namespace TracedValueCallback {

/**
 * \ingroup network
 * TracedValue callback signature for DataRate
 *
 * \param [in] oldValue original value of the traced variable
 * \param [in] newValue new value of the traced variable
 */
typedef void (* DataRate)(DataRate oldValue,
                          DataRate newValue);

}  // namespace TracedValueCallback

} // namespace ns3

#endif /* DATA_RATE_H */
