/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2018 Universita' degli Studi di Napoli Federico II
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
// Author: Stefano Avallone <stavallo@unina.it>
//

#ifndef QUEUE_SIZE_H
#define QUEUE_SIZE_H

#include <string>
#include <iostream>
#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/abort.h"

namespace ns3 {

/**
 * \ingroup network
 * \defgroup queuesize Queue size
 */

/**
 * \ingroup queuesize
 * \brief Enumeration of the operating modes of queues.
 *
 */
enum QueueSizeUnit
{
  PACKETS,     /**< Use number of packets for queue size */
  BYTES,       /**< Use number of bytes for queue size */
};

/**
 * \ingroup queuesize
 * \brief Class for representing queue sizes
 *
 * Allows for natural and familiar use of queue sizes.  Allows construction
 * from strings, natural sum e.g.:
 * \code
 *   QueueSize x("7kB");
 *   Ptr<Packet> p = Create<Packet> (1000);
 *   QueueSize y = x + p;    // 8kB
 * \endcode
 * This class also supports the regular comparison operators \c <, \c >,
 * \c <=, \c >=, \c ==, and \c !=
 *
 * Queue size specifiers consist of
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
 *
 * Supported unit strings:
 *
 * | Symbol   | Meaning     |
 * | :------- | :---------- |
 * | "B"      | 8-bit bytes |
 * | "p"      | packets     |
 *
 * Examples:
 * * "56kB" = 56,000 bytes
 * * "128 kB" = 128,000 bytes
 * * "8KiB" = 8,192 bytes
 * * "1000p" = 1,000 packets
 *
 * \see attribute_QueueSize
 */
class QueueSize
{
public:
  QueueSize ();
  /**
   * \brief Integer constructor
   *
   * Construct a queue size from a mode and a value.
   * \param unit whether the value is expressed in terms of packets or bytes
   * \param value the value
   */
  QueueSize (QueueSizeUnit unit, uint32_t value);
  /**
   * \brief String constructor
   *
   * Construct a queue size from a string.  Many different unit strings are supported
   * Supported unit strings:
   * B, p \n
   * kB, KB, KiB, kp, Kp, Kip \n
   * MB, MiB, Mp, Mip \n
   *
   * Examples:
   * "56kB" = 56,000 bytes
   * "128 kB" = 128,000 bytes
   * "8KiB" = 8,192 bytes
   * "1000p" = 1,000 packets
   *
   * \param size string representing the size
   */
  QueueSize (std::string size);

  /**
   * \return true if this size is less than rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator <  (const QueueSize& rhs) const;

  /**
   * \return true if this size is less than or equal to rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator <= (const QueueSize& rhs) const;

  /**
   * \return true if this size is greater than rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator >  (const QueueSize& rhs) const;

  /**
   * \return true if this size is greater than or equal to rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator >= (const QueueSize& rhs) const;

  /**
   * \return true if this size is equal to rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator == (const QueueSize& rhs) const;

  /**
   * \return true if this size is not equal to rhs
   *
   * \param rhs the queue size to compare to this queue size
   */
  bool operator != (const QueueSize& rhs) const;

  /**
   * Get the underlying unit
   * \return The underlying unit
   */
  QueueSizeUnit GetUnit () const;

  /**
   * Get the underlying value
   * \return The underlying value
   */
  uint32_t GetValue () const;

private:

  /**
   * \brief Parse a string representing a QueueSize
   *
   * Allowed unit representations include all combinations of
   *
   * * An SI prefix: k, K, M
   * * Bytes (8 bits) or packets
   *
   * \param [in] s The string representation, including unit
   * \param [in,out] unit The location to put the unit.
   * \param [in,out] value The location to put the value, in bytes or packets.
   * \return true if parsing was successful.
   */
  static bool DoParse (const std::string s, QueueSizeUnit *unit, uint32_t *value);

  // Uses DoParse
  friend std::istream &operator >> (std::istream &is, QueueSize &size);

  QueueSizeUnit m_unit; //!< unit
  uint32_t m_value;     //!< queue size [bytes or packets]
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param size the queue size
 * \returns a reference to the stream
 */
std::ostream &operator << (std::ostream &os, const QueueSize &size);

/**
 * \brief Stream extraction operator.
 *
 * \param is the stream
 * \param size the queue size
 * \returns a reference to the stream
 */
std::istream &operator >> (std::istream &is, QueueSize &size);

ATTRIBUTE_HELPER_HEADER (QueueSize);


/**
 * \brief Increase the queue size by a packet size
 *
 * \param lhs queue size
 * \param rhs packet
 * \return the queue size increased by the packet size
 */
template <typename Item>
QueueSize operator+ (const QueueSize& lhs, const Ptr<Item>& rhs);
/**
 * \brief Increase the queue size by a packet size
 *
 * \param lhs packet
 * \param rhs queue size
 * \return the queue size increased by the packet size
 */
template <typename Item>
QueueSize operator+ (const Ptr<Item>& lhs, const QueueSize& rhs);


/**
 * Implementation of the templates declared above.
 */


template <typename Item>
QueueSize operator+ (const QueueSize& lhs, const Ptr<Item>& rhs)
{
  if (lhs.GetUnit () == QueueSizeUnit::PACKETS)
    {
      return QueueSize (lhs.GetUnit (), lhs.GetValue () + 1);
    }
  if (lhs.GetUnit () == QueueSizeUnit::BYTES)
    {
      return QueueSize (lhs.GetUnit (), lhs.GetValue () + rhs->GetSize ());
    }
  NS_FATAL_ERROR ("Unknown queue size mode");
}

template <typename Item>
QueueSize operator+ (const Ptr<Item>& lhs, const QueueSize& rhs)
{
  if (rhs.GetUnit () == QueueSizeUnit::PACKETS)
    {
      return QueueSize (rhs.GetUnit (), rhs.GetValue () + 1);
    }
  if (rhs.GetUnit () == QueueSizeUnit::BYTES)
    {
      return QueueSize (rhs.GetUnit (), rhs.GetValue () + lhs->GetSize ());
    }
  NS_FATAL_ERROR ("Unknown queue size mode");
}


} // namespace ns3

#endif /* QUEUE_SIZE_H */
