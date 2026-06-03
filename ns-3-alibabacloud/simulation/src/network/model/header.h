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

#ifndef HEADER_H
#define HEADER_H

#include "chunk.h"
#include "buffer.h"
#include <stdint.h>

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief Protocol header serialization and deserialization.
 *
 * Every Protocol header which needs to be inserted or removed
 * from a Packet instance must derive from this base class and
 * implement the pure virtual methods defined here. 
 *
 * Sample code which shows how to create a new type of Header, and how to use it, 
 * is shown in the sample file samples/main-packet-header.cc
 */
class Header : public Chunk
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual ~Header ();

  using Chunk::Deserialize;
  /**
   * \returns the expected size of the header.
   *
   * This method is used by Packet::AddHeader
   * to store a header into the byte buffer of a packet. This method
   * should return the number of bytes which are needed to store
   * the full header data by Serialize.
   */
  virtual uint32_t GetSerializedSize (void) const = 0;
  /**
   * \param start an iterator which points to where the header should
   *        be written.
   *
   * This method is used by Packet::AddHeader to
   * store a header into the byte buffer of a packet.
   * The data written
   * is expected to match bit-for-bit the representation of this
   * header in a real network.
   */
  virtual void Serialize (Buffer::Iterator start) const = 0;
  /**
   * \param start an iterator which points to where the header should
   *        read from.
   * \returns the number of bytes read.
   *
   * This method is used by Packet::RemoveHeader to
   * re-create a header from the byte buffer of a packet. 
   * The data read is expected to
   * match bit-for-bit the representation of this header in real
   * networks.
   *
   * Note that data is not actually removed from the buffer to 
   * which the iterator points.  Both Packet::RemoveHeader() and
   * Packet::PeekHeader() call Deserialize(), but only the RemoveHeader()
   * has additional statements to remove the header bytes from the
   * underlying buffer and associated metadata.
   */
  virtual uint32_t Deserialize (Buffer::Iterator start) = 0;
  /**
   * \param os output stream
   * This method is used by Packet::Print to print the 
   * content of a header as ascii data to a c++ output stream.
   * Although the header is free to format its output as it
   * wishes, it is recommended to follow a few rules to integrate
   * with the packet pretty printer: start with flags, small field 
   * values located between a pair of parens. Values should be separated 
   * by whitespace. Follow the parens with the important fields, 
   * separated by whitespace.
   * i.e.: (field1 val1 field2 val2 field3 val3) field4 val4 field5 val5
   */
  virtual void Print (std::ostream &os) const = 0;
};


/**
 * \brief Stream insertion operator.
 *
 * \param os the stream
 * \param header the header
 * \returns a reference to the stream
 */
std::ostream & operator << (std::ostream &os, const Header &header);

} // namespace ns3

#endif /* HEADER_H */
