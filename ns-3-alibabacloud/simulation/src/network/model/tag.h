/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
#ifndef TAG_H
#define TAG_H

#include "ns3/object-base.h"
#include "tag-buffer.h"
#include <stdint.h>

namespace ns3 {

/**
 * \ingroup packet
 *
 * \brief tag a set of bytes in a packet
 *
 * New kinds of tags can be created by subclassing from this abstract base class.
 */
class Tag : public ObjectBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \returns the number of bytes required to serialize the data of the tag.
   *
   * This method is typically invoked by Packet::AddPacketTag or Packet::AddByteTag
   * just prior to calling Tag::Serialize.
   */
  virtual uint32_t GetSerializedSize (void) const = 0;
  /**
   * \param i the buffer to write data into.
   *
   * Write the content of the tag in the provided tag buffer.
   * DO NOT attempt to write more bytes than you requested
   * with Tag::GetSerializedSize.
   */
  virtual void Serialize (TagBuffer i) const = 0;
  /**
   * \param i the buffer to read data from.
   *
   * Read the content of the tag from the provided tag buffer.
   * DO NOT attempt to read more bytes than you wrote with
   * Tag::Serialize.
   */
  virtual void Deserialize (TagBuffer i) = 0;

  /**
   * \param os the stream to print to
   *
   * This method is typically invoked from the Packet::PrintByteTags
   * or Packet::PrintPacketTags methods.
   */
  virtual void Print (std::ostream &os) const = 0;
};

} // namespace ns3

#endif /* TAG_H */
