/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef MESH_ID_H
#define MESH_ID_H

#include <stdint.h>
#include "ns3/buffer.h"
#include "ns3/mesh-information-element-vector.h"

namespace ns3 {
namespace dot11s {

/**
 * \brief a IEEE 802.11 Mesh ID element (Section 8.4.2.101 of IEEE 802.11-2012)
 * \see attribute_IeMeshId
 */
class IeMeshId : public WifiInformationElement
{
public:
  // broadcast meshId
  IeMeshId ();
  /**
   * Constructor
   *
   * \param s reference id
   */
  IeMeshId (std::string s);

  /**
   * Equality test
   * \param o another IeMeshId
   * \returns true if equal
   */
  bool IsEqual (IeMeshId const &o) const;
  /**
   * Return true if broadcast (if first octet of Mesh ID is zero)
   * \returns true if broadcast
   */
  bool IsBroadcast (void) const;
  //uint32_t GetLength (void) const;
  /**
   * Peek the IeMeshId as a string value
   * \returns the mesh ID as a string
   */
  char *PeekString (void) const;

  // Inherited from WifiInformationElement
  virtual WifiInformationElementId ElementId () const;
  virtual void SerializeInformationField (Buffer::Iterator i) const;
  virtual uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length);
  virtual void Print (std::ostream& os) const;
  virtual uint8_t GetInformationFieldSize () const;

private:
  uint8_t m_meshId[33]; ///< mesh ID
  /**
   * equality operator
   *
   * \param a lhs
   * \param b lhs
   * \returns true if equal
   */
  friend bool operator== (const IeMeshId & a, const IeMeshId & b);

};

/**
 * \brief Stream insertion operator.
 *
 * \param [in] os The reference to the output stream.
 * \param [in] meshId The IeMeshId object.
 * \returns The reference to the output stream.
 */
std::ostream &operator << (std::ostream &os, const IeMeshId &meshId);

  
} // namespace dot11s
} // namespace ns3
#endif /* MESH_ID_H */
