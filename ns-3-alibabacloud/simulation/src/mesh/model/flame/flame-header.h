/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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

#ifndef FLAME_HEADER_H
#define FLAME_HEADER_H

#include "ns3/header.h"
#include "ns3/mac48-address.h"

namespace ns3 {
namespace flame {
/**
 * \ingroup flame
 *
 * \brief Flame header
 *
 * Header format: | Reserved: 1 | cost: 1 | Sequence number: 2 | OrigDst: 6 | OrigSrc: 6 | Flame port : 2 |
 */

class FlameHeader : public Header
{
public:

  FlameHeader ();
  ~FlameHeader ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // Inherited from Header class:
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  // Seeters/Getters for fields:
  /**
   * Add cost value
   * \param cost the cost
   */
  void AddCost (uint8_t cost);
  /**
   * Get cost value
   * \returns the cost
   */
  uint8_t GetCost () const;
  /**
   * Set sequence number value
   * \param seqno the sequence number
   */
  void SetSeqno (uint16_t seqno);
  /**
   * Get sequence number value
   * \returns the sequence number
   */
  uint16_t GetSeqno () const;
  /**
   * Set origin destination address
   * \param dst the MAC address of the destination
   */
  void SetOrigDst (Mac48Address dst);
  /**
   * Get origin destination address
   * \returns the MAC address of the destination
   */
  Mac48Address GetOrigDst () const;
  /**
   * Set origin source function
   * \param OrigSrc the MAC address of the origina source
   */
  void SetOrigSrc (Mac48Address OrigSrc);
  /**
   * Get origin source address
   * \returns the MAC address of the origin source
   */
  Mac48Address GetOrigSrc () const;
  /**
   * Set protocol value
   * \param protocol the protocol
   */
  void SetProtocol (uint16_t protocol);
  /**
   * Get protocol value
   * \returns the protocol
   */
  uint16_t GetProtocol () const;

private:
  uint8_t m_cost; ///< cost
  uint16_t m_seqno; ///< sequence number
  Mac48Address m_origDst; ///< origin destination
  Mac48Address m_origSrc; ///< origin source
  uint16_t m_protocol; ///< protocol
  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const FlameHeader & a, const FlameHeader & b);
};
bool operator== (const FlameHeader & a, const FlameHeader & b);
} // namespace flame
} // namespace ns3
#endif /* FLAME_HEADER_H */
