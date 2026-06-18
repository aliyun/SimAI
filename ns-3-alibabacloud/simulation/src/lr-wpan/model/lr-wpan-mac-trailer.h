/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  kwong yin <kwong-sang.yin@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 *  Erwan Livolant <erwan.livolant@inria.fr>
 */

#ifndef LR_WPAN_MAC_TRAILER_H
#define LR_WPAN_MAC_TRAILER_H

#include <ns3/trailer.h>

namespace ns3 {

class Packet;

/**
 * \ingroup lr-wpan
 *
 * Represent the Mac Trailer with the Frame Check Sequence field.
 */
class LrWpanMacTrailer : public Trailer
{
public:
  /**
   * The length in octets of the IEEE 802.15.4 MAC FCS field
   */
  static const uint16_t LR_WPAN_MAC_FCS_LENGTH;

  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Default constructor for a MAC trailer with disabled FCS calculation.
   */
  LrWpanMacTrailer (void);

  // Inherited from the Trailer class.
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * Get this trailers FCS value. If FCS calculation is disabled for this
   * trailer, the returned value is always 0.
   *
   * \return the FCS value.
   */
  uint16_t GetFcs (void) const;

  /**
   * Calculate and set the FCS value based on the given packet.
   *
   * \param p the packet for which the FCS should be calculated
   */
  void SetFcs (Ptr<const Packet> p);

  /**
   * Check the FCS of a given packet against the FCS value stored in the
   * trailer. The packet itself should contain no trailer. If FCS calculation is
   * disabled for this trailer, CheckFcs() will always return true.
   *
   * \param p the packet to be checked
   * \return false, if the FCS values do not match, true otherwise
   */
  bool CheckFcs (Ptr<const Packet> p);

  /**
   * Enable or disable FCS calculation for this trailer.
   *
   * \param enable flag, indicating if FCS calculation should be enabled or not
   */
  void EnableFcs (bool enable);

  /**
   * Query if FCS calculation is enabled for this trailer.
   *
   * \return true, if FCS calculation is enabled, false otherwise.
   */
  bool IsFcsEnabled (void);

private:
  /**
   * Calculate the 16-bit FCS value.
   * CRC16-CCITT with a generator polynomial = ^16 + ^12 + ^5 + 1, LSB first and
   * initial value = 0x0000.
   *
   * \param data the checksum will be calculated over this data
   * \param length the length of the data
   * \return the checksum
   */
  uint16_t GenerateCrc16 (uint8_t *data, int length);

  /**
   * The FCS value stored in this trailer.
   */
  uint16_t m_fcs;

  /**
   * Only if m_calcFcs is true, FCS values will be calculated and used in the
   * trailer
   */
  bool m_calcFcs;

};

} // namespace ns3

#endif /* LR_WPAN_MAC_TRAILER_H */
