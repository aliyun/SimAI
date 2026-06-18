/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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

#ifndef CAPABILITY_INFORMATION_H
#define CAPABILITY_INFORMATION_H

#include "ns3/buffer.h"

namespace ns3 {

/**
 * \ingroup wifi
 *
 * Capability information
 */
class CapabilityInformation
{
public:
  CapabilityInformation ();

  /**
   * Set the Extended Service Set (ESS) bit
   * in the capability information field.
   */
  void SetEss (void);
  /**
   * Set the Independent BSS (IBSS) bit
   * in the capability information field.
   */
  void SetIbss (void);
  /**
   * Set the short preamble bit
   * in the capability information field.
   *
   * \param shortPreamble the short preamble bit
   *
   */
  void SetShortPreamble (bool shortPreamble);
  /**
   * Set the short slot time bit
   * in the capability information field.
   *
   * \param shortSlotTime the short preamble bit
   *
   */
  void SetShortSlotTime (bool shortSlotTime);
  /**
   * Set the CF-Pollable bit
   * in the capability information field.
   */
  void SetCfPollable (void);

  /**
   * Check if the Extended Service Set (ESS) bit
   * in the capability information field is set to 1.
   *
   * \return ESS bit in the capability information
   *         field is set to 1
   */
  bool IsEss (void) const;
  /**
   * Check if the Independent BSS (IBSS) bit
   * in the capability information field is set to 1.
   *
   * \return IBSS bit in the capability information
   *         field is set to 1
   */
  bool IsIbss (void) const;
  /**
   * Check if the short preamble bit
   * in the capability information field is set to 1.
   *
   * \return short preamble bit in the capability information
   *         field is set to 1
   */
  bool IsShortPreamble (void) const;
  /**
   * Check if the short slot time
   * in the capability information field is set to 1.
   *
   * \return short slot time bit in the capability information
   *         field is set to 1
   */
  bool IsShortSlotTime (void) const;
  /**
   * Check if the CF-Pollable bit
   * in the capability information field is set to 1.
   *
   * \return CF-Pollable bit in the capability information
   *         field is set to 1
   */
  bool IsCfPollable (void) const;

  /**
   * Return the serialized size of capability
   * information.
   *
   * \return the serialized size
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize capability information to the given buffer.
   *
   * \param start an iterator to a buffer
   * \return an iterator to a buffer after capability information
   *         was serialized
   */
  Buffer::Iterator Serialize (Buffer::Iterator start) const;
  /**
   * Deserialize capability information from the given buffer.
   *
   * \param start an iterator to a buffer
   * \return an iterator to a buffer after capability information
   *         was deserialized
   */
  Buffer::Iterator Deserialize (Buffer::Iterator start);


private:
  /**
   * Check if bit n is set to 1.
   *
   * \param n the bit position
   *
   * \return true if bit n is set to 1,
   *         false otherwise
   */
  bool Is (uint8_t n) const;
  /**
   * Set bit n to 1.
   *
   * \param n the bit position
   */
  void Set (uint8_t n);
  /**
   * Set bit n to 0.
   *
   * \param n the bit position
   */
  void Clear (uint8_t n);

  uint16_t m_capability; ///< capability
};

} //namespace ns3

#endif /* CAPABILITY_INFORMATION_H */
