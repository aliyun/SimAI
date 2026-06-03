/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013
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
 * Author: Ghada Badawy <gbadawy@gmail.com>
 */

#ifndef AMPDU_SUBFRAME_HEADER_H
#define AMPDU_SUBFRAME_HEADER_H

#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup wifi
 * \brief Headers for A-MPDU subframes
 */
class AmpduSubframeHeader : public Header
{
public:
  AmpduSubframeHeader ();
  virtual ~AmpduSubframeHeader ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  TypeId GetInstanceTypeId (void) const override;
  void Print (std::ostream &os) const override;
  uint32_t GetSerializedSize (void) const override;
  void Serialize (Buffer::Iterator start) const override;
  uint32_t Deserialize (Buffer::Iterator start) override;

  /**
   * Set the length field.
   *
   * \param length in bytes
   */
  void SetLength (uint16_t length);
  /**
  * Set the EOF field.
  *
  * \param eof set EOF field if true
  */
  void SetEof (bool eof);
  /**
   * Return the length field.
   *
   * \return the length field in bytes
   */
  uint16_t GetLength (void) const;
  /**
   * Return the EOF field.
   *
   * \return the EOF field
   */
  bool GetEof (void) const;
  /**
   * Return whether the pattern stored in the delimiter
   * signature field is correct, i.e. corresponds to the
   * unique pattern 0x4E.
   *
   * \return true if the signature is valid, false otherwise
   */
  bool IsSignatureValid (void) const;

private:
  uint16_t m_length;   //!< length field in bytes
  bool m_eof;          //!< EOF field
  uint8_t m_signature; //!< delimiter signature (should correspond to pattern 0x4E in order to be assumed valid)
};

} //namespace ns3

#endif /* AMPDU_SUBFRAME_HEADER_H */
