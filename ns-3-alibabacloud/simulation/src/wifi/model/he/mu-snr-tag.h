/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#ifndef MU_SNR_TAG_H
#define MU_SNR_TAG_H

#include "ns3/tag.h"
#include <map>

namespace ns3 {

/**
 * \ingroup wifi
 *
 * A tag to be attached to a response to a multi-user UL frame, that carries the SNR
 * values with which the individual frames have been received.
 */
class MuSnrTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TypeId GetInstanceTypeId (void) const override;

  /**
   * Create an empty MuSnrTag
   */
  MuSnrTag ();

  uint32_t GetSerializedSize (void) const override;
  void Serialize (TagBuffer i) const override;
  void Deserialize (TagBuffer i) override;
  void Print (std::ostream &os) const override;

  /**
   * Reset the content of the tag.
   */
  void Reset (void);
  /**
   * Set the SNR for the given sender to the given value.
   *
   * \param staId the STA-ID of the given sender
   * \param snr the value of the SNR to set in linear scale
   */
  void Set (uint16_t staId, double snr);
  /**
   * Return true if the SNR value for the given STA-ID is present
   *
   * \param staId the STA-ID
   * \return true if the SNR value for the given STA-ID is present
   */
  bool IsPresent (uint16_t staId) const;
  /**
   * Return the SNR value for the given sender.
   *
   * \param staId the STA-ID of the given sender
   * \return the SNR value in linear scale
   */
  double Get (uint16_t staId) const;


private:
  std::map<uint16_t, double> m_snrMap;     //!< Map containing (STA-ID, SNR) pairs
};

}

#endif /* MU_SNR_TAG_H */
