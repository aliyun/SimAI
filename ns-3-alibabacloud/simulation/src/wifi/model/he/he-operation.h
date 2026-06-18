/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef HE_OPERATION_H
#define HE_OPERATION_H

#include "ns3/wifi-information-element.h"

namespace ns3 {

/**
 * \brief The HE Operation Information Element
 * \ingroup wifi
 *
 * This class knows how to serialise and deserialise
 * the HE Operation Information Element
 */
class HeOperation : public WifiInformationElement
{
public:
  HeOperation ();

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const override;
  WifiInformationElementId ElementIdExt () const override;
  uint8_t GetInformationFieldSize () const override;
  void SerializeInformationField (Buffer::Iterator start) const override;
  uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length) override;
  /* This information element is a bit special in that it is only
     included if the STA is a HE STA. To support this we
     override the Serialize and GetSerializedSize methods of
     WifiInformationElement. */
  Buffer::Iterator Serialize (Buffer::Iterator start) const override;
  uint16_t GetSerializedSize () const override;

  /**
   * Set the HE supported information element.
   *
   * \param heSupported the HE supported information element
   */
  void SetHeSupported (uint8_t heSupported);

  /**
   * Set the HE Operation Parameters field in the HE Operation information element.
   *
   * \param ctrl the HE Operation Parameters field in the HE Operation information element
   */
  void SetHeOperationParameters (uint32_t ctrl);
  /**
   * Set the Basic HE-MCS and NSS field in the HE Operation information element
   * by specifying the tuple (<i>nss</i>, <i>maxMcs</i>).
   *
   * \param nss the NSS
   * \param maxHeMcs the maximum supported HE-MCS value corresponding to that NSS
   */
  void SetMaxHeMcsPerNss (uint8_t nss, uint8_t maxHeMcs);

  /**
   * Return the HE Operation Parameters field in the HE Operation information element.
   *
   * \return the HE Operation Parameters field in the HE Operation information element
   */
  uint32_t GetHeOperationParameters (void) const;
  /**
   * Return the Basic HE-MCS And Nss field in the HE Operation information element.
   *
   * \return the Basic HE-MCS And Nss field in the HE Operation information element
   */
  uint16_t GetBasicHeMcsAndNssSet (void) const;
  /**
   * Set the BSS color
   * \param bssColor the BSS color value
   */
  void SetBssColor (uint8_t bssColor);
  /**
   * Get the BSS color
   * \return the BSS color value
   */
  uint8_t GetBssColor (void) const;


private:
  //HE Operation Parameters fields
  uint8_t m_bssColor;                     //!< BSS color
  uint8_t m_defaultPEDuration;            //!< default PE duration
  uint8_t m_twtRequired;                  //!< TWT required
  uint16_t m_heDurationBasedRtsThreshold; //!< HE duration based RTS threshold
  uint8_t m_partialBssColor;              //!< partial BSS color
  uint8_t m_maxBssidIndicator;            //!< max BSSID indicator
  uint8_t m_txBssidIndicator;             //!< TX BSSID indicator
  uint8_t m_bssColorDisabled;             //!< BSS color disabled
  uint8_t m_dualBeacon;                   //!< Dual Beacon

  //Basic HE-MCS and NSS Set
  uint16_t m_basicHeMcsAndNssSet; ///< basic HE MCS NSS set

  //TODO: VHT Operation Information subfields not defined in the standard yet.

  /// This is used to decide whether this element should be added to the frame or not
  uint8_t m_heSupported;
};

/**
 * output stream output operator
 *
 * \param os output stream
 * \param HeOperation the HE operation
 *
 * \returns output stream
 */
std::ostream &operator << (std::ostream &os, const HeOperation &HeOperation);

} //namespace ns3

#endif /* HE_OPERATION_H */
