/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 University of Washington
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
 */

#ifndef HE_CONFIGURATION_H
#define HE_CONFIGURATION_H

#include "ns3/object.h"
#include "ns3/nstime.h"

namespace ns3 {

/**
 * \brief HE configuration
 * \ingroup wifi
 *
 * This object stores HE configuration information, for use in modifying
 * AP or STA behavior and for constructing HE-related information elements.
 *
 */
class HeConfiguration : public Object
{
public:
  HeConfiguration ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * \param guardInterval the supported HE guard interval
   */
  void SetGuardInterval (Time guardInterval);
  /**
   * \return the supported HE guard interval
   */
  Time GetGuardInterval (void) const;
  /**
   * \param bssColor the BSS color
   */
  void SetBssColor (uint8_t bssColor);
  /**
   * \return the BSS color
   */
  uint8_t GetBssColor (void) const;
  /**
   * \param size the MPDU buffer size to receive A-MPDUs
   */
  void SetMpduBufferSize (uint16_t size);
  /**
   * \return the MPDU buffer size to receive A-MPDUs
   */
  uint16_t GetMpduBufferSize (void) const;


private:
  Time m_guardInterval;      //!< Supported HE guard interval
  uint8_t m_bssColor;        //!< BSS color
  uint16_t m_mpduBufferSize; //!< MPDU buffer size
  uint8_t m_muBeAifsn;       //!< AIFSN for BE in MU EDCA Parameter Set
  uint8_t m_muBkAifsn;       //!< AIFSN for BK in MU EDCA Parameter Set
  uint8_t m_muViAifsn;       //!< AIFSN for VI in MU EDCA Parameter Set
  uint8_t m_muVoAifsn;       //!< AIFSN for VO in MU EDCA Parameter Set
  uint16_t m_muBeCwMin;      //!< CWmin for BE in MU EDCA Parameter Set
  uint16_t m_muBkCwMin;      //!< CWmin for BK in MU EDCA Parameter Set
  uint16_t m_muViCwMin;      //!< CWmin for VI in MU EDCA Parameter Set
  uint16_t m_muVoCwMin;      //!< CWmin for VO in MU EDCA Parameter Set
  uint16_t m_muBeCwMax;      //!< CWmax for BE in MU EDCA Parameter Set
  uint16_t m_muBkCwMax;      //!< CWmax for BK in MU EDCA Parameter Set
  uint16_t m_muViCwMax;      //!< CWmax for VI in MU EDCA Parameter Set
  uint16_t m_muVoCwMax;      //!< CWmax for VO in MU EDCA Parameter Set
  Time m_beMuEdcaTimer;      //!< Timer for BE in MU EDCA Parameter Set
  Time m_bkMuEdcaTimer;      //!< Timer for BK in MU EDCA Parameter Set
  Time m_viMuEdcaTimer;      //!< Timer for VI in MU EDCA Parameter Set
  Time m_voMuEdcaTimer;      //!< Timer for VO in MU EDCA Parameter Set
};

} //namespace ns3

#endif /* HE_CONFIGURATION_H */
