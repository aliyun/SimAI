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

#ifndef MU_EDCA_PARAMETER_SET_H
#define MU_EDCA_PARAMETER_SET_H

#include "ns3/wifi-information-element.h"
#include "ns3/nstime.h"
#include <array>

namespace ns3 {

/**
 * \brief The MU EDCA Parameter Set
 * \ingroup wifi
 *
 * The 802.11ax MU EDCA Parameter Set.
 */
class MuEdcaParameterSet : public WifiInformationElement
{
public:
  MuEdcaParameterSet ();
  /**
   * Return true if a valid MU EDCA Parameter Set is present in this object
   * \return true if a valid MU EDCA Parameter Set is present in this object
   */
  bool IsPresent (void) const;

  /**
   * Set the QoS Info field in the MuEdcaParameterSet information element.
   *
   * \param qosInfo the QoS Info field in the MuEdcaParameterSet information element
   */
  void SetQosInfo (uint8_t qosInfo);
  /**
   * Set the AIFSN subfield of the ACI/AIFSN field in the MU AC Parameter Record
   * field corresponding to the given AC Index (<i>aci</i>). Note that <i>aifsn</i>
   * can either be zero (EDCA disabled) or in the range from 2 to 15.
   *
   * \param aci the AC Index
   * \param aifsn the value for the AIFSN subfield
   */
  void SetMuAifsn (uint8_t aci, uint8_t aifsn);
  /**
   * Set the ECWmin subfield of the ECWmin/ECWmax field in the MU AC Parameter Record
   * field corresponding to the given AC Index (<i>aci</i>). Note that <i>cwMin</i>
   * must be a power of 2 minus 1 in the range from 0 to 32767.
   *
   * \param aci the AC Index
   * \param cwMin the CWmin value encoded by the ECWmin field
   */
  void SetMuCwMin (uint8_t aci, uint16_t cwMin);
  /**
   * Set the ECWmax subfield of the ECWmin/ECWmax field in the MU AC Parameter Record
   * field corresponding to the given AC Index (<i>aci</i>). Note that <i>cwMax</i>
   * must be a power of 2 minus 1 in the range from 0 to 32767.
   *
   * \param aci the AC Index
   * \param cwMax the CWmax value encoded by the ECWmax field
   */
  void SetMuCwMax (uint8_t aci, uint16_t cwMax);
  /**
   * Set the MU EDCA Timer field in the MU AC Parameter Record field corresponding
   * to the given AC Index (<i>aci</i>). Note that <i>timer</i> must be an integer
   * multiple of 8 TUs (i.e., 8 * 1024 = 8192 microseconds) in the range from
   * 8.192 milliseconds to 2088.96 milliseconds. A value of 0 is used to indicate
   * that the MU EDCA Parameter Set element must not be sent and therefore it is
   * only allowed if the MU EDCA Timer is set to 0 for all ACs.
   *
   * \param aci the AC Index
   * \param timer the value for the timer encoded by the MU EDCA Timer field
   */
  void SetMuEdcaTimer (uint8_t aci, Time timer);

  /**
   * Return the QoS Info field in the MuEdcaParameterSet information element.
   *
   * \return the QoS Info field in the MuEdcaParameterSet information element
   */
  uint8_t GetQosInfo (void) const;
  /**
   * Get the AIFSN subfield of the ACI/AIFSN field in the MU AC Parameter Record
   * field corresponding to the given AC Index (<i>aci</i>).
   *
   * \param aci the AC Index
   * \return the value of the AIFSN subfield
   */
  uint8_t GetMuAifsn (uint8_t aci) const;
  /**
   * Get the CWmin value encoded by the ECWmin subfield of the ECWmin/ECWmax field
   * in the MU AC Parameter Record field corresponding to the given AC Index (<i>aci</i>).
   *
   * \param aci the AC Index
   * \return the CWmin value
   */
  uint16_t GetMuCwMin (uint8_t aci) const;
  /**
   * Get the CWmax value encoded by the ECWmax subfield of the ECWmin/ECWmax field
   * in the MU AC Parameter Record field corresponding to the given AC Index (<i>aci</i>).
   *
   * \param aci the AC Index
   * \return the CWmax value
   */
  uint16_t GetMuCwMax (uint8_t aci) const;
  /**
   * Get the MU EDCA Timer value encoded in the MU AC Parameter Record field corresponding
   * to the given AC Index (<i>aci</i>).
   *
   * \param aci the AC Index
   * \return the MU EDCA Timer value
   */
  Time GetMuEdcaTimer (uint8_t aci) const;

  /**
   * Get the wifi information element ID
   * \return the wifi information element ID
   */
  WifiInformationElementId ElementId () const;
  /**
   * Get the wifi information element ID extension
   * \return the wifi information element ID extension
   */
  WifiInformationElementId ElementIdExt () const;
  /**
   * Get information field size function
   * \return the information field size
   */
  uint8_t GetInformationFieldSize () const;
  /**
   * Serialize information field function
   * \param start the iterator
   */
  void SerializeInformationField (Buffer::Iterator start) const;
  /**
   * Deserialize information field function
   * \param start the iterator
   * \param length the length
   * \return the size
   */
  uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length);

  /**
   * This information element is a bit special in that it is only
   * included if the STA is an HE STA. To support this we
   * override the Serialize and GetSerializedSize methods of
   * WifiInformationElement.
   *
   * \param start iterator start
   *
   * \return an iterator
   */
  Buffer::Iterator Serialize (Buffer::Iterator start) const;
  /**
   * Return the serialized size of this EDCA Parameter Set.
   *
   * \return the serialized size of this EDCA Parameter Set
   */
  uint16_t GetSerializedSize () const;


private:
  /**
   * MU AC Parameter Record type
   */
  struct ParameterRecord
    {
      uint8_t aifsnField;                    ///< the ACI/AIFSN field
      uint8_t cwMinMax;                      ///< the ECWmin/ECWmax field
      uint8_t muEdcaTimer;                   ///< the MU EDCA Timer field
    };

  uint8_t m_qosInfo;                         ///< QoS info field
  std::array<ParameterRecord, 4> m_records;  ///< MU AC Parameter Record fields
};

} //namespace ns3

#endif /* MU_EDCA_PARAMETER_SET_H */
