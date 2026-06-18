/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Sébastien Deronne
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

#ifndef VHT_OPERATION_H
#define VHT_OPERATION_H

#include "ns3/wifi-information-element.h"

namespace ns3 {

/**
 * \brief The VHT Operation Information Element
 * \ingroup wifi
 *
 * This class knows how to serialise and deserialise
 * the VHT Operation Information Element
 */
class VhtOperation : public WifiInformationElement
{
public:
  VhtOperation ();

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const override;
  uint8_t GetInformationFieldSize () const override;
  void SerializeInformationField (Buffer::Iterator start) const override;
  uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length) override;
  /* This information element is a bit special in that it is only
     included if the STA is a VHT STA. To support this we
     override the Serialize and GetSerializedSize methods of
     WifiInformationElement. */
  Buffer::Iterator Serialize (Buffer::Iterator start) const override;
  uint16_t GetSerializedSize () const override;

  /**
   * Set the VHT supported information element.
   *
   * \param vhtSupported the VHT supported information element
   */
  void SetVhtSupported (uint8_t vhtSupported);

  /**
   * Set the Channel Width field in the VHT Operation information element.
   *
   * \param channelWidth the Channel Width field in the VHT Operation information element
   */
  void SetChannelWidth (uint8_t channelWidth);
  /**
   * Set the Channel Center Frequency Segment 0 field in the VHT Operation information element.
   *
   * \param channelCenterFrequencySegment0 the Channel Center Frequency Segment 0 field in the VHT Operation information element
   */
  void SetChannelCenterFrequencySegment0 (uint8_t channelCenterFrequencySegment0);
  /**
   * Set the Channel Center Frequency Segment 1 field in the VHT Operation information element.
   *
   * \param channelCenterFrequencySegment1 the Channel Center Frequency Segment 1 field in the VHT Operation information element
   */
  void SetChannelCenterFrequencySegment1 (uint8_t channelCenterFrequencySegment1);
  /**
   * Set the Basic VHT-MCS and NSS field in the VHT Operation information element.
   *
   * \param basicVhtMcsAndNssSet the Basic VHT-MCS and NSS field in the VHT Operation information element
   */
  void SetBasicVhtMcsAndNssSet (uint16_t basicVhtMcsAndNssSet);
  /**
   * Set the Basic VHT-MCS and NSS field in the VHT Operation information element
   * by specifying the tuple (nss, maxMcs).
   *
   * \param nss the NSS
   * \param maxVhtMcs the maximum supported VHT-MCS value corresponding to that NSS
   */
  void SetMaxVhtMcsPerNss (uint8_t nss, uint8_t maxVhtMcs);

  /**
   * Return the Channel Width field in the VHT Operation information element.
   *
   * \return the Channel Width field in the VHT Operation information element
   */
  uint8_t GetChannelWidth (void) const;
  /**
   * Return the Channel Center Frequency Segment 0 field in the VHT Operation information element.
   *
   * \return the Channel Center Frequency Segment 0 field in the VHT Operation information element
   */
  uint8_t GetChannelCenterFrequencySegment0 (void) const;
  /**
   * Return the Channel Center Frequency Segment 1 field in the VHT Operation information element.
   *
   * \return the Channel Center Frequency Segment 1 field in the VHT Operation information element
   */
  uint8_t GetChannelCenterFrequencySegment1 (void) const;
  /**
   * Return the Basic VHT-MCS And Nss field in the VHT Operation information element.
   *
   * \return the Basic VHT-MCS And Nss field in the VHT Operation information element
   */
  uint16_t GetBasicVhtMcsAndNssSet (void) const;


private:
  //VHT Operation Information
  uint8_t m_channelWidth; ///< channel width
  uint8_t m_channelCenterFrequencySegment0; ///< channel center frequency segment 0
  uint8_t m_channelCenterFrequencySegment1; ///< channel center frequency segment 1

  //Basic VHT-MCS and NSS Set
  uint16_t m_basicVhtMcsAndNssSet; ///< basic VHT MCS NSS set

  /// This is used to decide whether this element should be added to the frame or not
  uint8_t m_vhtSupported;
};

/**
 * output stream output operator
 *
 * \param os the output stream
 * \param VhtOperation the VHT operation
 *
 * \returns output stream
 */
std::ostream &operator << (std::ostream &os, const VhtOperation &VhtOperation);

} //namespace ns3

#endif /* VHT_OPERATION_H */
