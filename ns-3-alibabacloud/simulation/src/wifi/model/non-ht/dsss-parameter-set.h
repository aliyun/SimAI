/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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

#ifndef DSSS_PARAMETER_SET_H
#define DSSS_PARAMETER_SET_H

#include "ns3/wifi-information-element.h"

namespace ns3 {

/**
 * \brief The DSSS Parameter Set
 * \ingroup wifi
 *
 * This class knows how to serialise and deserialise the DSSS Parameter Set.
 */
class DsssParameterSet : public WifiInformationElement
{
public:
  DsssParameterSet ();

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const override;
  uint8_t GetInformationFieldSize () const override;
  void SerializeInformationField (Buffer::Iterator start) const override;
  uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length) override;
  /* This information element is a bit special in that it is only
     included if the STA does support DSSS. To support this we
     override the Serialize and GetSerializedSize methods of
     WifiInformationElement. */
  Buffer::Iterator Serialize (Buffer::Iterator start) const override;
  uint16_t GetSerializedSize () const override;

  /**
   * Set DSSS supported
   * \param dsssSupported the DSSS supported indicator
   */
  void SetDsssSupported (uint8_t dsssSupported);

  /**
   * Set the Current Channel field in the DsssParameterSet information element.
   *
   * \param currentChannel the CurrentChannel field in the DsssParameterSet information element
   */
  void SetCurrentChannel (uint8_t currentChannel);


private:
  uint8_t m_currentChannel; ///< current channel number

  /// This is used to decide whether this element should be added to the frame or not
  bool m_dsssSupported;
};

} //namespace ns3

#endif /* DSSS_PARAMETER_SET_H */
