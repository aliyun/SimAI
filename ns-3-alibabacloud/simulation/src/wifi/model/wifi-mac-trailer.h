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

#ifndef WIFI_MAC_TRAILER_H
#define WIFI_MAC_TRAILER_H

#include "ns3/trailer.h"

namespace ns3 {

/**
 * The length in octects of the IEEE 802.11 MAC FCS field
 */
static const uint16_t WIFI_MAC_FCS_LENGTH = 4;

/**
 * \ingroup wifi
 *
 * Implements the IEEE 802.11 MAC trailer
 */
class WifiMacTrailer : public Trailer
{
public:
  WifiMacTrailer ();
  virtual ~WifiMacTrailer ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
};

} //namespace ns3

#endif /* WIFI_MAC_TRAILER_H */
