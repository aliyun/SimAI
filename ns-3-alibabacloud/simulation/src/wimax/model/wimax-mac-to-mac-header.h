/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2010 INRIA, UDcast
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
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *
 */
#ifndef WIMAX_MAC_TO_MAC_HEADER_H
#define WIMAX_MAC_TO_MAC_HEADER_H

#include <stdint.h>
#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup wimax
 * \brief this class implements the mac to mac header needed to dump a wimax pcap file
 * The header format was reverse-engineered by looking  at existing live pcap traces which
 * could be opened with wireshark  i.e., we have no idea where this is coming from.
 */
class WimaxMacToMacHeader : public Header
{
public:
  WimaxMacToMacHeader ();
  ~WimaxMacToMacHeader ();
  /**
   * Constructor
   *
   * \param len length
   */
  WimaxMacToMacHeader (uint32_t len);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
  /**
   * Get size of length field
   * \returns the size of length field
   */
  uint8_t GetSizeOfLen (void) const;
  virtual void Print (std::ostream &os) const;
private:
  uint32_t m_len; ///< length
};
};
#endif

