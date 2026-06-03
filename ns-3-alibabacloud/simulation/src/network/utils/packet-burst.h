/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#ifndef PACKET_BURST_H
#define PACKET_BURST_H

#include <stdint.h>
#include <list>
#include "ns3/object.h"

namespace ns3 {

class Packet;

/**
 * \brief this class implement a burst as a list of packets
 */
class PacketBurst : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  PacketBurst (void);
  virtual ~PacketBurst (void);
  /**
   * \return a copy the packetBurst
   */
  Ptr<PacketBurst> Copy (void) const;
  /**
   * \brief add a packet to the list of packet
   * \param packet the packet to add
   */
  void AddPacket (Ptr<Packet> packet);
  /**
   * \return the list of packet of this burst
   */
  std::list<Ptr<Packet> > GetPackets (void) const;
  /**
   * \return the number of packet in the burst
   */
  uint32_t GetNPackets (void) const;
  /**
   * \return the size of the burst in byte (the size of all packets)
   */
  uint32_t GetSize (void) const;

  /**
   * \brief Returns an iterator to the begin of the burst
   * \return iterator to the burst list start
   */
  std::list<Ptr<Packet> >::const_iterator Begin (void) const;
  /**
   * \brief Returns an iterator to the end of the burst
   * \return iterator to the burst list end
   */
  std::list<Ptr<Packet> >::const_iterator End (void) const;

  /**
   * TracedCallback signature for Ptr<PacketBurst>
   *
   * \param [in] burst The PacketBurst
   */
  typedef void (* TracedCallback)(Ptr<const PacketBurst> burst);

  
private:
  void DoDispose (void);
  std::list<Ptr<Packet> > m_packets; //!< the list of packets in the burst
};
} // namespace ns3

#endif /* PACKET_BURST */
