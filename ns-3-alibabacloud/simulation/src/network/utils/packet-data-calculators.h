/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Drexel University
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
 * Author: Joe Kopena (tjkopena@cs.drexel.edu)
 */

#ifndef PACKET_DATA_CALCULATORS_H
#define PACKET_DATA_CALCULATORS_H

#include "ns3/packet.h"
#include "ns3/mac48-address.h"
#include "ns3/data-calculator.h"
#include "ns3/basic-data-calculators.h"

namespace ns3 {

/**
 * \ingroup stats
 *
 *  A stat for counting packets
 *
 */
class PacketCounterCalculator : public CounterCalculator<uint32_t> {
public:
  PacketCounterCalculator();
  virtual ~PacketCounterCalculator();

  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * Increments the packet counter by one
   *
   * \param path not used in this method
   * \param packet not used in this method
   */
  void PacketUpdate (std::string path, Ptr<const Packet> packet);

  /**
   * Increments the packet counter by one
   *
   * \param path not used in this method
   * \param packet not used in this method
   * \param realto not used in this method
   */

  void FrameUpdate (std::string path, Ptr<const Packet> packet,
                    Mac48Address realto);

protected:
  virtual void DoDispose (void);

  // end class PacketCounterCalculator
};


/**
 * \ingroup stats
 *
 * A stat for collecting packet size statistics: min, max and average
 *
 */
class PacketSizeMinMaxAvgTotalCalculator :
  public MinMaxAvgTotalCalculator<uint32_t>
{
public:
  PacketSizeMinMaxAvgTotalCalculator();
  virtual ~PacketSizeMinMaxAvgTotalCalculator();
  
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  
  /**
   * Increments the packet stats by the size of the packet
   *
   * \param path not used in this method
   * \param packet packet size used to update stats
   */
  void PacketUpdate (std::string path, Ptr<const Packet> packet);
  
  /**
   * Increments the packet stats by the size of the packet
   *
   * \param path not used in this method
   * \param packet packet size used to update stats
   * \param realto not used in this method
   */  
  void FrameUpdate (std::string path, Ptr<const Packet> packet,
                    Mac48Address realto);

protected:
  virtual void DoDispose (void);

  // end class PacketSizeMinMaxAvgTotalCalculator
};


// end namespace ns3
};


#endif /* PACKET_DATA_CALCULATORS_H */
