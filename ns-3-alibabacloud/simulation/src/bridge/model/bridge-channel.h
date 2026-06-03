/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Gustavo Carneiro  <gjc@inescporto.pt>
 */
#ifndef BRIDGE_CHANNEL_H
#define BRIDGE_CHANNEL_H

#include "ns3/net-device.h"
#include "ns3/channel.h"
#include <vector>

/**
 * \file
 * \ingroup bridge
 * ns3::BridgeChannel declaration.
 */

namespace ns3 {

/**
 * \ingroup bridge
 * 
 * \brief Virtual channel implementation for bridges (BridgeNetDevice).
 *
 * Just like BridgeNetDevice aggregates multiple NetDevices,
 * BridgeChannel aggregates multiple channels and make them appear as
 * a single channel to upper layers.
 */
class BridgeChannel : public Channel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  BridgeChannel ();
  virtual ~BridgeChannel ();

  // Delete copy constructor and assignment operator to avoid misuse
  BridgeChannel (const BridgeChannel &) = delete;
  BridgeChannel & operator = (const BridgeChannel &) = delete;

  /**
   * Adds a channel to the bridged pool
   * \param bridgedChannel  the channel to add to the pool
   */
  void AddChannel (Ptr<Channel> bridgedChannel);

  // virtual methods implementation, from Channel
  virtual std::size_t GetNDevices (void) const;
  virtual Ptr<NetDevice> GetDevice (std::size_t i) const;

private:
  std::vector< Ptr<Channel> > m_bridgedChannels; //!< pool of bridged channels
};

} // namespace ns3

#endif /* BRIDGE_CHANNEL_H */
