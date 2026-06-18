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

#ifndef WIMAX_CHANNEL_H
#define WIMAX_CHANNEL_H

#include <list>
#include "ns3/channel.h"
#include "wimax-connection.h"
#include "ns3/log.h"

namespace ns3 {

class WimaxPhy;
class Packet;
class Position;
class PacketBurst;

/**
 * \ingroup wimax
 * The channel object to attach Wimax NetDevices
 */
class WimaxChannel : public Channel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  WimaxChannel (void);
  virtual ~WimaxChannel (void);
  /**
   * \brief attach the channel to a physical layer of a device
   * \param phy the physical layer to which the channel will be attached
   */
  void Attach (Ptr<WimaxPhy> phy);
  /**
   * \return the number of attached devices
   */
  std::size_t GetNDevices (void) const;
  /**
   * \param i the ith device
   * \return the ith attached device
   */
  Ptr<NetDevice> GetDevice (std::size_t i) const;

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  virtual int64_t AssignStreams (int64_t stream) = 0;

private:
  /**
   * Attach a phy to the channel
   * \param phy the phy object to attach
   */
  virtual void DoAttach (Ptr<WimaxPhy> phy) = 0;

  /**
   * Get number of devices on the channel
   * \returns the number of devices
   */
  virtual std::size_t DoGetNDevices (void) const = 0;
  /**
   * Get device corresponding to index
   * \param i the device index
   * \returns the device
   */
  virtual Ptr<NetDevice> DoGetDevice (std::size_t i) const = 0;
};

} // namespace ns3

#endif /* WIMAX_CHANNEL_H */
