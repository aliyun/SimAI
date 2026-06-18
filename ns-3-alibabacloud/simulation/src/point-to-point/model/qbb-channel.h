/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 University of Washington
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
 */

#ifndef QBB_CHANNEL_H
#define QBB_CHANNEL_H

#include <list>
#include "ns3/channel.h"
#include "ns3/point-to-point-channel.h"
#include "ns3/ptr.h"
#include "ns3/nstime.h"
#include "ns3/data-rate.h"
#include "ns3/traced-callback.h"

namespace ns3 {

class QbbNetDevice;
class Packet;

/**
 * \ingroup point-to-point
 * \brief Simple Point To Point Channel.
 *
 * This class represents a very simple point to point channel.  Think full
 * duplex RS-232 or RS-422 with null modem and no handshaking.  There is no
 * multi-drop capability on this channel -- there can be a maximum of two 
 * point-to-point net devices connected.
 *
 * There are two "wires" in the channel.  The first device connected gets the
 * [0] wire to transmit on.  The second device gets the [1] wire.  There is a
 * state (IDLE, TRANSMITTING) associated with each wire.
 */
class QbbChannel : public PointToPointChannel 
{
public:
  static TypeId GetTypeId (void);

  /**
   * \brief Create a QbbChannel
   *
   * By default, you get a channel that
   * has zero transmission delay.
   */
  QbbChannel ();

  /**
   * \brief Attach a given netdevice to this channel
   * \param device pointer to the netdevice to attach to the channel
   */
  void Attach (Ptr<QbbNetDevice> device);

  /**
   * \brief Transmit a packet over this channel
   * \param p Packet to transmit
   * \param src Source QbbNetDevice
   * \param txTime Transmit time to apply
   * \returns true if successful (currently always true)
   */
  virtual bool TransmitStart (Ptr<Packet> p, Ptr<QbbNetDevice> src, Time txTime);

  /**
   * \brief Get number of devices on this channel
   * \returns number of devices on this channel
   */
  virtual size_t GetNDevices (void) const;

  /*
   * \brief Get QbbNetDevice corresponding to index i on this channel
   * \param i Index number of the device requested
   * \returns Ptr to QbbNetDevice requested
   */
  Ptr<QbbNetDevice> GetQbbDevice (uint32_t i) const;

  /*
   * \brief Get NetDevice corresponding to index i on this channel
   * \param i Index number of the device requested
   * \returns Ptr to NetDevice requested
   */
  virtual Ptr<NetDevice> GetDevice (std::size_t i) const;

  /*
   * \brief Get the delay associated with this channel
   * \returns Time delay
   */
  Time GetDelay (void) const;

protected:
  /*
   * \brief Check to make sure the link is initialized
   * \returns true if initialized, asserts otherwise
   */
  bool IsInitialized (void) const;

  /*
   * \brief Get the net-device source 
   * \param i the link requested
   * \returns Ptr to QbbNetDevice source for the 
   * specified link
   */
  Ptr<QbbNetDevice> GetSource (uint32_t i) const;

  /*
   * \brief Get the net-device destination
   * \param i the link requested
   * \returns Ptr to QbbNetDevice destination for 
   * the specified link
   */
  Ptr<QbbNetDevice> GetDestination (uint32_t i) const;

private:
  // Each point to point link has exactly two net devices
  static const int N_DEVICES = 2;

  Time          m_delay;
  int32_t       m_nDevices;

  /**
   * The trace source for the packet transmission animation events that the 
   * device can fire.
   * Arguments to the callback are the packet, transmitting
   * net device, receiving net device, transmission time and 
   * packet receipt time.
   *
   * @see class CallBackTraceSource
   */
  TracedCallback<Ptr<const Packet>, // Packet being transmitted
                 Ptr<NetDevice>,    // Transmitting NetDevice
                 Ptr<NetDevice>,    // Receiving NetDevice
                 Time,              // Amount of time to transmit the pkt
                 Time               // Last bit receive time (relative to now)
                 > m_txrxQbb;

  enum WireState
  {
    INITIALIZING,
    IDLE,
    TRANSMITTING,
    PROPAGATING
  };

  class Link
  {
public:
    Link() : m_state (INITIALIZING), m_src (0), m_dst (0) {}
    WireState                  m_state;
    Ptr<QbbNetDevice> m_src;
    Ptr<QbbNetDevice> m_dst;
  };

  Link    m_link[N_DEVICES];
};

} // namespace ns3

#endif /* POINT_TO_POINT_CHANNEL_H */
