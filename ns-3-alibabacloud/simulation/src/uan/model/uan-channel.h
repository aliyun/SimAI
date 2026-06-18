/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_CHANNEL_H
#define UAN_CHANNEL_H

#include "ns3/net-device.h"
#include "ns3/channel.h"
#include "ns3/packet.h"
#include "ns3/uan-prop-model.h"
#include "ns3/uan-noise-model.h"

#include <list>
#include <vector>

namespace ns3 {

class UanNetDevice;
class UanPhy;
class UanTransducer;
class UanTxMode;

/**
 * \ingroup uan
 *
 * Channel class used by UAN devices.
 */
class UanChannel : public Channel
{
public:
  /**
   * UanDeviceList is a standard template vector of pairs
   * (UanNetDevice, UanTransducer)
   */
  typedef std::vector<std::pair<Ptr<UanNetDevice>, Ptr<UanTransducer> > > UanDeviceList;

  UanChannel ();           //!< Constructor
  virtual ~UanChannel ();  //!< Dummy destructor, see DoDispose.

  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // Inherited methods
  virtual std::size_t GetNDevices (void) const;
  virtual Ptr<NetDevice> GetDevice (std::size_t i) const;

  /**
   * Send a packet out on the channel.
   *
   * \param src Transducer transmitting packet.
   * \param packet Packet to be transmitted.
   * \param txPowerDb Transmission power in dB.
   * \param txmode UanTxMode defining modulation of transmitted packet.
   */
  virtual void TxPacket  (Ptr<UanTransducer> src, Ptr<Packet> packet, double txPowerDb,
                          UanTxMode txmode);

  /**
   * Adds device to receiver list for this channel.
   *
   * \param dev Net Device of node.
   * \param trans Transducer of net device attached to this channel.
   */
  void AddDevice (Ptr<UanNetDevice> dev, Ptr<UanTransducer> trans);

  /**
   * Set the propagation model this channel will use
   * for path loss/propagation delay.
   *
   * \param prop The propagation model.
   */
  void SetPropagationModel (Ptr<UanPropModel> prop);

  /**
   * Set the noise model this channel will use
   * to determine ambient channel noise.
   *
   * \param noise The noise model.
   */
  void SetNoiseModel  (Ptr<UanNoiseModel> noise);

  /**
   * Get the noise level on the channel.
   *
   * \param fKhz Frequency in kHz.
   * \return Ambient noise in dB/Hz on channel at a frequency.
   */
  double GetNoiseDbHz (double fKhz);

  /**
   * Clear all pointer references. */
  void Clear (void);

protected:
  UanDeviceList m_devList;     //!< The list of devices on this channel.
  Ptr<UanPropModel> m_prop;    //!< The propagation model.
  Ptr<UanNoiseModel> m_noise;  //!< The noise model.
  /** Has Clear ever been called on the channel. */
  bool m_cleared;

  /**
   * Send a packet up to the receiving UanTransducer.
   *
   * \param i Device number.
   * \param packet The received packet.
   * \param rxPowerDb Signal power in dB of arriving packet.
   * \param txMode Mode arriving packet is using.
   * \param pdp PDP of arriving signal.
   */
  void SendUp (uint32_t i, Ptr<Packet> packet, double rxPowerDb, UanTxMode txMode, UanPdp pdp);

  virtual void DoDispose (void);

};  // class UanChannel

} // namespace ns3

#endif /* UAN_CHANNEL_H */
