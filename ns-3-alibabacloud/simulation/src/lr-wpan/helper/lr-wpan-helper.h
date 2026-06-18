/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Authors:
 *  Gary Pei <guangyu.pei@boeing.com>
 *  Tom Henderson <thomas.r.henderson@boeing.com>
 */
#ifndef LR_WPAN_HELPER_H
#define LR_WPAN_HELPER_H

#include <ns3/node-container.h>
#include <ns3/lr-wpan-phy.h>
#include <ns3/lr-wpan-mac.h>
#include <ns3/trace-helper.h>

namespace ns3 {

class SpectrumChannel;
class MobilityModel;

/**
 * \ingroup lr-wpan
 *
 * \brief helps to manage and create IEEE 802.15.4 NetDevice objects
 *
 * This class can help to create IEEE 802.15.4 NetDevice objects
 * and to configure their attributes during creation.  It also contains
 * additional helper functions used by client code.
 *
 * Only one channel is created, and all devices attached to it.  If
 * multiple channels are needed, multiple helper objects must be used,
 * or else the channel object must be replaced.
 */

class LrWpanHelper : public PcapHelperForDevice,
                     public AsciiTraceHelperForDevice
{
public:
  /**
   * \brief Create a LrWpan helper in an empty state.  By default, a
   * SingleModelSpectrumChannel is created, with a 
   * LogDistancePropagationLossModel and a ConstantSpeedPropagationDelayModel.
   *
   * To change the channel type, loss model, or delay model, the Get/Set
   * Channel methods may be used.
   */
  LrWpanHelper (void);

  /**
   * \brief Create a LrWpan helper in an empty state with either a
   * SingleModelSpectrumChannel or a MultiModelSpectrumChannel.
   * \param useMultiModelSpectrumChannel use a MultiModelSpectrumChannel if true, a SingleModelSpectrumChannel otherwise
   *
   * A LogDistancePropagationLossModel and a 
   * ConstantSpeedPropagationDelayModel are added to the channel.
   */
  LrWpanHelper (bool useMultiModelSpectrumChannel);

  virtual ~LrWpanHelper (void);

  /**
   * \brief Get the channel associated to this helper
   * \returns the channel
   */
  Ptr<SpectrumChannel> GetChannel (void);

  /**
   * \brief Set the channel associated to this helper
   * \param channel the channel
   */
  void SetChannel (Ptr<SpectrumChannel> channel);

  /**
   * \brief Set the channel associated to this helper
   * \param channelName the channel name
   */
  void SetChannel (std::string channelName);

  /**
   * \brief Add mobility model to a physical device
   * \param phy the physical device
   * \param m the mobility model
   */
  void AddMobility (Ptr<LrWpanPhy> phy, Ptr<MobilityModel> m);

  /**
   * \brief Install a LrWpanNetDevice and the associated structures (e.g., channel) in the nodes.
   * \param c a set of nodes
   * \returns A container holding the added net devices.
   */
  NetDeviceContainer Install (NodeContainer c);

  /**
   * \brief Associate the nodes to the same PAN
   *
   * \param c a set of nodes
   * \param panId the PAN Id
   */
  void AssociateToPan (NetDeviceContainer c, uint16_t panId);

  /**
   * \brief Associate the nodes to the same PAN and initiate beacon enabled mode.
   *
   * \param c a set of nodes
   * \param panId the PAN id
   * \param coor the address of the PAN coordinator
   * \param bcnOrd indicates the interval between beacons.
   *               The value must be an int between 0 and 14.
   * \param sfrmOrd indicates the length of the superframe.
   *                The value must be an int between 0 and 14 and less or equal to the bcnOrd
   */
  void AssociateToBeaconPan (NetDeviceContainer c, uint16_t panId, Mac16Address coor, uint8_t bcnOrd, uint8_t sfrmOrd);

  /**
   * Helper to enable all LrWpan log components with one statement
   */
  void EnableLogComponents (void);

  /**
   * \brief Transform the LrWpanPhyEnumeration enumeration into a printable string.
   * \param e the LrWpanPhyEnumeration
   * \return a string
   */
  static std::string LrWpanPhyEnumerationPrinter (LrWpanPhyEnumeration e);

  /**
   * \brief Transform the LrWpanMacState enumeration into a printable string.
   * \param e the LrWpanMacState
   * \return a string
   */
  static std::string LrWpanMacStatePrinter (LrWpanMacState e);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams that have been
   * assigned. The Install() method should have previously been
   * called by the user.
   *
   * \param c NetDeviceContainer of the set of net devices for which the
   *          CsmaNetDevice should be modified to use a fixed stream
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this helper
   */
  int64_t AssignStreams (NetDeviceContainer c, int64_t stream);

private:
  // Disable implicit constructors
  /**
   * \brief Copy constructor - defined and not implemented.
   */
  LrWpanHelper (LrWpanHelper const &);
  /**
   * \brief Copy constructor - defined and not implemented.
   * \returns
   */
  LrWpanHelper& operator= (LrWpanHelper const &);
  /**
   * \brief Enable pcap output on the indicated net device.
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param nd Net device for which you want to enable tracing.
   * \param promiscuous If true capture all possible packets available at the device.
   * \param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnablePcapInternal (std::string prefix, Ptr<NetDevice> nd, bool promiscuous, bool explicitFilename);

  /**
   * \brief Enable ascii trace output on the indicated net device.
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param stream The output stream object to use when logging ascii traces.
   * \param prefix Filename prefix to use for ascii trace files.
   * \param nd Net device for which you want to enable tracing.
   * \param explicitFilename Treat the prefix as an explicit filename if true
   */
  virtual void EnableAsciiInternal (Ptr<OutputStreamWrapper> stream,
                                    std::string prefix,
                                    Ptr<NetDevice> nd,
                                    bool explicitFilename);

private:
  Ptr<SpectrumChannel> m_channel; //!< channel to be used for the devices

};

}

#endif /* LR_WPAN_HELPER_H */
