/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDCAST
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef WIMAX_HELPER_H
#define WIMAX_HELPER_H

#include <string>
#include "ns3/object-factory.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/bs-net-device.h"
#include "ns3/ss-net-device.h"
#include "ns3/service-flow.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/simple-ofdm-wimax-channel.h"
#include "ns3/bs-uplink-scheduler.h"
#include "ns3/bs-uplink-scheduler-mbqos.h"
#include "ns3/bs-uplink-scheduler-simple.h"
#include "ns3/bs-uplink-scheduler-rtps.h"
#include "ns3/bs-scheduler.h"
#include "ns3/bs-scheduler-simple.h"
#include "ns3/bs-scheduler-rtps.h"
#include "ns3/trace-helper.h"

namespace ns3 {

class WimaxChannel;
class WimaxPhy;
class UplinkScheduler;


/**
 * \brief helps to manage and create WimaxNetDevice objects
 *
 * This class can help to create a large set of similar
 * WimaxNetDevice objects and to configure their attributes
 * during creation.
 */

class WimaxHelper : public PcapHelperForDevice, public AsciiTraceHelperForDevice
{
public:
  /**
   * Net Device Type
   * Distinguish a subscriber station(SS) device from base station(BS) device
   */
  enum NetDeviceType
  {
    DEVICE_TYPE_SUBSCRIBER_STATION, /**< Subscriber station(SS) device */
    DEVICE_TYPE_BASE_STATION
    /**< Base station(BS) device */
  };

  /**
   * WiMAX Physical layer
   * WiMAX Physical layers with different levels of detail
   */
  enum PhyType
  {
    SIMPLE_PHY_TYPE_OFDM
  };

  /**
   * Scheduler Type
   * Different implementations of uplink/downlink scheduler
   */
  enum SchedulerType
  {
    SCHED_TYPE_SIMPLE, /**< A simple priority-based FCFS scheduler */
    SCHED_TYPE_RTPS, /**< A simple scheduler - rtPS based scheduler */
    SCHED_TYPE_MBQOS
    /**< An migration-based uplink scheduler */

  };
  /**
   * \brief Create a Wimax helper in an empty state.
   */
  WimaxHelper (void);
  ~WimaxHelper (void);
  /**
   *  \brief Enable ascii trace output on the indicated net device for a given connection
   *  \param oss The output stream object to use when logging ascii traces.
   *  \param nodeid the id of the node for which you want to enable tracing.
   *  \param deviceid the id of the net device for which you want to enable tracing.
   *  \param netdevice the type of net device for which you want to enable tracing (SubscriberStationNetDevice,
   *   BaseStationNetDevice or WimaxNetDevice)
   *  \param connection the connection for which you want to enable tracing (InitialRangingConnection,
   *   BroadcastConnection, BasicConnection, PrimaryConnection).
   */
  static void EnableAsciiForConnection (Ptr<OutputStreamWrapper> oss,
                                        uint32_t nodeid,
                                        uint32_t deviceid,
                                        char *netdevice,
                                        char *connection);

  /**
   * \param phyType WiMAX Physical layer type
   * \return WiMAX Phy object
   *
   * Creates a physical layer object to be used in simulation.
   */
  Ptr<WimaxPhy> CreatePhy (PhyType phyType);

  /**
   * \param schedulerType Scheduling mechanism
   * \return Uplink scheduler
   *
   * Creates a uplink scheduler to be used by base station
   * according to selected scheduling mechanism.
   */
  Ptr<UplinkScheduler> CreateUplinkScheduler (SchedulerType schedulerType);

  /**
   * \param schedulerType Scheduling mechanism
   * \return Downlink scheduler
   *
   * Creates a downlink scheduler to be used by base station
   * according to selected scheduling mechanism.
   */
  Ptr<BSScheduler> CreateBSScheduler (SchedulerType schedulerType);

  /**
   * \param c a set of nodes
   * \param type device type to create
   * \param phyType a phy to use
   * \param schedulerType the type of the scheduling algorithm to install
   * \return NetDeviceContainer
   *
   * For each of the input nodes, a new WiMAX net device (either
   * ns3::SubscriberStationNetDevice or ns3::BaseStationNetDevice
   * depending on the type parameter) is attached to the shared input channel.
   */
  NetDeviceContainer Install (NodeContainer c, NetDeviceType type, PhyType phyType, SchedulerType schedulerType);

  /**
   * \param c A set of nodes.
   * \param deviceType Device type to create.
   * \param phyType PHY type to create.
   * \param channel A channel to use.
   * \param schedulerType The scheduling mechanism.
   * \return NetDeviceContainer
   *
   * For each of the input nodes, a new WiMAX net device (either
   * ns3::SubscriberStationNetDevice or ns3::BaseStationNetDevice
   * depending on the type parameter) is attached to the shared input channel.
   */
  NetDeviceContainer Install (NodeContainer c,
                              NetDeviceType deviceType,
                              PhyType phyType,
                              Ptr<WimaxChannel> channel,
                              SchedulerType schedulerType);
  /**
   * \param c A set of nodes.
   * \param deviceType Device type to create.
   * \param phyType PHY type to create.
   * \param schedulerType The scheduling mechanism.
   * \param frameDuration the farme duration in seconds
   * \return NetDeviceContainer
   *
   * For each of the input nodes, a new WiMAX net device (either
   * ns3::SubscriberStationNetDevice or ns3::BaseStationNetDevice
   * depending on the type parameter) is attached to the shared input channel.
   */
  NetDeviceContainer Install (NodeContainer c,
                              NetDeviceType deviceType,
                              PhyType phyType,
                              SchedulerType schedulerType,
                              double frameDuration);

  /**
   * \brief Set the propagation and loss model of the channel. By default the channel
   *  uses a COST231 propagation and loss model.
   * \param propagationModel The propagation and loss model to set
   */
  void SetPropagationLossModel (SimpleOfdmWimaxChannel::PropModel propagationModel);

  /**
   * \param phyType WiMAX Physical layer type
   * \return WiMAX Phy object
   *
   * Creates a physical layer without a channel
   */
  Ptr<WimaxPhy> CreatePhyWithoutChannel (PhyType phyType);

  /**
   * \param phyType WiMAX Physical layer type
   * \param SNRTraceFilePath of the repository containing the SNR traces files
   * \param activateLoss set to 1 to activate losses 0 otherwise
   * \return WiMAX Phy object
   *
   * Creates a physical layer without creating a channel
   */
  Ptr<WimaxPhy> CreatePhyWithoutChannel (PhyType phyType, char * SNRTraceFilePath, bool activateLoss);

  /**
   * \param phyType WiMAX Physical layer type
   * \param SNRTraceFilePath the path to the repository containing the SNR traces files
   * \param activateLoss set to 1 if you want ton activate losses 0 otherwise
   * \return WiMAX Phy object
   *
   * Creates a physical layer
   */
  Ptr<WimaxPhy> CreatePhy (PhyType phyType, char * SNRTraceFilePath, bool activateLoss);
  /**
   * \param node Node to be installed.
   * \param deviceType Device type to create.
   * \param phyType PHY type to create.
   * \param channel A channel to use.
   * \param schedulerType The scheduling mechanism to install on the device.
   * \return Ptr<WimaxNetDevice>
   *
   * For each of the input nodes, a new WiMAX net device (either
   * ns3::SubscriberStationNetDevice or ns3::BaseStationNetDevice
   * depending on the type parameter) is attached to the shared input channel.
   */
  Ptr<WimaxNetDevice> Install (Ptr<Node> node,
                               NetDeviceType deviceType,
                               PhyType phyType,
                               Ptr<WimaxChannel> channel,
                               SchedulerType schedulerType);

  /**
   * \brief Creates a transport service flow.
   * \param direction the direction of the service flow: UP or DOWN.
   * \param schedulinType The service scheduling type to be used: UGS, RTPS, NRTPS, BE
   * \param classifier The classifier to be used for this service flow
   * \return ServiceFlow
   *
   */
  ServiceFlow CreateServiceFlow (ServiceFlow::Direction direction,
                                 ServiceFlow::SchedulingType schedulinType,
                                 IpcsClassifierRecord classifier);

  /**
   * Helper to enable all WimaxNetDevice log components with one statement
   */
  static void EnableLogComponents (void);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

  /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model. Return the number of streams (possibly zero) that
  * have been assigned. The Install() method should have previously been
  * called by the user.
  *
  * \param c NetDeviceContainer of the set of net devices for which the 
  *          WimaxNetDevice should be modified to use a fixed stream
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this helper
  */
  int64_t AssignStreams (NetDeviceContainer c, int64_t stream);

private:
  /**
   * ASCII trace receive event
   * \param stream the output stream
   * \param path the path name
   * \param packet the packet
   * \param source the source MAC address
   */
  static void AsciiRxEvent (Ptr<OutputStreamWrapper> stream, std::string path, Ptr<const Packet> packet, const Mac48Address &source);
  /**
   * ASCII trace transmit event
   * \param stream the output stream
   * \param path the path
   * \param packet the packet
   * \param dest the destination MAC address
   */
  static void AsciiTxEvent (Ptr<OutputStreamWrapper> stream, std::string path, Ptr<const Packet> packet, const Mac48Address &dest);
  /**
   * \brief Enable pcap output on the indicated net device.
   *
   * NetDevice-specific implementation mechanism for hooking the trace and
   * writing to the trace file.
   *
   * \param prefix Filename prefix to use for pcap files.
   * \param nd Net device for which you want to enable tracing.
   * \param explicitFilename Treat the prefix as an explicit filename if true
   * \param promiscuous If true capture all possible packets available at the device.
   */
  virtual void EnablePcapInternal (std::string prefix, Ptr<NetDevice> nd, bool explicitFilename, bool promiscuous);

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

  Ptr<WimaxChannel> m_channel; ///< wifi channel
};

} // namespace ns3

#endif /* WIMAX_HELPER_H */
