/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INESC Porto
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
 * C++ helper functions for use by the python visualizer (for things
 * Python is too slow at).
 *
 * Author: Gustavo Carneiro  <gjc@inescporto.pt>
 */
#ifndef NS3_PYVIZ_H
#define NS3_PYVIZ_H

#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/node.h"
#include "ns3/channel.h"
#include "ns3/packet.h"
#include "ns3/mac48-address.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-l3-protocol.h"

#include <map>
#include <set>

namespace ns3 {

/**
 * \ingroup visualizer
 *
 * \brief helper class to be used by the visualizer
 * \internal
 *
 * This class is not meant to be used by simulations.  It is only
 * meant to be used by the visualizer tool (PyViz).  The only reason
 * it is public is because Python bindings for it are needed,
 * otherwise it should be considered private.
 **/
class PyViz
{
public:
  PyViz ();
  ~PyViz ();

  /**
   * Register drop trace path function
   * \param tracePath the path to trace
   */
  void RegisterDropTracePath (std::string const &tracePath);

  /**
   * Register CSMA like device function
   * \param deviceTypeName the device type name
   */
  void RegisterCsmaLikeDevice (std::string const &deviceTypeName);
  /**
   * Register WIFI like device function
   * \param deviceTypeName the device type name
   */
  void RegisterWifiLikeDevice (std::string const &deviceTypeName);
  /**
   * Register point to point like device function
   * \param deviceTypeName the device type name
   */
  void RegisterPointToPointLikeDevice (std::string const &deviceTypeName);

  /**
   * Run simulation until a given (simulated, absolute) time is reached
   * \param time the run time
   */
  void SimulatorRunUntil (Time time);

  /**
   * Pause function
   * \param message the pause message
   */
  static void Pause (std::string const &message);
  /**
   * Get pause message function
   * \returns the pause message
   */
  std::vector<std::string> GetPauseMessages () const;

  /// TransmissionSample structure
  struct TransmissionSample
  {
    Ptr<Node> transmitter; ///< transmitter
    Ptr<Node> receiver; ///< NULL if broadcast
    Ptr<Channel> channel; ///< channel
    uint32_t bytes; ///< bytes
  };
  typedef std::vector<TransmissionSample> TransmissionSampleList; ///< TransmissionSampleList typedef
  /**
   * Get transmission samples
   * \returns the transmission sample list
   */
  TransmissionSampleList GetTransmissionSamples () const;

  /// PacketDropSample structure
  struct PacketDropSample
  {
    Ptr<Node> transmitter; ///< transmitter
    uint32_t bytes; ///< bytes
  };
  typedef std::vector<PacketDropSample> PacketDropSampleList; ///< PacketDropSampleList typedef
  /**
   * Get packet drop samples
   * \returns the packet drop sample list
   */
  PacketDropSampleList GetPacketDropSamples () const;


  /// PacketSample structure
  struct PacketSample
  {
    Time time; ///< time
    Ptr<Packet> packet; ///< packet
    Ptr<NetDevice> device; ///< device
  };
  /// TxPacketSample structure
  struct TxPacketSample : PacketSample
  {
    Mac48Address to; ///< to
  };
  /// RxPacketSample structure
  struct RxPacketSample : PacketSample
  {
    Mac48Address from; ///< from
  };

  /// LastPacketsSample structure
  struct LastPacketsSample
  {
    std::vector<RxPacketSample> lastReceivedPackets; ///< last received packets
    std::vector<TxPacketSample> lastTransmittedPackets; ///< last transmitted packets
    std::vector<PacketSample> lastDroppedPackets; ///< last dropped packets
  };
  /**
   * Get last packets function
   * \param nodeId the node ID
   * \returns the last packets
   */
  LastPacketsSample GetLastPackets (uint32_t nodeId) const;


  /**
   * Set nodes of interest function
   * \param nodes the collection of nodes
   */
  void SetNodesOfInterest (std::set<uint32_t> nodes);

  /// NetDeviceStatistics structure
  struct NetDeviceStatistics
  {
    /// constructor
    NetDeviceStatistics () : transmittedBytes (0), receivedBytes (0),
                             transmittedPackets (0), receivedPackets (0) {}
    uint64_t transmittedBytes; ///< transmitted bytes
    uint64_t receivedBytes; ///< received bytes
    uint32_t transmittedPackets; ///< transmitted packets
    uint32_t receivedPackets; ///< received packets
  };

  /// NodeStatistics structure
  struct NodeStatistics
  {
    uint32_t nodeId; ///< node ID
    std::vector<NetDeviceStatistics> statistics; ///< statistics
  };

  /**
   * Get node statistics
   * \returns the node statistics
   */
  std::vector<NodeStatistics> GetNodesStatistics () const;

  /// PacketCaptureMode enumeration
  enum PacketCaptureMode {
    PACKET_CAPTURE_DISABLED=1, // packet capture is disabled
    PACKET_CAPTURE_FILTER_HEADERS_OR, // packet capture if any of the indicated headers is present
    PACKET_CAPTURE_FILTER_HEADERS_AND, // packet capture if all of the indicated headers are present
  };

  /// PacketCaptureOptions structure
  struct PacketCaptureOptions
  {
    std::set<TypeId> headers; ///< headers
    uint32_t numLastPackets; ///< num last packets
    PacketCaptureMode mode; ///< mode
  };

  /**
   * Set packet capture options function
   * \param nodeId the node ID
   * \param options the capture options
   */
  void SetPacketCaptureOptions (uint32_t nodeId, PacketCaptureOptions options);


  // Yes, I know, this is just a utility function, not really related to the class in any way.
  /**
   * Utility function - clips a line to a bounding box.
   * \param [in] boundsX1 Bounding box, minimum X coord
   * \param [in] boundsY1 Bounding box, minimum Y coord
   * \param [in] boundsX2 Bounding box, maximum X coord
   * \param [in] boundsY2 Bounding box, maximum Y coord
   * \param [in,out] lineX1 Line, minimum X coord (any on input, clipped to the bounding box on output)
   * \param [in,out] lineY1 Line, minimum Y coord (any on input, clipped to the bounding box on output)
   * \param [in,out] lineX2 Line, maximum X coord (any on input, clipped to the bounding box on output)
   * \param [in,out] lineY2 Line, maximum Y coord (any on input, clipped to the bounding box on output)
   */
  // -#- @lineX1(direction=inout); @lineY1(direction=inout); @lineX2(direction=inout); @lineY2(direction=inout) -#-
  static void LineClipping (double boundsX1, double boundsY1, double boundsX2, double boundsY2, double &lineX1, double &lineY1, double &lineX2, double &lineY2);
  // Don't break the above line or pybindgen will not be able to pick up the above annotation :(

private:

  /**
   * Get packet capture options function
   * \param nodeId the node ID
   * \param outOptions the packet capture options
   * \returns true if successful
   */
  bool GetPacketCaptureOptions (uint32_t nodeId, const PacketCaptureOptions **outOptions) const;
  /**
   * Filter packet function
   * \param packet the packet
   * \param options the capture options
   * \returns true if successful
   */
  static bool FilterPacket (Ptr<const Packet> packet, const PacketCaptureOptions &options);


  typedef std::pair<Ptr<Channel>, uint32_t> TxRecordKey; ///< TxRecordKey typedef

  /// TxRecordValue structure
  struct TxRecordValue
  {
    Time time; ///< time
    Ptr<Node> srcNode; ///< source node
    bool isBroadcast; ///< is broadcast?
  };

  /// TransmissionSampleKey structure
  struct TransmissionSampleKey
  {
    /**
     * less than operator
     *
     * \param other object to compare
     * \return true if less than
     */
    bool operator < (TransmissionSampleKey const &other) const;
    /**
     * equality operator
     *
     * \param other object to compare
     * \return true if equal
     */
    bool operator == (TransmissionSampleKey const &other) const;
    Ptr<Node> transmitter; ///< transmitter
    Ptr<Node> receiver; ///< NULL if broadcast
    Ptr<Channel> channel; ///< channel
  };

  /// TransmissionSampleValue structure
  struct TransmissionSampleValue
  {
    uint32_t bytes; ///< bytes
  };

  // data
  std::map<uint32_t, PacketCaptureOptions> m_packetCaptureOptions; ///< packet capture options
  std::vector<std::string> m_pauseMessages; ///< pause message
  std::map<TxRecordKey, TxRecordValue> m_txRecords; ///< transmit records
  std::map<TransmissionSampleKey, TransmissionSampleValue> m_transmissionSamples; ///< transmission samples
  std::map<Ptr<Node>, uint32_t> m_packetDrops; ///< packt drops
  std::set<uint32_t> m_nodesOfInterest; ///< list of node IDs whose transmissions will be monitored
  std::map<uint32_t, Time> m_packetsOfInterest; ///< list of packet UIDs that will be monitored
  std::map<uint32_t, LastPacketsSample> m_lastPackets; ///< last packets
  std::map<uint32_t, std::vector<NetDeviceStatistics> > m_nodesStatistics; ///< node statsitics

  // Trace callbacks
  /**
   * network transmit common trace callback function
   * \param context the context
   * \param packet the packet
   * \param destination the destination MAC address
   */
  void TraceNetDevTxCommon (std::string const &context, Ptr<const Packet> packet, Mac48Address const &destination);
  /**
   * network receive common trace callback function
   * \param context the context
   * \param packet the packet
   * \param source the source MAC address
   */
  void TraceNetDevRxCommon (std::string const &context, Ptr<const Packet> packet, Mac48Address const &source);

  /**
   * WIFI transmit trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevTxWifi (std::string context, Ptr<const Packet> packet);
  /**
   * WIFI receive trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevRxWifi (std::string context, Ptr<const Packet> packet);

  /**
   * queue drop trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceDevQueueDrop (std::string context, Ptr<const Packet> packet);
  /**
   * ipv4 drop trace callback function
   * \param context the context
   * \param hdr the header
   * \param packet the packet
   * \param reason the drop reason
   * \param dummy_ipv4
   * \param interface the interface
   */
  void TraceIpv4Drop (std::string context, ns3::Ipv4Header const &hdr, Ptr<const Packet> packet, 
                      ns3::Ipv4L3Protocol::DropReason reason, Ptr<Ipv4> dummy_ipv4, uint32_t interface);

  /**
   * CSMA transmit trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevTxCsma (std::string context, Ptr<const Packet> packet);
  /**
   * CSMA receive trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevRxCsma (std::string context, Ptr<const Packet> packet);
  /**
   * CSMA promiscious receive function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevPromiscRxCsma (std::string context, Ptr<const Packet> packet);

  /**
   * Point to point transmit trace calllback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevTxPointToPoint (std::string context, Ptr<const Packet> packet);
  /**
   * Point to point receive trace callback function
   * \param context the context
   * \param packet the packet
   */
  void TraceNetDevRxPointToPoint (std::string context, Ptr<const Packet> packet);

  /**
   * WIMax transmit trace callback function
   * \param context the context
   * \param packet the packet
   * \param destination the destination MAC address
   */
  void TraceNetDevTxWimax (std::string context, Ptr<const Packet> packet, Mac48Address const &destination);
  /**
   * WIMax transmit trace callback function
   * \param context the context
   * \param packet the packet
   * \param source the source MAC address
   */
  void TraceNetDevRxWimax (std::string context, Ptr<const Packet> packet, Mac48Address const &source);

  /**
   * LTE transmit trace callback function
   * \param context the context
   * \param packet the packet
   * \param destination the destination MAC address
   */
  void TraceNetDevTxLte (std::string context, Ptr<const Packet> packet, Mac48Address const &destination);
  /**
   * LTE receive trace callback function
   * \param context the context
   * \param packet the packet
   * \param source the MAC address of the source
   */
  void TraceNetDevRxLte (std::string context, Ptr<const Packet> packet, Mac48Address const &source);

  /**
   * Findnet device statistics function
   * \param node the node
   * \param interface the interface number
   * \returns the device statistics
   */
  inline NetDeviceStatistics & FindNetDeviceStatistics (int node, int interface);

  /**
   * Do pause function
   * \param message the pause message
   */
  void DoPause (std::string const &message);

  bool m_stop; ///< stop?
  Time m_runUntil; ///< run until time
  /// stop simulation callback function
  void CallbackStopSimulation ();
};


}

#endif /* NS3_PYVIZ_H */
