/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef HWMP_PROTOCOL_H
#define HWMP_PROTOCOL_H

#include "ns3/mesh-l2-routing-protocol.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "ns3/traced-value.h"
#include <vector>
#include <map>

namespace ns3 {
class MeshPointDevice;
class Packet;
class Mac48Address;
class UniformRandomVariable;
class RandomVariableStream;
namespace dot11s {
class HwmpProtocolMac;
class HwmpRtable;
class IePerr;
class IePreq;
class IePrep;

/**
 * Structure to encapsulate route change information
 */
struct RouteChange
{
  std::string type;             ///< type of change
  Mac48Address destination;     ///< route destination
  Mac48Address retransmitter;   ///< route source
  uint32_t interface;           ///< interface index
  uint32_t metric;              ///< metric of route
  Time lifetime;                ///< lifetime of route
  uint32_t seqnum;              ///< sequence number of route
};
/**
 * \ingroup dot11s
 *
 * \brief Hybrid wireless mesh protocol -- a mesh routing protocol defined
 * in IEEE 802.11-2012 standard.
 */
class HwmpProtocol : public MeshL2RoutingProtocol
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  HwmpProtocol ();
  ~HwmpProtocol ();
  void DoDispose ();

  /**
   * \brief structure of unreachable destination - address and sequence number
   */
  struct FailedDestination
  {
    Mac48Address destination; ///< destination address
    uint32_t seqnum; ///< sequence number
  };

  /**
   * Route request, inherited from MeshL2RoutingProtocol
   *
   * \param sourceIface the source interface
   * \param source the source address
   * \param destination the destination address
   * \param packet the packet to route
   * \param protocolType the protocol type
   * \param routeReply the route reply
   * \returns true if route exists
   */
  bool RequestRoute (uint32_t  sourceIface, const Mac48Address source, const Mac48Address destination,
                     Ptr<const Packet>  packet, uint16_t  protocolType, RouteReplyCallback  routeReply);
  /**
   * Clean HWMP packet tag from packet; only the packet parameter is used
   *
   * \param fromIface the from interface
   * \param source the source address
   * \param destination the destination address
   * \param packet the packet to route
   * \param protocolType the protocol type
   * \returns true if successful
   */
  bool RemoveRoutingStuff (uint32_t fromIface, const Mac48Address source,
                           const Mac48Address destination, Ptr<Packet>  packet, uint16_t&  protocolType);
  /**
   * \brief Install HWMP on given mesh point.
   * \param mp the MeshPointDevice
   * \returns true if successful
   *
   * Installing protocol causes installation of its interface MAC plugins.
   *
   * Also MP aggregates all installed protocols, HWMP protocol can be accessed
   * via MeshPointDevice::GetObject<dot11s::HwmpProtocol>();
   */
  bool Install (Ptr<MeshPointDevice> mp);
  /**
   * Peer link status function
   * \param meshPontAddress The MAC address of the mesh point
   * \param peerAddress The MAC address of the peer
   * \param interface The interface number
   * \param status The status of the peer link
   */
  void PeerLinkStatus (Mac48Address meshPontAddress, Mac48Address peerAddress, uint32_t interface,bool status);
  /**
   * \brief This callback is used to obtain active neighbours on a given interface
   * \param cb is a callback, which returns a list of addresses on given interface (uint32_t)
   */
  void SetNeighboursCallback (Callback<std::vector<Mac48Address>, uint32_t> cb);
  /// \name Proactive PREQ mechanism:
  ///@{
  /// Set the current node as root
  void SetRoot ();
  /// Unset the current node as root
  void UnsetRoot ();
  ///@}

  /// \brief Statistics:
  ///@{
  /**
   *  Print statistics counters
   *  \param os the output stream
   */
  void Report (std::ostream & os) const;
  ///\brief Reset Statistics:
  void ResetStats ();
  ///@}

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
   * \brief Get pointer to HWMP routing table
   * \return pointer to routing table
   */
  Ptr<HwmpRtable> GetRoutingTable (void) const;

private:
  /// allow HwmpProtocolMac class friend access
  friend class HwmpProtocolMac;

  virtual void DoInitialize ();

  /**
   * assignment operator
   * \param hwmp the HWMP protocol to assign
   * \returns the assigned value
   */
  HwmpProtocol& operator= (const HwmpProtocol & hwmp);
  /**
   * Copy constructor - defined but not implemented (on purpose)
   * \param hwmp the HWMP protocol
   */
  HwmpProtocol (const HwmpProtocol & hwmp);

  /**
   * \brief Structure of path error: IePerr and list of receivers:
   * interfaces and MAC address
   */
  struct PathError
  {
    std::vector<FailedDestination> destinations; ///< destination list: Mac48Address and sequence number
    std::vector<std::pair<uint32_t, Mac48Address> > receivers; ///< list of PathError receivers (in case of unicast PERR)
  };
  /// Packet waiting its routing information
  struct QueuedPacket
  {
    Ptr<Packet> pkt; ///< the packet
    Mac48Address src; ///< src address
    Mac48Address dst; ///< dst address
    uint16_t protocol; ///< protocol number
    uint32_t inInterface; ///< incoming device interface ID. (if packet has come from upper layers, this is Mesh point ID)
    RouteReplyCallback reply; ///< how to reply

    QueuedPacket ();
  };
  typedef std::map<uint32_t, Ptr<HwmpProtocolMac> > HwmpProtocolMacMap; ///< HwmpProtocolMacMap typedef
  /**
   * Like RequestRoute, but for unicast packets
   *
   * \param sourceIface the source interface
   * \param source the source address
   * \param destination the destination address
   * \param packet the packet to route
   * \param protocolType the protocol type
   * \param routeReply the route reply callback
   * \param ttl the TTL
   * \returns true if forwarded
   */
  bool ForwardUnicast (uint32_t sourceIface, const Mac48Address source, const Mac48Address destination,
                       Ptr<Packet> packet, uint16_t protocolType, RouteReplyCallback routeReply, uint32_t ttl);

  /// \name Interaction with HWMP MAC plugin
  ///@{
  /**
   * \brief Handler for receiving Path Request
   *
   * \param preq the IE preq
   * \param from the from address
   * \param interface the interface
   * \param fromMp the 'from MP' address
   * \param metric the metric
   */
  void ReceivePreq (IePreq preq, Mac48Address from, uint32_t interface, Mac48Address fromMp, uint32_t metric);
  /**
   * \brief Handler for receiving Path Reply
   *
   * \param prep the IE prep
   * \param from the from address
   * \param interface the interface
   * \param fromMp the 'from MP' address
   * \param metric the metric
   */
  void ReceivePrep (IePrep prep, Mac48Address from, uint32_t interface, Mac48Address fromMp, uint32_t metric);
  /**
   * \brief Handler for receiving Path Error
   *
   * \param destinations the list of failed destinations
   * \param from the from address
   * \param interface the interface
   * \param fromMp the from MP address
   */
  void ReceivePerr (std::vector<FailedDestination> destinations, Mac48Address from, uint32_t interface, Mac48Address fromMp);
   /**
    * \brief Send Path Reply
    * \param src the source address
    * \param dst the destination address
    * \param retransmitter the retransmitter address
    * \param initMetric the initial metric
    * \param originatorDsn the originator DSN
    * \param destinationSN the destination DSN
    * \param lifetime the lifetime
    * \param interface the interface
    */
  void SendPrep (
    Mac48Address src,
    Mac48Address dst,
    Mac48Address retransmitter,
    uint32_t initMetric,
    uint32_t originatorDsn,
    uint32_t destinationSN,
    uint32_t lifetime,
    uint32_t interface);
  /**
   * \brief forms a path error information element when list of destination fails on a given interface
   * \attention removes all entries from routing table!
   *
   * \param destinations vector of failed destinations
   * \return PathError
   */
  PathError MakePathError (std::vector<FailedDestination> destinations);
  /**
   * \brief Forwards a received path error
   * \param perr the path error
   */
  void ForwardPathError (PathError perr);
  /**
   * \brief Passes a self-generated PERR to interface-plugin
   * \param perr the path error
   */
  void InitiatePathError (PathError perr);
  /**
   * Get PERR receivers
   *
   * \param failedDest
   * \return list of addresses where a PERR should be sent to
   */
  std::vector<std::pair<uint32_t, Mac48Address> > GetPerrReceivers (std::vector<FailedDestination> failedDest);

  /**
   * Get PREQ receivers
   *
   * \param interface
   * \return list of addresses where a PREQ should be sent to
   */
  std::vector<Mac48Address> GetPreqReceivers (uint32_t interface);
  /**
   * Get broadcast receivers
   *
   * \param interface
   * \return list of addresses where a broadcast should be retransmitted
   */
  std::vector<Mac48Address> GetBroadcastReceivers (uint32_t interface);
  /**
   * \brief MAC-plugin asks whether the frame can be dropped. Protocol automatically updates seqno.
   *
   * \return true if frame can be dropped
   * \param seqno is the sequence number of source
   * \param source is the source address
   */
  bool DropDataFrame (uint32_t seqno, Mac48Address source);
  ///@}

  /// Route discovery time:
  TracedCallback<Time> m_routeDiscoveryTimeCallback;
  /// RouteChangeTracedCallback typedef
  typedef TracedCallback <struct RouteChange> RouteChangeTracedCallback;
  /// Route change trace source
  TracedCallback<struct RouteChange> m_routeChangeTraceSource;

  // /\name Methods related to Queue/Dequeue procedures
  ///@{
  /**
   * Queue a packet
   * \param packet the packet to be queued
   * \return true on success
   */
  bool QueuePacket (QueuedPacket packet);
  /**
   * Dequeue the first packet for a given destination
   * \param dst the destination
   * \return the dequeued packet
   */
  QueuedPacket  DequeueFirstPacketByDst (Mac48Address dst);
  /**
   * Dequeue the first packet in the queue
   * \return the dequeued packet
   */
  QueuedPacket  DequeueFirstPacket ();
  /**
   * Signal the protocol that the reactive path toward a destination is now available
   * \param dst the destination
   */
  void ReactivePathResolved (Mac48Address dst);
  /**
   * Signal the protocol that the proactive path is now available
   */
  void ProactivePathResolved ();
  ///@}

  /// \name Methods responsible for path discovery retry procedure:
  ///@{
  /**
   * \brief checks when the last path discovery procedure was started for a given destination.
   * \return true if should send PREQ
   * \param dst is the destination address
   *
   * If the retry counter has not achieved the maximum level - preq should not be sent
   */
  bool  ShouldSendPreq (Mac48Address dst);

  /**
   * \brief Generates PREQ retry when retry timeout has expired and route is still unresolved.
   * \param dst is the destination address
   * \param numOfRetry is the number of retries
   *
   * When PREQ retry has achieved the maximum level - retry mechanism should be canceled
   */
  void  RetryPathDiscovery (Mac48Address dst, uint8_t numOfRetry);
  /// Proactive Preq routines:
  void SendProactivePreq ();
  ///@}

  ///\return address of MeshPointDevice
  Mac48Address GetAddress ();
  /// \name Methods needed by HwmpMacLugin to access protocol parameters:
  ///@{
  /**
   * Get do flag function
   * \returns DO flag
   */
  bool GetDoFlag ();
  /**
   * Get rf flag function
   * \returns the RF flag
   */
  bool GetRfFlag ();
  /**
   * Get PREQ minimum interval function
   * \returns the PREQ
   */
  Time GetPreqMinInterval ();
  /**
   * Get PERR minimum interval function
   * \returns the PERR minimum interval
   */
  Time GetPerrMinInterval ();
  /**
   * Get maximum TTL function
   * \returns the maximum TTL
   */
  uint8_t GetMaxTtl ();
  /**
   * Get next period function
   * \returns the next period
   */
  uint32_t GetNextPreqId ();
  /**
   * Get next HWMP sequence no function
   * \returns the next HWMP sequence number
   */
  uint32_t GetNextHwmpSeqno ();
  /**
   * Get active path lifetime function
   * \returns the active path lifetime
   */
  uint32_t GetActivePathLifetime ();
  /**
   * Get unicast PERR threshold function
   * \returns the unicast PERR threshold
   */
  uint8_t GetUnicastPerrThreshold ();
  ///@}

private:
  /// Statistics structure
  struct Statistics
  {
    uint16_t txUnicast; ///< transmit unicast
    uint16_t txBroadcast; ///< transmit broadcast
    uint32_t txBytes; ///< transmit bytes
    uint16_t droppedTtl; ///< dropped TTL
    uint16_t totalQueued; ///< total queued
    uint16_t totalDropped; ///< total dropped
    uint16_t initiatedPreq; ///< initiated PREQ
    uint16_t initiatedPrep; ///< initiated PREP
    uint16_t initiatedPerr; ///< initiated PERR

    /**
     * Print function
     * \param os The output stream
     */
    void Print (std::ostream & os) const;
    /// constructor
    Statistics ();
  };
  Statistics m_stats;  ///< statistics 

  HwmpProtocolMacMap m_interfaces; ///< interfaces
  Mac48Address m_address; ///< address
  uint32_t m_dataSeqno; ///< data sequence no
  uint32_t m_hwmpSeqno; ///< HWMP sequence no
  uint32_t m_preqId; ///< PREQ ID
  /// \name Sequence number filters
  ///@{
  /// Data sequence number database
  std::map<Mac48Address, uint32_t> m_lastDataSeqno;
  /// keeps HWMP seqno (first in pair) and HWMP metric (second in pair) for each address
  std::map<Mac48Address, std::pair<uint32_t, uint32_t> > m_hwmpSeqnoMetricDatabase;
  ///@}

  /// Routing table
  Ptr<HwmpRtable> m_rtable;

  /// PreqEvent structure
  struct PreqEvent {
    EventId preqTimeout; ///< PREQ timeout
    Time whenScheduled; ///< scheduled time
  };

  std::map<Mac48Address, PreqEvent> m_preqTimeouts; ///< PREQ timeouts
  EventId m_proactivePreqTimer; ///< proactive PREQ timer
  /// Random start in Proactive PREQ propagation
  Time m_randomStart;
  /// Packet Queue
  std::vector<QueuedPacket> m_rqueue;
  
  /// \name HWMP-protocol parameters
  /// These are all Attributes
  ///@{
  uint16_t m_maxQueueSize;                        //!< Maximum number of packets we can store when resolving route
  uint8_t m_dot11MeshHWMPmaxPREQretries;          //!< Maximum number of retries before we suppose the destination to be unreachable
  Time m_dot11MeshHWMPnetDiameterTraversalTime;   //!< Time we suppose the packet to go from one edge of the network to another
  Time m_dot11MeshHWMPpreqMinInterval;            //!< Minimal interval between to successive PREQs
  Time m_dot11MeshHWMPperrMinInterval;            //!< Minimal interval between to successive PREQs
  Time m_dot11MeshHWMPactiveRootTimeout;          //!< Lifetime of proactive routing information
  Time m_dot11MeshHWMPactivePathTimeout;          //!< Lifetime of reactive routing information
  Time m_dot11MeshHWMPpathToRootInterval;         //!< Interval between two successive proactive PREQs
  Time m_dot11MeshHWMPrannInterval;               //!< Lifetime of proactive routing information
  bool m_isRoot;                                  //!< True if the node is a root
  uint8_t m_maxTtl;                               //!< Initial value of Time To Live field
  uint8_t m_unicastPerrThreshold;                 //!< Maximum number of PERR receivers, when we send a PERR as a chain of unicasts
  uint8_t m_unicastPreqThreshold;                 //!< Maximum number of PREQ receivers, when we send a PREQ as a chain of unicasts
  uint8_t m_unicastDataThreshold;                 //!< Maximum number of broadcast receivers, when we send a broadcast as a chain of unicasts
  bool m_doFlag;                                  //!< Destination only HWMP flag
  bool m_rfFlag;                                  //!< Reply and forward flag
  ///@}
  
  /// Random variable for random start time
  Ptr<UniformRandomVariable> m_coefficient; ///< coefficient
  Callback <std::vector<Mac48Address>, uint32_t> m_neighboursCallback; ///< neighbors callback
};
} // namespace dot11s
} // namespace ns3
#endif
