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
 * Author: George F. Riley<riley@ece.gatech.edu>
 * Author: John Abraham <john.abraham@gatech.edu>
 * Contributions: Eugene Kalishenko <ydginster@gmail.com> (Open Source and Linux Laboratory http://dev.osll.ru/)
 */

// Interface between ns3 and the network animator

#ifndef ANIMATION_INTERFACE__H
#define ANIMATION_INTERFACE__H

#include <string>
#include <cstdio>
#include <map>

#include "ns3/ptr.h"
#include "ns3/net-device.h"
#include "ns3/node-container.h"
#include "ns3/nstime.h"
#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/random-variable-stream.h"
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/mac48-address.h"
#include "ns3/lte-ue-net-device.h"
#include "ns3/lte-enb-net-device.h"
#include "ns3/uan-phy-gen.h"
#include "ns3/wifi-phy.h"
#include "ns3/rectangle.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/wifi-phy.h"

namespace ns3 {

#define MAX_PKTS_PER_TRACE_FILE 100000
#define PURGE_INTERVAL 5
#define NETANIM_VERSION "netanim-3.108"
#define CHECK_STARTED_INTIMEWINDOW {if (!m_started || !IsInTimeWindow ()) {return; }}
#define CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS {if (!m_started || !IsInTimeWindow () || !m_trackPackets) {return; }}


struct NodeSize;
class WifiPsdu;

/**
 * \defgroup netanim Network Animation
 *
 * This section documents the API of the ns-3 netanim module. For a generic functional description, please refer to the ns-3 manual.
 */

/**
 * \ingroup netanim
 *
 * \brief Interface to network animator
 *
 * Provides functions that facilitate communications with an
 * external or internal network animator.
 */
class AnimationInterface
{
public:
  /**
   * \brief Constructor
   * \param filename The Filename for the trace file used by the Animator
   *
   */
  AnimationInterface (const std::string filename);

  /**
   * Counter Types
   */
  typedef enum
  {
    UINT32_COUNTER,
    DOUBLE_COUNTER
  } CounterType;

  /**
   * \brief typedef for WriteCallBack used for listening to AnimationInterface
   * write messages
   *
   */
  typedef void (*AnimWriteCallback)(const char * str);

  /**
   * \brief Destructor for the animator interface.
   *
   */
  ~AnimationInterface ();

  /**
   * \brief Enable tracking of Ipv4 L3 Protocol Counters such as Tx, Rx, Drop
   *
   * \param startTime Start Time for capturing values
   * \param stopTime Stop Time for capturing values
   * \param pollInterval The periodic interval at which the counters are written to the trace file
   *        Default: 1s
   */
  void EnableIpv4L3ProtocolCounters (Time startTime, Time stopTime, Time pollInterval = Seconds (1));

  /**
   * \brief Enable tracking of Queue Counters such as Enqueue, Dequeue, Queue Drops
   *
   * \param startTime Start Time for capturing values
   * \param stopTime Stop Time for capturing values
   * \param pollInterval The periodic interval at which the counters are written to the trace file
   *        Default: 1s
   */
  void EnableQueueCounters (Time startTime, Time stopTime, Time pollInterval = Seconds (1));

  /**
   * \brief Enable tracking of Wifi Mac Counters such as Tx, TxDrop, Rx, RxDrop
   *
   * \param startTime Start Time for capturing values
   * \param stopTime Stop Time for capturing values
   * \param pollInterval The periodic interval at which the counters are written to the trace file
   *        Default: 1s
   */
  void EnableWifiMacCounters (Time startTime, Time stopTime, Time pollInterval = Seconds (1));

  /**
   * \brief Enable tracking of Wifi Phy Counters such as TxDrop, RxDrop
   *
   * \param startTime Start Time for capturing values
   * \param stopTime Stop Time for capturing values
   * \param pollInterval The periodic interval at which the counters are written to the trace file
   *        Default: 1s
   */
  void EnableWifiPhyCounters (Time startTime, Time stopTime, Time pollInterval = Seconds (1));

  /**
   * \brief Enable tracking of the Ipv4 routing table for all Nodes
   *
   * \param fileName Trace file for storing routing table information
   * \param startTime Start time for capture
   * \param stopTime  End time for capture
   * \param pollInterval The periodic interval at which routing table information is polled
   *        Default: 5s
   *
   * \returns reference to this AnimationInterface object
   */
  AnimationInterface & EnableIpv4RouteTracking (std::string fileName, Time startTime, Time stopTime, Time pollInterval = Seconds (5));

  /**
   * \brief Enable tracking of the Ipv4 routing table for a set of Nodes
   *
   * \param fileName Trace file for storing routing table information
   * \param startTime Start time for capture
   * \param stopTime  End time for capture
   * \param nc A NodeContainer containing nodes for which Routing table has to be tracked
   * \param pollInterval The periodic interval at which routing table information is polled
   *        Default: 5s
   *
   * \returns reference to this AnimationInterface object
   */
  AnimationInterface & EnableIpv4RouteTracking (std::string fileName, Time startTime, Time stopTime, NodeContainer nc, Time pollInterval = Seconds (5));

  /**
   * \brief Check if AnimationInterface is initialized
   *
   * \returns true if AnimationInterface was already initialized
   *
   */
  static bool IsInitialized (void);

  /**
   * \brief Specify the time at which capture should start
   *
   * \param t The time at which AnimationInterface should begin capture of traffic info
   *
   */
  void SetStartTime (Time t);

  /**
   * \brief Specify the time at which capture should stop
   *
   * \param t The time at which AnimationInterface should stop capture of traffic info
   *
   */
  void SetStopTime (Time t);

  /**
   * \brief Set Max packets per trace file
   * \param maxPktsPerFile The maximum number of packets per trace file.
            AnimationInterface will create trace files with the following
            filenames : filename, filename-1, filename-2..., filename-N
            where each file contains packet info for 'maxPktsPerFile' number of packets
   *
   */
  void SetMaxPktsPerTraceFile (uint64_t maxPktsPerFile);

  /**
   * \brief Set mobility poll interval:WARNING: setting a low interval can
   * cause slowness
   *
   * \param t Time interval between fetching mobility/position information
   * Default: 0.25s
   *
   */
  void SetMobilityPollInterval (Time t);

  /**
   * \brief Set a callback function to listen to AnimationInterface write events
   *
   * \param cb Address of callback function
   *
   */
  void SetAnimWriteCallback (AnimWriteCallback cb);

  /**
   * \brief Reset the write callback function
   *
   */
  void ResetAnimWriteCallback ();

  /**
   * \brief Helper function to set Constant Position for a given node
   * \param n Ptr to the node
   * \param x X co-ordinate of the node
   * \param y Y co-ordinate of the node
   * \param z Z co-ordinate of the node
   *
   */
  static void SetConstantPosition (Ptr <Node> n, double x, double y, double z = 0);

  /**
   * \brief Helper function to update the description for a given node
   * \param n Ptr to the node
   * \param descr A string to briefly describe the node
   *
   */
  void UpdateNodeDescription (Ptr <Node> n, std::string descr);

  /**
   * \brief Helper function to update the description for a given node
   * \param nodeId Id of the node
   * \param descr A string to briefly describe the node
   *
   */
  void UpdateNodeDescription (uint32_t nodeId, std::string descr);

  /**
   * \brief Helper function to update the image of a node
   * \param nodeId Id of the node
   * \param resourceId Id of the image resource that was previously added
   *
   */
  void UpdateNodeImage (uint32_t nodeId, uint32_t resourceId);

  /**
   * \brief Helper function to update the size of a node
   * \param nodeId Id of the node
   * \param width Width of the node
   * \param height Height of the node
   *
   */
  void UpdateNodeSize (uint32_t nodeId, double width, double height);

  /**
   * \brief Helper function to update the node color
   * \param n Ptr to the node
   * \param r Red component value (0-255)
   * \param g Green component value (0-255)
   * \param b Blue component value (0-255)
   *
   */
  void UpdateNodeColor (Ptr <Node> n, uint8_t r, uint8_t g, uint8_t b);

  /**
   * \brief Helper function to update the node color
   * \param nodeId Id of the node
   * \param r Red component value (0-255)
   * \param g Green component value (0-255)
   * \param b Blue component value (0-255)
   *
   */
  void UpdateNodeColor (uint32_t nodeId, uint8_t r, uint8_t g, uint8_t b);

  /**
   * \brief Helper function to update a node's counter referenced by the nodeCounterId
   * \param nodeCounterId The counter Id obtained from AddNodeCounter
   * \param nodeId Node Id of the node
   * \param counter Current value of the counter
   *
   */
  void UpdateNodeCounter (uint32_t nodeCounterId, uint32_t nodeId, double counter);

  /**
   * \brief Helper function to set the background image
   * \param fileName File name of the background image
   * \param x X co-ordinate of the image
   * \param y Y co-ordinate of the image
   * \param scaleX X scale of the image
   * \param scaleY Y scale of the image
   * \param opacity Opacity of the background: A value between 0.0 and 1.0. 0.0 is transparent,
   *        1.0 is opaque
   *
   */
  void SetBackgroundImage (std::string fileName, double x, double y, double scaleX, double scaleY, double opacity);

  /**
   * \brief Helper function to update the description for a link
   * \param fromNode Node Id of the "from Node" of the p2p link
   * \param toNode Node Id of the "to Node" of the p2p link
   * \param linkDescription Description of the link such as link bandwidth
   *
   */
  void UpdateLinkDescription (uint32_t fromNode, uint32_t toNode,
                              std::string linkDescription);

  /**
   * \brief Helper function to update the description for a link
   * \param fromNode Ptr to the "from Node" of the p2p link
   * \param toNode Ptr to the "to Node" of the p2p link
   * \param linkDescription Description of the link such as link bandwidth
   *
   */
  void UpdateLinkDescription (Ptr <Node> fromNode, Ptr <Node> toNode,
                              std::string linkDescription);

  /**
   * \brief Helper function to print the routing path from a source node to destination IP
   * \param fromNodeId The source node
   * \param destinationIpv4Address The destination Ipv4 Address
   *
   * \returns reference to this AnimationInterface object
   */
  AnimationInterface & AddSourceDestination (uint32_t fromNodeId, std::string destinationIpv4Address);

  /**
   * \brief Is AnimationInterface started
   *
   * \returns true if AnimationInterface was started
   */
  bool IsStarted (void);

  /**
   * \brief Do not trace packets. This helps reduce the trace file size if AnimationInterface is solely
   *        used for tracking mobility, routing paths and counters
   */
  void SkipPacketTracing ();

  /**
   *
   * \brief Enable Packet metadata
   * \param enable if true enables writing the packet metadata to the XML trace file
   *        if false disables writing the packet metadata
   *
   */
  void EnablePacketMetadata (bool enable = true);

  /**
   *
   * \brief Get trace file packet count (This used only for testing)
   *
   * \returns Number of packets recorded in the current trace file
   */
  uint64_t GetTracePktCount ();

  /**
   *
   * \brief Setup a node counter
   * \param counterName A string to identify the counter
   * \param counterType The type of the counter, such as uint32, double etc
   *
   * \returns The id of the counter to be used as a reference for future
   */
  uint32_t AddNodeCounter (std::string counterName, CounterType counterType);

  /**
   *
   * \brief Add a resource such as the path to an image file
   * \param resourcePath Absolute Path to an image/resource
   *
   * \returns a number identifying the resource
   */
  uint32_t AddResource (std::string resourcePath);

  /**
   *
   * \brief Get node's energy fraction (This used only for testing)
   * \param node
   *
   * \returns current node's remaining energy (between [0, 1])
   */
  double GetNodeEnergyFraction (Ptr <const Node> node) const;

private:
  /**
   * AnimPacketInfo class
   */
  class AnimPacketInfo

  {
public:
    AnimPacketInfo ();
    /**
     * Constructor
     *
     * \param pInfo anim packet info
     */
    AnimPacketInfo (const AnimPacketInfo & pInfo);
    /**
     * Constructor
     *
     * \param tx_nd transmit device
     * \param fbTx fb transmit
     * \param txNodeId transmit node ID
     */
    AnimPacketInfo (Ptr <const NetDevice> tx_nd, const Time fbTx, uint32_t txNodeId = 0);
    Ptr <const NetDevice> m_txnd; ///< transmit device
    uint32_t m_txNodeId; ///< node ID
    double m_fbTx; ///< fb transmit
    double m_lbTx; ///< lb transmit
    double m_fbRx; ///< fb receive
    double m_lbRx; ///< lb receive
    Ptr <const NetDevice> m_rxnd; ///< receive device
    /**
     * Process receive begin
     * \param nd the device
     * \param fbRx
     */
    void ProcessRxBegin (Ptr <const NetDevice> nd, const double fbRx);
  };

  /// RGB structure
  typedef struct
  {
    uint8_t r;   ///< r
    uint8_t g;   ///< g
    uint8_t b;   ///< b
  } Rgb;   ///< RGB structure

  /// P2pLinkNodeIdPair structure
  typedef struct
  {
    uint32_t fromNode; ///< from node
    uint32_t toNode; ///< to node
  } P2pLinkNodeIdPair; ///< P2P link node id pair

  /// LinkProperties structure
  typedef struct
  {
    std::string fromNodeDescription; ///< from node description
    std::string toNodeDescription; ///< to node description
    std::string linkDescription; ///< link description
  } LinkProperties; ///< link properties

  /// LinkPairCompare structure
  struct LinkPairCompare
  {
    /**
     * comparison operator
     *
     * \param first
     * \param second
     * \return true if equal
     */
    bool operator () (P2pLinkNodeIdPair first, P2pLinkNodeIdPair second) const
    {
      //Check if they are the same node pairs but flipped
      if (  ((first.fromNode == second.fromNode) && (first.toNode == second.toNode))
            || ((first.fromNode == second.toNode) && (first.toNode == second.fromNode)) )
        {
          return false;
        }
      std::ostringstream oss1;
      oss1 << first.fromNode << first.toNode;
      std::ostringstream oss2;
      oss2 << second.fromNode << second.toNode;
      return oss1.str () < oss2.str ();
    }

  };

  /// Ipv4RouteTrackElement structure
  typedef struct
  {
    std::string destination;   ///< destination
    uint32_t fromNodeId;   ///< from node ID
  } Ipv4RouteTrackElement;   ///< IPv4 route track element

  /// Ipv4RoutePathElement structure
  typedef struct
  {
    uint32_t nodeId;   ///< node ID
    std::string nextHop;   ///< next hop
  } Ipv4RoutePathElement;   ///< IPv4 route path element

  /// ProtocolType enumeration
  typedef enum
  {
    UAN,
    LTE,
    WIFI,
    WIMAX,
    CSMA,
    LRWPAN,
    WAVE
  } ProtocolType;

  /// NodeSize structure
  typedef struct
  {
    double width;   ///< width
    double height;   ///< height
  } NodeSize;   ///< node size
  typedef std::map <P2pLinkNodeIdPair, LinkProperties, LinkPairCompare> LinkPropertiesMap; ///< LinkPropertiesMap typedef
  typedef std::map <uint32_t, std::string> NodeDescriptionsMap; ///< NodeDescriptionsMap typedef
  typedef std::map <uint32_t, Rgb> NodeColorsMap; ///< NodeColorsMap typedef
  typedef std::map<uint64_t, AnimPacketInfo> AnimUidPacketInfoMap; ///< AnimUidPacketInfoMap typedef
  typedef std::map <uint32_t, double> EnergyFractionMap; ///< EnergyFractionMap typedef
  typedef std::vector <Ipv4RoutePathElement> Ipv4RoutePathElements; ///< Ipv4RoutePathElements typedef
  typedef std::multimap <uint32_t, std::string> NodeIdIpv4Map; ///< NodeIdIpv4Map typedef
  typedef std::multimap <uint32_t, std::string> NodeIdIpv6Map; ///< NodeIdIpv6Map typedef
  typedef std::pair <uint32_t, std::string> NodeIdIpv4Pair; ///< NodeIdIpv4Pair typedef
  typedef std::pair <uint32_t, std::string> NodeIdIpv6Pair; ///< NodeIdIpv6Pair typedef


  // Node Counters
  typedef std::map <uint32_t, uint64_t> NodeCounterMap64; ///< NodeCounterMap64 typedef


  /// AnimXmlElement class
  class AnimXmlElement
  {
public:
    /**
     * Constructor
     *
     * \param tagName tag name
     * \param emptyElement empty element?
     */
    AnimXmlElement (std::string tagName, bool emptyElement = true);
    template <typename T>
    /**
     * Add attribute function
     * \param attribute the attribute name
     * \param value the attribute value
     * \param xmlEscape true to escape
     */
    void AddAttribute (std::string attribute, T value, bool xmlEscape = false);
    /**
     * Set text function
     * \param text the text for the element
     */
    void SetText (std::string text);
    /**
     * Append child function
     * \param e the element to add as a child
     */
    void AppendChild (AnimXmlElement e);
    /**
     * Get text for the element function
     * \param autoClose auto close the element
     * \returns the text
     */
    std::string ToString (bool autoClose = true);

private:
    std::string m_tagName; ///< tag name
    std::string m_text; ///< element string
    std::vector<std::string> m_attributes; ///< list of attributes
    std::vector<std::string> m_children; ///< list of children

  };



  // ##### State #####

  FILE * m_f; ///< File handle for output (0 if none)
  FILE * m_routingF; ///< File handle for routing table output (0 if None);
  Time m_mobilityPollInterval; ///< mobility poll interval
  std::string m_outputFileName; ///< output file name
  uint64_t gAnimUid;     ///< Packet unique identifier used by AnimationInterface
  AnimWriteCallback m_writeCallback; ///< write callback
  bool m_started; ///< started
  bool m_enablePacketMetadata; ///< enable packet metadata
  Time m_startTime; ///< start time
  Time m_stopTime; ///< stop time
  uint64_t m_maxPktsPerFile; ///< maximum pakets per file
  std::string m_originalFileName; ///< original file name
  Time m_routingStopTime; ///< routing stop time
  std::string m_routingFileName; ///< routing file name
  Time m_routingPollInterval; ///< routing poll interval
  NodeContainer m_routingNc; ///< routing node container
  Time m_ipv4L3ProtocolCountersStopTime; ///< IPv4 L3 protocol counters stop time
  Time m_ipv4L3ProtocolCountersPollInterval; ///< IPv4 L3 protocol counters poll interval
  Time m_queueCountersStopTime; ///< queue counters stop time
  Time m_queueCountersPollInterval; ///< queue counters poll interval
  Time m_wifiMacCountersStopTime; ///< wifi MAC counters stop time
  Time m_wifiMacCountersPollInterval; ///< wifi MAC counters poll interval
  Time m_wifiPhyCountersStopTime; ///< wifi Phy counters stop time
  Time m_wifiPhyCountersPollInterval; ///< wifi Phy counters poll interval
  static Rectangle * userBoundary; ///< user boundary
  bool m_trackPackets; ///< track packets

  // Counter ID
  uint32_t m_remainingEnergyCounterId; ///< remaining energy counter ID

  uint32_t m_ipv4L3ProtocolTxCounterId; ///< IPv4 L3 protocol transmit counter ID
  uint32_t m_ipv4L3ProtocolRxCounterId; ///< IPv4 L3 protocol receive counter ID
  uint32_t m_ipv4L3ProtocolDropCounterId; ///< IPv4 protocol drop counter ID

  uint32_t m_queueEnqueueCounterId; ///< queue enqueue counter ID
  uint32_t m_queueDequeueCounterId; ///< queue dequeue counter ID
  uint32_t m_queueDropCounterId; ///< queue drop counter ID

  uint32_t m_wifiMacTxCounterId; ///< wifi MAC transmit counter ID
  uint32_t m_wifiMacTxDropCounterId; ///< wifi MAC transmit drop counter ID
  uint32_t m_wifiMacRxCounterId; ///< wifi MAC receive counter ID
  uint32_t m_wifiMacRxDropCounterId; ///< wifi MAC receive drop counter ID

  uint32_t m_wifiPhyTxDropCounterId; ///< wifi Phy transmit drop counter ID
  uint32_t m_wifiPhyRxDropCounterId; ///< wifi Phy receive drop counter ID

  AnimUidPacketInfoMap m_pendingWifiPackets; ///< pending wifi packets
  AnimUidPacketInfoMap m_pendingWimaxPackets; ///< pending wimax packets
  AnimUidPacketInfoMap m_pendingLrWpanPackets; ///< pending LR-WPAN packets
  AnimUidPacketInfoMap m_pendingLtePackets; ///< pending LTE packets
  AnimUidPacketInfoMap m_pendingCsmaPackets; ///< pending CSMA packets
  AnimUidPacketInfoMap m_pendingUanPackets; ///< pending UAN packets
  AnimUidPacketInfoMap m_pendingWavePackets; ///< pending WAVE packets

  std::map <uint32_t, Vector> m_nodeLocation; ///< node location
  std::map <std::string, uint32_t> m_macToNodeIdMap; ///< MAC to node ID map
  std::map <std::string, uint32_t> m_ipv4ToNodeIdMap; ///< IPv4 to node ID map
  std::map <std::string, uint32_t> m_ipv6ToNodeIdMap; ///< IPv6 to node ID map
  NodeIdIpv4Map m_nodeIdIpv4Map; ///< node ID to IPv4 map
  NodeIdIpv6Map m_nodeIdIpv6Map; ///< node ID to IPv6 map

  NodeColorsMap m_nodeColors; ///< node colors
  NodeDescriptionsMap m_nodeDescriptions; ///< node description
  LinkPropertiesMap m_linkProperties; ///< link properties
  EnergyFractionMap m_nodeEnergyFraction; ///< node energy fraction
  uint64_t m_currentPktCount; ///< current packet count
  std::vector <Ipv4RouteTrackElement> m_ipv4RouteTrackElements; ///< IPv route track elements
  std::map <uint32_t, NodeSize> m_nodeSizes; ///< node sizes
  std::vector <std::string> m_resources; ///< resources
  std::vector <std::string> m_nodeCounters; ///< node counters

  /* Value-added custom counters */
  NodeCounterMap64 m_nodeIpv4Drop; ///< node IPv4 drop
  NodeCounterMap64 m_nodeIpv4Tx; ///< node IPv4 transmit
  NodeCounterMap64 m_nodeIpv4Rx; ///< node IPv4 receive
  NodeCounterMap64 m_nodeQueueEnqueue; ///< node queue enqueue
  NodeCounterMap64 m_nodeQueueDequeue; ///< node queue dequeue
  NodeCounterMap64 m_nodeQueueDrop; ///< node queue drop
  NodeCounterMap64 m_nodeWifiMacTx; ///< node wifi MAC transmit
  NodeCounterMap64 m_nodeWifiMacTxDrop; ///< node wifi MAC transmit drop
  NodeCounterMap64 m_nodeWifiMacRx; ///< node wifi MAC receive
  NodeCounterMap64 m_nodeWifiMacRxDrop; ///< node wifi MAC receive drop
  NodeCounterMap64 m_nodeWifiPhyTxDrop; ///< node wifi Phy transmit drop
  NodeCounterMap64 m_nodeWifiPhyRxDrop; ///< node wifi Phy receive drop
  NodeCounterMap64 m_nodeLrWpanMacTx; ///< node LR-WPAN MAC transmit
  NodeCounterMap64 m_nodeLrWpanMacTxDrop; ///< node LR-WPAN MAC transmit drop
  NodeCounterMap64 m_nodeLrWpanMacRx; ///< node LR-WPAN MAC receive
  NodeCounterMap64 m_nodeLrWpanMacRxDrop; ///< node LR-WPAN MAC receive drop

  /**
   * Get elements from context
   * \param context the context string
   * \returns the elements
   */
  const std::vector<std::string> GetElementsFromContext (const std::string& context) const;
  /**
   * Get node from context
   * \param context the context string
   * \returns the node
   */
  Ptr <Node> GetNodeFromContext (const std::string& context) const;
  /**
   * Get net device from context
   * \param context the context string
   * \returns the device
   */
  Ptr <NetDevice> GetNetDeviceFromContext (std::string context);

  // ##### General #####
  /**
   * Start animation function
   *
   * \param restart
   */
  void StartAnimation (bool restart = false);
  /**
   * Set output file function
   *
   * \param fn the file name
   * \param routing
   */
  void SetOutputFile (const std::string& fn, bool routing = false);
  /**
   * Stop animation function
   *
   * \param onlyAnimation
   */
  void StopAnimation (bool onlyAnimation = false);
  /**
   * Counter type to string function
   * \param counterType the counter type
   * \returns the string
   */
  std::string CounterTypeToString (CounterType counterType);
  /**
   * Get packet metadata function
   * \param p the packet
   * \returns the meta data
   */
  std::string GetPacketMetadata (Ptr<const Packet> p);
  /**
   * Add byte tag function
   * \param animUid the UID
   * \param p the packet
   */
  void AddByteTag (uint64_t animUid, Ptr<const Packet> p);
  /**
   * WriteN function
   * \param data the data t write
   * \param count the number of bytes to write
   * \param f the file to write to
   * \returns the number of bytes written
   */
  int WriteN (const char* data, uint32_t count, FILE * f);
  /**
   * WriteN function
   * \param st the string to output
   * \param f the file to write to
   * \returns the number of bytes written
   */
  int WriteN (const std::string& st, FILE * f);
  /**
   * Get MAC address function
   * \param nd the device
   * \returns the MAC address
   */
  std::string GetMacAddress (Ptr <NetDevice> nd);
  /**
   * Get IPv4 address
   * \param nd the device
   * \returns the IPv4 address
   */
  std::string GetIpv4Address (Ptr <NetDevice> nd);
  /**
   * Get IPv6 address
   * \param nd the device
   * \returns the IPv6 address
   */
  std::string GetIpv6Address (Ptr <NetDevice> nd);
  /**
   * Get IPv4 addresses
   * \param nd the device
   * \returns the IPv4 address list
   */
  std::vector<std::string> GetIpv4Addresses (Ptr <NetDevice> nd);
  /**
   * Get IPv6 addresses
   * \param nd the device
   * \returns the IPv6 address list
   */
  std::vector<std::string> GetIpv6Addresses (Ptr <NetDevice> nd);

  /**
   * Get netanim version function
   * \returns the net anim version string
   */
  std::string GetNetAnimVersion ();
  /// Mobility auto check function
  void MobilityAutoCheck ();
  /**
   * Is packet pending function
   * \param animUid the UID
   * \param protocolType the protocol type
   * \returns true if a packet is pending
   */
  bool IsPacketPending (uint64_t animUid, ProtocolType protocolType);
  /**
   * Purge pending packets function
   * \param protocolType the protocol type
   */
  void PurgePendingPackets (ProtocolType protocolType);
  /**
   * Protocol type to pending packets function
   * \param protocolType the protocol type
   * \returns AnimUidPacketInfoMap *
   */
  AnimUidPacketInfoMap * ProtocolTypeToPendingPackets (ProtocolType protocolType);
  /**
   * Protocol type to string function
   * \param protocolType the protocol type
   * \returns the protocol type string
   */
  std::string ProtocolTypeToString (ProtocolType protocolType);
  /**
   * Add pending packet function
   * \param protocolType the protocol type
   * \param animUid the UID
   * \param pktInfo the packet info
   */
  void AddPendingPacket (ProtocolType protocolType, uint64_t animUid, AnimPacketInfo pktInfo);
  /**
   * Get anim UID from packet function
   * \param p the packet
   * \returns the UID
   */
  uint64_t GetAnimUidFromPacket (Ptr <const Packet>);
  /**
   * Add to IPv4 address node ID table function
   * \param ipv4Address the IPv4 address
   * \param nodeId the node ID
   */
  void AddToIpv4AddressNodeIdTable (std::string ipv4Address, uint32_t nodeId);
  /**
   * Add to IPv4 address node ID table function
   * \param ipv4Addresses the list of IPv4 addresses
   * \param nodeId the node ID
   */
  void AddToIpv4AddressNodeIdTable (std::vector<std::string> ipv4Addresses, uint32_t nodeId);
  /**
   * Add to IPv6 address node ID table function
   * \param ipv6Address the IPv6 address
   * \param nodeId the node ID
   */
  void AddToIpv6AddressNodeIdTable (std::string ipv6Address, uint32_t nodeId);
  /**
   * Add to IPv6 address node ID table function
   * \param ipv6Addresses the list of IPv6 addresses
   * \param nodeId the node ID
   */
  void AddToIpv6AddressNodeIdTable (std::vector<std::string> ipv6Addresses, uint32_t nodeId);
  /**
   * Is in time window function
   * \returns true if in the time window
   */
  bool IsInTimeWindow ();
  /// Check maximum packets per trace file function
  void CheckMaxPktsPerTraceFile ();

  /// Track wifi phy counters function
  void TrackWifiPhyCounters ();
  /// Track wifi MAC counters function
  void TrackWifiMacCounters ();
  /// Track IPv4 L3 protocol counters function
  void TrackIpv4L3ProtocolCounters ();
  /// Track queue counters function
  void TrackQueueCounters ();
  // ##### Routing #####
  /// Track IPv4 router function
  void TrackIpv4Route ();
  /// Track IPv4 route paths function
  void TrackIpv4RoutePaths ();
  /**
   * Get IPv4 routing table function
   * \param n the node
   * \returns the IPv4 routing table
   */
  std::string GetIpv4RoutingTable (Ptr <Node> n);
  /**
   * Recursive IPv4 route path search function
   * \param from the source node
   * \param to the destination node
   * \param rpElements the IPv4 routing path elements
   */
  void RecursiveIpv4RoutePathSearch (std::string from, std::string to, Ipv4RoutePathElements &rpElements);
  /**
   * Write route path function
   * \param nodeId the node ID
   * \param destination the destination
   * \param rpElements the IPv4 routing path elements
   */
  void WriteRoutePath (uint32_t nodeId, std::string destination, Ipv4RoutePathElements rpElements);


  // ##### Trace #####
  /**
   * Enqueue trace function
   * \param context the context
   * \param p the packet
   */
  void EnqueueTrace (std::string context, Ptr<const Packet>);
  /**
   * Dequeue trace function
   * \param context the context
   * \param p the packet
   */
  void DequeueTrace (std::string context, Ptr<const Packet>);
  /**
   * Queue trace function
   * \param context the context
   * \param p the packet
   */
  void QueueDropTrace (std::string context, Ptr<const Packet>);
  /**
   * IPv4 transmit trace function
   * \param context the context
   * \param p the packet
   * \param ipv4 the IP
   * \param interfaceIndex the interface index
   */
  void Ipv4TxTrace (std::string context,
                    Ptr<const Packet> p, Ptr<Ipv4> ipv4,
                    uint32_t interfaceIndex);
  /**
   * IPv4 receive trace function
   * \param context the context
   * \param p the packet
   * \param ipv4 the IP
   * \param interfaceIndex the interface index
   */
  void Ipv4RxTrace (std::string context,
                    Ptr<const Packet> p, Ptr<Ipv4> ipv4,
                    uint32_t interfaceIndex);
  /**
   * IPv4 drop trace function
   * \param context the context
   * \param ipv4Header the IPv4 header
   * \param p the packet
   * \param dropReason the reason for the drop
   * \param ipv4 the IP
   * \param interfaceIndex the interface index
   */
  void Ipv4DropTrace (std::string context,
                      const Ipv4Header & ipv4Header,
                      Ptr<const Packet> p,
                      Ipv4L3Protocol::DropReason dropReason, Ptr<Ipv4> ipv4,
                      uint32_t interfaceIndex);

  /**
   * wifi MAC transmit trace function
   * \param context the context
   * \param p the packet
   */
  void WifiMacTxTrace (std::string context, Ptr<const Packet> p);
  /**
   * wifi MAC transmit drop trace function
   * \param context the context
   * \param p the packet
   */
  void WifiMacTxDropTrace (std::string context, Ptr<const Packet> p);
  /**
   * wifi MAC receive trace function
   * \param context the context
   * \param p the packet
   */
  void WifiMacRxTrace (std::string context, Ptr<const Packet> p);
  /**
   * wifi MAC receive drop trace function
   * \param context the context
   * \param p the packet
   */
  void WifiMacRxDropTrace (std::string context, Ptr<const Packet> p);
  /**
   * wifi Phy transmit drop trace function
   * \param context the context
   * \param p the packet
   */
  void WifiPhyTxDropTrace (std::string context, Ptr<const Packet> p);
  /**
   * wifi Phy receive drop trace function
   * \param context the context
   * \param p the packet
   * \param reason the reason
   */
  void WifiPhyRxDropTrace (std::string context, Ptr<const Packet> p, WifiPhyRxfailureReason reason);
  /**
   * LR-WPAN MAC transmit trace function
   * \param context the context
   * \param p the packet
   */
  void LrWpanMacTxTrace (std::string context,
                         Ptr<const Packet> p);
  /**
   * LR-WPAN MAC transmit drop trace function
   * \param context the context
   * \param p the packet
   */
  void LrWpanMacTxDropTrace (std::string context,
                             Ptr<const Packet> p);
  /**
   * LR-WPAN MAC receive trace function
   * \param context the context
   * \param p the packet
   */
  void LrWpanMacRxTrace (std::string context,
                         Ptr<const Packet> p);
  /**
   * LR-WPAN MAC receive drop trace function
   * \param context the context
   * \param p the packet
   */
  void LrWpanMacRxDropTrace (std::string context,
                             Ptr<const Packet> p);
  /**
   * Device transmit trace function
   * \param context the context
   * \param p the packet
   * \param tx the transmit device
   * \param rx the receive device
   * \param txTime the transmit time
   * \param rxTime the reeive time
   */
  void DevTxTrace (std::string context,
                   Ptr<const Packet> p,
                   Ptr<NetDevice> tx,
                   Ptr<NetDevice> rx,
                   Time txTime,
                   Time rxTime);
  /**
   * wifi Phy transmit PSDU begin trace function
   * \param context the context
   * \param psduMap the PSDU map
   * \param txVector the TXVECTOR
   * \param txPowerW the tx power in Watts
   */
  void WifiPhyTxBeginTrace (std::string context, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);
  /**
   * wifi Phy receive begin trace function
   *
   * \param context the context
   * \param p the packet
   * \param rxPowersW the receive power per channel band in Watts
   */
  void WifiPhyRxBeginTrace (std::string context, Ptr<const Packet> p, RxPowerWattPerChannelBand rxPowersW);
  /**
   * WAVE Phy transmit begin trace function
   * \param context the context
   * \param p the packet
   */
  void WavePhyTxBeginTrace (std::string context,
                            Ptr<const Packet> p);
  /**
   * WAVE Phy receive begin trace function
   *
   * \param context the context
   * \param p the packet
   */
  void WavePhyRxBeginTrace (std::string context,
                            Ptr<const Packet> p);
  /**
   * LR-WPAN Phy receive begin trace function
   *
   * \param context the context
   * \param p the packet
   */
  void LrWpanPhyTxBeginTrace (std::string context,
                              Ptr<const Packet> p);
  /**
   * LR-WPAN Phy receive begin trace function
   *
   * \param context the context
   * \param p the packet
   */
  void LrWpanPhyRxBeginTrace (std::string context,
                              Ptr<const Packet> p);
  /**
   * WIMax transmit trace function
   * \param context the context
   * \param p the packet
   * \param m the MAC address
   */
  void WimaxTxTrace (std::string context,
                     Ptr<const Packet> p,
                     const Mac48Address & m);
  /**
   * WIMax receive trace function
   * \param context the context
   * \param p the packet
   * \param m the MAC address
   */
  void WimaxRxTrace (std::string context,
                     Ptr<const Packet> p,
                     const Mac48Address & m);
  /**
   * CSMA Phy transmit begin trace function
   * \param context the context
   * \param p the packet
   */
  void CsmaPhyTxBeginTrace (std::string context, Ptr<const Packet> p);
  /**
   * CSMA Phy transmit end trace function
   *
   * \param context the context
   * \param p the packet
   */
  void CsmaPhyTxEndTrace (std::string context, Ptr<const Packet> p);
  /**
   * CSMA Phy receive end trace function
   *
   * \param context the context
   * \param p the packet
   */
  void CsmaPhyRxEndTrace (std::string context, Ptr<const Packet> p);
  /**
   * CSMA MAC receive trace function
   *
   * \param context the context
   * \param p the packet
   */
  void CsmaMacRxTrace (std::string context, Ptr<const Packet> p);
  /**
   * LTE transmit trace function
   * \param context the context
   * \param p the packet
   * \param m the MAC address
   */
  void LteTxTrace (std::string context,
                   Ptr<const Packet> p,
                   const Mac48Address & m);
  /**
   * LTE receive trace function
   * \param context the context
   * \param p the packet
   * \param m the MAC address
   */
  void LteRxTrace (std::string context,
                   Ptr<const Packet> p,
                   const Mac48Address & m);
  /**
   * LTE Spectrum Phy transmit start function
   * \param context the context
   * \param pb the packet burst
   */
  void LteSpectrumPhyTxStart (std::string context, Ptr<const PacketBurst> pb);
  /**
   * LTE Spectrum Phy receive start function
   * \param context the context
   * \param pb the packet burst
   */
  void LteSpectrumPhyRxStart (std::string context, Ptr<const PacketBurst> pb);
  /**
   * UAN Phy gen transmit trace function
   * \param context the context
   * \param p the packet
   */
  void UanPhyGenTxTrace (std::string context, Ptr<const Packet>);
  /**
   * UAN Phy gen receive trace function
   * \param context the context
   * \param p the packet
   */
  void UanPhyGenRxTrace (std::string context, Ptr<const Packet>);
  /**
   * Remaining energy trace function
   * \param context the context
   * \param previousEnergy The previous energy
   * \param currentEnergy The current energy
   */
  void RemainingEnergyTrace (std::string context, double previousEnergy, double currentEnergy);
  /**
   * Generic wireless transmit trace function
   * \param context the context
   * \param p the packet
   * \param protocolType the protocol type
   */
  void GenericWirelessTxTrace (std::string context, Ptr<const Packet> p, ProtocolType protocolType);
  /**
   * Generic wireless receive trace function
   * \param context the context
   * \param p the packet
   * \param protocolType the protocol type
   */
  void GenericWirelessRxTrace (std::string context, Ptr<const Packet> p, ProtocolType protocolType);


  /// Connect callbacks function
  void ConnectCallbacks ();
  /// Connect LTE function
  void ConnectLte ();
  /**
   * Connect LTE ue function
   * \param n the node
   * \param nd the device
   * \param devIndex the device index
   */
  void ConnectLteUe (Ptr <Node> n, Ptr <LteUeNetDevice> nd, uint32_t devIndex);
  /**
   * Connect LTE ENB function
   * \param n the node
   * \param nd the device
   * \param devIndex the device index
   */
  void ConnectLteEnb (Ptr <Node> n, Ptr <LteEnbNetDevice> nd, uint32_t devIndex);


  // ##### Mobility #####
  /**
   * Get position function
   * \param n the node
   * \returns the position vector
   */
  Vector GetPosition (Ptr <Node> n);
  /**
   * Update position function
   * \param n the node
   * \returns the position vector
   */
  Vector UpdatePosition (Ptr <Node> n);
  /**
   * Update position function
   * \param n the node
   * \param v the vector
   * \returns the position vector
   */
  Vector UpdatePosition (Ptr <Node> n, Vector v);
  /**
   * Update position function
   * \param ndev the device
   * \returns the position vector
   */
  Vector UpdatePosition (Ptr <NetDevice> ndev);
  /**
   * Node has moved function
   * \param n the node
   * \param newLocation the new location vector
   * \returns true if the node has moved
   */
  bool NodeHasMoved (Ptr <Node> n, Vector newLocation);
  /**
   * Get moved nodes function
   * \returns the list of moved nodes
   */
  std::vector < Ptr <Node> > GetMovedNodes ();
  /**
   * Mobility course change trace function
   * \param mob the mobility model
   */
  void MobilityCourseChangeTrace (Ptr <const MobilityModel> mob);


  // ##### XML Helpers #####

  /**
   * Write non P2P link properties function
   * \param id the ID
   * \param ipv4Address the IP address
   * \param channelType the channel type
   */
  void WriteNonP2pLinkProperties (uint32_t id, std::string ipv4Address, std::string channelType);
  /**
   * Write node update function
   * \param nodeId the node ID
   */
  void WriteNodeUpdate (uint32_t nodeId);
  /**
   * Output wireless packet transmit info
   * \param p the packet
   * \param pktInfo the packet info
   * \param animUid the UID
   */
  void OutputWirelessPacketTxInfo (Ptr<const Packet> p, AnimPacketInfo& pktInfo, uint64_t animUid);
  /**
   * Output wireless packet receive info
   * \param p the packet
   * \param pktInfo the packet info
   * \param animUid the UID
   */
  void OutputWirelessPacketRxInfo (Ptr<const Packet> p, AnimPacketInfo& pktInfo, uint64_t animUid);
  /**
   * Output CSMA packet function
   * \param p the packet
   * \param pktInfo the packet info
   */
  void OutputCsmaPacket (Ptr<const Packet> p, AnimPacketInfo& pktInfo);
  /// Write link properties function
  void WriteLinkProperties ();
  /// Write IPv4 Addresses function
  void WriteIpv4Addresses ();
  /// Write IPv6 Addresses function
  void WriteIpv6Addresses ();
  /// Write nodes function
  void WriteNodes ();
  /// Write node colors function
  void WriteNodeColors ();
  /// Write node sizes function
  void WriteNodeSizes ();
  /// Write node energies function
  void WriteNodeEnergies ();
  /**
   * Write XML anim function
   * \param routing the routing
   */
  void WriteXmlAnim (bool routing = false);
  /**
   * Write XML update node position function
   * \param nodeId the node ID
   * \param x the X position
   * \param y the Y position
   */
  void WriteXmlUpdateNodePosition (uint32_t nodeId, double x, double y);
  /**
   * Write XML update node color function
   * \param nodeId the node ID
   * \param r the red color
   * \param g the green color
   * \param b the blue color
   */
  void WriteXmlUpdateNodeColor (uint32_t nodeId, uint8_t r, uint8_t g, uint8_t b);
  /**
   * Write XML update node description function
   * \param nodeId the node ID
   */
  void WriteXmlUpdateNodeDescription (uint32_t nodeId);
  /**
   * Write XML update node size function
   * \param nodeId the node ID
   * \param width the width
   * \param height the height
   */
  void WriteXmlUpdateNodeSize (uint32_t nodeId, double width, double height);
  /**
   * Write XML add resource function
   * \param resourceId the resource ID
   * \param resourcePath the resource path
   */
  void WriteXmlAddResource (uint32_t resourceId, std::string resourcePath);
  /**
   * Write XML add node counter function
   * \param counterId the counter ID
   * \param counterName the counter name
   * \param counterType the counter type
   */
  void WriteXmlAddNodeCounter (uint32_t counterId, std::string counterName, CounterType counterType);
  /**
   * Write XML update node image function
   * \param nodeId the node ID
   * \param resourceId the resource ID
   */
  void WriteXmlUpdateNodeImage (uint32_t nodeId, uint32_t resourceId);
  /**
   * Write XML update node counter function
   * \param counterId the counter ID
   * \param nodeId the node ID
   * \param value the node counter value
   */
  void WriteXmlUpdateNodeCounter (uint32_t counterId, uint32_t nodeId, double value);
  /**
   * Write XML node function
   * \param id the ID
   * \param sysId the system ID
   * \param locX the x location
   * \param locY the y location
   */
  void WriteXmlNode (uint32_t id, uint32_t sysId, double locX, double locY);
  /**
   * Write XML link counter function
   * \param fromId the from device
   * \param toLp the to device
   * \param toId the to ID
   */
  void WriteXmlLink (uint32_t fromId, uint32_t toLp, uint32_t toId);
  /**
   * Write XML update link counter function
   * \param fromId the from device
   * \param toId the to device
   * \param linkDescription the link description
   */
  void WriteXmlUpdateLink (uint32_t fromId, uint32_t toId, std::string linkDescription);
  /**
   * Write XMLP function
   * \param pktType the packet type
   * \param fId the FID
   * \param fbTx the FB transmit
   * \param lbTx the LB transmit
   * \param tId the TID
   * \param fbRx the FB receive
   * \param lbRx the LB receive
   * \param metaInfo the meta info
   */
  void WriteXmlP (std::string pktType,
                  uint32_t fId,
                  double fbTx,
                  double lbTx,
                  uint32_t tId,
                  double fbRx,
                  double lbRx,
                  std::string metaInfo = "");
  /**
   * Write XMLP function
   * \param animUid the UID
   * \param pktType the packet type
   * \param fId the FID
   * \param fbTx the FB transmit
   * \param lbTx the LB transmit
   */
  void WriteXmlP (uint64_t animUid, std::string pktType, uint32_t fId, double fbTx, double lbTx);
  /**
   * Write XMLP Ref function
   * \param animUid the UID
   * \param fId the FID
   * \param fbTx the FB transmit
   * \param metaInfo the meta info
   */
  void WriteXmlPRef (uint64_t animUid, uint32_t fId, double fbTx, std::string metaInfo = "");
  /**
   * Write XML close function
   * \param name the name
   * \param routing true if routing
   */
  void WriteXmlClose (std::string name, bool routing = false);
  /**
   * Write XML non P2P link properties function
   * \param id the ID
   * \param ipAddress the IP address
   * \param channelType the channel type
   */
  void WriteXmlNonP2pLinkProperties (uint32_t id, std::string ipAddress, std::string channelType);
  /**
   * Write XML routing function
   * \param id the ID
   * \param routingInfo the routing info
   */
  void WriteXmlRouting (uint32_t id, std::string routingInfo);
  /**
   * Write XMLRP function
   * \param nodeId the node ID
   * \param destination the destination
   * \param rpElements the route path elements
   */
  void WriteXmlRp (uint32_t nodeId, std::string destination, Ipv4RoutePathElements rpElements);
  /**
   * Write XML update background function
   * \param fileName the file name
   * \param x the X value
   * \param y the Y value
   * \param scaleX the X scale
   * \param scaleY the Y scale
   * \param opacity the opacity
   */
  void WriteXmlUpdateBackground (std::string fileName, double x, double y, double scaleX, double scaleY, double opacity);
  /**
   * Write XML Ipv4 addresses function
   * \param nodeId the node ID
   * \param ipv4Addresses the list of Ipv4 addresses
   */
  void WriteXmlIpv4Addresses (uint32_t nodeId, std::vector<std::string> ipv4Addresses);
  /**
   * Write XML Ipv6 addresses function
   * \param nodeId the node ID
   * \param ipv6Addresses the list of Ipv6 addresses
   */
  void WriteXmlIpv6Addresses (uint32_t nodeId, std::vector<std::string> ipv6Addresses);

};




/**
 * \ingroup netanim
 *
 * \brief Byte tag using by Anim to uniquely identify packets
 *
 * When Anim receives a Tx Notification we tag the packet with a unique global uint64_t identifier
 * before recording Tx information
 * When Anim receives Rx notifications the tag is used to retrieve Tx information recorded earlier
 *
 */

class AnimByteTag : public Tag
{
public:
  /**
   * \brief Get Type Id
   * \returns Type Id
   *
   */
  static TypeId GetTypeId (void);

  /**
   * \brief Get Instance Type Id
   * \returns Type Id
   *
   */
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief Get Serialized Size
   * \returns Serialized Size (i.e size of uint64_t)
   *
   */
  virtual uint32_t GetSerializedSize (void) const;

  /**
   * \brief Serialize function
   * \param i Tag Buffer
   *
   */
  virtual void Serialize (TagBuffer i) const;

  /**
   * \brief Deserialize function
   * \param i Tag Buffer
   *
   */
  virtual void Deserialize (TagBuffer i);

  /**
   * \brief Print tag info
   * \param os Reference of ostream object
   *
   */
  virtual void Print (std::ostream &os) const;

  /**
   * \brief Set global Uid in tag
   * \param AnimUid global Uid
   *
   */
  void Set (uint64_t AnimUid);

  /**
   * \brief Get Uid in tag
   * \returns Uid in tag
   *
   */
  uint64_t Get (void) const;

private:
  uint64_t m_AnimUid; ///< the UID
};



}
#endif

