/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef FLAME_PROTOCOL_H
#define FLAME_PROTOCOL_H


#include "ns3/mesh-l2-routing-protocol.h"
#include "ns3/nstime.h"
#include "ns3/tag.h"
#include <map>

/**
 * \ingroup mesh
 * \defgroup flame FLAME
 *
 * \brief Forwarding LAyer for MEshing protocol
 *
 * Simple L2.5 mesh routing protocol developed by
 * Herman Elfrink <herman.elfrink@ti-wmc.nl> and presented in
 * "Easy Wireless: broadband ad-hoc networking for emergency services"
 * by Maurits de Graaf et. al. at The Sixth Annual Mediterranean Ad Hoc
 * Networking WorkShop, Corfu, Greece, June 12-15, 2007
 *
 * see also Linux kernel mailing list discussion at
 * http://lkml.org/lkml/2006/5/23/82
 */
namespace ns3 {
namespace flame {
class FlameProtocolMac;
class FlameHeader;
class FlameRtable;
/**
 * \ingroup flame
 * \brief Transmitter and receiver addresses
 */
class FlameTag : public Tag
{
public:
  /// transmitter for incoming:
  Mac48Address  transmitter;
  /// Receiver of the packet:
  Mac48Address  receiver;

  /**
   * Constructor
   *
   * \param a receiver MAC address
   */
  FlameTag (Mac48Address a = Mac48Address ()) :
    receiver (a){}

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static  TypeId  GetTypeId ();
  // Inherited from Tag
  TypeId  GetInstanceTypeId () const;
  uint32_t GetSerializedSize () const;
  void  Serialize (TagBuffer i) const;
  void  Deserialize (TagBuffer i);
  void  Print (std::ostream &os) const;

};

/**
 * \ingroup flame
 * \brief FLAME routing protocol
 */
class FlameProtocol : public MeshL2RoutingProtocol
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  FlameProtocol ();
  ~FlameProtocol ();
  void DoDispose ();

  /**
   * Route request, inherited from MeshL2RoutingProtocol
   *
   * \param sourceIface the source interface
   * \param source the source address
   * \param destination the destination address
   * \param packet the packet to route
   * \param protocolType the protocol type
   * \param routeReply the route reply
   * \returns if route exists
   */
  bool RequestRoute (uint32_t sourceIface, const Mac48Address source, const Mac48Address destination,
                     Ptr<const Packet> packet, uint16_t protocolType, RouteReplyCallback routeReply);
  /**
   * Cleanup flame headers!
   *
   * \param fromIface the from interface
   * \param source the source address
   * \param destination the destination address
   * \param packet the packet
   * \param protocolType the protocol type
   * \returns if the route removed
   */
  bool RemoveRoutingStuff (uint32_t fromIface, const Mac48Address source,
                           const Mac48Address destination, Ptr<Packet> packet, uint16_t& protocolType);
  /**
   * \brief Install FLAME on given mesh point.
   * \param mp the MeshPointDevice
   * \returns true if successful
   *
   * Installing protocol causes installation of its interface MAC plugins.
   *
   * Also MP aggregates all installed protocols, FLAME protocol can be accessed
   * via MeshPointDevice::GetObject<flame::FlameProtocol>();
   */
  bool Install (Ptr<MeshPointDevice> mp);
  /**
   * Get address of this instance
   * \returns the MAC address
   */
  Mac48Address GetAddress ();
  /**
   * Statistics
   * \param os the output stream
   */
  void Report (std::ostream & os) const;
  /// Reset statistics function
  void ResetStats ();
private:
  /**
   * assignment operator
   *
   * \param flame the object to assign
   * \returns the assigned value
   */
  FlameProtocol& operator= (const FlameProtocol & flame);
  /// type conversion operator
  FlameProtocol (const FlameProtocol &);

  /// LLC protocol number reserved by flame
  static const uint16_t FLAME_PROTOCOL = 0x4040;
  /**
   * \brief Handles a packet: adds a routing information and drops packets by TTL or Seqno
   *
   * \param seqno
   * \param source
   * \param flameHdr
   * \param receiver
   * \param fromIface
   * \return true if packet shall be dropped
   */
  bool HandleDataFrame (uint16_t seqno, Mac48Address source, const FlameHeader flameHdr, Mac48Address receiver, uint32_t fromIface);
  /**
   * \name Information about MeshPointDeviceaddress, plugins
   * \{
   */
  typedef std::map<uint32_t, Ptr<FlameProtocolMac> > FlamePluginMap;
  FlamePluginMap m_interfaces; ///< interfaces
  Mac48Address m_address; ///< address
  //\}
  /**
   * \name Broadcast timers:
   * \{
   */
  Time m_broadcastInterval;
  Time m_lastBroadcast;
  //\}
  /// Max Cost value (or TTL, because cost is actually hopcount)
  uint8_t m_maxCost;
  /// Sequence number:
  uint16_t m_myLastSeqno;
  /// Routing table:
  Ptr<FlameRtable> m_rtable;
  /// Statistics structure
  struct Statistics
  {
    uint16_t txUnicast; ///< transmit unicast
    uint16_t txBroadcast; ///< transmit broadcast
    uint32_t txBytes; ///< transmit bytes
    uint16_t droppedTtl; ///< dropped TTL
    uint16_t totalDropped; ///< total dropped
    /**
     * Print function
     * \param os the output stream to print to
     */
    void Print (std::ostream & os) const;
    /// constructor
    Statistics ();
  };
  Statistics m_stats; ///< statistics

};
} // namespace flame
} // namespace ns3
#endif /* FLAME_PROTOCOL_H */
