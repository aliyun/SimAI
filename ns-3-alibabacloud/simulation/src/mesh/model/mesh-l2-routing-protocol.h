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
 *          Pavel Boyko <boyko@iitp.ru>
 */

#ifndef MESH_L2_ROUTING_PROTOCOL_H
#define MESH_L2_ROUTING_PROTOCOL_H

#include "ns3/object.h"
#include "ns3/mac48-address.h"
#include "ns3/packet.h"

namespace ns3 {

class Packet;
class MeshPointDevice;

/**
 * \ingroup mesh
 *
 * \brief Interface for L2 mesh routing protocol and mesh point communication.
 *
 * Every mesh routing protocol must implement this interface. Each mesh point (MeshPointDevice) is supposed
 * to know single L2RoutingProtocol to work with, see MeshPointDevice::SetRoutingProtocol ().
 *
 * This interface is similar to ipv4 routing protocol base class.
 */
class MeshL2RoutingProtocol : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  /// virtual D-tor for subclasses
  virtual ~MeshL2RoutingProtocol ();
  /**
   * Callback to be invoked when route discovery  procedure is completed.
   *
   * \param flag        indicating whether a route was actually found and all needed information is
   *                    added to the packet successfully
   *
   * \param packet      for which the route was resolved. All routing information for MAC layer
   *                    must be stored in proper tags (like in case of HWMP, when WifiMacHeader
   *                    needs address of next hop), or must be added as a packet header (if MAC
   *                    does not need any additional information). So, the packet is returned back
   *                    to MeshPointDevice looks like a pure packet with ethernet header
   *                    (i.e data + src +dst + protocol). The only special information addressed
   *                    to MeshPointDevice is an outcoming interface ID.
   *
   * \param src         source address of the packet
   *
   * \param dst         destination address of the packet
   *
   * \param protocol    ethernet 'Protocol' field, needed to form a proper MAC-layer header
   *
   * \param uint32_t    outcoming interface to use or 0xffffffff if packet should be sent by ALL interfaces
   */
  typedef Callback<void, /* return type */
                   bool, /* flag */
                   Ptr<Packet>, /* packet */
                   Mac48Address, /* src */
                   Mac48Address, /* dst */
                   uint16_t, /* protocol */
                   uint32_t /* out interface ID */
                   > RouteReplyCallback;
  /**
   * Request routing information, all packets must go through this request.
   *
   * Note that route discovery works async. -- RequestRoute returns immediately, while
   * reply callback will be called when routing information will be available.
   * \return true if valid route is already known
   * \param sourceIface the incoming interface of the packet
   * \param source        source address
   * \param destination   destination address
   * \param packet        the packet to be resolved (needed the whole packet, because
   *                      routing information is added as tags or headers). The packet
   *                      will be returned to reply callback.
   * \param protocolType  protocol ID, needed to form a proper MAC-layer header
   * \param routeReply    callback to be invoked after route discovery procedure, supposed
   *                      to really send packet using routing information.
   */
  virtual bool RequestRoute (uint32_t sourceIface, const Mac48Address source, const Mac48Address destination, Ptr<
                               const Packet> packet, uint16_t protocolType, RouteReplyCallback routeReply) = 0;
  /**
   * \brief When packet is ready to go to upper layer, protocol must
   * remove all its information: tags, header, etc. So,
   * MeshPointDevice must call this method when passing a packet to
   * upper layer.
   * \returns true if packet shall not be dropped, false otherwise.
   * \param fromIface     the incoming interface of the packet
   * \param source        source address
   * \param destination   destination address
   * \param packet        the packet to be handled
   * \param protocolType  protocol ID, needed to form a proper MAC-layer header
   * \attention protocol type is passed by reference, because may be
   * changed
   */
  virtual bool RemoveRoutingStuff (uint32_t fromIface, const Mac48Address source, const Mac48Address destination, Ptr<
                                     Packet> packet, uint16_t & protocolType) = 0;
  /**
   * Set host mesh point, analog of SetNode (...) methods for upper layer protocols.
   *
   * \param mp the mesh point device
   */
  void
  SetMeshPoint (Ptr<MeshPointDevice> mp);
  /**
   * Each mesh protocol must be installed on the mesh point to work.
   *
   * \returns the mesh point device
   */
  Ptr<MeshPointDevice>
  GetMeshPoint () const;
protected:
  /// Host mesh point
  Ptr<MeshPointDevice> m_mp;
};
} // namespace ns3
#endif
