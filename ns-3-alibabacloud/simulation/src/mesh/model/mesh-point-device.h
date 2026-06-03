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

#ifndef L2ROUTING_NET_DEVICE_H
#define L2ROUTING_NET_DEVICE_H

#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/mac48-address.h"
#include "ns3/bridge-channel.h"
#include "ns3/mesh-l2-routing-protocol.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

/**
 * \ingroup mesh
 *
 * \brief Virtual net device modeling mesh point.
 *
 * Mesh point is a virtual net device which is responsible for
 *   - Aggregating and coordinating 1..* real devices -- mesh interfaces, see MeshInterfaceDevice class.
 *   - Hosting all mesh-related level 2 protocols.
 *
 * One of hosted L2 protocols must implement L2RoutingProtocol interface and is used for packets forwarding.
 *
 * From the level 3 point of view MeshPointDevice is similar to BridgeNetDevice, but the packets,
 * which going through may be changed (because L2 protocols may require their own headers or tags).
 *
 * Attributes: \todo
 */
class MeshPointDevice : public NetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  /// C-tor create empty (without interfaces and protocols) mesh point
  MeshPointDevice ();
  /// D-tor
  virtual ~MeshPointDevice ();

  /// \name Interfaces
  ///@{
  /**
   * \brief Attach new interface to the station. Interface must support 48-bit MAC address and SendFrom method.
   * \param port the port used
   *
   * \attention Only MeshPointDevice can have IP address, but not individual interfaces.
   */
  void AddInterface (Ptr<NetDevice> port);
  /**
   * \return number of interfaces
   */
  uint32_t GetNInterfaces () const;
  /**
   * \return interface device by its index (aka ID)
   * \param id is interface id, 0 <= id < GetNInterfaces
   */
  Ptr<NetDevice> GetInterface (uint32_t id) const;
  /**
   * \return vector of interfaces
   */
  std::vector<Ptr<NetDevice> > GetInterfaces () const;
  ///@}


  /// \name Protocols
  ///@{
  /**
   * Register routing protocol to be used. Protocol must be already installed on this mesh point.
   *
   * \param protocol the routing protocol
   */
  void SetRoutingProtocol (Ptr<MeshL2RoutingProtocol> protocol);
  /**
   * Access current routing protocol
   *
   * \return the current routing protocol
   */
  Ptr<MeshL2RoutingProtocol> GetRoutingProtocol () const;
  ///@}

  // Inherited from NetDevice
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex () const;
  virtual Ptr<Channel> GetChannel () const;
  virtual Address GetAddress () const;
  virtual void SetAddress (Address a);
  virtual bool SetMtu (const uint16_t mtu);
  virtual uint16_t GetMtu () const;
  virtual bool IsLinkUp () const;
  virtual void AddLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast () const;
  virtual Address GetBroadcast () const;
  virtual bool IsMulticast () const;
  virtual Address GetMulticast (Ipv4Address multicastGroup) const;
  virtual bool IsPointToPoint () const;
  virtual bool IsBridge () const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode () const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp () const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom () const;
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual void DoDispose ();

  /// \name Statistics
  ///@{
  /**
   *  Print statistics counters
   *  \param os the output stream
   */
  void Report (std::ostream & os) const;
  /// Reset statistics counters
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
   * Return a (random) forwarding delay value from the random variable
   * ForwardingDelay attribute.
   * 
   * \return A Time value from the ForwardingDelay random variable
   */
  Time GetForwardingDelay () const;

private:
  /**
   * Receive packet from interface
   *
   * \param device the device to receive from
   * \param packet the received packet
   * \param protocol the protocol
   * \param source the source address
   * \param destination the destination address
   * \param packetType the packet type
   */
  void ReceiveFromDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                          Address const &source, Address const &destination, PacketType packetType);
  /**
   * Forward packet down to interfaces
   *
   * \param incomingPort the incoming port
   * \param packet the packet to forward
   * \param protocol the protocol
   * \param src the source MAC address
   * \param dst the destination MAC address
   */
  void Forward (Ptr<NetDevice> incomingPort, Ptr<const Packet> packet,
                uint16_t protocol, const Mac48Address src,
                const Mac48Address dst);
  /**
   * \brief Response callback for L2 routing protocol. This will be executed when routing information is ready.
   *
   * \param success     True is route found. 
   * \param packet      Packet to send
   * \param src         Source MAC address
   * \param dst         Destination MAC address
   * \param protocol    Protocol ID
   * \param iface       Interface to use (ID) for send (decided by routing protocol). All interfaces will be used if outIface = 0xffffffff
   * \todo diagnose routing errors
   */
  void
  DoSend (bool success, Ptr<Packet> packet, Mac48Address src, Mac48Address dst, uint16_t protocol,
          uint32_t iface);

private:
  /// Receive action
  NetDevice::ReceiveCallback   m_rxCallback;
  /// Promisc receive action
  NetDevice::PromiscReceiveCallback  m_promiscRxCallback;
  /// Mesh point MAC address, supposed to be the address of the first added interface
  Mac48Address m_address;
  /// Parent node
  Ptr<Node> m_node;
  /// List of interfaces
  std::vector< Ptr<NetDevice> > m_ifaces;
  /// If index
  uint32_t m_ifIndex;
  /// MTU in bytes
  uint16_t m_mtu;
  /// Virtual channel for upper layers
  Ptr<BridgeChannel> m_channel;
  /// Current routing protocol, used mainly by GetRoutingProtocol
  Ptr<MeshL2RoutingProtocol> m_routingProtocol;
  /// Random variable used for forwarding delay and jitter
  Ptr<RandomVariableStream> m_forwardingRandomVariable;

  /// statistics counters
  struct Statistics
  {
    uint32_t unicastData; ///< unicast data
    uint32_t unicastDataBytes; ///< unicast data bytes
    uint32_t broadcastData; ///< broadcast data
    uint32_t broadcastDataBytes; ///< broadcast data bytes

    /// constructor
    Statistics ();
  };
  // Counters
  Statistics m_rxStats; ///< receive statistics
  Statistics m_txStats; ///< transmit statistics
  Statistics m_fwdStats; ///< forward statistics
};
} // namespace ns3
#endif
