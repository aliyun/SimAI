/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: George F. Riley <riley@ece.gatech.edu>
// Author: Lalith Suresh <suresh.lalith@gmail.com>
//

#ifndef IPV4_L3_CLICK_PROTOCOL_H
#define IPV4_L3_CLICK_PROTOCOL_H

#include "ns3/ipv4.h"
#include "ns3/net-device.h"
#include "ns3/packet.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/traced-callback.h"
#include "ns3/ipv4-interface.h"
#include "ns3/log.h"

namespace ns3 {

class Packet;
class NetDevice;
class Ipv4Interface;
class Ipv4Address;
class Ipv4Header;
class Ipv4RoutingTableEntry;
class Ipv4Route;
class Node;
class Socket;
class Ipv4RawSocketImpl;
class IpL4Protocol;
class Icmpv4L4Protocol;

/**
 * \brief Implement the Ipv4 layer specifically for Click nodes
 * to allow a clean integration of Click.
 * \ingroup click
 *
 * This is code is mostly repeated from the Ipv4L3Protocol implementation.
 * Changes include:
 *   - A stripped down version of Send().
 *   - A stripped down version of Receive().
 *   - A public version of LocalDeliver().
 *   - Modifications to AddInterface().
 */

class Ipv4L3ClickProtocol : public Ipv4
{
#ifdef NS3_CLICK
public:
  static TypeId GetTypeId (void);

  /**
   * Protocol number for Ipv4 L3 (0x0800).
   */
  static const uint16_t PROT_NUMBER;

  Ipv4L3ClickProtocol ();
  virtual ~Ipv4L3ClickProtocol ();

  virtual void Insert (Ptr<IpL4Protocol> protocol);
  virtual void Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual void Remove (Ptr<IpL4Protocol> protocol);
  virtual void Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex);

  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber) const;
  virtual Ptr<IpL4Protocol> GetProtocol (int protocolNumber, int32_t interfaceIndex) const;

  virtual Ipv4Address SourceAddressSelection (uint32_t interface, Ipv4Address dest);

  /**
   * \param ttl default ttl to use
   *
   * When we need to send an ipv4 packet, we use this default
   * ttl value.
   */
  void SetDefaultTtl (uint8_t ttl);

  /**
   * \param packet packet to send
   * \param source source address of packet
   * \param destination address of packet
   * \param protocol number of packet
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet
   * to Click
   */
  void Send (Ptr<Packet> packet, Ipv4Address source,
             Ipv4Address destination, uint8_t protocol, Ptr<Ipv4Route> route);

  /**
   * \param packet packet to send
   * \param ipHeader IP Header
   * \param route route entry
   *
   * Higher-level layers call this method to send a packet with IPv4 Header
   * (Intend to be used with IpHeaderInclude attribute.)
   */
  void SendWithHeader (Ptr<Packet> packet, Ipv4Header ipHeader, Ptr<Ipv4Route> route);

  /**
   * \param packet packet to send down the stack
   * \param ifid interface to be used for sending down packet
   *
   * Ipv4ClickRouting calls this method to send a packet further
   * down the stack
   */
  void SendDown (Ptr<Packet> packet, int ifid);

  /**
   * Lower layer calls this method to send a packet to Click
   * \param device network device
   * \param p the packet
   * \param protocol protocol value
   * \param from address of the correspondent
   * \param to address of the destination
   * \param packetType type of the packet
   */
  void Receive ( Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from,
                 const Address &to, NetDevice::PacketType packetType);

  /**
   * Ipv4ClickRouting calls this to locally deliver a packet
   * \param p the packet
   * \param ip The Ipv4Header of the packet
   * \param iif The interface on which the packet was received
   */
  void LocalDeliver (Ptr<const Packet> p, Ipv4Header const&ip, uint32_t iif);

  /**
   * Get a pointer to the i'th Ipv4Interface
   * \param i index of interface, pointer to which is to be returned
   * \returns Pointer to the i'th Ipv4Interface if any.
   */
  Ptr<Ipv4Interface> GetInterface (uint32_t i) const;

  /**
   * Adds an Ipv4Interface to the interfaces list
   * \param interface Pointer to the Ipv4Interface to be added
   * \returns Index of the device which was added
   */
  uint32_t AddIpv4Interface (Ptr<Ipv4Interface> interface);

  /**
   * Calls m_node = node and sets up Loopback if needed
   * \param node Pointer to the node
   */
  void SetNode (Ptr<Node> node);

  /**
   * Returns the Icmpv4L4Protocol for the node
   * \returns Icmpv4L4Protocol instance of the node
   */
  Ptr<Icmpv4L4Protocol> GetIcmp (void) const;

  /**
   * Sets up a Loopback device
   */
  void SetupLoopback (void);

  /**
   * Creates a raw-socket
   * \returns Pointer to the created socket
   */
  Ptr<Socket> CreateRawSocket (void);

  /**
   * Deletes a particular raw socket
   * \param socket Pointer of socket to be deleted
   */
  void DeleteRawSocket (Ptr<Socket> socket);

  // functions defined in base class Ipv4
  void SetRoutingProtocol (Ptr<Ipv4RoutingProtocol> routingProtocol);
  Ptr<Ipv4RoutingProtocol> GetRoutingProtocol (void) const;

  Ptr<NetDevice> GetNetDevice (uint32_t i);

  uint32_t AddInterface (Ptr<NetDevice> device);
  uint32_t GetNInterfaces (void) const;

  int32_t GetInterfaceForAddress (Ipv4Address addr) const;
  int32_t GetInterfaceForPrefix (Ipv4Address addr, Ipv4Mask mask) const;
  int32_t GetInterfaceForDevice (Ptr<const NetDevice> device) const;
  bool IsDestinationAddress (Ipv4Address address, uint32_t iif) const;

  bool AddAddress (uint32_t i, Ipv4InterfaceAddress address);
  Ipv4InterfaceAddress GetAddress (uint32_t interfaceIndex, uint32_t addressIndex) const;
  uint32_t GetNAddresses (uint32_t interface) const;
  bool RemoveAddress (uint32_t interfaceIndex, uint32_t addressIndex);
  bool RemoveAddress (uint32_t interfaceIndex, Ipv4Address address);
  Ipv4Address SelectSourceAddress (Ptr<const NetDevice> device,
                                   Ipv4Address dst, Ipv4InterfaceAddress::InterfaceAddressScope_e scope);

  void SetMetric (uint32_t i, uint16_t metric);
  uint16_t GetMetric (uint32_t i) const;
  uint16_t GetMtu (uint32_t i) const;
  bool IsUp (uint32_t i) const;
  void SetUp (uint32_t i);
  void SetDown (uint32_t i);
  bool IsForwarding (uint32_t i) const;
  void SetForwarding (uint32_t i, bool val);
  void SetPromisc (uint32_t i);
protected:
  virtual void DoDispose (void);
  /**
   * This function will notify other components connected to the node that a new stack member is now connected
   * This will be used to notify Layer 3 protocol of layer 4 protocol stack to connect them together.
   */
  virtual void NotifyNewAggregate ();

private:
  Ipv4Header BuildHeader (
    Ipv4Address source,
    Ipv4Address destination,
    uint8_t protocol,
    uint16_t payloadSize,
    uint8_t ttl,
    bool mayFragment);

  virtual void SetIpForward (bool forward);
  virtual bool GetIpForward (void) const;
  virtual void SetWeakEsModel (bool model);
  virtual bool GetWeakEsModel (void) const;

  typedef std::vector<Ptr<Ipv4Interface> > Ipv4InterfaceList;
  /**
   * \brief Container of NetDevices registered to IPv4 and their interface indexes.
   */
  typedef std::map<Ptr<const NetDevice>, uint32_t > Ipv4InterfaceReverseContainer;
  typedef std::list<Ptr<Ipv4RawSocketImpl> > SocketList;
  /**
   * \brief Container of the IPv4 L4 keys: protocol number, interface index
   */
  typedef std::pair<int, int32_t> L4ListKey_t;
  /**
   * \brief Container of the IPv4 L4 instances.
   */
  typedef std::map<L4ListKey_t, Ptr<IpL4Protocol> > L4List_t;

  Ptr<Ipv4RoutingProtocol> m_routingProtocol;
  bool m_ipForward;
  bool m_weakEsModel;
  L4List_t m_protocols;
  Ipv4InterfaceList m_interfaces;
  Ipv4InterfaceReverseContainer m_reverseInterfacesContainer; //!< Container of NetDevice / Interface index associations.
  uint8_t m_defaultTtl;
  uint16_t m_identification;

  Ptr<Node> m_node;

  /** \todo Remove; this TracedCallback is never invoked. */
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_sendOutgoingTrace;
  /** \todo Remove: this TracedCallback is never invoked. */
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_unicastForwardTrace;
  /** \todo This TracedCallback is invoked but not accessible. */
  TracedCallback<const Ipv4Header &, Ptr<const Packet>, uint32_t> m_localDeliverTrace;

  SocketList m_sockets;

  std::vector<bool> m_promiscDeviceList;

#endif /* NS3_CLICK */
};

} // namespace ns3

#endif /* IPV4_L3_CLICK_ROUTING_H */
