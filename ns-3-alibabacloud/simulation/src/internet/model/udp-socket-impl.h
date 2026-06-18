/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef UDP_SOCKET_IMPL_H
#define UDP_SOCKET_IMPL_H

#include <stdint.h>
#include <queue>
#include "ns3/callback.h"
#include "ns3/traced-callback.h"
#include "ns3/socket.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/udp-socket.h"
#include "ns3/ipv4-interface.h"
#include "icmpv4.h"

namespace ns3 {

class Ipv4EndPoint;
class Ipv6EndPoint;
class Node;
class Packet;
class UdpL4Protocol;
class Ipv6Header;
class Ipv6Interface;

/**
 * \ingroup socket
 * \ingroup udp
 *
 * \brief A sockets interface to UDP
 * 
 * This class subclasses ns3::UdpSocket, and provides a socket interface
 * to ns3's implementation of UDP.
 *
 * For IPv4 packets, the TOS is set according to the following rules:
 * - if the socket is connected, the TOS set for the socket is used
 * - if the socket is not connected, the TOS specified in the destination address
 *   passed to SendTo is used, while the TOS set for the socket is ignored
 * In both cases, a SocketIpTos tag is only added to the packet if the resulting
 * TOS is non-null. The Bind and Connect operations set the TOS for the
 * socket to the value specified in the provided address.
 * If the TOS determined for a packet (as described above) is not null, the
 * packet is assigned a priority based on that TOS value (according to the
 * Socket::IpTos2Priority function). Otherwise, the priority set for the
 * socket is assigned to the packet. Setting a TOS for a socket also sets a
 * priority for the socket (according to the Socket::IpTos2Priority function).
 * A SocketPriority tag is only added to the packet if the resulting priority
 * is non-null.
 */

class UdpSocketImpl : public UdpSocket
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * Create an unbound udp socket.
   */
  UdpSocketImpl ();
  virtual ~UdpSocketImpl ();

  /**
   * \brief Set the associated node.
   * \param node the node
   */
  void SetNode (Ptr<Node> node);
  /**
   * \brief Set the associated UDP L4 protocol.
   * \param udp the UDP L4 protocol
   */
  void SetUdp (Ptr<UdpL4Protocol> udp);

  virtual enum SocketErrno GetErrno (void) const;
  virtual enum SocketType GetSocketType (void) const;
  virtual Ptr<Node> GetNode (void) const;
  virtual int Bind (void);
  virtual int Bind6 (void);
  virtual int Bind (const Address &address);
  virtual int Close (void);
  virtual int ShutdownSend (void);
  virtual int ShutdownRecv (void);
  virtual int Connect (const Address &address);
  virtual int Listen (void);
  virtual uint32_t GetTxAvailable (void) const;
  virtual int Send (Ptr<Packet> p, uint32_t flags);
  virtual int SendTo (Ptr<Packet> p, uint32_t flags, const Address &address);
  virtual uint32_t GetRxAvailable (void) const;
  virtual Ptr<Packet> Recv (uint32_t maxSize, uint32_t flags);
  virtual Ptr<Packet> RecvFrom (uint32_t maxSize, uint32_t flags,
                                Address &fromAddress);
  virtual int GetSockName (Address &address) const; 
  virtual int GetPeerName (Address &address) const;
  virtual int MulticastJoinGroup (uint32_t interfaceIndex, const Address &groupAddress);
  virtual int MulticastLeaveGroup (uint32_t interfaceIndex, const Address &groupAddress);
  virtual void BindToNetDevice (Ptr<NetDevice> netdevice);
  virtual bool SetAllowBroadcast (bool allowBroadcast);
  virtual bool GetAllowBroadcast () const;
  virtual void Ipv6JoinGroup (Ipv6Address address, Socket::Ipv6MulticastFilterMode filterMode, std::vector<Ipv6Address> sourceAddresses);

private:
  // Attributes set through UdpSocket base class 
  virtual void SetRcvBufSize (uint32_t size);
  virtual uint32_t GetRcvBufSize (void) const;
  virtual void SetIpMulticastTtl (uint8_t ipTtl);
  virtual uint8_t GetIpMulticastTtl (void) const;
  virtual void SetIpMulticastIf (int32_t ipIf);
  virtual int32_t GetIpMulticastIf (void) const;
  virtual void SetIpMulticastLoop (bool loop);
  virtual bool GetIpMulticastLoop (void) const;
  virtual void SetMtuDiscover (bool discover);
  virtual bool GetMtuDiscover (void) const;


  /**
   * \brief UdpSocketFactory friend class.
   * \relates UdpSocketFactory
   */
  friend class UdpSocketFactory;
  // invoked by Udp class

  /**
   * Finish the binding process
   * \returns 0 on success, -1 on failure
   */
  int FinishBind (void);

  /**
   * \brief Called by the L3 protocol when it received a packet to pass on to TCP.
   *
   * \param packet the incoming packet
   * \param header the packet's IPv4 header
   * \param port the remote port
   * \param incomingInterface the incoming interface
   */
  void ForwardUp (Ptr<Packet> packet, Ipv4Header header, uint16_t port, Ptr<Ipv4Interface> incomingInterface);

  /**
   * \brief Called by the L3 protocol when it received a packet to pass on to TCP.
   *
   * \param packet the incoming packet
   * \param header the packet's IPv6 header
   * \param port the remote port
   * \param incomingInterface the incoming interface
   */
  void ForwardUp6 (Ptr<Packet> packet, Ipv6Header header, uint16_t port, Ptr<Ipv6Interface> incomingInterface);

  /**
   * \brief Kill this socket by zeroing its attributes (IPv4)
   *
   * This is a callback function configured to m_endpoint in
   * SetupCallback(), invoked when the endpoint is destroyed.
   */
  void Destroy (void);

  /**
   * \brief Kill this socket by zeroing its attributes (IPv6)
   *
   * This is a callback function configured to m_endpoint in
   * SetupCallback(), invoked when the endpoint is destroyed.
   */
  void Destroy6 (void);

  /**
   * \brief Deallocate m_endPoint and m_endPoint6
   */
  void DeallocateEndPoint (void);

  /**
   * \brief Send a packet
   * \param p packet
   * \returns 0 on success, -1 on failure
   */
  int DoSend (Ptr<Packet> p);
  /**
   * \brief Send a packet to a specific destination and port (IPv4)
   * \param p packet
   * \param daddr destination address
   * \param dport destination port
   * \param tos ToS
   * \returns 0 on success, -1 on failure
   */
  int DoSendTo (Ptr<Packet> p, Ipv4Address daddr, uint16_t dport, uint8_t tos);
  /**
   * \brief Send a packet to a specific destination and port (IPv6)
   * \param p packet
   * \param daddr destination address
   * \param dport destination port
   * \returns 0 on success, -1 on failure
   */
  int DoSendTo (Ptr<Packet> p, Ipv6Address daddr, uint16_t dport);

  /**
   * \brief Called by the L3 protocol when it received an ICMP packet to pass on to TCP.
   *
   * \param icmpSource the ICMP source address
   * \param icmpTtl the ICMP Time to Live
   * \param icmpType the ICMP Type
   * \param icmpCode the ICMP Code
   * \param icmpInfo the ICMP Info
   */
  void ForwardIcmp (Ipv4Address icmpSource, uint8_t icmpTtl, uint8_t icmpType, uint8_t icmpCode, uint32_t icmpInfo);

  /**
   * \brief Called by the L3 protocol when it received an ICMPv6 packet to pass on to TCP.
   *
   * \param icmpSource the ICMP source address
   * \param icmpTtl the ICMP Time to Live
   * \param icmpType the ICMP Type
   * \param icmpCode the ICMP Code
   * \param icmpInfo the ICMP Info
   */
  void ForwardIcmp6 (Ipv6Address icmpSource, uint8_t icmpTtl, uint8_t icmpType, uint8_t icmpCode, uint32_t icmpInfo);

  // Connections to other layers of TCP/IP
  Ipv4EndPoint*       m_endPoint;   //!< the IPv4 endpoint
  Ipv6EndPoint*       m_endPoint6;  //!< the IPv6 endpoint
  Ptr<Node>           m_node;       //!< the associated node
  Ptr<UdpL4Protocol> m_udp;         //!< the associated UDP L4 protocol
  Callback<void, Ipv4Address,uint8_t,uint8_t,uint8_t,uint32_t> m_icmpCallback;  //!< ICMP callback
  Callback<void, Ipv6Address,uint8_t,uint8_t,uint8_t,uint32_t> m_icmpCallback6; //!< ICMPv6 callback

  Address m_defaultAddress; //!< Default address
  uint16_t m_defaultPort;   //!< Default port
  TracedCallback<Ptr<const Packet> > m_dropTrace; //!< Trace for dropped packets

  mutable enum SocketErrno m_errno;           //!< Socket error code
  bool                     m_shutdownSend;    //!< Send no longer allowed
  bool                     m_shutdownRecv;    //!< Receive no longer allowed
  bool                     m_connected;       //!< Connection established
  bool                     m_allowBroadcast;  //!< Allow send broadcast packets

  std::queue<std::pair<Ptr<Packet>, Address> > m_deliveryQueue; //!< Queue for incoming packets
  uint32_t m_rxAvailable;                   //!< Number of available bytes to be received

  // Socket attributes
  uint32_t m_rcvBufSize;    //!< Receive buffer size
  uint8_t m_ipMulticastTtl; //!< Multicast TTL
  int32_t m_ipMulticastIf;  //!< Multicast Interface
  bool m_ipMulticastLoop;   //!< Allow multicast loop
  bool m_mtuDiscover;       //!< Allow MTU discovery
};

} // namespace ns3

#endif /* UDP_SOCKET_IMPL_H */
