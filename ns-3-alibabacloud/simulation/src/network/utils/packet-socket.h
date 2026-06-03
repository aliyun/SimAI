/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Emmanuelle Laprise, INRIA
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
 * Authors: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>,
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef PACKET_SOCKET_H
#define PACKET_SOCKET_H

#include <stdint.h>
#include <queue>
#include "ns3/callback.h"
#include "ns3/traced-callback.h"
#include "ns3/ptr.h"
#include "ns3/socket.h"
#include "ns3/net-device.h"

namespace ns3 {

class Node;
class Packet;
class NetDevice;
class PacketSocketAddress;

/**
 * \ingroup socket
 *
 * \brief A PacketSocket is a link between an application and a net device.
 *
 * A PacketSocket can be used to connect an application to a net
 * device. The application provides the buffers of data, the socket
 * converts them to a raw packet and the net device then adds the
 * protocol specific headers and trailers. This socket type
 * is very similar to the linux and BSD "packet" sockets.
 *
 * Here is a summary of the semantics of this class:
 * - Bind: Bind uses only the protocol and device fields of the 
 *       PacketSocketAddress. If none are provided, Bind uses 
 *       zero for both, which means that the socket is bound
 *       to all protocols on all devices on the node.
 *
 * - Connect: uses only the protocol, device and "physical address" 
 *       field of the PacketSocketAddress. It is used to set the default
 *       destination address for outgoing packets.
 *
 * - Send: send the input packet to the underlying NetDevices
 *       with the default destination address. The socket must
 *       be bound and connected.
 *
 * - SendTo: uses the protocol, device, and "physical address" 
 *       fields of the PacketSocketAddress. The device value is 
 *       used to specialize the packet transmission to a single 
 *       device, the protocol value specifies the protocol of this
 *       packet only and the "physical address" field is used to override the 
 *       default destination address. The socket must be bound.
 *
 * - Recv: The address represents the address of the packer originator.
 *       The fields "physical address", device, and protocol are filled.
 *
 * - Accept: not allowed
 *
 * - Listen: returns -1 (OPNOTSUPP)
 *
 * Socket data that is read from this socket using the methods returning
 * an ns3::Packet object (i.e., Recv (), RecvMsg (), RecvFrom ()) will
 * return Packet objects with three PacketTag objects attached.
 * Applications may wish to read the extra out-of-band data provided in
 * these tags, and may safely remove the tags from the Packet.
 *
 * - PacketSocketTag: contains destination address (type PacketSocketAddress)
 *   and packet type of the received packet
 *
 * - DeviceNameTag:  contains the TypeId string of the relevant NetDevice
 *
 * \see class PacketSocketTag
 * \see class DeviceNameTag
 */

class PacketSocket : public Socket
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  PacketSocket ();
  virtual ~PacketSocket ();

  /**
   * \brief Set the associated node.
   * \param node the node
   */
  void SetNode (Ptr<Node> node);

  virtual enum SocketErrno GetErrno (void) const;
  virtual enum SocketType GetSocketType (void) const;
  virtual Ptr<Node> GetNode (void) const;
  /**
   * \brief Bind the socket to the NetDevice and register the protocol handler.
   *
   * \warning this will actually bind protocol "0".
   *
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind (void);
  /**
   * \brief Bind the socket to the NetDevice and register the protocol handler.
   *
   * \warning this will actually bind protocol "0".
   *
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind6 (void);
  /**
   * \brief Bind the socket to the NetDevice and register the
   *        protocol handler specified in the address.
   *
   * \param address the packet socket address
   * \returns 0 on success, -1 on failure.
   */
  virtual int Bind (const Address & address);
  virtual int Close (void);
  virtual int ShutdownSend (void);
  virtual int ShutdownRecv (void);
  virtual int Connect (const Address &address);
  virtual int Listen (void);
  virtual uint32_t GetTxAvailable (void) const;
  virtual int Send (Ptr<Packet> p, uint32_t flags);
  virtual int SendTo (Ptr<Packet> p, uint32_t flags, const Address &toAddress);
  virtual uint32_t GetRxAvailable (void) const;
  virtual Ptr<Packet> Recv (uint32_t maxSize, uint32_t flags);
  virtual Ptr<Packet> RecvFrom (uint32_t maxSize, uint32_t flags,
                                Address &fromAddress);
  virtual int GetSockName (Address &address) const; 
  virtual int GetPeerName (Address &address) const;
  virtual bool SetAllowBroadcast (bool allowBroadcast);
  virtual bool GetAllowBroadcast () const;

private:
  /**
   * \brief Called by the L3 protocol when it received a packet to pass on to TCP.
   *
   * \param device the incoming NetDevice
   * \param packet the incoming packet
   * \param protocol the protocol
   * \param from sender address
   * \param to destination address
   * \param packetType packet type
   */
  void ForwardUp (Ptr<NetDevice> device, Ptr<const Packet> packet, 
                  uint16_t protocol, const Address &from, const Address &to,
                  NetDevice::PacketType packetType);
  /**
   * \brief Bind the socket to the NetDevice and register the
   *        protocol handler specified in the address.
   * \param address the packet socket address
   * \returns 0 on success, -1 on failure.
   */
  int DoBind (const PacketSocketAddress &address);

  /**
   * \brief Get the minimum MTU supported by the NetDevices bound to a specific address
   * \param ad the socket address to check for
   * \returns The minimum MTU
   */
  uint32_t GetMinMtu (PacketSocketAddress ad) const;
  virtual void DoDispose (void);

  /**
   * \brief States of the socket
   */
  enum State {
    STATE_OPEN,
    STATE_BOUND,     // open and bound
    STATE_CONNECTED, // open, bound and connected
    STATE_CLOSED
  };

  Ptr<Node> m_node;         //!< the associated node
  mutable enum SocketErrno m_errno; //!< Socket error code
  bool m_shutdownSend;      //!< Send no longer allowed
  bool m_shutdownRecv;      //!< Receive no longer allowed
  enum State m_state;       //!< Socket state
  uint16_t m_protocol;      //!< Socket protocol
  bool m_isSingleDevice;    //!< Is bound to a single netDevice
  uint32_t m_device;        //!< index of the bound NetDevice
  Address m_destAddr;       //!< Default destination address

  std::queue<std::pair<Ptr<Packet>, Address> > m_deliveryQueue; //!< Rx queue
  uint32_t m_rxAvailable; //!< Rx queue size [Bytes]

  /// Traced callback: dropped packets
  TracedCallback<Ptr<const Packet> > m_dropTrace;

  // Socket options (attributes)
  uint32_t m_rcvBufSize; //!< Rx buffer size [Bytes]

};

/**
 * \brief  This class implements a tag that carries the dest address of a packet and the packet type.
 */
class PacketSocketTag : public Tag
{
public:
  /**
   *  Create an empty PacketSocketTag
   */
  PacketSocketTag ();
  /**
   * Set the packet type
   * @param t the packet type of the corresponding packet
   */
  void SetPacketType (NetDevice::PacketType t);
  /**
   * Get the packet type 
   * @return the packet type of the corresponding packet
   */
  NetDevice::PacketType GetPacketType (void) const;
  /**
   * Set the destination address of the corresponding packet
   * @param a the destination address of the corresponding packet
   */
  void SetDestAddress(Address a);
  /**
   * Get the destination address of the corresponding packet
   * @return the destination address of the corresponding packet
   */
  Address GetDestAddress (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

private:
  NetDevice::PacketType m_packetType; //!< Packet type
  Address m_destAddr; //!< Destination address
};
/**
 * \brief  This class implements a tag that carries the ns3 device name from where a packet is coming.
 */
class DeviceNameTag : public Tag
{
public:
  /**
   * Create an empty DeviceNameTag
   */
  DeviceNameTag ();
  /**
   * Set the device name
   * @param n the device name from where the corresponding packet is coming.
   */
  void SetDeviceName (std::string n);
  /**
   * Get the device name from where the corresponding packet is coming.
   * @return the device name from where the corresponding packet is coming.
   */
  std::string GetDeviceName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

private:
  std::string m_deviceName; //!< Device name
};

} // namespace ns3

#endif /* PACKET_SOCKET_H */


