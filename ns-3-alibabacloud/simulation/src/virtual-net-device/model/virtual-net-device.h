/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 INESC Porto
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
 * Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt>
 */

#ifndef VIRTUAL_NET_DEVICE_H
#define VIRTUAL_NET_DEVICE_H

#include "ns3/address.h"
#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/ptr.h"
#include "ns3/traced-callback.h"

namespace ns3 {


/**
 * \defgroup virtual-net-device Virtual Device
 *
 */

/**
 * \ingroup virtual-net-device
 *
 * \class VirtualNetDevice
 * \brief A virtual device, similar to Linux TUN/TAP interfaces.
 *
 * A VirtualNetDevice is a "virtual" NetDevice implementation which
 * delegates to a user callback (see method SetSendCallback()) the
 * task of actually transmitting a packet.  It also allows the user
 * code to inject the packet as if it had been received by the
 * VirtualNetDevice.  Together, these features allow one to build tunnels.
 * For instance, by transmitting packets into a UDP socket we end up
 * building an IP-over-UDP-over-IP tunnel, or IP-over-IP tunnels.
 *
 * The same thing could be accomplished by subclassing NetDevice
 * directly.  However, VirtualNetDevice is usually much simpler to program
 * than a NetDevice subclass.
 */
class VirtualNetDevice : public NetDevice
{
public:
  /**
   * Callback the be invoked when the VirtualNetDevice is asked to queue/transmit a packet.
   * For more information, consult the documentation of NetDevice::SendFrom().
   */
  typedef Callback<bool, Ptr<Packet>, const Address&, const Address&, uint16_t> SendCallback;

  /**
   * \brief Get the type ID.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);
  VirtualNetDevice ();

  virtual ~VirtualNetDevice ();

  /**
   * \brief Set the user callback to be called when a L2 packet is to be transmitted
   * \param transmitCb the new transmit callback
   */
  void SetSendCallback (SendCallback transmitCb);

  /**
   * \brief Configure whether the virtual device needs ARP
   *
   * \param needsArp the the 'needs arp' value that will be returned
   * by the NeedsArp() method.  The method IsBroadcast() will also
   * return this value.
   */
  void SetNeedsArp (bool needsArp);

  /**
   * \brief Configure whether the virtual device is point-to-point
   *
   * \param isPointToPoint the value that should be returned by the
   * IsPointToPoint method for this instance.
   */
  void SetIsPointToPoint (bool isPointToPoint);

  /**
   * \brief Configure whether the virtual device supports SendFrom
   * \param supportsSendFrom true if the device supports SendFrom
   */
  void SetSupportsSendFrom (bool supportsSendFrom);

  /**
   * \brief Configure the reported MTU for the virtual device.
   * \param mtu MTU value to set
   * \return whether the MTU value was within legal bounds
   */
  bool SetMtu (const uint16_t mtu);


  /**
   * \param packet packet sent from below up to Network Device
   * \param protocol Protocol type
   * \param source the address of the sender of this packet.
   * \param destination the address of the receiver of this packet.
   * \param packetType type of packet received (broadcast/multicast/unicast/otherhost)
   * \returns true if the packet was forwarded successfully, false otherwise.
   *
   * Forward a "virtually received" packet up
   * the node's protocol stack.
   */
  bool Receive (Ptr<Packet> packet, uint16_t protocol,
                const Address &source, const Address &destination,
                PacketType packetType);


  // inherited from NetDevice base class.
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex (void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  virtual void SetAddress (Address address);
  virtual Address GetAddress (void) const;
  virtual uint16_t GetMtu (void) const;
  virtual bool IsLinkUp (void) const;
  virtual void AddLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast (void) const;
  virtual Address GetBroadcast (void) const;
  virtual bool IsMulticast (void) const;
  virtual Address GetMulticast (Ipv4Address multicastGroup) const;
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom () const;
  virtual bool IsBridge (void) const;

protected:

  virtual void DoDispose (void);

private:

  Address m_myAddress;    //!< MAC address
  SendCallback m_sendCb;  //!< send callback
  TracedCallback<Ptr<const Packet> > m_macRxTrace;            //!< Rx trace
  TracedCallback<Ptr<const Packet> > m_macTxTrace;            //!< Tx trace
  TracedCallback<Ptr<const Packet> > m_macPromiscRxTrace;     //!< Promisc Rx trace
  TracedCallback<Ptr<const Packet> > m_snifferTrace;          //!< Sniffer trace
  TracedCallback<Ptr<const Packet> > m_promiscSnifferTrace;   //!< Promisc Sniffer trace
  Ptr<Node> m_node;                               //!< Pointer to the node
  ReceiveCallback m_rxCallback;                   //!< Rx callback
  PromiscReceiveCallback m_promiscRxCallback;     //!< Promisc Rx callback
  std::string m_name;       //!< Name of the device
  uint32_t m_index;         //!< Device index
  uint16_t m_mtu;           //!< MTU
  bool m_needsArp;          //!< True if the device needs ARP
  bool m_supportsSendFrom;  //!< True if the device supports SendFrm
  bool m_isPointToPoint;    //!< True if the device is a PointToPoint type device
};

} // namespace ns3

#endif

