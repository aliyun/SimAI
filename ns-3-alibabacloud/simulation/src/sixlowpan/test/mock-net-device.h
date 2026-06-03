/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */
#ifndef MOCK_NET_DEVICE_H
#define MOCK_NET_DEVICE_H

#include <stdint.h>
#include <string>

#include "ns3/traced-callback.h"
#include "ns3/net-device.h"

namespace ns3 {

class Node;

/**
 * \ingroup netdevice
 *
 * This device assumes 48-bit mac addressing; there is also the possibility to
 * add an ErrorModel if you want to force losses on the device.
 * 
 * The device can be installed on a node through the MockNetDeviceHelper.
 * In case of manual creation, the user is responsible for assigning an unique
 * address to the device.
 *
 * By default the device is in Broadcast mode, with infinite bandwidth.
 *
 * \brief simple net device for simple things and testing
 */
class MockNetDevice : public NetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  MockNetDevice ();

  /**
   * Pretend that a packet has been received from a connected Channel.
   *
   * Note that no analysis is performed on the addresses, and the
   * packet is forwarded to the callbacks according to the packetType.
   *
   * \param packet Packet received on the channel
   * \param protocol protocol number
   * \param to address packet should be sent to
   * \param from address packet was sent from
   * \param packetType type of the packet (e.g., NetDevice::PACKET_HOST, NetDevice::PACKET_OTHERHOST, etc.)
   */
  void Receive (Ptr<Packet> packet, uint16_t protocol, Address to, Address from, NetDevice::PacketType packetType);

  // inherited from NetDevice base class.
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex (void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  virtual void SetAddress (Address address);
  virtual Address GetAddress (void) const;
  virtual bool SetMtu (const uint16_t mtu);
  virtual uint16_t GetMtu (void) const;
  virtual bool IsLinkUp (void) const;
  virtual void AddLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast (void) const;
  virtual Address GetBroadcast (void) const;
  virtual bool IsMulticast (void) const;
  virtual Address GetMulticast (Ipv4Address multicastGroup) const;
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool IsBridge (void) const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom (void) const;

  /**
   *
   * Add a callback to be invoked when the MockNetDevice has a packet to "send".
   *
   * In the callback the PacketType is always set to NetDevice::PACKET_HOST.
   *
   * \param cb callback to invoke whenever the MockNetDevice has one packet to "send".
   *
   */
  void SetSendCallback (PromiscReceiveCallback cb);

protected:
  virtual void DoDispose (void);

private:

  NetDevice::ReceiveCallback m_rxCallback; //!< Receive callback
  NetDevice::PromiscReceiveCallback m_promiscCallback; //!< Promiscuous receive callback
  NetDevice::PromiscReceiveCallback m_sendCallback; //!< Send callback
  Ptr<Node> m_node; //!< Node this netDevice is associated to
  uint16_t m_mtu;   //!< MTU
  uint32_t m_ifIndex; //!< Interface index
  Address m_address; //!< MAC address

  bool m_linkUp; //!< Flag indicating whether or not the link is up
  bool m_pointToPointMode; //!< Enabling this will disable Broadcast and Arp.

  /**
   * List of callbacks to fire if the link changes state (up or down).
   */
  TracedCallback<> m_linkChangeCallbacks;
};

} // namespace ns3

#endif /* MOCK_NET_DEVICE_H */
