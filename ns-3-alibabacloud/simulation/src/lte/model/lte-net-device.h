/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
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
 * Author: Giuseppe Piro <g.piro@poliba.it>
 *         Nicola Baldo  <nbaldo@cttc.es>
 */

#ifndef LTE_NET_DEVICE_H
#define LTE_NET_DEVICE_H

#include <ns3/net-device.h>
#include <ns3/event-id.h>
#include <ns3/mac64-address.h>
#include <ns3/traced-callback.h>
#include <ns3/nstime.h>
#include <ns3/lte-phy.h>
#include <ns3/lte-control-messages.h>

namespace ns3 {

class Node;
class Packet;

/**
 * \defgroup lte LTE Models
 *
 */

/**
 * \ingroup lte
 *
 * LteNetDevice provides  basic implementation for all LTE network devices
 */
class LteNetDevice : public NetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LteNetDevice (void);
  virtual ~LteNetDevice (void);

  virtual void DoDispose (void);

  // inherited from NetDevice
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex (void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  virtual bool SetMtu (const uint16_t mtu);
  virtual uint16_t GetMtu (void) const;
  virtual void SetAddress (Address address);
  virtual Address GetAddress (void) const;
  virtual bool IsLinkUp (void) const;
  virtual void AddLinkChangeCallback (Callback<void> callback);
  virtual bool IsBroadcast (void) const;
  virtual Address GetBroadcast (void) const;
  virtual bool IsMulticast (void) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool IsBridge (void) const;
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual Address GetMulticast (Ipv4Address addr) const;
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual void SetPromiscReceiveCallback (PromiscReceiveCallback cb); 
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual bool SupportsSendFrom (void) const;

  /** 
   * receive a packet from the lower layers in order to forward it to the upper layers
   * 
   * \param p the packet
   */
  void Receive (Ptr<Packet> p);
  
protected:
  
  NetDevice::ReceiveCallback m_rxCallback; ///< receive callback
  
private:
  /// type conversion operator
  LteNetDevice (const LteNetDevice &);
  /**
   * assignment operator
   * \returns LteNetDevice
   */
  LteNetDevice & operator= (const LteNetDevice &);

  Ptr<Node> m_node; ///< the node

  TracedCallback<> m_linkChangeCallbacks; ///< link change callback

  uint32_t m_ifIndex; ///< interface index
  bool m_linkUp; ///< link uo
  mutable uint16_t m_mtu; ///< MTU

  Mac64Address m_address; ///< MAC address - only relevant for UEs.
};


} // namespace ns3

#endif /* LTE_NET_DEVICE_H */
