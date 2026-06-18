/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  Tom Henderson <thomas.r.henderson@boeing.com>
 *  Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *  Margherita Filippetti <morag87@gmail.com>
 */
#ifndef LR_WPAN_NET_DEVICE_H
#define LR_WPAN_NET_DEVICE_H

#include <ns3/net-device.h>
#include <ns3/traced-callback.h>
#include <ns3/lr-wpan-mac.h>

namespace ns3 {

class LrWpanPhy;
class LrWpanCsmaCa;
class SpectrumChannel;
class Node;

/**
 * \ingroup lr-wpan
 *
 * \brief Network layer to device interface.
 *
 * The ns3::NetDevice includes IP-specific API such as GetMulticast(), Send()
 * and SendTo() methods, which do not map well the the 802.15.4 MAC MCPS
 * DataRequest primitive.  So, the basic design is to provide, as
 * much as makes sense, the class ns3::NetDevice API, but rely on the user
 * accessing the LrWpanMac pointer to make 802.15.4-specific API calls.
 * As such, this is really just an encapsulating class.
 */
class LrWpanNetDevice : public NetDevice
{
public:
  /**
   * Get the type ID.
   *
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LrWpanNetDevice (void);
  virtual ~LrWpanNetDevice (void);

  /**
   * How the pseudo-MAC address is built from
   * the short address (XXXX) and the PanId (YYYY).
   *
   * See \RFC{4944} and \RFC{6282}.
   */
  enum PseudoMacAddressMode_e {
    RFC4944,  //!< YYYY:0000:XXXX (with U/L bit set to local)
    RFC6282   //!< 0200:0000:XXXX
  };

  /**
   * Set the MAC to be used by this NetDevice.
   *
   * \param mac the MAC to be used
   */
  void SetMac (Ptr<LrWpanMac> mac);

  /**
   * Set the PHY to be used by the MAC and this NetDevice.
   *
   * \param phy the PHY to be used
   */
  void SetPhy (Ptr<LrWpanPhy> phy);

  /**
   * Set the CSMA/CA implementation to be used by the MAC and this NetDevice.
   *
   * \param csmaca the CSMA/CA implementation to be used
   */
  void SetCsmaCa (Ptr<LrWpanCsmaCa> csmaca);

  /**
   * Set the channel to which the NetDevice, and therefore the PHY, should be
   * attached to.
   *
   * \param channel the channel to be used
   */
  void SetChannel (Ptr<SpectrumChannel> channel);

  /**
   * Get the MAC used by this NetDevice.
   *
   * \return the MAC object
   */
  Ptr<LrWpanMac> GetMac (void) const;

  /**
   * Get the PHY used by this NetDevice.
   *
   * \return the PHY object
   */
  Ptr<LrWpanPhy> GetPhy (void) const;

  /**
   * Get the CSMA/CA implementation used by this NetDevice.
   *
   * \return the CSMA/CA implementation object
   */
  Ptr<LrWpanCsmaCa> GetCsmaCa (void) const;

  // From class NetDevice
  virtual void SetIfIndex (const uint32_t index);
  virtual uint32_t GetIfIndex (void) const;
  virtual Ptr<Channel> GetChannel (void) const;
  /**
   * This method indirects to LrWpanMac::SetShortAddress ()
   * \param address The short address.
   */
  virtual void SetAddress (Address address);
  /**
   * This method indirects to LrWpanMac::SetShortAddress ()
   * \returns The short address.
   */
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
  virtual bool IsBridge (void) const;
  virtual bool IsPointToPoint (void) const;
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual Ptr<Node> GetNode (void) const;
  virtual void SetNode (Ptr<Node> node);
  virtual bool NeedsArp (void) const;

  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual void SetPromiscReceiveCallback (PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom (void) const;

  /**
   * The callback used by the MAC to hand over incoming packets to the
   * NetDevice. This callback will in turn use the ReceiveCallback set by
   * SetReceiveCallback() to notify upper layers.
   *
   * \param params 802.15.4 specific parameters, including source and destination addresses
   * \param pkt the packet do be delivered
   */
  void McpsDataIndication (McpsDataIndicationParams params, Ptr<Packet> pkt);

  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model.  Return the number of streams that have been assigned.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  int64_t AssignStreams (int64_t stream);

private:
  // Inherited from NetDevice/Object
  virtual void DoDispose (void);
  virtual void DoInitialize (void);

  /**
   * Mark NetDevice link as up.
   */
  void LinkUp (void);

  /**
   * Mark NetDevice link as down.
   */
  void LinkDown (void);

  /**
   * Attribute accessor method for the "Channel" attribute.
   *
   * \return the channel to which this NetDevice is attached
   */
  Ptr<SpectrumChannel> DoGetChannel (void) const;

  /**
   * Configure PHY, MAC and CSMA/CA.
   */
  void CompleteConfig (void);

  /**
   * Builds a "pseudo 48-bit address" from the PanId and Short Address
   * The form is PanId : 0x0 : 0x0 : ShortAddress
   *
   * The address follows RFC 4944, section 6, and it is used to build an
   * Interface ID.
   *
   * The Interface ID should have its U/L bit is set to zero, to indicate that
   * this interface ID is not globally unique.
   * However, the U/L bit flipping is performed when the IPv6 address is created.
   *
   * As a consequence, here we set it to 1.
   *
   * \param panId The PanID
   * \param shortAddr The Short MAC address
   * \return a Pseudo-Mac48Adress
   */
  Mac48Address BuildPseudoMacAddress (uint16_t panId, Mac16Address shortAddr) const;

  /**
   * The MAC for this NetDevice.
   */
  Ptr<LrWpanMac> m_mac;

  /**
   * The PHY for this NetDevice.
   */
  Ptr<LrWpanPhy> m_phy;

  /**
   * The CSMA/CA implementation for this NetDevice.
   */
  Ptr<LrWpanCsmaCa> m_csmaca;

  /**
   * The node associated with this NetDevice.
   */
  Ptr<Node> m_node;

  /**
   * True if MAC, PHY and CSMA/CA where successfully configured and the
   * NetDevice is ready for being used.
   */
  bool m_configComplete;

  /**
   * Configure the NetDevice to request MAC layer acknowledgments when sending
   * packets using the Send() API.
   */
  bool m_useAcks;

  /**
   * Is the link/device currently up and running?
   */
  bool m_linkUp;

  /**
   * The interface index of this NetDevice.
   */
  uint32_t m_ifIndex;

  /**
   * Trace source for link up/down changes.
   */
  TracedCallback<> m_linkChanges;

  /**
   * Upper layer callback used for notification of new data packet arrivals.
   */
  ReceiveCallback m_receiveCallback;

  /**
   * How the pseudo MAC address is created.
   *
   * According to \RFC{4944} the psudo-MAC is YYYY:0000:XXXX (with U/L bit set to local)
   * According to \RFC{6282} the psudo-MAC is 0200:0000:XXXX
   */
  PseudoMacAddressMode_e m_pseudoMacMode;
};

} // namespace ns3

#endif /* LR_WPAN_NET_DEVICE_H */
