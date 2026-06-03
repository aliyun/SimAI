/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
 *         Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef WAVE_NET_DEVICE_H
#define WAVE_NET_DEVICE_H

#include <map>
#include <vector>
#include "ns3/packet.h"
#include "ns3/traced-callback.h"
#include "ns3/mac48-address.h"
#include "ns3/wifi-net-device.h"
#include "ocb-wifi-mac.h"
#include "vendor-specific-action.h"
#include "channel-coordinator.h"
#include "channel-manager.h"
#include "channel-scheduler.h"
#include "vsa-manager.h"
namespace ns3 {
struct SchInfo;
struct VsaInfo;
class NetDevice;
class ChannelScheduler;
class VsaManager;
class OcbWifiMac;

/**
 * \defgroup wave WAVE module
 */


/**
 * \ingroup wave
 *
 * \param channelNumber the specific channel
 * \param priority the priority of packet with range 0-7
 * \param dataRate the transmit data rate of packet
 * \param txPowerLevel the transmit power level with range 0-7
 * \param expireTime indicate how many milliseconds the packet
 * can stay before sent
 *
 * typically these parameters are used by higher layer to control
 * the transmission characteristics of WSMP packets.
 * When dataRate is an invalid WifiMode and txPowerLevel is 8, this
 * indicates that high layer does not set tx parameters.
 */
struct TxInfo
{
  uint32_t channelNumber; ///< channel number
  uint32_t priority; ///< priority
  WifiMode dataRate; ///< data rate
  WifiPreamble preamble; ///< preamble
  uint32_t txPowerLevel; ///< transmit power level
  // Time expiryTime;   // unsupported
  /// Initializer
  TxInfo ()
    : channelNumber (CCH),
      priority (7),
      txPowerLevel (8)
  {

  }
  /**
   * Initializer
   * \param channel the channel
   * \param prio the priority
   * \param rate the wifi mode
   * \param preamble the preamble
   * \param powerLevel the power level
   */
  TxInfo (uint32_t channel, uint32_t prio = 7, WifiMode rate = WifiMode (), WifiPreamble preamble = WIFI_PREAMBLE_LONG, uint32_t powerLevel = 8)
    : channelNumber (channel),
      priority (prio),
      dataRate (rate),
      preamble (preamble),
      txPowerLevel (powerLevel)
  {

  }
};

/**
 * \ingroup wave
 *
 * \param channelNumber the channel number for the SCH.
 * \param adaptable if true, the actual power level and data
 * rate for transmission are adaptable. TxPwr_Level is the maximum
 * transmit power that sets the upper bound for the actual transmit
 * power; DataRate is the minimum data rate that sets the lower
 * bound for the actual data rate. If false, the actual power level
 * and data rate for transmission are non-adaptable. TxPwr_Level and
 * DataRate are the actual values to be used for transmission.
 * \param txPowerLevel transmit power level.
 * \param dataRate transmit data rate
 */
struct TxProfile
{
  uint32_t channelNumber; ///< channel number
  bool adaptable; ///< adaptable
  uint32_t txPowerLevel; ///< transmit power level
  WifiMode dataRate; ///< data rate
  WifiPreamble preamble; ///< preamble
  /// Initializer
  TxProfile (void)
    : channelNumber (SCH1),
      adaptable (false),
      txPowerLevel (4),
      preamble (WIFI_PREAMBLE_LONG)
  {
      dataRate = WifiMode ("OfdmRate6MbpsBW10MHz");
  }
  /**
   * Initializer
   * \param channel the channel number
   * \param adapt true to adapt
   * \param powerLevel the power level
   */
  TxProfile (uint32_t channel, bool adapt = true, uint32_t powerLevel = 4)
    : channelNumber (channel),
      adaptable (adapt),
      txPowerLevel (powerLevel)
  {
      dataRate = WifiMode ("OfdmRate6MbpsBW10MHz");
      preamble = WIFI_PREAMBLE_LONG;
  }
};

/**
 * \ingroup wave
 *
 * This class holds together multiple, ns3::WifiPhy,
 * and ns3::OcbWifiMac (including ns3::WifiRemoteStationManager).
 * Besides that, to support multiple channel operation this
 * class also holds ns3::ChannelScheduler, ns3::ChannelManager,
 * ns3::ChannelCoordinator and ns3::VsaManager.
 *
 * these primitives specified in the standard will not be implemented
 * because of limited use in simulation:
 * void StartTimingAdvertisement ();
 * void StopTimingAdvertisement ();
 * UtcTime GetUtcTime ();
 * void SetUtcTime ();
 */
class WaveNetDevice : public WifiNetDevice
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  WaveNetDevice (void);
  virtual ~WaveNetDevice (void);

  /**
   * \param channelNumber the specific channel
   * \param mac add a new available MAC entity for specific channel
   */
  void AddMac (uint32_t channelNumber, Ptr<OcbWifiMac> mac);
  /**
   * \param channelNumber the specific channel number
   * \return corresponding MAC entity
   */
  Ptr<OcbWifiMac> GetMac (uint32_t channelNumber) const;
  /**
   * \return all inserted MAC entities.
   */
  std::map<uint32_t, Ptr<OcbWifiMac> >  GetMacs (void) const;
  /**
   * \param phy add a new available PHY entity
   */
  void AddPhy (Ptr<WifiPhy> phy);
  /**
   * \param index the index of PHY entity
   * \return corresponding PHY entity
   */
  Ptr<WifiPhy> GetPhy (uint32_t index) const;
  /**
   * \return all inserted PHY entities.
   */
  std::vector<Ptr<WifiPhy> > GetPhys (void) const;

  /**
   * \param channelScheduler the channel scheduler for multiple channel operation
   */
  void SetChannelScheduler (Ptr<ChannelScheduler> channelScheduler);
  /**
   * \return current channel scheduler for multiple channel operation
   */
  Ptr<ChannelScheduler> GetChannelScheduler (void) const;
  /**
   * \param channelManager the channel manager for multiple channel operation
   */
  void SetChannelManager (Ptr<ChannelManager> channelManager);
  /**
   * \return currentc channel manager for multiple channel operation
   */
  Ptr<ChannelManager> GetChannelManager (void) const;
  /**
   * \param channelCoordinator  the channel coordinator for multiple channel operation
   */
  void SetChannelCoordinator (Ptr<ChannelCoordinator> channelCoordinator);
  /**
   * \return current channel coordinator for multiple channel operation
   */
  Ptr<ChannelCoordinator> GetChannelCoordinator (void) const;
  /**
   * \param vsaManager the VSA manager for multiple channel operation
   */
  void SetVsaManager (Ptr<VsaManager> vsaManager);
  /**
   * \return current VSA manager for multiple channel operation
   */
  Ptr<VsaManager> GetVsaManager (void) const;

  /**
   * \param schInfo the parameters about how to start SCH service
   * \return whether channel access is assigned successfully
   */
  bool StartSch (const SchInfo & schInfo);
  /**
   *  \param channelNumber the channel which access resource will be released.
   *  \return whether channel access is released successfully
   */
  bool StopSch (uint32_t channelNumber);

  /**
   * \param vsaInfo the parameters about how to send VSA frame
   * \return whether the request for VSA transmission is completed
   */
  bool StartVsa (const VsaInfo & vsaInfo);
  /**
   * \param channelNumber the channel on which VSA transmit event will be canceled.
   * \return whether the request for stopping VSA transmission is completed
   */
  bool StopVsa (uint32_t channelNumber);
  /**
   * \param packet the packet is Vendor Specific Action frame.
   * \param address the address of the MAC from which the management frame
   *  was received.
   * \param managementID identify the originator of the data.
   *  Values are specified in IEEE P1609.0 with range 0-15.
   * \param channelNumber the channel on which the frame was received.
   * \returns true if the callback could handle the packet successfully, false
   *          otherwise.
   */
  typedef Callback<bool, Ptr<const Packet>,const Address &, uint32_t, uint32_t> WaveVsaCallback;
  /**
   * \param vsaCallback the VSA receive callback for 1609 management information
   */
  void SetWaveVsaCallback (WaveVsaCallback vsaCallback);

  /**
   * \param txprofile transmit profile for IP-based data
   * register a transmitter profile in the MLME before
   * the associated IP-based data transfer starts.
   * \return whether the tx profile is registered successfully
   *
   * note: This method should be called before WaveNetDevice::Send method
   */
  bool RegisterTxProfile (const TxProfile &txprofile);
  /**
   * \param channelNumber the specific channel number
   * delete a registered transmitter profile in the MLME
   * after the associated IP-based data transfer is complete
   * \return whether the tx profile is unregistered successfully
   */
  bool DeleteTxProfile (uint32_t channelNumber);

  /**
   * \param packet packet sent from above down to Network Device
   * \param dest mac address of the destination (already resolved)
   * \param protocol identifies the type of payload contained in the packet.
   *  Used to call the right L3Protocol when the packet is received.
   * \param txInfo WSMP or other packets parameters for sending
   * \return whether the SendX operation succeeded
   *
   * Normally this method is called by 1609.3 standard to
   * send WSMP packets, however high layers can also send packets
   * in other types except IP-based packets in CCH.
   */
  bool SendX (Ptr<Packet> packet, const Address& dest, uint32_t protocol, const TxInfo & txInfo);
  /**
   * \param newAddress  an immediate MAC-layer address change is required
   *
   * This method is similar with SetAddress method, but
   * SetAddress is suggested for initializing a device, while this method
   * is preferred for changing address and a addressChange TracedCallback
   * will be called.
   */
  void ChangeAddress (Address newAddress);
  /**
   * \param channelNumber the specific channel number
   * \param ac the specific access category
   *
   * Cancel all transmissions with the particular category and channel number.
   */
  void CancelTx (uint32_t channelNumber, enum AcIndex ac);

  /**
   * \param packet packet sent from above down to Network Device
   * \param dest mac address of the destination (already resolved)
   * \param protocolNumber identifies the type of payload contained in
   *        this packet. Used to call the right L3Protocol when the packet
   *        is received.
   * \return whether the Send operation succeeded
   *
   * Normally this method is called by 1609.3 standard to
   * send IP-based packets, however high layers can also send packets
   * in other types except IP-based packets in CCH.
   */
  virtual bool Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber);

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
  virtual bool IsPointToPoint (void) const;
  virtual bool IsBridge (void) const;
  virtual bool NeedsArp (void) const;
  virtual void SetReceiveCallback (NetDevice::ReceiveCallback cb);
  virtual Address GetMulticast (Ipv6Address addr) const;
  virtual bool SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber);
  virtual void SetPromiscReceiveCallback (PromiscReceiveCallback cb);
  virtual bool SupportsSendFrom (void) const;

private:
  /// This value conforms to the 802.11 specification
  static const uint16_t MAX_MSDU_SIZE = 2304;

  /// IP v4 Protocol number
  static const uint16_t IPv4_PROT_NUMBER = 0x0800;
  /// IP v6 Protocol number
  static const uint16_t IPv6_PROT_NUMBER = 0x86DD;

  virtual void DoDispose (void);
  virtual void DoInitialize (void);
  /**
   * \param channelNumber the specific channel
   * \return whether this channel is valid and available for use
   */
  bool IsAvailableChannel (uint32_t channelNumber) const;
  /**
   * Receive a packet from the lower layer and pass the
   * packet up the stack.
   *
   * \param packet
   * \param from
   * \param to
   */
  void ForwardUp (Ptr<const Packet> packet, Mac48Address from, Mac48Address to);

  /// MacEntities typedef
  typedef std::map<uint32_t, Ptr<OcbWifiMac> > MacEntities;
  /// MacEntities iterator typedef
  typedef std::map<uint32_t, Ptr<OcbWifiMac> >::const_iterator MacEntitiesI;
  MacEntities m_macEntities; ///< MAC entities
  /// PhyEntities typedef
  typedef std::vector<Ptr<WifiPhy> > PhyEntities; 
  /// PhyEntities iterator typedef
  typedef std::vector<Ptr<WifiPhy> >::const_iterator PhyEntitiesI;
  PhyEntities m_phyEntities; ///< Phy entities

  Ptr<ChannelManager> m_channelManager; ///< the channel manager
  Ptr<ChannelScheduler> m_channelScheduler; ///< the channel scheduler
  Ptr<ChannelCoordinator> m_channelCoordinator; ///< the channel coordinator
  Ptr<VsaManager> m_vsaManager; ///< the VSA manager 
  TxProfile *m_txProfile; ///< transmit profile
  /**
   * \todo The Address arguments should be passed
   * by const reference, since they are large.
   */
  TracedCallback<Address, Address> m_addressChange;

  // copy from WifiNetDevice
  NetDevice::ReceiveCallback m_forwardUp; ///< forward up receive callback
  NetDevice::PromiscReceiveCallback m_promiscRx; ///< promiscious receive callback
  uint32_t m_ifIndex; ///< IF index
  mutable uint16_t m_mtu; ///< MTU
};

} // namespace ns3

#endif /* WAVE_NET_DEVICE_H */
