/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Dalian University of Technology
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef VSA_MANAGER_H
#define VSA_MANAGER_H
#include <vector>
#include "wave-net-device.h"

namespace ns3 {

/**
 * \ingroup wave
 *
 * \brief indicate which interval the VSA frames will be transmitted in.
 * VSA_TRANSMIT_IN_CCHI will only allow in CCH Interval;
 * VSA_TRANSMIT_IN_SCHI will only allow in SCH Interval;
 * VSA_TRANSMIT_IN_BOTHI will allow anytime.
 */
enum VsaTransmitInterval
{
  VSA_TRANSMIT_IN_CCHI = 1,
  VSA_TRANSMIT_IN_SCHI = 2,
  VSA_TRANSMIT_IN_BOTHI = 3,
};

/**
 * \ingroup wave
 *
 * \param peer The address of the peer MAC entity to which the
 * VSA is sent.
 * \param oi Identifies the source of the data when the source
 * is not an IEEE 1609 entity. See IEEE Std 802.11p.
 * \param managementId Identifies the source of the data when the source
 * is an IEEE 1609 entity. Values are specified in IEEE P1609.0.
 * Valid range: 0-15
 * \param vsc pointer to Information that will be sent as vendor specific content.
 * \param channelNumber The channel on which the transmissions are to occur.
 * While section 7.2 of the standard specifies that channel identification
 * comprises Country String, Operating Class, and Channel Number, the channel
 * number is enough for simulation.
 * \param repeatRate The number of Vendor Specific Action frames to
 * be transmitted per 5 s. A value of 0 indicates a single message is to be sent.
 * If Destination MAC Address is an individual address, Repeat Rate is ignored.
 * \param sendInterval The channel interval in which the transmissions
 * are to occur.
 */
struct VsaInfo
{
  Mac48Address peer; ///< peer
  OrganizationIdentifier oi; ///< OI
  uint8_t managementId; ///< management ID
  Ptr<Packet> vsc; ///< VSC
  uint32_t channelNumber; ///< channel number
  uint8_t repeatRate; ///< repeat rate
  enum VsaTransmitInterval sendInterval; ///< send interval

  /**
   * Initializer
   * \param peer the peer MAC address
   * \param identifier the organization identifier
   * \param manageId the manage ID
   * \param vscPacket the VSC packet
   * \param channel the channel
   * \param repeat the repeat value
   * \param interval the transmit interval
   */
  VsaInfo (Mac48Address peer, OrganizationIdentifier identifier, uint8_t manageId, Ptr<Packet> vscPacket,
           uint32_t channel, uint8_t repeat, enum VsaTransmitInterval interval)
    : peer (peer),
      oi (identifier),
      managementId (manageId),
      vsc (vscPacket),
      channelNumber (channel),
      repeatRate (repeat),
      sendInterval (interval)
  {

  }
};

/**
 * \ingroup wave
 *
 * refer to 1609.4-2010 chapter 6.4
 * Vendor Specific Action (VSA) frames transmission.
 *
 * The channel interval and channel number indicating how to transmit VSA frames.
 * However making the VSA transmitted strictly in required channel interval and channel number
 * is hard to achieve. For example, if current channel is assigned “Alternating Access” and higher layer wants
 * VSA transmitted only in CCHI (VSA_TRANSMIT_IN_CCHI), but when packet is dequeued from
 * the internal queue of MAC layer in SCHI and can contend for channel access, how to deal with
 * this case? Reinserting into the head of the queue and dequeuing the second packet is not a good
 * choice, because it will require queue traversal. Reinserting the packet into the tail of the queue is
 * worse, because it will make queue not in order. And both solutions may affect many MAC classes
 * of Wifi module. Current approach is to guarantee packets can be inserted into the  MAC
 * internal queue strictly in channel interval and channel number required by higher layers. This solution
 * will result in high probability that packets will be sent in channel interval as higher layer wants,
 * while some packet may be sent practically in other channel interval that not follows the requirement
 * of higher layer request due to queue delay of MAC layer.
 * Therefore, we suggest users assign alternating access for sending VSAs in CCH Interval (VSA_TRANSMIT_IN_CCHI)
 * or SCH Interval (VSA_TRANSMIT_IN_SCHI), and assign continuous access or extended access for
 * sending VSAs in both interval (VSA_TRANSMIT_IN_BOTHI) to avoid the above cases.
 */
class VsaManager : public Object
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  VsaManager (void);
  virtual ~VsaManager (void);

  /**
   * \param device WaveNetDevice associated with VsaManager
   */
  void SetWaveNetDevice (Ptr<WaveNetDevice> device);

  /**
   * Set wave vsa callback function
   * \param vsaCallback the callback
   */
  void SetWaveVsaCallback (Callback<bool, Ptr<const Packet>,const Address &, uint32_t, uint32_t>  vsaCallback);

  /**
   * \param vsaInfo the tx information for VSA transmissions
   */
  void SendVsa (const VsaInfo &vsaInfo);
  /**
   * cancel all VSA transmissions
   */
  void RemoveAll (void);
  /**
   * \param channelNumber cancel VSA transmission specified by channel number
   */
  void RemoveByChannel (uint32_t channelNumber);
  /**
   * \param oi cancel VSA transmission specified by organization identifier
   */
  void RemoveByOrganizationIdentifier (const OrganizationIdentifier &oi);
private:
  void DoDispose (void);
  void DoInitialize (void);

  /**
   * \param mac the MAC entity which receives VSA frame
   * \param oi the Organization Identifier of received VSA frame
   * \param vsc the vendor specific content of received VSA frame
   * \param src the source address of received VSA frame
   * \return true if successful
   */
  bool ReceiveVsc (Ptr<WifiMac> mac, const OrganizationIdentifier &oi, Ptr<const Packet> vsc, const Address &src);

  /// A number of VSA frames will be transmitted repeatedly during the period of 5s.
  const static uint32_t VSA_REPEAT_PERIOD = 5;

  /// VsaWork structure
  struct VsaWork
  {
    Mac48Address peer; ///< peer
    OrganizationIdentifier oi; ///< OI
    Ptr<Packet> vsc; ///< VSC
    uint32_t channelNumber; ///< channel number
    enum VsaTransmitInterval sentInterval; ///< VSA transmit interval
    Time repeatPeriod; ///< repeat period
    EventId repeat; ///< repeat ID
  };

  /**
   * \param vsa the specific VSA repeat work
   *
   * Repeat to send VSA frames
   */
  void DoRepeat (VsaWork *vsa);
  /**
   * \param interval the specific channel interval for VSA transmission
   * \param channel the specific channel number for VSA transmission
   * \param vsc the data field of VSA frame that contains vendor specific content
   * \param oi the Organization Identifier for VSA frame
   * \param peer the destination address
   */
  void DoSendVsa (enum VsaTransmitInterval  interval, uint32_t channel, Ptr<Packet> vsc, OrganizationIdentifier oi, Mac48Address peer);

  Callback<bool, Ptr<const Packet>,const Address &, uint32_t, uint32_t> m_vsaReceived; ///< VSA received callback
  std::vector<VsaWork *> m_vsas; ///< VSAs
  Ptr<WaveNetDevice> m_device; ///< the device
};

}
#endif /* VSA_MANAGER_H */
