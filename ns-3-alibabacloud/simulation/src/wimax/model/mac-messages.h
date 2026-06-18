/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */


/*
 *This file does not contain all MAC messages, the rest of MAC messages have
 *This been categorized as DL and UL messages and are placed in
 *This dl-mac-messages.h and ul-mac-messages.h files.
 */

#ifndef MANAGEMENT_MESSAGE_TYPE_H
#define MANAGEMENT_MESSAGE_TYPE_H

#include <stdint.h>
#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup wimax
 * Mac Management messages
 * Section 6.3.2.3 MAC Management messages page 42, Table 14 page 43
 */
class ManagementMessageType : public Header
{
public:
  /// Message type enumeration
  enum MessageType
  {
    MESSAGE_TYPE_UCD = 0,
    MESSAGE_TYPE_DCD = 1,
    MESSAGE_TYPE_DL_MAP = 2,
    MESSAGE_TYPE_UL_MAP = 3,
    MESSAGE_TYPE_RNG_REQ = 4,
    MESSAGE_TYPE_RNG_RSP = 5,
    MESSAGE_TYPE_REG_REQ = 6,
    MESSAGE_TYPE_REG_RSP = 7,
    MESSAGE_TYPE_DSA_REQ = 11,
    MESSAGE_TYPE_DSA_RSP = 12,
    MESSAGE_TYPE_DSA_ACK = 13
  };

  ManagementMessageType (void);
  /**
   * Constructor
   *
   * \param type message type
   */
  ManagementMessageType (uint8_t type);
  virtual ~ManagementMessageType (void);
  /**
   * Set type field
   * \param type the type
   */
  void SetType (uint8_t type);
  /**
   * Get type field
   * \returns the type value
   */
  uint8_t GetType (void) const;

  /** \returns the name field */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_type; ///< type
};

} // namespace ns3

#endif /* MANAGEMENT_MESSAGE_TYPE_H */


// ---------------------------------------------------------------------

#ifndef RNG_RSP_H
#define RNG_RSP_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/mac48-address.h"
#include "service-flow.h"
#include "cid.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the ranging response message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.6 Ranging response (RNG-RSP) message, page 50
 */
class RngRsp : public Header
{
public:
  RngRsp (void);
  virtual ~RngRsp (void);
  /**
   * \brief set the  Tx timing offset adjustment (signed 32-bit).
   * \param timingAdjust The time required to advance SS transmission so frames
   *  arrive at the expected time instance at the BS.
   */
  void SetTimingAdjust (uint32_t timingAdjust);
  /**
   *  \brief set the relative change in transmission power level that the SS should make in order that
   * transmissions arrive at the BS at the desired power. When subchannelization is employed, the
   * subscriber shall interpret the power offset adjustment as a required change to the transmitted power
   * density.
   * \param powerLevelAdjust the relative change in transmission power level
   */
  void SetPowerLevelAdjust (uint8_t powerLevelAdjust);
  /**
   * \brief set the relative change in transmission frequency that the SS should take in order to better match
   * the BS. This is fine-frequency adjustment within a channel, not reassignment to a different channel
   * \param offsetFreqAdjust
   */
  void SetOffsetFreqAdjust (uint32_t offsetFreqAdjust);
  /**
   * \brief set the range status.
   * \param rangStatus
   */
  void SetRangStatus (uint8_t rangStatus);
  /**
   * \brief set the Center frequency, in kHz, of new downlink channel where the SS should redo initial ranging.
   * \param dlFreqOverride the Center frequency in kHz
   */
  void SetDlFreqOverride (uint32_t dlFreqOverride);
  /**
   * \brief set the identifier of the uplink channel with which the SS is to redo initial ranging
   * \param ulChnlIdOverride the uplink channel index
   */
  void SetUlChnlIdOverride (uint8_t ulChnlIdOverride);
  /**
   * \brief set the DL oper burst profile
   * \param dlOperBurstProfile the oper burt profile
   */
  void SetDlOperBurstProfile (uint16_t dlOperBurstProfile);
  /**
   * \brief set the MAC address
   * \param macAddress the MAC address
   */
  void SetMacAddress (Mac48Address macAddress);

  /**
   * \brief set basic CID.
   * \param basicCid
   */
  void SetBasicCid (Cid basicCid);
  /**
   * \brief set primary CID.
   * \param primaryCid
   */
  void SetPrimaryCid (Cid primaryCid);

  /**
   * \brief set AAS broadcast permission.
   * \param aasBdcastPermission
   */
  void SetAasBdcastPermission (uint8_t aasBdcastPermission);
  /**
   * \brief set frame number.
   * \param frameNumber
   */
  void SetFrameNumber (uint32_t frameNumber);
  /**
   * \brief set initial range opp number.
   * \param initRangOppNumber
   */
  void SetInitRangOppNumber (uint8_t initRangOppNumber);
  /**
   * \brief set range sub channel.
   * \param rangSubchnl
   */
  void SetRangSubchnl (uint8_t rangSubchnl);
  /**
   * \return Tx timing offset adjustment (signed 32-bit). The time required to advance SS transmission so frames
   *  arrive at the expected time instance at the BS.
   */
  uint32_t GetTimingAdjust (void) const;
  /**
   *  \return the relative change in transmission power level that the SS should take in order that
   * transmissions arrive at the BS at the desired power. When subchannelization is employed, the
   * subscriber shall interpret the power offset adjustment as a required change to the transmitted power
   * density.
   */
  uint8_t GetPowerLevelAdjust (void) const;
  /**
   * \return the relative change in transmission frequency that the SS should take in order to better match
   * the BS. This is fine-frequency adjustment within a channel, not reassignment to a different channel.
   */
  uint32_t GetOffsetFreqAdjust (void) const;
  /**
   * \return the range status.
   */
  uint8_t GetRangStatus (void) const;
  /**
   * \return Center frequency, in kHz, of new downlink channel where the SS should redo initial ranging.
   */
  uint32_t GetDlFreqOverride (void) const;
  /**
   * \return The identifier of the uplink channel with which the SS is to redo initial ranging
   */
  uint8_t GetUlChnlIdOverride (void) const;
  /**
   * \return DlOperBurstProfile: This parameter is sent in response to the RNG-REQ Requested Downlink Burst Profile parameter
   */
  uint16_t GetDlOperBurstProfile (void) const;
  /**
   * \return MAC address
   */
  Mac48Address GetMacAddress (void) const;
  /**
   * \return basic CID
   */
  Cid GetBasicCid (void) const;
  /**
   * \return primary CID
   */
  Cid GetPrimaryCid (void) const;
  /**
   * \return AAS broadcast permission
   */
  uint8_t GetAasBdcastPermission (void) const;
  /**
   * \return frame number
   */
  uint32_t GetFrameNumber (void) const;
  /**
   * \return initial range opp number
   */
  uint8_t GetInitRangOppNumber (void) const;
  /**
   * \return range sub channel
   */
  uint8_t GetRangSubchnl (void) const;

  /**
   * \return name string
   */
  std::string GetName (void) const;
  /**
   * Register this type.
   * \return The TypeId.
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_reserved; ///< changed as per the amendment 802.16e-2005

  // TLV Encoded Information

  /**
   * Tx timing offset adjustment (signed 32-bit). The time required to advance SS transmission so frames
   * arrive at the expected time instance at the BS.
   */
  uint32_t m_timingAdjust;

  /**
   * Specifies the relative change in transmission power level that the SS is to make in order that
   * transmissions arrive at the BS at the desired power. When subchannelization is employed, the
   * subscriber shall interpret the power offset adjustment as a required change to the transmitted power
   * density.
   */
  uint8_t m_powerLevelAdjust;

  /**
   * Specifies the relative change in transmission frequency that the SS is to make in order to better match
   * the BS. This is fine-frequency adjustment within a channel, not reassignment to a different channel.
   */
  uint32_t m_offsetFreqAdjust;

  /**
   * range status.
   */
  uint8_t m_rangStatus;

  /// Center frequency, in kHz, of new downlink channel where the SS should redo initial ranging.
  uint32_t m_dlFreqOverride;

  /**
   * Licensed bands: The identifier of the uplink channel with which the SS is to redo initial ranging (not
   * used with PHYs without channelized uplinks).
   */
  uint8_t m_ulChnlIdOverride;

  /**
   * This parameter is sent in response to the RNG-REQ Requested Downlink Burst Profile parameter.
   * Byte 0: Specifies the least robust DIUC that may be used by the BS for transmissions to the SS.
   * Byte 1: Configuration Change Count value of DCD defining the burst profile associated with DIUC.
   */
  uint16_t m_dlOperBurstProfile;

  Mac48Address m_macAddress; ///< MAC address
  Cid m_basicCid; ///< basic CID
  Cid m_primaryCid; ///< primary CID
  uint8_t m_aasBdcastPermission; ///< AAS broadcast permission

  /**
   * Frame number where the associated RNG_REQ message was detected by the BS. Usage is mutually
   * exclusive with SS MAC Address
   */
  uint32_t m_frameNumber;

  /**
   * Initial Ranging opportunity (1–255) in which the associated RNG_REQ message was detected by the BS.
   * Usage is mutually exclusive with SS MAC Address
   */
  uint8_t m_initRangOppNumber;

  /**
   * Used to indicate the OFDM subchannel reference that was used to transmit the initial ranging message
   * (OFDM with subchannelization).
   */
  uint8_t m_rangSubchnl;
};

} // namespace ns3

#endif /* RNG_RSP_H */

// ---------------------------------------------------------------------

#ifndef DSA_REQ_H
#define DSA_REQ_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/buffer.h"
#include "cid.h"
#include "service-flow.h"

namespace ns3 {
class serviceFlow;

/**
 * \ingroup wimax
 * This class implements the DSA-REQ message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.10 DSA-REQ message, page 62
 */
class DsaReq : public Header
{
public:
  DsaReq ();
  ~DsaReq ();
  /**
   * Constructor
   *
   * \param sf service flow
   */
  DsaReq (ServiceFlow sf);
  /**
   * \brief set the transaction ID
   * \param transactionId
   */
  void SetTransactionId (uint16_t transactionId);
  /**
   * \brief set the service flow identifier
   * \param sfid the service flow identifier
   */
  void SetSfid (uint32_t sfid);
  /**
   * \brief set the connection identifier
   * \param cid the connection identifier
   */
  void SetCid (Cid cid);
  /**
   * \brief specify a service flow to be requested by this message
   * \param sf the service flow
   */
  void SetServiceFlow (ServiceFlow sf);
  /**
   * \return the service flow requested by this message
   */
  ServiceFlow GetServiceFlow (void) const;
  /**
   * \return the transaction ID
   */
  uint16_t GetTransactionId (void) const;
  /**
   * \return the service flow identifier
   */
  uint32_t GetSfid (void) const;
  /**
   * \return the connection identifier
   */
  Cid GetCid (void) const;
  /**
   * \return the service name
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);

private:
  uint16_t m_transactionId; ///< transaction ID
  // TLV Encoded Information
  uint32_t m_sfid; ///< SFID
  Cid m_cid; ///< CID
  ServiceFlow m_serviceFlow; ///< service flow
};

} // namespace ns3

#endif /* DSA_REQ_H */

// ---------------------------------------------------------------------

#ifndef DSA_RSP_H
#define DSA_RSP_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/buffer.h"
#include "cid.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the DSA-RSP message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.11 DSA-RSP message, page 63
 *
 * \verbatim
 *   0             7             15            23
 *  +-------------+-------------+-------------+
 *  |Mngt msg type|       Transaction ID      |
 *  +-------------+-------------+-------------+
 *  | Conf Code   | Service Flow TLV          |
 *  +~~~~~~~~~~~~~+~~~~~~~~~~~~~+~~~~~~~~~~~~~+
 * \endverbatim
 */
class DsaRsp : public Header
{
public:
  DsaRsp (void);
  ~DsaRsp (void);

  /**
   * \brief set the transaction ID
   * \param transactionId
   */
  void SetTransactionId (uint16_t transactionId);
  /**
   * \return the transaction ID
   */
  uint16_t GetTransactionId (void) const;

  /**
   * \brief set the confirmation code
   * \param confirmationCode
   */
  void SetConfirmationCode (uint16_t confirmationCode);
  /**
   * \return the confirmation code
   */
  uint16_t GetConfirmationCode (void) const;
  /**
   * \brief set the service flow identifier
   * \param sfid the service flow identifier
   */
  void SetSfid (uint32_t sfid);
  /**
   * \return the service flow identifier
   */
  uint32_t GetSfid (void) const;
  /**
   * \brief set the connection identifier
   * \param cid the connection identifier
   */
  void SetCid (Cid cid);
  /**
   * \return the connection identifier
   */
  Cid GetCid (void) const;
  /**
   * \brief specify a service flow to be requested by this message
   * \param sf the service flow
   */
  void SetServiceFlow (ServiceFlow sf);
  /**
   * \return the service flow requested by this message
   */
  ServiceFlow GetServiceFlow (void) const;

  /**
   * \return the service name
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint16_t m_transactionId; ///< transaction ID
  uint8_t m_confirmationCode; ///< confirmation code
  // TLV Encoded Information
  ServiceFlow m_serviceFlow; ///< service flow
  uint32_t m_sfid; ///< SFID
  Cid m_cid; ///< CID

};

} // namespace ns3

#endif /* DSA_RSP_H */

// ---------------------------------------------------------------------

#ifndef DSA_ACK_H
#define DSA_ACK_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/buffer.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the DSA-ACK message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.12 DSA-ACK message, page 64
 */
class DsaAck : public Header
{

public:
  DsaAck (void);
  ~DsaAck (void);

  /**
   * Set transaction ID field
   * \param transactionId the transaction ID
   */
  void SetTransactionId (uint16_t transactionId);
  /**
   * Get transaction ID field
   * \returns the transaction ID
   */
  uint16_t GetTransactionId (void) const;

  /**
   * Set confirmation code field
   * \param confirmationCode the confirmation code
   */
  void SetConfirmationCode (uint16_t confirmationCode);
  /**
   * Get confirmation code field
   * \returns the confirmation code
   */
  uint16_t GetConfirmationCode (void) const;

  /**
   * Get name field
   * \return the name string
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint16_t m_transactionId; ///< transaction ID 
  uint8_t m_confirmationCode; ///< confirmation code
};

} // namespace ns3

#endif /* DSA_ACK_H */

// ---------------------------------------------------------------------

#ifndef RNG_REQ_H
#define RNG_REQ_H

#include <stdint.h>
#include "ns3/header.h"
#include "ns3/mac48-address.h"
#include "service-flow.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the ranging request message described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 */
class RngReq : public Header
{
public:
  RngReq (void);
  virtual ~RngReq (void);

  /**
   * Set request DL burst profile field
   * \param reqDlBurstProfile the request DL burst profile
   */
  void SetReqDlBurstProfile (uint8_t reqDlBurstProfile);
  /**
   * Set MAC address field
   * \param macAddress the MAC address
   */
  void SetMacAddress (Mac48Address macAddress);
  /**
   * Set ranging anomalies field
   * \param rangingAnomalies the rnaging anomalies
   */
  void SetRangingAnomalies (uint8_t rangingAnomalies);

  /**
   * Get request DL burst profile field
   * \returns the request DL burst profile
   */
  uint8_t GetReqDlBurstProfile (void) const;
  /**
   * Get MAC address field
   * \returns the MAC address
   */
  Mac48Address GetMacAddress (void) const;
  /**
   * Get ranging anomalies field
   * \returns the ranging anomalies
   */
  uint8_t GetRangingAnomalies (void) const;

  /**
   * \brief Get name field
   * \returns the name string
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  /// Print debug function
  void PrintDebug (void) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_reserved; ///< changed as per the amendment 802.16e-2005

  // TLV Encoded Information
  uint8_t m_reqDlBurstProfile; ///< request DL burst profile
  Mac48Address m_macAddress; ///< MAC address
  uint8_t m_rangingAnomalies; ///< ranging anomalies
};

} // namespace ns3

#endif /* RNG_REQ_H */

