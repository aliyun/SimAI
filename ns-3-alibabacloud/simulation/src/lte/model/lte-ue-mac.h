/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo  <nbaldo@cttc.es>
 * Author: Marco Miozzo <mmiozzo@cttc.es>
 */

#ifndef LTE_UE_MAC_ENTITY_H
#define LTE_UE_MAC_ENTITY_H



#include <map>

#include <ns3/lte-mac-sap.h>
#include <ns3/lte-ue-cmac-sap.h>
#include <ns3/lte-ue-phy-sap.h>
#include <ns3/nstime.h>
#include <ns3/event-id.h>
#include <vector>
#include <ns3/packet.h>
#include <ns3/packet-burst.h>
#include <ns3/traced-callback.h>


namespace ns3 {

class UniformRandomVariable;

class LteUeMac :   public Object
{
  /// allow UeMemberLteUeCmacSapProvider class friend access
  friend class UeMemberLteUeCmacSapProvider;
  /// allow UeMemberLteMacSapProvider class friend access
  friend class UeMemberLteMacSapProvider;
  /// allow UeMemberLteUePhySapUser class friend access
  friend class UeMemberLteUePhySapUser;

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LteUeMac ();
  virtual ~LteUeMac ();
  virtual void DoDispose (void);

  /**
   * \brief TracedCallback signature for RA response timeout events
   * exporting IMSI, contention flag, preamble transmission counter
   * and the max limit of preamble transmission
   *
   * \param [in] imsi
   * \param [in] contention
   * \param [in] preambleTxCounter
   * \param [in] maxPreambleTxLimit
   */
  typedef void (* RaResponseTimeoutTracedCallback)
    (uint64_t imsi, bool contention, uint8_t preambleTxCounter, uint8_t maxPreambleTxLimit);

  /**
  * \brief Get the LTE MAC SAP provider
  * \return a pointer to the LTE MAC SAP provider
  */
  LteMacSapProvider*  GetLteMacSapProvider (void);
  /**
  * \brief Set the LTE UE CMAC SAP user
  * \param s the LTE UE CMAC SAP User
  */
  void  SetLteUeCmacSapUser (LteUeCmacSapUser* s);
  /**
  * \brief Get the LTE CMAC SAP provider
  * \return a pointer to the LTE CMAC SAP provider
  */
  LteUeCmacSapProvider*  GetLteUeCmacSapProvider (void);
  
  /**
  * \brief Set the component carried ID
  * \param index the component carrier ID
  */
  void SetComponentCarrierId (uint8_t index);

  /**
  * \brief Get the PHY SAP user
  * \return a pointer to the SAP user of the PHY
  */
  LteUePhySapUser* GetLteUePhySapUser ();

  /**
  * \brief Set the PHY SAP Provider
  * \param s a pointer to the PHY SAP Provider
  */
  void SetLteUePhySapProvider (LteUePhySapProvider* s);
  
  /**
  * \brief Forwarded from LteUePhySapUser: trigger the start from a new frame
  *
  * \param frameNo frame number
  * \param subframeNo subframe number
  */
  void DoSubframeIndication (uint32_t frameNo, uint32_t subframeNo);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

private:
  // forwarded from MAC SAP
 /**
  * Transmit PDU function
  *
  * \param params LteMacSapProvider::TransmitPduParameters
  */
  void DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params);
 /**
  * Report buffers status function
  *
  * \param params LteMacSapProvider::ReportBufferStatusParameters
  */
  void DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params);

  // forwarded from UE CMAC SAP
 /**
  * Configure RACH function
  *
  * \param rc LteUeCmacSapProvider::RachConfig
  */
  void DoConfigureRach (LteUeCmacSapProvider::RachConfig rc);
 /**
  * Start contention based random access procedure function
  */
  void DoStartContentionBasedRandomAccessProcedure ();
 /**
  * Set RNTI
  *
  * \param rnti the RNTI
  */
  void DoSetRnti (uint16_t rnti);
 /**
  * Start non contention based random access procedure function
  *
  * \param rnti the RNTI
  * \param rapId the RAPID
  * \param prachMask the PRACH mask
  */
  void DoStartNonContentionBasedRandomAccessProcedure (uint16_t rnti, uint8_t rapId, uint8_t prachMask);
 /**
  * Add LC function
  *
  * \param lcId the LCID
  * \param lcConfig the logical channel config
  * \param msu the MSU
  */
  void DoAddLc (uint8_t lcId, LteUeCmacSapProvider::LogicalChannelConfig lcConfig, LteMacSapUser* msu);
 /**
  * Remove LC function
  *
  * \param lcId the LCID
  */
  void DoRemoveLc (uint8_t lcId);
  /**
   * \brief Reset function
   */
  void DoReset ();
  /**
   * \brief Notify MAC about the successful RRC connection
   * establishment.
   */
  void DoNotifyConnectionSuccessful ();
  /**
   * Set IMSI
   *
   * \param imsi the IMSI of the UE
   */
  void DoSetImsi (uint64_t imsi);

  // forwarded from PHY SAP
 /**
  * Receive Phy PDU function
  *
  * \param p the packet
  */
  void DoReceivePhyPdu (Ptr<Packet> p);
 /**
  * Receive LTE control message function
  *
  * \param msg the LTE control message
  */
  void DoReceiveLteControlMessage (Ptr<LteControlMessage> msg);
  
  // internal methods
  /// Randomly select and send RA preamble function
  void RandomlySelectAndSendRaPreamble ();
 /**
  * Send RA preamble function
  *
  * \param contention if true randomly select and send the RA preamble
  */
  void SendRaPreamble (bool contention);
  /// Start waiting for RA response function
  void StartWaitingForRaResponse ();
 /**
  * Receive the RA response function
  *
  * \param raResponse RA response received
  */
  void RecvRaResponse (BuildRarListElement_s raResponse);
 /**
  * RA response timeout function
  *
  * \param contention if true randomly select and send the RA preamble
  */
  void RaResponseTimeout (bool contention);
  /// Send report buffer status
  void SendReportBufferStatus (void);
  /// Refresh HARQ processes packet buffer function
  void RefreshHarqProcessesPacketBuffer (void);

  /// component carrier Id --> used to address sap
  uint8_t m_componentCarrierId;

private:

  /// LcInfo structure
  struct LcInfo
  {
    LteUeCmacSapProvider::LogicalChannelConfig lcConfig; ///< logical channel config
    LteMacSapUser* macSapUser; ///< MAC SAP user
  };

  std::map <uint8_t, LcInfo> m_lcInfoMap; ///< logical channel info map

  LteMacSapProvider* m_macSapProvider; ///< MAC SAP provider

  LteUeCmacSapUser* m_cmacSapUser; ///< CMAC SAP user
  LteUeCmacSapProvider* m_cmacSapProvider; ///< CMAC SAP provider

  LteUePhySapProvider* m_uePhySapProvider; ///< UE Phy SAP provider
  LteUePhySapUser* m_uePhySapUser; ///< UE Phy SAP user
  
  std::map <uint8_t, LteMacSapProvider::ReportBufferStatusParameters> m_ulBsrReceived; ///< BSR received from RLC (the last one)
  
  
  Time m_bsrPeriodicity; ///< BSR periodicity
  Time m_bsrLast; ///< BSR last
  
  bool m_freshUlBsr; ///< true when a BSR has been received in the last TTI

  uint8_t m_harqProcessId; ///< HARQ process ID
  std::vector < Ptr<PacketBurst> > m_miUlHarqProcessesPacket; ///< Packets under transmission of the UL HARQ processes
  std::vector < uint8_t > m_miUlHarqProcessesPacketTimer; ///< timer for packet life in the buffer

  uint16_t m_rnti; ///< RNTI
  uint16_t m_imsi; ///< IMSI

  bool m_rachConfigured; ///< is RACH configured?
  LteUeCmacSapProvider::RachConfig m_rachConfig; ///< RACH configuration
  uint8_t m_raPreambleId; ///< RA preamble ID
  uint8_t m_preambleTransmissionCounter; ///< preamble tranamission counter
  uint16_t m_backoffParameter; ///< backoff parameter
  EventId m_noRaResponseReceivedEvent; ///< no RA response received event ID
  Ptr<UniformRandomVariable> m_raPreambleUniformVariable; ///< RA preamble random variable

  uint32_t m_frameNo; ///< frame number
  uint32_t m_subframeNo; ///< subframe number
  uint8_t m_raRnti; ///< RA RNTI
  bool m_waitingForRaResponse; ///< waiting for RA response

  /**
   * \brief The `RaResponseTimeout` trace source. Fired RA response timeout.
   * Exporting IMSI, contention flag, preamble transmission counter
   * and the max limit of preamble transmission.
   */
  TracedCallback<uint64_t, bool, uint8_t, uint8_t> m_raResponseTimeoutTrace;
};

} // namespace ns3

#endif // LTE_UE_MAC_ENTITY
