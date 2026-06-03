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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo  <nbaldo@cttc.es>
 * Modified by:
 *          Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *          Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 */

#ifndef LTE_ENB_MAC_H
#define LTE_ENB_MAC_H


#include <map>
#include <vector>
#include <ns3/lte-common.h>
#include <ns3/lte-mac-sap.h>
#include <ns3/lte-enb-cmac-sap.h>
#include <ns3/ff-mac-csched-sap.h>
#include <ns3/ff-mac-sched-sap.h>
#include <ns3/lte-enb-phy-sap.h>
#include "ns3/traced-value.h"
#include "ns3/trace-source-accessor.h"
#include <ns3/packet.h>
#include <ns3/packet-burst.h>
#include <ns3/lte-ccm-mac-sap.h>

namespace ns3 {

class DlCqiLteControlMessage;
class UlCqiLteControlMessage;
class PdcchMapLteControlMessage;

/// DlHarqProcessesBuffer_t typedef
typedef std::vector <std::vector < Ptr<PacketBurst> > > DlHarqProcessesBuffer_t;

/**
 * This class implements the MAC layer of the eNodeB device
 */
class LteEnbMac :   public Object
{
  /// allow EnbMacMemberLteEnbCmacSapProvider class friend access
  friend class EnbMacMemberLteEnbCmacSapProvider;
  /// allow EnbMacMemberLteMacSapProvider<LteEnbMac> class friend access
  friend class EnbMacMemberLteMacSapProvider<LteEnbMac>;
  /// allow EnbMacMemberFfMacSchedSapUser class friend access
  friend class EnbMacMemberFfMacSchedSapUser;
  /// allow EnbMacMemberFfMacCschedSapUser class friend access
  friend class EnbMacMemberFfMacCschedSapUser;
  /// allow EnbMacMemberLteEnbPhySapUser class friend access
  friend class EnbMacMemberLteEnbPhySapUser;
  /// allow MemberLteCcmMacSapProvider<LteEnbMac> class friend access
  friend class MemberLteCcmMacSapProvider<LteEnbMac>;

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  LteEnbMac (void);
  virtual ~LteEnbMac (void);
  virtual void DoDispose (void);

  /**
   * \brief Set the component carrier ID
   * \param index the component carrier ID
   */
  void SetComponentCarrierId (uint8_t index);
  /**
   * \brief Set the scheduler SAP provider
   * \param s a pointer SAP provider of the FF packet scheduler
   */
  void SetFfMacSchedSapProvider (FfMacSchedSapProvider* s);
  /**
   * \brief Get the scheduler SAP user
   * \return a pointer to the SAP user of the scheduler
   */
  FfMacSchedSapUser* GetFfMacSchedSapUser (void);
  /**
   * \brief Set the control scheduler SAP provider
   * \param s a pointer to the control scheduler SAP provider
   */
  void SetFfMacCschedSapProvider (FfMacCschedSapProvider* s);
  /**
   * \brief Get the control scheduler SAP user
   * \return a pointer to the control scheduler SAP user
   */
  FfMacCschedSapUser* GetFfMacCschedSapUser (void);



  /**
   * \brief Set the MAC SAP user
   * \param s a pointer to the MAC SAP user
   */
  void SetLteMacSapUser (LteMacSapUser* s);
  /**
   * \brief Get the MAC SAP provider
   * \return a pointer to the SAP provider of the MAC
   */
  LteMacSapProvider* GetLteMacSapProvider (void);
  /**
   * \brief Set the control MAC SAP user
   * \param s a pointer to the control MAC SAP user
   */
  void SetLteEnbCmacSapUser (LteEnbCmacSapUser* s);
  /**
   * \brief Get the control MAC SAP provider
   * \return a pointer to the control MAC SAP provider
   */
  LteEnbCmacSapProvider* GetLteEnbCmacSapProvider (void);


  /**
  * \brief Get the eNB-PHY SAP User
  * \return a pointer to the SAP User of the PHY
  */
  LteEnbPhySapUser* GetLteEnbPhySapUser ();

  /**
  * \brief Set the PHY SAP Provider
  * \param s a pointer to the PHY SAP provider
  */
  void SetLteEnbPhySapProvider (LteEnbPhySapProvider* s);

  /**
  * \brief Get the eNB-ComponentCarrierManager SAP User
  * \return a pointer to the SAP User of the ComponentCarrierManager
  */
  LteCcmMacSapProvider* GetLteCcmMacSapProvider ();

  /**
  * \brief Set the ComponentCarrierManager SAP user
  * \param s a pointer to the ComponentCarrierManager provider
  */
  void SetLteCcmMacSapUser (LteCcmMacSapUser* s);
  

  /**
   * TracedCallback signature for DL scheduling events.
   *
   * \param [in] frame Frame number.
   * \param [in] subframe Subframe number.
   * \param [in] rnti The C-RNTI identifying the UE.
   * \param [in] mcs0 The MCS for transport block.. 
   * \param [in] tbs0Size
   * \param [in] mcs1 The MCS for transport block.
   * \param [in] tbs1Size
   * \param [in] component carrier id
   */
  typedef void (* DlSchedulingTracedCallback)
    (const uint32_t frame, const uint32_t subframe, const uint16_t rnti,
     const uint8_t mcs0, const uint16_t tbs0Size,
     const uint8_t mcs1, const uint16_t tbs1Size, const uint8_t ccId);

  /**
   *  TracedCallback signature for UL scheduling events.
   *
   * \param [in] frame Frame number.
   * \param [in] subframe Subframe number.
   * \param [in] rnti The C-RNTI identifying the UE.
   * \param [in] mcs  The MCS for transport block
   * \param [in] tbsSize
   */
  typedef void (* UlSchedulingTracedCallback)
    (const uint32_t frame, const uint32_t subframe, const uint16_t rnti,
     const uint8_t mcs, const uint16_t tbsSize);
  
private:

  /**
  * \brief Receive a DL CQI ideal control message
  * \param msg the DL CQI message
  */
  void ReceiveDlCqiLteControlMessage  (Ptr<DlCqiLteControlMessage> msg);

  /**
  * \brief Receive a DL CQI ideal control message
  * \param msg the DL CQI message
  */
  void DoReceiveLteControlMessage (Ptr<LteControlMessage> msg);

  /**
  * \brief Receive a CE element containing the buffer status report
  * \param bsr the BSR message
  */
  void ReceiveBsrMessage  (MacCeListElement_s bsr);

  /**
  * \brief UL CQI report
  * \param ulcqi FfMacSchedSapProvider::SchedUlCqiInfoReqParameters
  */
  void DoUlCqiReport (FfMacSchedSapProvider::SchedUlCqiInfoReqParameters ulcqi);



  // forwarded from LteEnbCmacSapProvider
  /**
  * \brief Configure MAC function
  * \param ulBandwidth the UL bandwidth
  * \param dlBandwidth the DL bandwidth
  */
  void DoConfigureMac (uint16_t ulBandwidth, uint16_t dlBandwidth);
  /**
  * \brief Add UE function
  * \param rnti the RNTI
  */
  void DoAddUe (uint16_t rnti);
  /**
  * \brief Remove UE function
  * \param rnti the RNTI
  */
  void DoRemoveUe (uint16_t rnti);
  /**
  * \brief Add LC function
  * \param lcinfo the LC info
  * \param msu the LTE MAC SAP user
  */
  void DoAddLc (LteEnbCmacSapProvider::LcInfo lcinfo, LteMacSapUser* msu);
  /**
  * \brief Reconfigure LC function
  * \param lcinfo the LC info
  */
  void DoReconfigureLc (LteEnbCmacSapProvider::LcInfo lcinfo);
  /**
  * \brief Release LC function
  * \param rnti the RNTI
  * \param lcid the LCID
  */
  void DoReleaseLc (uint16_t  rnti, uint8_t lcid);
  /**
  * \brief UE Update configuration request function
  * \param params LteEnbCmacSapProvider::UeConfig
  */
  void DoUeUpdateConfigurationReq (LteEnbCmacSapProvider::UeConfig params);
  /**
  * \brief Get RACH configuration function
  * \returns LteEnbCmacSapProvider::RachConfig
  */
  LteEnbCmacSapProvider::RachConfig DoGetRachConfig ();
  /**
  * \brief Allocate NC RA preamble function
  * \param rnti the RNTI
  * \returns LteEnbCmacSapProvider::AllocateNcRaPreambleReturnValue
  */
  LteEnbCmacSapProvider::AllocateNcRaPreambleReturnValue DoAllocateNcRaPreamble (uint16_t rnti);

  // forwarded from LteMacSapProvider
  /**
  * \brief Transmit PDU function
  * \param params LteMacSapProvider::TransmitPduParameters
  */
  void DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params);
  /**
  * \brief Report Buffer Status function
  * \param params LteMacSapProvider::ReportBufferStatusParameters
  */
  void DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params);


  // forwarded from FfMacCchedSapUser
  /**
  * \brief CSched Cell Config configure function
  * \param params FfMacCschedSapUser::CschedCellConfigCnfParameters
  */
  void DoCschedCellConfigCnf (FfMacCschedSapUser::CschedCellConfigCnfParameters params);
  /**
  * \brief CSched UE Config configure function
  * \param params FfMacCschedSapUser::CschedUeConfigCnfParameters
  */
  void DoCschedUeConfigCnf (FfMacCschedSapUser::CschedUeConfigCnfParameters params);
  /**
  * \brief CSched LC Config configure function
  * \param params FfMacCschedSapUser::CschedLcConfigCnfParameters
  */
  void DoCschedLcConfigCnf (FfMacCschedSapUser::CschedLcConfigCnfParameters params);
  /**
  * \brief CSched LC Release configure function
  * \param params FfMacCschedSapUser::CschedLcReleaseCnfParameters
  */
  void DoCschedLcReleaseCnf (FfMacCschedSapUser::CschedLcReleaseCnfParameters params);
  /**
  * \brief CSched UE Release configure function
  * \param params FfMacCschedSapUser::CschedUeReleaseCnfParameters
  */
  void DoCschedUeReleaseCnf (FfMacCschedSapUser::CschedUeReleaseCnfParameters params);
  /**
  * \brief CSched UE Config Update Indication function
  * \param params FfMacCschedSapUser::CschedUeConfigUpdateIndParameters
  */
  void DoCschedUeConfigUpdateInd (FfMacCschedSapUser::CschedUeConfigUpdateIndParameters params);
  /**
  * \brief CSched Cell Config Update Indication function
  * \param params FfMacCschedSapUser::CschedCellConfigUpdateIndParameters
  */
  void DoCschedCellConfigUpdateInd (FfMacCschedSapUser::CschedCellConfigUpdateIndParameters params);

  // forwarded from FfMacSchedSapUser
  /**
  * \brief Sched DL Config Indication function
  * \param ind FfMacSchedSapUser::SchedDlConfigIndParameters
  */
  void DoSchedDlConfigInd (FfMacSchedSapUser::SchedDlConfigIndParameters ind);
  /**
  * \brief Sched UL Config Indication function
  * \param params FfMacSchedSapUser::SchedUlConfigIndParameters
  */
  void DoSchedUlConfigInd (FfMacSchedSapUser::SchedUlConfigIndParameters params);

  // forwarded from LteEnbPhySapUser
  /**
  * \brief Subrame Indication function
  * \param frameNo frame number
  * \param subframeNo subframe number
  */
  void DoSubframeIndication (uint32_t frameNo, uint32_t subframeNo);
  /**
  * \brief Receive RACH Preamble function
  * \param prachId PRACH ID number
  */
  void DoReceiveRachPreamble (uint8_t prachId);

  // forwarded by LteCcmMacSapProvider
  /**
   * Report MAC CE to scheduler
   * \param bsr the BSR
   */
  void DoReportMacCeToScheduler (MacCeListElement_s bsr);

  /**
   * \brief Report SR to scheduler
   * \param rnti RNTI of the UE that requested the SR
   *
   * Since SR is not implemented in LTE, this method does nothing.
   */
  void DoReportSrToScheduler ([[maybe_unused]] uint16_t rnti)
  {
  }
  
public:
  /**
   * legacy public for use the Phy callback
   * \param p packet
   */
  void DoReceivePhyPdu (Ptr<Packet> p);

private:
  /**
  * \brief UL Info List ELements HARQ Feedback function
  * \param params UlInfoListElement_s
  */
  void DoUlInfoListElementHarqFeeback (UlInfoListElement_s params);
  /**
  * \brief DL Info List ELements HARQ Feedback function
  * \param params DlInfoListElement_s
  */
  void DoDlInfoListElementHarqFeeback (DlInfoListElement_s params);

  /// RNTI, LC ID, SAP of the RLC instance
  std::map <uint16_t, std::map<uint8_t, LteMacSapUser*> > m_rlcAttached;

  std::vector <CqiListElement_s> m_dlCqiReceived; ///< DL-CQI received
  std::vector <FfMacSchedSapProvider::SchedUlCqiInfoReqParameters> m_ulCqiReceived; ///< UL-CQI received
  std::vector <MacCeListElement_s> m_ulCeReceived; ///< CE received (BSR up to now)

  std::vector <DlInfoListElement_s> m_dlInfoListReceived; ///< DL HARQ feedback received

  std::vector <UlInfoListElement_s> m_ulInfoListReceived; ///< UL HARQ feedback received


  /*
  * Map of UE's info element (see 4.3.12 of FF MAC Scheduler API)
  */
  //std::map <uint16_t,UlInfoListElement_s> m_ulInfoListElements; 



  LteMacSapProvider* m_macSapProvider; ///< the MAC SAP provider
  LteEnbCmacSapProvider* m_cmacSapProvider; ///< the CMAC SAP provider
  LteMacSapUser* m_macSapUser; ///< the MAC SAP user
  LteEnbCmacSapUser* m_cmacSapUser; ///< the CMAC SAP user


  FfMacSchedSapProvider* m_schedSapProvider; ///< the Sched SAP provider
  FfMacCschedSapProvider* m_cschedSapProvider; ///< the Csched SAP provider
  FfMacSchedSapUser* m_schedSapUser; ///< the Sched SAP user
  FfMacCschedSapUser* m_cschedSapUser; ///< the CSched SAP user

  // PHY-SAP
  LteEnbPhySapProvider* m_enbPhySapProvider; ///< the ENB Phy SAP provider
  LteEnbPhySapUser* m_enbPhySapUser; ///< the ENB Phy SAP user

  // Sap For ComponentCarrierManager 'Uplink case'
  LteCcmMacSapProvider* m_ccmMacSapProvider; ///< CCM MAC SAP provider
  LteCcmMacSapUser* m_ccmMacSapUser; ///< CCM MAC SAP user
  /**
   * frame number of current subframe indication
   */
  uint32_t m_frameNo;
  /**
   * subframe number of current subframe indication
   */
  uint32_t m_subframeNo;
  /**
   * Trace information regarding DL scheduling
   * Frame number, Subframe number, RNTI, MCS of TB1, size of TB1,
   * MCS of TB2 (0 if not present), size of TB2 (0 if not present)
   */
  TracedCallback<DlSchedulingCallbackInfo> m_dlScheduling;

  /**
   * Trace information regarding UL scheduling
   * Frame number, Subframe number, RNTI, MCS of TB, size of TB, component carrier id
   */
  TracedCallback<uint32_t, uint32_t, uint16_t,
                 uint8_t, uint16_t, uint8_t> m_ulScheduling;
  
  uint8_t m_macChTtiDelay; ///< delay of MAC, PHY and channel in terms of TTIs


  std::map <uint16_t, DlHarqProcessesBuffer_t> m_miDlHarqProcessesPackets; ///< Packet under transmission of the DL HARQ process
  
  uint8_t m_numberOfRaPreambles; ///< number of RA preambles
  uint8_t m_preambleTransMax; ///< preamble transmit maximum
  uint8_t m_raResponseWindowSize; ///< RA response window size
  uint8_t m_connEstFailCount; ///< the counter value for T300 timer expiration

  /**
   * info associated with a preamble allocated for non-contention based RA
   * 
   */
  struct NcRaPreambleInfo
  {   
    uint16_t rnti; ///< rnti previously allocated for this non-contention based RA procedure
    Time expiryTime; ///< value the expiration time of this allocation (so that stale preambles can be reused)
  };

  /**
   * map storing as key the random access preamble IDs allocated for
   * non-contention based access, and as value the associated info
   * 
   */
  std::map<uint8_t, NcRaPreambleInfo> m_allocatedNcRaPreambleMap;
 
  std::map<uint8_t, uint32_t> m_receivedRachPreambleCount; ///< received RACH preamble count

  std::map<uint16_t, uint32_t> m_rapIdRntiMap; ///< RAPID RNTI map

  /// component carrier Id used to address sap
  uint8_t m_componentCarrierId;
 
};

} // end namespace ns3

#endif /* LTE_ENB_MAC_ENTITY_H */
