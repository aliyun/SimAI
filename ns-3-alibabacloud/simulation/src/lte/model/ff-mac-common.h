/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef FF_MAC_COMMON_H
#define FF_MAC_COMMON_H

#include <ns3/simple-ref-count.h>
#include <ns3/ptr.h>
#include <vector>


/**
 * Constants. See section 4.4
 */
#define MAX_SCHED_CFG_LIST    10
#define MAX_LC_LIST           10

#define MAX_RACH_LIST         30
#define MAX_DL_INFO_LIST      30
#define MAX_BUILD_DATA_LIST   30
#define MAX_BUILD_RAR_LIST    10
#define MAX_BUILD_BC_LIST     3
#define MAX_UL_INFO_LIST      30
#define MAX_DCI_LIST          30
#define MAX_PHICH_LIST        30
#define MAX_TB_LIST           2
#define MAX_RLC_PDU_LIST      30
#define MAX_NR_LCG            4
#define MAX_MBSFN_CONFIG      5
#define MAX_SI_MSG_LIST       32
#define MAX_SI_MSG_SIZE       65535

#define MAX_CQI_LIST          30
#define MAX_UE_SELECTED_SB    6
#define MAX_HL_SB             25
#define MAX_SINR_RB_LIST      100
#define MAX_SR_LIST           30
#define MAX_MAC_CE_LIST       30

namespace ns3 {

/// Result_e enumeration
enum Result_e
{
  SUCCESS,
  FAILURE
};

/// SetupRelease_e enumeration
enum SetupRelease_e
{
  setup,
  release
};

/// CeBitmap_e
enum CeBitmap_e
{
  TA,
  DRX,
  CR
};

/// NormalExtended_e enumeration
enum NormalExtended_e
{
  normal,
  extended
};


/**
 * \brief See section 4.3.1 dlDciListElement
 * \struct DlDciListElement_s
 */
struct DlDciListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  uint32_t  m_rbBitmap {UINT8_MAX}; ///< RB bitmap
  uint8_t   m_rbShift {UINT8_MAX}; ///< RB shift
  uint8_t   m_resAlloc {UINT8_MAX}; ///< The type of resource allocation
  std::vector <uint16_t>  m_tbsSize; ///< The TBs size
  std::vector <uint8_t>   m_mcs; ///< MCS
  std::vector <uint8_t>   m_ndi; ///< New data indicator
  std::vector <uint8_t>   m_rv; ///< Redundancy version
  uint8_t   m_cceIndex {UINT8_MAX}; ///< Control Channel Element index
  uint8_t   m_aggrLevel {UINT8_MAX}; ///< The aggregation level
  uint8_t   m_precodingInfo {UINT8_MAX}; ///< precoding info
  /// Format enumeration
  enum Format_e
  {
    ONE, ONE_A, ONE_B, ONE_C, ONE_D, TWO, TWO_A, TWO_B, NotValid_Dci_Format
  } m_format {NotValid_Dci_Format}; ///< the format
  uint8_t   m_tpc {UINT8_MAX}; ///< Tx power control command
  uint8_t   m_harqProcess {UINT8_MAX}; ///< HARQ process
  uint8_t   m_dai {UINT8_MAX}; ///< DL assignment index
  /// Vrb Format enum
  enum VrbFormat_e
  {
    VRB_DISTRIBUTED,
    VRB_LOCALIZED,
    NotValid_VRB_Format
  } m_vrbFormat {NotValid_VRB_Format}; ///< the format
  bool      m_tbSwap {false}; ///< swap?
  bool      m_spsRelease {false}; ///< release?
  bool      m_pdcchOrder {false}; ///< cch order?
  uint8_t   m_preambleIndex {UINT8_MAX}; ///< preamble index
  uint8_t   m_prachMaskIndex {UINT8_MAX}; ///< RACH mask index
  /// Ngap enum
  enum Ngap_e
  {
    GAP1, GAP2, NotValid_Ngap
  } m_nGap {NotValid_Ngap}; ///< the gap
  uint8_t   m_tbsIdx {UINT8_MAX}; ///< tbs index
  uint8_t   m_dlPowerOffset {UINT8_MAX}; ///< DL power offset
  uint8_t   m_pdcchPowerOffset {UINT8_MAX}; ///<  CCH power offset
};

/**
 * \brief See section 4.3.2 ulDciListElement
 * \struct UlDciListElement_s
 */
struct UlDciListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  uint8_t   m_rbStart {UINT8_MAX}; ///< start
  uint8_t   m_rbLen {UINT8_MAX}; ///< length
  uint16_t  m_tbSize {UINT16_MAX}; ///< size
  uint8_t   m_mcs {UINT8_MAX}; ///< MCS
  uint8_t   m_ndi {UINT8_MAX}; ///< NDI
  uint8_t   m_cceIndex {UINT8_MAX}; ///< Control Channel Element index
  uint8_t   m_aggrLevel {UINT8_MAX}; ///< The aggregation level
  uint8_t   m_ueTxAntennaSelection {UINT8_MAX}; ///< UE antenna selection
  bool      m_hopping {false}; ///< hopping?
  uint8_t   m_n2Dmrs {UINT8_MAX}; ///< n2 DMRS
  int8_t    m_tpc {INT8_MIN}; ///< Tx power control command
  bool      m_cqiRequest {false}; ///< CQI request
  uint8_t   m_ulIndex {UINT8_MAX}; ///< UL index
  uint8_t   m_dai {UINT8_MAX}; ///< DL assignment index
  uint8_t   m_freqHopping {UINT8_MAX}; ///< freq hopping
  int8_t    m_pdcchPowerOffset {INT8_MIN}; ///< CCH power offset
};

/**
* \brief Base class for storing the values of vendor specific parameters
*/
struct VendorSpecificValue : public SimpleRefCount<VendorSpecificValue>
{ 
  virtual ~VendorSpecificValue ();

};

/**
 * \brief See section 4.3.3 vendorSpecifiListElement
 * \struct VendorSpecificListElement_s
 */
struct VendorSpecificListElement_s
{
  uint32_t m_type {UINT32_MAX}; ///< type
  uint32_t m_length {UINT32_MAX}; ///< length
  Ptr<VendorSpecificValue> m_value; ///< value
};

/**
 * \brief See section 4.3.4 logicalChannelConfigListElement
 * \struct LogicalChannelConfigListElement_s
 */
struct LogicalChannelConfigListElement_s
{
  uint8_t   m_logicalChannelIdentity {UINT8_MAX}; ///< logical channel identity
  uint8_t   m_logicalChannelGroup {UINT8_MAX}; ///< logical channel group

  /// Direction enum
  enum Direction_e
  {
    DIR_UL,
    DIR_DL,
    DIR_BOTH,
    NotValid
  } m_direction {NotValid}; ///< the direction

  /// QosBearerType enum
  enum QosBearerType_e
  {
    QBT_NON_GBR,
    QBT_GBR,
    NotValid_QosBearerType
  } m_qosBearerType {NotValid_QosBearerType}; ///< the QOS bearer type

  uint8_t   m_qci {UINT8_MAX}; ///< QCI
  uint64_t  m_eRabMaximulBitrateUl {UINT64_MAX}; ///< ERAB maximum bit rate UL
  uint64_t  m_eRabMaximulBitrateDl {UINT64_MAX}; ///< ERAB maximum bit rate DL
  uint64_t  m_eRabGuaranteedBitrateUl {UINT64_MAX}; ///< ERAB guaranteed bit rate UL
  uint64_t  m_eRabGuaranteedBitrateDl {UINT64_MAX}; ///< ERAB guaranteed bit rate DL
};

/**
 * \brief See section 4.3.6 rachListElement
 * \struct RachListElement_s 
 */
struct RachListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  uint16_t  m_estimatedSize {UINT16_MAX}; ///< estimated size
};

/**
 * \brief See section 4.3.7 phichListElement
 * \struct PhichListElement_s
 */
struct PhichListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  /// Phich enum
  enum Phich_e
  {
    ACK, NACK, NotValid
  } m_phich {NotValid}; ///< the phich
};

/**
 * \brief See section 4.3.9 rlcPDU_ListElement
 */
struct RlcPduListElement_s
{
  uint8_t   m_logicalChannelIdentity {UINT8_MAX}; ///< logical channel identity
  uint16_t  m_size {UINT16_MAX}; ///< size
};

/**
 * \brief See section 4.3.8 builDataListElement
 * \struct BuildDataListElement_s
 */
struct BuildDataListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  struct DlDciListElement_s m_dci; ///< DCI
  std::vector <enum CeBitmap_e> m_ceBitmap; ///< CE bitmap
  std::vector < std::vector <struct RlcPduListElement_s> > m_rlcPduList; ///< RLC PDU list
};

/**
 * \brief Substitutive structure for specifying BuildRarListElement_s::m_grant field
 */
struct UlGrant_s
{
  uint16_t m_rnti {UINT16_MAX}; ///< RNTI
  uint8_t m_rbStart {UINT8_MAX}; ///< start
  uint8_t m_rbLen {UINT8_MAX}; ///< length
  uint16_t m_tbSize {UINT16_MAX}; ///< size
  uint8_t m_mcs {UINT8_MAX}; ///< MCS
  bool m_hopping {false}; ///< hopping?
  int8_t m_tpc {INT8_MIN}; ///< Tx power control command
  bool m_cqiRequest {false}; ///< CQI request?
  bool m_ulDelay {false}; ///< UL delay?
}; 

/**
 * \brief See section 4.3.10 buildRARListElement
 */
struct BuildRarListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  //uint32_t  m_grant; // Substituted with type UlGrant_s
  UlGrant_s m_grant; ///< grant
  struct DlDciListElement_s m_dci; ///< DCI
};

/**
 * \brief See section 4.3.11 buildBroadcastListElement
 */
struct BuildBroadcastListElement_s
{
  /// Type enum
  enum Type_e
  {
    BCCH, PCCH, NotValid
  } m_type {NotValid}; ///< the type
  uint8_t m_index {UINT8_MAX}; ///< index
  struct DlDciListElement_s m_dci; ///< DCI
};

/**
 * \brief See section 4.3.12 ulInfoListElement
 */
struct UlInfoListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  std::vector <uint16_t> m_ulReception; ///< UL reception
  /// Reception status enum
  enum ReceptionStatus_e 
  {
    Ok, NotOk, NotValid
  } m_receptionStatus {NotValid}; ///< the status
  uint8_t   m_tpc {UINT8_MAX}; ///< Tx power control command
};

/**
 * \brief See section 4.3.13 srListElement
 */
struct SrListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
};

/**
 * \brief See section 4.3.15 macCEValue
 */
struct MacCeValue_u
{
  uint8_t   m_phr {UINT8_MAX}; ///< phr
  uint8_t   m_crnti {UINT8_MAX}; ///< NRTI
  std::vector <uint8_t> m_bufferStatus; ///< buffer status
};

/**
 * \brief See section 4.3.14 macCEListElement
 */
struct MacCeListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  /// MAC CE type enum
  enum MacCeType_e
  {
    BSR, PHR, CRNTI, NotValid
  } m_macCeType {NotValid}; ///< MAC CE type
  struct MacCeValue_u m_macCeValue; ///< MAC CE value
};

/**
 * \brief See section 4.3.16 drxConfig
 */
struct DrxConfig_s
{
  uint8_t   m_onDurationTimer {UINT8_MAX}; ///< on duration timer
  uint16_t  m_drxInactivityTimer {UINT16_MAX}; ///< inactivity timer
  uint16_t  m_drxRetransmissionTimer {UINT16_MAX}; ///< retransmission timer
  uint16_t  m_longDrxCycle {UINT16_MAX}; ///< long DRX cycle
  uint16_t  m_longDrxCycleStartOffset {UINT16_MAX}; ///< long DRX cycle start offset
  uint16_t  m_shortDrxCycle {UINT16_MAX}; ///< short DRX cycle
  uint8_t   m_drxShortCycleTimer {UINT8_MAX}; ///< short DRX cycle timer
};

/**
 * \brief See section 4.3.17 spsConfig
 */
struct SpsConfig_s
{
  uint16_t  m_semiPersistSchedIntervalUl {UINT16_MAX}; ///< UL semi persist schedule interval
  uint16_t  m_semiPersistSchedIntervalDl {UINT16_MAX}; ///< DL semi persist schedule interval
  uint8_t   m_numberOfConfSpsProcesses {UINT8_MAX}; ///< number of conf SPS process
  uint8_t   m_n1PucchAnPersistentListSize {UINT8_MAX}; ///< N1pu CCH  persistent list size
  std::vector <uint16_t> m_n1PucchAnPersistentList; ///< N1pu CCH persistent list
  uint8_t   m_implicitReleaseAfter {UINT8_MAX}; ///< implicit release after
};

/**
 * \brief See section 4.3.18 srConfig
 */
struct SrConfig_s
{
  enum SetupRelease_e m_action {setup}; ///< action
  uint8_t   m_schedInterval {UINT8_MAX}; ///< sched interval
  uint8_t   m_dsrTransMax {UINT8_MAX}; ///< trans max
};

/**
 * \brief See section 4.3.19 cqiConfig
 */
struct CqiConfig_s
{
  enum SetupRelease_e m_action {setup}; ///< action
  uint16_t  m_cqiSchedInterval {UINT16_MAX}; ///< CQI schedule interval
  uint8_t   m_riSchedInterval {UINT8_MAX}; ///< RI schedule interval
};

/**
 * \brief See section 4.3.20 ueCapabilities
 */
struct UeCapabilities_s
{
  bool      m_halfDuplex {false}; ///< half duplex
  bool      m_intraSfHopping {false}; ///< intra SF hopping
  bool      m_type2Sb1 {false}; ///< type 2Sb1
  uint8_t   m_ueCategory {UINT8_MAX}; ///< UE category
  bool      m_resAllocType1 {false}; ///< alloc type 1
};

/**
 * \brief See section 4.3.22 siMessageListElement
 */
struct SiMessageListElement_s
{
  uint16_t  m_periodicity {UINT16_MAX}; ///< periodicity
  uint16_t  m_length {UINT16_MAX}; ///< length
};

/**
 * \brief See section 4.3.21 siConfiguration
 */
struct SiConfiguration_s
{
  uint16_t  m_sfn {UINT16_MAX}; ///< sfn
  uint16_t  m_sib1Length {UINT16_MAX}; ///< sib1 length
  uint8_t   m_siWindowLength {UINT8_MAX}; ///< window length
  std::vector <struct SiMessageListElement_s> m_siMessageList; ///< message list
};

/**
 * \brief See section 4.3.23 dlInfoListElement
 */
struct DlInfoListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  uint8_t   m_harqProcessId {UINT8_MAX}; ///< HARQ process ID
  /// HARQ status enum
  enum HarqStatus_e
  {
    ACK, NACK, DTX
  };
  std::vector <enum HarqStatus_e> m_harqStatus; ///< HARQ status
};

/**
 * \brief See section 4.3.28 bwPart
 */
struct BwPart_s
{
  uint8_t   m_bwPartIndex {UINT8_MAX}; ///< bw part index
  uint8_t   m_sb {UINT8_MAX}; ///< sb
  uint8_t   m_cqi {UINT8_MAX}; ///< CQI
};

/**
 * \brief See section 4.3.27 higherLayerSelected
 */
struct HigherLayerSelected_s
{
  uint8_t   m_sbPmi {UINT8_MAX}; ///< sb PMI
  std::vector <uint8_t> m_sbCqi; ///< sb CQI
};

/**
 * \brief See section 4.3.26 ueSelected
 */
struct UeSelected_s
{
  std::vector <uint8_t> m_sbList; ///< sb list
  uint8_t   m_sbPmi {UINT8_MAX}; ///< sb PMI
  std::vector <uint8_t> m_sbCqi; ///< sb CQI
};

/**
 * \brief See section 4.3.25 sbMeasResult
 */
struct SbMeasResult_s
{
  struct UeSelected_s           m_ueSelected; ///< UE selected
  std::vector <struct HigherLayerSelected_s> m_higherLayerSelected; ///< higher layer selected
  struct BwPart_s               m_bwPart; ///< bw part
};

/**
 * \brief See section 4.3.24 cqiListElement
 */
struct CqiListElement_s
{
  uint16_t  m_rnti {UINT16_MAX}; ///< RNTI
  uint8_t   m_ri {UINT8_MAX}; ///< RI
  /// CqiType_e enumeration
  enum CqiType_e
  {
    P10, P11, P20, P21, A12, A22, A20, A30, A31, NotValid
  } m_cqiType {NotValid}; ///< CQI type
  std::vector <uint8_t> m_wbCqi; ///< wb CQI
  uint8_t   m_wbPmi {UINT8_MAX}; ///< wb PMI

  struct SbMeasResult_s m_sbMeasResult; ///< sb measure result
};

/**
 * \brief See section 4.3.29 ulCQI
 */
struct UlCqi_s
{
  std::vector <uint16_t> m_sinr; ///< SINR
  /// Type_e enumeration
  enum Type_e
  {
    SRS,
    PUSCH,
    PUCCH_1,
    PUCCH_2,
    PRACH,
    NotValid
  } m_type {NotValid}; ///< type
};

/**
 * \brief See section 4.3.30 pagingInfoListElement
 */
struct PagingInfoListElement_s
{
  uint8_t   m_pagingIndex {UINT8_MAX}; ///< paging index
  uint16_t  m_pagingMessageSize {UINT16_MAX}; ///< paging message size
  uint8_t   m_pagingSubframe {UINT8_MAX}; ///< paging subframe
};

} // namespace ns3

#endif /* FF_MAC_COMMON_H */
