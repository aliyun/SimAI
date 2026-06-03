/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 * Copyright (c) 2018 Fraunhofer ESK : RLF extensions
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 *         Budiarto Herman <budiarto.herman@magister.fi>
 * Modified by:
 *          Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *          Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 *          Vignesh Babu <ns3-dev@esk.fraunhofer.de> (RLF extensions)
 */

#ifndef LTE_UE_RRC_H
#define LTE_UE_RRC_H

#include <ns3/object.h>
#include <ns3/packet.h>
#include <ns3/lte-ue-cmac-sap.h>
#include <ns3/lte-pdcp-sap.h>
#include <ns3/lte-as-sap.h>
#include <ns3/lte-ue-cphy-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/traced-callback.h>
#include "ns3/component-carrier-ue.h"
#include <ns3/lte-ue-ccm-rrc-sap.h>
#include <vector>

#include <map>
#include <set>

#define MIN_NO_CC 1
#define MAX_NO_CC 5 // this is the maximum number of carrier components allowed by 3GPP up to R13

namespace ns3 {


/**
 * \brief Artificial delay of UE measurements procedure.
 *
 * i.e. the period between the time layer-1-filtered measurements from PHY
 * layer is received and the earliest time the actual measurement report
 * submission to the serving cell is invoked.
 *
 * This delay exists because of racing condition between several UE measurements
 * functions which happen to be scheduled at the same time. The delay ensures
 * that:
 *  - measurements (e.g., layer-3 filtering) are always performed before
 *    reporting, thus the latter always use the latest measured RSRP and RSRQ;
 *    and
 *  - time-to-trigger check is always performed before the reporting, so there
 *    would still be chance for it to cancel the reporting if necessary.
 */
static const Time UE_MEASUREMENT_REPORT_DELAY = MicroSeconds (1);


class LteRlc;
class LteMacSapProvider;
class LteUeCmacSapUser;
class LteUeCmacSapProvider;
class LteDataRadioBearerInfo;
class LteSignalingRadioBearerInfo;

/**
 *
 *
 */
class LteUeRrc : public Object
{

  /// allow UeMemberLteUeCmacSapUser class friend access
  friend class UeMemberLteUeCmacSapUser;
  /// allow UeRrcMemberLteEnbCmacSapUser class friend access
  friend class UeRrcMemberLteEnbCmacSapUser;
  /// allow LtePdcpSpecificLtePdcpSapUser<LteUeRrc> class friend access
  friend class LtePdcpSpecificLtePdcpSapUser<LteUeRrc>;
  /// allow MemberLteAsSapProvider<LteUeRrc> class friend access
  friend class MemberLteAsSapProvider<LteUeRrc>;
  /// allow MemberLteUeCphySapUser<LteUeRrc> class friend access
  friend class MemberLteUeCphySapUser<LteUeRrc>;
  /// allow MemberLteUeRrcSapProvider<LteUeRrc> class friend access
  friend class MemberLteUeRrcSapProvider<LteUeRrc>;
  /// allow MemberLteUeCcmRrcSapUser<LteUeRrc> class friend access
  friend class MemberLteUeCcmRrcSapUser<LteUeRrc>;

public:

  /**
   * The states of the UE RRC entity
   *
   */
  enum State
  {
    IDLE_START = 0,
    IDLE_CELL_SEARCH,
    IDLE_WAIT_MIB_SIB1,
    IDLE_WAIT_MIB,
    IDLE_WAIT_SIB1,
    IDLE_CAMPED_NORMALLY,
    IDLE_WAIT_SIB2,
    IDLE_RANDOM_ACCESS,
    IDLE_CONNECTING,
    CONNECTED_NORMALLY,
    CONNECTED_HANDOVER,
    CONNECTED_PHY_PROBLEM,
    CONNECTED_REESTABLISHING,
    NUM_STATES
  };


  /**
   * create an RRC instance for use within an ue
   *
   */
  LteUeRrc ();


  /**
   * Destructor
   */
  virtual ~LteUeRrc ();


  // inherited from Object
private:
  virtual void DoInitialize (void);
  virtual void DoDispose (void);
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /// Initiaize SAP
  void InitializeSap (void);

  /**
   * set the CPHY SAP this RRC should use to interact with the PHY
   *
   * \param s the CPHY SAP Provider
   */
  void SetLteUeCphySapProvider (LteUeCphySapProvider * s);
  /**
   * set the CPHY SAP this RRC should use to interact with the PHY
   *
   * \param s the CPHY SAP Provider
   * \param index the index
   */
  void SetLteUeCphySapProvider (LteUeCphySapProvider * s, uint8_t index);

  /**
   *
   *
   * \return s the CPHY SAP User interface offered to the PHY by this RRC
   */
  LteUeCphySapUser* GetLteUeCphySapUser ();
  /**
   *
   * \param index the index
   * \return s the CPHY SAP User interface offered to the PHY by this RRC
   */
  LteUeCphySapUser* GetLteUeCphySapUser (uint8_t index);

  /**
   * set the CMAC SAP this RRC should interact with
   * \brief This function is overloaded to maintain backward compatibility
   * \param s the CMAC SAP Provider to be used by this RRC
   */
  void SetLteUeCmacSapProvider (LteUeCmacSapProvider * s);
  /**
   * set the CMAC SAP this RRC should interact with
   * \brief This function is overloaded to maintain backward compatibility
   * \param s the CMAC SAP Provider to be used by this RRC
   * \param index the index
   */
  void SetLteUeCmacSapProvider (LteUeCmacSapProvider * s, uint8_t index);

  /**
   * \brief This function is overloaded to maintain backward compatibility
   * \return s the CMAC SAP User interface offered to the MAC by this RRC
   */
  LteUeCmacSapUser* GetLteUeCmacSapUser ();
  /**
   * \brief This function is overloaded to maintain backward compatibility
   * \param index the index
   * \return s the CMAC SAP User interface offered to the MAC by this RRC
   */
  LteUeCmacSapUser* GetLteUeCmacSapUser (uint8_t index);


  /**
   * set the RRC SAP this RRC should interact with
   *
   * \param s the RRC SAP User to be used by this RRC
   */
  void SetLteUeRrcSapUser (LteUeRrcSapUser * s);

  /**
   *
   *
   * \return s the RRC SAP Provider interface offered to the MAC by this RRC
   */
  LteUeRrcSapProvider* GetLteUeRrcSapProvider ();

  /**
   * set the MAC SAP provider. The ue RRC does not use this
   * directly, but it needs to provide it to newly created RLC instances.
   *
   * \param s the MAC SAP provider that will be used by all
   * newly created RLC instances
   */
  void SetLteMacSapProvider (LteMacSapProvider* s);

  /**
   * Set the AS SAP user to interact with the NAS entity
   *
   * \param s the AS SAP user
   */
  void SetAsSapUser (LteAsSapUser* s);

  /**
   *
   *
   * \return the AS SAP provider exported by this RRC
   */
  LteAsSapProvider* GetAsSapProvider ();

  /**
   * set the Component Carrier Management SAP this RRC should interact with
   *
   * \param s the Component Carrier Management SAP Provider to be used by this RRC
   */
  void SetLteCcmRrcSapProvider (LteUeCcmRrcSapProvider * s);

  /**
   * Get the Component Carrier Management SAP offered by this RRC
   * \return s the Component Carrier Management SAP User interface offered to the
   *           carrier component selection algorithm by this RRC
   */
  LteUeCcmRrcSapUser* GetLteCcmRrcSapUser ();

  /**
   *
   * \param imsi the unique UE identifier
   */
  void SetImsi (uint64_t imsi);

  /**
   * \brief Store the previous cell id
   *
   * \param cellId The cell id of the previous cell the UE was attached to
   */
  void StorePreviousCellId (uint16_t cellId);

  /**
   *
   * \return imsi the unique UE identifier
   */
  uint64_t GetImsi (void) const;

  /**
   *
   * \return the C-RNTI of the user
   */
  uint16_t GetRnti () const;

  /**
   *
   * \return the CellId of the attached Enb
   */
  uint16_t GetCellId () const;

  /**
   * \param cellId cell identifier
   * \return true if cellId is the serving cell for this UE
   */
  bool IsServingCell (uint16_t cellId) const;

  /**
   * \return the uplink bandwidth in RBs
   */
  uint8_t GetUlBandwidth () const;

  /**
   * \return the downlink bandwidth in RBs
   */
  uint8_t GetDlBandwidth () const;

  /**
   * \return the downlink carrier frequency (EARFCN)
   */
  uint32_t GetDlEarfcn () const;

  /**
   * \return the uplink carrier frequency (EARFCN)
   */
  uint32_t GetUlEarfcn () const;

  /**
   *
   * \return the current state
   */
  State GetState () const;

  /**
   * \brief Get the previous cell id
   *
   * \return The cell Id of the previous cell the UE was attached to.
   */
  uint16_t GetPreviousCellId () const;

  /**
   *
   *
   * \param val true if RLC SM is to be used, false if RLC UM/AM are to be used
   */
  void SetUseRlcSm (bool val);

  /**
   * \param s The UE RRC state.
   * \return The string representation of the given state.
   */
  static const std::string ToString (LteUeRrc::State s);

  /**
   * TracedCallback signature for imsi, cellId and rnti events.
   *
   * \param [in] imsi
   * \param [in] cellId
   */
  typedef void (* CellSelectionTracedCallback)
    (uint64_t imsi, uint16_t cellId);

  /**
   * TracedCallback signature for imsi, cellId and rnti events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   */
  typedef void (* ImsiCidRntiTracedCallback)
    (uint64_t imsi, uint16_t cellId, uint16_t rnti);

  /**
   * TracedCallback signature for MIBRecieved, Sib1Received and
   * HandoverStart events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] otherCid
   */
  typedef void (* MibSibHandoverTracedCallback)
    (uint64_t imsi, uint16_t cellId, uint16_t rnti, uint16_t otherCid);

  /**
   * TracedCallback signature for state transition events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] oldState
   * \param [in] newState
   */
  typedef void (* StateTracedCallback)
    (uint64_t imsi, uint16_t cellId, uint16_t rnti,
     State oldState, State newState);

  /**
    * TracedCallback signature for secondary carrier configuration events.
    *
    * \param [in] Pointer to UE RRC
    * \param [in] List of LteRrcSap::SCellToAddMod
    */
  typedef void (* SCarrierConfiguredTracedCallback)
    (Ptr<LteUeRrc>, std::list<LteRrcSap::SCellToAddMod>);

  /**
   * TracedCallback signature for in-sync and out-of-sync detection events.
   *
   *
   * \param [in] imsi
   * \param [in] rnti
   * \param [in] cellId
   * \param [in] type
   * \param [in] count
   */
  typedef void (*PhySyncDetectionTracedCallback)
      (uint64_t imsi, uint16_t rnti, uint16_t cellId, std::string type, uint16_t count);

  /**
   * TracedCallback signature for imsi, cellId, rnti and counter for
   * random access events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] count
   */
  typedef void (* ImsiCidRntiCountTracedCallback)
  (uint64_t imsi, uint16_t cellId, uint16_t rnti, uint8_t count);


private:


  // PDCP SAP methods
  /**
   * Receive PDCP SDU function
   *
   * \param params LtePdcpSapUser::ReceivePdcpSduParameters
   */
  void DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params);

  // CMAC SAP methods
  /**
   * Set temporary cell rnti function
   *
   * \param rnti RNTI
   */
  void DoSetTemporaryCellRnti (uint16_t rnti);
  /// Notify random access successful function
  void DoNotifyRandomAccessSuccessful ();
  /// Notify random access failed function
  void DoNotifyRandomAccessFailed ();

  // LTE AS SAP methods
  /**
   * Set CSG white list function
   *
   * \param csgId CSG ID
   */
  void DoSetCsgWhiteList (uint32_t csgId);
  /**
   * Force camped on ENB function
   *
   * \param cellId the cell ID
   * \param dlEarfcn the DL EARFCN
   */
  void DoForceCampedOnEnb (uint16_t cellId, uint32_t dlEarfcn);
  /**
   * Start cell selection function
   *
   * \param dlEarfcn the DL EARFCN
   */
  void DoStartCellSelection (uint32_t dlEarfcn);
  /// Connect function
  void DoConnect ();
  /**
   * Send data function
   *
   * \param packet the packet
   * \param bid the BID
   */
  void DoSendData (Ptr<Packet> packet, uint8_t bid);
  /// Disconnect function
  void DoDisconnect ();

  // CPHY SAP methods
  /**
   * Receive master information block function
   *
   * \param cellId the cell ID
   * \param msg LteRrcSap::MasterInformationBlock
   */
  void DoRecvMasterInformationBlock (uint16_t cellId,
                                     LteRrcSap::MasterInformationBlock msg);
  /**
   * Receive system information block type 1 function
   *
   * \param cellId the cell ID
   * \param msg LteRrcSap::SystemInformationBlockType1
   */
  void DoRecvSystemInformationBlockType1 (uint16_t cellId,
                                          LteRrcSap::SystemInformationBlockType1 msg);
  /**
   * Report UE measurements function
   *
   * \param params LteUeCphySapUser::UeMeasurementsParameters
   */
  void DoReportUeMeasurements (LteUeCphySapUser::UeMeasurementsParameters params);

  // RRC SAP methods

  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::CompleteSetup interface.
   * \param params the LteUeRrcSapProvider::CompleteSetupParameters
   */
  void DoCompleteSetup (LteUeRrcSapProvider::CompleteSetupParameters params);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvSystemInformation interface.
   * \param msg the LteRrcSap::SystemInformation
   */
  void DoRecvSystemInformation (LteRrcSap::SystemInformation msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionSetup interface.
   * \param msg the LteRrcSap::RrcConnectionSetup
   */
  void DoRecvRrcConnectionSetup (LteRrcSap::RrcConnectionSetup msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionReconfiguration interface.
   * \param msg the LteRrcSap::RrcConnectionReconfiguration
   */
  void DoRecvRrcConnectionReconfiguration (LteRrcSap::RrcConnectionReconfiguration msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionReestablishment interface.
   * \param msg LteRrcSap::RrcConnectionReestablishment
   */
  void DoRecvRrcConnectionReestablishment (LteRrcSap::RrcConnectionReestablishment msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionReestablishmentReject interface.
   * \param msg LteRrcSap::RrcConnectionReestablishmentReject
   */
  void DoRecvRrcConnectionReestablishmentReject (LteRrcSap::RrcConnectionReestablishmentReject msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionRelease interface.
   * \param msg LteRrcSap::RrcConnectionRelease
   */
  void DoRecvRrcConnectionRelease (LteRrcSap::RrcConnectionRelease msg);
  /**
   * Part of the RRC protocol. Implement the LteUeRrcSapProvider::RecvRrcConnectionReject interface.
   * \param msg the LteRrcSap::RrcConnectionReject
   */
  void DoRecvRrcConnectionReject (LteRrcSap::RrcConnectionReject msg);

  /**
   * RRC CCM SAP USER Method
   * \param noOfComponentCarriers the number of component carriers
   */
  void DoSetNumberOfComponentCarriers (uint16_t noOfComponentCarriers);


  // INTERNAL METHODS

  /**
   * \brief Go through the list of measurement results, choose the one with the
   *        strongest RSRP, and tell PHY to synchronize to it.
   *
   * \warning This function is a part of the *initial cell selection* procedure,
   *          hence must be only executed during IDLE mode.
   */
  void SynchronizeToStrongestCell ();

  /**
   * \brief Performs cell selection evaluation to the current serving cell.
   *
   * \warning This function is a part of the *initial cell selection* procedure,
   *          hence must be only executed during IDLE mode and specifically
   *          during the state when the UE just received the first SIB1 message
   *          from the serving cell.
   *
   * This function assumes that the required information for the evaluation
   * procedure have been readily gathered, such as *measurement results*, MIB,
   * and SIB1. Please refer to the LTE module's Design Documentation for more
   * details on the evaluation process.
   *
   * If the cell passes the evaluation, the UE will immediately camp to it.
   * Otherwise, the UE will pick another cell and restart the cell selection
   * procedure.
   */
  void EvaluateCellForSelection ();

  /**
   * \brief Update the current measurement configuration #m_varMeasConfig.
   * \param mc measurements to be performed by the UE
   *
   * Implements Section 5.5.2 "Measurement configuration" of 3GPP TS 36.331.
   * The supported subfunctions are:
   * - Measurement object removal
   * - Measurement object addition/ modification
   * - Reporting configuration removal
   * - Reporting configuration addition/ modification
   * - Quantity configuration
   * - Measurement identity removal
   * - Measurement identity addition/ modification
   *
   * The subfunctions that will be invoked are determined by the content of
   * the given measurement configuration.
   *
   * Note the existence of some chain reaction behaviours:
   * - Removal of measurement object or reporting configuration also removes any
   *   impacted measurement identities.
   * - Removal of measurement identity also removes any associated *reporting
   *   entry* from #m_varMeasReportList.
   * - Modification to measurement object or reporting configuration also
   *   removes any reporting entries of the impacted measurement identities
   *   from #m_varMeasReportList.
   * - Modification to quantity configuration also removes all existing
   *   reporting entries from #m_varMeasReportList, regardless of measurement
   *   identity.
   *
   * Some unsupported features:
   * - List of neighbouring cells
   * - List of black cells
   * - CGI reporting
   * - Periodical reporting configuration
   * - Measurement gaps
   * - s-Measure
   * - Speed-dependent scaling
   *
   * \warning There is a possibility that the input argument (of type
   *          LteRrcSap::MeasConfig) may contain information in fields related
   *          to the unsupported features. In such case, the function will raise
   *          an error.
   *
   * The measurement configuration given as an argument is typically provided by
   * the serving eNodeB. It is transmitted through the RRC protocol when the UE
   * joins the cell, e.g., by connection establishment or by incoming handover.
   * The information inside the argument can be configured from the eNodeB side,
   * which would then equally apply to all other UEs attached to the same
   * eNodeB. See the LTE module's User Documentation for more information on
   * configuring this.
   *
   * \sa LteRrcSap::MeasConfig, LteUeRrc::m_varMeasReportList
   */
  void ApplyMeasConfig (LteRrcSap::MeasConfig mc);

  /**
   * \brief Keep the given measurement result as the latest measurement figures,
   *        to be utilised by UE RRC functions.
   * \param cellId the cell ID of the measured cell
   * \param rsrp measured RSRP value to be saved (in dBm)
   * \param rsrq measured RSRQ value to be saved (in dB)
   * \param useLayer3Filtering
   * \param componentCarrierId
   * \todo Remove the useLayer3Filtering argument
   *
   * Implements Section 5.5.3.2 "Layer 3 filtering" of 3GPP TS 36.331. *Layer-3
   * filtering* is applied to the given measurement results before saved to
   * #m_storedMeasValues. The filtering is however disabled when the UE is in
   * IDLE mode, i.e., saving unfiltered values.
   *
   * Layer-3 filtering is influenced by a filter coefficient, which determines
   * the strength of the filtering. This coefficient is provided by the active
   * *quantity configuration* in #m_varMeasConfig, which is configured by the
   * LteUeRrc::ApplyMeasConfig. Details on how the coefficient works and how to
   * modify it can be found in LTE module's Design Documentation.
   *
   * \sa LteUeRrc::m_storedMeasValues
   */
  void SaveUeMeasurements (uint16_t cellId, double rsrp, double rsrq,
                           bool useLayer3Filtering, uint8_t componentCarrierId);

  /**
   * \brief Evaluate the reporting criteria of a measurement identity and
   *        invoke some reporting actions based on the result.
   * \param measId the measurement identity to be evaluated
   *
   * Implements Section 5.5.4.1 "Measurement report triggering - General" of
   * 3GPP TS 36.331. This function take into use the latest measurement results
   * and evaluate them against the *entering condition* and the *leaving
   * condition* of the measurement identity's reporting criteria. The evaluation
   * also take into account other defined criteria, such as *hysteresis* and
   * *time-to-trigger*.
   *
   * The entering and leaving condition to be evaluated are determined by the
   * *event type* of the measurement identity's reporting criteria. As defined
   * in LteRrcSap::ReportConfigEutra, there 5 supported events. The gore details
   * of these events can be found in Section 5.5.4 of 3GPP TS 36.331.
   *
   * An applicable entering condition (i.e., the condition evaluates to true)
   * will insert a new *reporting entry* to #m_varMeasReportList, so
   * *measurement reports* would be produced and submitted to eNodeB. On the
   * other hand, an applicable leaving condition will remove the related
   * reporting entry from #m_varMeasReportList, so submission of related
   * measurement reports to eNodeB will be suspended.
   */
  void MeasurementReportTriggering (uint8_t measId);

  /**
   * \brief Produce a proper measurement report from the given measurement
   *        identity's reporting entry in #m_varMeasReportList and then submit
   *        it to the serving eNodeB.
   * \param measId the measurement identity which report is to be submitted.
   *
   * Implements Section 5.5.5 "Measurement reporting" of 3GPP TS 36.331.
   * Producing a *measurement report* involves several tasks such as:
   * - including the measurement results of the serving cell into the report;
   * - selecting some neighbour cells which triggered the reporting (i.e., those
   *   in *cellsTriggeredList*) to be included in the report;
   * - sorting the order of neighbour cells in the report by their RSRP or RSRQ
   *   measurement results (the highest comes first); and
   * - ensuring the number of neighbour cells in the report is under the
   *   *maxReportCells* limit defined by the measurement identity's reporting
   *   configuration.
   *
   * The RSRP and RSRQ measurement results included in the report are expressed
   * in 3GPP-specified range format. They are converted from dBm and dB units
   * using EutranMeasurementMapping::Dbm2RsrpRange and
   * EutranMeasurementMapping::Db2RsrqRange functions.
   *
   * Measurement report is submitted to the serving eNodeB through the *RRC
   * protocol*. The LteUeRrcSapUser::SendMeasurementReport method of the *UE RRC
   * SAP* facilitates this submission.
   *
   * After the submission, the function will repeat itself after a certain
   * interval. The interval length may vary from 120 ms to 60 minutes and is
   * determined by the *report interval* parameter specified by the measurement
   * identity's reporting configuration.
   */
  void SendMeasurementReport (uint8_t measId);

  /**
   * Apply radio resource config dedicated.
   * \param rrcd LteRrcSap::RadioResourceConfigDedicated
   */
  void ApplyRadioResourceConfigDedicated (LteRrcSap::RadioResourceConfigDedicated rrcd);
  /**
   * Apply radio resource config dedicated secondary carrier.
   * \param nonCec LteRrcSap::NonCriticalExtensionConfiguration
   */
  void ApplyRadioResourceConfigDedicatedSecondaryCarrier (LteRrcSap::NonCriticalExtensionConfiguration nonCec);
  /// Start connection function
  void StartConnection ();
  /**
   * \brief Leave connected mode method
   * Resets the UE back to an appropiate state depending
   * on the nature of cause. For example, the UE is move
   * to the IDLE_START state upon radio link failure. At
   * RRC, all radio bearers except SRB 0 are removed,
   * measurement reports are cleared and the appropriate
   * flags are reset to their default values. This method
   * in turn triggers the reset methods of UE PHY and MAC layers.
   */
  void LeaveConnectedMode ();
  /// Dispose old SRB1
  void DisposeOldSrb1 ();
  /**
   * Bid 2 DR bid.
   * \param bid the BID
   * \returns the DR bid
   */
  uint8_t Bid2Drbid (uint8_t bid);
  /**
   * Switch the UE RRC to the given state.
   * \param s the destination state
   */
  void SwitchToState (State s);

  std::map<uint8_t, uint8_t> m_bid2DrbidMap; ///< bid to DR bid map

  std::vector<LteUeCphySapUser*> m_cphySapUser; ///< UE CPhy SAP user
  std::vector<LteUeCphySapProvider*> m_cphySapProvider; ///< UE CPhy SAP provider

  std::vector<LteUeCmacSapUser*> m_cmacSapUser; ///< UE CMac SAP user
  std::vector<LteUeCmacSapProvider*> m_cmacSapProvider; ///< UE CMac SAP provider

  LteUeRrcSapUser* m_rrcSapUser; ///< RRC SAP user
  LteUeRrcSapProvider* m_rrcSapProvider; ///< RRC SAP provider

  LteMacSapProvider* m_macSapProvider; ///< MAC SAP provider
  LtePdcpSapUser* m_drbPdcpSapUser; ///< DRB PDCP SAP user

  LteAsSapProvider* m_asSapProvider; ///< AS SAP provider
  LteAsSapUser* m_asSapUser; ///< AS SAP user

  // Receive API calls from the LteUeComponentCarrierManager  instance.
  // LteCcmRrcSapUser* m_ccmRrcSapUser;
  /// Interface to the LteUeComponentCarrierManage instance.
  LteUeCcmRrcSapProvider* m_ccmRrcSapProvider; ///< CCM RRC SAP provider
  LteUeCcmRrcSapUser* m_ccmRrcSapUser; ///< CCM RRC SAP user

  /// The current UE RRC state.
  State m_state;

  /// The unique UE identifier.
  uint64_t m_imsi;
  /**
   * The `C-RNTI` attribute. Cell Radio Network Temporary Identifier.
   */
  uint16_t m_rnti;
  /**
   * The `CellId` attribute. Serving cell identifier.
   */
  uint16_t m_cellId;

  /**
   * The `Srb0` attribute. SignalingRadioBearerInfo for SRB0.
   */
  Ptr<LteSignalingRadioBearerInfo> m_srb0;
  /**
   * The `Srb1` attribute. SignalingRadioBearerInfo for SRB1.
   */
  Ptr<LteSignalingRadioBearerInfo> m_srb1;
  /**
   * SRB1 configuration before RRC connection reconfiguration. To be deleted
   * soon by DisposeOldSrb1().
   */
  Ptr<LteSignalingRadioBearerInfo> m_srb1Old;
  /**
   * The `DataRadioBearerMap` attribute. List of UE RadioBearerInfo for Data
   * Radio Bearers by LCID.
   */
  std::map <uint8_t, Ptr<LteDataRadioBearerInfo> > m_drbMap;

  /**
   * True if RLC SM is to be used, false if RLC UM/AM are to be used.
   * Can be modified using SetUseRlcSm().
   */
  bool m_useRlcSm;

  uint8_t m_lastRrcTransactionIdentifier; ///< last RRC transaction identifier

  LteRrcSap::PdschConfigDedicated m_pdschConfigDedicated; ///< the PDSCH condig dedicated

  uint16_t m_dlBandwidth; /**< Downlink bandwidth in RBs. */
  uint16_t m_ulBandwidth; /**< Uplink bandwidth in RBs. */

  uint32_t m_dlEarfcn;  /**< Downlink carrier frequency. */
  uint32_t m_ulEarfcn;  /**< Uplink carrier frequency. */
  std::list<LteRrcSap::SCellToAddMod> m_sCellToAddModList; /**< Secondary carriers. */

  /**
   * The `MibReceived` trace source. Fired upon reception of Master Information
   * Block. Exporting IMSI, the serving cell ID, RNTI, and the source cell ID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint16_t> m_mibReceivedTrace;
  /**
   * The `Sib1Received` trace source. Fired upon reception of System
   * Information Block Type 1. Exporting IMSI, the serving cell ID, RNTI, and
   * the source cell ID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint16_t> m_sib1ReceivedTrace;
  /**
   * The `Sib2Received` trace source. Fired upon reception of System
   * Information Block Type 2. Exporting IMSI, the serving cell ID, RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_sib2ReceivedTrace;
  /**
   * The `StateTransition` trace source. Fired upon every UE RRC state
   * transition. Exporting IMSI, the serving cell ID, RNTI, old state, and new
   * state.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, State, State> m_stateTransitionTrace;
  /**
   * The `InitialCellSelectionEndOk` trace source. Fired upon successful
   * initial cell selection procedure. Exporting IMSI and the selected cell ID.
   */
  TracedCallback<uint64_t, uint16_t> m_initialCellSelectionEndOkTrace;
  /**
   * The `InitialCellSelectionEndError` trace source. Fired upon failed initial
   * cell selection procedure. Exporting IMSI and the cell ID under evaluation.
   */
  TracedCallback<uint64_t, uint16_t> m_initialCellSelectionEndErrorTrace;
  /**
   * The `RandomAccessSuccessful` trace source. Fired upon successful
   * completion of the random access procedure. Exporting IMSI, cell ID, and
   * RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_randomAccessSuccessfulTrace;
  /**
   * The `RandomAccessError` trace source. Fired upon failure of the random
   * access procedure. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_randomAccessErrorTrace;
  /**
   * The `ConnectionEstablished` trace source. Fired upon successful RRC
   * connection establishment. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_connectionEstablishedTrace;
  /**
   * The `ConnectionTimeout` trace source. Fired upon timeout RRC connection
   * establishment because of T300. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint8_t> m_connectionTimeoutTrace;
  /**
   * The `ConnectionReconfiguration` trace source. Fired upon RRC connection
   * reconfiguration. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_connectionReconfigurationTrace;
  /**
   * The `HandoverStart` trace source. Fired upon start of a handover
   * procedure. Exporting IMSI, source cell ID, RNTI, and target cell ID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint16_t> m_handoverStartTrace;
  /**
   * The `HandoverEndOk` trace source. Fired upon successful termination of a
   * handover procedure. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_handoverEndOkTrace;
  /**
   * The `HandoverEndError` trace source. Fired upon failure of a handover
   * procedure. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_handoverEndErrorTrace;
  /**
   * The `SCarrierConfigured` trace source. Fired after the configuration
   * of secondary carriers received through RRC Connection Reconfiguration
   * message.
   */
  TracedCallback<Ptr<LteUeRrc>, std::list<LteRrcSap::SCellToAddMod> > m_sCarrierConfiguredTrace;
  /**
   * The `Srb1Created` trace source. Fired when SRB1 is created, i.e.
   * the RLC and PDCP entities are created for logical channel = 1.
   * Exporting IMSI, cell ID, and RNTI
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_srb1CreatedTrace;
  /**
   * The `DrbCreated` trace source. Fired when DRB is created, i.e.
   * the RLC and PDCP entities are created for one logical channel.
   * Exporting IMSI, cell ID, RNTI, LCID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint8_t> m_drbCreatedTrace;
  /**
   * The 'PhySyncDetection' trace source. Fired when UE RRC
   * receives in-sync or out-of-sync indications from UE PHY
   *
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, std::string, uint8_t> m_phySyncDetectionTrace;
  /**
   * The 'RadioLinkFailure' trace source. Fired when T310 timer expires.
   *
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_radioLinkFailureTrace;

  /// True if a connection request by upper layers is pending.
  bool m_connectionPending;
  /// True if MIB was received for the current cell.
  bool m_hasReceivedMib;
  /// True if SIB1 was received for the current cell.
  bool m_hasReceivedSib1;
  /// True if SIB2 was received for the current cell.
  bool m_hasReceivedSib2;

  /// Stored content of the last SIB1 received.
  LteRrcSap::SystemInformationBlockType1 m_lastSib1;

  /// List of cell ID of acceptable cells for cell selection that have been detected.
  std::set<uint16_t> m_acceptableCell;

  /// List of CSG ID which this UE entity has access to.
  uint32_t m_csgWhiteList;


  // INTERNAL DATA STRUCTURE RELATED TO UE MEASUREMENTS

  /**
   * \brief Includes the accumulated configuration of the measurements to be
   *        performed by the UE.
   *
   * Based on 3GPP TS 36.331 section 7.1. Also note that some optional variables
   * in the specification are omitted.
   */
  struct VarMeasConfig
  {
    std::map<uint8_t, LteRrcSap::MeasIdToAddMod> measIdList; ///< measure ID list
    std::map<uint8_t, LteRrcSap::MeasObjectToAddMod> measObjectList; ///< measure object list
    std::map<uint8_t, LteRrcSap::ReportConfigToAddMod> reportConfigList; ///< report config list
    LteRrcSap::QuantityConfig quantityConfig; ///< quantity config
    double aRsrp; ///< RSRP
    double aRsrq; ///< RSRQ
  };

  /**
   * \brief Includes the accumulated configuration of the measurements to be
   *        performed by the UE.
   *
   * Based on 3GPP TS 36.331 section 7.1.
   */
  VarMeasConfig m_varMeasConfig;

  /**
   * \brief Represents a single measurement reporting entry., which includes
   *        information about a measurement for which the triggering conditions
   *        have been met.
   *
   * Based on 3GPP TS 36.331 section 7.1.
   */
  struct VarMeasReport
  {
    uint8_t measId; ///< measure ID
    std::set<uint16_t> cellsTriggeredList; ///< note: only E-UTRA is supported.
    uint32_t numberOfReportsSent; ///< number of reports sent
    EventId periodicReportTimer; ///< periodic report timer
  };

  /**
   * \brief The list of active reporting entries, indexed by the measurement
   *        identity which triggered the reporting. Includes information about
   *        measurements for which the triggering conditions have been met.
   */
  std::map<uint8_t, VarMeasReport> m_varMeasReportList;

  /**
   * \brief List of cell IDs which are responsible for a certain trigger.
   */
  typedef std::list<uint16_t> ConcernedCells_t;

  /**
   * \brief Compose a new reporting entry of the given measurement identity,
   *        insert it into #m_varMeasReportList, and set it up for submission
   *        to eNodeB.
   * \param measId the measurement identity which the new reporting entry will
   *               be based upon
   * \param enteringCells the cells which are responsible for triggering the
   *                      reporting (i.e., successfully fulfilling the entering
   *                      condition of the measurement identity) and will be
   *                      included in the measurement report.
   *
   * \note If an existing reporting entry with the same measurement identity has
   *       already existed in #m_varMeasReportList, the function will update it
   *       by adding the entering cells into the existing reporting entry.
   * \note When time-to-trigger is enabled for this measurement identity, the
   *       function will also remove the related trigger from the
   *       #m_enteringTriggerQueue.
   */
  void VarMeasReportListAdd (uint8_t measId, ConcernedCells_t enteringCells);

  /**
   * \brief Remove some cells from an existing reporting entry in
   *        #m_varMeasReportList.
   * \param measId the measurement identity to be removed from
   *               #m_varMeasReportList, must already exists there, otherwise
   *               an error would be raised
   * \param leavingCells the cells which are about to be removed
   * \param reportOnLeave when true, will make the function send one last
   *                      measurement report to eNodeB before removing it
   *
   * \note If a given cell is not found in the reporting entry, the function
   *       will quietly continue.
   * \note If the removal has removed all the cells in the reporting entry, the
   *       function will remove the reporting entry as well.
   * \note When time-to-trigger is enabled for this measurement identity, the
   *       function will also remove the related trigger from the
   *       #m_leavingTriggerQueue.
   */
  void VarMeasReportListErase (uint8_t measId, ConcernedCells_t leavingCells,
                               bool reportOnLeave);

  /**
   * \brief Remove the reporting entry of the given measurement identity from
   *        #m_varMeasReportList.
   * \param measId the measurement identity to be removed from
   *               #m_varMeasReportList, must already exists there, otherwise
   *               an error would be raised
   *
   * Any events or triggers related with this measurement identity will be
   * canceled as well.
   */
  void VarMeasReportListClear (uint8_t measId);

  /**
   * \brief Represents a measurement result from a certain cell.
   */
  struct MeasValues
  {
    double rsrp; ///< Measured RSRP in dBm.
    double rsrq; ///< Measured RSRQ in dB.
    uint32_t carrierFreq; ///< Measurement object frequency
  };

  /**
   * \brief Internal storage of the latest measurement results from all detected
   *        detected cells, indexed by the cell ID where the measurement was
   *        taken from.
   *
   * Each *measurement result* comprises of RSRP (in dBm) and RSRQ (in dB).
   *
   * In IDLE mode, the measurement results are used by the *initial cell
   * selection* procedure. While in CONNECTED mode, *layer-3 filtering* is
   * applied to the measurement results and they are used by *UE measurements*
   * function (LteUeRrc::MeasurementReportTriggering and
   * LteUeRrc::SendMeasurementReport).
   */
  std::map<uint16_t, MeasValues> m_storedMeasValues;

  /**
   * \brief Stored measure values per carrier.
   */
  std::map<uint16_t, std::map <uint8_t, MeasValues> > m_storedMeasValuesPerCarrier;

  /**
   * \brief Internal storage of the latest measurement results from all detected
   *        detected Secondary carrier component, indexed by the carrier component ID
   *        where the measurement was taken from.
   *
   * Each *measurement result* comprises of RSRP (in dBm) and RSRQ (in dB).
   *
   * In IDLE mode, the measurement results are used by the *initial cell
   * selection* procedure. While in CONNECTED mode, *layer-3 filtering* is
   * applied to the measurement results and they are used by *UE measurements*
   * function:
   * - LteUeRrc::MeasurementReportTriggering: in this case it is not set any
   *   measurement related to secondary carrier components since the
   *   A6 event is not implemented
   * - LteUeRrc::SendMeasurementReport: in this case the report are sent.
   */
  std::map<uint16_t, MeasValues> m_storedScellMeasValues;

  /**
   * \brief Represents a single triggered event from a measurement identity
   *        which reporting criteria have been fulfilled, but delayed by
   *        time-to-trigger.
   */
  struct PendingTrigger_t
  {
    uint8_t measId; ///< The measurement identity which raised the trigger.
    ConcernedCells_t concernedCells; ///< The list of cells responsible for this trigger.
    EventId timer; ///< The pending reporting event, scheduled at the end of the time-to-trigger.
  };

  /**
   * \brief List of triggers that were raised because entering condition have
   *        been true, but are still delayed from reporting it by
   *        time-to-trigger.
   *
   * The list is indexed by the measurement identity where the trigger
   * originates from. The enclosed event will run at the end of the
   * time-to-trigger and insert a *reporting entry* to #m_varMeasReportList.
   */
  std::map<uint8_t, std::list<PendingTrigger_t> > m_enteringTriggerQueue;

  /**
   * \brief List of triggers that were raised because leaving condition have
   *        been true, but are still delayed from stopping the reporting by
   *        time-to-trigger.
   *
   * The list is indexed by the measurement identity where the trigger
   * originates from. The enclosed event will run at the end of the
   * time-to-trigger and remove the associated *reporting entry* from
   * #m_varMeasReportList.
   */
  std::map<uint8_t, std::list<PendingTrigger_t> > m_leavingTriggerQueue;

  /**
   * \brief Clear all the waiting triggers in #m_enteringTriggerQueue which are
   *        associated with the given measurement identity.
   * \param measId the measurement identity to be processed, must already exists
   *               in #m_enteringTriggerQueue, otherwise an error would be
   *               raised
   *
   * \note The function may conclude that there is nothing to be removed. In
   *       this case, the function will simply ignore quietly.
   *
   * This function is used when the entering condition of the measurement
   * identity becomes no longer true. Therefore all the waiting triggers for
   * this measurement identity in #m_enteringTriggerQueue have become invalid
   * and must be canceled.
   *
   * \sa LteUeRrc::m_enteringTriggerQueue
   */
  void CancelEnteringTrigger (uint8_t measId);

  /**
   * \brief Remove a specific cell from the waiting triggers in
   *        #m_enteringTriggerQueue which belong to the given measurement
   *        identity.
   * \param measId the measurement identity to be processed, must already exists
   *               in #m_enteringTriggerQueue, otherwise an error would be
   *               raised
   * \param cellId the cell ID to be removed from the waiting triggers
   *
   * \note The function may conclude that there is nothing to be removed. In
   *       this case, the function will simply ignore quietly.
   *
   * This function is used when a specific neighbour cell no longer fulfills
   * the entering condition of the measurement identity. Thus the cell must be
   * removed from all the waiting triggers for this measurement identity in
   * #m_enteringTriggerQueue.
   *
   * \sa LteUeRrc::m_enteringTriggerQueue
   */
  void CancelEnteringTrigger (uint8_t measId, uint16_t cellId);

  /**
   * \brief Clear all the waiting triggers in #m_leavingTriggerQueue which are
   *        associated with the given measurement identity.
   * \param measId the measurement identity to be processed, must already exists
   *               in #m_leavingTriggerQueue, otherwise an error would be
   *               raised
   *
   * \note The function may conclude that there is nothing to be removed. In
   *       this case, the function will simply ignore quietly.
   *
   * This function is used when the leaving condition of the measurement
   * identity becomes no longer true. Therefore all the waiting triggers for
   * this measurement identity in #m_leavingTriggerQueue have become invalid
   * and must be canceled.
   *
   * \sa LteUeRrc::m_leavingTriggerQueue
   */
  void CancelLeavingTrigger (uint8_t measId);

  /**
   * \brief Remove a specific cell from the waiting triggers in
   *        #m_leavingTriggerQueue which belong to the given measurement
   *        identity.
   * \param measId the measurement identity to be processed, must already exists
   *               in #m_leavingTriggerQueue, otherwise an error would be
   *               raised
   * \param cellId the cell ID to be removed from the waiting triggers
   *
   * \note The function may conclude that there is nothing to be removed. In
   *       this case, the function will simply ignore quietly.
   *
   * This function is used when a specific neighbour cell no longer fulfills
   * the leaving condition of the measurement identity. Thus the cell must be
   * removed from all the waiting triggers for this measurement identity in
   * #m_leavingTriggerQueue.
   *
   * \sa LteUeRrc::m_leavingTriggerQueue
   */
  void CancelLeavingTrigger (uint8_t measId, uint16_t cellId);

  /**
   * The `T300` attribute. Timer for RRC connection establishment procedure
   * (i.e., the procedure is deemed as failed if it takes longer than this).
   * See Section 7.3 of 3GPP TS 36.331.
   */
  Time m_t300;

  /**
   * \brief Invokes ConnectionEstablishmentTimeout() if RRC connection
   *        establishment procedure for this UE takes longer than T300.
   */
  EventId m_connectionTimeout;

  /**
   * \brief Invoked after timer T300 expires, notifying upper layers that RRC
   *        connection establishment procedure has failed.
   */
  void ConnectionTimeout ();

  /**
   * The 'T310' attribute. After detecting N310 out-of-sync indications,
   * if number of in-sync indications detected is less than N311 before this
   * time, then the radio link is considered to have failed and the UE
   * transitions to state CONNECTED_PHY_PROMLEM and eventually IDLE_START
   * and UE context at eNodeB is destroyed. RRC connection re-establishment
   * is not initiated after this time. See 3GPP TS 36.331 7.3.
   */
  Time m_t310;

  /**
   * The 'N310' attribute. This specifies the maximum
   * consecutive out-of-sync indications from lower layers.
   */
  uint8_t m_n310;

  /**
   *  The 'N311' attribute. This specifies the minimum
   *  consecutive in-sync indications from lower layers.
   */
  uint8_t m_n311;

  /**
   * Time limit (given by m_t310) before the radio link is considered to have failed.
   * It is set upon detecting physical layer problems i.e. upon receiving
   * N310 consecutive out-of-sync indications from lower layers. Calling
   * LteUeRrc::RadioLinkFailureDetected() when it expires.
   * It is cancelled upon receiving N311 consecutive in-sync indications. Upon
   * expiry, the UE transitions to RRC_IDLE and no RRC connection
   * re-establishment is initiated.
   */
  EventId m_radioLinkFailureDetected;

  uint8_t m_noOfSyncIndications; ///< number of in-sync or out-of-sync indications coming from PHY layer

  bool m_leaveConnectedMode; ///< true if UE NAS ask UE RRC to leave connected mode, e.g., after RLF, i.e. T310 has expired

  uint16_t m_previousCellId; ///< the cell id of the previous cell UE was attached to

  uint8_t m_connEstFailCountLimit; ///< the counter value for T300 timer expiration received from the eNB

  uint8_t m_connEstFailCount; ///< the counter to count T300 timer expiration
  /**
   * \brief Radio link failure detected function
   *
   * Upon detection of radio link failure, the UE is reverted
   * back to idle state and the UE context at eNodeB and EPC
   * is deleted, thus releasing the RRC connection. The eNodeB is notified
   * in an ideal way since there is no radio link failure detection
   * implemented at the eNodeB. If the deletion process is not synchronous,
   * then errors occur due to triggering of assert messages.
   */
  void RadioLinkFailureDetected ();

  /**
   * \brief Do notify in sync function
   *
   * Triggered upon receiving an in sync indication from UE PHY.
   * When the count equals N311, then T310 is cancelled.
   */
  void DoNotifyInSync ();

  /**
   * \brief Do notify out of sync function
   *
   * Triggered upon receiving an out of sync indication from UE PHY.
   * When the count equals N310, then T310 is started.
   */
  void DoNotifyOutOfSync ();

  /**
   * \brief Do reset sync indication counter function
   *
   * Reset the sync indication counter
   * if the Qin or Qout condition at PHY
   * is not fulfilled for the number of
   * consecutive frames.
   */
  void DoResetSyncIndicationCounter ();

  /**
   * \brief Reset radio link failure parameters
   *
   * RLF timers and counters should be rest upon:
   *
   * - If the UE received N311 in Sync indications from the UE
   *   PHY.
   * - If the UE receives RRCConnectionReconfiguration including
   *   the mobilityControlInfo (TS 36.331 sec 5.3.5.4)
   *
   * Inside this method the UE RRC also instructs the UE PHY to reset the
   * RLF parameters so, it can start RLF detection again.
   *
   */
  void ResetRlfParams ();

public:
  /**
   * The number of component carriers.
   */
  uint16_t m_numberOfComponentCarriers;

}; // end of class LteUeRrc


} // namespace ns3

#endif // LTE_UE_RRC_H
