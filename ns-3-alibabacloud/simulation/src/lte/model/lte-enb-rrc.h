/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Marco Miozzo <mmiozzo@cttc.es>
 *          Manuel Requena <manuel.requena@cttc.es> 
 * Modified by:
 *          Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *          Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 *          Vignesh Babu <ns3-dev@esk.fraunhofer.de> (RLF extensions)
 */

#ifndef LTE_ENB_RRC_H
#define LTE_ENB_RRC_H

#include <ns3/nstime.h>
#include <ns3/object.h>
#include <ns3/traced-callback.h>
#include <ns3/event-id.h>

#include <ns3/lte-enb-cmac-sap.h>
#include <ns3/lte-mac-sap.h>
#include <ns3/ff-mac-sched-sap.h>
#include <ns3/ff-mac-csched-sap.h>
#include <ns3/lte-pdcp-sap.h>
#include <ns3/epc-x2-sap.h>
#include <ns3/epc-enb-s1-sap.h>
#include <ns3/lte-handover-management-sap.h>
#include <ns3/lte-ccm-rrc-sap.h>
#include <ns3/lte-enb-cphy-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/lte-anr-sap.h>
#include <ns3/lte-ffr-rrc-sap.h>
#include <ns3/lte-rlc.h>

#include <map>
#include <set>
#include <ns3/component-carrier-enb.h>
#include <vector>

#define MIN_NO_CC 1
#define MAX_NO_CC 5 // this is the maximum number of carrier components allowed by 3GPP up to R13

namespace ns3 {

class LteRadioBearerInfo;
class LteSignalingRadioBearerInfo;
class LteDataRadioBearerInfo;
class LteEnbRrc;
class Packet;



/**
 * \ingroup lte
 * Manages all the radio bearer information possessed by the ENB RRC for a
 * single UE.
 */
class UeManager : public Object
{
  /// allow LtePdcpSpecificLtePdcpSapUser<UeManager> class friend access
  friend class LtePdcpSpecificLtePdcpSapUser<UeManager>;

public:


  /**
   * The state of the UeManager at the eNB RRC
   * 
   */
  enum State
  {
    INITIAL_RANDOM_ACCESS = 0,
    CONNECTION_SETUP,
    CONNECTION_REJECTED,
    ATTACH_REQUEST,
    CONNECTED_NORMALLY,
    CONNECTION_RECONFIGURATION,
    CONNECTION_REESTABLISHMENT,
    HANDOVER_PREPARATION,
    HANDOVER_JOINING,
    HANDOVER_PATH_SWITCH,
    HANDOVER_LEAVING,
    NUM_STATES
  };

  UeManager ();
 
  /** 
   * UeManager constructor
   * 
   * \param rrc pointer to the LteEnbRrc holding this UeManager
   * \param rnti RNTI of the UE
   * \param s initial state of the UeManager
   * \param componentCarrierId primary component carrier ID
   */
  UeManager (Ptr<LteEnbRrc> rrc, uint16_t rnti, State s, uint8_t componentCarrierId);

  virtual ~UeManager (void);

  // inherited from Object
protected:
  virtual void DoInitialize ();
  virtual void DoDispose ();
public: 
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /** 
   * Set the identifiers of the source eNB for the case where a UE
   * joins the current eNB as part of a handover procedure 
   * 
   * \param sourceCellId 
   * \param sourceX2apId 
   */
  void SetSource (uint16_t sourceCellId, uint16_t sourceX2apId);

  /** 
   * Set the IMSI
   * 
   * \param imsi the IMSI
   */
  void SetImsi (uint64_t imsi);

  /**
   * Process Initial context setup request message from the MME.
   * It triggers RRC connection reconfiguration.
   */
  void InitialContextSetupRequest ();

  /** 
   * Setup a new data radio bearer, including both the configuration
   * within the eNB and the necessary RRC signaling with the UE
   * 
   * \param bearer the QoS characteristics of the bearer
   * \param bearerId the EPS bearer identifier
   * \param gtpTeid S1-bearer GTP tunnel endpoint identifier, see 36.423 9.2.1
   * \param transportLayerAddress  IP Address of the SGW, see 36.423 9.2.1
   * 
   */
  void SetupDataRadioBearer (EpsBearer bearer, uint8_t bearerId, uint32_t gtpTeid, Ipv4Address transportLayerAddress);

  /** 
   * Start all configured data radio bearers. It is safe to call this
   * method if any bearer had been already started previously.
   * 
   */
  void RecordDataRadioBearersToBeStarted ();

  /** 
   * Start the data radio bearers that have been previously recorded
   * to be started using RecordDataRadioBearersToBeStarted() 
   * 
   */
  void StartDataRadioBearers ();

  /**
   *
   * Release a given radio bearer
   *
   * \param drbid the data radio bearer id of the bearer to be released
   */
  void ReleaseDataRadioBearer (uint8_t drbid);

  /** 
   * schedule an RRC Connection Reconfiguration procedure with the UE
   * 
   */
  void ScheduleRrcConnectionReconfiguration ();

  /** 
   * Start the handover preparation and send the handover request
   * 
   * \param cellId id of the target cell
   */
  void PrepareHandover (uint16_t cellId);

  /** 
   * take the necessary actions in response to the reception of an X2 HANDOVER REQUEST ACK message
   * 
   * \param params 
   */
  void RecvHandoverRequestAck (EpcX2SapUser::HandoverRequestAckParams params);

  /** 
   * 
   * \return the HandoverPreparationInfo sent by the source eNB to the
   * target eNB in the X2-based handover procedure
   */
  LteRrcSap::RadioResourceConfigDedicated GetRadioResourceConfigForHandoverPreparationInfo ();

  /** 
   * \param componentCarrierId target component carrier ID
   * 
   * \return retrieve the data that the target eNB needs to send to the source
   * eNB as the Handover Command in the X2-based handover
   * procedure.
   *
   * \note mobility control info is not expected to be filled in
   * (shall be filled in by the caller). 
   */
  LteRrcSap::RrcConnectionReconfiguration GetRrcConnectionReconfigurationForHandover (uint8_t componentCarrierId);

  /**
   * Send a data packet over the appropriate Data Radio Bearer.
   * If state is HANDOVER_JOINING (i.e. target eNB has received the
   * Handover Request), the packet is buffered.
   * If state is HANDOVER_LEAVING (i.e. source eNB has received the
   * RRC Connection Reconfiguration, the packet is sent through the
   * X2 interface.
   *
   * \param bid the corresponding EPS Bearer ID
   * \param p the packet
   */
  void SendData (uint8_t bid, Ptr<Packet> p);

  /** 
   * 
   * \return a list of ERAB-to-be-setup items to be put in a X2 HO REQ message
   */
  std::vector<EpcX2Sap::ErabToBeSetupItem> GetErabList ();

  /** 
   * send the UE CONTEXT RELEASE X2 message to the source eNB, thus
   * successfully terminating an X2 handover procedure 
   * 
   */
  void SendUeContextRelease ();

  /** 
   * Take the necessary actions in response to the reception of an X2 HO preparation failure message
   * 
   * \param cellId id of the target cell
   */
  void RecvHandoverPreparationFailure (uint16_t cellId);

  /** 
   * Take the necessary actions in response to the reception of an X2 SN STATUS TRANSFER message
   * 
   * \param params the SN STATUS
   */
  void RecvSnStatusTransfer (EpcX2SapUser::SnStatusTransferParams params);
 
  /** 
   * Take the necessary actions in response to the reception of an X2 UE CONTEXT RELEASE message
   * 
   * \param params the SN STATUS
   */
  void RecvUeContextRelease (EpcX2SapUser::UeContextReleaseParams params); 


  // METHODS FORWARDED FROM ENB RRC SAP ///////////////////////////////////////

  /**
   * Implement the LteEnbRrcSapProvider::CompleteSetupUe interface.
   * \param params CompleteSetupUeParameters
   */
  void CompleteSetupUe (LteEnbRrcSapProvider::CompleteSetupUeParameters params);
  /**
   * Implement the LteEnbRrcSapProvider::RecvRrcConnectionRequest interface.
   * \param msg the RRC connection request message
   */
  void RecvRrcConnectionRequest (LteRrcSap::RrcConnectionRequest msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvRrcConnectionSetupCompleted interface.
   * \param msg RRC connection setup completed message
   */
  void RecvRrcConnectionSetupCompleted (LteRrcSap::RrcConnectionSetupCompleted msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvRrcConnectionReconfigurationCompleted interface.
   * \param msg RRC connection reconfiguration completed message
   */
  void RecvRrcConnectionReconfigurationCompleted (LteRrcSap::RrcConnectionReconfigurationCompleted msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvRrcConnectionReestablishmentRequest interface.
   * \param msg the RRC connection reestablishment request message
   */
  void RecvRrcConnectionReestablishmentRequest (LteRrcSap::RrcConnectionReestablishmentRequest msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvRrcConnectionReestablishmentComplete interface.
   * \param msg the RRC connection reestablsihment complete message
   */
  void RecvRrcConnectionReestablishmentComplete (LteRrcSap::RrcConnectionReestablishmentComplete msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvMeasurementReport interface.
   * \param msg the measrurement report
   */
  void RecvMeasurementReport (LteRrcSap::MeasurementReport msg);
  /**
   * Implement the LteEnbRrcSapProvider::RecvIdealUeContextRemoveRequest interface.
   *
   * \param rnti the C-RNTI identifying the user
   */
  void RecvIdealUeContextRemoveRequest (uint16_t rnti);


  // METHODS FORWARDED FROM ENB CMAC SAP //////////////////////////////////////

  /**
   * CMAC UE config update indication function
   * \param cmacParams the UE config parameters
   */
  void CmacUeConfigUpdateInd (LteEnbCmacSapUser::UeConfig cmacParams);

  // METHODS FORWARDED FROM ENB PDCP SAP //////////////////////////////////////

  /**
   * Receive PDCP SDU function
   * \param params the receive PDCP SDU parameters
   */
  void DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params);

  /** 
   * 
   * \return the RNTI, i.e., an UE identifier that is unique within
   * the cell
   */
  uint16_t GetRnti (void) const;

  /** 
   *
   * \return the IMSI, i.e., a globally unique UE identifier
   */
  uint64_t GetImsi (void) const;

  /**
   *
   * \return the primary component carrier ID
   */
  uint8_t GetComponentCarrierId () const;

  /** 
   * 
   * \return the SRS Configuration Index
   */
  uint16_t GetSrsConfigurationIndex (void) const;

  /** 
   * Set the SRS configuration index and do the necessary reconfiguration
   * 
   * \param srsConfIndex 
   */
  void SetSrsConfigurationIndex (uint16_t srsConfIndex);

  /** 
   * 
   * \return the current state
   */
  State GetState () const;

  /**
   * Configure PdschConfigDedicated (i.e. P_A value) for UE and start RrcConnectionReconfiguration
   * to inform UE about new PdschConfigDedicated
   *
   * \param pdschConfigDedicated new pdschConfigDedicated (i.e. P_A value) to be set
   */
  void SetPdschConfigDedicated (LteRrcSap::PdschConfigDedicated pdschConfigDedicated);

  /**
   * Cancel all timers which are running for the UE
   *
   */
  void CancelPendingEvents ();

  /**
   * TracedCallback signature for state transition events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] oldState
   * \param [in] newState
   */
  typedef void (*StateTracedCallback)
    (const uint64_t imsi, const uint16_t cellId, const uint16_t rnti,
    const State oldState, const State newState);

private:

  /** 
   * Add a new LteDataRadioBearerInfo structure to the UeManager
   * 
   * \param radioBearerInfo 
   * 
   * \return the id of the newly added data radio bearer structure
   */
  uint8_t AddDataRadioBearerInfo (Ptr<LteDataRadioBearerInfo> radioBearerInfo);

  /** 
   * \param drbid the Data Radio Bearer id
   * 
   * \return the corresponding LteDataRadioBearerInfo
   */
  Ptr<LteDataRadioBearerInfo> GetDataRadioBearerInfo (uint8_t drbid);

  /** 
   * remove the LteDataRadioBearerInfo corresponding to a bearer being released
   *
   * \param drbid the Data Radio Bearer id
   */
  void RemoveDataRadioBearerInfo (uint8_t drbid);

  /** 
   * 
   * \return an RrcConnectionReconfiguration struct built based on the
   * current configuration
   */
  LteRrcSap::RrcConnectionReconfiguration BuildRrcConnectionReconfiguration ();

  /** 
   * 
   * \return an NonCriticalExtensionConfiguration struct built based on the
   * current configuration
   */
  LteRrcSap::NonCriticalExtensionConfiguration BuildNonCriticalExtensionConfigurationCa ();

  /** 
   * 
   * \return a RadioResourceConfigDedicated struct built based on the
   * current configuration
   */
  LteRrcSap::RadioResourceConfigDedicated BuildRadioResourceConfigDedicated ();

  /** 
   * 
   * \return a newly allocated identifier for a new RRC transaction
   */
  uint8_t GetNewRrcTransactionIdentifier ();

  /** 
   * \param lcid a Logical Channel Identifier
   * 
   * \return the corresponding Data Radio Bearer Id
   */
  uint8_t Lcid2Drbid (uint8_t lcid);

  /** 
   * \param drbid a Data Radio Bearer Id
   * 
   * \return the corresponding  Logical Channel Identifier
   */
  uint8_t Drbid2Lcid (uint8_t drbid);

  /** 
   * \param lcid a  Logical Channel Identifier
   * 
   * \return the corresponding EPS Bearer Identifier
   */
  uint8_t Lcid2Bid (uint8_t lcid);

  /** 
   * \param bid  an EPS Bearer Identifier
   * 
   * \return the corresponding Logical Channel Identifier
   */
  uint8_t Bid2Lcid (uint8_t bid);

  /** 
   * \param drbid Data Radio Bearer Id
   * 
   * \return the corresponding EPS Bearer Identifier
   */
  uint8_t Drbid2Bid (uint8_t drbid);

  /** 
   * \param bid an EPS Bearer Identifier
   * 
   * \return the corresponding Data Radio Bearer Id
   */
  uint8_t Bid2Drbid (uint8_t bid);

  /**
   * Send a data packet over the appropriate Data Radio Bearer.
   * It is called by SendData if the UE is in a connected state
   * or when the RRC Connection Reconfiguration Complete message
   * is received and the packets are debuffered.
   *
   * \param bid the corresponding EPS Bearer ID
   * \param p the packet
   */
  void SendPacket (uint8_t bid, Ptr<Packet> p);

  /** 
   * Switch the UeManager to the given state
   * 
   * \param s the given state
   */
  void SwitchToState (State s);

  uint8_t m_lastAllocatedDrbid; ///< last allocated Data Radio Bearer ID

  /**
   * The `DataRadioBearerMap` attribute. List of UE DataRadioBearerInfo by
   * DRBID.
   */
  std::map <uint8_t, Ptr<LteDataRadioBearerInfo> > m_drbMap;

  /**
   * The `Srb0` attribute. SignalingRadioBearerInfo for SRB0.
   */
  Ptr<LteSignalingRadioBearerInfo> m_srb0;
  /**
   * The `Srb1` attribute. SignalingRadioBearerInfo for SRB1.
   */
  Ptr<LteSignalingRadioBearerInfo> m_srb1;

  /**
   * The `C-RNTI` attribute. Cell Radio Network Temporary Identifier.
   */
  uint16_t m_rnti;
  /**
   * International Mobile Subscriber Identity assigned to this UE. A globally
   * unique UE identifier.
   */
  uint64_t m_imsi;
  /**
   * ID of the primary CC for this UE
   */
  uint8_t m_componentCarrierId;
  
  uint8_t m_lastRrcTransactionIdentifier; ///< last RRC transaction identifier

  LteRrcSap::PhysicalConfigDedicated m_physicalConfigDedicated; ///< physical config dedicated
  /// Pointer to the parent eNodeB RRC.
  Ptr<LteEnbRrc> m_rrc;
  /// The current UeManager state.
  State m_state;
  
  LtePdcpSapUser* m_drbPdcpSapUser; ///< DRB PDCP SAP user

  bool m_pendingRrcConnectionReconfiguration; ///< pending RRC connection reconfiguration

  /**
   * The `StateTransition` trace source. Fired upon every UE state transition
   * seen by the UeManager at the eNB RRC. Exporting IMSI, cell ID, RNTI, old
   * state, and new state.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, State, State> m_stateTransitionTrace;

  /**
   * The `DrbCreated` trace source. Fired when DRB is created, i.e.
   * the RLC and PDCP entities are created for one logical channel.
   * Exporting IMSI, cell ID, RNTI, LCID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint8_t> m_drbCreatedTrace;

  uint16_t m_sourceX2apId; ///< source X2 ap ID
  uint16_t m_targetX2apId; ///< target X2 ap ID
  uint16_t m_sourceCellId; ///< source cell ID
  uint16_t m_targetCellId; ///< target cell ID
  std::list<uint8_t> m_drbsToBeStarted; ///< DRBS to be started
  bool m_needPhyMacConfiguration; ///< need Phy MAC configuration

  /**
   * Time limit before a _connection request timeout_ occurs. Set after a new
   * UE context is added after a successful Random Access. Calling
   * LteEnbRrc::ConnectionRequestTimeout() when it expires. Cancelled when RRC
   * CONNECTION REQUEST is received.
   */
  EventId m_connectionRequestTimeout;
  /**
   * Time limit before a _connection setup timeout_ occurs. Set after an RRC
   * CONNECTION SETUP is sent. Calling LteEnbRrc::ConnectionSetupTimeout() when
   * it expires. Cancelled when RRC CONNECTION SETUP COMPLETE is received.
   */
  EventId m_connectionSetupTimeout;
  /**
   * The delay before a _connection rejected timeout_ occurs. Set after an RRC
   * CONNECTION REJECT is sent. Calling LteEnbRrc::ConnectionRejectedTimeout()
   * when it expires.
   */
  EventId m_connectionRejectedTimeout;
  /**
   * Time limit before a _handover joining timeout_ occurs. Set after a new UE
   * context is added after receiving a handover request. Calling
   * LteEnbRrc::HandoverJoiningTimeout() when it expires. Cancelled when
   * RRC CONNECTION RECONFIGURATION COMPLETE is received.
   */
  EventId m_handoverJoiningTimeout;
  /**
   * Time limit before a _handover leaving timeout_ occurs. Set after a
   * handover command is sent. Calling LteEnbRrc::HandoverLeavingTimeout()
   * when it expires. Cancelled when RRC CONNECTION RE-ESTABLISHMENT or X2
   * UE CONTEXT RELEASE is received.
   */
  EventId m_handoverLeavingTimeout;

  /// Define if the Carrier Aggregation was already configure for the current UE on not
  bool m_caSupportConfigured;

  /// Pending start data radio bearers
  bool m_pendingStartDataRadioBearers;


  /**
   * Packet buffer for when UE is doing the handover.
   * The packets are stored with the bid (bearer ID).
   *
   * Source eNB starts forwarding data to target eNB through the X2 interface
   * when it sends RRC Connection Reconfiguration to the UE.
   * Target eNB buffers data until it receives RRC Connection Reconfiguration
   * Complete from the UE.
   */
  std::list<std::pair<uint8_t, Ptr<Packet> > > m_packetBuffer;

}; // end of `class UeManager`



/**
 * \ingroup lte
 * 
 * The LTE Radio Resource Control entity at the eNB
 */
class LteEnbRrc : public Object
{

  /// allow EnbRrcMemberLteEnbCmacSapUser class friend access
  friend class EnbRrcMemberLteEnbCmacSapUser;
  /// allow MemberLteHandoverManagementSapUser<LteEnbRrc> class friend access
  friend class MemberLteHandoverManagementSapUser<LteEnbRrc>;
  /// allow MemberLteAnrSapUser<LteEnbRrc> class friend access
  friend class MemberLteAnrSapUser<LteEnbRrc>;
  /// allow MemberLteFfrRrcSapUser<LteEnbRrc> class friend access
  friend class MemberLteFfrRrcSapUser<LteEnbRrc>;
  /// allow MemberLteEnbRrcSapProvider<LteEnbRrc> class friend access
  friend class MemberLteEnbRrcSapProvider<LteEnbRrc>;
  /// allow MemberLteEnbRrcSapProvider<LteEnbRrc> class friend access
  friend class MemberEpcEnbS1SapUser<LteEnbRrc>;
  /// allow MemberEpcEnbS1SapUser<LteEnbRrc> class friend access
  friend class EpcX2SpecificEpcX2SapUser<LteEnbRrc>;
  /// allow UeManager class friend access
  friend class UeManager;
  /// allow  MemberLteCcmRrcSapUser<LteEnbRrc> class friend access
  friend class MemberLteCcmRrcSapUser<LteEnbRrc>;

public:
  /**
   * create an RRC instance for use within an eNB
   *
   */
  LteEnbRrc ();

  /**
   * Destructor
   */
  virtual ~LteEnbRrc ();


  // inherited from Object
protected:
  virtual void DoDispose (void);
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Set the X2 SAP this RRC should interact with
   * \param s the X2 SAP Provider to be used by this RRC entity
   */
  void SetEpcX2SapProvider (EpcX2SapProvider* s);

  /** 
   * Get the X2 SAP offered by this RRC
   * \return s the X2 SAP User interface offered to the X2 entity by this RRC entity
   */
  EpcX2SapUser* GetEpcX2SapUser ();


  /**
   * set the CMAC SAP this RRC should interact with
   *
   * \param s the CMAC SAP Provider to be used by this RRC
   */
  void SetLteEnbCmacSapProvider (LteEnbCmacSapProvider * s);

  /**
   * set the CMAC SAP this RRC should interact with
   *
   * \param s the CMAC SAP Provider to be used by this RRC
   * \param pos the position
   */
  void SetLteEnbCmacSapProvider (LteEnbCmacSapProvider * s, uint8_t pos);

  /** 
   * Get the CMAC SAP offered by this RRC
   * \returns the CMAC SAP User interface offered to the MAC by this RRC
   */
  LteEnbCmacSapUser* GetLteEnbCmacSapUser ();

  /** 
   * Get the CMAC SAP offered by this RRC
   * \param pos the position
   * \returns the CMAC SAP User interface offered to the MAC by this RRC
   */
  LteEnbCmacSapUser* GetLteEnbCmacSapUser (uint8_t pos);

  /**
   * set the Handover Management SAP this RRC should interact with
   *
   * \param s the Handover Management SAP Provider to be used by this RRC
   */
  void SetLteHandoverManagementSapProvider (LteHandoverManagementSapProvider * s);

  /**
   * Get the Handover Management SAP offered by this RRC
   * \returns the Handover Management SAP User interface offered to the
   *           handover algorithm by this RRC
   */
  LteHandoverManagementSapUser* GetLteHandoverManagementSapUser ();


  /**
   * set the Component Carrier Management SAP this RRC should interact with
   *
   * \param s the Component Carrier Management SAP Provider to be used by this RRC
   */
  void SetLteCcmRrcSapProvider (LteCcmRrcSapProvider * s);

  /**
   * Get the Component Carrier Management SAP offered by this RRC
   * \return s the Component Carrier Management SAP User interface offered to the
   *           carrier component selection algorithm by this RRC
   */
  LteCcmRrcSapUser* GetLteCcmRrcSapUser ();


  /**
   * set the ANR SAP this RRC should interact with
   *
   * \param s the ANR SAP Provider to be used by this RRC
   */
  void SetLteAnrSapProvider (LteAnrSapProvider * s);

  /**
   * Get the ANR SAP offered by this RRC
   * \return s the ANR SAP User interface offered to the ANR instance by this
   *           RRC
   */
  LteAnrSapUser* GetLteAnrSapUser ();



  /**
   * set the FFR SAP this RRC should interact with
   *
   * \param s the FFR SAP Provider to be used by this RRC
   */
  void SetLteFfrRrcSapProvider (LteFfrRrcSapProvider * s);
  /**
   * set the FFR SAP this RRC should interact with
   *
   * \param s the FFR SAP Provider to be used by this RRC
   * \param index the index
   */
  void SetLteFfrRrcSapProvider (LteFfrRrcSapProvider * s, uint8_t index);

  /**
   * Get the FFR SAP offered by this RRC
   * \return s the FFR SAP User interface offered to the ANR instance by this
   *           RRC
   */
  LteFfrRrcSapUser* GetLteFfrRrcSapUser ();
  /**
   * Get the FFR SAP offered by this RRC
   * \param index the index
   * \return s the FFR SAP User interface offered to the ANR instance by this
   *           RRC
   */
  LteFfrRrcSapUser* GetLteFfrRrcSapUser (uint8_t index);

  /**
   * set the RRC SAP this RRC should interact with
   *
   * \param s the RRC SAP User to be used by this RRC
   */
  void SetLteEnbRrcSapUser (LteEnbRrcSapUser * s);

  /**
   *
   *
   * \return s the RRC SAP Provider interface offered to the MAC by this RRC
   */
  LteEnbRrcSapProvider* GetLteEnbRrcSapProvider ();

  /**
   * set the MAC SAP provider. The eNB RRC does not use this
   * directly, but it needs to provide it to newly created RLC instances.
   *
   * \param s the MAC SAP provider that will be used by all
   * newly created RLC instances
   */
  void SetLteMacSapProvider (LteMacSapProvider* s);


  /** 
   * Set the S1 SAP Provider
   * 
   * \param s the S1 SAP Provider
   */
  void SetS1SapProvider (EpcEnbS1SapProvider * s);

  /** 
   * 
   * \return the S1 SAP user
   */
  EpcEnbS1SapUser* GetS1SapUser ();


  /**
   * set the CPHY SAP this RRC should use to interact with the PHY
   *
   * \param s the CPHY SAP Provider
   */
  void SetLteEnbCphySapProvider (LteEnbCphySapProvider * s);

  /**
   * set the CPHY SAP this RRC should use to interact with the PHY
   *
   * \param s the CPHY SAP Provider
   * \param pos the position
   */
  void SetLteEnbCphySapProvider (LteEnbCphySapProvider * s, uint8_t pos);

  /**
   *
   *
   * \return s the CPHY SAP User interface offered to the PHY by this RRC
   */
  LteEnbCphySapUser* GetLteEnbCphySapUser ();

  /**
   * Get the ENB CPhy SAP user
   *
   * \param pos the position
   * \return s the CPHY SAP User interface offered to the PHY by this RRC
   */
  LteEnbCphySapUser* GetLteEnbCphySapUser (uint8_t pos);

  /** 
   * 
   * 
   * \param rnti the identifier of an UE
   * 
   * \return true if the corresponding UeManager instance exists
   */
  bool HasUeManager (uint16_t rnti) const;

  /**
   *
   *
   * \param rnti the identifier of an UE
   *
   * \return the corresponding UeManager instance
   */
  Ptr<UeManager> GetUeManager (uint16_t rnti);

  /**
   * \brief Add a new UE measurement reporting configuration
   * \param config the new reporting configuration
   * \return the measurement IDs (measId) referring to the newly added
   *         reporting configuration
   *
   * Assuming intra-frequency environment, the new measurement reporting
   * configuration will be automatically associated to measurement
   * objects corresponding to serving cell frequencies.
   *
   * Can only be called before the start of simulation.
   */
  std::vector<uint8_t> AddUeMeasReportConfig (LteRrcSap::ReportConfigEutra config);

  /**
   * \brief Configure cell-specific parameters.
   *
   * Configure cell-specific parameters and propagate them to lower layers.
   * The parameters include bandwidth, EARFCN (E-UTRA Absolute Radio Frequency
   * Channel Number), and cell ID.
   *
   * In addition to parameter configuration, this function also performs several
   * other tasks:
   *  - Initializing UE measurement (i.e. measurement object and quantity
   *    configuration), which is expected to be further configured through
   *    `LteEnbRrc::AddUeMeasReportConfig`;
   *  - Enabling MIB (Master Information Block) broadcast transmission
   *  - Enabling SIB (System Information Block) broadcast transmission
   *
   * Typically runs when the eNodeB NetDevice is installed, for instance by
   * `LteHelper::InstallEnbDevice` (i.e. before the simulation starts).
   *
   * \warning Raises an error when executed more than once.
   *
   * \param ccPhyConf the component carrier configuration
   */
  void ConfigureCell (std::map<uint8_t, Ptr<ComponentCarrierBaseStation> > ccPhyConf);

  /**
   * \brief Configure carriers.
   * \param ccPhyConf the component carrier configuration
   */
  void ConfigureCarriers (std::map<uint8_t, Ptr<ComponentCarrierBaseStation>> ccPhyConf);

  /** 
   * set the cell id of this eNB
   * 
   * \param m_cellId 
   */
  void SetCellId (uint16_t m_cellId);

  /** 
   * set the cell id of this eNB
   * 
   * \param m_cellId 
   * \param ccIndex 
   */
  void SetCellId (uint16_t m_cellId, uint8_t ccIndex);

  /**
   * convert the cell id to component carrier id
   *
   * \param cellId Cell ID
   *
   * \return corresponding component carrier id
   */
  uint8_t CellToComponentCarrierId (uint16_t cellId);

  /**
   * convert the component carrier id to cell id
   *
   * \param componentCarrierId component carrier ID
   *
   * \return corresponding cell ID
   */
  uint16_t ComponentCarrierToCellId (uint8_t componentCarrierId);

  /**
   * \param cellId cell ID
   * \return true if cellId is served by this eNB
   */
  bool HasCellId (uint16_t cellId) const;

  /** 
   * Enqueue an IP data packet on the proper bearer for downlink
   * transmission. Normally expected to be called by the NetDevice
   * forwarding a packet coming from the EpcEnbApplication 
   * 
   * \param p the packet
   * 
   * \return true if successful, false if an error occurred
   */
  bool SendData (Ptr<Packet> p);

  /** 
   * set the callback used to forward data packets up the stack
   * 
   * \param cb 
   */
  void SetForwardUpCallback (Callback <void, Ptr<Packet> > cb);

  /** 
   * Method triggered when a UE is expected to request for connection but does
   * not do so in a reasonable time. The method will remove the UE context.
   * 
   * \param rnti the T-C-RNTI whose timeout expired
   */
  void ConnectionRequestTimeout (uint16_t rnti);

  /** 
   * Method triggered when a UE is expected to complete a connection setup
   * procedure but does not do so in a reasonable time. The method will remove
   * the UE context.
   *
   * \param rnti the T-C-RNTI whose timeout expired
   */
  void ConnectionSetupTimeout (uint16_t rnti);

  /**
   * Method triggered a while after sending RRC Connection Rejected. The method
   * will remove the UE context.
   * 
   * \param rnti the T-C-RNTI whose timeout expired
   */
  void ConnectionRejectedTimeout (uint16_t rnti);

  /** 
   * Method triggered when a UE is expected to join the cell for a handover 
   * but does not do so in a reasonable time. The method will remove the UE
   * context.
   * 
   * \param rnti the C-RNTI whose timeout expired
   */
  void HandoverJoiningTimeout (uint16_t rnti);

  /** 
   * Method triggered when a UE is expected to leave a cell for a handover
   * but no feedback is received in a reasonable time. The method will remove
   * the UE context.
   * 
   * \param rnti the C-RNTI whose timeout expired
   */
  void HandoverLeavingTimeout (uint16_t rnti);

  /** 
   * Send a HandoverRequest through the X2 SAP interface. This method will
   * trigger a handover which is started by the RRC by sending a handover
   * request to the target eNB over the X2 interface
   *
   * \param rnti the ID of the UE to be handed over
   * \param cellId the ID of the target eNB
   */
  void SendHandoverRequest (uint16_t rnti, uint16_t cellId);

  /**
   *  \brief This function acts as an interface to trigger Release indication messages towards eNB and EPC
   *  \param imsi the IMSI
   *  \param rnti the RNTI
   *  \param bearerId Bearer Identity which is to be de-activated
   */
  void DoSendReleaseDataRadioBearer (uint64_t imsi, uint16_t rnti, uint8_t bearerId);

  /**
   * Identifies how EPS Bearer parameters are mapped to different RLC types
   * 
   */
  enum LteEpsBearerToRlcMapping_t {RLC_SM_ALWAYS = 1,
                                   RLC_UM_ALWAYS = 2,
                                   RLC_AM_ALWAYS = 3,
                                   PER_BASED = 4};

  /**
   * TracedCallback signature for new Ue Context events.
   *
   * \param [in] cellId
   * \param [in] rnti
   */
  typedef void (*NewUeContextTracedCallback)
    (const uint16_t cellId, const uint16_t rnti);

  /**
   * TracedCallback signature for connection and handover end events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   */
  typedef void (*ConnectionHandoverTracedCallback)
    (const uint64_t imsi, const uint16_t cellId, const uint16_t rnti);

  /**
   * TracedCallback signature for handover start events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] targetCid
   */
  typedef void (*HandoverStartTracedCallback)
    (const uint64_t imsi, const uint16_t cellId, const uint16_t rnti,
     const uint16_t targetCid);

  /**
   * TracedCallback signature for receive measurement report events.
   *
   * \param [in] imsi
   * \param [in] cellId
   * \param [in] rnti
   * \param [in] report
   * \todo The \c LteRrcSap::MeasurementReport argument should be
   * changed to a const reference since the argument is large.
   */
  typedef void (*ReceiveReportTracedCallback)
    (const uint64_t imsi, const uint16_t cellId, const uint16_t rnti,
     const LteRrcSap::MeasurementReport report);

  /**
   * TracedCallback signature for timer expiry events
   *
   * \param [in] imsi
   * \param [in] rnti
   * \param [in] cellId
   * \param [in] cause
   */
  typedef void (*TimerExpiryTracedCallback)
      (const uint64_t imsi, const uint16_t rnti, const uint16_t cellId,
       const std::string cause);


private:


  // RRC SAP methods

  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::CompleteSetupUe interface to UeManager::CompleteSetupUe
   * \param rnti the RNTI
   * \param params the LteEnbRrcSapProvider::CompleteSetupUeParameters
   */
  void DoCompleteSetupUe (uint16_t rnti, LteEnbRrcSapProvider::CompleteSetupUeParameters params);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvRrcConnectionRequest interface to UeManager::RecvRrcConnectionRequest
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::RrcConnectionRequest
   */
  void DoRecvRrcConnectionRequest (uint16_t rnti, LteRrcSap::RrcConnectionRequest msg);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvRrcConnectionSetupCompleted interface to UeManager::RecvRrcConnectionSetupCompleted
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::RrcConnectionSetupCompleted
   */
  void DoRecvRrcConnectionSetupCompleted (uint16_t rnti, LteRrcSap::RrcConnectionSetupCompleted msg);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvRrcConnectionReconfigurationCompleted interface to UeManager::RecvRrcConnectionReconfigurationCompleted
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::RrcConnectionReconfigurationCompleted
   */
  void DoRecvRrcConnectionReconfigurationCompleted (uint16_t rnti, LteRrcSap::RrcConnectionReconfigurationCompleted msg);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvRrcConnectionReestablishmentRequest interface to UeManager::RecvRrcConnectionReestablishmentRequest
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::RrcConnectionReestablishmentRequest
   */
  void DoRecvRrcConnectionReestablishmentRequest (uint16_t rnti, LteRrcSap::RrcConnectionReestablishmentRequest msg);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvRrcConnectionReestablishmentComplete interface to UeManager::RecvRrcConnectionReestablishmentComplete
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::RrcConnectionReestablishmentComplete
   */
  void DoRecvRrcConnectionReestablishmentComplete (uint16_t rnti, LteRrcSap::RrcConnectionReestablishmentComplete msg);
  /**
   * Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvMeasurementReport interface to UeManager::RecvMeasurementReport
   *
   * \param rnti the RNTI
   * \param msg the LteRrcSap::MeasurementReport
   */
  void DoRecvMeasurementReport (uint16_t rnti, LteRrcSap::MeasurementReport msg);
  /**
   * \brief Part of the RRC protocol. Forwarding LteEnbRrcSapProvider::RecvIdealUeContextRemoveRequest interface to UeManager::RecvIdealUeContextRemoveRequest.
   *
   * Remove the UE context at eNodeB and also remove the bearers established
   * at SGW/PGW node. Bearer info at MME is not deleted since they are added at
   * MME only at the beginning of simulation and if they are removed,
   * the bearers cannot be activated again.
   *
   * \param rnti the C-RNTI identifying the user
   */
  void DoRecvIdealUeContextRemoveRequest (uint16_t rnti);

  // S1 SAP methods

  /**
   * Initial context setup request function
   *
   * \param params EpcEnbS1SapUser::InitialContextSetupRequestParameters
   */
  void DoInitialContextSetupRequest (EpcEnbS1SapUser::InitialContextSetupRequestParameters params);
  /**
   * Data radio beaerer setup request function
   *
   * \param params EpcEnbS1SapUser::DataRadioBearerSetupRequestParameters
   */
  void DoDataRadioBearerSetupRequest (EpcEnbS1SapUser::DataRadioBearerSetupRequestParameters params);
  /**
   * Path switch request acknowledge function
   *
   * \param params EpcEnbS1SapUser::PathSwitchRequestAcknowledgeParameters
   */
  void DoPathSwitchRequestAcknowledge (EpcEnbS1SapUser::PathSwitchRequestAcknowledgeParameters params);

  // X2 SAP methods

  /**
   * Receive handover request function
   *
   * \param params EpcX2SapUser::HandoverRequestParams
   */
  void DoRecvHandoverRequest (EpcX2SapUser::HandoverRequestParams params);
  /**
   * Receive handover request acknowledge function
   *
   * \param params EpcX2SapUser::HandoverRequestAckParams
   */
  void DoRecvHandoverRequestAck (EpcX2SapUser::HandoverRequestAckParams params);
  /**
   * Receive handover preparation failure function
   *
   * \param params EpcX2SapUser::HandoverPreparationFailureParams
   */
  void DoRecvHandoverPreparationFailure (EpcX2SapUser::HandoverPreparationFailureParams params);
  /**
   * Receive SN status transfer function
   *
   * \param params EpcX2SapUser::SnStatusTransferParams
   */
  void DoRecvSnStatusTransfer (EpcX2SapUser::SnStatusTransferParams params);
  /**
   * Receive UE context release function
   *
   * \param params EpcX2SapUser::UeContextReleaseParams
   */
  void DoRecvUeContextRelease (EpcX2SapUser::UeContextReleaseParams params);
  /**
   * Receive load information function
   *
   * \param params EpcX2SapUser::LoadInformationParams
   */
  void DoRecvLoadInformation (EpcX2SapUser::LoadInformationParams params);
  /**
   * Receive resource status update function
   *
   * \param params EpcX2SapUser::ResourceStatusUpdateParams
   */
  void DoRecvResourceStatusUpdate (EpcX2SapUser::ResourceStatusUpdateParams params);
  /**
   * Receive UE data function
   *
   * \param params EpcX2SapUser::UeDataParams
   */
  void DoRecvUeData (EpcX2SapUser::UeDataParams params);

  // CMAC SAP methods

  /**
   * Allocate temporary cell RNTI function
   *
   * \param componentCarrierId ID of the primary component carrier
   * \return temporary RNTI
   */
  uint16_t DoAllocateTemporaryCellRnti (uint8_t componentCarrierId);
  /**
   * Notify LC config result function
   *
   * \param rnti RNTI
   * \param lcid LCID
   * \param success the success indicator
   */
  void DoNotifyLcConfigResult (uint16_t rnti, uint8_t lcid, bool success);
  /**
   * RRC configuration update indication function
   *
   * \param params LteEnbCmacSapUser::UeConfig
   */
  void DoRrcConfigurationUpdateInd (LteEnbCmacSapUser::UeConfig params);

  // Handover Management SAP methods

  /**
   * Add UE measure report config for handover function
   *
   * \param reportConfig LteRrcSap::ReportConfigEutra
   * \returns measure ID
   */
  std::vector<uint8_t> DoAddUeMeasReportConfigForHandover (LteRrcSap::ReportConfigEutra reportConfig);
  /**
   * Add UE measure report config for component carrier function
   *
   * \param reportConfig LteRrcSap::ReportConfigEutra
   * \returns measure ID
   */
  uint8_t DoAddUeMeasReportConfigForComponentCarrier (LteRrcSap::ReportConfigEutra reportConfig);
  /**
   * \brief Set number of component carriers
   * \param numberOfComponentCarriers the number of component carriers
   */
  void DoSetNumberOfComponentCarriers (uint16_t numberOfComponentCarriers);


  /**
   * Trigger handover function
   *
   * \param rnti RNTI
   * \param targetCellId target cell ID
   */
  void DoTriggerHandover (uint16_t rnti, uint16_t targetCellId);

  // ANR SAP methods

  /**
   * Add UE measure report config for ANR function
   *
   * \param reportConfig LteRrcSap::ReportConfigEutra
   * \returns measure ID
   */
  uint8_t DoAddUeMeasReportConfigForAnr (LteRrcSap::ReportConfigEutra reportConfig);

  // FFR RRC SAP methods
  /**
   * Add UE measure report config for FFR function
   *
   * \param reportConfig LteRrcSap::ReportConfigEutra
   * \returns measure ID
   */
  uint8_t DoAddUeMeasReportConfigForFfr (LteRrcSap::ReportConfigEutra reportConfig);
  /**
   * Set PDSCH config dedicated function
   *
   * \param rnti the RNTI
   * \param pa LteRrcSap::PdschConfigDedicated
   */
  void DoSetPdschConfigDedicated (uint16_t rnti, LteRrcSap::PdschConfigDedicated pa);
  /**
   * Send load information function
   *
   * \param params EpcX2Sap::LoadInformationParams
   */
  void DoSendLoadInformation (EpcX2Sap::LoadInformationParams params);

  // Internal methods

  /**
   * Allocate a new RNTI for a new UE. This is done in the following cases:
   *   * T-C-RNTI allocation upon contention-based MAC Random Access procedure
   *   * target cell RNTI allocation upon handover
   *
   * \param state the initial state of the UeManager
   * \param componentCarrierId primary component carrier ID of the UeManager
   *
   * \return the newly allocated RNTI
   */
  uint16_t AddUe (UeManager::State state, uint8_t componentCarrierId);

  /**
   * remove a UE from the cell
   *
   * \param rnti the C-RNTI identiftying the user
   */
  void RemoveUe (uint16_t rnti);


  /** 
   * 
   * \param bearer the specification of an EPS bearer
   * 
   * \return the type of RLC that is to be created for the given EPS bearer
   */
  TypeId GetRlcType (EpsBearer bearer);

  /**
   * \brief Is random access completed function
   *
   * This method is executed to decide if the non contention based
   * preamble has to reused or not upon preamble expiry. If the random access
   * in connected mode is completed, then the preamble can be reused by other UEs.
   * If not, the same UE retains the preamble and other available preambles is
   * assigned to the required UEs.
   *
   * \param rnti the C-RNTI identifying the user
   * \return true if the random access in connected mode is completed
   */
  bool IsRandomAccessCompleted (uint16_t rnti);



public:

  /** 
   * Add a neighbour with an X2 interface
   *
   * \param cellId neighbouring cell id
   */
  void AddX2Neighbour (uint16_t cellId);

  /** 
   * 
   * \param p the SRS periodicity in num TTIs
   */
  void SetSrsPeriodicity (uint32_t p);

  /** 
   *
   * \return the current SRS periodicity
   */
  uint32_t GetSrsPeriodicity () const;

  /**
   * \brief Associate this RRC entity with a particular CSG information.
   * \param csgId the intended Closed Subscriber Group identity
   * \param csgIndication if TRUE, only CSG members are allowed to access the
   *                      cell
   *
   * CSG identity is a number identifying a Closed Subscriber Group which the
   * cell belongs to. eNodeB is associated with a single CSG identity.
   *
   * The same CSG identity can also be associated to several UEs, which is
   * equivalent as enlisting these UEs as the members of this particular CSG.
   * When the CSG indication field is set to TRUE, only UEs which are members of
   * the CSG (i.e. same CSG ID) can gain access to the eNodeB, therefore
   * enforcing closed access mode. Otherwise, the eNodeB operates as a non-CSG
   * cell and implements open access mode.
   *
   * This restriction only applies to initial cell selection and EPC-enabled
   * simulation.
   */
  void SetCsgId (uint32_t csgId, bool csgIndication);

private:

  /** 
   * Allocate a new SRS configuration index for a new UE. 
   *
   * \note this method can have the side effect of updating the SRS
   * configuration index of all UEs
   * 
   * \return the newly allocated SRS configuration index
   */
  uint16_t GetNewSrsConfigurationIndex (void);

  /** 
   * remove a previously allocated SRS configuration index
   *
   * \note this method can have the side effect of updating the SRS
   * configuration index of all UEs
   * 
   * \param srcCi the indext to be removed
   */
  void RemoveSrsConfigurationIndex (uint16_t srcCi);

 

  /** 
   * 
   * \param bearer the characteristics of the bearer
   * 
   * \return the Logical Channel Group in a bearer with these
   * characteristics is put. Used for MAC Buffer Status Reporting purposes. 
   */
  uint8_t GetLogicalChannelGroup (EpsBearer bearer);

  /** 
   * 
   * \param bearer the characteristics of the bearer
   * 
   * \return the priority level of a bearer with these
   * characteristics is put. Used for the part of UL MAC Scheduling
   * carried out by the UE
   */
  uint8_t GetLogicalChannelPriority (EpsBearer bearer);


  /** 
   * method used to periodically send System Information
   * 
   */
  void SendSystemInformation ();

  Callback <void, Ptr<Packet> > m_forwardUpCallback;  ///< forward up callback function

  /// Interface to receive messages from neighbour eNodeB over the X2 interface.
  EpcX2SapUser* m_x2SapUser;
  /// Interface to send messages to neighbour eNodeB over the X2 interface.
  EpcX2SapProvider* m_x2SapProvider;

  /// Receive API calls from the eNodeB MAC instance.
  std::vector<LteEnbCmacSapUser*> m_cmacSapUser;
  /// Interface to the eNodeB MAC instance.
  std::vector<LteEnbCmacSapProvider*> m_cmacSapProvider;

  /// Receive API calls from the handover algorithm instance.
  LteHandoverManagementSapUser* m_handoverManagementSapUser;
  /// Interface to the handover algorithm instance.
  LteHandoverManagementSapProvider* m_handoverManagementSapProvider;

  /// Receive API calls from the LteEnbComponentCarrierManager instance.
  LteCcmRrcSapUser* m_ccmRrcSapUser;
  /// Interface to the LteEnbComponentCarrierManager instance.
  LteCcmRrcSapProvider* m_ccmRrcSapProvider;

  /// Receive API calls from the ANR instance.
  LteAnrSapUser* m_anrSapUser;
  /// Interface to the ANR instance.
  LteAnrSapProvider* m_anrSapProvider;

  /// Receive API calls from the FFR algorithm instance.
  std::vector<LteFfrRrcSapUser*> m_ffrRrcSapUser;
  /// Interface to the FFR algorithm instance.
  std::vector<LteFfrRrcSapProvider*> m_ffrRrcSapProvider;

  /// Interface to send messages to UE over the RRC protocol.
  LteEnbRrcSapUser* m_rrcSapUser;
  /// Interface to receive messages from UE over the RRC protocol.
  LteEnbRrcSapProvider* m_rrcSapProvider;

  /// Interface to the eNodeB MAC instance, to be used by RLC instances.
  LteMacSapProvider* m_macSapProvider;

  /// Interface to send messages to core network over the S1 protocol.
  EpcEnbS1SapProvider* m_s1SapProvider;
  /// Interface to receive messages from core network over the S1 protocol.
  EpcEnbS1SapUser* m_s1SapUser;

  /// Receive API calls from the eNodeB PHY instances.
  std::vector<LteEnbCphySapUser*> m_cphySapUser;
  /// Interface to the eNodeB PHY instances.
  std::vector<LteEnbCphySapProvider*> m_cphySapProvider;

  /// True if ConfigureCell() has been completed.
  bool m_configured;
  /// Downlink E-UTRA Absolute Radio Frequency Channel Number.
  uint32_t m_dlEarfcn;
  /// Uplink E-UTRA Absolute Radio Frequency Channel Number.
  uint32_t m_ulEarfcn;
  /// Downlink transmission bandwidth configuration in number of Resource Blocks.
  uint16_t m_dlBandwidth;
  /// Uplink transmission bandwidth configuration in number of Resource Blocks.
  uint16_t m_ulBandwidth;
  /// Last allocated RNTI
  uint16_t m_lastAllocatedRnti;

  /// The System Information Block Type 1 that is currently broadcasted over BCH.
  std::vector<LteRrcSap::SystemInformationBlockType1> m_sib1;

  /**
   * The `UeMap` attribute. List of UeManager by C-RNTI.
   */
  std::map<uint16_t, Ptr<UeManager> > m_ueMap;

  /**
   * List of measurement configuration which are active in every UE attached to
   * this eNodeB instance.
   */
  LteRrcSap::MeasConfig m_ueMeasConfig;

  /// List of measurement identities which are intended for handover purpose.
  std::set<uint8_t> m_handoverMeasIds;
  /// List of measurement identities which are intended for ANR purpose.
  std::set<uint8_t> m_anrMeasIds;
  /// List of measurement identities which are intended for FFR purpose.
  std::set<uint8_t> m_ffrMeasIds;
  /// List of measurement identities which are intended for component carrier management purposes.
  std::set<uint8_t> m_componentCarrierMeasIds;

  /// X2uTeidInfo structure
  struct X2uTeidInfo
  {
    uint16_t rnti; ///< RNTI
    uint8_t drbid; ///< DRBID
  };

  /// TEID, RNTI, DRBID
  std::map<uint32_t, X2uTeidInfo> m_x2uTeidInfoMap;

  /**
   * The `DefaultTransmissionMode` attribute. The default UEs' transmission
   * mode (0: SISO).
   */
  uint8_t m_defaultTransmissionMode;
  /**
   * The `EpsBearerToRlcMapping` attribute. Specify which type of RLC will be
   * used for each type of EPS bearer.
   */
  enum LteEpsBearerToRlcMapping_t m_epsBearerToRlcMapping;
  /**
   * The `SystemInformationPeriodicity` attribute. The interval for sending
   * system information.
   */
  Time m_systemInformationPeriodicity;
  /**
   * The `SrsPeriodicity` attribute. The SRS periodicity in milliseconds.
   */
  uint16_t m_srsCurrentPeriodicityId; ///< SRS current periodicity ID
  std::set<uint16_t> m_ueSrsConfigurationIndexSet; ///< UE SRS configuration index set
  uint16_t m_lastAllocatedConfigurationIndex; ///< last allocated configuration index
  bool m_reconfigureUes; ///< reconfigure UEs?

  /**
   * The `QRxLevMin` attribute. One of information transmitted within the SIB1
   * message, indicating the required minimum RSRP level that any UE must
   * receive from this cell before it is allowed to camp to this cell.
   */
  int8_t m_qRxLevMin;
  /**
   * The `AdmitHandoverRequest` attribute. Whether to admit an X2 handover
   * request from another eNB.
   */
  bool m_admitHandoverRequest;
  /**
   * The `AdmitRrcConnectionRequest` attribute. Whether to admit a connection
   * request from a UE.
   */
  bool m_admitRrcConnectionRequest;
  /**
   * The `RsrpFilterCoefficient` attribute. Determines the strength of
   * smoothing effect induced by layer 3 filtering of RSRP in all attached UE.
   * If equals to 0, no layer 3 filtering is applicable.
   */
  uint8_t m_rsrpFilterCoefficient;
  /**
   * The `RsrqFilterCoefficient` attribute. Determines the strength of
   * smoothing effect induced by layer 3 filtering of RSRQ in all attached UE.
   * If equals to 0, no layer 3 filtering is applicable.
   */
  uint8_t m_rsrqFilterCoefficient;
  /**
   * The `ConnectionRequestTimeoutDuration` attribute. After a RA attempt, if
   * no RRC CONNECTION REQUEST is received before this time, the UE context is
   * destroyed. Must account for reception of RAR and transmission of RRC
   * CONNECTION REQUEST over UL GRANT.
   */
  Time m_connectionRequestTimeoutDuration;
  /**
   * The `ConnectionSetupTimeoutDuration` attribute. After accepting connection
   * request, if no RRC CONNECTION SETUP COMPLETE is received before this time,
   * the UE context is destroyed. Must account for the UE's reception of RRC
   * CONNECTION SETUP and transmission of RRC CONNECTION SETUP COMPLETE.
   */
  Time m_connectionSetupTimeoutDuration;
  /**
   * The `ConnectionRejectedTimeoutDuration` attribute. Time to wait between
   * sending a RRC CONNECTION REJECT and destroying the UE context.
   */
  Time m_connectionRejectedTimeoutDuration;
  /**
   * The `HandoverJoiningTimeoutDuration` attribute. After accepting a handover
   * request, if no RRC CONNECTION RECONFIGURATION COMPLETE is received before
   * this time, the UE context is destroyed. Must account for reception of X2
   * HO REQ ACK by source eNB, transmission of the Handover Command,
   * non-contention-based random access and reception of the RRC CONNECTION
   * RECONFIGURATION COMPLETE message.
   */
  Time m_handoverJoiningTimeoutDuration;
  /**
   * The `HandoverLeavingTimeoutDuration` attribute. After issuing a Handover
   * Command, if neither RRC CONNECTION RE-ESTABLISHMENT nor X2 UE Context
   * Release has been previously received, the UE context is destroyed.
   */
  Time m_handoverLeavingTimeoutDuration;

  /**
   * The `NewUeContext` trace source. Fired upon creation of a new UE context.
   * Exporting cell ID and RNTI.
   */
  TracedCallback<uint16_t, uint16_t> m_newUeContextTrace;
  /**
   * The `ConnectionEstablished` trace source. Fired upon successful RRC
   * connection establishment. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_connectionEstablishedTrace;
  /**
   * The `ConnectionReconfiguration` trace source. Fired upon RRC connection
   * reconfiguration. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_connectionReconfigurationTrace;
  /**
   * The `HandoverStart` trace source. Fired upon start of a handover
   * procedure. Exporting IMSI, cell ID, RNTI, and target cell ID.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, uint16_t> m_handoverStartTrace;
  /**
   * The `HandoverEndOk` trace source. Fired upon successful termination of a
   * handover procedure. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_handoverEndOkTrace;
  /**
   * The `RecvMeasurementReport` trace source. Fired when measurement report is
   * received. Exporting IMSI, cell ID, and RNTI.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, LteRrcSap::MeasurementReport> m_recvMeasurementReportTrace;
  /**
   * The `NotifyConnectionRelease` trace source. Fired when an UE leaves the eNB.
   * Exporting IMSI, cell ID, RNTI.
   *
   */
  TracedCallback<uint64_t, uint16_t, uint16_t> m_connectionReleaseTrace;
  /**
   * The 'TimerExpiry' Trace source. Fired when any of the RRC timers maintained
   * at eNB expires. Exporting IMSI, cell ID, and RNTI and name of timer
   * which expired.
   */
  TracedCallback<uint64_t, uint16_t, uint16_t, std::string> m_rrcTimeoutTrace;

  uint16_t m_numberOfComponentCarriers; ///< number of component carriers

  bool m_carriersConfigured; ///< are carriers configured

  std::map<uint8_t, Ptr<ComponentCarrierBaseStation>> m_componentCarrierPhyConf; ///< component carrier phy configuration

}; // end of `class LteEnbRrc`


} // namespace ns3

#endif // LTE_ENB_RRC_H
