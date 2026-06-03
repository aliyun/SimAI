/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 *          Lluis Parcerisa <lparcerisa@cttc.cat>
 */


#ifndef LTE_RRC_PROTOCOL_REAL_H
#define LTE_RRC_PROTOCOL_REAL_H

#include <stdint.h>
#include <map>

#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/lte-rrc-sap.h>
#include <ns3/lte-pdcp-sap.h>
#include <ns3/lte-rlc-sap.h>
#include <ns3/lte-rrc-header.h>

namespace ns3 {

class LteUeRrcSapProvider;
class LteUeRrcSapUser;
class LteEnbRrcSapProvider;
class LteUeRrc;


/**
 * \ingroup lte
 *
 * Models the transmission of RRC messages from the UE to the eNB in
 * a real fashion, by creating real RRC PDUs and transmitting them
 * over Signaling Radio Bearers using radio resources allocated by the
 * LTE MAC scheduler.
 * 
 */
class LteUeRrcProtocolReal : public Object
{
  /// allow MemberLteUeRrcSapUser<LteUeRrcProtocolReal> class friend access
  friend class MemberLteUeRrcSapUser<LteUeRrcProtocolReal>;
  /// allow LteRlcSpecificLteRlcSapUser<LteUeRrcProtocolReal> class friend access
  friend class LteRlcSpecificLteRlcSapUser<LteUeRrcProtocolReal>;
  /// allow LtePdcpSpecificLtePdcpSapUser<LteUeRrcProtocolReal> class friend access
  friend class LtePdcpSpecificLtePdcpSapUser<LteUeRrcProtocolReal>;

public:
  LteUeRrcProtocolReal ();
  virtual ~LteUeRrcProtocolReal ();

  // inherited from Object
  virtual void DoDispose (void);
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Set LTE UE RRC SAP provider function
   *
   * \param p the LTE UE RRC SAP provider 
   */
  void SetLteUeRrcSapProvider (LteUeRrcSapProvider* p);
  /**
   * Get LTE UE RRC SAP user function
   *
   * \returns LTE UE RRC SAP user
   */
  LteUeRrcSapUser* GetLteUeRrcSapUser ();

  /**
   * Set UE RRC function
   *
   * \param rrc the LTE UE RRC 
   */
  void SetUeRrc (Ptr<LteUeRrc> rrc);


private:
  // methods forwarded from LteUeRrcSapUser
  /**
   * Setup function
   *
   * \param params LteUeRrcSapUser::SetupParameters 
   */
  void DoSetup (LteUeRrcSapUser::SetupParameters params);
  /**
   * Send RRC connection request function
   *
   * \param msg LteRrcSap::RrcConnectionRequest 
   */
  void DoSendRrcConnectionRequest (LteRrcSap::RrcConnectionRequest msg);
  /**
   * Send RRC connection setup completed function
   *
   * \param msg LteRrcSap::RrcConnectionSetupCompleted 
   */
  void DoSendRrcConnectionSetupCompleted (LteRrcSap::RrcConnectionSetupCompleted msg);
  /**
   * Send RRC connection reconfiguration setup completed function
   *
   * \param msg LteRrcSap::RrcConnectionReconfigurationCompleted 
   */
  void DoSendRrcConnectionReconfigurationCompleted (LteRrcSap::RrcConnectionReconfigurationCompleted msg);
  /**
   * Send RRC connection reestablishment request function
   *
   * \param msg LteRrcSap::RrcConnectionReestablishmentRequest 
   */
  void DoSendRrcConnectionReestablishmentRequest (LteRrcSap::RrcConnectionReestablishmentRequest msg);
  /**
   * Send RRC connection reestablishment complete function
   *
   * \param msg LteRrcSap::RrcConnectionReestablishmentComplete 
   */
  void DoSendRrcConnectionReestablishmentComplete (LteRrcSap::RrcConnectionReestablishmentComplete msg);
  /**
   * Send measurement report function
   *
   * \param msg LteRrcSap::MeasurementReport 
   */
  void DoSendMeasurementReport (LteRrcSap::MeasurementReport msg);
  /**
   * \brief Send ideal UE context remove request function
   *
   * Notify eNodeB to release UE context once radio link failure
   * or random access failure is detected. It is needed since no
   * RLF detection mechanism at eNodeB is implemented
   *
   * \param rnti the RNTI of the UE
   */
  void DoSendIdealUeContextRemoveRequest (uint16_t rnti);

  /// Set ENB RRC SAP provider
  void SetEnbRrcSapProvider ();
  /**
   * Receive PDCP PDU function
   *
   * \param p the packet 
   */
  void DoReceivePdcpPdu (Ptr<Packet> p);
  /**
   * Receive PDCP SDU function
   *
   * \param params LtePdcpSapUser::ReceivePdcpSduParameters 
   */
  void DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params);

  Ptr<LteUeRrc> m_rrc; ///< the RRC
  uint16_t m_rnti; ///< the RNTI
  LteUeRrcSapProvider* m_ueRrcSapProvider; ///< UE RRC SAP provider
  LteUeRrcSapUser* m_ueRrcSapUser; ///< UE RRC SAP user
  LteEnbRrcSapProvider* m_enbRrcSapProvider; ///< ENB RRC SAP provider

  LteUeRrcSapUser::SetupParameters m_setupParameters; ///< setup parameters
  LteUeRrcSapProvider::CompleteSetupParameters m_completeSetupParameters; ///< complete setup parameters

};


/**
 * Models the transmission of RRC messages from the UE to the eNB in
 * a real fashion, by creating real RRC PDUs and transmitting them
 * over Signaling Radio Bearers using radio resources allocated by the
 * LTE MAC scheduler.
 *
 */
class LteEnbRrcProtocolReal : public Object
{
  /// allow MemberLteEnbRrcSapUser<LteEnbRrcProtocolReal> class friend access
  friend class MemberLteEnbRrcSapUser<LteEnbRrcProtocolReal>;
  /// allow LtePdcpSpecificLtePdcpSapUser<LteEnbRrcProtocolReal> class friend access
  friend class LtePdcpSpecificLtePdcpSapUser<LteEnbRrcProtocolReal>;
  /// allow LteRlcSpecificLteRlcSapUser<LteEnbRrcProtocolReal> class friend access
  friend class LteRlcSpecificLteRlcSapUser<LteEnbRrcProtocolReal>;
  /// allow RealProtocolRlcSapUser class friend access
  friend class RealProtocolRlcSapUser;

public:
  LteEnbRrcProtocolReal ();
  virtual ~LteEnbRrcProtocolReal ();

  // inherited from Object
  virtual void DoDispose (void);
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Set LTE ENB RRC SAP provider function
   *
   * \param p LteEnbRrcSapProvider * 
   */
  void SetLteEnbRrcSapProvider (LteEnbRrcSapProvider* p);
  /**
   * Get LTE ENB RRC SAP user function
   *
   * \returns LteEnbRrcSapUser * 
   */
  LteEnbRrcSapUser* GetLteEnbRrcSapUser ();

  /**
   * Set cell ID function
   *
   * \param cellId the cell ID 
   */
  void SetCellId (uint16_t cellId);

  /**
   * Get UE RRC SAP provider function
   *
   * \param rnti the RNTI
   * \returns LteUeRrcSapProvider * 
   */
  LteUeRrcSapProvider* GetUeRrcSapProvider (uint16_t rnti);
  /**
   * Set UE RRC SAP provider function
   *
   * \param rnti the RNTI
   * \param p LteUeRrcSapProvider *
   */
  void SetUeRrcSapProvider (uint16_t rnti, LteUeRrcSapProvider* p);

private:
  // methods forwarded from LteEnbRrcSapUser
  /**
   * Setup UE function
   *
   * \param rnti the RNTI
   * \param params LteEnbRrcSapUser::SetupUeParameters
   */
  void DoSetupUe (uint16_t rnti, LteEnbRrcSapUser::SetupUeParameters params);
  /**
   * Remove UE function
   *
   * \param rnti the RNTI
   */
  void DoRemoveUe (uint16_t rnti);
  /**
   * Send system information function
   *
   * \param cellId cell ID
   * \param msg LteRrcSap::SystemInformation
   */
  void DoSendSystemInformation (uint16_t cellId, LteRrcSap::SystemInformation msg);
  /**
   * Send system information function
   *
   * \param cellId cell ID
   * \param msg LteRrcSap::SystemInformation
   */
  void SendSystemInformation (uint16_t cellId, LteRrcSap::SystemInformation msg);
  /**
   * Send RRC connection setup function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionSetup
   */
  void DoSendRrcConnectionSetup (uint16_t rnti, LteRrcSap::RrcConnectionSetup msg);
  /**
   * Send RRC connection reconfiguration function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionReconfiguration
   */
  void DoSendRrcConnectionReconfiguration (uint16_t rnti, LteRrcSap::RrcConnectionReconfiguration msg);
  /**
   * Send RRC connection reestabishment function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionReestablishment
   */
  void DoSendRrcConnectionReestablishment (uint16_t rnti, LteRrcSap::RrcConnectionReestablishment msg);
  /**
   * Send RRC connection reestabishment reject function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionReestablishmentReject
   */
  void DoSendRrcConnectionReestablishmentReject (uint16_t rnti, LteRrcSap::RrcConnectionReestablishmentReject msg);
  /**
   * Send RRC connection release function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionRelease
   */
  void DoSendRrcConnectionRelease (uint16_t rnti, LteRrcSap::RrcConnectionRelease msg);
  /**
   * Send RRC connection reject function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionReject
   */
  void DoSendRrcConnectionReject (uint16_t rnti, LteRrcSap::RrcConnectionReject msg);
  /**
   * Encode handover preparation information function
   *
   * \param msg LteRrcSap::HandoverPreparationInfo
   * \returns the packet
   */
  Ptr<Packet> DoEncodeHandoverPreparationInformation (LteRrcSap::HandoverPreparationInfo msg);
  /**
   * Decode handover preparation information function
   *
   * \param p the packet
   * \returns LteRrcSap::HandoverPreparationInfo
   */
  LteRrcSap::HandoverPreparationInfo DoDecodeHandoverPreparationInformation (Ptr<Packet> p);
  /**
   * Encode handover command function
   *
   * \param msg LteRrcSap::RrcConnectionReconfiguration
   * \returns the packet
   */
  Ptr<Packet> DoEncodeHandoverCommand (LteRrcSap::RrcConnectionReconfiguration msg);
  /**
   * Decode handover command function
   *
   * \param p the packet
   * \returns LteRrcSap::RrcConnectionReconfiguration
   */
  LteRrcSap::RrcConnectionReconfiguration DoDecodeHandoverCommand (Ptr<Packet> p);

  /**
   * Receive PDCP SDU function
   *
   * \param params LtePdcpSapUser::ReceivePdcpSduParameters
   */
  void DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params);
  /**
   * Receive PDCP PDU function
   *
   * \param rnti the RNTI
   * \param p the packet
   */
  void DoReceivePdcpPdu (uint16_t rnti, Ptr<Packet> p);

  uint16_t m_rnti; ///< the RNTI
  uint16_t m_cellId; ///< the cell ID
  LteEnbRrcSapProvider* m_enbRrcSapProvider; ///< ENB RRC SAP provider
  LteEnbRrcSapUser* m_enbRrcSapUser; ///< ENB RRC SAP user
  std::map<uint16_t, LteUeRrcSapProvider*> m_enbRrcSapProviderMap; ///< ENB RRC SAP provider map
  std::map<uint16_t, LteEnbRrcSapUser::SetupUeParameters> m_setupUeParametersMap; ///< setup UE parameters map
  std::map<uint16_t, LteEnbRrcSapProvider::CompleteSetupUeParameters> m_completeSetupUeParametersMap; ///< complete setup UE parameters map

};


/// RealProtocolRlcSapUser class
class RealProtocolRlcSapUser : public LteRlcSapUser
{
public:
  /**
   * Real protocol RC SAP user
   *
   * \param pdcp LteEnbRrcProtocolReal *
   * \param rnti the RNTI
   */
  RealProtocolRlcSapUser (LteEnbRrcProtocolReal* pdcp, uint16_t rnti);

  // Interface implemented from LteRlcSapUser
  virtual void ReceivePdcpPdu (Ptr<Packet> p);

private:
  RealProtocolRlcSapUser ();
  LteEnbRrcProtocolReal* m_pdcp; ///< PDCP
  uint16_t m_rnti; ///< RNTI
};


}


#endif // LTE_RRC_PROTOCOL_REAL_H
