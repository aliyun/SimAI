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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */


#ifndef LTE_RRC_PROTOCOL_IDEAL_H
#define LTE_RRC_PROTOCOL_IDEAL_H

#include <stdint.h>
#include <map>

#include <ns3/ptr.h>
#include <ns3/object.h>
#include <ns3/lte-rrc-sap.h>

namespace ns3 {

class LteUeRrcSapProvider;
class LteUeRrcSapUser;
class LteEnbRrcSapProvider;
class LteUeRrc;


/**
 * \ingroup lte
 *
 * Models the transmission of RRC messages from the UE to the eNB in
 * an ideal fashion, without errors and without consuming any radio
 * resources. 
 * 
 */
class LteUeRrcProtocolIdeal : public Object
{
  /// allow MemberLteUeRrcSapUser<LteUeRrcProtocolIdeal> class friend access
  friend class MemberLteUeRrcSapUser<LteUeRrcProtocolIdeal>;

public:

  LteUeRrcProtocolIdeal ();
  virtual ~LteUeRrcProtocolIdeal ();

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
   * \param p LTE UE RRC SAP provider
   */
  void SetLteUeRrcSapProvider (LteUeRrcSapProvider* p);
  /**
   * Get LTE UE RRC SAP user function
   *
   * \returns LTE UE RRC SAP user
   */
  LteUeRrcSapUser* GetLteUeRrcSapUser ();
  
  /**
   * Set LTE UE RRC  function
   *
   * \param rrc LTE UE RRC 
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
   * Send RRC connection reconfiguration completed function
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
   * \param msg LteRrcSap::RrcConnectionReestablishmentRequest 
   */
  void DoSendRrcConnectionReestablishmentComplete (LteRrcSap::RrcConnectionReestablishmentComplete msg);
  /**
   * Send measurement report function
   *
   * \param msg LteRrcSap::MeasurementReport 
   */
  void DoSendMeasurementReport (LteRrcSap::MeasurementReport msg);

  /**
   * \brief Send Ideal UE context remove request function
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

  Ptr<LteUeRrc> m_rrc; ///< the RRC
  uint16_t m_rnti; ///< the RNTI
  LteUeRrcSapProvider* m_ueRrcSapProvider; ///< the UE RRC SAP provider
  LteUeRrcSapUser* m_ueRrcSapUser; ///< the RRC SAP user
  LteEnbRrcSapProvider* m_enbRrcSapProvider; ///< the ENB RRC SAP provider
  
};


/**
 * Models the transmission of RRC messages from the UE to the eNB in
 * an ideal fashion, without errors and without consuming any radio
 * resources. 
 * 
 */
class LteEnbRrcProtocolIdeal : public Object
{
  /// allow MemberLteEnbRrcSapUser<LteEnbRrcProtocolIdeal> class friend access
  friend class MemberLteEnbRrcSapUser<LteEnbRrcProtocolIdeal>;

public:

  LteEnbRrcProtocolIdeal ();
  virtual ~LteEnbRrcProtocolIdeal ();

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
   * \param p the LTE ENB RRC SAP provider 
   */
  void SetLteEnbRrcSapProvider (LteEnbRrcSapProvider* p);
  /**
   * Get LTE ENB RRC SAP user function
   *
   * \returns LTE ENB RRC SAP user
   */
  LteEnbRrcSapUser* GetLteEnbRrcSapUser ();

  /**
   * Set the cell ID function
   *
   * \param cellId the cell ID 
   */
  void SetCellId (uint16_t cellId);

  /**
   * Get LTE UE RRC SAP provider function
   *
   * \param rnti the RNTI
   * \returns LTE UE RRC SAP provider
   */
  LteUeRrcSapProvider* GetUeRrcSapProvider (uint16_t rnti);
  /**
   * Set UE RRC SAP provider function
   *
   * \param rnti the RNTI
   * \param p the UE RRC SAP provider 
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
   * \param msg LteRrcSap::SystemInformation
   */
  void SendSystemInformation (LteRrcSap::SystemInformation msg);
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
   * Send RRC connection reestablishment function
   *
   * \param rnti the RNTI
   * \param msg LteRrcSap::RrcConnectionReestablishment
   */
  void DoSendRrcConnectionReestablishment (uint16_t rnti, LteRrcSap::RrcConnectionReestablishment msg);
  /**
   * Send RRC connection reestablishment reject function
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
   * Encode handover preparation information function
   *
   * \param p the packet
   * \returns LteRrcSap::HandoverPreparationInfo
   */
  LteRrcSap::HandoverPreparationInfo DoDecodeHandoverPreparationInformation (Ptr<Packet> p);
  /**
   * Encode handover command function
   *
   * \param msg LteRrcSap::RrcConnectionReconfiguration
   * \returns rnti the RNTI
   */
  Ptr<Packet> DoEncodeHandoverCommand (LteRrcSap::RrcConnectionReconfiguration msg);
  /**
   * Decode handover command function
   *
   * \param p the packet
   * \returns LteRrcSap::RrcConnectionReconfiguration
   */
  LteRrcSap::RrcConnectionReconfiguration DoDecodeHandoverCommand (Ptr<Packet> p);


  uint16_t m_rnti; ///< the RNTI
  uint16_t m_cellId; ///< the cell ID
  LteEnbRrcSapProvider* m_enbRrcSapProvider; ///< the ENB RRC SAP provider
  LteEnbRrcSapUser* m_enbRrcSapUser; ///< the ENB RRC SAP user
  std::map<uint16_t, LteUeRrcSapProvider*> m_enbRrcSapProviderMap; ///< the LTE UE RRC SAP provider
  
};



}


#endif // LTE_RRC_PROTOCOL_IDEAL_H
