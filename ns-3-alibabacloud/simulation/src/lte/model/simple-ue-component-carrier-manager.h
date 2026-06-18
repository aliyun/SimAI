/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 *
 */

#ifndef SIMPLE_UE_COMPONENT_CARRIER_MANAGER_H
#define SIMPLE_UE_COMPONENT_CARRIER_MANAGER_H

#include <ns3/lte-ue-component-carrier-manager.h>
#include <ns3/lte-ue-ccm-rrc-sap.h>
#include <ns3/lte-rrc-sap.h>
#include <map>

namespace ns3 {
  class LteUeCcmRrcSapProvider;

/**
 * \brief Component carrier manager implementation which simply does nothing.
 *
 * Selecting this component carrier selection algorithm is equivalent to disabling automatic
 * triggering of component carrier selection. This is the default choice.
 *
 */
class SimpleUeComponentCarrierManager : public LteUeComponentCarrierManager
{
public:
  /// Creates a No-op CCS algorithm instance.
  SimpleUeComponentCarrierManager ();

  virtual ~SimpleUeComponentCarrierManager ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();

  // inherited from LteComponentCarrierManager
  virtual LteMacSapProvider* GetLteMacSapProvider ();



  /// let the forwarder class access the protected and private members
  friend class MemberLteUeCcmRrcSapProvider<SimpleUeComponentCarrierManager>;
  //friend class MemberLteUeCcmRrcSapUser<SimpleUeComponentCarrierManager>;
  
  /// allow SimpleUeCcmMacSapProvider class friend access
  friend class SimpleUeCcmMacSapProvider;
  /// allow SimpleUeCcmMacSapUser class friend access
  friend class SimpleUeCcmMacSapUser;

protected:

  // inherited from Object
  virtual void DoInitialize ();
  virtual void DoDispose ();
  // inherited from LteCcsAlgorithm as a Component Carrier Management SAP implementation
  /**
   * \brief Report Ue Measure function
   * \param rnti the RNTI
   * \param measResults the measure results
   */
  void DoReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults);
  // forwarded from LteMacSapProvider
  /**
   * \brief Transmit PDU function
   * \param params LteMacSapProvider::TransmitPduParameters
   */
  void DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params);
  /**
   * \brief Report buffer status function
   * \param params LteMacSapProvider::ReportBufferStatusParameters
   */
  virtual void DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params);
  /// Notify HARQ deliver failure
  void DoNotifyHarqDeliveryFailure ();
  // forwarded from LteMacSapUser
  /**
   * \brief Notify TX opportunity function
   *
   * \param txOpParams the LteMacSapUser::TxOpportunityParameters
   */
  void DoNotifyTxOpportunity (LteMacSapUser::TxOpportunityParameters txOpParams);
  /**
   * \brief Receive PDU function
   *
   * \param rxPduParams the LteMacSapUser::ReceivePduParameters
   */
  void DoReceivePdu (LteMacSapUser::ReceivePduParameters rxPduParams);
  //forwarded from LteUeCcmRrcSapProvider
  /**
   * \brief Add LC function
   * \param lcId the LCID
   * \param lcConfig the logical channel config
   * \param msu the MSU
   * \returns updated LC config list
   */
  virtual std::vector<LteUeCcmRrcSapProvider::LcsConfig> DoAddLc (uint8_t lcId,  LteUeCmacSapProvider::LogicalChannelConfig lcConfig, LteMacSapUser* msu);
  /**
   * \brief Remove LC function
   * \param lcid the LCID
   * \returns updated LC list
   */
  std::vector<uint16_t> DoRemoveLc (uint8_t lcid);
  /**
   * \brief Configure signal bearer function
   * \param lcId the LCID
   * \param lcConfig the logical channel config
   * \param msu the MSU
   * \returns LteMacSapUser *
   */
  virtual LteMacSapUser* DoConfigureSignalBearer (uint8_t lcId,  LteUeCmacSapProvider::LogicalChannelConfig lcConfig, LteMacSapUser* msu);
  /**
   * \brief Reset LC map
   *
   */
  void DoReset ();
  
protected:
  
  LteMacSapUser* m_ccmMacSapUser;//!< Interface to the UE RLC instance.
  LteMacSapProvider* m_ccmMacSapProvider; //!< Receive API calls from the UE RLC instance

}; // end of class SimpleUeComponentCarrierManager


} // end of namespace ns3


#endif /* SIMPLE_UE_COMPONENT_CARRIER_MANAGER_H */
