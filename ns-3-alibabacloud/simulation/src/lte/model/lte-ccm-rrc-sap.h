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

#ifndef LTE_CCM_RRC_SAP_H
#define LTE_CCM_RRC_SAP_H

#include <ns3/lte-rrc-sap.h>
#include <ns3/eps-bearer.h>
#include <ns3/lte-enb-cmac-sap.h>
#include <ns3/lte-mac-sap.h>
#include <map>


namespace ns3 {
  class LteUeCmacSapProvider;
  class UeManager;
  class LteEnbCmacSapProvider;
  class LteMacSapUser;
  class LteRrcSap;

/**
 * \brief Service Access Point (SAP) offered by the Component Carrier Manager (CCM)
 * instance to the eNodeB RRC instance.
 *
 * This is the *Component Carrier Manager SAP Provider*, i.e., the part of the SAP
 * that contains the CCM methods called by the eNodeB RRC instance.
 */
class LteCcmRrcSapProvider
{

/// allow UeManager class friend access
friend class UeManager;
/// allow LteMacSapUser class friend access
friend class LteMacSapUser;
 
public:
  
  virtual ~LteCcmRrcSapProvider ();
  
  /// LcsConfig structure
  struct LcsConfig
  {
    uint16_t componentCarrierId; ///< component carrier ID
    LteEnbCmacSapProvider::LcInfo lc; ///< LC info
    LteMacSapUser *msu; ///< MSU
  };

  /**
   * \brief Reports UE measurements to the component carrier manager.
   * \param rnti Radio Network Temporary Identity, an integer identifying 
   * the UE where the measurement report originates from.
   * \param measResults a single report of one measurement identity
   *
   * The received measurement report is a result of the UE measurements configuration
   * previously configured by calling LteCcmRrcSapProvider::AddUeMeasReportConfigForComponentCarrier.
   * The report may be stored and utilized for the purpose of making decision if and when
   * to use the secondary carriers.
   */
  virtual void ReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults) = 0;

  /**
   * \brief Add a new UE in the LteEnbComponentCarrierManager.
   * \param rnti Radio Network Temporary Identity, an integer identifying the UE.
   * \param state The current rrc state of the UE.
   */
  virtual void AddUe (uint16_t rnti, uint8_t state) = 0;

  /**
   * \brief Add a new logical channel.
   * \param lcInfo - information about newly created logical channel
   * \param msu - pointer to corresponding rlc interface
   *
   */
  virtual void AddLc (LteEnbCmacSapProvider::LcInfo lcInfo, LteMacSapUser* msu) = 0;

  /**
   * \brief Remove an existing UE.
   * \param rnti Radio Network Temporary Identity, an integer identifying the UE
   *             where the report originates from
   */
  virtual void RemoveUe (uint16_t rnti) = 0;

  /**
   * \brief Add a new Bearer for the Ue in the LteEnbComponentCarrierManager.
   * \param bearer a pointer to the EpsBearer object
   * \param bearerId a unique identifier for the bearer
   * \param rnti Radio Network Temporary Identity, an integer identifying the UE
   *             where the report originates from
   * \param lcid the Logical Channel id
   * \param lcGroup the Logical Channel group
   * \param msu a pointer to the LteMacSapUser, the LteEnbComponentCarrierManager
   *             has to store a LteMacSapUser for each Rlc istance, in order to 
   *             properly redirect the packet
   * \return vector of LcsConfig contains the lc configuration for each Mac
   *                the size of the vector is equal to the number of component
   *                carrier enabled.
   *
   * The Logical Channel configurations for each component carrier depend on the 
   * algorithm used to split the traffic between the component carriers themself.
   */  
  virtual std::vector<LteCcmRrcSapProvider::LcsConfig> SetupDataRadioBearer (EpsBearer bearer, uint8_t bearerId, uint16_t rnti, uint8_t lcid, uint8_t lcGroup, LteMacSapUser *msu) = 0;

   /**
   * \brief Release an existing Data Radio Bearer for a Ue in the LteEnbComponentCarrierManager
   * \param rnti Radio Network Temporary Identity, an integer identifying the UE
   *             where the report originates from
   * \param lcid the Logical Channel Id
   * \return vector of integer the componentCarrierId of the componentCarrier
   *                where the bearer is enabled
   */

  virtual std::vector<uint8_t> ReleaseDataRadioBearer (uint16_t rnti, uint8_t lcid) = 0;

  /**
   * \brief Add the Signal Bearer for a specific Ue in LteEnbComponenCarrierManager
   * \param lcInfo this structure it is hard-coded in the LteEnbRrc
   * \param rlcMacSapUser it is the MacSapUser of the Rlc istance
   * \return the LteMacSapUser of the ComponentCarrierManager
   *
   */
  virtual LteMacSapUser* ConfigureSignalBearer(LteEnbCmacSapProvider::LcInfo lcInfo,  LteMacSapUser* rlcMacSapUser) = 0;

}; // end of class LteCcmRrcSapProvider


/**
 * \brief Service Access Point (SAP) offered by the eNodeB RRC instance to the
 *        component carrier manager (CCM) instance.
 *
 * This is the *Component Carrier Management SAP User*, i.e., the part of the SAP that
 * contains the eNodeB RRC methods called by the CCM.
 */
class LteCcmRrcSapUser
{
  /// allow LteEnbRrc class friend access
  friend class LteEnbRrc;
public:
  virtual ~LteCcmRrcSapUser ();

  /**
   * \brief Request a certain reporting configuration to be fulfilled by the UEs
   *        attached to the eNodeB entity.
   * \param reportConfig the UE measurement reporting configuration
   * \return the measurement identity associated with this newly added
   *         reporting configuration
   *
   * The eNodeB RRC entity is expected to configure the same reporting
   * configuration in each of the attached UEs. When later in the simulation a
   * UE measurement report is received from a UE as a result of this
   * configuration, the eNodeB RRC entity shall forward this report to the
   * ComponentCarrier algorithm through the LteCcmRrcSapProvider::ReportUeMeas
   * SAP function.
   *
   * \note This function is only valid before the simulation begins.
   */
  virtual uint8_t AddUeMeasReportConfigForComponentCarrier (LteRrcSap::ReportConfigEutra reportConfig) = 0;

  /**
   * \brief Instruct the eNodeB RRC entity to prepare a component carrier.
   * \param rnti Radio Network Temporary Identity, an integer identifying the
   *             UE which shall perform the ComponentCarrier
   * \param targetCellId the cell ID of the target eNodeB
   *
   * This function is used by the ComponentCarrier manager when a decision on 
   * component carriers configurations.
   *
   * The process to produce the decision is up to the implementation of ComponentCarrier
   * algorithm. It is typically based on the reported UE measurements, which are
   * received through the LteCcmRrcSapProvider::ReportUeMeas function.
   */
  virtual void TriggerComponentCarrier (uint16_t rnti, uint16_t targetCellId) = 0;

  /** 
   * add a new Logical Channel (LC) 
   * 
   * \param lcConfig is a single structure contains logical Channel Id, Logical Channel config and Component Carrier Id
   */
  virtual void AddLcs (std::vector<LteEnbRrcSapProvider::LogicalChannelConfig> lcConfig) = 0;

  /** 
   * remove an existing LC
   * 
   * \param rnti 
   * \param lcid
   */
  virtual void ReleaseLcs (uint16_t rnti, uint8_t lcid) = 0;

  /**
   * Get UE manager by RNTI
   *
   * \param rnti RNTI
   * \return UE manager
   */
  virtual Ptr<UeManager> GetUeManager (uint16_t rnti) = 0;

  /**
   * \brief Set the number of component carriers
   *
   * \param noOfComponentCarriers The number of component carriers
   */
  virtual void SetNumberOfComponentCarriers (uint16_t noOfComponentCarriers) = 0;

}; // end of class LteCcmRrcSapUser

/// MemberLteCcmRrcSapProvider class
template <class C>
class MemberLteCcmRrcSapProvider : public LteCcmRrcSapProvider
{
public:
  /**
   * Constructor
   * 
   * \param owner the owner class
   */
  MemberLteCcmRrcSapProvider (C* owner);

  // inherited from LteCcmRrcSapProvider
  virtual void ReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults);
  virtual void AddUe (uint16_t rnti, uint8_t state);
  virtual void AddLc (LteEnbCmacSapProvider::LcInfo lcInfo, LteMacSapUser* msu);
  virtual void RemoveUe (uint16_t rnti);
  virtual std::vector<LteCcmRrcSapProvider::LcsConfig> SetupDataRadioBearer (EpsBearer bearer, uint8_t bearerId, uint16_t rnti, uint8_t lcid, uint8_t lcGroup, LteMacSapUser *msu);
  virtual std::vector<uint8_t> ReleaseDataRadioBearer (uint16_t rnti, uint8_t lcid);
  virtual LteMacSapUser* ConfigureSignalBearer(LteEnbCmacSapProvider::LcInfo lcInfo,  LteMacSapUser* rlcMacSapUser);

private:
  C* m_owner; ///< the owner class
};

template <class C>
MemberLteCcmRrcSapProvider<C>::MemberLteCcmRrcSapProvider (C* owner)
  : m_owner (owner)
{
}

template <class C>
void MemberLteCcmRrcSapProvider<C>::ReportUeMeas (uint16_t rnti, LteRrcSap::MeasResults measResults)
{
  m_owner->DoReportUeMeas (rnti, measResults);
}

template <class C>
void MemberLteCcmRrcSapProvider<C>::AddUe (uint16_t rnti, uint8_t state)
{
  m_owner->DoAddUe (rnti, state);
}

template <class C>
void MemberLteCcmRrcSapProvider<C>::AddLc (LteEnbCmacSapProvider::LcInfo lcInfo, LteMacSapUser* msu)
{
  m_owner->DoAddLc (lcInfo, msu);
}

template <class C>
void MemberLteCcmRrcSapProvider<C>::RemoveUe (uint16_t rnti)
{
  m_owner->DoRemoveUe (rnti);
}

template <class C>
std::vector<LteCcmRrcSapProvider::LcsConfig> MemberLteCcmRrcSapProvider<C>::SetupDataRadioBearer (EpsBearer bearer, uint8_t bearerId, uint16_t rnti, uint8_t lcid, uint8_t lcGroup, LteMacSapUser *msu)
{
  return m_owner->DoSetupDataRadioBearer (bearer, bearerId, rnti, lcid, lcGroup, msu);
}

template <class C>
std::vector<uint8_t> MemberLteCcmRrcSapProvider<C>::ReleaseDataRadioBearer (uint16_t rnti, uint8_t lcid)
{
  return m_owner->DoReleaseDataRadioBearer (rnti, lcid);
}

template <class C>
LteMacSapUser* MemberLteCcmRrcSapProvider<C>::ConfigureSignalBearer(LteEnbCmacSapProvider::LcInfo lcInfo,  LteMacSapUser* rlcMacSapUser)
{
  return m_owner->DoConfigureSignalBearer (lcInfo, rlcMacSapUser);
}


/// MemberLteCcmRrcSapUser class
template <class C>
class MemberLteCcmRrcSapUser : public LteCcmRrcSapUser
{
public:
  /**
   * Constructor
   * 
   * \param owner the owner class
   */
  MemberLteCcmRrcSapUser (C* owner);

  // inherited from LteCcmRrcSapUser
  virtual void AddLcs (std::vector<LteEnbRrcSapProvider::LogicalChannelConfig> lcConfig);
  virtual void ReleaseLcs (uint16_t rnti, uint8_t lcid);
  virtual uint8_t AddUeMeasReportConfigForComponentCarrier (LteRrcSap::ReportConfigEutra reportConfig);
  virtual void TriggerComponentCarrier (uint16_t rnti, uint16_t targetCellId);
  virtual Ptr<UeManager> GetUeManager (uint16_t rnti);
  virtual void SetNumberOfComponentCarriers (uint16_t noOfComponentCarriers);

private:
  C* m_owner; ///< the owner class
};

template <class C>
MemberLteCcmRrcSapUser<C>::MemberLteCcmRrcSapUser (C* owner)
  : m_owner (owner)
{
}

template <class C>
void MemberLteCcmRrcSapUser<C>::AddLcs (std::vector<LteEnbRrcSapProvider::LogicalChannelConfig> lcConfig)
{
  NS_FATAL_ERROR ("Function should not be called because it is not implemented.");
  //m_owner->DoAddLcs (lcConfig);
}

template <class C>
void MemberLteCcmRrcSapUser<C>::ReleaseLcs (uint16_t rnti, uint8_t lcid)
{
  NS_FATAL_ERROR ("Function should not be called because it is not implemented.");
  //m_owner->DoReleaseLcs (rnti, lcid);

}

template <class C>
uint8_t
MemberLteCcmRrcSapUser<C>::AddUeMeasReportConfigForComponentCarrier (LteRrcSap::ReportConfigEutra reportConfig)
{
  return m_owner->DoAddUeMeasReportConfigForComponentCarrier (reportConfig);
}


template <class C>
void
MemberLteCcmRrcSapUser<C>::TriggerComponentCarrier (uint16_t rnti, uint16_t targetCellId)
{
  NS_FATAL_ERROR ("Function should not be called because it is not implemented.");
}

template <class C>
Ptr<UeManager>
MemberLteCcmRrcSapUser<C>::GetUeManager (uint16_t rnti)
{
  return m_owner->GetUeManager (rnti);
}

template <class C>
void
MemberLteCcmRrcSapUser<C>::SetNumberOfComponentCarriers (uint16_t noOfComponentCarriers)
{
  return m_owner->DoSetNumberOfComponentCarriers (noOfComponentCarriers);
}

} // end of namespace ns3


#endif /* LTE_CCM_RRC_SAP_H */

