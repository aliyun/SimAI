/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
 * Copyright (c) 2016 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Authors: Danilo Abrignani <danilo.abrignani@unibo.it>
 *          Biljana Bojovic <biljana.bojovic@cttc.es>
 *
 */

#include "no-op-component-carrier-manager.h"
#include <ns3/log.h>
#include <ns3/random-variable-stream.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NoOpComponentCarrierManager");
NS_OBJECT_ENSURE_REGISTERED (NoOpComponentCarrierManager);
  
NoOpComponentCarrierManager::NoOpComponentCarrierManager ()
{
  NS_LOG_FUNCTION (this);
  m_ccmRrcSapProvider = new MemberLteCcmRrcSapProvider<NoOpComponentCarrierManager> (this);
  m_ccmMacSapUser = new MemberLteCcmMacSapUser<NoOpComponentCarrierManager> (this);
  m_macSapProvider = new EnbMacMemberLteMacSapProvider <NoOpComponentCarrierManager> (this);
  m_ccmRrcSapUser  = 0;
}

NoOpComponentCarrierManager::~NoOpComponentCarrierManager ()
{
  NS_LOG_FUNCTION (this);
}

void
NoOpComponentCarrierManager::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ccmRrcSapProvider;
  delete m_ccmMacSapUser;
  delete m_macSapProvider;
}


TypeId
NoOpComponentCarrierManager::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::NoOpComponentCarrierManager")
          .SetParent<LteEnbComponentCarrierManager> ()
          .SetGroupName("Lte")
          .AddConstructor<NoOpComponentCarrierManager> ()
          ;
  return tid;
}


void
NoOpComponentCarrierManager::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteEnbComponentCarrierManager::DoInitialize ();
}

//////////////////////////////////////////////
// MAC SAP
/////////////////////////////////////////////


void
NoOpComponentCarrierManager::DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params)
{
  NS_LOG_FUNCTION (this);
  std::map <uint8_t, LteMacSapProvider*>::iterator it =  m_macSapProvidersMap.find (params.componentCarrierId);
  NS_ASSERT_MSG (it != m_macSapProvidersMap.end (), "could not find Sap for ComponentCarrier " << params.componentCarrierId);
  // with this algorithm all traffic is on Primary Carrier
  it->second->TransmitPdu (params);
}

void
NoOpComponentCarrierManager::DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params)
{
  NS_LOG_FUNCTION (this);
  auto ueManager = m_ccmRrcSapUser->GetUeManager (params.rnti);
  std::map <uint8_t, LteMacSapProvider*>::iterator it = m_macSapProvidersMap.find (ueManager->GetComponentCarrierId ());
  NS_ASSERT_MSG (it != m_macSapProvidersMap.end (), "could not find Sap for ComponentCarrier ");
  it->second->ReportBufferStatus (params);
}

void
NoOpComponentCarrierManager::DoNotifyTxOpportunity (LteMacSapUser::TxOpportunityParameters txOpParams)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG (this << " rnti= " << txOpParams.rnti << " lcid= " << +txOpParams.lcid << " layer= " << +txOpParams.layer << " ccId=" << +txOpParams.componentCarrierId);
  m_ueInfo.at (txOpParams.rnti).m_ueAttached.at (txOpParams.lcid)->NotifyTxOpportunity (txOpParams);
}

void
NoOpComponentCarrierManager::DoReceivePdu (LteMacSapUser::ReceivePduParameters rxPduParams)
{
  NS_LOG_FUNCTION (this);
  auto lcidIt = m_ueInfo.at (rxPduParams.rnti).m_ueAttached.find (rxPduParams.lcid);
  if (lcidIt != m_ueInfo.at (rxPduParams.rnti).m_ueAttached.end ())
    {
      lcidIt->second->ReceivePdu (rxPduParams);
    }
}

void
NoOpComponentCarrierManager::DoNotifyHarqDeliveryFailure ()
{
  NS_LOG_FUNCTION (this);
}

void
NoOpComponentCarrierManager::DoReportUeMeas (uint16_t rnti,
                                             LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);
}

void
NoOpComponentCarrierManager::DoAddUe (uint16_t rnti, uint8_t state)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) state);
  std::map<uint16_t, uint8_t>::iterator eccIt; // m_enabledComponentCarrier iterator
  auto ueInfoIt = m_ueInfo.find (rnti);
  if (ueInfoIt == m_ueInfo.end ())
    {
      NS_LOG_DEBUG (this << " UE " << rnti << " was not found, now it is added in the map");
      UeInfo info;
      info.m_ueState = state;

      // the Primary carrier (PC) is enabled by default
      // on the PC the SRB0 and SRB1 are enabled when the Ue is connected
      // these are hard-coded and the configuration not pass through the
      // Component Carrier Manager which is responsible of configure
      // only DataRadioBearer on the different Component Carrier
      info.m_enabledComponentCarrier = 1;
      m_ueInfo.emplace (rnti, info);
    }
  else
    {
      NS_LOG_DEBUG (this << " UE " << rnti << "found, updating the state from " << +ueInfoIt->second.m_ueState << " to " << +state);
      ueInfoIt->second.m_ueState = state;
    }
}

void
NoOpComponentCarrierManager::DoAddLc (LteEnbCmacSapProvider::LcInfo lcInfo, LteMacSapUser* msu)
{
  NS_LOG_FUNCTION (this);
  m_ueInfo.at (lcInfo.rnti).m_rlcLcInstantiated.emplace (lcInfo.lcId, lcInfo);
}


void
NoOpComponentCarrierManager::DoRemoveUe (uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  auto rntiIt = m_ueInfo.find (rnti);
  NS_ASSERT_MSG (rntiIt != m_ueInfo.end (), "request to remove UE info with unknown RNTI " << rnti);
  m_ueInfo.erase (rntiIt);
}

std::vector<LteCcmRrcSapProvider::LcsConfig>
NoOpComponentCarrierManager::DoSetupDataRadioBearer (EpsBearer bearer, uint8_t bearerId, uint16_t rnti, uint8_t lcid, uint8_t lcGroup, LteMacSapUser *msu)
{
  NS_LOG_FUNCTION (this << rnti);
  auto rntiIt = m_ueInfo.find (rnti);
  NS_ASSERT_MSG (rntiIt != m_ueInfo.end (), "SetupDataRadioBearer on unknown RNTI " << rnti);

  // enable by default all carriers
  rntiIt->second.m_enabledComponentCarrier = m_noOfComponentCarriers;

  std::vector<LteCcmRrcSapProvider::LcsConfig> res;
  LteCcmRrcSapProvider::LcsConfig entry;
  LteEnbCmacSapProvider::LcInfo lcinfo;
  // NS_LOG_DEBUG (this << " componentCarrierEnabled " << (uint16_t) eccIt->second);
  for (uint16_t ncc = 0; ncc < m_noOfComponentCarriers; ncc++)
    {
      // NS_LOG_DEBUG (this << " res size " << (uint16_t) res.size ());
      LteEnbCmacSapProvider::LcInfo lci;
      lci.rnti = rnti;
      lci.lcId = lcid;
      lci.lcGroup = lcGroup;
      lci.qci = bearer.qci;
      if (ncc == 0)
        {
          lci.isGbr = bearer.IsGbr ();
          lci.mbrUl = bearer.gbrQosInfo.mbrUl;
          lci.mbrDl = bearer.gbrQosInfo.mbrDl;
          lci.gbrUl = bearer.gbrQosInfo.gbrUl;
          lci.gbrDl = bearer.gbrQosInfo.gbrDl;
        }
      else
        {
          lci.isGbr = 0;
          lci.mbrUl = 0;
          lci.mbrDl = 0;
          lci.gbrUl = 0;
          lci.gbrDl = 0;
        } // data flows only on PC
      NS_LOG_DEBUG (this << " RNTI " << lci.rnti << "Lcid " << (uint16_t) lci.lcId << " lcGroup " << (uint16_t) lci.lcGroup);
      entry.componentCarrierId = ncc;
      entry.lc = lci;
      entry.msu = m_ccmMacSapUser;
      res.push_back (entry);
    } // end for

  auto lcidIt = rntiIt->second.m_rlcLcInstantiated.find (lcid);
  if (lcidIt == rntiIt->second.m_rlcLcInstantiated.end ())
    {
      lcinfo.rnti = rnti;
      lcinfo.lcId = lcid;
      lcinfo.lcGroup = lcGroup;
      lcinfo.qci = bearer.qci;
      lcinfo.isGbr = bearer.IsGbr ();
      lcinfo.mbrUl = bearer.gbrQosInfo.mbrUl;
      lcinfo.mbrDl = bearer.gbrQosInfo.mbrDl;
      lcinfo.gbrUl = bearer.gbrQosInfo.gbrUl;
      lcinfo.gbrDl = bearer.gbrQosInfo.gbrDl;
      rntiIt->second.m_rlcLcInstantiated.emplace (lcinfo.lcId, lcinfo);
      rntiIt->second.m_ueAttached.emplace (lcinfo.lcId, msu);
    }
  else
    {
      NS_LOG_ERROR ("LC already exists");
    }
  return res;

}

std::vector<uint8_t>
NoOpComponentCarrierManager::DoReleaseDataRadioBearer (uint16_t rnti, uint8_t lcid)
{
  NS_LOG_FUNCTION (this << rnti << +lcid);
  
  // Here we receive directly the RNTI and the LCID, instead of only DRB ID
  // DRB ID are mapped as DRBID = LCID + 2
  auto rntiIt = m_ueInfo.find (rnti);
  NS_ASSERT_MSG (rntiIt != m_ueInfo.end (), "request to Release Data Radio Bearer on UE with unknown RNTI " << rnti);

  NS_LOG_DEBUG (this << " remove LCID " << +lcid << " for RNTI " << rnti);
  std::vector<uint8_t> res;
  for (uint16_t i = 0; i < rntiIt->second.m_enabledComponentCarrier; i++)
    {
      res.insert (res.end (), i);
    }

  auto lcIt = rntiIt->second.m_ueAttached.find (lcid);
  NS_ASSERT_MSG (lcIt != rntiIt->second.m_ueAttached.end (), "Logical Channel not found");
  rntiIt->second.m_ueAttached.erase (lcIt);

  auto rlcInstancesIt = rntiIt->second.m_rlcLcInstantiated.find (rnti);
  NS_ASSERT_MSG (rlcInstancesIt != rntiIt->second.m_rlcLcInstantiated.end (), "Logical Channel not found");
  rntiIt->second.m_rlcLcInstantiated.erase (rlcInstancesIt);

  return res;
}

LteMacSapUser*
NoOpComponentCarrierManager::DoConfigureSignalBearer(LteEnbCmacSapProvider::LcInfo lcinfo, LteMacSapUser* msu)
{
  NS_LOG_FUNCTION (this);

  auto rntiIt = m_ueInfo.find (lcinfo.rnti);
  NS_ASSERT_MSG (rntiIt != m_ueInfo.end (), "request to add a signal bearer to unknown RNTI " << lcinfo.rnti);

  auto lcidIt = rntiIt->second.m_ueAttached.find (lcinfo.lcId);
  if (lcidIt == rntiIt->second.m_ueAttached.end ())
    {
      rntiIt->second.m_ueAttached.emplace (lcinfo.lcId, msu);
    }
  else
    {
      NS_LOG_ERROR ("LC already exists");
    }

  return m_ccmMacSapUser;
}

void
NoOpComponentCarrierManager::DoNotifyPrbOccupancy (double prbOccupancy, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("Update PRB occupancy:"<<prbOccupancy<<" at carrier:"<< (uint32_t) componentCarrierId);
  m_ccPrbOccupancy.insert(std::pair<uint8_t, double> (componentCarrierId, prbOccupancy));
}

void
NoOpComponentCarrierManager::DoUlReceiveMacCe (MacCeListElement_s bsr, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (bsr.m_macCeType == MacCeListElement_s::BSR, "Received a Control Message not allowed " << bsr.m_macCeType);
  if ( bsr.m_macCeType == MacCeListElement_s::BSR)
    {
      MacCeListElement_s newBsr;
      newBsr.m_rnti = bsr.m_rnti;
      newBsr.m_macCeType = bsr.m_macCeType;
      newBsr.m_macCeValue.m_phr = bsr.m_macCeValue.m_phr;
      newBsr.m_macCeValue.m_crnti = bsr.m_macCeValue.m_crnti;
      newBsr.m_macCeValue.m_bufferStatus.resize (4);
      for (uint16_t i = 0; i < 4; i++)
        {
          uint8_t bsrId = bsr.m_macCeValue.m_bufferStatus.at (i);
          uint32_t buffer = BufferSizeLevelBsr::BsrId2BufferSize (bsrId);
          // here the buffer should be divide among the different sap
          // since the buffer status report are compressed information
          // it is needed to use BsrId2BufferSize to uncompress
          // after the split over all component carriers is is needed to
          // compress again the information to fit MacCeListEkement_s structure
          // verify how many Component Carrier are enabled per UE
          // in this simple code the BufferStatus will be notify only
          // to the primary carrier component
          newBsr.m_macCeValue.m_bufferStatus.at (i) = BufferSizeLevelBsr::BufferSize2BsrId (buffer);
        }
      auto sapIt = m_ccmMacSapProviderMap.find (componentCarrierId);
      if (sapIt == m_ccmMacSapProviderMap.end ())
        {
          NS_FATAL_ERROR ("Sap not found in the CcmMacSapProviderMap");
        }
      else
        {
          // in the current implementation bsr in uplink is forwarded only to the primary carrier.
          // above code demonstrates how to resize buffer status if more carriers are being used in future
          sapIt->second->ReportMacCeToScheduler (newBsr);
        }
    }
  else
    {
      NS_FATAL_ERROR ("Expected BSR type of message.");
    }
}

void
NoOpComponentCarrierManager::DoUlReceiveSr (uint16_t rnti, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this);

  auto sapIt = m_ccmMacSapProviderMap.find (componentCarrierId);
  NS_ABORT_MSG_IF (sapIt == m_ccmMacSapProviderMap.end (),
                   "Sap not found in the CcmMacSapProviderMap");

  sapIt->second->ReportSrToScheduler (rnti);
}


//////////////////////////////////////////

NS_OBJECT_ENSURE_REGISTERED (RrComponentCarrierManager);

RrComponentCarrierManager::RrComponentCarrierManager ()
{
  NS_LOG_FUNCTION (this);

}

RrComponentCarrierManager::~RrComponentCarrierManager ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
RrComponentCarrierManager::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::RrComponentCarrierManager")
                .SetParent<NoOpComponentCarrierManager> ()
                .SetGroupName("Lte")
                .AddConstructor<RrComponentCarrierManager> ()
                ;
  return tid;
}


void
RrComponentCarrierManager::DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params)
{
  NS_LOG_FUNCTION (this);

  uint32_t numberOfCarriersForUe = m_ueInfo.at (params.rnti).m_enabledComponentCarrier;
  if (params.lcid == 0 || params.lcid == 1 || numberOfCarriersForUe == 1)
    {
      NS_LOG_INFO("Buffer status forwarded to the primary carrier.");
      auto ueManager = m_ccmRrcSapUser->GetUeManager (params.rnti);
      m_macSapProvidersMap.at (ueManager->GetComponentCarrierId ())->ReportBufferStatus (params);
    }
  else
    {
      params.retxQueueSize /= numberOfCarriersForUe;
      params.txQueueSize /= numberOfCarriersForUe;
      for ( uint16_t i = 0;  i < numberOfCarriersForUe ; i++)
        {
          NS_ASSERT_MSG (m_macSapProvidersMap.find(i)!=m_macSapProvidersMap.end(), "Mac sap provider does not exist.");
          m_macSapProvidersMap.at (i)->ReportBufferStatus(params);
        }
    }
}


void
RrComponentCarrierManager::DoUlReceiveMacCe (MacCeListElement_s bsr, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (componentCarrierId == 0, "Received BSR from a ComponentCarrier not allowed, ComponentCarrierId = " << componentCarrierId);
  NS_ASSERT_MSG (bsr.m_macCeType == MacCeListElement_s::BSR, "Received a Control Message not allowed " << bsr.m_macCeType);

  // split traffic in uplink equally among carriers
  uint32_t numberOfCarriersForUe = m_ueInfo.at (bsr.m_rnti).m_enabledComponentCarrier;

  if ( bsr.m_macCeType == MacCeListElement_s::BSR)
    {
      MacCeListElement_s newBsr;
      newBsr.m_rnti = bsr.m_rnti;
      // mac control element type, values can be BSR, PHR, CRNTI
      newBsr.m_macCeType = bsr.m_macCeType;
      // the power headroom, 64 means no valid phr is available
      newBsr.m_macCeValue.m_phr = bsr.m_macCeValue.m_phr;
      // indicates that the CRNTI MAC CE was received. The value is not used.
      newBsr.m_macCeValue.m_crnti = bsr.m_macCeValue.m_crnti;
      // and value 64 means that the buffer status should not be updated
      newBsr.m_macCeValue.m_bufferStatus.resize (4);
      // always all 4 LCGs are present see 6.1.3.1 of 3GPP TS 36.321.
      for (uint16_t i = 0; i < 4; i++)
        {
          uint8_t bsrStatusId = bsr.m_macCeValue.m_bufferStatus.at (i);
          uint32_t bufferSize = BufferSizeLevelBsr::BsrId2BufferSize (bsrStatusId);
          // here the buffer should be divide among the different sap
          // since the buffer status report are compressed information
          // it is needed to use BsrId2BufferSize to uncompress
          // after the split over all component carriers is is needed to
          // compress again the information to fit MacCeListElement_s structure
          // verify how many Component Carrier are enabled per UE
          newBsr.m_macCeValue.m_bufferStatus.at(i) = BufferSizeLevelBsr::BufferSize2BsrId (bufferSize/numberOfCarriersForUe);
        }
      // notify MAC of each component carrier that is enabled for this UE
      for ( uint16_t i = 0;  i < numberOfCarriersForUe ; i++)
        {
          NS_ASSERT_MSG (m_ccmMacSapProviderMap.find(i)!=m_ccmMacSapProviderMap.end(), "Mac sap provider does not exist.");
          m_ccmMacSapProviderMap.find(i)->second->ReportMacCeToScheduler(newBsr);
        }
    }
  else
    {
      auto ueManager = m_ccmRrcSapUser->GetUeManager (bsr.m_rnti);
      m_ccmMacSapProviderMap.at (ueManager->GetComponentCarrierId ())->ReportMacCeToScheduler (bsr);
    }
}

void
RrComponentCarrierManager::DoUlReceiveSr(uint16_t rnti, [[maybe_unused]] uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this);
  // split traffic in uplink equally among carriers
  uint32_t numberOfCarriersForUe = m_ueInfo.at (rnti).m_enabledComponentCarrier;

  m_ccmMacSapProviderMap.find (m_lastCcIdForSr)->second->ReportSrToScheduler (rnti);

  m_lastCcIdForSr++;
  if (m_lastCcIdForSr > numberOfCarriersForUe - 1)
    {
      m_lastCcIdForSr = 0;
    }
}

} // end of namespace ns3
