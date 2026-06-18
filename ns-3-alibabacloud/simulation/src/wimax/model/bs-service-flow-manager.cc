/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#include <stdint.h>
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "service-flow.h"
#include "service-flow-manager.h"
#include "ns3/log.h"
#include "bs-net-device.h"
#include "ss-record.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "wimax-connection.h"
#include "ss-manager.h"
#include "connection-manager.h"
#include "bs-uplink-scheduler.h"
#include "ss-scheduler.h"
#include "ns3/buffer.h"
#include "service-flow-record.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BsServiceFlowManager");

BsServiceFlowManager::BsServiceFlowManager (Ptr<BaseStationNetDevice> device)
  : m_device (device),
    m_sfidIndex (100),
    m_maxDsaRspRetries (100)                                     // default value
{
  m_inuseScheduleDsaRspCid = Cid::InitialRanging ();
}

BsServiceFlowManager::~BsServiceFlowManager (void)
{
}

/* static */
TypeId
BsServiceFlowManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BsServiceFlowManager")
    .SetParent<ServiceFlowManager> ()
    .SetGroupName ("Wifi")
    // No AddConstructor because this class has no default constructor.
  ;
  return tid;
}

void
BsServiceFlowManager::DoDispose (void)
{
  ServiceFlowManager::DoDispose ();
}

void
BsServiceFlowManager::SetMaxDsaRspRetries (uint8_t maxDsaRspRetries)
{
  m_maxDsaRspRetries = maxDsaRspRetries;
}

uint8_t
BsServiceFlowManager::GetMaxDsaRspRetries (void) const
{
  return m_maxDsaRspRetries;
}

EventId
BsServiceFlowManager::GetDsaAckTimeoutEvent (void) const
{
  return m_dsaAckTimeoutEvent;
}

void
BsServiceFlowManager::AddServiceFlow (ServiceFlow *serviceFlow)
{
  ServiceFlowManager::AddServiceFlow (serviceFlow);
}

ServiceFlow*
BsServiceFlowManager::GetServiceFlow (uint32_t sfid) const
{
  return ServiceFlowManager::GetServiceFlow (sfid);
}

ServiceFlow*
BsServiceFlowManager::GetServiceFlow (Cid cid) const
{
  return ServiceFlowManager::GetServiceFlow (cid);
}

std::vector<ServiceFlow*>
BsServiceFlowManager::GetServiceFlows (ServiceFlow::SchedulingType schedulingType) const
{
  return ServiceFlowManager::GetServiceFlows (schedulingType);
}

DsaRsp
BsServiceFlowManager::CreateDsaRsp (const ServiceFlow *serviceFlow, uint16_t transactionId)
{
  DsaRsp dsaRsp;
  dsaRsp.SetTransactionId (transactionId);
  dsaRsp.SetServiceFlow (*serviceFlow);
  // assuming SS can supports all of the service flow parameters
  dsaRsp.SetConfirmationCode (CONFIRMATION_CODE_SUCCESS);

  return dsaRsp;
}

void
BsServiceFlowManager::ScheduleDsaRsp (ServiceFlow *serviceFlow, Cid cid)
{
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();

  SSRecord *ssRecord = bs->GetSSManager ()->GetSSRecord (cid);
  if (ssRecord == 0)
    {
      NS_LOG_INFO ("SS not registered with the BS CID:" << cid);
      return;
    }

  serviceFlow->SetIsEnabled (true);
  serviceFlow->SetType (ServiceFlow::SF_TYPE_ACTIVE);
  ssRecord->AddServiceFlow (serviceFlow);


  bs->GetUplinkScheduler ()->SetupServiceFlow (ssRecord, serviceFlow);

  Ptr<Packet> p = Create<Packet> ();
  DsaRsp dsaRsp;

  if (ssRecord->GetDsaRspRetries () == 0)
    {
      dsaRsp = CreateDsaRsp (serviceFlow, ssRecord->GetSfTransactionId ());
      p->AddHeader (dsaRsp);
      ssRecord->SetDsaRsp (dsaRsp);
    }
  else
    {
      if (ssRecord->GetDsaRspRetries () < m_maxDsaRspRetries)
        {
          p->AddHeader (ssRecord->GetDsaRsp ());
        }
      else
        {
          NS_LOG_DEBUG ("Service flows could not be initialized!");
        }
    }

  ssRecord->IncrementDsaRspRetries ();
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_DSA_RSP));

  if (m_dsaAckTimeoutEvent.IsRunning ())
    {
      Simulator::Cancel (m_dsaAckTimeoutEvent);
    }

  m_inuseScheduleDsaRspCid = cid;

  m_dsaAckTimeoutEvent = Simulator::Schedule (bs->GetIntervalT8 (),
                                              &BsServiceFlowManager::ScheduleDsaRsp,
                                              this,
                                              serviceFlow,
                                              cid);
  m_device->Enqueue (p, MacHeaderType (), bs->GetConnection (ssRecord->GetPrimaryCid ()));
}

ServiceFlow*
BsServiceFlowManager::ProcessDsaReq (const DsaReq &dsaReq, Cid cid)
{
  ServiceFlow * serviceFlow;
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();
  SSRecord *ssRecord = bs->GetSSManager ()->GetSSRecord (cid);

  NS_LOG_INFO ("BsServiceFlowManager: Processing DSA-REQ...");
  if (ssRecord->GetSfTransactionId () != 0)
    {
      // had already received DSA-REQ. DSA-RSP was lost
      NS_ASSERT_MSG (dsaReq.GetTransactionId () == ssRecord->GetSfTransactionId (),
                     "Error while processing DSA request:the received transaction ID is not expected");
      serviceFlow = GetServiceFlow (ssRecord->GetDsaRsp ().GetSfid ());
    }
  else
    {
      ServiceFlow sf = dsaReq.GetServiceFlow ();
      Ptr<WimaxConnection> transportConnection;
      Ptr<ConnectionManager> BsConManager = bs->GetConnectionManager ();
      transportConnection = BsConManager->CreateConnection (Cid::TRANSPORT);
      serviceFlow = new ServiceFlow (m_sfidIndex++, sf.GetDirection (), transportConnection);
      transportConnection->SetServiceFlow (serviceFlow);
      serviceFlow->CopyParametersFrom (sf);
      serviceFlow->SetUnsolicitedGrantInterval (1);
      serviceFlow->SetUnsolicitedPollingInterval (1);
      serviceFlow->SetConvergenceSublayerParam (sf.GetConvergenceSublayerParam ());
      AddServiceFlow (serviceFlow);
      ssRecord->SetSfTransactionId (dsaReq.GetTransactionId ());
      NS_LOG_INFO ("BsServiceFlowManager: Creating a new Service flow: SFID = " << serviceFlow->GetSfid () << " CID = "
                                                                                << serviceFlow->GetCid ());
    }
  return serviceFlow;
}

void
BsServiceFlowManager::AddMulticastServiceFlow (ServiceFlow  sf, enum WimaxPhy::ModulationType modulation)
{
  ServiceFlow * serviceFlow = new ServiceFlow ();
  serviceFlow->CopyParametersFrom (sf);
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();
  Ptr<WimaxConnection> multicastConnection = bs->GetConnectionManager ()->CreateConnection (Cid::MULTICAST);
  serviceFlow->SetConnection (multicastConnection);
  AddServiceFlow (serviceFlow);
  serviceFlow->SetIsEnabled (true);
  serviceFlow->SetType (ServiceFlow::SF_TYPE_ACTIVE);
  serviceFlow->SetIsMulticast (true);
  serviceFlow->SetModulation (modulation);
  bs->GetUplinkScheduler ()->SetupServiceFlow (0, serviceFlow);
}

void
BsServiceFlowManager::AllocateServiceFlows (const DsaReq &dsaReq, Cid cid)
{
  ServiceFlow *serviceFlow = ProcessDsaReq (dsaReq, cid);
  if (serviceFlow) {
      ScheduleDsaRsp (serviceFlow, cid);
    } else {
      NS_LOG_INFO ("No service Flow. Could not connect.");
    }
}

void
BsServiceFlowManager::ProcessDsaAck (const DsaAck &dsaAck, Cid cid)
{
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();
  SSRecord *ssRecord = bs->GetSSManager ()->GetSSRecord (cid);

  if (dsaAck.GetTransactionId () != ssRecord->GetSfTransactionId ())
    {
      return;
    }

  ssRecord->SetDsaRspRetries (0);
  ssRecord->SetSfTransactionId (0);

  // check if all service flow have been initiated
  if (AreServiceFlowsAllocated (ssRecord->GetServiceFlows (ServiceFlow::SF_TYPE_ALL)))
    {
      ssRecord->SetAreServiceFlowsAllocated (true);
    }
}
} // namespace ns3
