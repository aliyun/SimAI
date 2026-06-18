/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 *                               <amine.ismail@UDcast.com>
 */

#include "bandwidth-manager.h"
#include "ns3/node.h"
#include "bs-net-device.h"
#include "ss-net-device.h"
#include "ns3/simulator.h"
#include "burst-profile-manager.h"
#include "ss-manager.h"
#include "ss-record.h"
#include "service-flow.h"
#include "service-flow-record.h"
#include "service-flow-manager.h"
#include "connection-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BandwidthManager");

NS_OBJECT_ENSURE_REGISTERED (BandwidthManager);

TypeId BandwidthManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BandwidthManager")
    .SetParent<Object> ()
    .SetGroupName("Wimax");
  return tid;
}

BandwidthManager::BandwidthManager (Ptr<WimaxNetDevice> device)
  : m_device (device),
    m_nrBwReqsSent (0)
{
}

BandwidthManager::~BandwidthManager (void)
{
}

void
BandwidthManager::DoDispose (void)
{
  m_device = 0;
}

uint32_t
BandwidthManager::CalculateAllocationSize (const SSRecord *ssRecord, const ServiceFlow *serviceFlow)
{
  Time currentTime = Simulator::Now ();
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();
  uint32_t allocationSize = 0;

  // if SS has a UGS flow then it must set poll-me bit in order to be polled for non-UGS flows
  if (serviceFlow->GetSchedulingType () != ServiceFlow::SF_TYPE_UGS
      && ssRecord->GetHasServiceFlowUgs ()
      && !ssRecord->GetPollMeBit ())
    {
      return 0;
    }

  switch (serviceFlow->GetSchedulingType ())
    {
    case ServiceFlow::SF_TYPE_UGS:
      {
        if ((currentTime - serviceFlow->GetRecord ()->GetGrantTimeStamp ()).GetMilliSeconds ()
            >= serviceFlow->GetUnsolicitedGrantInterval ())
          {
            allocationSize = serviceFlow->GetRecord ()->GetGrantSize ();
            serviceFlow->GetRecord ()->SetGrantTimeStamp (currentTime);
          }
      }
      break;
    case ServiceFlow::SF_TYPE_RTPS:
      {
        if ((currentTime - serviceFlow->GetRecord ()->GetGrantTimeStamp ()).GetMilliSeconds ()
            >= serviceFlow->GetUnsolicitedPollingInterval ())
          {
            allocationSize = bs->GetBwReqOppSize ();
            serviceFlow->GetRecord ()->SetGrantTimeStamp (currentTime);
          }
      }
      break;
    case ServiceFlow::SF_TYPE_NRTPS:
      {
        /* nrtPS shall be serviced only if sufficient bandwidth is available after servicing
         UGS and rtPS scheduling types, hence no specific service interval is used */

        allocationSize = bs->GetBwReqOppSize ();
      }
      break;
    case ServiceFlow::SF_TYPE_BE:
      {
        /* BE shall be serviced only if sufficient bandwidth is available after servicing
         the rest of three scheduling types, hence no specific service interval is used */

        allocationSize = bs->GetBwReqOppSize ();
      }
      break;
    default:
      NS_FATAL_ERROR ("Invalid scheduling type");
    }

  return allocationSize;
}

ServiceFlow*
BandwidthManager::SelectFlowForRequest (uint32_t &bytesToRequest)
{
  Ptr<Packet> packet;
  ServiceFlow *serviceFlow = 0;

  Ptr<SubscriberStationNetDevice> ss = m_device->GetObject<SubscriberStationNetDevice> ();
  std::vector<ServiceFlow*> serviceFlows = ss->GetServiceFlowManager ()->GetServiceFlows (ServiceFlow::SF_TYPE_ALL);

  for (std::vector<ServiceFlow*>::iterator iter = serviceFlows.begin (); iter != serviceFlows.end (); ++iter)
    {
      serviceFlow = *iter;
      if (serviceFlow->GetSchedulingType () == ServiceFlow::SF_TYPE_RTPS
          || serviceFlow->GetSchedulingType () == ServiceFlow::SF_TYPE_NRTPS
          || serviceFlow->GetSchedulingType () == ServiceFlow::SF_TYPE_BE)
        {
          if (serviceFlow->HasPackets (MacHeaderType::HEADER_TYPE_GENERIC))
            {
              // bandwidth is requested for all packets
              bytesToRequest = serviceFlow->GetQueue ()->GetQueueLengthWithMACOverhead ();
              break;
            }
        }
    }

  return serviceFlow;
}

void
BandwidthManager::SendBandwidthRequest (uint8_t uiuc, uint16_t allocationSize)
{
  Ptr<SubscriberStationNetDevice> ss = m_device->GetObject<SubscriberStationNetDevice> ();

  uint32_t bytesToRequest = 0;
  ServiceFlow *serviceFlow = SelectFlowForRequest (bytesToRequest);

  if (!serviceFlow || !bytesToRequest)
    {
      return;
    }
  BandwidthRequestHeader bwRequestHdr;

  // bytesToRequest is the queue length of Service Flow and so,
  // the header type must be HEADER_TYPE_AGGREGATE!

  bwRequestHdr.SetType ((uint8_t) BandwidthRequestHeader::HEADER_TYPE_AGGREGATE);
  bwRequestHdr.SetCid (serviceFlow->GetConnection ()->GetCid ());
  bwRequestHdr.SetBr (bytesToRequest);

  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (bwRequestHdr);
  ss->Enqueue (packet, MacHeaderType (MacHeaderType::HEADER_TYPE_BANDWIDTH), serviceFlow->GetConnection ());
  m_nrBwReqsSent++;
  NS_ASSERT_MSG (uiuc == OfdmUlBurstProfile::UIUC_REQ_REGION_FULL, "Send Bandwidth Request: !UIUC_REQ_REGION_FULL");
  ss->SendBurst (uiuc, allocationSize, serviceFlow->GetConnection (), MacHeaderType::HEADER_TYPE_BANDWIDTH);
}

void
BandwidthManager::ProcessBandwidthRequest (const BandwidthRequestHeader &bwRequestHdr)
{
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();

  ServiceFlow *serviceFlow = bs->GetConnectionManager ()->GetConnection (bwRequestHdr.GetCid ())->GetServiceFlow ();
  if (bwRequestHdr.GetType () == (uint8_t) BandwidthRequestHeader::HEADER_TYPE_INCREMENTAL)
    {
      serviceFlow->GetRecord ()->UpdateRequestedBandwidth (bwRequestHdr.GetBr ());
    }
  else
    {
      serviceFlow->GetRecord ()->SetRequestedBandwidth (bwRequestHdr.GetBr ());
      bs->GetUplinkScheduler ()->OnSetRequestedBandwidth (serviceFlow->GetRecord ());
    }
  bs->GetUplinkScheduler ()->ProcessBandwidthRequest (bwRequestHdr);
  // update backlogged
  serviceFlow->GetRecord ()->IncreaseBacklogged (bwRequestHdr.GetBr ());
}

void
BandwidthManager::SetSubframeRatio (void)
{
  // sets ratio of the DL and UL subframes

  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();

  uint32_t symbolsPerFrame = bs->GetPhy ()->GetSymbolsPerFrame ();

  /* temporarily divided in half (360 symbols each), shall actually be determined based on UL and DL traffic*/
  bs->SetNrDlSymbols (symbolsPerFrame / 2);
  bs->SetNrUlSymbols (symbolsPerFrame / 2);
}

uint32_t
BandwidthManager::GetSymbolsPerFrameAllocated (void)
{
  Ptr<BaseStationNetDevice> bs = m_device->GetObject<BaseStationNetDevice> ();

  uint32_t allocationPerFrame = 0;

  std::vector<SSRecord*> *ssRecords = bs->GetSSManager ()->GetSSRecords ();
  for (std::vector<SSRecord*>::const_iterator iter1 = ssRecords->begin (); iter1 != ssRecords->end (); ++iter1)
    {
      std::vector<ServiceFlow*> ssServiceFlows = (*iter1)->GetServiceFlows (ServiceFlow::SF_TYPE_ALL);
      for (std::vector<ServiceFlow*>::const_iterator iter2 = ssServiceFlows.begin (); iter2 != ssServiceFlows.end (); ++iter2)
        {
          allocationPerFrame += (*iter2)->GetRecord ()->GetGrantSize ();
        }
    }
  return allocationPerFrame;
}

} // namespace ns3
