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

#include "bs-link-manager.h"
#include <stdint.h>
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "burst-profile-manager.h"
#include "ss-record.h"
#include "ss-manager.h"
#include "bs-uplink-scheduler.h"
#include "connection-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BSLinkManager");

NS_OBJECT_ENSURE_REGISTERED (BSLinkManager);

TypeId BSLinkManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BSLinkManager")
    .SetParent<Object> ()
    .SetGroupName("Wimax");
  return tid;
}

BSLinkManager::BSLinkManager (Ptr<BaseStationNetDevice> bs)
  : m_bs (bs),
    m_signalQuality (10),
    m_signalQualityThreshold (10)                                  // arbitrary value
{
  tries = 0;
}

BSLinkManager::~BSLinkManager (void)
{
  m_bs = 0;
}

uint8_t
BSLinkManager::CalculateRangingOppsToAllocate (void)
{
  // randomly selecting TOs up to 10, shall actually be decided by scheduler
  return rand () % 8 + 2;
}

/*
 * Function mainly to avoid duplicate code in DoReceive ()
 */
void
BSLinkManager::ProcessRangingRequest (Cid cid, RngReq rngreq)
{
  NS_ASSERT_MSG (m_bs->GetState () == BaseStationNetDevice::BS_STATE_UL_SUB_FRAME,
                 "Base station: Error while processing ranging request: !BS_STATE_UL_SUB_FRAME");

  Time irIntervalBoundary = Seconds (0);

  if (m_bs->GetUplinkScheduler ()->GetIsInvIrIntrvlAllocated ())
    {
      if (m_bs->GetUplinkScheduler ()->GetIsIrIntrvlAllocated ())
        {
          irIntervalBoundary = m_bs->GetUlSubframeStartTime ()
            + Seconds ((m_bs->GetUplinkScheduler ()->GetNrIrOppsAllocated () + 1) * m_bs->GetRangReqOppSize ()
                       * m_bs->GetSymbolDuration ().GetSeconds ());
        }
      else
        {
          irIntervalBoundary = m_bs->GetUlSubframeStartTime () + Seconds (m_bs->GetRangReqOppSize ()
                                                                          * m_bs->GetSymbolDuration ().GetSeconds ());
        }
    }
  else
    {
      NS_ASSERT_MSG (m_bs->GetUplinkScheduler ()->GetIsIrIntrvlAllocated (),
                     "Base station: Error while processing ranging request: IR interval not allocated");

      irIntervalBoundary = m_bs->GetUlSubframeStartTime ()
        + Seconds (m_bs->GetUplinkScheduler ()->GetNrIrOppsAllocated () * m_bs->GetRangReqOppSize ()
                   * m_bs->GetSymbolDuration ().GetSeconds ());
    }

  tries++;

  if (Simulator::Now () >= m_bs->GetUlSubframeStartTime () && Simulator::Now () < irIntervalBoundary)
    {
      PerformRanging (cid, rngreq);
    }
}

void
BSLinkManager::PerformRanging (Cid cid, RngReq rngreq)
{
  RngRsp rngrsp;
  bool decodable = false;

  // assuming low power, packet lost or undecodable first 2 times
  if (tries < 2)
    {
      return;
    }
  if (tries >= 3)
    {
      decodable = true;
    }

  NS_LOG_DEBUG ("RNG-REQ:");
  rngreq.PrintDebug ();

  if (!decodable)
    {
      rngrsp.SetFrameNumber (m_bs->GetNrFrames ());
      rngrsp.SetInitRangOppNumber (m_bs->GetRangingOppNumber ());

      SetParametersToAdjust (&rngrsp);
      rngrsp.SetRangStatus (WimaxNetDevice::RANGING_STATUS_CONTINUE); // see Figure 64
      ScheduleRngRspMessage (cid, &rngrsp);
    }
  else
    {
      if (cid.IsInitialRanging ())
        {
          PerformInitialRanging (cid, &rngreq, &rngrsp);
        }
      else
        {
          // invited initial ranging or periodic ranging
          PerformInvitedRanging (cid, &rngrsp);
        }
    }
}

void
BSLinkManager::PerformInitialRanging (Cid cid, RngReq *rngreq, RngRsp *rngrsp)
{

  SSRecord *ssRecord = 0;
  bool isOldSS = m_bs->GetSSManager ()->IsInRecord (rngreq->GetMacAddress ());
  if (isOldSS)
    {
      ssRecord = m_bs->GetSSManager ()->GetSSRecord (rngreq->GetMacAddress ());
      // if this fails it would mean the RNG-RSP with success status was not received by the SS
    }
  else
    {
      ssRecord = m_bs->GetSSManager ()->CreateSSRecord (rngreq->GetMacAddress ());
    }

  if (ChangeDlChannel ())
    {
      rngrsp->SetDlFreqOverride (GetNewDlChannel ());
      AbortRanging (cid, rngrsp, ssRecord, isOldSS);
      return;
    }

  if (isOldSS)
    {
      // CIDs already assigned, e.g., RNG-REQ was lost and resent after timeout. reusing old CIDs
      ssRecord->ResetRangingCorrectionRetries ();
      ssRecord->ResetInvitedRangingRetries ();
    }
  else
    {
      m_bs->GetConnectionManager ()->AllocateManagementConnections (ssRecord, rngrsp);

      WimaxPhy::ModulationType modulationType;
      uint8_t diuc = m_bs->GetBurstProfileManager ()->GetBurstProfileForSS (ssRecord, rngreq, modulationType);
      ssRecord->SetModulationType (modulationType);

      // specify in RNG-RSP only if different than what SS requested
      if (rngreq->GetReqDlBurstProfile () != diuc)
        {
          rngrsp->SetDlOperBurstProfile (diuc);
        }

      // add SS (Basic CID) to poll list for invited ranging intervals, see Table 115
      ssRecord->EnablePollForRanging ();
    }

  rngrsp->SetMacAddress (rngreq->GetMacAddress ());

  if (isOldSS) // CIDs had already been allocated
    {
      cid = ssRecord->GetBasicCid ();
    }

  if (IsRangingAcceptable ())
    {
      AcceptRanging (cid, rngrsp, ssRecord);
    }
  else
    {
      ContinueRanging (cid, rngrsp, ssRecord);
    }
}

void
BSLinkManager::PerformInvitedRanging (Cid cid, RngRsp *rngrsp)
{
  SSRecord *ssRecord = m_bs->GetSSManager ()->GetSSRecord (cid);
  ssRecord->IncrementRangingCorrectionRetries ();
  ssRecord->ResetInvitedRangingRetries ();

  if (IsRangingAcceptable ())
    {
      AcceptRanging (cid, rngrsp, ssRecord);
    }
  else
    {
      if (ssRecord->GetRangingCorrectionRetries () == m_bs->GetMaxRangingCorrectionRetries ())
        {
          AbortRanging (cid, rngrsp, ssRecord, true);
        }
      else
        {
          ContinueRanging (cid, rngrsp, ssRecord);
        }
    }
}

void
BSLinkManager::VerifyInvitedRanging (Cid cid, uint8_t uiuc)
{
  if (uiuc == OfdmUlBurstProfile::UIUC_INITIAL_RANGING)
    {
      SSRecord *ssRecord = m_bs->GetSSManager ()->GetSSRecord (cid);
      if (ssRecord->GetInvitedRangRetries () > 0)
        {
          ssRecord->IncrementInvitedRangingRetries ();

          if (ssRecord->GetInvitedRangRetries () == m_bs->GetMaxInvitedRangRetries ())
            {
              RngRsp *rngrsp = new RngRsp ();
              AbortRanging (ssRecord->GetBasicCid (), rngrsp, ssRecord, true);
            } // else keep polling
        }
    }
}

void
BSLinkManager::SetParametersToAdjust (RngRsp *rngrsp)
{
  // code to calculate parameter adjustment values goes here
  rngrsp->SetTimingAdjust (40);
  rngrsp->SetPowerLevelAdjust (8);
  rngrsp->SetOffsetFreqAdjust (30);
}

void
BSLinkManager::AbortRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord, bool isOldSS)
{
  rngrsp->SetRangStatus (WimaxNetDevice::RANGING_STATUS_ABORT);
  ScheduleRngRspMessage (cid, rngrsp);

  if (isOldSS)
    {
      ssRecord->SetRangingStatus (WimaxNetDevice::RANGING_STATUS_ABORT);
    }

  ssRecord->DisablePollForRanging ();
  DeallocateCids (cid);
}

void
BSLinkManager::AcceptRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord)
{
  rngrsp->SetRangStatus (WimaxNetDevice::RANGING_STATUS_SUCCESS);
  ScheduleRngRspMessage (cid, rngrsp);

  /*Shall not be set until the SS receives the RNG-RSP, as it may be lost etc. may be state field
   is also added to SSRecord which then set to SS_STATE_REGISTERED once confirmed that SS has received
   this RNG-RSP, but how to determine that, may be as a data packet is received by the SS*/
  ssRecord->SetRangingStatus (WimaxNetDevice::RANGING_STATUS_SUCCESS);

  ssRecord->DisablePollForRanging ();
}

void
BSLinkManager::ContinueRanging (Cid cid, RngRsp *rngrsp, SSRecord *ssRecord)
{
  rngrsp->SetRangStatus (WimaxNetDevice::RANGING_STATUS_CONTINUE);
  ScheduleRngRspMessage (cid, rngrsp);
  ssRecord->SetRangingStatus (WimaxNetDevice::RANGING_STATUS_CONTINUE);
}

void
BSLinkManager::ScheduleRngRspMessage (Cid cid, RngRsp *rngrsp)
{
  if (rngrsp->GetRangStatus () == WimaxNetDevice::RANGING_STATUS_SUCCESS || rngrsp->GetRangStatus ()
      == WimaxNetDevice::RANGING_STATUS_CONTINUE)
    {
      SetParametersToAdjust (rngrsp);
    }

  Ptr<Packet> p = Create<Packet> ();
  p->AddHeader (*rngrsp);
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_RNG_RSP));

  m_bs->Enqueue (p, MacHeaderType (), m_bs->GetConnection (cid));
}

void
BSLinkManager::DeallocateCids (Cid cid)
{
  // if necessary, delete entire connections or simply set CIDs to 0
}

uint64_t
BSLinkManager::SelectDlChannel (void)
{
  // Values according to WirelessMAN-OFDM RF profile for 10 MHz channelization
  // Section 12.3.3.1 from IEEE 802.16-2004 standard
  // profR10_3 :
  // channels: 5000 + n â‹… 5 MHz, âˆ€n âˆˆ { 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167 }
  // temporarily set to 1 for quick scanning. To be standard compliant, use a value in the list above
  return m_bs->GetChannel (1);
}

bool
BSLinkManager::ChangeDlChannel (void)
{
  // code to decide if SS shall move to a new channel/frequency goes here
  return false;
}

uint32_t
BSLinkManager::GetNewDlChannel (void)
{
  // code to determine suggested new frequency goes here
  return 100;
}

uint8_t
BSLinkManager::GetSignalQuality (void)
{
  // code to measure signal quality goes here
  uint8_t signalQuality = m_signalQuality;
  m_signalQuality++;
  return signalQuality;
}

bool
BSLinkManager::IsRangingAcceptable (void)
{
  return GetSignalQuality () > m_signalQualityThreshold;
}

} // namespace ns3
