/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#include "bs-uplink-scheduler-simple.h"
#include "bs-net-device.h"
#include "ns3/simulator.h"
#include "cid.h"
#include "burst-profile-manager.h"
#include "ss-manager.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ss-record.h"
#include "service-flow.h"
#include "service-flow-record.h"
#include "bs-link-manager.h"
#include "bandwidth-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("UplinkSchedulerSimple");
  
NS_OBJECT_ENSURE_REGISTERED (UplinkSchedulerSimple);

UplinkSchedulerSimple::UplinkSchedulerSimple (void)
{
  SetBs (0);
  SetTimeStampIrInterval (Seconds (0));
  SetNrIrOppsAllocated (0);
  SetIsIrIntrvlAllocated (false);
  SetIsInvIrIntrvlAllocated (false);
  SetDcdTimeStamp (Simulator::Now ());
  SetUcdTimeStamp (Simulator::Now ());
}

UplinkSchedulerSimple::UplinkSchedulerSimple (Ptr<BaseStationNetDevice> bs)
{
  SetBs (bs);
  SetTimeStampIrInterval (Seconds (0));
  SetNrIrOppsAllocated (0);
  SetIsIrIntrvlAllocated (false);
  SetIsInvIrIntrvlAllocated (false);
  SetDcdTimeStamp (Simulator::Now ());
  SetUcdTimeStamp (Simulator::Now ());
}

UplinkSchedulerSimple::~UplinkSchedulerSimple (void)
{
  SetBs (0);
  m_uplinkAllocations.clear ();
}

void
UplinkSchedulerSimple::InitOnce ()
{

}

TypeId
UplinkSchedulerSimple::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UplinkSchedulerSimple")
    .SetParent<UplinkScheduler> ()
    .SetGroupName("Wimax")
    .AddConstructor<UplinkSchedulerSimple> ()
    ;
  return tid;
}

std::list<OfdmUlMapIe>
UplinkSchedulerSimple::GetUplinkAllocations (void) const
{
  return m_uplinkAllocations;
}

void
UplinkSchedulerSimple::GetChannelDescriptorsToUpdate (bool &updateDcd,
                                                      bool &updateUcd,
                                                      bool &sendDcd,
                                                      bool &sendUcd)
{
  /*DCD and UCD shall actually be updated when channel or burst profile definitions
   change. burst profiles are updated based on number of SSs, network conditions and etc.
   for now temporarily assuming DCD/UCD shall be updated every time */

  uint32_t randNr = rand ();
  if (randNr % 5 == 0 || GetBs ()->GetNrDcdSent () == 0)
    {
      sendDcd = true;
    }

  randNr = rand ();
  if (randNr % 5 == 0 || GetBs ()->GetNrUcdSent () == 0)
    {
      sendUcd = true;
    }

  // -------------------------------------
  // additional, just to send more frequently
  if (!sendDcd)
    {
      randNr = rand ();
      if (randNr % 4 == 0)
        {
          sendDcd = true;
        }
    }

  if (!sendUcd)
    {
      randNr = rand ();
      if (randNr % 4 == 0)
        {
          sendUcd = true;
        }
    }
  // -------------------------------------

  Time timeSinceLastDcd = Simulator::Now () - GetDcdTimeStamp ();
  Time timeSinceLastUcd = Simulator::Now () - GetUcdTimeStamp ();

  if (timeSinceLastDcd > GetBs ()->GetDcdInterval ())
    {
      sendDcd = true;
      SetDcdTimeStamp (Simulator::Now ());
    }

  if (timeSinceLastUcd > GetBs ()->GetUcdInterval ())
    {
      sendUcd = true;
      SetUcdTimeStamp (Simulator::Now ());
    }
}

uint32_t
UplinkSchedulerSimple::CalculateAllocationStartTime (void)
{
  return GetBs ()->GetNrDlSymbols () * GetBs ()->GetPhy ()->GetPsPerSymbol () + GetBs ()->GetTtg ();
}

void
UplinkSchedulerSimple::AddUplinkAllocation (OfdmUlMapIe &ulMapIe,
                                            const uint32_t &allocationSize,
                                            uint32_t &symbolsToAllocation,
                                            uint32_t &availableSymbols)
{
  ulMapIe.SetDuration (allocationSize);
  ulMapIe.SetStartTime (symbolsToAllocation);
  m_uplinkAllocations.push_back (ulMapIe);
  symbolsToAllocation += allocationSize;
  availableSymbols -= allocationSize;
}

void
UplinkSchedulerSimple::Schedule (void)
{
  m_uplinkAllocations.clear ();
  SetIsIrIntrvlAllocated (false);
  SetIsInvIrIntrvlAllocated (false);
  bool allocationForDsa = false;

  uint32_t symbolsToAllocation = 0;
  uint32_t allocationSize = 0; // size in symbols
  uint32_t availableSymbols = GetBs ()->GetNrUlSymbols ();

  AllocateInitialRangingInterval (symbolsToAllocation, availableSymbols);

  std::vector<SSRecord*> *ssRecords = GetBs ()->GetSSManager ()->GetSSRecords ();
  for (std::vector<SSRecord*>::iterator iter = ssRecords->begin (); iter != ssRecords->end (); ++iter)
    {
      SSRecord *ssRecord = *iter;

      if (ssRecord->GetIsBroadcastSS ())
        {
          continue;
        }
      Cid cid = ssRecord->GetBasicCid ();
      OfdmUlMapIe ulMapIe;
      ulMapIe.SetCid (cid);

      if (ssRecord->GetPollForRanging () && ssRecord->GetRangingStatus () == WimaxNetDevice::RANGING_STATUS_CONTINUE)
        {

          // SS's ranging is not yet complete
          // allocating invited initial ranging interval
          ulMapIe.SetUiuc (OfdmUlBurstProfile::UIUC_INITIAL_RANGING);
          allocationSize = GetBs ()->GetRangReqOppSize ();
          SetIsInvIrIntrvlAllocated (true);

          if (availableSymbols >= allocationSize)
            {
              AddUplinkAllocation (ulMapIe, allocationSize, symbolsToAllocation, availableSymbols);
            }
          else
            {

              break;
            }
        }
      else
        {
          WimaxPhy::ModulationType modulationType = ssRecord->GetModulationType ();

          // need to update because modulation/FEC to UIUC mapping may vary over time
          ulMapIe.SetUiuc (GetBs ()->GetBurstProfileManager ()->GetBurstProfile (modulationType,
                                                                                 WimaxNetDevice::DIRECTION_UPLINK));

          // establish service flows for SS
          if (ssRecord->GetRangingStatus () == WimaxNetDevice::RANGING_STATUS_SUCCESS
              && !ssRecord->GetAreServiceFlowsAllocated ())
            {

              // allocating grant (with arbitrary size) to allow SS to send DSA messages DSA-REQ and DSA-ACK
              // only one DSA allocation per frame
              if (!allocationForDsa)
                {
                  allocationSize = GetBs ()->GetPhy ()->GetNrSymbols (sizeof(DsaReq), modulationType);

                  if (availableSymbols >= allocationSize)
                    {
                      AddUplinkAllocation (ulMapIe, allocationSize, symbolsToAllocation, availableSymbols);
                      allocationForDsa = true;
                    }
                  else
                    {
                      break;
                    }
                }
            }
          else
            {
              // all service flows associated to SS are established now

              /*allocating grants for data transmission for UGS flows (Data Grant Burst Type IEs, 6.3.7.4.3.3)
               (grant has been referred by different names e.g. transmission opportunity, slot, uplink allocation, etc)*/
              ServiceUnsolicitedGrants (ssRecord,
                                        ServiceFlow::SF_TYPE_UGS,
                                        ulMapIe,
                                        modulationType,
                                        symbolsToAllocation,
                                        availableSymbols);

              // allocate unicast polls for rtPS flows if bandwidth is available
              if (availableSymbols)
                {
                  ServiceUnsolicitedGrants (ssRecord,
                                            ServiceFlow::SF_TYPE_RTPS,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }
              // allocate unicast polls for nrtPS flows if bandwidth is available
              if (availableSymbols)
                {
                  ServiceUnsolicitedGrants (ssRecord,
                                            ServiceFlow::SF_TYPE_NRTPS,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }
              // finally allocate unicast polls for BE flows if bandwidth is available
              if (availableSymbols)
                {
                  ServiceUnsolicitedGrants (ssRecord,
                                            ServiceFlow::SF_TYPE_BE,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }

              // now allocating grants for non-UGS flows (i.e., in response of bandwidth requests)

              if (availableSymbols)
                {
                  ServiceBandwidthRequests (ssRecord,
                                            ServiceFlow::SF_TYPE_RTPS,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }
              // allocate unicast polls for nrtPS flows if bandwidth is available
              if (availableSymbols)
                {
                  ServiceBandwidthRequests (ssRecord,
                                            ServiceFlow::SF_TYPE_NRTPS,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }
              // finally allocate unicast polls for BE flows if bandwidth is available
              if (availableSymbols)
                {
                  ServiceBandwidthRequests (ssRecord,
                                            ServiceFlow::SF_TYPE_BE,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                }
            }
        }
    }
  OfdmUlMapIe ulMapIeEnd;

  ulMapIeEnd.SetCid (Cid::InitialRanging ());
  ulMapIeEnd.SetStartTime (symbolsToAllocation);
  ulMapIeEnd.SetUiuc (OfdmUlBurstProfile::UIUC_END_OF_MAP);
  ulMapIeEnd.SetDuration (0);
  m_uplinkAllocations.push_back (ulMapIeEnd);

  // setting DL/UL subframe allocation for the next frame
  GetBs ()->GetBandwidthManager ()->SetSubframeRatio ();
}

void
UplinkSchedulerSimple::ServiceUnsolicitedGrants (const SSRecord *ssRecord,
                                                 enum ServiceFlow::SchedulingType schedulingType,
                                                 OfdmUlMapIe &ulMapIe,
                                                 const WimaxPhy::ModulationType modulationType,
                                                 uint32_t &symbolsToAllocation,
                                                 uint32_t &availableSymbols)
{
  uint32_t allocationSize = 0; // size in symbols
  uint8_t uiuc = ulMapIe.GetUiuc (); // SS's burst profile
  std::vector<ServiceFlow*> serviceFlows = ssRecord->GetServiceFlows (schedulingType);

  for (std::vector<ServiceFlow*>::iterator iter = serviceFlows.begin (); iter != serviceFlows.end (); ++iter)
    {
      ServiceFlow *serviceFlow = *iter;

      /* in case of rtPS, nrtPS and BE, allocating unicast polls for bandwidth requests (Request IEs, 6.3.7.4.3.1).
       in case of UGS, allocating grants for data transmission (Data Grant Burst Type IEs, 6.3.7.4.3.3) (grant has
       been referred in this code by different names e.g. transmission opportunity, slot, allocation, etc) */

      allocationSize = GetBs ()->GetBandwidthManager ()->CalculateAllocationSize (ssRecord, serviceFlow);
      // verifying that minimum reserved traffic rate of nrtPS flow is maintained
      if (serviceFlow->GetSchedulingType () == ServiceFlow::SF_TYPE_NRTPS)
        {
          Time currentTime = Simulator::Now ();
          ServiceFlowRecord *record = serviceFlow->GetRecord ();
          if (currentTime - record->GetGrantTimeStamp () > Seconds (1))
            {
              uint32_t bps = (record->GetBwSinceLastExpiry () * 8);
              if (bps < serviceFlow->GetMinReservedTrafficRate ())
                {
                  ServiceBandwidthRequests (serviceFlow,
                                            schedulingType,
                                            ulMapIe,
                                            modulationType,
                                            symbolsToAllocation,
                                            availableSymbols);
                  record->SetBwSinceLastExpiry (0);
                  record->SetGrantTimeStamp (currentTime);
                }
            }
        }

      if (availableSymbols < allocationSize)
        {
          break;
        }

      if (allocationSize > 0)
        {
          ulMapIe.SetStartTime (symbolsToAllocation);
          if (serviceFlow->GetSchedulingType () != ServiceFlow::SF_TYPE_UGS)
            {
              // special burst profile with most robust modulation type is used for unicast polls (Request IEs)
              ulMapIe.SetUiuc (OfdmUlBurstProfile::UIUC_REQ_REGION_FULL);
            }
        }
      else
        {
          continue;
        }

      NS_LOG_DEBUG (", CID: " << serviceFlow->GetConnection ()->GetCid () << ", SFID: " << serviceFlow->GetSfid ());

      AddUplinkAllocation (ulMapIe, allocationSize, symbolsToAllocation, availableSymbols);
      ulMapIe.SetUiuc (uiuc);
    }
}

void
UplinkSchedulerSimple::ServiceBandwidthRequests (const SSRecord *ssRecord,
                                                 enum ServiceFlow::SchedulingType schedulingType,
                                                 OfdmUlMapIe &ulMapIe,
                                                 const WimaxPhy::ModulationType modulationType,
                                                 uint32_t &symbolsToAllocation,
                                                 uint32_t &availableSymbols)
{
  std::vector<ServiceFlow*> serviceFlows = ssRecord->GetServiceFlows (schedulingType);

  for (std::vector<ServiceFlow*>::iterator iter = serviceFlows.begin (); iter != serviceFlows.end (); ++iter)
    {
      if (!ServiceBandwidthRequests (*iter,
                                     schedulingType,
                                     ulMapIe,
                                     modulationType,
                                     symbolsToAllocation,
                                     availableSymbols))
        {
          break;
        }
    }
}

bool
UplinkSchedulerSimple::ServiceBandwidthRequests (ServiceFlow *serviceFlow,
                                                 enum ServiceFlow::SchedulingType schedulingType,
                                                 OfdmUlMapIe &ulMapIe,
                                                 const WimaxPhy::ModulationType modulationType,
                                                 uint32_t &symbolsToAllocation,
                                                 uint32_t &availableSymbols)
{
  uint32_t allocSizeBytes = 0;
  uint32_t allocSizeSymbols = 0;
  uint16_t sduSize = 0;

  ServiceFlowRecord *record = serviceFlow->GetRecord ();
  sduSize = serviceFlow->GetSduSize ();

  uint32_t requiredBandwidth = record->GetRequestedBandwidth () - record->GetGrantedBandwidth ();
  if (requiredBandwidth > 0)
    {
      if (sduSize > 0)
        {
          // if SDU size is mentioned, allocate grant of that size
          allocSizeBytes = sduSize;
          allocSizeSymbols = GetBs ()->GetPhy ()->GetNrSymbols (sduSize, modulationType);
        }
      else
        {
          allocSizeBytes = requiredBandwidth;
          allocSizeSymbols = GetBs ()->GetPhy ()->GetNrSymbols (requiredBandwidth, modulationType);
        }

      if (availableSymbols >= allocSizeSymbols)
        {
          record->UpdateGrantedBandwidth (allocSizeBytes);

          if (schedulingType == ServiceFlow::SF_TYPE_NRTPS)
            {
              record->SetBwSinceLastExpiry (allocSizeBytes);
            }

          AddUplinkAllocation (ulMapIe, allocSizeSymbols, symbolsToAllocation, availableSymbols);
        }
      else
        {
          return false;
        }
    }
  return true;
}

void
UplinkSchedulerSimple::AllocateInitialRangingInterval (uint32_t &symbolsToAllocation, uint32_t &availableSymbols)
{
  Time ssUlStartTime = Seconds (CalculateAllocationStartTime () * GetBs ()->GetPsDuration ().GetSeconds ());
  SetNrIrOppsAllocated (GetBs ()->GetLinkManager ()->CalculateRangingOppsToAllocate ());
  uint32_t allocationSize = GetNrIrOppsAllocated () * GetBs ()->GetRangReqOppSize ();
  Time timeSinceLastIrInterval = Simulator::Now () - GetTimeStampIrInterval ();

  // adding one frame because may be the time has not elapsed now but will elapse before the next frame is sent
  if (timeSinceLastIrInterval + GetBs ()->GetPhy ()->GetFrameDuration () > GetBs ()->GetInitialRangingInterval ()
      && availableSymbols >= allocationSize)
    {
      SetIsIrIntrvlAllocated (true);
      OfdmUlMapIe ulMapIeIr;
      ulMapIeIr.SetCid ((GetBs ()->GetBroadcastConnection ())->GetCid ());
      ulMapIeIr.SetStartTime (symbolsToAllocation);
      ulMapIeIr.SetUiuc (OfdmUlBurstProfile::UIUC_INITIAL_RANGING);

      NS_LOG_DEBUG ("BS uplink scheduler, initial ranging allocation, size: " << allocationSize << " symbols"
                                                                              << ", modulation: BPSK 1/2");

      // marking start and end of each TO, only for debugging
      for (uint8_t i = 0; i < GetNrIrOppsAllocated (); i++)
        {
          GetBs ()->MarkRangingOppStart (ssUlStartTime + Seconds (symbolsToAllocation
                                                                  * GetBs ()->GetSymbolDuration ().GetSeconds ()) + Seconds (i * GetBs ()->GetRangReqOppSize ()
                                                                                                                             * GetBs ()->GetSymbolDuration ().GetSeconds ()));
        }

      AddUplinkAllocation (ulMapIeIr, allocationSize, symbolsToAllocation, availableSymbols);
      SetTimeStampIrInterval (Simulator::Now ());
    }
}

void
UplinkSchedulerSimple::SetupServiceFlow (SSRecord *ssRecord, ServiceFlow *serviceFlow)
{
  uint8_t delayNrFrames = 1;
  uint32_t bitsPerSecond = serviceFlow->GetMinReservedTrafficRate ();
  WimaxPhy::ModulationType modulation;
  uint32_t bytesPerFrame =
    (uint32_t ((double)(bitsPerSecond) * GetBs ()->GetPhy ()->GetFrameDuration ().GetSeconds ())) / 8;
  uint32_t frameDurationMSec = GetBs ()->GetPhy ()->GetFrameDuration ().GetMilliSeconds ();

  switch (serviceFlow->GetSchedulingType ())
    {
    case ServiceFlow::SF_TYPE_UGS:
      {
        if (serviceFlow->GetIsMulticast () == true)
          {
            modulation = serviceFlow->GetModulation ();
          }
        else
          {
            modulation = ssRecord->GetModulationType ();
          }
        uint32_t grantSize = GetBs ()->GetPhy ()->GetNrSymbols (bytesPerFrame, modulation);
        serviceFlow->GetRecord ()->SetGrantSize (grantSize);

        uint32_t toleratedJitter = serviceFlow->GetToleratedJitter ();

        if (toleratedJitter > frameDurationMSec)
          {
            delayNrFrames = (uint8_t)(toleratedJitter / frameDurationMSec);
          }

        uint16_t interval = delayNrFrames * frameDurationMSec;
        serviceFlow->SetUnsolicitedGrantInterval (interval);
      }
      break;
    case ServiceFlow::SF_TYPE_RTPS:
      {
        if (serviceFlow->GetSduSize () > bytesPerFrame)
          {
            delayNrFrames = (uint8_t)(serviceFlow->GetSduSize () / bytesPerFrame);
          }

        uint16_t interval = delayNrFrames * frameDurationMSec;
        serviceFlow->SetUnsolicitedPollingInterval (interval);
      }
      break;
    case ServiceFlow::SF_TYPE_NRTPS:
      {
        // no real-time guarantees are given to NRTPS, serviced based on available bandwidth
      }
      break;
    case ServiceFlow::SF_TYPE_BE:
      {
        // no real-time guarantees are given to BE, serviced based on available bandwidth
      }
      break;
    default:
      NS_FATAL_ERROR ("Invalid scheduling type");
    }
}

void
UplinkSchedulerSimple::ProcessBandwidthRequest (const BandwidthRequestHeader &bwRequestHdr)
{
  // virtual function on UplinkScheduler
  // this is not necessary on this implementation
}

void
UplinkSchedulerSimple::OnSetRequestedBandwidth (ServiceFlowRecord *sfr)
{
  // m_grantedBandwidth must be reset to zero
  uint32_t grantedBandwidth = 0;
  sfr->SetGrantedBandwidth (grantedBandwidth);
}

} // namespace ns3
