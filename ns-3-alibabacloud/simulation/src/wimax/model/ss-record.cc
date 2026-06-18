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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#include "ss-record.h"
#include "service-flow.h"
#include <stdint.h>

namespace ns3 {

SSRecord::SSRecord (void)
{
  Initialize ();
}

SSRecord::SSRecord (Mac48Address macAddress)
{
  m_macAddress = macAddress;
  Initialize ();
}

SSRecord::SSRecord (Mac48Address macAddress, Ipv4Address IPaddress)
{
  m_macAddress = macAddress;
  m_IPAddress = IPaddress;
  Initialize ();
}

void
SSRecord::Initialize (void)
{
  m_basicCid = Cid ();
  m_primaryCid = Cid ();

  m_rangingCorrectionRetries = 0;
  m_invitedRangingRetries = 0;
  m_modulationType = WimaxPhy::MODULATION_TYPE_BPSK_12;
  m_rangingStatus = WimaxNetDevice::RANGING_STATUS_EXPIRED;
  m_pollForRanging = false;
  m_areServiceFlowsAllocated = false;
  m_pollMeBit = false;

  m_sfTransactionId = 0;
  m_dsaRspRetries = 0;

  m_serviceFlows = new std::vector<ServiceFlow*> ();
  m_dsaRsp = DsaRsp ();
  m_broadcast = 0;
}

SSRecord::~SSRecord (void)
{
  delete m_serviceFlows;
  m_serviceFlows = 0;
}

void
SSRecord::SetIPAddress (Ipv4Address IPAddress)
{
  m_IPAddress = IPAddress;
}

Ipv4Address SSRecord::GetIPAddress (void)
{
  return m_IPAddress;
}

void
SSRecord::SetBasicCid (Cid basicCid)
{
  m_basicCid = basicCid;
}

Cid
SSRecord::GetBasicCid (void) const
{
  return m_basicCid;
}

void
SSRecord::SetPrimaryCid (Cid primaryCid)
{
  m_primaryCid = primaryCid;
}

Cid
SSRecord::GetPrimaryCid (void) const
{
  return m_primaryCid;
}

void
SSRecord::SetMacAddress (Mac48Address macAddress)
{
  m_macAddress = macAddress;
}

Mac48Address
SSRecord::GetMacAddress (void) const
{
  return m_macAddress;
}

uint8_t
SSRecord::GetRangingCorrectionRetries (void) const
{
  return m_rangingCorrectionRetries;
}

void
SSRecord::ResetRangingCorrectionRetries (void)
{
  m_rangingCorrectionRetries = 0;
}

void
SSRecord::IncrementRangingCorrectionRetries (void)
{
  m_rangingCorrectionRetries++;
}

uint8_t
SSRecord::GetInvitedRangRetries (void) const
{
  return m_invitedRangingRetries;
}

void
SSRecord::ResetInvitedRangingRetries (void)
{
  m_invitedRangingRetries = 0;
}

void
SSRecord::IncrementInvitedRangingRetries (void)
{
  m_invitedRangingRetries++;
}

void
SSRecord::SetModulationType (WimaxPhy::ModulationType modulationType)
{
  m_modulationType = modulationType;
}

WimaxPhy::ModulationType
SSRecord::GetModulationType (void) const
{
  return m_modulationType;
}

void
SSRecord::SetRangingStatus (WimaxNetDevice::RangingStatus rangingStatus)
{
  m_rangingStatus = rangingStatus;
}

WimaxNetDevice::RangingStatus
SSRecord::GetRangingStatus (void) const
{
  return m_rangingStatus;
}

void
SSRecord::EnablePollForRanging (void)
{
  m_pollForRanging = true;
}

void
SSRecord::DisablePollForRanging (void)
{
  m_pollForRanging = false;
}

bool
SSRecord::GetPollForRanging (void) const
{
  return m_pollForRanging;
}

void
SSRecord::SetAreServiceFlowsAllocated (bool val)
{
  m_areServiceFlowsAllocated = val;
}

bool
SSRecord::GetAreServiceFlowsAllocated (void) const
{
  return m_areServiceFlowsAllocated;
}

void
SSRecord::SetPollMeBit (bool pollMeBit)
{
  m_pollMeBit = pollMeBit;
}

bool
SSRecord::GetPollMeBit (void) const
{
  return m_pollMeBit;
}

void
SSRecord::AddServiceFlow (ServiceFlow *serviceFlow)
{
  m_serviceFlows->push_back (serviceFlow);
}

std::vector<ServiceFlow*> SSRecord::GetServiceFlows (enum ServiceFlow::SchedulingType schedulingType) const
{
  std::vector<ServiceFlow*> tmpServiceFlows;
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if (((*iter)->GetSchedulingType () == schedulingType) || (schedulingType == ServiceFlow::SF_TYPE_ALL))
        {
          tmpServiceFlows.push_back ((*iter));
        }
    }
  return tmpServiceFlows;
}

void
SSRecord::SetIsBroadcastSS (bool broadcast_enable)
{
  m_broadcast = broadcast_enable;
}

bool
SSRecord::GetIsBroadcastSS (void)
{
  return m_broadcast;
}

bool
SSRecord::GetHasServiceFlowUgs (void) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetSchedulingType () == ServiceFlow::SF_TYPE_UGS)
        {
          return true;
        }
    }
  return false;
}

bool
SSRecord::GetHasServiceFlowRtps (void) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetSchedulingType () == ServiceFlow::SF_TYPE_RTPS)
        {
          return true;
        }
    }
  return false;
}

bool
SSRecord::GetHasServiceFlowNrtps (void) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetSchedulingType () == ServiceFlow::SF_TYPE_NRTPS)
        {
          return true;
        }
    }
  return false;
}

bool
SSRecord::GetHasServiceFlowBe (void) const
{
  for (std::vector<ServiceFlow*>::iterator iter = m_serviceFlows->begin (); iter != m_serviceFlows->end (); ++iter)
    {
      if ((*iter)->GetSchedulingType () == ServiceFlow::SF_TYPE_BE)
        {
          return true;
        }
    }
  return false;
}

void
SSRecord::SetSfTransactionId (uint16_t sfTransactionId)
{
  m_sfTransactionId = sfTransactionId;
}

uint16_t SSRecord::GetSfTransactionId (void) const
{
  return m_sfTransactionId;
}

void
SSRecord::SetDsaRspRetries (uint8_t dsaRspRetries)
{
  m_dsaRspRetries = dsaRspRetries;
}

void
SSRecord::IncrementDsaRspRetries (void)
{
  m_dsaRspRetries++;
}

uint8_t
SSRecord::GetDsaRspRetries (void) const
{
  return m_dsaRspRetries;
}

void
SSRecord::SetDsaRsp (DsaRsp dsaRsp)
{
  m_dsaRsp = dsaRsp;
}

DsaRsp
SSRecord::GetDsaRsp (void) const
{
  return m_dsaRsp;
}

} // namespace ns3
