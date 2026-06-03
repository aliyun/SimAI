/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA
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

#include "service-flow-record.h"

namespace ns3 {

ServiceFlowRecord::ServiceFlowRecord (void)
  : m_grantSize (0),
    m_grantTimeStamp (Seconds (0)),
    m_dlTimeStamp (Seconds (0)),
    m_pktsSent (0),
    m_pktsRcvd (0),
    m_bytesSent (0),
    m_bytesRcvd (0),
    m_requestedBandwidth (0),
    m_grantedBandwidth (0),
    m_bwSinceLastExpiry (0)
{

  m_lastGrantTime = Seconds (0);
  m_backlogged = 0;
  m_backloggedTemp = 0;
  m_grantedBandwidthTemp = 0;
}

ServiceFlowRecord::~ServiceFlowRecord (void)
{
}

void
ServiceFlowRecord::SetGrantSize (uint32_t grantSize)
{
  m_grantSize = grantSize;
}

uint32_t
ServiceFlowRecord::GetGrantSize (void) const
{
  return m_grantSize;
}

void
ServiceFlowRecord::SetGrantTimeStamp (Time grantTimeStamp)
{
  m_grantTimeStamp = grantTimeStamp;
}

Time
ServiceFlowRecord::GetGrantTimeStamp (void) const
{
  return m_grantTimeStamp;
}

void
ServiceFlowRecord::SetDlTimeStamp (Time dlTimeStamp)
{
  m_dlTimeStamp = dlTimeStamp;
}

Time
ServiceFlowRecord::GetDlTimeStamp (void) const
{
  return m_dlTimeStamp;
}

void
ServiceFlowRecord::SetPktsSent (uint32_t pktsSent)
{
  m_pktsSent = pktsSent;
}

void
ServiceFlowRecord::UpdatePktsSent (uint32_t pktsSent)
{
  m_pktsSent += pktsSent;
}

uint32_t
ServiceFlowRecord::GetPktsSent (void) const
{
  return m_pktsSent;
}

void
ServiceFlowRecord::SetPktsRcvd (uint32_t pktsRcvd)
{
  m_pktsRcvd = pktsRcvd;
}

void
ServiceFlowRecord::UpdatePktsRcvd (uint32_t pktsRcvd)
{
  m_pktsRcvd += pktsRcvd;
}

uint32_t
ServiceFlowRecord::GetPktsRcvd (void) const
{
  return m_pktsRcvd;
}

void
ServiceFlowRecord::SetBytesSent (uint32_t bytesSent)
{
  m_bytesSent = bytesSent;
}

void
ServiceFlowRecord::UpdateBytesSent (uint32_t bytesSent)
{
  m_bytesSent += bytesSent;
}

uint32_t
ServiceFlowRecord::GetBytesSent (void) const
{
  return m_bytesSent;
}

void
ServiceFlowRecord::SetBytesRcvd (uint32_t bytesRcvd)
{
  m_bytesRcvd = bytesRcvd;
}

void
ServiceFlowRecord::UpdateBytesRcvd (uint32_t bytesRcvd)
{
  m_bytesRcvd += bytesRcvd;
}

uint32_t
ServiceFlowRecord::GetBytesRcvd (void) const
{
  return m_bytesRcvd;
}

void
ServiceFlowRecord::SetRequestedBandwidth (uint32_t requestedBandwidth)
{
  m_requestedBandwidth = requestedBandwidth;
}
void
ServiceFlowRecord::UpdateRequestedBandwidth (uint32_t requestedBandwidth)
{
  m_requestedBandwidth += requestedBandwidth;
}

uint32_t
ServiceFlowRecord::GetRequestedBandwidth (void)
{
  return m_requestedBandwidth;
}

void
ServiceFlowRecord::SetGrantedBandwidth (uint32_t grantedBandwidth)
{
  m_grantedBandwidth = grantedBandwidth;
}

void
ServiceFlowRecord::UpdateGrantedBandwidth (uint32_t grantedBandwidth)
{
  m_grantedBandwidth += grantedBandwidth;
}

uint32_t
ServiceFlowRecord::GetGrantedBandwidth (void)
{
  return m_grantedBandwidth;
}
void
ServiceFlowRecord::SetGrantedBandwidthTemp (uint32_t grantedBandwidthTemp)
{
  m_grantedBandwidthTemp = grantedBandwidthTemp;
}

void
ServiceFlowRecord::UpdateGrantedBandwidthTemp (
  uint32_t grantedBandwidthTemp)
{
  m_grantedBandwidthTemp += grantedBandwidthTemp;
}

uint32_t
ServiceFlowRecord::GetGrantedBandwidthTemp (void)
{
  return m_grantedBandwidthTemp;
}

void
ServiceFlowRecord::SetLastGrantTime (Time grantTime)
{
  m_lastGrantTime = grantTime;
}

Time
ServiceFlowRecord::GetLastGrantTime (void) const
{
  return m_lastGrantTime;
}

void
ServiceFlowRecord::SetBacklogged (uint32_t backlogged)
{
  m_backlogged = backlogged;
}

void
ServiceFlowRecord::IncreaseBacklogged (uint32_t backlogged)
{
  m_backlogged += backlogged;
}

uint32_t
ServiceFlowRecord::GetBacklogged (void) const
{
  return m_backlogged;
}

void
ServiceFlowRecord::SetBackloggedTemp (uint32_t backloggedTemp)
{
  m_backloggedTemp = backloggedTemp;
}

void
ServiceFlowRecord::IncreaseBackloggedTemp (uint32_t backloggedTemp)
{
  m_backloggedTemp += backloggedTemp;
}

uint32_t
ServiceFlowRecord::GetBackloggedTemp (void) const
{
  return m_backloggedTemp;
}

void
ServiceFlowRecord::SetBwSinceLastExpiry (uint32_t bwSinceLastExpiry)
{
  m_bwSinceLastExpiry = bwSinceLastExpiry;
}

void
ServiceFlowRecord::UpdateBwSinceLastExpiry (uint32_t bwSinceLastExpiry)
{
  m_bwSinceLastExpiry += bwSinceLastExpiry;
}

uint32_t
ServiceFlowRecord::GetBwSinceLastExpiry (void)
{
  return m_bwSinceLastExpiry;
}

} // namespace ns3
