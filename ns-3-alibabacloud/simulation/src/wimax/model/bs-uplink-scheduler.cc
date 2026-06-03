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

#include "bs-uplink-scheduler.h"
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

NS_LOG_COMPONENT_DEFINE ("UplinkScheduler");

NS_OBJECT_ENSURE_REGISTERED (UplinkScheduler);

UplinkScheduler::UplinkScheduler (void)
  : m_bs (0),
    m_timeStampIrInterval (Seconds (0)),
    m_nrIrOppsAllocated (0),
    m_isIrIntrvlAllocated (false),
    m_isInvIrIntrvlAllocated (false),
    m_dcdTimeStamp (Simulator::Now ()),
    m_ucdTimeStamp (Simulator::Now ())
{
}

UplinkScheduler::UplinkScheduler (Ptr<BaseStationNetDevice> bs)
  : m_bs (bs),
    m_timeStampIrInterval (Seconds (0)),
    m_nrIrOppsAllocated (0),
    m_isIrIntrvlAllocated (false),
    m_isInvIrIntrvlAllocated (false),
    m_dcdTimeStamp (Simulator::Now ()),
    m_ucdTimeStamp (Simulator::Now ())
{
}

UplinkScheduler::~UplinkScheduler (void)
{
  m_bs = 0;
  m_uplinkAllocations.clear ();
}
void
UplinkScheduler::InitOnce ()
{

}

TypeId
UplinkScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UplinkScheduler")
    .SetParent<Object> ()
    .SetGroupName("Wimax")
  ;
  return tid;
}

uint8_t
UplinkScheduler::GetNrIrOppsAllocated (void) const
{
  return m_nrIrOppsAllocated;
}

void
UplinkScheduler::SetNrIrOppsAllocated (uint8_t nrIrOppsAllocated)
{
  m_nrIrOppsAllocated = nrIrOppsAllocated;
}

bool
UplinkScheduler::GetIsIrIntrvlAllocated (void) const
{
  return m_isIrIntrvlAllocated;
}

void
UplinkScheduler::SetIsIrIntrvlAllocated (bool isIrIntrvlAllocated)
{

  m_isIrIntrvlAllocated = isIrIntrvlAllocated;
}

bool
UplinkScheduler::GetIsInvIrIntrvlAllocated (void) const
{

  return m_isInvIrIntrvlAllocated;
}

void
UplinkScheduler::SetIsInvIrIntrvlAllocated (bool isInvIrIntrvlAllocated)
{
  m_isInvIrIntrvlAllocated = isInvIrIntrvlAllocated;
}

Time
UplinkScheduler::GetDcdTimeStamp (void) const
{
  return m_dcdTimeStamp;
}

void
UplinkScheduler::SetDcdTimeStamp (Time dcdTimeStamp)
{
  m_dcdTimeStamp = dcdTimeStamp;
}

Time
UplinkScheduler::GetUcdTimeStamp (void) const
{
  return m_ucdTimeStamp;
}

void
UplinkScheduler::SetUcdTimeStamp (Time ucdTimeStamp)
{
  m_ucdTimeStamp = ucdTimeStamp;
}

std::list<OfdmUlMapIe>
UplinkScheduler::GetUplinkAllocations (void) const
{
  return m_uplinkAllocations;
}

Time
UplinkScheduler::GetTimeStampIrInterval (void)
{
  return m_timeStampIrInterval;
}

void
UplinkScheduler::SetTimeStampIrInterval (Time timeStampIrInterval)
{
  m_timeStampIrInterval = timeStampIrInterval;
}

Ptr<BaseStationNetDevice>
UplinkScheduler::GetBs (void)
{
  return m_bs;
}
void
UplinkScheduler::SetBs (Ptr<BaseStationNetDevice> bs)
{
  m_bs = bs;
}
} // namespace ns3
