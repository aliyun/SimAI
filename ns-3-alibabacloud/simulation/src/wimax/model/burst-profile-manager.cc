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

#include <stdint.h>
#include "burst-profile-manager.h"
#include "bs-net-device.h"
#include "ss-net-device.h"
#include "ss-record.h"
#include "ss-manager.h"
#include "ns3/log.h"
#include "mac-messages.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BurstProfileManager");

NS_OBJECT_ENSURE_REGISTERED (BurstProfileManager);

TypeId BurstProfileManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BurstProfileManager")
    .SetParent<Object> ()
    .SetGroupName("Wimax");
  return tid;
}

BurstProfileManager::BurstProfileManager (Ptr<WimaxNetDevice> device)
  : m_device (device)
{
}

BurstProfileManager::~BurstProfileManager (void)
{
  m_device = 0;
}

void
BurstProfileManager::DoDispose (void)
{
  m_device = 0;
}


uint16_t BurstProfileManager::GetNrBurstProfilesToDefine (void)
{
  /*
   * 7 modulation types
   */
  return 7;
}

WimaxPhy::ModulationType
BurstProfileManager::GetModulationType (uint8_t iuc,
                                        WimaxNetDevice::Direction direction) const
{
  if (direction == WimaxNetDevice::DIRECTION_DOWNLINK)
    {
      std::vector<OfdmDlBurstProfile> dlBurstProfiles =
        m_device->GetCurrentDcd ().GetDlBurstProfiles ();
      for (std::vector<OfdmDlBurstProfile>::iterator iter =
             dlBurstProfiles.begin (); iter != dlBurstProfiles.end (); ++iter)
        {
          if (iter->GetDiuc () == iuc)
            {
              return (WimaxPhy::ModulationType) iter->GetFecCodeType ();
            }
        }
    }
  else
    {
      std::vector<OfdmUlBurstProfile> ulBurstProfiles =
        m_device->GetCurrentUcd ().GetUlBurstProfiles ();
      for (std::vector<OfdmUlBurstProfile>::iterator iter =
             ulBurstProfiles.begin (); iter != ulBurstProfiles.end (); ++iter)
        {
          if (iter->GetUiuc () == iuc)
            {
              return (WimaxPhy::ModulationType) iter->GetFecCodeType ();
            }
        }
    }

  // burst profile got to be there in DCD/UCD, assuming always all profiles are defined in DCD/UCD
  NS_FATAL_ERROR ("burst profile got to be there in DCD/UCD");

  return (WimaxPhy::ModulationType) -1;
}

uint8_t
BurstProfileManager::GetBurstProfile (
  WimaxPhy::ModulationType modulationType,
  WimaxNetDevice::Direction direction) const
{
  if (direction == WimaxNetDevice::DIRECTION_DOWNLINK)
    {
      std::vector<OfdmDlBurstProfile> dlBurstProfiles =
        m_device->GetCurrentDcd ().GetDlBurstProfiles ();
      for (std::vector<OfdmDlBurstProfile>::iterator iter =
             dlBurstProfiles.begin (); iter != dlBurstProfiles.end (); ++iter)
        {
          if (iter->GetFecCodeType () == modulationType)
            {
              return iter->GetDiuc ();
            }
        }
    }
  else
    {
      std::vector<OfdmUlBurstProfile> ulBurstProfiles =
        m_device->GetCurrentUcd ().GetUlBurstProfiles ();
      for (std::vector<OfdmUlBurstProfile>::iterator iter =
             ulBurstProfiles.begin (); iter != ulBurstProfiles.end (); ++iter)
        {
          if (iter->GetFecCodeType () == modulationType)
            {
              return iter->GetUiuc ();
            }
        }
    }

  // burst profile got to be there in DCD/UCD, assuming always all profiles are defined in DCD/UCD
  NS_FATAL_ERROR ("burst profile got to be there in DCD/UCD");

  return ~0;
}

uint8_t
BurstProfileManager::GetBurstProfileForSS (const SSRecord *ssRecord,
                                           const RngReq *rngreq, WimaxPhy::ModulationType &modulationType)
{
  /*during initial ranging or periodic ranging (or when RNG-REQ is used instead of
   DBPC) calculates the least robust burst profile for SS, e.g., based on distance,
   power, signal etc, temporarily choosing same burst profile SS requested in RNG-REQ*/

  modulationType = GetModulationTypeForSS (ssRecord, rngreq);
  return GetBurstProfile (modulationType, WimaxNetDevice::DIRECTION_DOWNLINK);
}

WimaxPhy::ModulationType
BurstProfileManager::GetModulationTypeForSS (const SSRecord *ssRecord, const RngReq *rngreq)
{

  return GetModulationType (rngreq->GetReqDlBurstProfile (),
                            WimaxNetDevice::DIRECTION_DOWNLINK);
}

uint8_t
BurstProfileManager::GetBurstProfileToRequest (void)
{
  /*modulation type is currently set by user in simulation script, shall
   actually be determined based on SS's distance, power, signal etc*/

  return GetBurstProfile (
           m_device->GetObject<SubscriberStationNetDevice> ()->GetModulationType (),
           WimaxNetDevice::DIRECTION_DOWNLINK);
}

} // namespace ns3
