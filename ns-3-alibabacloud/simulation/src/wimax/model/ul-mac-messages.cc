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

#include <stdint.h>
#include "ul-mac-messages.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (Ucd);

UcdChannelEncodings::UcdChannelEncodings (void)
  : m_bwReqOppSize (0),
    m_rangReqOppSize (0),
    m_frequency (0)
{
}

UcdChannelEncodings::~UcdChannelEncodings (void)
{
}

void
UcdChannelEncodings::SetBwReqOppSize (uint16_t bwReqOppSize)
{
  m_bwReqOppSize = bwReqOppSize;
}

void
UcdChannelEncodings::SetRangReqOppSize (uint16_t rangReqOppSize)
{
  m_rangReqOppSize = rangReqOppSize;
}

void
UcdChannelEncodings::SetFrequency (uint32_t frequency)
{
  m_frequency = frequency;
}

uint16_t
UcdChannelEncodings::GetBwReqOppSize (void) const
{
  return m_bwReqOppSize;
}

uint16_t
UcdChannelEncodings::GetRangReqOppSize (void) const
{
  return m_rangReqOppSize;
}

uint32_t
UcdChannelEncodings::GetFrequency (void) const
{
  return m_frequency;
}

uint16_t
UcdChannelEncodings::GetSize (void) const
{
  return 2 + 2 + 4;
}

Buffer::Iterator
UcdChannelEncodings::Write (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU16 (m_bwReqOppSize);
  i.WriteU16 (m_rangReqOppSize);
  i.WriteU32 (m_frequency);
  return DoWrite (i);
}

Buffer::Iterator
UcdChannelEncodings::Read (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_bwReqOppSize = i.ReadU16 ();
  m_rangReqOppSize = i.ReadU16 ();
  m_frequency = i.ReadU32 ();
  return DoRead (i);
}

// ----------------------------------------------------------------------------------------------------------

OfdmUcdChannelEncodings::OfdmUcdChannelEncodings (void)
  : m_sbchnlReqRegionFullParams (0),
    m_sbchnlFocContCodes (0)
{
}

OfdmUcdChannelEncodings::~OfdmUcdChannelEncodings (void)
{
}

void
OfdmUcdChannelEncodings::SetSbchnlReqRegionFullParams (
  uint8_t sbchnlReqRegionFullParams)
{
  m_sbchnlReqRegionFullParams = sbchnlReqRegionFullParams;
}

void
OfdmUcdChannelEncodings::SetSbchnlFocContCodes (uint8_t sbchnlFocContCodes)
{
  m_sbchnlFocContCodes = sbchnlFocContCodes;
}

uint8_t
OfdmUcdChannelEncodings::GetSbchnlReqRegionFullParams (void) const
{
  return m_sbchnlReqRegionFullParams;
}

uint8_t
OfdmUcdChannelEncodings::GetSbchnlFocContCodes (void) const
{
  return m_sbchnlFocContCodes;
}

uint16_t
OfdmUcdChannelEncodings::GetSize (void) const
{
  return UcdChannelEncodings::GetSize () + 1 + 1;
}

Buffer::Iterator
OfdmUcdChannelEncodings::DoWrite (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_sbchnlReqRegionFullParams);
  i.WriteU8 (m_sbchnlFocContCodes);
  return i;
}

Buffer::Iterator
OfdmUcdChannelEncodings::DoRead (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_sbchnlReqRegionFullParams = i.ReadU8 ();
  m_sbchnlFocContCodes = i.ReadU8 ();
  return i;
}

// ----------------------------------------------------------------------------------------------------------

OfdmUlBurstProfile::OfdmUlBurstProfile (void)
  : m_type (0),
    m_length (0),
    m_uiuc (0),
    m_fecCodeType (0)
{
}

OfdmUlBurstProfile::~OfdmUlBurstProfile (void)
{
}

void
OfdmUlBurstProfile::SetType (uint8_t type)
{
  m_type = type;
}

void
OfdmUlBurstProfile::SetLength (uint8_t length)
{
  m_length = length;
}

void
OfdmUlBurstProfile::SetUiuc (uint8_t uiuc)
{
  m_uiuc = uiuc;
}

void
OfdmUlBurstProfile::SetFecCodeType (uint8_t fecCodeType)
{
  m_fecCodeType = fecCodeType;
}

uint8_t
OfdmUlBurstProfile::GetType (void) const
{
  return m_type;
}

uint8_t
OfdmUlBurstProfile::GetLength (void) const
{
  return m_length;
}

uint8_t
OfdmUlBurstProfile::GetUiuc (void) const
{
  return m_uiuc;
}

uint8_t
OfdmUlBurstProfile::GetFecCodeType (void) const
{
  return m_fecCodeType;
}

uint16_t
OfdmUlBurstProfile::GetSize (void) const
{
  return 1 + 1 + 1 + 1;
}

Buffer::Iterator
OfdmUlBurstProfile::Write (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_type);
  i.WriteU8 (m_length);
  i.WriteU8 (m_uiuc);
  i.WriteU8 (m_fecCodeType);
  return i;
}

Buffer::Iterator
OfdmUlBurstProfile::Read (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_type = i.ReadU8 ();
  m_length = i.ReadU8 ();
  m_uiuc = i.ReadU8 ();
  m_fecCodeType = i.ReadU8 ();
  return i;
}

// ----------------------------------------------------------------------------------------------------------

Ucd::Ucd (void)
  : m_configurationChangeCount (0),
    m_rangingBackoffStart (0),
    m_rangingBackoffEnd (0),
    m_requestBackoffStart (0),
    m_requestBackoffEnd (
      0),
    m_nrUlBurstProfiles (0)
{
}

Ucd::~Ucd (void)
{
}

void
Ucd::SetConfigurationChangeCount (uint8_t configurationChangeCount)
{
  m_configurationChangeCount = configurationChangeCount;
}

void
Ucd::SetRangingBackoffStart (uint8_t rangingBackoffStart)
{
  m_rangingBackoffStart = rangingBackoffStart;
}

void
Ucd::SetRangingBackoffEnd (uint8_t rangingBackoffEnd)
{
  m_rangingBackoffEnd = rangingBackoffEnd;
}

void
Ucd::SetRequestBackoffStart (uint8_t requestBackoffStart)
{
  m_requestBackoffStart = requestBackoffStart;
}

void
Ucd::SetRequestBackoffEnd (uint8_t requestBackoffEnd)
{
  m_requestBackoffEnd = requestBackoffEnd;
}

void
Ucd::SetChannelEncodings (OfdmUcdChannelEncodings channelEncodings)
{
  m_channelEncodings = channelEncodings;
}

void
Ucd::AddUlBurstProfile (OfdmUlBurstProfile ulBurstProfile)
{
  m_ulBurstProfiles.push_back (ulBurstProfile);
}

void
Ucd::SetNrUlBurstProfiles (uint8_t nrUlBurstProfiles)
{
  // number of burst profiles is set to number of UL-MAP IEs after processing UL-MAP
  m_nrUlBurstProfiles = nrUlBurstProfiles;
}

uint8_t
Ucd::GetConfigurationChangeCount (void) const
{
  return m_configurationChangeCount;
}

uint8_t
Ucd::GetRangingBackoffStart (void) const
{
  return m_rangingBackoffStart;
}

uint8_t
Ucd::GetRangingBackoffEnd (void) const
{
  return m_rangingBackoffEnd;
}

uint8_t
Ucd::GetRequestBackoffStart (void) const
{
  return m_requestBackoffStart;
}

uint8_t
Ucd::GetRequestBackoffEnd (void) const
{
  return m_requestBackoffEnd;
}

OfdmUcdChannelEncodings
Ucd::GetChannelEncodings (void) const
{
  return m_channelEncodings;
}

std::vector<OfdmUlBurstProfile>
Ucd::GetUlBurstProfiles (void) const
{
  return m_ulBurstProfiles;
}

uint8_t
Ucd::GetNrUlBurstProfiles (void) const
{
  return m_nrUlBurstProfiles;
}

std::string
Ucd::GetName (void) const
{
  return "UCD";
}

TypeId
Ucd::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ucd")
    .SetParent<Header> ()
    .SetGroupName("Wimax")
    .AddConstructor<Ucd> ()
  ;
  return tid;
}

TypeId
Ucd::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
Ucd::Print (std::ostream &os) const
{
  os << " configuration change count = "
     << (uint32_t) m_configurationChangeCount << ", ranging backoff start = "
     << (uint32_t) m_rangingBackoffStart << ", ranging backoff end = "
     << (uint32_t) m_rangingBackoffEnd << ", request backoff start = "
     << (uint32_t) m_requestBackoffStart << ", request backoff end = "
     << (uint32_t) m_requestBackoffEnd << ", number of ul burst profiles = "
     << m_ulBurstProfiles.size ();
}

uint32_t
Ucd::GetSerializedSize (void) const
{
  uint32_t ulBurstProfilesSize = 0;

  for (std::vector<OfdmUlBurstProfile>::const_iterator iter =
         m_ulBurstProfiles.begin (); iter != m_ulBurstProfiles.end (); ++iter)
    {
      OfdmUlBurstProfile burstProfile = *iter;
      ulBurstProfilesSize += burstProfile.GetSize ();
    }

  return 5 + m_channelEncodings.GetSize () + ulBurstProfilesSize;
}

void
Ucd::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_configurationChangeCount);
  i.WriteU8 (m_rangingBackoffStart);
  i.WriteU8 (m_rangingBackoffEnd);
  i.WriteU8 (m_requestBackoffStart);
  i.WriteU8 (m_requestBackoffEnd);
  i = m_channelEncodings.Write (i);

  for (std::vector<OfdmUlBurstProfile>::const_iterator iter =
         m_ulBurstProfiles.begin (); iter != m_ulBurstProfiles.end (); ++iter)
    {
      OfdmUlBurstProfile burstProfile = *iter;
      i = burstProfile.Write (i);
    }
}

uint32_t
Ucd::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_configurationChangeCount = i.ReadU8 ();
  m_rangingBackoffStart = i.ReadU8 ();
  m_rangingBackoffEnd = i.ReadU8 ();
  m_requestBackoffStart = i.ReadU8 ();
  m_requestBackoffEnd = i.ReadU8 ();
  i = m_channelEncodings.Read (i);

  for (uint8_t j = 0; j < m_nrUlBurstProfiles; j++)
    {
      OfdmUlBurstProfile burstProfile;
      i = burstProfile.Read (i);
      AddUlBurstProfile (burstProfile);
    }

  return i.GetDistanceFrom (start);
}

// ----------------------------------------------------------------------------------------------------------

OfdmUlMapIe::OfdmUlMapIe (void)
  : m_cid (),
    m_startTime (0),
    m_subchannelIndex (0),
    m_uiuc (0),
    m_duration (0),
    m_midambleRepetitionInterval (0)
{
}

OfdmUlMapIe::~OfdmUlMapIe (void)
{
}

void
OfdmUlMapIe::SetCid (const Cid &cid)
{
  m_cid = cid;
}

void
OfdmUlMapIe::SetStartTime (uint16_t startTime)
{
  m_startTime = startTime;
}

void
OfdmUlMapIe::SetSubchannelIndex (uint8_t subchannelIndex)
{
  m_subchannelIndex = subchannelIndex;
}

void
OfdmUlMapIe::SetUiuc (uint8_t uiuc)
{
  m_uiuc = uiuc;
}

void
OfdmUlMapIe::SetDuration (uint16_t duration)
{
  m_duration = duration;
}

void
OfdmUlMapIe::SetMidambleRepetitionInterval (
  uint8_t midambleRepetitionInterval)
{
  m_midambleRepetitionInterval = midambleRepetitionInterval;
}

Cid
OfdmUlMapIe::GetCid (void) const
{
  return m_cid;
}

uint16_t
OfdmUlMapIe::GetStartTime (void) const
{
  return m_startTime;
}

uint8_t
OfdmUlMapIe::GetSubchannelIndex (void) const
{
  return m_subchannelIndex;
}

uint8_t
OfdmUlMapIe::GetUiuc (void) const
{
  return m_uiuc;
}

uint16_t
OfdmUlMapIe::GetDuration (void) const
{
  return m_duration;
}

uint8_t
OfdmUlMapIe::GetMidambleRepetitionInterval (void) const
{
  return m_midambleRepetitionInterval;
}

uint16_t
OfdmUlMapIe::GetSize (void) const
{
  return 2 + 2 + 1 + 1 + 2 + 1;
}

Buffer::Iterator
OfdmUlMapIe::Write (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU16 (m_cid.GetIdentifier ());
  i.WriteU16 (m_startTime);
  i.WriteU8 (m_subchannelIndex);
  i.WriteU8 (m_uiuc);
  i.WriteU16 (m_duration);
  i.WriteU8 (m_midambleRepetitionInterval);
  return i;
}

Buffer::Iterator
OfdmUlMapIe::Read (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_cid = i.ReadU16 ();
  m_startTime = i.ReadU16 ();
  m_subchannelIndex = i.ReadU8 ();
  m_uiuc = i.ReadU8 ();
  m_duration = i.ReadU16 ();
  m_midambleRepetitionInterval = i.ReadU8 ();
  return i;
}

// ----------------------------------------------------------------------------------------------------------

NS_OBJECT_ENSURE_REGISTERED (UlMap);

UlMap::UlMap (void)
  : m_reserved (0),
    m_ucdCount (0),
    m_allocationStartTime (0)
{
}

UlMap::~UlMap (void)
{
}

void
UlMap::SetUcdCount (uint8_t ucdCount)
{
  m_ucdCount = ucdCount;
}

void
UlMap::SetAllocationStartTime (uint32_t allocationStartTime)
{
  m_allocationStartTime = allocationStartTime;
}

void
UlMap::AddUlMapElement (OfdmUlMapIe ulMapElement)
{
  m_ulMapElements.push_back (ulMapElement);
}

uint8_t
UlMap::GetUcdCount (void) const
{
  return m_ucdCount;
}

uint32_t
UlMap::GetAllocationStartTime (void) const
{
  return m_allocationStartTime;
}

std::list<OfdmUlMapIe>
UlMap::GetUlMapElements (void) const
{
  return m_ulMapElements;
}

std::string
UlMap::GetName (void) const
{
  return "UL-MAP";
}

TypeId
UlMap::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UlMap")
    .SetParent<Header> ()
    .SetGroupName("Wimax")
    .AddConstructor<UlMap> ()
  ;
  return tid;
}

TypeId
UlMap::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
UlMap::Print (std::ostream &os) const
{
  os << " ucd count = " << (uint32_t) m_ucdCount
     << ", allocation start time = " << m_allocationStartTime
     << ", number of ul-map elements = " << m_ulMapElements.size ();
}

uint32_t
UlMap::GetSerializedSize (void) const
{
  uint32_t ulMapElementsSize = 0;
  for (std::list<OfdmUlMapIe>::const_iterator iter = m_ulMapElements.begin (); iter
       != m_ulMapElements.end (); ++iter)
    {
      OfdmUlMapIe ulMapIe = *iter;
      ulMapElementsSize += ulMapIe.GetSize ();
    }

  return 1 + 1 + 4 + ulMapElementsSize;
}

void
UlMap::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  i.WriteU8 (m_reserved);
  i.WriteU8 (m_ucdCount);
  i.WriteU32 (m_allocationStartTime);

  for (std::list<OfdmUlMapIe>::const_iterator iter = m_ulMapElements.begin (); iter
       != m_ulMapElements.end (); ++iter)
    {
      OfdmUlMapIe ulMapIe = *iter;
      i = ulMapIe.Write (i);
    }
}

uint32_t
UlMap::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  m_reserved = i.ReadU8 ();
  m_ucdCount = i.ReadU8 ();
  m_allocationStartTime = i.ReadU32 ();

  m_ulMapElements.clear (); // only for printing, otherwise it shows wrong number of elements

  while (true)
    {
      OfdmUlMapIe ulMapIe;
      i = ulMapIe.Read (i);

      AddUlMapElement (ulMapIe);

      if (ulMapIe.GetUiuc () == 14) // End of Map IE
        {
          break;
        }
    }
  return i.GetDistanceFrom (start);
}

} // namespace ns3
