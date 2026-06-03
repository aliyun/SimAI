/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */

#include "channel-scheduler.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ChannelScheduler");

NS_OBJECT_ENSURE_REGISTERED (ChannelScheduler);

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
TypeId
ChannelScheduler::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ChannelScheduler")
    .SetParent<Object> ()
    .SetGroupName ("Wave")
  ;
  return tid;
}

ChannelScheduler::ChannelScheduler ()
{
  NS_LOG_FUNCTION (this);
}
ChannelScheduler::~ChannelScheduler ()
{
  NS_LOG_FUNCTION (this);
}

void
ChannelScheduler::DoInitialize (void)
{
  // assign default CCH access when the device is initialized
  AssignDefaultCchAccess ();
}

void
ChannelScheduler::SetWaveNetDevice (Ptr<WaveNetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  m_device = device;
}

bool
ChannelScheduler::IsChannelAccessAssigned (uint32_t channelNumber) const
{
  NS_LOG_FUNCTION (this << channelNumber);
  return (GetAssignedAccessType (channelNumber) != NoAccess);
}

bool
ChannelScheduler::IsCchAccessAssigned (void) const
{
  NS_LOG_FUNCTION (this);
  return (GetAssignedAccessType (CCH) != NoAccess);
}

bool
ChannelScheduler::IsSchAccessAssigned (void) const
{
  NS_LOG_FUNCTION (this);
  return (GetAssignedAccessType (SCH1) != NoAccess) || (GetAssignedAccessType (SCH2) != NoAccess)
         || (GetAssignedAccessType (SCH3) != NoAccess) || (GetAssignedAccessType (SCH4) != NoAccess)
         || (GetAssignedAccessType (SCH5) != NoAccess) || (GetAssignedAccessType (SCH6) != NoAccess);
}

bool
ChannelScheduler::IsContinuousAccessAssigned (uint32_t channelNumber) const
{
  NS_LOG_FUNCTION (this << channelNumber);
  return (GetAssignedAccessType (channelNumber) == ContinuousAccess);
}
bool
ChannelScheduler::IsAlternatingAccessAssigned (uint32_t channelNumber) const
{
  NS_LOG_FUNCTION (this << channelNumber);
  return (GetAssignedAccessType (channelNumber) == AlternatingAccess);
}
bool
ChannelScheduler::IsExtendedAccessAssigned (uint32_t channelNumber) const
{
  NS_LOG_FUNCTION (this << channelNumber);
  return (GetAssignedAccessType (channelNumber) == ExtendedAccess);
}
bool
ChannelScheduler::IsDefaultCchAccessAssigned (void) const
{
  NS_LOG_FUNCTION (this);
  return (GetAssignedAccessType (CCH) == DefaultCchAccess);
}
bool
ChannelScheduler::StartSch (const SchInfo & schInfo)
{
  NS_LOG_FUNCTION (this << &schInfo);
  uint32_t cn = schInfo.channelNumber;

  if (ChannelManager::IsCch (schInfo.channelNumber))
    {
      NS_LOG_DEBUG ("the channel access requirement for CCH is not allowed.");
      return false;
    }
  uint32_t extends = schInfo.extendedAccess;
  bool immediate = schInfo.immediateAccess;
  Ptr<OcbWifiMac> mac = m_device->GetMac (cn);
  for (EdcaParametersI i = schInfo.edcaParameters.begin (); i != schInfo.edcaParameters.end (); ++i)
    {
      EdcaParameter edca = i->second;
      mac->ConfigureEdca (edca.cwmin, edca.cwmax, edca.aifsn, i->first);
    }

  if (extends == EXTENDED_CONTINUOUS)
    {
      return AssignContinuousAccess (cn, immediate);
    }
  else if (extends == EXTENDED_ALTERNATING)
    {
      return AssignAlternatingAccess (cn, immediate);
    }
  else
    {
      return AssignExtendedAccess (cn, extends, immediate);
    }
}

bool
ChannelScheduler::StopSch (uint32_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  if (ChannelManager::IsCch (channelNumber))
    {
      NS_LOG_DEBUG ("the channel access for CCH is not allowed to be released.");
      return false;
    }
  if (!IsChannelAccessAssigned (channelNumber))
    {
      NS_LOG_DEBUG ("the channel access for channel " << channelNumber << " has already been released.");
      return true;
    }
  return ReleaseAccess (channelNumber);
}

void
ChannelScheduler::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_device = 0;
}

} // namespace ns3
