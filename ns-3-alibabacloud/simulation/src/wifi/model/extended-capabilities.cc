/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017
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
 * Authors: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "extended-capabilities.h"

namespace ns3 {

ExtendedCapabilities::ExtendedCapabilities ()
  : m_20_40_bssCoexistenceManagementSupport (0),
    m_extendedChannelSwitching (0),
    m_psmpCapability (0),
    m_spsmpSupport (0),
    m_event (0),
    m_diagnostics (0),
    m_multicastDiagnostics (0),
    m_locationTracking (0),
    m_fms (0),
    m_proxyArpService (0),
    m_collocatedInterferenceReporting (0),
    m_civicLocation (0),
    m_geospatialLocation (0),
    m_tfs (0),
    m_wnmSleepMode (0),
    m_timBroadcast (0),
    m_bssTransition (0),
    m_qosTrafficCapability (0),
    m_acStationCount (0),
    m_multipleBssid (0),
    m_timingMeasurement (0),
    m_channelUsage (0),
    m_ssidList (0),
    m_dms (0),
    m_utcTsfOffset (0),
    m_tpuBufferStaSupport (0),
    m_tdlsPeerPsmSupport (0),
    m_tdlsChannelSwitching (0),
    m_interworking (0),
    m_qosMap (0),
    m_ebr (0),
    m_sspnInterface (0),
    m_msgcfCapability (0),
    m_tdlsSupport (0),
    m_tdlsProhibited (0),
    m_tdlsChannelSwitchingProhibited (0),
    m_rejectUnadmittedFrame (0),
    m_serviceIntervalGranularity (0),
    m_identifierLocation (0),
    m_uapsdCoexistence (0),
    m_wnmNotification (0),
    m_qabCapability (0),
    m_utf8Ssid (0),
    m_qmfActivated (0),
    m_qmfReconfigurationActivated (0),
    m_robustAvStreaming (0),
    m_advancedGcr (0),
    m_meshGcr (0),
    m_scs (0),
    m_qloadReport (0),
    m_alternateEdca (0),
    m_unprotectedTxopNegotiation (0),
    m_protectedTxopNegotiation (0),
    m_protectedQloadReport (0),
    m_tdlsWiderBandwidth (0),
    m_operatingModeNotification (0),
    m_maxNumberOfMsdusInAmsdu (0),
    m_htSupported (0),
    m_vhtSupported (0)
{
}

WifiInformationElementId
ExtendedCapabilities::ElementId () const
{
  return IE_EXTENDED_CAPABILITIES;
}

void
ExtendedCapabilities::SetHtSupported (uint8_t htSupported)
{
  m_htSupported = htSupported;
}

void
ExtendedCapabilities::SetVhtSupported (uint8_t vhtSupported)
{
  m_vhtSupported = vhtSupported;
}

uint8_t
ExtendedCapabilities::GetInformationFieldSize () const
{
  //we should not be here if it is not supported
  NS_ASSERT (m_htSupported > 0 || m_vhtSupported > 0);
  if (!m_vhtSupported)
    {
      return 1;
    }
  return 8;
}

Buffer::Iterator
ExtendedCapabilities::Serialize (Buffer::Iterator i) const
{
  if (m_htSupported < 1 && m_vhtSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
ExtendedCapabilities::GetSerializedSize () const
{
  if (m_htSupported < 1 && m_vhtSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte1 (uint8_t ctrl)
{
  m_20_40_bssCoexistenceManagementSupport = ctrl & 0x01;
  m_extendedChannelSwitching = (ctrl >> 2) & 0x01;
  m_psmpCapability = (ctrl >> 4) & 0x01;
  m_spsmpSupport = (ctrl >> 6) & 0x01;
  m_event = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte1 (void) const
{
  uint8_t val = 0;
  val |= m_20_40_bssCoexistenceManagementSupport & 0x01;
  val |= (m_extendedChannelSwitching & 0x01) << 2;
  val |= (m_psmpCapability & 0x01) << 4;
  val |= (m_spsmpSupport & 0x01) << 6;
  val |= (m_event & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte2 (uint8_t ctrl)
{
  m_diagnostics = ctrl & 0x01;
  m_multicastDiagnostics = (ctrl >> 1) & 0x01;
  m_locationTracking = (ctrl >> 2) & 0x01;
  m_fms = (ctrl >> 3) & 0x01;
  m_proxyArpService = (ctrl >> 4) & 0x01;
  m_collocatedInterferenceReporting = (ctrl >> 5) & 0x01;
  m_civicLocation = (ctrl >> 6) & 0x01;
  m_geospatialLocation = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte2 (void) const
{
  uint8_t val = 0;
  val |= m_diagnostics & 0x01;
  val |= (m_multicastDiagnostics & 0x01) << 1;
  val |= (m_locationTracking & 0x01) << 2;
  val |= (m_fms & 0x01) << 3;
  val |= (m_proxyArpService & 0x01) << 4;
  val |= (m_collocatedInterferenceReporting & 0x01) << 5;
  val |= (m_civicLocation & 0x01) << 6;
  val |= (m_geospatialLocation & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte3 (uint8_t ctrl)
{
  m_tfs = ctrl & 0x01;
  m_wnmSleepMode = (ctrl >> 1) & 0x01;
  m_timBroadcast = (ctrl >> 2) & 0x01;
  m_bssTransition = (ctrl >> 3) & 0x01;
  m_qosTrafficCapability = (ctrl >> 4) & 0x01;
  m_acStationCount = (ctrl >> 5) & 0x01;
  m_multipleBssid = (ctrl >> 6) & 0x01;
  m_timingMeasurement = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte3 (void) const
{
  uint8_t val = 0;
  val |= m_tfs & 0x01;
  val |= (m_wnmSleepMode & 0x01) << 1;
  val |= (m_timBroadcast & 0x01) << 2;
  val |= (m_bssTransition & 0x01) << 3;
  val |= (m_qosTrafficCapability & 0x01) << 4;
  val |= (m_acStationCount & 0x01) << 5;
  val |= (m_multipleBssid & 0x01) << 6;
  val |= (m_timingMeasurement & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte4 (uint8_t ctrl)
{
  m_channelUsage = ctrl & 0x01;
  m_ssidList = (ctrl >> 1) & 0x01;
  m_dms = (ctrl >> 2) & 0x01;
  m_utcTsfOffset = (ctrl >> 3) & 0x01;
  m_tpuBufferStaSupport = (ctrl >> 4) & 0x01;
  m_tdlsPeerPsmSupport = (ctrl >> 5) & 0x01;
  m_tdlsChannelSwitching = (ctrl >> 6) & 0x01;
  m_interworking = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte4 (void) const
{
  uint8_t val = 0;
  val |= m_channelUsage & 0x01;
  val |= (m_ssidList & 0x01) << 1;
  val |= (m_dms & 0x01) << 2;
  val |= (m_utcTsfOffset & 0x01) << 3;
  val |= (m_tpuBufferStaSupport & 0x01) << 4;
  val |= (m_tdlsPeerPsmSupport & 0x01) << 5;
  val |= (m_tdlsChannelSwitching & 0x01) << 6;
  val |= (m_interworking & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte5 (uint8_t ctrl)
{
  m_qosMap = ctrl & 0x01;
  m_ebr = (ctrl >> 1) & 0x01;
  m_sspnInterface = (ctrl >> 2) & 0x01;
  m_msgcfCapability = (ctrl >> 4) & 0x01;
  m_tdlsSupport = (ctrl >> 5) & 0x01;
  m_tdlsProhibited = (ctrl >> 6) & 0x01;
  m_tdlsChannelSwitchingProhibited = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte5 (void) const
{
  uint8_t val = 0;
  val |= m_qosMap & 0x01;
  val |= (m_ebr & 0x01) << 1;
  val |= (m_sspnInterface & 0x01) << 2;
  val |= (m_msgcfCapability & 0x01) << 4;
  val |= (m_tdlsSupport & 0x01) << 5;
  val |= (m_tdlsProhibited & 0x01) << 6;
  val |= (m_tdlsChannelSwitchingProhibited & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte6 (uint8_t ctrl)
{
  m_rejectUnadmittedFrame = ctrl & 0x01;
  m_serviceIntervalGranularity = (ctrl >> 1) & 0x07;
  m_identifierLocation = (ctrl >> 4) & 0x01;
  m_uapsdCoexistence = (ctrl >> 5) & 0x01;
  m_wnmNotification = (ctrl >> 6) & 0x01;
  m_qabCapability = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte6 (void) const
{
  uint8_t val = 0;
  val |= m_rejectUnadmittedFrame & 0x01;
  val |= (m_serviceIntervalGranularity & 0x07) << 1;
  val |= (m_identifierLocation & 0x01) << 4;
  val |= (m_uapsdCoexistence & 0x01) << 5;
  val |= (m_wnmNotification & 0x01) << 6;
  val |= (m_qabCapability & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte7 (uint8_t ctrl)
{
  m_utf8Ssid = ctrl & 0x01;
  m_qmfActivated = (ctrl >> 1) & 0x01;
  m_qmfReconfigurationActivated = (ctrl >> 2) & 0x01;
  m_robustAvStreaming = (ctrl >> 3) & 0x01;
  m_advancedGcr = (ctrl >> 4) & 0x01;
  m_meshGcr = (ctrl >> 5) & 0x01;
  m_scs = (ctrl >> 6) & 0x01;
  m_qloadReport = (ctrl >> 7) & 0x01;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte7 (void) const
{
  uint8_t val = 0;
  val |= m_utf8Ssid & 0x01;
  val |= (m_qmfActivated & 0x01) << 1;
  val |= (m_qmfReconfigurationActivated & 0x01) << 2;
  val |= (m_robustAvStreaming & 0x01) << 3;
  val |= (m_advancedGcr & 0x01) << 4;
  val |= (m_meshGcr & 0x01) << 5;
  val |= (m_scs & 0x01) << 6;
  val |= (m_qloadReport & 0x01) << 7;
  return val;
}

void
ExtendedCapabilities::SetExtendedCapabilitiesByte8 (uint8_t ctrl)
{
  m_alternateEdca = ctrl & 0x01;
  m_unprotectedTxopNegotiation = (ctrl >> 1) & 0x01;
  m_protectedTxopNegotiation = (ctrl >> 2) & 0x01;
  m_protectedQloadReport = (ctrl >> 3) & 0x01;
  m_tdlsWiderBandwidth = (ctrl >> 4) & 0x01;
  m_operatingModeNotification = (ctrl >> 5) & 0x01;
  m_maxNumberOfMsdusInAmsdu = (ctrl >> 6) & 0x03;
}

uint8_t
ExtendedCapabilities::GetExtendedCapabilitiesByte8 (void) const
{
  uint8_t val = 0;
  val |= m_alternateEdca & 0x01;
  val |= (m_unprotectedTxopNegotiation & 0x01) << 1;
  val |= (m_protectedTxopNegotiation & 0x01) << 2;
  val |= (m_protectedQloadReport & 0x01) << 3;
  val |= (m_tdlsWiderBandwidth & 0x01) << 4;
  val |= (m_operatingModeNotification & 0x01) << 5;
  val |= (m_maxNumberOfMsdusInAmsdu & 0x03) << 6;
  return val;
}

void
ExtendedCapabilities::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_htSupported > 0 && m_vhtSupported == 0)
    {
      //write the corresponding value for each bit
      start.WriteU8 (GetExtendedCapabilitiesByte1 () & 0x7f);
    }
  else if (m_vhtSupported > 0)
    {
      //write the corresponding value for each bit
      start.WriteU8 (GetExtendedCapabilitiesByte1 ());
      start.WriteU8 (GetExtendedCapabilitiesByte2 ());
      start.WriteU8 (GetExtendedCapabilitiesByte3 ());
      start.WriteU8 (GetExtendedCapabilitiesByte4 ());
      start.WriteU8 (GetExtendedCapabilitiesByte5 ());
      start.WriteU8 (GetExtendedCapabilitiesByte6 ());
      start.WriteU8 (GetExtendedCapabilitiesByte7 ());
      start.WriteU8 (GetExtendedCapabilitiesByte8 ());
    }
}

uint8_t
ExtendedCapabilities::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  uint8_t byte1 = i.ReadU8 ();
  SetExtendedCapabilitiesByte1 (byte1);
  if (m_vhtSupported > 0)
    {
      uint8_t byte2 = i.ReadU8 ();
      uint8_t byte3 = i.ReadU8 ();
      uint8_t byte4 = i.ReadU8 ();
      uint8_t byte5 = i.ReadU8 ();
      uint8_t byte6 = i.ReadU8 ();
      uint8_t byte7 = i.ReadU8 ();
      uint8_t byte8 = i.ReadU8 ();
      SetExtendedCapabilitiesByte2 (byte2);
      SetExtendedCapabilitiesByte3 (byte3);
      SetExtendedCapabilitiesByte4 (byte4);
      SetExtendedCapabilitiesByte5 (byte5);
      SetExtendedCapabilitiesByte6 (byte6);
      SetExtendedCapabilitiesByte7 (byte7);
      SetExtendedCapabilitiesByte8 (byte8);
    }
  return length;
}

std::ostream &
operator << (std::ostream &os, const ExtendedCapabilities &extendedCapabilities)
{
  os << +extendedCapabilities.GetExtendedCapabilitiesByte1 ();
  return os;
}

} //namespace ns3
