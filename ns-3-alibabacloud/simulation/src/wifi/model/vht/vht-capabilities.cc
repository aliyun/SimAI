/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015
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
 * Authors: Ghada Badawy <gbadawy@rim.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "vht-capabilities.h"

namespace ns3 {

VhtCapabilities::VhtCapabilities ()
  : m_maxMpduLength (0),
    m_supportedChannelWidthSet (0),
    m_rxLdpc (0),
    m_shortGuardIntervalFor80Mhz (0),
    m_shortGuardIntervalFor160Mhz (0),
    m_txStbc (0),
    m_rxStbc (0),
    m_suBeamformerCapable (0),
    m_suBeamformeeCapable (0),
    m_beamformeeStsCapable (0),
    m_numberOfSoundingDimensions (0),
    m_muBeamformerCapable (0),
    m_muBeamformeeCapable (0),
    m_vhtTxopPs (0),
    m_htcVhtCapable (0),
    m_maxAmpduLengthExponent (0),
    m_vhtLinkAdaptationCapable (0),
    m_rxAntennaPatternConsistency (0),
    m_txAntennaPatternConsistency (0),
    m_rxHighestSupportedLongGuardIntervalDataRate (0),
    m_txHighestSupportedLongGuardIntervalDataRate (0),
    m_vhtSupported (0)
{
  m_rxMcsMap.resize (8,0);
  m_txMcsMap.resize (8,0);
  for (uint8_t i = 0; i < 8; i++) //set to 3 by default, i.e. #spatial streams not supported. 0 means supported up to MCS 7, not what we want to imply at this stage.
    {
      m_rxMcsMap[i] = 3;
      m_txMcsMap[i] = 3;
    }
}

WifiInformationElementId
VhtCapabilities::ElementId () const
{
  return IE_VHT_CAPABILITIES;
}

void
VhtCapabilities::SetVhtSupported (uint8_t vhtSupported)
{
  m_vhtSupported = vhtSupported;
}

uint8_t
VhtCapabilities::GetInformationFieldSize () const
{
  //we should not be here if vht is not supported
  NS_ASSERT (m_vhtSupported > 0);
  return 12;
}

Buffer::Iterator
VhtCapabilities::Serialize (Buffer::Iterator i) const
{
  if (m_vhtSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
VhtCapabilities::GetSerializedSize () const
{
  if (m_vhtSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
VhtCapabilities::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_vhtSupported == 1)
    {
      //write the corresponding value for each bit
      start.WriteHtolsbU32 (GetVhtCapabilitiesInfo ());
      start.WriteHtolsbU64 (GetSupportedMcsAndNssSet ());
    }
}

uint8_t
VhtCapabilities::DeserializeInformationField (Buffer::Iterator start,
                                              uint8_t length)
{
  Buffer::Iterator i = start;
  uint32_t vhtinfo = i.ReadLsbtohU32 ();
  uint64_t mcsset = i.ReadLsbtohU64 ();
  SetVhtCapabilitiesInfo (vhtinfo);
  SetSupportedMcsAndNssSet (mcsset);
  return length;
}

void
VhtCapabilities::SetVhtCapabilitiesInfo (uint32_t ctrl)
{
  m_maxMpduLength = ctrl & 0x03;
  m_supportedChannelWidthSet = (ctrl >> 2) & 0x03;
  m_rxLdpc = (ctrl >> 4) & 0x01;
  m_shortGuardIntervalFor80Mhz = (ctrl >> 5) & 0x01;
  m_shortGuardIntervalFor160Mhz = (ctrl >> 6) & 0x01;
  m_txStbc = (ctrl >> 7) & 0x01;
  m_rxStbc = (ctrl >> 8) & 0x07;
  m_suBeamformerCapable = (ctrl >> 11) & 0x01;
  m_suBeamformeeCapable = (ctrl >> 12) & 0x01;
  m_beamformeeStsCapable = (ctrl >> 13) & 0x07;
  m_numberOfSoundingDimensions = (ctrl >> 16) & 0x07;
  m_muBeamformerCapable = (ctrl >> 19) & 0x01;
  m_muBeamformeeCapable = (ctrl >> 20) & 0x01;
  m_vhtTxopPs = (ctrl >> 21) & 0x01;
  m_htcVhtCapable = (ctrl >> 22) & 0x01;
  m_maxAmpduLengthExponent = (ctrl >> 23) & 0x07;
  m_vhtLinkAdaptationCapable = (ctrl >> 26) & 0x03;
  m_rxAntennaPatternConsistency = (ctrl >> 28) & 0x01;
  m_txAntennaPatternConsistency = (ctrl >> 29) & 0x01;
}

uint32_t
VhtCapabilities::GetVhtCapabilitiesInfo () const
{
  uint32_t val = 0;
  val |= m_maxMpduLength & 0x03;
  val |= (m_supportedChannelWidthSet & 0x03) << 2;
  val |= (m_rxLdpc & 0x01) << 4;
  val |= (m_shortGuardIntervalFor80Mhz & 0x01) << 5;
  val |= (m_shortGuardIntervalFor160Mhz & 0x01) << 6;
  val |= (m_txStbc & 0x01) << 7;
  val |= (m_rxStbc & 0x07) << 8;
  val |= (m_suBeamformerCapable & 0x01) << 11;
  val |= (m_suBeamformeeCapable & 0x01) << 12;
  val |= (m_beamformeeStsCapable & 0x07) << 13;
  val |= (m_numberOfSoundingDimensions & 0x07) << 16;
  val |= (m_muBeamformerCapable & 0x01) << 19;
  val |= (m_muBeamformeeCapable & 0x01) << 20;
  val |= (m_vhtTxopPs & 0x01) << 21;
  val |= (m_htcVhtCapable & 0x01) << 22;
  val |= (m_maxAmpduLengthExponent & 0x07) << 23;
  val |= (m_vhtLinkAdaptationCapable & 0x03) << 26;
  val |= (m_rxAntennaPatternConsistency & 0x01) << 28;
  val |= (m_txAntennaPatternConsistency & 0x01) << 29;
  return val;
}

void
VhtCapabilities::SetSupportedMcsAndNssSet (uint64_t ctrl)
{
  uint16_t n;
  for (uint8_t i = 0; i < 8; i++)
    {
      n = i * 2;
      m_rxMcsMap[i] = (ctrl >> n) & 0x03;
    }
  m_rxHighestSupportedLongGuardIntervalDataRate = (ctrl >> 16) & 0x1fff;
  for (uint8_t i = 0; i < 8; i++)
    {
      n = (i * 2) + 32;
      m_txMcsMap[i] = (ctrl >> n) & 0x03;
    }
  m_txHighestSupportedLongGuardIntervalDataRate = (ctrl >> 48) & 0x1fff;
}

uint64_t
VhtCapabilities::GetSupportedMcsAndNssSet () const
{
  uint64_t val = 0;
  uint16_t n;
  for (uint8_t i = 0; i < 8; i++)
    {
      n = i * 2;
      val |= (static_cast<uint64_t> (m_rxMcsMap[i]) & 0x03) << n;
    }
  val |=  (static_cast<uint64_t> (m_rxHighestSupportedLongGuardIntervalDataRate) & 0x1fff) << 16;
  for (uint8_t i = 0; i < 8; i++)
    {
      n = (i * 2) + 32;
      val |= (static_cast<uint64_t> (m_txMcsMap[i]) & 0x03) << n;
    }
  val |= (static_cast<uint64_t> (m_txHighestSupportedLongGuardIntervalDataRate) & 0x1fff) << 48;
  return val;
}

void
VhtCapabilities::SetMaxMpduLength (uint16_t length)
{
  NS_ABORT_MSG_IF (length != 3895 && length != 7991 && length != 11454,
                   "Invalid MPDU Max Length value");
  if (length == 11454)
    {
      m_maxMpduLength = 2;
    }
  else if (length == 7991)
    {
      m_maxMpduLength = 1;
    }
  else
    {
      m_maxMpduLength = 0;
    }
}

void
VhtCapabilities::SetSupportedChannelWidthSet (uint8_t channelWidthSet)
{
  m_supportedChannelWidthSet = channelWidthSet;
}

void
VhtCapabilities::SetRxLdpc (uint8_t rxLdpc)
{
  m_rxLdpc = rxLdpc;
}

void
VhtCapabilities::SetShortGuardIntervalFor80Mhz (uint8_t shortGuardInterval)
{
  m_shortGuardIntervalFor80Mhz = shortGuardInterval;
}

void
VhtCapabilities::SetShortGuardIntervalFor160Mhz (uint8_t shortGuardInterval)
{
  m_shortGuardIntervalFor160Mhz = shortGuardInterval;
}

void
VhtCapabilities::SetRxStbc (uint8_t rxStbc)
{
  m_rxStbc = rxStbc;
}

void
VhtCapabilities::SetTxStbc (uint8_t txStbc)
{
  m_txStbc = txStbc;
}

void
VhtCapabilities::SetMaxAmpduLength (uint32_t maxampdulength)
{
  for (uint8_t i = 0; i <= 7; i++)
    {
      if ((1ul << (13 + i)) - 1 == maxampdulength)
        {
          m_maxAmpduLengthExponent = i;
          return;
        }
    }
  NS_ABORT_MSG ("Invalid A-MPDU Max Length value");
}

void
VhtCapabilities::SetRxMcsMap (uint8_t mcs, uint8_t nss)
{
  //MCS index should be at least 7 and should not exceed 9
  NS_ASSERT (mcs >= 7 && mcs <= 9);
  m_rxMcsMap[nss - 1] = mcs - 7; //1 = MCS 8; 2 = MCS 9
}

void
VhtCapabilities::SetTxMcsMap (uint8_t mcs, uint8_t nss)
{
  //MCS index should be at least 7 and should not exceed 9
  NS_ASSERT (mcs >= 7 && mcs <= 9);
  m_txMcsMap[nss - 1] = mcs - 7; //1 = MCS 8; 2 = MCS 9
}

bool
VhtCapabilities::IsSupportedTxMcs (uint8_t mcs) const
{
  NS_ASSERT (mcs >= 0 && mcs <= 9);
  if (mcs <= 7)
    {
      return true;
    }
  if (mcs == 8 && (m_txMcsMap[0] == 1 || m_txMcsMap[0] == 2))
    {
      return true;
    }
  if (mcs == 9 && m_txMcsMap[0] == 2)
    {
      return true;
    }
  return false;
}

bool
VhtCapabilities::IsSupportedRxMcs (uint8_t mcs) const
{
  NS_ASSERT (mcs >= 0 && mcs <= 9);
  if (mcs <= 7)
    {
      return true;
    }
  if (mcs == 8 && (m_rxMcsMap[0] == 1 || m_rxMcsMap[0] == 2))
    {
      return true;
    }
  if (mcs == 9 && m_rxMcsMap[0] == 2)
    {
      return true;
    }
  return false;
}

void
VhtCapabilities::SetRxHighestSupportedLgiDataRate (uint16_t supportedDatarate)
{
  m_rxHighestSupportedLongGuardIntervalDataRate = supportedDatarate;
}

void
VhtCapabilities::SetTxHighestSupportedLgiDataRate (uint16_t supportedDatarate)
{
  m_txHighestSupportedLongGuardIntervalDataRate = supportedDatarate;
}

uint16_t
VhtCapabilities::GetMaxMpduLength (void) const
{
  if (m_maxMpduLength == 0)
    {
      return 3895;
    }
  if (m_maxMpduLength == 1)
    {
      return 7991;
    }
  if (m_maxMpduLength == 2)
    {
      return 11454;
    }
  NS_ABORT_MSG ("The value 3 is reserved");
}

uint8_t
VhtCapabilities::GetSupportedChannelWidthSet () const
{
  return m_supportedChannelWidthSet;
}

uint8_t
VhtCapabilities::GetRxLdpc () const
{
  return m_rxLdpc;
}

uint8_t
VhtCapabilities::GetRxStbc () const
{
  return m_rxStbc;
}

uint8_t
VhtCapabilities::GetTxStbc () const
{
  return m_txStbc;
}

uint32_t
VhtCapabilities::GetMaxAmpduLength (void) const
{
  return (1ul << (13 + m_maxAmpduLengthExponent)) - 1;
}

bool
VhtCapabilities::IsSupportedMcs (uint8_t mcs, uint8_t nss) const
{
  //The MCS index starts at 0 and NSS starts at 1
  if (mcs <= 7 && m_rxMcsMap[nss - 1] < 3)
    {
      return true;
    }
  if (mcs == 8 && m_rxMcsMap[nss - 1] > 0 && m_rxMcsMap[nss - 1] < 3)
    {
      return true;
    }
  if (mcs == 9 && m_rxMcsMap[nss - 1] == 2)
    {
      return true;
    }
  return false;
}

uint16_t
VhtCapabilities::GetRxHighestSupportedLgiDataRate () const
{
  return m_rxHighestSupportedLongGuardIntervalDataRate;
}

std::ostream &
operator << (std::ostream &os, const VhtCapabilities &vhtCapabilities)
{
  os <<  vhtCapabilities.GetVhtCapabilitiesInfo () << "|" << vhtCapabilities.GetSupportedMcsAndNssSet ();

  return os;
}

} //namespace ns3
