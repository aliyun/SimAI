/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 CTTC
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
 * Authors: Nicola Baldo <nbaldo@cttc.es>
 *          Ghada Badawy <gbadawy@gmail.com>
 */

#include "wifi-tx-vector.h"
#include "wifi-phy-common.h"
#include "ns3/abort.h"

namespace ns3 {

WifiTxVector::WifiTxVector ()
  : m_preamble (WIFI_PREAMBLE_LONG),
    m_channelWidth (20),
    m_guardInterval (800),
    m_nTx (1),
    m_nss (1),
    m_ness (0),
    m_aggregation (false),
    m_stbc (false),
    m_ldpc (false),
    m_bssColor (0),
    m_length (0),
    m_modeInitialized (false)
{
}

WifiTxVector::WifiTxVector (WifiMode mode,
                            uint8_t powerLevel,
                            WifiPreamble preamble,
                            uint16_t guardInterval,
                            uint8_t nTx,
                            uint8_t nss,
                            uint8_t ness,
                            uint16_t channelWidth,
                            bool aggregation,
                            bool stbc,
                            bool ldpc,
                            uint8_t bssColor,
                            uint16_t length)
  : m_mode (mode),
    m_txPowerLevel (powerLevel),
    m_preamble (preamble),
    m_channelWidth (channelWidth),
    m_guardInterval (guardInterval),
    m_nTx (nTx),
    m_nss (nss),
    m_ness (ness),
    m_aggregation (aggregation),
    m_stbc (stbc),
    m_ldpc (ldpc),
    m_bssColor (bssColor),
    m_length (length),
    m_modeInitialized (true)
{
}

WifiTxVector::WifiTxVector (const WifiTxVector& txVector)
  : m_mode (txVector.m_mode),
    m_txPowerLevel (txVector.m_txPowerLevel),
    m_preamble (txVector.m_preamble),
    m_channelWidth (txVector.m_channelWidth),
    m_guardInterval (txVector.m_guardInterval),
    m_nTx (txVector.m_nTx),
    m_nss (txVector.m_nss),
    m_ness (txVector.m_ness),
    m_aggregation (txVector.m_aggregation),
    m_stbc (txVector.m_stbc),
    m_ldpc (txVector.m_ldpc),
    m_bssColor (txVector.m_bssColor),
    m_length (txVector.m_length),
    m_modeInitialized (txVector.m_modeInitialized)
{
  m_muUserInfos.clear ();
  if (!txVector.m_muUserInfos.empty ()) //avoids crashing for loop
    {
      for (auto & info : txVector.m_muUserInfos)
        {
          m_muUserInfos.insert (std::make_pair (info.first, info.second));
        }
    }
}

WifiTxVector::~WifiTxVector ()
{
  m_muUserInfos.clear ();
}

bool
WifiTxVector::GetModeInitialized (void) const
{
  return m_modeInitialized;
}

WifiMode
WifiTxVector::GetMode (uint16_t staId) const
{
  if (!m_modeInitialized)
    {
      NS_FATAL_ERROR ("WifiTxVector mode must be set before using");
    }
  if (IsMu ())
    {
      NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU (" << staId << ")");
      NS_ASSERT (m_muUserInfos.find (staId) != m_muUserInfos.end ());
      return m_muUserInfos.at (staId).mcs;
    }
  return m_mode;
}

WifiModulationClass
WifiTxVector::GetModulationClass (void) const
{
  NS_ABORT_MSG_IF (!m_modeInitialized, "WifiTxVector mode must be set before using");

  if (IsMu ())
    {
      NS_ASSERT (!m_muUserInfos.empty ());
      // all the modes belong to the same modulation class
      return m_muUserInfos.begin ()->second.mcs.GetModulationClass ();
    }
  return m_mode.GetModulationClass ();
}

uint8_t
WifiTxVector::GetTxPowerLevel (void) const
{
  return m_txPowerLevel;
}

WifiPreamble
WifiTxVector::GetPreambleType (void) const
{
  return m_preamble;
}

uint16_t
WifiTxVector::GetChannelWidth (void) const
{
  return m_channelWidth;
}

uint16_t
WifiTxVector::GetGuardInterval (void) const
{
  return m_guardInterval;
}

uint8_t
WifiTxVector::GetNTx (void) const
{
  return m_nTx;
}

uint8_t
WifiTxVector::GetNss (uint16_t staId) const
{
  if (IsMu ())
    {
      NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU (" << staId << ")");
      NS_ASSERT (m_muUserInfos.find (staId) != m_muUserInfos.end ());
      return m_muUserInfos.at (staId).nss;
    }
  return m_nss;
}

uint8_t
WifiTxVector::GetNssMax (void) const
{
  uint8_t nss = 0;
  if (IsMu ())
    {
      for (const auto & info : m_muUserInfos)
        {
          nss = (nss < info.second.nss) ? info.second.nss : nss;
        }
    }
  else
    {
      nss = m_nss;
    }
  return nss;
}

uint8_t
WifiTxVector::GetNess (void) const
{
  return m_ness;
}

bool
WifiTxVector::IsAggregation (void) const
{
  return m_aggregation;
}

bool
WifiTxVector::IsStbc (void) const
{
  return m_stbc;
}

bool
WifiTxVector::IsLdpc (void) const
{
  return m_ldpc;
}

void
WifiTxVector::SetMode (WifiMode mode)
{
  m_mode = mode;
  m_modeInitialized = true;
}

void
WifiTxVector::SetMode (WifiMode mode, uint16_t staId)
{
  NS_ABORT_MSG_IF (!IsMu (), "Not a MU transmission");
  NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU");
  m_muUserInfos[staId].mcs = mode;
  m_modeInitialized = true;
}

void
WifiTxVector::SetTxPowerLevel (uint8_t powerlevel)
{
  m_txPowerLevel = powerlevel;
}

void
WifiTxVector::SetPreambleType (WifiPreamble preamble)
{
  m_preamble = preamble;
}

void
WifiTxVector::SetChannelWidth (uint16_t channelWidth)
{
  m_channelWidth = channelWidth;
}

void
WifiTxVector::SetGuardInterval (uint16_t guardInterval)
{
  m_guardInterval = guardInterval;
}

void
WifiTxVector::SetNTx (uint8_t nTx)
{
  m_nTx = nTx;
}

void
WifiTxVector::SetNss (uint8_t nss)
{
  m_nss = nss;
}

void
WifiTxVector::SetNss (uint8_t nss, uint16_t staId)
{
  NS_ABORT_MSG_IF (!IsMu (), "Not a MU transmission");
  NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU");
  m_muUserInfos[staId].nss = nss;
}

void
WifiTxVector::SetNess (uint8_t ness)
{
  m_ness = ness;
}

void
WifiTxVector::SetAggregation (bool aggregation)
{
  m_aggregation = aggregation;
}

void
WifiTxVector::SetStbc (bool stbc)
{
  m_stbc = stbc;
}

void
WifiTxVector::SetLdpc (bool ldpc)
{
  m_ldpc = ldpc;
}

void
WifiTxVector::SetBssColor (uint8_t color)
{
  m_bssColor = color;
}

uint8_t
WifiTxVector::GetBssColor (void) const
{
  return m_bssColor;
}

void
WifiTxVector::SetLength (uint16_t length)
{
  m_length = length;
}

uint16_t
WifiTxVector::GetLength (void) const
{
  return m_length;
}

bool
WifiTxVector::IsValid (void) const
{
  if (!GetModeInitialized ())
    {
      return false;
    }
  std::string modeName = m_mode.GetUniqueName ();
  if (m_channelWidth == 20)
    {
      if (m_nss != 3 && m_nss != 6)
        {
          return (modeName != "VhtMcs9");
        }
    }
  else if (m_channelWidth == 80)
    {
      if (m_nss == 3 || m_nss == 7)
        {
          return (modeName != "VhtMcs6");
        }
      else if (m_nss == 6)
        {
          return (modeName != "VhtMcs9");
        }
    }
  else if (m_channelWidth == 160)
    {
      if (m_nss == 3)
        {
          return (modeName != "VhtMcs9");
        }
    }
  return true;
}

bool
WifiTxVector::IsMu (void) const
{
  return ns3::IsMu (m_preamble);
}

bool
WifiTxVector::IsDlMu (void) const
{
  return ns3::IsDlMu (m_preamble);
}

bool
WifiTxVector::IsUlMu (void) const
{
  return ns3::IsUlMu (m_preamble);
}

HeRu::RuSpec
WifiTxVector::GetRu (uint16_t staId) const
{
  NS_ABORT_MSG_IF (!IsMu (), "RU only available for MU");
  NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU");
  return m_muUserInfos.at (staId).ru;
}

void
WifiTxVector::SetRu (HeRu::RuSpec ru, uint16_t staId)
{
  NS_ABORT_MSG_IF (!IsMu (), "RU only available for MU");
  NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU");
  m_muUserInfos[staId].ru = ru;
}

HeMuUserInfo
WifiTxVector::GetHeMuUserInfo (uint16_t staId) const
{
  NS_ABORT_MSG_IF (!IsMu (), "HE MU user info only available for MU");
  return m_muUserInfos.at (staId);
}

void
WifiTxVector::SetHeMuUserInfo (uint16_t staId, HeMuUserInfo userInfo)
{
  NS_ABORT_MSG_IF (!IsMu (), "HE MU user info only available for MU");
  NS_ABORT_MSG_IF (staId > 2048, "STA-ID should be correctly set for MU");
  NS_ABORT_MSG_IF (userInfo.mcs.GetModulationClass () < WIFI_MOD_CLASS_HE, "Only HE (or newer) modes authorized for MU");
  m_muUserInfos[staId] = userInfo;
  m_modeInitialized = true;
}

const WifiTxVector::HeMuUserInfoMap&
WifiTxVector::GetHeMuUserInfoMap (void) const
{
  NS_ABORT_MSG_IF (!IsMu (), "HE MU user info map only available for MU");
  return m_muUserInfos;
}

WifiTxVector::HeMuUserInfoMap&
WifiTxVector::GetHeMuUserInfoMap (void)
{
  NS_ABORT_MSG_IF (!IsMu (), "HE MU user info map only available for MU");
  return m_muUserInfos;
}

std::pair<std::size_t, std::size_t>
WifiTxVector::GetNumRusPerHeSigBContentChannel (void) const
{
  //MU-MIMO is not handled for now, i.e. one station per RU

  if (m_channelWidth == 20)
    {
      return std::make_pair (m_muUserInfos.size (), 0); //all RUs are in HE-SIG-B content channel 1
    }

  HeRu::SubcarrierGroup toneRangesContentChannel1, toneRangesContentChannel2;
  // See section 27.3.10.8.3 of IEEE 802.11ax draft 4.0 for tone ranges per HE-SIG-B content channel
  switch (m_channelWidth)
    {
      case 40:
        toneRangesContentChannel1.push_back (std::make_pair (-244, -3));
        toneRangesContentChannel2.push_back (std::make_pair (3, 244));
        break;
      case 80:
        toneRangesContentChannel1.push_back (std::make_pair (-500, -259));
        toneRangesContentChannel2.push_back (std::make_pair (-258, -17));
        toneRangesContentChannel1.push_back (std::make_pair (-16, -4)); //first part of center carrier (in HE-SIG-B content channel 1)
        toneRangesContentChannel1.push_back (std::make_pair (4, 16)); //second part of center carrier (in HE-SIG-B content channel 1)
        toneRangesContentChannel1.push_back (std::make_pair (17, 258));
        toneRangesContentChannel2.push_back (std::make_pair (259, 500));
        break;
      case 160:
        toneRangesContentChannel1.push_back (std::make_pair (-1012, -771));
        toneRangesContentChannel2.push_back (std::make_pair (-770, -529));
        toneRangesContentChannel1.push_back (std::make_pair (-528, -516)); //first part of center carrier of lower 80 MHz band (in HE-SIG-B content channel 1)
        toneRangesContentChannel1.push_back (std::make_pair (-508, -496)); //second part of center carrier of lower 80 MHz band (in HE-SIG-B content channel 1)
        toneRangesContentChannel1.push_back (std::make_pair (-495, -254));
        toneRangesContentChannel2.push_back (std::make_pair (-253, -12));
        toneRangesContentChannel1.push_back (std::make_pair (12, 253));
        toneRangesContentChannel2.push_back (std::make_pair (254, 495));
        toneRangesContentChannel2.push_back (std::make_pair (496, 508)); //first part of center carrier of upper 80 MHz band (in HE-SIG-B content channel 2)
        toneRangesContentChannel2.push_back (std::make_pair (516, 528)); //second part of center carrier of upper 80 MHz band (in HE-SIG-B content channel 2)
        toneRangesContentChannel1.push_back (std::make_pair (529, 770));
        toneRangesContentChannel2.push_back (std::make_pair (771, 1012));
        break;
      default:
        NS_ABORT_MSG ("Unknown channel width: " << m_channelWidth);
    }

  std::size_t numRusContentChannel1 = 0;
  std::size_t numRusContentChannel2 = 0;
  for (auto & userInfo : m_muUserInfos)
    {
      HeRu::RuSpec ru = userInfo.second.ru;
      if (!ru.IsPhyIndexSet ())
        {
          // this method can be called when calculating the TX duration of a frame
          // and at that time the RU PHY index may have not been set yet
          ru.SetPhyIndex (m_channelWidth, 0);
        }
      if (HeRu::DoesOverlap (m_channelWidth, ru, toneRangesContentChannel1))
        {
          numRusContentChannel1++;
        }
      if (HeRu::DoesOverlap (m_channelWidth, ru, toneRangesContentChannel2))
        {
          numRusContentChannel2++;
        }
    }
  return std::make_pair (numRusContentChannel1, numRusContentChannel2);
}

std::ostream & operator << ( std::ostream &os, const WifiTxVector &v)
{
  if (!v.IsValid ())
    {
      os << "TXVECTOR not valid";
      return os;
    }
  os << "txpwrlvl: " << +v.GetTxPowerLevel ()
     << " preamble: " << v.GetPreambleType ()
     << " channel width: " << v.GetChannelWidth ()
     << " GI: " << v.GetGuardInterval ()
     << " NTx: " << +v.GetNTx ()
     << " Ness: " << +v.GetNess ()
     << " MPDU aggregation: " << v.IsAggregation ()
     << " STBC: " << v.IsStbc ()
     << " FEC coding: " << (v.IsLdpc () ? "LDPC" : "BCC");
  if (v.GetPreambleType () >= WIFI_PREAMBLE_HE_SU)
    {
      os << " BSS color: " << +v.GetBssColor ();
    }
  if (v.IsUlMu ())
    {
      os << " Length: " << v.GetLength ();
    }
  if (v.IsMu ())
    {
      WifiTxVector::HeMuUserInfoMap userInfoMap = v.GetHeMuUserInfoMap ();
      os << " num User Infos: " << userInfoMap.size ();
      for (auto & ui : userInfoMap)
        {
          os << ", {STA-ID: " << ui.first
             << ", " << ui.second.ru
             << ", MCS: " << ui.second.mcs
             << ", Nss: " << +ui.second.nss << "}";
        }
    }
  else
    {
      os << " mode: " << v.GetMode ()
         << " Nss: " << +v.GetNss ();
    }
  return os;
}

bool
HeMuUserInfo::operator== (const HeMuUserInfo& other) const
{
  return ru == other.ru
         && mcs.GetMcsValue () == other.mcs.GetMcsValue ()
         && nss == other.nss;
}

bool
HeMuUserInfo::operator!= (const HeMuUserInfo& other) const
{
  return !(*this == other);
}

} //namespace ns3
