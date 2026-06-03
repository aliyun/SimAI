/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#include "lte-ffr-enhanced-algorithm.h"
#include "ns3/ff-mac-common.h"
#include "ns3/lte-common.h"
#include "ns3/lte-vendor-specific-parameters.h"
#include <ns3/log.h>
#include "ns3/boolean.h"
#include <ns3/double.h>
#include <cfloat>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFfrEnhancedAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (LteFfrEnhancedAlgorithm);

/// Spectral efficiency for CQI table
static const double SpectralEfficiencyForCqi[16] = {
  0.0, // out of range
  0.15, 0.23, 0.38, 0.6, 0.88, 1.18,
  1.48, 1.91, 2.41,
  2.73, 3.32, 3.9, 4.52, 5.12, 5.55
};

/// FfrEnhancedDownlinkDefaultConfiguration structure
static const struct FfrEnhancedDownlinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t dlBandwidth; ///< DL bandwidth
  uint8_t dlSubBandOffset; ///< DL subband offset
  uint8_t dlReuse3SubBandwidth; ///< reuse 3 subbandwidth
  uint8_t dlReuse1SubBandwidth; ///< reuse 1 subbandwidth
} g_ffrEnhancedDownlinkDefaultConfiguration[] = {
  { 1, 25, 0, 4, 4},
  { 2, 25, 8, 4, 4},
  { 3, 25, 16, 4, 4},
  { 1, 50, 0, 9, 6},
  { 2, 50, 15, 9, 6},
  { 3, 50, 30, 9, 6},
  { 1, 75, 0, 8, 16},
  { 2, 75, 24, 8, 16},
  { 3, 75, 48, 8, 16},
  { 1, 100, 0, 16, 16},
  { 2, 100, 32, 16, 16},
  { 3, 100, 64, 16, 16}
}; ///< the enhanced downlink default configation

/// FfrEnhancedUplinkDefaultConfiguration structure
static const struct FfrEnhancedUplinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t ulBandwidth; ///< UL bandwidth
  uint8_t ulSubBandOffset; ///< UL subband offset
  uint8_t ulReuse3SubBandwidth; ///< UL reuse 3 subbandwidth
  uint8_t ulReuse1SubBandwidth; ///< UL reuse 1 subbandwidth
} g_ffrEnhancedUplinkDefaultConfiguration[] = {
  { 1, 25, 0, 4, 4},
  { 2, 25, 8, 4, 4},
  { 3, 25, 16, 4, 4},
  { 1, 50, 0, 9, 6},
  { 2, 50, 15, 9, 6},
  { 3, 50, 30, 9, 6},
  { 1, 75, 0, 8, 16},
  { 2, 75, 24, 8, 16},
  { 3, 75, 48, 8, 16},
  { 1, 100, 0, 16, 16},
  { 2, 100, 32, 16, 16},
  { 3, 100, 64, 16, 16}
}; ///< the enhanced uplink default configuration

/** \returns number of downlink configurations */
const uint16_t NUM_DOWNLINK_CONFS (sizeof (g_ffrEnhancedDownlinkDefaultConfiguration) / sizeof (FfrEnhancedDownlinkDefaultConfiguration));
/** \returns number of uplink configurations */
const uint16_t NUM_UPLINK_CONFS (sizeof (g_ffrEnhancedUplinkDefaultConfiguration) / sizeof (FfrEnhancedUplinkDefaultConfiguration));


LteFfrEnhancedAlgorithm::LteFfrEnhancedAlgorithm ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0),
    m_measId (0)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFfrEnhancedAlgorithm> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFfrEnhancedAlgorithm> (this);
}


LteFfrEnhancedAlgorithm::~LteFfrEnhancedAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFfrEnhancedAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFfrEnhancedAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFfrEnhancedAlgorithm")
    .SetParent<LteFfrAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<LteFfrEnhancedAlgorithm> ()
    .AddAttribute ("UlSubBandOffset",
                   "Uplink SubBand Offset for this cell in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_ulSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlReuse3SubBandwidth",
                   "Uplink Reuse 3 SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (4),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_ulReuse3SubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlReuse1SubBandwidth",
                   "Uplink Reuse 1 SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (4),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_ulReuse1SubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlSubBandOffset",
                   "Downlink SubBand Offset for this cell in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_dlSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlReuse3SubBandwidth",
                   "Downlink Reuse 3 SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (4),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_dlReuse3SubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlReuse1SubBandwidth",
                   "Downlink Reuse 1 SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (4),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_dlReuse1SubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Edge sub-band",
                   UintegerValue (26),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_rsrqThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaPowerOffset",
                   "PdschConfigDedicated::Pa value for Center Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_centerAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaPowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_edgeAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlCqiThreshold",
                   "If the DL-CQI for RBG of is higher than this threshold, transmission on RBG is possible",
                   UintegerValue (15),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_dlCqiThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlCqiThreshold",
                   "If the UL-CQI for RBG of is higher than this threshold, transmission on RBG is possible",
                   UintegerValue (15),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_ulCqiThreshold),
                   MakeUintegerChecker <uint8_t> ())
    .AddAttribute ("CenterAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in center area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_centerAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in edge area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrEnhancedAlgorithm::m_edgeAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
  ;
  return tid;
}


void
LteFfrEnhancedAlgorithm::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFfrEnhancedAlgorithm::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFfrEnhancedAlgorithm::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFfrEnhancedAlgorithm::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFfrEnhancedAlgorithm::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteFfrAlgorithm::DoInitialize ();

  NS_ASSERT_MSG (m_dlBandwidth > 24,"DlBandwidth must be at least 25 to use EFFR algorithm");
  NS_ASSERT_MSG (m_ulBandwidth > 24,"UlBandwidth must be at least 25 to use EFFR algorithm");

  if (m_frCellTypeId != 0)
    {
      SetDownlinkConfiguration (m_frCellTypeId, m_dlBandwidth);
      SetUplinkConfiguration (m_frCellTypeId, m_ulBandwidth);
    }

  NS_LOG_LOGIC (this << " requesting Event A1 measurements"
                     << " (threshold = 0" << ")");
  LteRrcSap::ReportConfigEutra reportConfig;
  reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A1;
  reportConfig.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRQ;
  reportConfig.threshold1.range = 0;
  reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRQ;
  reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS120;
  m_measId = m_ffrRrcSapUser->AddUeMeasReportConfigForFfr (reportConfig);
}

void
LteFfrEnhancedAlgorithm::Reconfigure ()
{
  NS_LOG_FUNCTION (this);
  if (m_frCellTypeId != 0)
    {
      SetDownlinkConfiguration (m_frCellTypeId, m_dlBandwidth);
      SetUplinkConfiguration (m_frCellTypeId, m_ulBandwidth);
    }
  InitializeDownlinkRbgMaps ();
  InitializeUplinkRbgMaps ();
  m_needReconfiguration = false;
}

void
LteFfrEnhancedAlgorithm::SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_DOWNLINK_CONFS; ++i)
    {
      if ((g_ffrEnhancedDownlinkDefaultConfiguration[i].cellId == cellId)
          && g_ffrEnhancedDownlinkDefaultConfiguration[i].dlBandwidth == m_dlBandwidth)
        {
          m_dlSubBandOffset = g_ffrEnhancedDownlinkDefaultConfiguration[i].dlSubBandOffset;
          m_dlReuse3SubBandwidth = g_ffrEnhancedDownlinkDefaultConfiguration[i].dlReuse3SubBandwidth;
          m_dlReuse1SubBandwidth = g_ffrEnhancedDownlinkDefaultConfiguration[i].dlReuse1SubBandwidth;
        }
    }
}

void
LteFfrEnhancedAlgorithm::SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_UPLINK_CONFS; ++i)
    {
      if ((g_ffrEnhancedUplinkDefaultConfiguration[i].cellId == cellId)
          && g_ffrEnhancedUplinkDefaultConfiguration[i].ulBandwidth == m_ulBandwidth)
        {
          m_ulSubBandOffset = g_ffrEnhancedUplinkDefaultConfiguration[i].ulSubBandOffset;
          m_ulReuse3SubBandwidth = g_ffrEnhancedUplinkDefaultConfiguration[i].ulReuse3SubBandwidth;
          m_ulReuse1SubBandwidth = g_ffrEnhancedUplinkDefaultConfiguration[i].ulReuse1SubBandwidth;
        }
    }
}

int
LteFfrEnhancedAlgorithm::GetCqiFromSpectralEfficiency (double s)
{
  NS_LOG_FUNCTION (s);
  NS_ASSERT_MSG (s >= 0.0, "negative spectral efficiency = " << s);
  int cqi = 0;
  while ((cqi < 15) && (SpectralEfficiencyForCqi[cqi + 1] < s))
    {
      ++cqi;
    }
  NS_LOG_LOGIC ("cqi = " << cqi);
  return cqi;
}

void
LteFfrEnhancedAlgorithm::InitializeDownlinkRbgMaps ()
{
  m_dlRbgMap.clear ();
  m_dlReuse3RbgMap.clear ();
  m_dlReuse1RbgMap.clear ();
  m_dlPrimarySegmentRbgMap.clear ();
  m_dlSecondarySegmentRbgMap.clear ();

  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgMap.resize (m_dlBandwidth / rbgSize, true);

  m_dlReuse3RbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlReuse1RbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlPrimarySegmentRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlSecondarySegmentRbgMap.resize (m_dlBandwidth / rbgSize, true);

  NS_ASSERT_MSG (m_dlSubBandOffset <= m_dlBandwidth,"DlSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlSubBandOffset + m_dlReuse3SubBandwidth + m_dlReuse1SubBandwidth <= m_dlBandwidth,
                 "DlSubBandOffset + DlReuse3SubBandwidth + DlReuse1SubBandwidth  higher than DlBandwidth");

  for (uint8_t i = 0; i < m_dlReuse3SubBandwidth / rbgSize; i++)
    {
      int offset = m_dlSubBandOffset / rbgSize;
      uint8_t index = offset + i;
      m_dlReuse3RbgMap[index] = true;
      m_dlPrimarySegmentRbgMap[index] = true;
      m_dlRbgMap[index] = false;
    }

  for (uint8_t i = 0; i < m_dlReuse1SubBandwidth / rbgSize; i++)
    {
      int offset = (m_dlSubBandOffset + m_dlReuse3SubBandwidth) / rbgSize;
      uint8_t index = offset + i;
      m_dlReuse1RbgMap[index] = true;
      m_dlPrimarySegmentRbgMap[index] = true;
      m_dlSecondarySegmentRbgMap[index] = false;
      m_dlRbgMap[index] = false;
    }

  for (uint8_t i = 0; i < m_dlReuse3SubBandwidth / rbgSize; i++)
    {
      uint8_t offset = (m_dlReuse3SubBandwidth + m_dlReuse1SubBandwidth) / rbgSize;

      uint8_t index = 0 * offset + i;
      m_dlSecondarySegmentRbgMap[index] = false;

      index = 1 * offset + i;
      m_dlSecondarySegmentRbgMap[index] = false;

      index = 2 * offset + i;
      m_dlSecondarySegmentRbgMap[index] = false;
    }
}


void
LteFfrEnhancedAlgorithm::InitializeUplinkRbgMaps ()
{
  m_ulRbgMap.clear ();
  m_ulReuse3RbgMap.clear ();
  m_ulReuse1RbgMap.clear ();
  m_ulPrimarySegmentRbgMap.clear ();
  m_ulSecondarySegmentRbgMap.clear ();

  if (!m_enabledInUplink)
    {
      m_ulRbgMap.resize (m_ulBandwidth, false);
      return;
    }

  m_ulRbgMap.resize (m_ulBandwidth, true);
  m_ulReuse3RbgMap.resize (m_ulBandwidth, false);
  m_ulReuse1RbgMap.resize (m_ulBandwidth, false);
  m_ulPrimarySegmentRbgMap.resize (m_ulBandwidth, false);
  m_ulSecondarySegmentRbgMap.resize (m_ulBandwidth, true);


  NS_ASSERT_MSG (m_ulSubBandOffset <= m_ulBandwidth, "UlSubBandOffset higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulSubBandOffset + m_ulReuse3SubBandwidth + m_ulReuse1SubBandwidth <= m_ulBandwidth,
                 "UlSubBandOffset + UlReuse3SubBandwidth + UlReuse1SubBandwidth higher than UlBandwidth");


  for (uint8_t i = 0; i < m_ulReuse3SubBandwidth; i++)
    {
      int offset = m_ulSubBandOffset;
      uint8_t index = offset + i;
      m_ulReuse3RbgMap[index] = true;
      m_ulPrimarySegmentRbgMap[index] = true;
      m_ulRbgMap[index] = false;
    }

  for (uint8_t i = 0; i < m_ulReuse1SubBandwidth; i++)
    {
      int offset = (m_ulSubBandOffset + m_ulReuse3SubBandwidth);
      uint8_t index = offset + i;
      m_ulReuse1RbgMap[index] = true;
      m_ulPrimarySegmentRbgMap[index] = true;
      m_ulSecondarySegmentRbgMap[index] = false;
      m_ulRbgMap[index] = false;
    }

  for (uint8_t i = 0; i < m_ulReuse3SubBandwidth; i++)
    {
      uint8_t offset = m_ulReuse3SubBandwidth + m_ulReuse1SubBandwidth;

      uint8_t index = 0 * offset + i;
      m_ulSecondarySegmentRbgMap[index] = false;

      index = 1 * offset + i;
      m_ulSecondarySegmentRbgMap[index] = false;

      index = 2 * offset + i;
      m_ulSecondarySegmentRbgMap[index] = false;

    }
}

std::vector <bool>
LteFfrEnhancedAlgorithm::DoGetAvailableDlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_needReconfiguration)
    {
      Reconfigure ();
    }

  if (m_dlRbgMap.empty ())
    {
      InitializeDownlinkRbgMaps ();
    }

  std::vector <bool> rbgMap = m_dlRbgMap;

  std::map <uint16_t, std::vector<bool> >::iterator it;
  for (it = m_dlRbgAvailableforUe.begin (); it != m_dlRbgAvailableforUe.end (); it++)
    {
      NS_LOG_INFO ("RNTI : " << it->first);
      std::vector<bool> rbgAvailableMap = it->second;
      for (uint32_t i = 0; i < rbgMap.size (); i++)
        {
          NS_LOG_INFO ("\t rbgId: " << i << " available " << (int)rbgAvailableMap.at (i));
          if ( rbgAvailableMap.at (i) == true)
            {
              rbgMap.at (i) = false;
            }
        }
    }

  return rbgMap;
}

bool
LteFfrEnhancedAlgorithm::DoIsDlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  bool isReuse3Rbg = m_dlReuse3RbgMap[rbgId];
  bool isReuse1Rbg = m_dlReuse1RbgMap[rbgId];
  bool isPrimarySegmentRbg     = m_dlPrimarySegmentRbgMap[rbgId];
  bool isSecondarySegmentRbg   = m_dlSecondarySegmentRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
    }

  it = m_ues.find (rnti);

  //if UE area is unknown, serve UE in edge area RBGs
  if (it->second == AreaUnset)
    {
      return isReuse3Rbg;
    }


  bool isCenterUe = false;
  bool isEdgeUe   = false;

  if (it->second == CenterArea )
    {
      isCenterUe = true;
    }
  else if (it->second == EdgeArea)
    {
      isEdgeUe = true;
    }

  if (isPrimarySegmentRbg)
    {
      NS_LOG_INFO ("PRIMARY SEGMENT RNTI: " << rnti  << "  rbgId: " << rbgId );
      return (isReuse1Rbg && isCenterUe) || (isReuse3Rbg && isEdgeUe);
    }
  else if (isSecondarySegmentRbg && isCenterUe)
    {
      //check if RB can be used by UE based on CQI information
      NS_LOG_INFO ("SECONDARY SEGMENT RNTI: " << rnti  << "  rbgId: " << rbgId );
      std::map <uint16_t, std::vector<bool> >::iterator it = m_dlRbgAvailableforUe.find (rnti);
      if (it != m_dlRbgAvailableforUe.end ())
        {
          NS_LOG_INFO ("RNTI: " << rnti  << "  rbgId: " << rbgId << "  available: " << it->second.at (rbgId));
          if (it->second.at (rbgId) == true)
            {
              return true;
            }
        }
      return false;
    }

  return false;
}

std::vector <bool>
LteFfrEnhancedAlgorithm::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      InitializeUplinkRbgMaps ();
    }

  if (!m_enabledInUplink)
    {
      return m_ulRbgMap;
    }

  std::vector <bool> rbgMap = m_ulRbgMap;

  std::map <uint16_t, std::vector<bool> >::iterator it;
  for (it = m_ulRbAvailableforUe.begin (); it != m_ulRbAvailableforUe.end (); it++)
    {
      NS_LOG_INFO ("RNTI : " << it->first);
      std::vector<bool> rbAvailableMap = it->second;
      for (uint32_t i = 0; i < rbgMap.size (); i++)
        {
          NS_LOG_INFO ("\t rbgId: " << i << " available " << (int)rbAvailableMap.at (i));
          if ( rbAvailableMap.at (i) == true)
            {
              rbgMap.at (i) = false;
            }
        }
    }

  return rbgMap;
}

bool
LteFfrEnhancedAlgorithm::DoIsUlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return true;
    }

  bool isReuse3Rbg = m_ulReuse3RbgMap[rbgId];
  bool isReuse1Rbg = m_ulReuse1RbgMap[rbgId];
  bool isPrimarySegmentRbg     = m_ulPrimarySegmentRbgMap[rbgId];
  bool isSecondarySegmentRbg   = m_ulSecondarySegmentRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
    }

  it = m_ues.find (rnti);

  //if UE area is unknown, serve UE in edge area RBGs
  if (it->second == AreaUnset)
    {
      return isReuse3Rbg;
    }

  bool isCenterUe = false;
  bool isEdgeUe   = false;

  if (it->second == CenterArea )
    {
      isCenterUe = true;
    }
  else if (it->second == EdgeArea)
    {
      isEdgeUe = true;
    }

  if (isPrimarySegmentRbg)
    {
      return (isReuse1Rbg && isCenterUe) || (isReuse3Rbg && isEdgeUe);
    }
  else if (isSecondarySegmentRbg && isCenterUe)
    {
      //check if RB can be used by UE based on CQI information
      NS_LOG_INFO ("UL SECONDARY SEGMENT RNTI: " << rnti  << "  rbgId: " << rbgId );
      std::map <uint16_t, std::vector<bool> >::iterator it = m_ulRbAvailableforUe.find (rnti);
      if (it != m_ulRbAvailableforUe.end ())
        {
          NS_LOG_INFO ("RNTI: " << rnti  << "  rbgId: " << rbgId << "  available: " << it->second.at (rbgId));
          if (it->second.at (rbgId) == true)
            {
              return true;
            }
        }
      return false;
    }

  return false;
}

void
LteFfrEnhancedAlgorithm::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);

  m_dlCqi.clear ();
  for (unsigned int i = 0; i < params.m_cqiList.size (); i++)
    {
      if ( params.m_cqiList.at (i).m_cqiType == CqiListElement_s::A30 )
        {
          NS_LOG_INFO ("subband CQI reporting high layer configured");
          // subband CQI reporting high layer configured
          std::map <uint16_t,SbMeasResult_s>::iterator it;
          uint16_t rnti = params.m_cqiList.at (i).m_rnti;

          std::map< uint16_t, uint8_t >::iterator ueIt = m_ues.find (rnti);
          if (ueIt != m_ues.end ())
            {
              if (ueIt->second != CenterArea )
                {
                  continue;
                }
            }
          else
            {
              continue;
            }

          it = m_dlCqi.find (rnti);
          if (it == m_dlCqi.end ())
            {
              // create the new entry
              m_dlCqi.insert ( std::pair<uint16_t, SbMeasResult_s > (rnti, params.m_cqiList.at (i).m_sbMeasResult) );
            }
          else
            {
              // update the CQI value and refresh correspondent timer
              (*it).second = params.m_cqiList.at (i).m_sbMeasResult;
            }
        }
      else
        {
          NS_LOG_ERROR (this << " CQI type unknown");
        }
    }

  uint32_t rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgAvailableforUe.clear ();
  std::map <uint16_t,SbMeasResult_s>::iterator it;
  for (it = m_dlCqi.begin (); it != m_dlCqi.end (); it++)
    {
      uint16_t rnti = it->first;
      std::vector<bool> rbgAvailableMap;

      for (uint32_t i = 0; i < (*it).second.m_higherLayerSelected.size (); i++)
        {
          uint8_t rbgCqi = (*it).second.m_higherLayerSelected.at (i).m_sbCqi.at (0);

          if (i > m_dlBandwidth / rbgSize)
            {
              continue;
            }
          NS_LOG_INFO (this << " RNTI " << rnti << " RBG  " << i << " DL-CQI: " << (int)rbgCqi);

          bool rbgAvailable = (rbgCqi > m_dlCqiThreshold);

          bool isSecondarySegmentRbg = false;
          if (i < m_dlSecondarySegmentRbgMap.size ())
            {
              isSecondarySegmentRbg = m_dlSecondarySegmentRbgMap[i];
            }

          rbgAvailable = (isSecondarySegmentRbg == true) ? rbgAvailable : false;

          rbgAvailableMap.push_back (rbgAvailable);
        }

      m_dlRbgAvailableforUe.insert ( std::pair<uint16_t, std::vector<bool> > (rnti, rbgAvailableMap ) );
    }

  m_ulRbAvailableforUe.clear ();
  for (std::map<uint16_t, std::vector<bool> >::iterator it = m_dlRbgAvailableforUe.begin ();
       it != m_dlRbgAvailableforUe.end (); it++)
    {
      uint16_t rnti = it->first;
      std::vector<bool> dlRbgAvailableMap = it->second;
      std::vector<bool> ulRbAvailableMap;
      ulRbAvailableMap.resize (m_ulBandwidth, false);

      for (uint32_t j = 0; j < dlRbgAvailableMap.size (); j++)
        {
          uint32_t index = rbgSize * j;
          for (uint32_t i = 0; i < rbgSize; i++)
            {
              index = index + i;
              ulRbAvailableMap[index] = dlRbgAvailableMap[j];
            }
        }

      m_ulRbAvailableforUe.insert ( std::pair<uint16_t, std::vector<bool> > (rnti, ulRbAvailableMap ) );
    }

  return;
}

void
LteFfrEnhancedAlgorithm::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  if (params.m_ulCqi.m_type == UlCqi_s::SRS)
    {
      // get the RNTI from vendor specific parameters
      uint16_t rnti = 0;
      for (uint32_t j = 0; j < m_ulBandwidth; j++)
        {
          double sinr = LteFfConverter::fpS11dot3toDouble (params.m_ulCqi.m_sinr.at (j));
          double s = log2 ( 1 + (
                              std::pow (10, sinr / 10 )  /
                              ( (-std::log (5.0 * 0.00005 )) / 1.5) ));
          int cqi = GetCqiFromSpectralEfficiency (s);
          NS_LOG_INFO (this << " RNTI " << rnti << " new SRS-CQI for RB  " << j << " value " << sinr << " UL-CQI: " << cqi);
        }
    }
}

void
LteFfrEnhancedAlgorithm::DoReportUlCqiInfo ( std::map <uint16_t, std::vector <double> > ulCqiMap )
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

double
LteFfrEnhancedAlgorithm::EstimateUlSinr (uint16_t rnti, uint16_t rb, std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  std::map <uint16_t, std::vector <double> >::iterator itCqi = ulCqiMap.find (rnti);
  if (itCqi == ulCqiMap.end ())
    {
      // no cqi info about this UE
      return (NO_SINR);
    }
  else
    {
      // take the average SINR value among the available
      double sinrSum = 0;
      unsigned int sinrNum = 0;
      for (uint32_t i = 0; i < m_ulBandwidth; i++)
        {
          double sinr = (*itCqi).second.at (i);
          if (sinr != NO_SINR)
            {
              sinrSum += sinr;
              sinrNum++;
            }
        }
      double estimatedSinr = (sinrNum > 0) ? (sinrSum / sinrNum) : DBL_MAX;
      // store the value
      (*itCqi).second.at (rb) = estimatedSinr;
      return (estimatedSinr);
    }
}

uint8_t
LteFfrEnhancedAlgorithm::DoGetTpc (uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return 1;     // 1 is mapped to 0 for Accumulated mode, and to -1 in Absolute mode TS36.213 Table 5.1.1.1-2
    }

  //TS36.213 Table 5.1.1.1-2
  //   TPC   |   Accumulated Mode  |  Absolute Mode
  //------------------------------------------------
  //    0    |         -1          |      -4
  //    1    |          0          |      -1
  //    2    |          1          |       1
  //    3    |          3          |       4
  //------------------------------------------------
  // here Absolute mode is used

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      return 1;
    }

  if (it->second == EdgeArea )
    {
      return m_edgeAreaTpc;
    }
  else
    {
      return m_centerAreaTpc;
    }

  return 1;
}

uint16_t
LteFfrEnhancedAlgorithm::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);

  uint8_t minContinuousUlBandwidth = m_ulBandwidth;

  if (!m_enabledInUplink)
    {
      return minContinuousUlBandwidth;
    }

  minContinuousUlBandwidth =
    ((m_ulReuse3SubBandwidth > 0 ) && (m_ulReuse3SubBandwidth < minContinuousUlBandwidth)) ? m_ulReuse3SubBandwidth : minContinuousUlBandwidth;

  minContinuousUlBandwidth =
    ((m_ulReuse1SubBandwidth > 0 ) && (m_ulReuse1SubBandwidth < minContinuousUlBandwidth)) ? m_ulReuse1SubBandwidth : minContinuousUlBandwidth;

  NS_LOG_INFO ("minContinuousUlBandwidth: " << (int)minContinuousUlBandwidth);

  return minContinuousUlBandwidth;
}


void
LteFfrEnhancedAlgorithm::DoReportUeMeas (uint16_t rnti,
                                         LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);
  NS_LOG_INFO ("RNTI :" << rnti << " MeasId: " << (uint16_t) measResults.measId
                        << " RSRP: " << (uint16_t)measResults.measResultPCell.rsrpResult
                        << " RSRQ: " << (uint16_t)measResults.measResultPCell.rsrqResult);

  if (measResults.measId != m_measId)
    {
      NS_LOG_WARN ("Ignoring measId " << (uint16_t) measResults.measId);
    }
  else
    {
      std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
      if (it == m_ues.end ())
        {
          m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
        }

      it = m_ues.find (rnti);
      if (measResults.measResultPCell.rsrqResult < m_rsrqThreshold)
        {
          if (it->second != EdgeArea)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Edge sub-band");
              it->second = EdgeArea;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_edgeAreaPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
      else
        {
          if (it->second != CenterArea)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Center sub-band");
              it->second = CenterArea;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_centerAreaPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
    }
}
void
LteFfrEnhancedAlgorithm::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

} // end of namespace ns3
