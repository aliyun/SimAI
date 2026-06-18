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

#include "lte-ffr-soft-algorithm.h"
#include <ns3/log.h>
#include "ns3/boolean.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFfrSoftAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (LteFfrSoftAlgorithm);

/// FfrSoftDownlinkDefaultConfiguration structure
static const struct FfrSoftDownlinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t dlBandwidth; ///< DL bandwidth
  uint8_t dlCommonSubBandwidth; ///< DL common subbandwidth
  uint8_t dlEdgeSubBandOffset; ///< DL edge subband offset
  uint8_t dlEdgeSubBandwidth; ///< DL edge subbandwidth
} g_ffrSoftDownlinkDefaultConfiguration[] = {
  { 1, 15, 2, 0, 4},
  { 2, 15, 2, 4, 4},
  { 3, 15, 2, 8, 4},
  { 1, 25, 6, 0, 6},
  { 2, 25, 6, 6, 6},
  { 3, 25, 6, 12, 6},
  { 1, 50, 21, 0, 9},
  { 2, 50, 21, 9, 9},
  { 3, 50, 21, 18, 11},
  { 1, 75, 36, 0, 12},
  { 2, 75, 36, 12, 12},
  { 3, 75, 36, 24, 15},
  { 1, 100, 28, 0, 24},
  { 2, 100, 28, 24, 24},
  { 3, 100, 28, 48, 24}
}; ///< the soft downlink default configuration

/// FfrSoftUplinkDefaultConfiguration structure
static const struct FfrSoftUplinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t ulBandwidth; ///< UL bandwidth
  uint8_t ulCommonSubBandwidth; ///< UL common subbandwidth
  uint8_t ulEdgeSubBandOffset; ///< UL edge subband offset
  uint8_t ulEdgeSubBandwidth; ///< edge subbandwidth
} g_ffrSoftUplinkDefaultConfiguration[] = {
  { 1, 15, 3, 0, 4},
  { 2, 15, 3, 4, 4},
  { 3, 15, 3, 8, 4},
  { 1, 25, 6, 0, 6},
  { 2, 25, 6, 6, 6},
  { 3, 25, 6, 12, 6},
  { 1, 50, 21, 0, 9},
  { 2, 50, 21, 9, 9},
  { 3, 50, 21, 18, 11},
  { 1, 75, 36, 0, 12},
  { 2, 75, 36, 12, 12},
  { 3, 75, 36, 24, 15},
  { 1, 100, 28, 0, 24},
  { 2, 100, 28, 24, 24},
  { 3, 100, 28, 48, 24}
}; ///< the soft uplink default configuration

/** \returns number of downlink configurations */
const uint16_t NUM_DOWNLINK_CONFS (sizeof (g_ffrSoftDownlinkDefaultConfiguration) / sizeof (FfrSoftDownlinkDefaultConfiguration));
/** \returns number of uplink configurations */
const uint16_t NUM_UPLINK_CONFS (sizeof (g_ffrSoftUplinkDefaultConfiguration) / sizeof (FfrSoftUplinkDefaultConfiguration));


LteFfrSoftAlgorithm::LteFfrSoftAlgorithm ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0),
    m_dlEdgeSubBandOffset (0),
    m_dlEdgeSubBandwidth (0),
    m_ulEdgeSubBandOffset (0),
    m_ulEdgeSubBandwidth (0),
    m_measId (0)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFfrSoftAlgorithm> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFfrSoftAlgorithm> (this);
}


LteFfrSoftAlgorithm::~LteFfrSoftAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFfrSoftAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFfrSoftAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFfrSoftAlgorithm")
    .SetParent<LteFfrAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<LteFfrSoftAlgorithm> ()
    .AddAttribute ("UlCommonSubBandwidth",
                   "Uplink Medium (Common) SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (6),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_ulCommonSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlEdgeSubBandOffset",
                   "Uplink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_ulEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlEdgeSubBandwidth",
                   "Uplink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (6),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_ulEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlCommonSubBandwidth",
                   "Downlink Medium (Common) SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (6),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_dlCommonSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandOffset",
                   "Downlink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_dlEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandwidth",
                   "Downlink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_dlEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterRsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Medium sub-band",
                   UintegerValue (30),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_centerSubBandThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeRsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Edge sub-band",
                   UintegerValue (20),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_edgeSubBandThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaPowerOffset",
                   "PdschConfigDedicated::Pa value for Center Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_centerAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("MediumAreaPowerOffset",
                   "PdschConfigDedicated::Pa value for Medium Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_mediumAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaPowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_edgeAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in center area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_centerAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("MediumAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in medium area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_mediumAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in edge area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrSoftAlgorithm::m_edgeAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
  ;
  return tid;
}


void
LteFfrSoftAlgorithm::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFfrSoftAlgorithm::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFfrSoftAlgorithm::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFfrSoftAlgorithm::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFfrSoftAlgorithm::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteFfrAlgorithm::DoInitialize ();

  NS_ASSERT_MSG (m_dlBandwidth > 14,"DlBandwidth must be at least 15 to use FFR algorithms");
  NS_ASSERT_MSG (m_ulBandwidth > 14,"UlBandwidth must be at least 15 to use FFR algorithms");

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
LteFfrSoftAlgorithm::Reconfigure ()
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
LteFfrSoftAlgorithm::SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_DOWNLINK_CONFS; ++i)
    {
      if ((g_ffrSoftDownlinkDefaultConfiguration[i].cellId == cellId)
          && g_ffrSoftDownlinkDefaultConfiguration[i].dlBandwidth == m_dlBandwidth)
        {
          m_dlCommonSubBandwidth = g_ffrSoftDownlinkDefaultConfiguration[i].dlCommonSubBandwidth;
          m_dlEdgeSubBandOffset = g_ffrSoftDownlinkDefaultConfiguration[i].dlEdgeSubBandOffset;
          m_dlEdgeSubBandwidth = g_ffrSoftDownlinkDefaultConfiguration[i].dlEdgeSubBandwidth;
        }
    }
}

void
LteFfrSoftAlgorithm::SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_UPLINK_CONFS; ++i)
    {
      if ((g_ffrSoftUplinkDefaultConfiguration[i].cellId == cellId)
          && g_ffrSoftUplinkDefaultConfiguration[i].ulBandwidth == m_ulBandwidth)
        {
          m_ulCommonSubBandwidth = g_ffrSoftUplinkDefaultConfiguration[i].ulCommonSubBandwidth;
          m_ulEdgeSubBandOffset = g_ffrSoftUplinkDefaultConfiguration[i].ulEdgeSubBandOffset;
          m_ulEdgeSubBandwidth = g_ffrSoftUplinkDefaultConfiguration[i].ulEdgeSubBandwidth;
        }
    }
}

void
LteFfrSoftAlgorithm::InitializeDownlinkRbgMaps ()
{
  m_dlRbgMap.clear ();
  m_dlCenterRbgMap.clear ();
  m_dlMediumRbgMap.clear ();
  m_dlEdgeRbgMap.clear ();

  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlCenterRbgMap.resize (m_dlBandwidth / rbgSize, true);
  m_dlMediumRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlEdgeRbgMap.resize (m_dlBandwidth / rbgSize, false);

  NS_ASSERT_MSG (m_dlCommonSubBandwidth <= m_dlBandwidth,"DlCommonSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlCommonSubBandwidth + m_dlEdgeSubBandOffset <= m_dlBandwidth,
                 "DlCommonSubBandwidth + DlEdgeSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlEdgeSubBandOffset <= m_dlBandwidth,"DlEdgeSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlEdgeSubBandwidth <= m_dlBandwidth,"DlEdgeSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG ((m_dlCommonSubBandwidth + m_dlEdgeSubBandOffset + m_dlEdgeSubBandwidth) <= m_dlBandwidth,
                 "(DlCommonSubBandwidth + DlEdgeSubBandOffset+DlEdgeSubBandwidth) higher than DlBandwidth");

  for (uint8_t i = 0;
       i < m_dlCommonSubBandwidth / rbgSize; i++)
    {
      m_dlMediumRbgMap[i] = true;
      m_dlCenterRbgMap[i] = false;
    }

  for (uint8_t i = (m_dlCommonSubBandwidth + m_dlEdgeSubBandOffset) / rbgSize;
       i < (m_dlCommonSubBandwidth + m_dlEdgeSubBandOffset + m_dlEdgeSubBandwidth) / rbgSize; i++)
    {
      m_dlEdgeRbgMap[i] = true;
      m_dlCenterRbgMap[i] = false;
    }
}


void
LteFfrSoftAlgorithm::InitializeUplinkRbgMaps ()
{
  m_ulRbgMap.clear ();
  m_ulCenterRbgMap.clear ();
  m_ulMediumRbgMap.clear ();
  m_ulEdgeRbgMap.clear ();

  m_ulRbgMap.resize (m_ulBandwidth, false);
  m_ulCenterRbgMap.resize (m_ulBandwidth, true);
  m_ulMediumRbgMap.resize (m_ulBandwidth, false);
  m_ulEdgeRbgMap.resize (m_ulBandwidth, false);

  NS_ASSERT_MSG (m_ulCommonSubBandwidth <= m_ulBandwidth,"UlCommonSubBandwidth higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset <= m_ulBandwidth,
                 "UlCommonSubBandwidth + UlEdgeSubBandOffset higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulEdgeSubBandOffset <= m_ulBandwidth,"UlEdgeSubBandOffset higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulEdgeSubBandwidth <= m_ulBandwidth,"UlEdgeSubBandwidth higher than UlBandwidth");
  NS_ASSERT_MSG ((m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset + m_ulEdgeSubBandwidth) <= m_ulBandwidth,
                 "(UlCommonSubBandwidth + UlEdgeSubBandOffset+UlEdgeSubBandwidth) higher than UlBandwidth");

  for (uint8_t i = 0;
       i < m_ulCommonSubBandwidth; i++)
    {
      m_ulMediumRbgMap[i] = true;
      m_ulCenterRbgMap[i] = false;
    }

  for (uint8_t i = (m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset);
       i < (m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset + m_ulEdgeSubBandwidth); i++)
    {
      m_ulEdgeRbgMap[i] = true;
      m_ulCenterRbgMap[i] = false;
    }
}

std::vector <bool>
LteFfrSoftAlgorithm::DoGetAvailableDlRbg ()
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

  return m_dlRbgMap;
}

bool
LteFfrSoftAlgorithm::DoIsDlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  bool isCenterRbg = m_dlCenterRbgMap[rbgId];
  bool isMediumRbg = m_dlMediumRbgMap[rbgId];
  bool isEdgeRbg   = m_dlEdgeRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
    }

  it = m_ues.find (rnti);

  //if UE area is unknown, serve UE in medium (common) RBGs
  if (it->second == AreaUnset)
    {
      return isMediumRbg;
    }


  bool isCenterUe = false;
  bool isMediumUe = false;
  bool isEdgeUe   = false;

  if (it->second == CenterArea )
    {
      isCenterUe = true;
    }
  else if (it->second == MediumArea)
    {
      isMediumUe = true;
    }
  else if (it->second == EdgeArea)
    {
      isEdgeUe = true;
    }

  return (isCenterRbg && isCenterUe) || (isMediumRbg && isMediumUe) || (isEdgeRbg && isEdgeUe);
}

std::vector <bool>
LteFfrSoftAlgorithm::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      InitializeUplinkRbgMaps ();
    }

  return m_ulRbgMap;
}

bool
LteFfrSoftAlgorithm::DoIsUlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return true;
    }

  bool isCenterRbg = m_ulCenterRbgMap[rbgId];
  bool isMediumRbg = m_ulMediumRbgMap[rbgId];
  bool isEdgeRbg   = m_ulEdgeRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
    }

  it = m_ues.find (rnti);

  //if UE area is unknown, serve UE in medium (common) RBGs
  if (it->second == AreaUnset)
    {
      return isMediumRbg;
    }


  bool isCenterUe = false;
  bool isMediumUe = false;
  bool isEdgeUe   = false;

  if (it->second == CenterArea )
    {
      isCenterUe = true;
    }
  else if (it->second == MediumArea)
    {
      isMediumUe = true;
    }
  else if (it->second == EdgeArea)
    {
      isEdgeUe = true;
    }

  return (isCenterRbg && isCenterUe) || (isMediumRbg && isMediumUe) || (isEdgeRbg && isEdgeUe);
}

void
LteFfrSoftAlgorithm::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFfrSoftAlgorithm::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFfrSoftAlgorithm::DoReportUlCqiInfo (std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

uint8_t
LteFfrSoftAlgorithm::DoGetTpc (uint16_t rnti)
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

  if (it->second == CenterArea )
    {
      return m_centerAreaTpc;
    }
  else if (it->second == MediumArea)
    {
      return m_mediumAreaTpc;
    }
  else if (it->second == EdgeArea)
    {
      return m_edgeAreaTpc;
    }

  return 1;
}

uint16_t
LteFfrSoftAlgorithm::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return m_ulBandwidth;
    }

  uint8_t centerSubBandwidth = 0;
  uint8_t mediumSubBandwidth = 0;
  uint8_t edgeSubBandwidth = 0;

  for (uint8_t i = 0; i < m_ulCenterRbgMap.size (); i++)
    {
      if ( m_ulCenterRbgMap[i] == true)
        {
          centerSubBandwidth++;
        }
    }

  for (uint8_t i = 0; i < m_ulMediumRbgMap.size (); i++)
    {
      if ( m_ulMediumRbgMap[i] == true)
        {
          mediumSubBandwidth++;
        }
    }

  for (uint8_t i = 0; i < m_ulEdgeRbgMap.size (); i++)
    {
      if ( m_ulEdgeRbgMap[i] == true)
        {
          edgeSubBandwidth++;
        }
    }

  uint8_t minContinuousUlBandwidth = m_ulBandwidth;

  minContinuousUlBandwidth =
    ((centerSubBandwidth > 0 ) && (centerSubBandwidth < minContinuousUlBandwidth)) ? centerSubBandwidth : minContinuousUlBandwidth;

  minContinuousUlBandwidth =
    ((mediumSubBandwidth > 0 ) && (mediumSubBandwidth < minContinuousUlBandwidth)) ? mediumSubBandwidth : minContinuousUlBandwidth;

  minContinuousUlBandwidth =
    ((edgeSubBandwidth > 0 ) && (edgeSubBandwidth < minContinuousUlBandwidth)) ? edgeSubBandwidth : minContinuousUlBandwidth;

  NS_LOG_INFO ("minContinuousUlBandwidth: " << (int)minContinuousUlBandwidth);

  return minContinuousUlBandwidth;
}

void
LteFfrSoftAlgorithm::DoReportUeMeas (uint16_t rnti,
                                     LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);
  NS_LOG_INFO ("RNTI :" << rnti << " MeasId: " << (uint16_t) measResults.measId
                        << " RSRP: " << (uint16_t)measResults.measResultPCell.rsrpResult
                        << " RSRQ: " << (uint16_t)measResults.measResultPCell.rsrqResult);

  NS_ASSERT_MSG (m_centerSubBandThreshold >= m_edgeSubBandThreshold,
                 "CenterSubBandThreshold must be higher than EdgeSubBandThreshold");

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
      if (measResults.measResultPCell.rsrqResult >= m_centerSubBandThreshold)
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
      else if (measResults.measResultPCell.rsrqResult < m_edgeSubBandThreshold)
        {
          if (it->second != EdgeArea )
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
          if (it->second != MediumArea)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Medium sub-band");
              it->second = MediumArea;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_mediumAreaPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
    }
}

void
LteFfrSoftAlgorithm::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

} // end of namespace ns3
