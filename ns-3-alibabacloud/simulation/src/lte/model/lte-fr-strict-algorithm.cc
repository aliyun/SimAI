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

#include "lte-fr-strict-algorithm.h"
#include <ns3/log.h>
#include "ns3/boolean.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFrStrictAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (LteFrStrictAlgorithm);

/// FrStrictDownlinkDefaultConfiguration structure
static const struct FrStrictDownlinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t dlBandwidth; ///< DL bandwidth
  uint8_t dlCommonSubBandwidth; ///< DL common subbandwidth
  uint8_t dlEdgeSubBandOffset; ///< DL edge subband offset
  uint8_t dlEdgeSubBandwidth; ///< DL edge subbandwidth
} g_frStrictDownlinkDefaultConfiguration[] = {
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
}; ///< the strict downlink default configuration

/// FrStrictUplinkDefaultConfiguration structure
static const struct FrStrictUplinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t ulBandwidth; ///< UL bandwidth
  uint8_t ulCommonSubBandwidth; ///< UL common subbandwidth
  uint8_t ulEdgeSubBandOffset; ///< UL edge subband offset
  uint8_t ulEdgeSubBandwidth; ///< UL edge subbandwidth
} g_frStrictUplinkDefaultConfiguration[] = {
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
}; ///< the strict uplink default configuration

/** \returns number of downlink configurations */
const uint16_t NUM_DOWNLINK_CONFS (sizeof (g_frStrictDownlinkDefaultConfiguration) / sizeof (FrStrictDownlinkDefaultConfiguration));
/** \returns number of uplink configurations */
const uint16_t NUM_UPLINK_CONFS (sizeof (g_frStrictUplinkDefaultConfiguration) / sizeof (FrStrictUplinkDefaultConfiguration));


LteFrStrictAlgorithm::LteFrStrictAlgorithm ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0),
    m_dlEdgeSubBandOffset (0),
    m_dlEdgeSubBandwidth (0),
    m_ulEdgeSubBandOffset (0),
    m_ulEdgeSubBandwidth (0),
    m_measId (0)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFrStrictAlgorithm> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFrStrictAlgorithm> (this);
}


LteFrStrictAlgorithm::~LteFrStrictAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFrStrictAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFrStrictAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFrStrictAlgorithm")
    .SetParent<LteFfrAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<LteFrStrictAlgorithm> ()
    .AddAttribute ("UlCommonSubBandwidth",
                   "Uplink Common SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (25),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_ulCommonSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlEdgeSubBandOffset",
                   "Uplink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_ulEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlEdgeSubBandwidth",
                   "Uplink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_ulEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlCommonSubBandwidth",
                   "Downlink Common SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (25),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_dlCommonSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandOffset",
                   "Downlink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_dlEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandwidth",
                   "Downlink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_dlEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Edge sub-band",
                   UintegerValue (20),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_edgeSubBandThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterPowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_centerAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgePowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_edgeAreaPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in center area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_centerAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in edge area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFrStrictAlgorithm::m_edgeAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
  ;
  return tid;
}


void
LteFrStrictAlgorithm::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFrStrictAlgorithm::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFrStrictAlgorithm::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFrStrictAlgorithm::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFrStrictAlgorithm::DoInitialize ()
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
LteFrStrictAlgorithm::Reconfigure ()
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
LteFrStrictAlgorithm::SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_DOWNLINK_CONFS; ++i)
    {
      if ((g_frStrictDownlinkDefaultConfiguration[i].cellId == cellId)
          && g_frStrictDownlinkDefaultConfiguration[i].dlBandwidth == m_dlBandwidth)
        {
          m_dlCommonSubBandwidth = g_frStrictDownlinkDefaultConfiguration[i].dlCommonSubBandwidth;
          m_dlEdgeSubBandOffset = g_frStrictDownlinkDefaultConfiguration[i].dlEdgeSubBandOffset;
          m_dlEdgeSubBandwidth = g_frStrictDownlinkDefaultConfiguration[i].dlEdgeSubBandwidth;
        }
    }
}

void
LteFrStrictAlgorithm::SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_UPLINK_CONFS; ++i)
    {
      if ((g_frStrictUplinkDefaultConfiguration[i].cellId == cellId)
          && g_frStrictUplinkDefaultConfiguration[i].ulBandwidth == m_ulBandwidth)
        {
          m_ulCommonSubBandwidth = g_frStrictUplinkDefaultConfiguration[i].ulCommonSubBandwidth;
          m_ulEdgeSubBandOffset = g_frStrictUplinkDefaultConfiguration[i].ulEdgeSubBandOffset;
          m_ulEdgeSubBandwidth = g_frStrictUplinkDefaultConfiguration[i].ulEdgeSubBandwidth;
        }
    }
}

void
LteFrStrictAlgorithm::InitializeDownlinkRbgMaps ()
{
  m_dlRbgMap.clear ();
  m_dlEdgeRbgMap.clear ();

  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgMap.resize (m_dlBandwidth / rbgSize, true);
  m_dlEdgeRbgMap.resize (m_dlBandwidth / rbgSize, false);

  NS_ASSERT_MSG (m_dlCommonSubBandwidth <= m_dlBandwidth,"DlCommonSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlEdgeSubBandOffset <= m_dlBandwidth,"DlEdgeSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlEdgeSubBandwidth <= m_dlBandwidth,"DlEdgeSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG ((m_dlCommonSubBandwidth + m_dlEdgeSubBandOffset + m_dlEdgeSubBandwidth) <= m_dlBandwidth,
                 "(DlCommonSubBandwidth+DlEdgeSubBandOffset+DlEdgeSubBandwidth) higher than DlBandwidth");

  for (uint8_t i = 0; i < m_dlCommonSubBandwidth / rbgSize; i++)
    {
      m_dlRbgMap[i] = false;

    }

  for (uint8_t i = m_dlCommonSubBandwidth / rbgSize + m_dlEdgeSubBandOffset / rbgSize;
       i < (m_dlCommonSubBandwidth / rbgSize + m_dlEdgeSubBandOffset / rbgSize
            + m_dlEdgeSubBandwidth / rbgSize); i++)
    {
      m_dlRbgMap[i] = false;
      m_dlEdgeRbgMap[i] = true;
    }
}

void
LteFrStrictAlgorithm::InitializeUplinkRbgMaps ()
{
  m_ulRbgMap.clear ();
  m_ulEdgeRbgMap.clear ();

  if (!m_enabledInUplink)
    {
      m_ulRbgMap.resize (m_ulBandwidth, false);
      return;
    }

  m_ulRbgMap.resize (m_ulBandwidth, true);
  m_ulEdgeRbgMap.resize (m_ulBandwidth, false);

  NS_ASSERT_MSG (m_ulCommonSubBandwidth <= m_ulBandwidth,"UlCommonSubBandwidth higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulEdgeSubBandOffset <= m_ulBandwidth,"UlEdgeSubBandOffset higher than UlBandwidth");
  NS_ASSERT_MSG (m_ulEdgeSubBandwidth <= m_ulBandwidth,"UlEdgeSubBandwidth higher than UlBandwidth");
  NS_ASSERT_MSG ((m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset + m_ulEdgeSubBandwidth) <= m_dlBandwidth,
                 "(UlCommonSubBandwidth+UlEdgeSubBandOffset+UlEdgeSubBandwidth) higher than UlBandwidth");

  for (uint8_t i = 0; i < m_ulCommonSubBandwidth; i++)
    {
      m_ulRbgMap[i] = false;

    }

  for (uint8_t i = m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset;
       i < (m_ulCommonSubBandwidth + m_ulEdgeSubBandOffset
            + m_ulEdgeSubBandwidth); i++)
    {
      m_ulRbgMap[i] = false;
      m_ulEdgeRbgMap[i] = true;
    }
}


std::vector <bool>
LteFrStrictAlgorithm::DoGetAvailableDlRbg ()
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
LteFrStrictAlgorithm::DoIsDlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  bool edgeRbg = m_dlEdgeRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
      return !edgeRbg;
    }

  bool edgeUe = false;
  if (it->second == CellEdge )
    {
      edgeUe = true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

std::vector <bool>
LteFrStrictAlgorithm::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      InitializeUplinkRbgMaps ();
    }

  return m_ulRbgMap;
}

bool
LteFrStrictAlgorithm::DoIsUlRbgAvailableForUe (int rbgId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return true;
    }

  bool edgeRbg = m_ulEdgeRbgMap[rbgId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
      return !edgeRbg;
    }

  bool edgeUe = false;
  if (it->second == CellEdge )
    {
      edgeUe = true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

void
LteFrStrictAlgorithm::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFrStrictAlgorithm::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFrStrictAlgorithm::DoReportUlCqiInfo (std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

uint8_t
LteFrStrictAlgorithm::DoGetTpc (uint16_t rnti)
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

  if (it->second == CellEdge )
    {
      return m_edgeAreaTpc;
    }
  else if (it->second == CellCenter )
    {
      return m_centerAreaTpc;
    }

  return 1;
}

uint16_t
LteFrStrictAlgorithm::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return m_ulBandwidth;
    }

  uint8_t minContinuousUlBandwidth = m_ulCommonSubBandwidth < m_ulEdgeSubBandwidth ? m_ulCommonSubBandwidth : m_ulEdgeSubBandwidth;
  NS_LOG_INFO ("minContinuousUlBandwidth: " << (int)minContinuousUlBandwidth);

  return minContinuousUlBandwidth;
}

void
LteFrStrictAlgorithm::DoReportUeMeas (uint16_t rnti,
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

      if (measResults.measResultPCell.rsrqResult < m_edgeSubBandThreshold)
        {
          if (it->second != CellEdge )
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Edge sub-band");
              it->second = CellEdge;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_edgeAreaPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
      else
        {
          if (it->second != CellCenter)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Center sub-band");
              it->second = CellCenter;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_centerAreaPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
    }
}

void
LteFrStrictAlgorithm::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, since it is empty");

}

} // end of namespace ns3
