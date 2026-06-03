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

#include "lte-fr-soft-algorithm.h"
#include <ns3/log.h>
#include "ns3/boolean.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFrSoftAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (LteFrSoftAlgorithm);

/// FrSoftDownlinkDefaultConfiguration structure
static const struct FrSoftDownlinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t dlBandwidth; ///< DL bandwidth
  uint8_t dlEdgeSubBandOffset; ///< DL edge subband offset
  uint8_t dlEdgeSubBandwidth; ///< Dl edge subbandwidth
} g_frSoftDownlinkDefaultConfiguration[] = {
  { 1, 15, 0, 4},
  { 2, 15, 4, 4},
  { 3, 15, 8, 6},
  { 1, 25, 0, 8},
  { 2, 25, 8, 8},
  { 3, 25, 16, 9},
  { 1, 50, 0, 16},
  { 2, 50, 16, 16},
  { 3, 50, 32, 18},
  { 1, 75, 0, 24},
  { 2, 75, 24, 24},
  { 3, 75, 48, 27},
  { 1, 100, 0, 32},
  { 2, 100, 32, 32},
  { 3, 100, 64, 36}
}; ///< soft downlink default configuration

/// soft uplink default configuration
static const struct FrSoftUplinkDefaultConfiguration
{
  uint8_t cellId; ///< cell ID
  uint8_t ulBandwidth; ///< UL bandwidth
  uint8_t ulEdgeSubBandOffset; ///< UL edge subband offset  
  uint8_t ulEdgeSubBandwidth; ///< UL endge subbandwidth
} g_frSoftUplinkDefaultConfiguration[] = {
  { 1, 15, 0, 5},
  { 2, 15, 5, 5},
  { 3, 15, 10, 5},
  { 1, 25, 0, 8},
  { 2, 25, 8, 8},
  { 3, 25, 16, 9},
  { 1, 50, 0, 16},
  { 2, 50, 16, 16},
  { 3, 50, 32, 18},
  { 1, 75, 0, 24},
  { 2, 75, 24, 24},
  { 3, 75, 48, 27},
  { 1, 100, 0, 32},
  { 2, 100, 32, 32},
  { 3, 100, 64, 36}
}; ///< soft uplink default configuration

/** \returns number of downlink configurations */
const uint16_t NUM_DOWNLINK_CONFS (sizeof (g_frSoftDownlinkDefaultConfiguration) / sizeof (FrSoftDownlinkDefaultConfiguration));
/** \returns number of uplink configurations */
const uint16_t NUM_UPLINK_CONFS (sizeof (g_frSoftUplinkDefaultConfiguration) / sizeof (FrSoftUplinkDefaultConfiguration));


LteFrSoftAlgorithm::LteFrSoftAlgorithm ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0),
    m_dlEdgeSubBandOffset (0),
    m_dlEdgeSubBandwidth (0),
    m_ulEdgeSubBandOffset (0),
    m_ulEdgeSubBandwidth (0),
    m_measId (0)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFrSoftAlgorithm> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFrSoftAlgorithm> (this);
}


LteFrSoftAlgorithm::~LteFrSoftAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFrSoftAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFrSoftAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFrSoftAlgorithm")
    .SetParent<LteFfrAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<LteFrSoftAlgorithm> ()
    .AddAttribute ("UlEdgeSubBandOffset",
                   "Uplink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_ulEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlEdgeSubBandwidth",
                   "Uplink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_ulEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandOffset",
                   "Downlink Edge SubBand Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_dlEdgeSubBandOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlEdgeSubBandwidth",
                   "Downlink Edge SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_dlEdgeSubBandwidth),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("AllowCenterUeUseEdgeSubBand",
                   "If true center UEs can receive on Edge SubBand RBGs",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteFrSoftAlgorithm::m_isEdgeSubBandForCenterUe),
                   MakeBooleanChecker ())
    .AddAttribute ("RsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Edge sub-band",
                   UintegerValue (20),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_edgeSubBandThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterPowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_centerPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgePowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_edgePowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in center area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_centerAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in edge area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFrSoftAlgorithm::m_edgeAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
  ;
  return tid;
}


void
LteFrSoftAlgorithm::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFrSoftAlgorithm::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFrSoftAlgorithm::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFrSoftAlgorithm::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFrSoftAlgorithm::DoInitialize ()
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
LteFrSoftAlgorithm::Reconfigure ()
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
LteFrSoftAlgorithm::SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_DOWNLINK_CONFS; ++i)
    {
      if ((g_frSoftDownlinkDefaultConfiguration[i].cellId == cellId)
          && g_frSoftDownlinkDefaultConfiguration[i].dlBandwidth == m_dlBandwidth)
        {
          m_dlEdgeSubBandOffset = g_frSoftDownlinkDefaultConfiguration[i].dlEdgeSubBandOffset;
          m_dlEdgeSubBandwidth = g_frSoftDownlinkDefaultConfiguration[i].dlEdgeSubBandwidth;
        }
    }
}

void
LteFrSoftAlgorithm::SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
  for (uint16_t i = 0; i < NUM_UPLINK_CONFS; ++i)
    {
      if ((g_frSoftUplinkDefaultConfiguration[i].cellId == cellId)
          && g_frSoftUplinkDefaultConfiguration[i].ulBandwidth == m_ulBandwidth)
        {
          m_ulEdgeSubBandOffset = g_frSoftUplinkDefaultConfiguration[i].ulEdgeSubBandOffset;
          m_ulEdgeSubBandwidth = g_frSoftUplinkDefaultConfiguration[i].ulEdgeSubBandwidth;
        }
    }
}

void
LteFrSoftAlgorithm::InitializeDownlinkRbgMaps ()
{
  m_dlRbgMap.clear ();
  m_dlEdgeRbgMap.clear ();

  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_dlEdgeRbgMap.resize (m_dlBandwidth / rbgSize, false);

  NS_ASSERT_MSG (m_dlEdgeSubBandOffset <= m_dlBandwidth,"DlEdgeSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_dlEdgeSubBandwidth <= m_dlBandwidth,"DlEdgeSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG ((m_dlEdgeSubBandOffset + m_dlEdgeSubBandwidth) <= m_dlBandwidth,
                 "(DlEdgeSubBandOffset+DlEdgeSubBandwidth) higher than DlBandwidth");

  for (uint8_t i = m_dlEdgeSubBandOffset / rbgSize;
       i < (m_dlEdgeSubBandOffset + m_dlEdgeSubBandwidth) / rbgSize; i++)
    {
      m_dlEdgeRbgMap[i] = true;
    }
}

void
LteFrSoftAlgorithm::InitializeUplinkRbgMaps ()
{
  m_ulRbgMap.clear ();
  m_ulEdgeRbgMap.clear ();

  m_ulRbgMap.resize (m_ulBandwidth, false);
  m_ulEdgeRbgMap.resize (m_ulBandwidth, false);

  NS_ASSERT_MSG (m_ulEdgeSubBandOffset <= m_dlBandwidth,"UlEdgeSubBandOffset higher than DlBandwidth");
  NS_ASSERT_MSG (m_ulEdgeSubBandwidth <= m_dlBandwidth,"UlEdgeSubBandwidth higher than DlBandwidth");
  NS_ASSERT_MSG ((m_ulEdgeSubBandOffset + m_ulEdgeSubBandwidth) <= m_dlBandwidth,
                 "(UlEdgeSubBandOffset+UlEdgeSubBandwidth) higher than DlBandwidth");

  for (uint8_t i = m_ulEdgeSubBandOffset; i < (m_ulEdgeSubBandOffset + m_ulEdgeSubBandwidth); i++)
    {
      m_ulEdgeRbgMap[i] = true;
    }
}

std::vector <bool>
LteFrSoftAlgorithm::DoGetAvailableDlRbg ()
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
LteFrSoftAlgorithm::DoIsDlRbgAvailableForUe (int rbgId, uint16_t rnti)
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

  if (!edgeUe && m_isEdgeSubBandForCenterUe)
    {
      return true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

std::vector <bool>
LteFrSoftAlgorithm::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      InitializeUplinkRbgMaps ();
    }

  return m_ulRbgMap;
}

bool
LteFrSoftAlgorithm::DoIsUlRbgAvailableForUe (int rbgId, uint16_t rnti)
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

  if (!edgeUe && m_isEdgeSubBandForCenterUe)
    {
      return true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

void
LteFrSoftAlgorithm::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFrSoftAlgorithm::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFrSoftAlgorithm::DoReportUlCqiInfo (std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

uint8_t
LteFrSoftAlgorithm::DoGetTpc (uint16_t rnti)
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
LteFrSoftAlgorithm::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);

  uint8_t minContinuousUlBandwidth = m_ulBandwidth;

  if (!m_enabledInUplink)
    {
      return minContinuousUlBandwidth;
    }

  uint8_t leftBandwidth = m_ulEdgeSubBandOffset;
  uint8_t centerBandwidth = m_ulEdgeSubBandwidth;
  uint8_t rightBandwidth = m_ulBandwidth - m_ulEdgeSubBandwidth - m_ulEdgeSubBandOffset;

  minContinuousUlBandwidth =
    ((leftBandwidth > 0 ) && (leftBandwidth < minContinuousUlBandwidth)) ? leftBandwidth : minContinuousUlBandwidth;

  minContinuousUlBandwidth =
    ((centerBandwidth > 0 ) && (centerBandwidth < minContinuousUlBandwidth)) ? centerBandwidth : minContinuousUlBandwidth;

  minContinuousUlBandwidth =
    ((rightBandwidth > 0 ) && (rightBandwidth < minContinuousUlBandwidth)) ? rightBandwidth : minContinuousUlBandwidth;

  NS_LOG_INFO ("minContinuousUlBandwidth: " << (int)minContinuousUlBandwidth);

  return minContinuousUlBandwidth;
}

void
LteFrSoftAlgorithm::DoReportUeMeas (uint16_t rnti,
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
          if (it->second != CellEdge)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Edge sub-band");
              it->second = CellEdge;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_edgePowerOffset;
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
              pdschConfigDedicated.pa = m_centerPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
    }
}

void
LteFrSoftAlgorithm::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

} // end of namespace ns3
