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

#include "lte-ffr-distributed-algorithm.h"
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFfrDistributedAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (LteFfrDistributedAlgorithm);


LteFfrDistributedAlgorithm::LteFfrDistributedAlgorithm ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFfrDistributedAlgorithm> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFfrDistributedAlgorithm> (this);
}


LteFfrDistributedAlgorithm::~LteFfrDistributedAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFfrDistributedAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFfrDistributedAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFfrDistributedAlgorithm")
    .SetParent<LteFfrAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<LteFfrDistributedAlgorithm> ()
    .AddAttribute ("CalculationInterval",
                   "Time interval between calculation of Edge sub-band, Default value 1 second",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&LteFfrDistributedAlgorithm::m_calculationInterval),
                   MakeTimeChecker ())
    .AddAttribute ("RsrqThreshold",
                   "If the RSRQ of is worse than this threshold, UE should be served in Edge sub-band",
                   UintegerValue (20),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_edgeSubBandRsrqThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("RsrpDifferenceThreshold",
                   "If the difference between the power of the signal received by UE from "
                   "the serving cell and the power of the signal received from the adjacent cell is less "
                   "than a RsrpDifferenceThreshold value, the cell weight is incremented",
                   UintegerValue (20),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_rsrpDifferenceThreshold),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterPowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_centerPowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgePowerOffset",
                   "PdschConfigDedicated::Pa value for Edge Sub-band, default value dB0",
                   UintegerValue (5),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_edgePowerOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeRbNum",
                   "Number of RB that can be used in edge Sub-band",
                   UintegerValue (8),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_edgeRbNum),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("CenterAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in center area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_centerAreaTpc),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("EdgeAreaTpc",
                   "TPC value which will be set in DL-DCI for UEs in edge area"
                   "Absolute mode is used, default value 1 is mapped to -1 according to"
                   "TS36.213 Table 5.1.1.1-2",
                   UintegerValue (1),
                   MakeUintegerAccessor (&LteFfrDistributedAlgorithm::m_edgeAreaTpc),
                   MakeUintegerChecker<uint8_t> ())

  ;
  return tid;
}


void
LteFfrDistributedAlgorithm::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFfrDistributedAlgorithm::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFfrDistributedAlgorithm::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFfrDistributedAlgorithm::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFfrDistributedAlgorithm::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteFfrAlgorithm::DoInitialize ();

  if (m_frCellTypeId != 0)
    {
      SetDownlinkConfiguration (m_frCellTypeId, m_dlBandwidth);
      SetUplinkConfiguration (m_frCellTypeId, m_ulBandwidth);
    }

  NS_LOG_LOGIC (this << " requesting Event A1 and A4 measurements"
                     << " (threshold = 0" << ")");
  LteRrcSap::ReportConfigEutra reportConfig;
  reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A1;
  reportConfig.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRQ;
  reportConfig.threshold1.range = 0;
  reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRQ;
  reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS120;
  m_rsrqMeasId = m_ffrRrcSapUser->AddUeMeasReportConfigForFfr (reportConfig);

  LteRrcSap::ReportConfigEutra reportConfigA4;
  reportConfigA4.eventId = LteRrcSap::ReportConfigEutra::EVENT_A4;
  reportConfigA4.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRP;
  reportConfigA4.threshold1.range = 0; // intentionally very low threshold
  reportConfigA4.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRP;
  reportConfigA4.reportInterval = LteRrcSap::ReportConfigEutra::MS120;
  m_rsrpMeasId = m_ffrRrcSapUser->AddUeMeasReportConfigForFfr (reportConfigA4);

  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlEdgeRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_ulEdgeRbgMap.resize (m_ulBandwidth, false);
  m_calculationEvent = Simulator::ScheduleNow (&LteFfrDistributedAlgorithm::Calculate, this);

}

void
LteFfrDistributedAlgorithm::Reconfigure ()
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
LteFfrDistributedAlgorithm::SetDownlinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
}

void
LteFfrDistributedAlgorithm::SetUplinkConfiguration (uint16_t cellId, uint8_t bandwidth)
{
  NS_LOG_FUNCTION (this);
}

void
LteFfrDistributedAlgorithm::InitializeDownlinkRbgMaps ()
{
  NS_LOG_FUNCTION (this);
  m_dlRbgMap.clear ();
  int rbgSize = GetRbgSize (m_dlBandwidth);
  m_dlRbgMap.resize (m_dlBandwidth / rbgSize, false);
}

void
LteFfrDistributedAlgorithm::InitializeUplinkRbgMaps ()
{
  NS_LOG_FUNCTION (this);
  m_ulRbgMap.clear ();
  m_ulRbgMap.resize (m_ulBandwidth, false);
}

std::vector <bool>
LteFfrDistributedAlgorithm::DoGetAvailableDlRbg ()
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
LteFfrDistributedAlgorithm::DoIsDlRbgAvailableForUe (int rbgId, uint16_t rnti)
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
  if (it->second == EdgeArea )
    {
      edgeUe = true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

std::vector <bool>
LteFfrDistributedAlgorithm::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      InitializeUplinkRbgMaps ();
    }

  return m_ulRbgMap;
}

bool
LteFfrDistributedAlgorithm::DoIsUlRbgAvailableForUe (int rbId, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (!m_enabledInUplink)
    {
      return true;
    }

  bool edgeRbg = m_ulEdgeRbgMap[rbId];

  std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
  if (it == m_ues.end ())
    {
      m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
      return !edgeRbg;
    }

  bool edgeUe = false;
  if (it->second == EdgeArea )
    {
      edgeUe = true;
    }

  return (edgeRbg && edgeUe) || (!edgeRbg && !edgeUe);
}

void
LteFfrDistributedAlgorithm::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFfrDistributedAlgorithm::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

void
LteFfrDistributedAlgorithm::DoReportUlCqiInfo (std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_WARN ("Method should not be called, because it is empty");
}

uint8_t
LteFfrDistributedAlgorithm::DoGetTpc (uint16_t rnti)
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
LteFfrDistributedAlgorithm::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);

  uint8_t minContinuousUlBandwidth = m_ulBandwidth;

  if (!m_enabledInUplink)
    {
      return minContinuousUlBandwidth;
    }

  minContinuousUlBandwidth =
    ((m_edgeRbNum > 0 ) && (m_edgeRbNum < minContinuousUlBandwidth)) ? m_edgeRbNum : minContinuousUlBandwidth;

  return minContinuousUlBandwidth;
}

void
LteFfrDistributedAlgorithm::DoReportUeMeas (uint16_t rnti,
                                            LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);
  NS_LOG_INFO ("CellId: " << m_cellId << " RNTI :" << rnti << " MeasId: " << (uint16_t) measResults.measId
                          << " RSRP: " << (uint16_t)measResults.measResultPCell.rsrpResult
                          << " RSRQ: " << (uint16_t)measResults.measResultPCell.rsrqResult);

  if (measResults.measId == m_rsrqMeasId)
    {
      //check if it is center or edge UE
      std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
      if (it == m_ues.end ())
        {
          m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
        }

      it = m_ues.find (rnti);
      if (measResults.measResultPCell.rsrqResult >= m_edgeSubBandRsrqThreshold)
        {
          if (it->second != CenterArea)
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Center sub-band");
              it->second = CenterArea;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_centerPowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
      else
        {
          if (it->second != EdgeArea )
            {
              NS_LOG_INFO ("UE RNTI: " << rnti << " will be served in Edge sub-band");
              it->second = EdgeArea;

              LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
              pdschConfigDedicated.pa = m_edgePowerOffset;
              m_ffrRrcSapUser->SetPdschConfigDedicated (rnti, pdschConfigDedicated);
            }
        }
    }
  else if (measResults.measId == m_rsrpMeasId)
    {
      std::map< uint16_t, uint8_t >::iterator it = m_ues.find (rnti);
      if (it == m_ues.end ())
        {
          m_ues.insert (std::pair< uint16_t, uint8_t > (rnti, AreaUnset));
        }

      UpdateNeighbourMeasurements (rnti, m_cellId, measResults.measResultPCell.rsrpResult, measResults.measResultPCell.rsrqResult);

      if (measResults.haveMeasResultNeighCells
          && !measResults.measResultListEutra.empty ())
        {
          for (std::list <LteRrcSap::MeasResultEutra>::iterator it = measResults.measResultListEutra.begin ();
               it != measResults.measResultListEutra.end ();
               ++it)
            {
              NS_ASSERT_MSG (it->haveRsrpResult == true,
                             "RSRP measurement is missing from cellId " << it->physCellId);
              NS_ASSERT_MSG (it->haveRsrqResult == true,
                             "RSRQ measurement is missing from cellId " << it->physCellId);
              UpdateNeighbourMeasurements (rnti, it->physCellId, it->rsrpResult, it->rsrqResult);

              bool found = false;
              for (std::vector<uint16_t>::iterator ncIt = m_neigborCell.begin (); ncIt != m_neigborCell.end (); ncIt++)
                {
                  if ((*ncIt) == it->physCellId)
                    {
                      found = true;
                    }
                }
              if (found == false)
                {
                  m_neigborCell.push_back (it->physCellId);
                }
            }
        }
      else
        {
          NS_LOG_WARN (this << " Event A4 received without measurement results from neighbouring cells");
        }
    }
  else
    {
      NS_LOG_WARN ("Ignoring measId " << (uint16_t) measResults.measId);
    }
}

void
LteFfrDistributedAlgorithm::Calculate ()
{
  NS_LOG_FUNCTION (this);
  m_calculationEvent = Simulator::Schedule (m_calculationInterval, &LteFfrDistributedAlgorithm::Calculate, this);

  int rbgSize = GetRbgSize (m_dlBandwidth);
  uint16_t rbgNum = m_dlBandwidth / rbgSize;

  m_cellWeightMap.clear ();
  m_dlEdgeRbgMap.clear ();
  m_dlEdgeRbgMap.resize (m_dlBandwidth / rbgSize, false);
  m_ulEdgeRbgMap.clear ();
  m_ulEdgeRbgMap.resize (m_ulBandwidth, false);

  MeasurementTable_t::iterator it1;
  MeasurementRow_t::iterator it2;
  Ptr<UeMeasure> servingCellMeasures;
  Ptr<UeMeasure> neighbourCellMeasures;

  uint32_t edgeUeNum = 0;
  std::map< uint16_t, uint8_t >::iterator areaIt;
  for (areaIt = m_ues.begin (); areaIt != m_ues.end (); areaIt++)
    {
      if (areaIt->second == EdgeArea)
        {
          edgeUeNum++;
        }
    }

  if (edgeUeNum != 0)
    {
      for (it1 = m_ueMeasures.begin (); it1 != m_ueMeasures.end (); it1++)
        {
          std::map< uint16_t, uint8_t >::iterator areaIt = m_ues.find (it1->first);
          if (areaIt->second != EdgeArea)
            {
              continue;
            }

          servingCellMeasures = 0;
          neighbourCellMeasures = 0;

          it2 = it1->second.find (m_cellId);
          if (it2 != it1->second.end ())
            {
              servingCellMeasures = it2->second;
            }
          else
            {
              continue;
            }

          for (it2 = it1->second.begin (); it2 != it1->second.end (); it2++)
            {
              if (it2->first != m_cellId)
                {
                  neighbourCellMeasures = it2->second;
                }
              else
                {
                  continue;
                }

              if (servingCellMeasures && neighbourCellMeasures)
                {
                  int16_t rsrpDifference = servingCellMeasures->m_rsrp - neighbourCellMeasures->m_rsrp;
                  NS_LOG_INFO ("CellId: " << m_cellId << " UE RNTI: " << it1->first
                                          << " NeighborCellId: " << neighbourCellMeasures->m_cellId
                                          << " RSRP Serving: " << (int)servingCellMeasures->m_rsrp
                                          << " RSRP Neighbor: " << (int)neighbourCellMeasures->m_rsrp
                                          << " RSRP Difference: " << (int)rsrpDifference);

                  if (rsrpDifference < m_rsrpDifferenceThreshold)
                    {
                      m_cellWeightMap[neighbourCellMeasures->m_cellId]++;
                    }
                }
            }
        }

      std::map< uint16_t, uint64_t > metricA;
      for (uint16_t i = 0; i < rbgNum; i++)
        {
          metricA[i] = 0;
        }

      std::map<uint16_t, uint32_t>::iterator cellIt;
      for (cellIt = m_cellWeightMap.begin (); cellIt != m_cellWeightMap.end (); cellIt++)
        {
          NS_LOG_INFO ("CellId: " << m_cellId << " NeighborCellId: " << cellIt->first << " Weight: " << cellIt->second);

          std::map<uint16_t, std::vector <bool> >::iterator rntpIt = m_rntp.find (cellIt->first);
          if (rntpIt == m_rntp.end ())
            {
              continue;
            }

          for (uint8_t i = 0; i < rbgNum; i++)
            {
              metricA[i] += cellIt->second * rntpIt->second[i];
            }
        }

      std::vector<uint16_t> sortedRbgByMetric;
      std::multimap< uint64_t, uint16_t > sortedMetricA;
      for (std::map<uint16_t, uint64_t>::const_iterator it = metricA.begin (); it != metricA.end (); ++it)
        {
          sortedMetricA.insert (std::pair<uint64_t, uint16_t> (it->second, it->first));
        }

      for (std::multimap< uint64_t, uint16_t >::const_iterator it = sortedMetricA.begin ();
           it != sortedMetricA.end (); ++it)
        {
          sortedRbgByMetric.push_back (it->second);
        }

      for (int i = 0; i < m_edgeRbNum / rbgSize; i++)
        {
          m_dlEdgeRbgMap[ sortedRbgByMetric[i] ] = true;
        }

      for (int i = 0; i < m_edgeRbNum / rbgSize; i++)
        {
          uint32_t rbgIndex = sortedRbgByMetric[i];
          for (int k = 0; k < rbgSize; k++)
            {
              uint32_t rbIndex = rbgSize * rbgIndex + k;
              m_ulEdgeRbgMap[ rbIndex ] = true;
            }
        }
    }

  for (std::vector<uint16_t>::iterator ncIt = m_neigborCell.begin (); ncIt != m_neigborCell.end (); ncIt++)
    {
      SendLoadInformation ((*ncIt));
    }
}

void
LteFfrDistributedAlgorithm::SendLoadInformation (uint16_t targetCellId)
{
  NS_LOG_FUNCTION (this);

  NS_LOG_INFO ("SendLoadInformation to CellId : " << targetCellId );

  std::vector<EpcX2Sap::UlInterferenceOverloadIndicationItem> m_currentUlInterferenceOverloadIndicationList;
  std::vector <EpcX2Sap::UlHighInterferenceInformationItem>  m_currentUlHighInterferenceInformationList;
  EpcX2Sap::RelativeNarrowbandTxBand m_currentRelativeNarrowbandTxBand;

  m_currentRelativeNarrowbandTxBand.rntpPerPrbList = m_dlEdgeRbgMap;

  EpcX2Sap::CellInformationItem cii;
  cii.sourceCellId = m_cellId;
  cii.ulInterferenceOverloadIndicationList = m_currentUlInterferenceOverloadIndicationList;
  cii.ulHighInterferenceInformationList = m_currentUlHighInterferenceInformationList;
  cii.relativeNarrowbandTxBand = m_currentRelativeNarrowbandTxBand;

  EpcX2Sap::LoadInformationParams params;
  params.targetCellId = targetCellId;
  params.cellInformationList.push_back (cii);

  m_ffrRrcSapUser->SendLoadInformation (params);
}

void
LteFfrDistributedAlgorithm::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_INFO ("CellId: " << m_cellId
                          << " Recv X2 message: LOAD INFORMATION from CellId:" << params.cellInformationList[0].sourceCellId);

  if (params.cellInformationList[0].sourceCellId > m_cellId)
    {
      return;
    }

  uint16_t neighborCellId = params.cellInformationList[0].sourceCellId;
  std::map<uint16_t, std::vector <bool> >::iterator it = m_rntp.find (neighborCellId);
  if (it != m_rntp.end ())
    {
      it->second = params.cellInformationList[0].relativeNarrowbandTxBand.rntpPerPrbList;
    }
  else
    {
      m_rntp.insert (std::pair<uint16_t, std::vector <bool> > (neighborCellId, params.cellInformationList[0].relativeNarrowbandTxBand.rntpPerPrbList));
    }
}

void
LteFfrDistributedAlgorithm::UpdateNeighbourMeasurements (uint16_t rnti,
                                                         uint16_t cellId,
                                                         uint8_t rsrp,
                                                         uint8_t rsrq)
{
  NS_LOG_FUNCTION (this << rnti << cellId << (uint16_t) rsrq);

  MeasurementTable_t::iterator it1;
  it1 = m_ueMeasures.find (rnti);

  if (it1 == m_ueMeasures.end ())
    {
      // insert a new UE entry
      MeasurementRow_t row;
      std::pair<MeasurementTable_t::iterator, bool> ret;
      ret = m_ueMeasures.insert (std::pair<uint16_t, MeasurementRow_t> (rnti, row));
      NS_ASSERT (ret.second);
      it1 = ret.first;
    }

  NS_ASSERT (it1 != m_ueMeasures.end ());
  Ptr<UeMeasure> cellMeasures;
  std::map<uint16_t, Ptr<UeMeasure> >::iterator it2;
  it2 = it1->second.find (cellId);

  if (it2 != it1->second.end ())
    {
      cellMeasures = it2->second;
      cellMeasures->m_cellId = cellId;
      cellMeasures->m_rsrp = rsrp;
      cellMeasures->m_rsrq = rsrq;
    }
  else
    {
      // insert a new cell entry
      cellMeasures = Create<UeMeasure> ();
      cellMeasures->m_cellId = cellId;
      cellMeasures->m_rsrp = rsrp;
      cellMeasures->m_rsrq = rsrq;
      it1->second[cellId] = cellMeasures;
    }

} // end of UpdateNeighbourMeasurements

} // end of namespace ns3
