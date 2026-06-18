/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Budiarto Herman
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
 * Author: Budiarto Herman <budiarto.herman@magister.fi>
 *
 */

#include "a3-rsrp-handover-algorithm.h"
#include <ns3/log.h>
#include <ns3/double.h>
#include <ns3/lte-common.h>
#include <algorithm>
#include <list>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("A3RsrpHandoverAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (A3RsrpHandoverAlgorithm);


A3RsrpHandoverAlgorithm::A3RsrpHandoverAlgorithm ()
  : m_handoverManagementSapUser (0)
{
  NS_LOG_FUNCTION (this);
  m_handoverManagementSapProvider = new MemberLteHandoverManagementSapProvider<A3RsrpHandoverAlgorithm> (this);
}


A3RsrpHandoverAlgorithm::~A3RsrpHandoverAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


TypeId
A3RsrpHandoverAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::A3RsrpHandoverAlgorithm")
    .SetParent<LteHandoverAlgorithm> ()
    .SetGroupName ("Lte")
    .AddConstructor<A3RsrpHandoverAlgorithm> ()
    .AddAttribute ("Hysteresis",
                   "Handover margin (hysteresis) in dB "
                   "(rounded to the nearest multiple of 0.5 dB)",
                   DoubleValue (3.0),
                   MakeDoubleAccessor (&A3RsrpHandoverAlgorithm::m_hysteresisDb),
                   MakeDoubleChecker<uint8_t> (0.0, 15.0)) // Hysteresis IE value range is [0..30] as per Section 6.3.5 of 3GPP TS 36.331
    .AddAttribute ("TimeToTrigger",
                   "Time during which neighbour cell's RSRP "
                   "must continuously higher than serving cell's RSRP "
                   "in order to trigger a handover",
                   TimeValue (MilliSeconds (256)), // 3GPP time-to-trigger median value as per Section 6.3.5 of 3GPP TS 36.331
                   MakeTimeAccessor (&A3RsrpHandoverAlgorithm::m_timeToTrigger),
                   MakeTimeChecker ())
  ;
  return tid;
}


void
A3RsrpHandoverAlgorithm::SetLteHandoverManagementSapUser (LteHandoverManagementSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_handoverManagementSapUser = s;
}


LteHandoverManagementSapProvider*
A3RsrpHandoverAlgorithm::GetLteHandoverManagementSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_handoverManagementSapProvider;
}


void
A3RsrpHandoverAlgorithm::DoInitialize ()
{
  NS_LOG_FUNCTION (this);

  uint8_t hysteresisIeValue = EutranMeasurementMapping::ActualHysteresis2IeValue (m_hysteresisDb);
  NS_LOG_LOGIC (this << " requesting Event A3 measurements"
                     << " (hysteresis=" << (uint16_t) hysteresisIeValue << ")"
                     << " (ttt=" << m_timeToTrigger.As (Time::MS) << ")");

  LteRrcSap::ReportConfigEutra reportConfig;
  reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A3;
  reportConfig.a3Offset = 0;
  reportConfig.hysteresis = hysteresisIeValue;
  reportConfig.timeToTrigger = m_timeToTrigger.GetMilliSeconds ();
  reportConfig.reportOnLeave = false;
  reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRP;
  reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS1024;
  m_measIds = m_handoverManagementSapUser->AddUeMeasReportConfigForHandover (reportConfig);

  LteHandoverAlgorithm::DoInitialize ();
}


void
A3RsrpHandoverAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_handoverManagementSapProvider;
}


void
A3RsrpHandoverAlgorithm::DoReportUeMeas (uint16_t rnti,
                                         LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);

  if (std::find (begin (m_measIds), end (m_measIds), measResults.measId) == std::end (m_measIds))
    {
      NS_LOG_WARN ("Ignoring measId " << (uint16_t) measResults.measId);
      return;
    }

  if (measResults.haveMeasResultNeighCells
      && !measResults.measResultListEutra.empty ())
    {
      uint16_t bestNeighbourCellId = 0;
      uint8_t bestNeighbourRsrp = 0;

      for (std::list <LteRrcSap::MeasResultEutra>::iterator it = measResults.measResultListEutra.begin ();
           it != measResults.measResultListEutra.end ();
           ++it)
        {
          if (it->haveRsrpResult)
            {
              if ((bestNeighbourRsrp < it->rsrpResult)
                  && IsValidNeighbour (it->physCellId))
                {
                  bestNeighbourCellId = it->physCellId;
                  bestNeighbourRsrp = it->rsrpResult;
                }
            }
          else
            {
              NS_LOG_WARN ("RSRP measurement is missing from cell ID " << it->physCellId);
            }
        }

      if (bestNeighbourCellId > 0)
        {
          NS_LOG_LOGIC ("Trigger Handover to cellId " << bestNeighbourCellId);
          NS_LOG_LOGIC ("target cell RSRP " << (uint16_t) bestNeighbourRsrp);
          NS_LOG_LOGIC ("serving cell RSRP " << (uint16_t) measResults.measResultPCell.rsrpResult);

          // Inform eNodeB RRC about handover
          m_handoverManagementSapUser->TriggerHandover (rnti,
                                                        bestNeighbourCellId);
        }
    }
  else
    {
      NS_LOG_WARN (this << " Event A3 received without measurement results from neighbouring cells");
    }

} // end of DoReportUeMeas


bool
A3RsrpHandoverAlgorithm::IsValidNeighbour (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << cellId);

  /**
   * \todo In the future, this function can be expanded to validate whether the
   *       neighbour cell is a valid target cell, e.g., taking into account the
   *       NRT in ANR and whether it is a CSG cell with closed access.
   */

  return true;
}


} // end of namespace ns3
