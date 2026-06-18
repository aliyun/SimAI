/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Original work authors (from lte-enb-rrc.cc):
 * - Nicola Baldo <nbaldo@cttc.es>
 * - Marco Miozzo <mmiozzo@cttc.es>
 * - Manuel Requena <manuel.requena@cttc.es>
 *
 * Converted to ANR interface by:
 * - Budiarto Herman <budiarto.herman@magister.fi>
 */

#include "lte-anr.h"
#include <ns3/log.h>
#include <ns3/uinteger.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteAnr");

NS_OBJECT_ENSURE_REGISTERED (LteAnr);


LteAnr::LteAnr (uint16_t servingCellId)
  : m_anrSapUser (0),
    m_threshold (0),
    m_measId (0),
    m_servingCellId (servingCellId)
{
  NS_LOG_FUNCTION (this << servingCellId);
  m_anrSapProvider = new MemberLteAnrSapProvider<LteAnr> (this);
}


LteAnr::~LteAnr ()
{
  NS_LOG_FUNCTION (this << m_servingCellId);
}


TypeId
LteAnr::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteAnr")
    .SetParent<Object> ()
    .SetGroupName("Lte")
    .AddAttribute ("Threshold",
                   "Minimum RSRQ range value required for detecting a neighbour cell",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteAnr::m_threshold),
                   MakeUintegerChecker<uint8_t> (0, 34)) // RSRQ range is [0..34] as per Section 9.1.7 of 3GPP TS 36.133
  ;
  return tid;
}


void
LteAnr::AddNeighbourRelation (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << m_servingCellId << cellId);

  if (cellId == m_servingCellId)
    {
      NS_FATAL_ERROR ("Serving cell ID " << cellId << " may not be added into NRT");
    }

  if (m_neighbourRelationTable.find (cellId) != m_neighbourRelationTable.end ())
    {
      NS_FATAL_ERROR ("There is already an entry in the NRT for cell ID " << cellId);
    }

  NeighbourRelation_t neighbourRelation;
  neighbourRelation.noRemove = true;
  neighbourRelation.noHo = true;
  neighbourRelation.noX2 = false;
  neighbourRelation.detectedAsNeighbour = false;
  m_neighbourRelationTable[cellId] = neighbourRelation;
}


void
LteAnr::RemoveNeighbourRelation (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << m_servingCellId << cellId);

  NeighbourRelationTable_t::iterator it = m_neighbourRelationTable.find (cellId);
  if (it != m_neighbourRelationTable.end ())
    {
      NS_FATAL_ERROR ("Cell ID " << cellId << " cannot be found in NRT");
    }

  m_neighbourRelationTable.erase (it);
}


void
LteAnr::SetLteAnrSapUser (LteAnrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_anrSapUser = s;
}


LteAnrSapProvider*
LteAnr::GetLteAnrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_anrSapProvider;
}


void
LteAnr::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  NS_LOG_LOGIC (this << " requesting Event A4 measurements"
                     << " (threshold=" << (uint16_t) m_threshold << ")");
  LteRrcSap::ReportConfigEutra reportConfig;
  reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A4;
  reportConfig.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRQ;
  reportConfig.threshold1.range = m_threshold;
  reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRQ;
  reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS480;
  m_measId = m_anrSapUser->AddUeMeasReportConfigForAnr (reportConfig);
}


void
LteAnr::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_anrSapProvider;
  m_neighbourRelationTable.clear ();
}


void
LteAnr::DoReportUeMeas (LteRrcSap::MeasResults measResults)
{
  uint8_t measId = measResults.measId;
  NS_LOG_FUNCTION (this << m_servingCellId << (uint16_t) measId);

  if (measId != m_measId)
    {
      NS_LOG_WARN (this << " Skipping unexpected measurement identity " << (uint16_t) measId);
    }
  else
    {
      if (measResults.haveMeasResultNeighCells
          && !(measResults.measResultListEutra.empty ()))
        {
          for (std::list <LteRrcSap::MeasResultEutra>::iterator it = measResults.measResultListEutra.begin ();
               it != measResults.measResultListEutra.end ();
               ++it)
            {
              // Keep new RSRQ value reported for the neighbour cell
              NS_ASSERT_MSG (it->haveRsrqResult == true,
                             "RSRQ measure missing for cellId " << it->physCellId);

              // Update Neighbour Relation Table
              NeighbourRelationTable_t::iterator itNrt = m_neighbourRelationTable.find (it->physCellId);
              if (itNrt != m_neighbourRelationTable.end ())
                {
                  // Update neighbour relation entry
                  NS_LOG_LOGIC (this << " updating NRT of cell " << m_servingCellId
                                     << " with entry of cell " << it->physCellId);
                  if (itNrt->second.noX2 == false)
                    {
                      NS_LOG_LOGIC (this << " enabling handover"
                                         << " from cell " << m_servingCellId
                                         << " to cell " << it->physCellId);
                      itNrt->second.noHo = false;
                    }
                  itNrt->second.detectedAsNeighbour = true;
                }
              else
                {
                  // Discovered new neighbour
                  NS_LOG_LOGIC (this << " inserting NRT of cell " << m_servingCellId
                                     << " with newly discovered neighbouring cell "
                                     << it->physCellId);
                  NeighbourRelation_t neighbourRelation;
                  neighbourRelation.noRemove = false;
                  neighbourRelation.noHo = true;
                  neighbourRelation.noX2 = true;
                  neighbourRelation.detectedAsNeighbour = true;
                  m_neighbourRelationTable[it->physCellId] = neighbourRelation;
                }

            } // end of for (it = measResults.measResultListEutra.begin ())

        } // end of if (measResults.haveMeasResultNeighCells && !(measResults.measResultListEutra.empty ()))
      else
        {
          NS_LOG_WARN (this << " Event A4 received without measurement results from neighbouring cells");
          /// \todo Remove neighbours in the NRT.
        }

    } // end of else of if (measId != m_measId)

} // end of DoReportUeMeas


void
LteAnr::DoAddNeighbourRelation (uint16_t cellId)
{
  NS_LOG_FUNCTION (this << cellId);
  AddNeighbourRelation (cellId);
}


bool
LteAnr::DoGetNoRemove (uint16_t cellId) const
{
  NS_LOG_FUNCTION (this << m_servingCellId << cellId);
  return Find (cellId)->noRemove;
}


bool
LteAnr::DoGetNoHo (uint16_t cellId) const
{
  NS_LOG_FUNCTION (this << m_servingCellId << cellId);
  return Find (cellId)->noHo;
}


bool
LteAnr::DoGetNoX2 (uint16_t cellId) const
{
  NS_LOG_FUNCTION (this << m_servingCellId << cellId);
  return Find (cellId)->noX2;
}


const LteAnr::NeighbourRelation_t *
LteAnr::Find (uint16_t cellId) const
{
  NeighbourRelationTable_t::const_iterator it = m_neighbourRelationTable.find (cellId);
  if (it == m_neighbourRelationTable.end ())
    {
      NS_FATAL_ERROR ("Cell ID " << cellId << " cannot be found in NRT");
    }
  return &(it->second);
}


} // end of namespace ns3

