/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Jaume Nin <jnin@cttc.es>
 * Modified by: Danilo Abrignani <danilo.abrignani@unibo.it> (Carrier Aggregation - GSoC 2015)
 *              Biljana Bojovic <biljana.bojovic@cttc.es> (Carrier Aggregation)
 */

#include "mac-stats-calculator.h"
#include "ns3/string.h"
#include <ns3/simulator.h>
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MacStatsCalculator");

NS_OBJECT_ENSURE_REGISTERED (MacStatsCalculator);

MacStatsCalculator::MacStatsCalculator ()
  : m_dlFirstWrite (true),
    m_ulFirstWrite (true)
{
  NS_LOG_FUNCTION (this);

}

MacStatsCalculator::~MacStatsCalculator ()
{
  NS_LOG_FUNCTION (this);
  if (m_dlOutFile.is_open())
    {
      m_dlOutFile.close();
    }

  if (m_ulOutFile.is_open())
    {
      m_ulOutFile.close();
    }
}

TypeId
MacStatsCalculator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MacStatsCalculator")
    .SetParent<LteStatsCalculator> ()
    .SetGroupName ("Lte")
    .AddConstructor<MacStatsCalculator> ()
    .AddAttribute ("DlOutputFilename",
                   "Name of the file where the downlink results will be saved.",
                   StringValue ("DlMacStats.txt"),
                   MakeStringAccessor (&MacStatsCalculator::SetDlOutputFilename),
                   MakeStringChecker ())
    .AddAttribute ("UlOutputFilename",
                   "Name of the file where the uplink results will be saved.",
                   StringValue ("UlMacStats.txt"),
                   MakeStringAccessor (&MacStatsCalculator::SetUlOutputFilename),
                   MakeStringChecker ())
  ;
  return tid;
}

void
MacStatsCalculator::SetUlOutputFilename (std::string outputFilename)
{
  LteStatsCalculator::SetUlOutputFilename (outputFilename);
}

std::string
MacStatsCalculator::GetUlOutputFilename (void)
{
  return LteStatsCalculator::GetUlOutputFilename ();
}

void
MacStatsCalculator::SetDlOutputFilename (std::string outputFilename)
{
  LteStatsCalculator::SetDlOutputFilename (outputFilename);
}

std::string
MacStatsCalculator::GetDlOutputFilename (void)
{
  return LteStatsCalculator::GetDlOutputFilename ();
}

void
MacStatsCalculator::DlScheduling (uint16_t cellId, uint64_t imsi, DlSchedulingCallbackInfo dlSchedulingCallbackInfo)
{
  NS_LOG_FUNCTION (this << cellId << imsi << dlSchedulingCallbackInfo.frameNo << dlSchedulingCallbackInfo.subframeNo <<
                   dlSchedulingCallbackInfo.rnti << (uint32_t) dlSchedulingCallbackInfo.mcsTb1 << dlSchedulingCallbackInfo.sizeTb1 << (uint32_t) dlSchedulingCallbackInfo.mcsTb2 << dlSchedulingCallbackInfo.sizeTb2);
  NS_LOG_INFO ("Write DL Mac Stats in " << GetDlOutputFilename ().c_str ());

  if ( m_dlFirstWrite == true )
    {
      m_dlOutFile.open (GetDlOutputFilename ().c_str ());
      if (!m_dlOutFile.is_open ())
        {
          NS_LOG_ERROR ("Can't open file " << GetDlOutputFilename ().c_str ());
          return;
        }
      m_dlFirstWrite = false;
      m_dlOutFile << "% time\tcellId\tIMSI\tframe\tsframe\tRNTI\tmcsTb1\tsizeTb1\tmcsTb2\tsizeTb2\tccId";
      m_dlOutFile << "\n";
    }

  m_dlOutFile << Simulator::Now ().GetSeconds () << "\t";
  m_dlOutFile << (uint32_t) cellId << "\t";
  m_dlOutFile << imsi << "\t";
  m_dlOutFile << dlSchedulingCallbackInfo.frameNo << "\t";
  m_dlOutFile << dlSchedulingCallbackInfo.subframeNo << "\t";
  m_dlOutFile << dlSchedulingCallbackInfo.rnti << "\t";
  m_dlOutFile << (uint32_t) dlSchedulingCallbackInfo.mcsTb1 << "\t";
  m_dlOutFile << dlSchedulingCallbackInfo.sizeTb1 << "\t";
  m_dlOutFile << (uint32_t) dlSchedulingCallbackInfo.mcsTb2 << "\t";
  m_dlOutFile << dlSchedulingCallbackInfo.sizeTb2 << "\t";
  m_dlOutFile << (uint32_t) dlSchedulingCallbackInfo.componentCarrierId << std::endl;
}

void
MacStatsCalculator::UlScheduling (uint16_t cellId, uint64_t imsi, uint32_t frameNo,
                                  uint32_t subframeNo, uint16_t rnti,uint8_t mcsTb, uint16_t size, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (this << cellId << imsi << frameNo << subframeNo << rnti << (uint32_t) mcsTb << size);
  NS_LOG_INFO ("Write UL Mac Stats in " << GetUlOutputFilename ().c_str ());

  if ( m_ulFirstWrite == true )
    {
      m_ulOutFile.open (GetUlOutputFilename ().c_str ());
      if (!m_ulOutFile.is_open ())
        {
          NS_LOG_ERROR ("Can't open file " << GetUlOutputFilename ().c_str ());
          return;
        }
      m_ulFirstWrite = false;
      m_ulOutFile << "% time\tcellId\tIMSI\tframe\tsframe\tRNTI\tmcs\tsize\tccId";
      m_ulOutFile << "\n";
    }

  m_ulOutFile << Simulator::Now ().GetSeconds () << "\t";
  m_ulOutFile << (uint32_t) cellId << "\t";
  m_ulOutFile << imsi << "\t";
  m_ulOutFile << frameNo << "\t";
  m_ulOutFile << subframeNo << "\t";
  m_ulOutFile << rnti << "\t";
  m_ulOutFile << (uint32_t) mcsTb << "\t";
  m_ulOutFile << size << "\t";
  m_ulOutFile << (uint32_t) componentCarrierId << std::endl;
}

void
MacStatsCalculator::DlSchedulingCallback (Ptr<MacStatsCalculator> macStats, std::string path, DlSchedulingCallbackInfo dlSchedulingCallbackInfo)
{
  NS_LOG_FUNCTION (macStats << path);
  uint64_t imsi = 0;
  std::ostringstream pathAndRnti;
  std::string pathEnb  = path.substr (0, path.find ("/ComponentCarrierMap"));
  pathAndRnti << pathEnb << "/LteEnbRrc/UeMap/" << dlSchedulingCallbackInfo.rnti;
  if (macStats->ExistsImsiPath (pathAndRnti.str ()) == true)
    {
      imsi = macStats->GetImsiPath (pathAndRnti.str ());
    }
  else
    {
      imsi = FindImsiFromEnbRlcPath (pathAndRnti.str ());
      macStats->SetImsiPath (pathAndRnti.str (), imsi);
    }
  uint16_t cellId = 0;
  if (macStats->ExistsCellIdPath (pathAndRnti.str ()) == true)
    {
      cellId = macStats->GetCellIdPath (pathAndRnti.str ());
    }
  else
    {
      cellId = FindCellIdFromEnbRlcPath (pathAndRnti.str ());
      macStats->SetCellIdPath (pathAndRnti.str (), cellId);
    }

  macStats->DlScheduling (cellId, imsi, dlSchedulingCallbackInfo);
}

void
MacStatsCalculator::UlSchedulingCallback (Ptr<MacStatsCalculator> macStats, std::string path,
                                          uint32_t frameNo, uint32_t subframeNo, uint16_t rnti,
                                          uint8_t mcs, uint16_t size, uint8_t componentCarrierId)
{
  NS_LOG_FUNCTION (macStats << path);

  uint64_t imsi = 0;
  std::ostringstream pathAndRnti;
  std::string pathEnb  = path.substr (0, path.find ("/ComponentCarrierMap"));
  pathAndRnti << pathEnb << "/LteEnbRrc/UeMap/" << rnti;
  if (macStats->ExistsImsiPath (pathAndRnti.str ()) == true)
    {
      imsi = macStats->GetImsiPath (pathAndRnti.str ());
    }
  else
    {
      imsi = FindImsiFromEnbRlcPath (pathAndRnti.str ());
      macStats->SetImsiPath (pathAndRnti.str (), imsi);
    }
  uint16_t cellId = 0;
  if (macStats->ExistsCellIdPath (pathAndRnti.str ()) == true)
    {
      cellId = macStats->GetCellIdPath (pathAndRnti.str ());
    }
  else
    {
      cellId = FindCellIdFromEnbRlcPath (pathAndRnti.str ());
      macStats->SetCellIdPath (pathAndRnti.str (), cellId);
    }

  macStats->UlScheduling (cellId, imsi, frameNo, subframeNo, rnti, mcs, size, componentCarrierId);
}


} // namespace ns3
