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

#include "lte-ffr-simple.h"
#include <ns3/log.h>
#include "ns3/lte-rrc-sap.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteFfrSimple");

NS_OBJECT_ENSURE_REGISTERED (LteFfrSimple);


LteFfrSimple::LteFfrSimple ()
  : m_ffrSapUser (0),
    m_ffrRrcSapUser (0),
    m_dlOffset (0),
    m_dlSubBand (0),
    m_ulOffset (0),
    m_ulSubBand (0),
    m_measId (0),
    m_changePdschConfigDedicated (false),
    m_tpc (1),
    m_tpcNum (0),
    m_accumulatedMode (false)
{
  NS_LOG_FUNCTION (this);
  m_ffrSapProvider = new MemberLteFfrSapProvider<LteFfrSimple> (this);
  m_ffrRrcSapProvider = new MemberLteFfrRrcSapProvider<LteFfrSimple> (this);
}


LteFfrSimple::~LteFfrSimple ()
{
  NS_LOG_FUNCTION (this);
}


void
LteFfrSimple::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_ffrSapProvider;
  delete m_ffrRrcSapProvider;
}


TypeId
LteFfrSimple::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteFfrSimple")
    .SetParent<LteFfrAlgorithm> ()
    .AddConstructor<LteFfrSimple> ()
    .AddAttribute ("UlSubBandOffset",
                   "Uplink Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrSimple::m_ulOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("UlSubBandwidth",
                   "Uplink Transmission SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (25),
                   MakeUintegerAccessor (&LteFfrSimple::m_ulSubBand),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlSubBandOffset",
                   "Downlink Offset in number of Resource Block Groups",
                   UintegerValue (0),
                   MakeUintegerAccessor (&LteFfrSimple::m_dlOffset),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DlSubBandwidth",
                   "Downlink Transmission SubBandwidth Configuration in number of Resource Block Groups",
                   UintegerValue (12),
                   MakeUintegerAccessor (&LteFfrSimple::m_dlSubBand),
                   MakeUintegerChecker<uint8_t> ())
    .AddTraceSource ("ChangePdschConfigDedicated",
                     "trace fired upon change of PdschConfigDedicated",
                     MakeTraceSourceAccessor (&LteFfrSimple::m_changePdschConfigDedicatedTrace),
                     "ns3::LteFfrSimple::PdschTracedCallback")
  ;
  return tid;
}


void
LteFfrSimple::SetLteFfrSapUser (LteFfrSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrSapUser = s;
}


LteFfrSapProvider*
LteFfrSimple::GetLteFfrSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrSapProvider;
}

void
LteFfrSimple::SetLteFfrRrcSapUser (LteFfrRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ffrRrcSapUser = s;
}


LteFfrRrcSapProvider*
LteFfrSimple::GetLteFfrRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ffrRrcSapProvider;
}


void
LteFfrSimple::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteFfrAlgorithm::DoInitialize ();

  NS_LOG_LOGIC (this << " requesting Event A4 measurements"
                     << " (threshold = 0" << ")");
  LteRrcSap::ReportConfigEutra reportConfig;
  reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A1;
  reportConfig.threshold1.choice = LteRrcSap::ThresholdEutra::THRESHOLD_RSRQ;
  reportConfig.threshold1.range = 0;
  reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRQ;
  reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS120;
  m_measId = m_ffrRrcSapUser->AddUeMeasReportConfigForFfr (reportConfig);

  m_pdschConfigDedicated.pa = LteRrcSap::PdschConfigDedicated::dB0;
}

void
LteFfrSimple::Reconfigure ()
{
  NS_LOG_FUNCTION (this);
}

void
LteFfrSimple::ChangePdschConfigDedicated (bool change)
{
  m_changePdschConfigDedicated = change;
}

void
LteFfrSimple::SetPdschConfigDedicated (LteRrcSap::PdschConfigDedicated pdschConfigDedicated)
{
  m_pdschConfigDedicated = pdschConfigDedicated;
}

void
LteFfrSimple::SetTpc (uint32_t tpc, uint32_t num, bool acculumatedMode)
{
  m_tpc = tpc;
  m_tpcNum = num;
  m_accumulatedMode = acculumatedMode;
}

std::vector <bool>
LteFfrSimple::DoGetAvailableDlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_dlRbgMap.empty ())
    {
      int rbgSize = GetRbgSize (m_dlBandwidth);
      m_dlRbgMap.resize (m_dlBandwidth / rbgSize, true);

      for (uint8_t i = m_dlOffset; i < (m_dlOffset + m_dlSubBand); i++)
        {
          m_dlRbgMap[i] = false;

        }
    }

  return m_dlRbgMap;
}

bool
LteFfrSimple::DoIsDlRbgAvailableForUe (int i, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  return true;
}

std::vector <bool>
LteFfrSimple::DoGetAvailableUlRbg ()
{
  NS_LOG_FUNCTION (this);

  if (m_ulRbgMap.empty ())
    {
      m_ulRbgMap.resize (m_ulBandwidth, true);

      for (uint8_t i = m_ulOffset; i < (m_ulOffset + m_ulSubBand); i++)
        {
          m_ulRbgMap[i] = false;
        }
    }

  return m_ulRbgMap;
}

bool
LteFfrSimple::DoIsUlRbgAvailableForUe (int i, uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  return true;
}

void
LteFfrSimple::DoReportDlCqiInfo (const struct FfMacSchedSapProvider::SchedDlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
}

void
LteFfrSimple::DoReportUlCqiInfo (const struct FfMacSchedSapProvider::SchedUlCqiInfoReqParameters& params)
{
  NS_LOG_FUNCTION (this);
}

void
LteFfrSimple::DoReportUlCqiInfo (std::map <uint16_t, std::vector <double> > ulCqiMap)
{
  NS_LOG_FUNCTION (this);
}

uint8_t
LteFfrSimple::DoGetTpc (uint16_t rnti)
{
  NS_LOG_FUNCTION (this);

  if (m_accumulatedMode)
    {
      if (m_tpcNum > 0)
        {
          m_tpcNum--;
          return m_tpc;
        }
      else
        {
          return 1;
        }
    }
  else
    {
      return m_tpc;
    }

  return 1; // 1 is mapped to 0 for Accumulated mode, and to -1 in Absolute mode TS36.213 Table 5.1.1.1-2
}

uint16_t
LteFfrSimple::DoGetMinContinuousUlBandwidth ()
{
  NS_LOG_FUNCTION (this);
  return m_ulBandwidth;
}

void
LteFfrSimple::DoReportUeMeas (uint16_t rnti,
                              LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);

  std::map<uint16_t, LteRrcSap::PdschConfigDedicated>::iterator it;

  it = m_ues.find (rnti);

  if (it == m_ues.end ())
    {
      LteRrcSap::PdschConfigDedicated pdschConfigDedicated;
      pdschConfigDedicated.pa = LteRrcSap::PdschConfigDedicated::dB0;
      m_ues.insert (std::pair<uint16_t, LteRrcSap::PdschConfigDedicated> (rnti,
                                                                          pdschConfigDedicated));
    }

  if (m_changePdschConfigDedicated)
    {
      UpdatePdschConfigDedicated ();
    }
}

void
LteFfrSimple::UpdatePdschConfigDedicated ()
{
  NS_LOG_FUNCTION (this);

  std::map<uint16_t, LteRrcSap::PdschConfigDedicated>::iterator it;
  for (it = m_ues.begin (); it != m_ues.end (); it++)
    {
      if (it->second.pa != m_pdschConfigDedicated.pa)
        {
          m_changePdschConfigDedicatedTrace (it->first, m_pdschConfigDedicated.pa);
          LteRrcSap::PdschConfigDedicated pdschConfigDedicated = m_pdschConfigDedicated;
          m_ffrRrcSapUser->SetPdschConfigDedicated (it->first, pdschConfigDedicated );
        }
    }
}

void
LteFfrSimple::DoRecvLoadInformation (EpcX2Sap::LoadInformationParams params)
{
  NS_LOG_FUNCTION (this);
}

} // end of namespace ns3
