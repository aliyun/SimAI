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

#include "no-op-handover-algorithm.h"
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NoOpHandoverAlgorithm");

NS_OBJECT_ENSURE_REGISTERED (NoOpHandoverAlgorithm);


NoOpHandoverAlgorithm::NoOpHandoverAlgorithm ()
  : m_handoverManagementSapUser (0)
{
  NS_LOG_FUNCTION (this);
  m_handoverManagementSapProvider = new MemberLteHandoverManagementSapProvider<NoOpHandoverAlgorithm> (this);
}


NoOpHandoverAlgorithm::~NoOpHandoverAlgorithm ()
{
  NS_LOG_FUNCTION (this);
}


void
NoOpHandoverAlgorithm::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_handoverManagementSapProvider;
}


TypeId
NoOpHandoverAlgorithm::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::NoOpHandoverAlgorithm")
    .SetParent<LteHandoverAlgorithm> ()
    .SetGroupName("Lte")
    .AddConstructor<NoOpHandoverAlgorithm> ()
  ;
  return tid;
}


void
NoOpHandoverAlgorithm::SetLteHandoverManagementSapUser (LteHandoverManagementSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_handoverManagementSapUser = s;
}


LteHandoverManagementSapProvider*
NoOpHandoverAlgorithm::GetLteHandoverManagementSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_handoverManagementSapProvider;
}


void
NoOpHandoverAlgorithm::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  LteHandoverAlgorithm::DoInitialize ();
}


void
NoOpHandoverAlgorithm::DoReportUeMeas (uint16_t rnti,
                                       LteRrcSap::MeasResults measResults)
{
  NS_LOG_FUNCTION (this << rnti << (uint16_t) measResults.measId);
}


} // end of namespace ns3
