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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/node.h"

#include "ns3/lte-rlc-header.h"
#include "ns3/lte-rlc-am-header.h"
#include "ns3/lte-pdcp-header.h"

#include "lte-test-entities.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteTestEntities");

/////////////////////////////////////////////////////////////////////

TypeId
LteTestRrc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteTestRrc")
    .SetParent<Object> ()
    .AddConstructor<LteTestRrc> ()
    ;

  return tid;
}

LteTestRrc::LteTestRrc ()
{
  NS_LOG_FUNCTION (this);

  m_txPdus = 0;
  m_txBytes = 0;
  m_rxPdus = 0;
  m_rxBytes = 0;
  m_txLastTime = Time (0);
  m_rxLastTime = Time (0);

  m_pdcpSapUser = new LtePdcpSpecificLtePdcpSapUser<LteTestRrc> (this);
//   Simulator::ScheduleNow (&LteTestRrc::Start, this);
}

LteTestRrc::~LteTestRrc ()
{
  NS_LOG_FUNCTION (this);
}

void
LteTestRrc::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_pdcpSapUser;
}

void
LteTestRrc::SetDevice (Ptr<NetDevice> device)
{
  m_device = device;
}

void
LteTestRrc::SetLtePdcpSapProvider (LtePdcpSapProvider* s)
{
  m_pdcpSapProvider = s;
}

LtePdcpSapUser*
LteTestRrc::GetLtePdcpSapUser (void)
{
  return m_pdcpSapUser;
}


std::string
LteTestRrc::GetDataReceived (void)
{
  NS_LOG_FUNCTION (this);
  return m_receivedData;
}

// Stats
uint32_t
LteTestRrc::GetTxPdus (void)
{
  NS_LOG_FUNCTION (this << m_txPdus);
  return m_txPdus;
}

uint32_t
LteTestRrc::GetTxBytes (void)
{
  NS_LOG_FUNCTION (this << m_txBytes);
  return m_txBytes;
}

uint32_t
LteTestRrc::GetRxPdus (void)
{
  NS_LOG_FUNCTION (this << m_rxPdus);
  return m_rxPdus;
}

uint32_t
LteTestRrc::GetRxBytes (void)
{
  NS_LOG_FUNCTION (this << m_rxBytes);
  return m_rxBytes;
}

Time
LteTestRrc::GetTxLastTime (void)
{
  NS_LOG_FUNCTION (this << m_txLastTime);
  return m_txLastTime;
}

Time
LteTestRrc::GetRxLastTime (void)
{
  NS_LOG_FUNCTION (this << m_rxLastTime);
  return m_rxLastTime;
}


void
LteTestRrc::SetArrivalTime (Time arrivalTime)
{
  NS_LOG_FUNCTION (this << arrivalTime);
  m_arrivalTime = arrivalTime;
}

void
LteTestRrc::SetPduSize (uint32_t pduSize)
{
  NS_LOG_FUNCTION (this << pduSize);
  m_pduSize = pduSize;
}


/**
 * PDCP SAP
 */

void
LteTestRrc::DoReceivePdcpSdu (LtePdcpSapUser::ReceivePdcpSduParameters params)
{
  NS_LOG_FUNCTION (this << params.pdcpSdu->GetSize ());
  Ptr<Packet> p = params.pdcpSdu;
//   NS_LOG_LOGIC ("PDU received = " << (*p));

  uint32_t dataLen = p->GetSize ();
  uint8_t *buf = new uint8_t[dataLen];

  // Stats
  m_rxPdus++;
  m_rxBytes += dataLen;
  m_rxLastTime = Simulator::Now ();

  p->CopyData (buf, dataLen);
  m_receivedData = std::string ((char *)buf, dataLen);

//   NS_LOG_LOGIC (m_receivedData);

  delete [] buf;
}

/**
 * START
 */

void
LteTestRrc::Start ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_arrivalTime != Time (0), "Arrival time must be different from 0");

  // Stats
  m_txPdus++;
  m_txBytes += m_pduSize;
  m_txLastTime = Simulator::Now ();

  LtePdcpSapProvider::TransmitPdcpSduParameters p;
  p.rnti = 1111;
  p.lcid = 222;
  p.pdcpSdu = Create<Packet> (m_pduSize);
  
  bool haveContext = false;
  Ptr<Node> node;
  if (m_device != 0)
    {
      node = m_device->GetNode ();
      if (node != 0)
        {                    
          haveContext = true;
        }
    }
  if (haveContext)
    {
      Simulator::ScheduleWithContext (node->GetId (), Seconds (0), &LtePdcpSapProvider::TransmitPdcpSdu, m_pdcpSapProvider, p);
    }
  else
    {
      Simulator::Schedule (Seconds (0), &LtePdcpSapProvider::TransmitPdcpSdu, m_pdcpSapProvider, p);
    }

  m_nextPdu = Simulator::Schedule (m_arrivalTime, &LteTestRrc::Start, this);
//   Simulator::Run ();
}

void
LteTestRrc::Stop ()
{
  NS_LOG_FUNCTION (this);
  m_nextPdu.Cancel ();
}

void
LteTestRrc::SendData (Time at, std::string dataToSend)
{
  NS_LOG_FUNCTION (this << at << dataToSend.length () << dataToSend);

  // Stats
  m_txPdus++;
  m_txBytes += dataToSend.length ();

  LtePdcpSapProvider::TransmitPdcpSduParameters p;
  p.rnti = 1111;
  p.lcid = 222;

  NS_LOG_LOGIC ("Data(" << dataToSend.length () << ") = " << dataToSend.data ());
  p.pdcpSdu = Create<Packet> ((uint8_t *) dataToSend.data (), dataToSend.length ());

  NS_LOG_LOGIC ("Packet(" << p.pdcpSdu->GetSize () << ")");
  Simulator::Schedule (at, &LtePdcpSapProvider::TransmitPdcpSdu, m_pdcpSapProvider, p);
}

/////////////////////////////////////////////////////////////////////

TypeId
LteTestPdcp::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteTestPdcp")
    .SetParent<Object> ()
    .AddConstructor<LteTestPdcp> ()
    ;

  return tid;
}

LteTestPdcp::LteTestPdcp ()
{
  NS_LOG_FUNCTION (this);
  m_rlcSapUser = new LteRlcSpecificLteRlcSapUser<LteTestPdcp> (this);
  Simulator::ScheduleNow (&LteTestPdcp::Start, this);
}

LteTestPdcp::~LteTestPdcp ()
{
  NS_LOG_FUNCTION (this);
}

void
LteTestPdcp::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_rlcSapUser;
}

void
LteTestPdcp::SetLteRlcSapProvider (LteRlcSapProvider* s)
{
  m_rlcSapProvider = s;
}

LteRlcSapUser*
LteTestPdcp::GetLteRlcSapUser (void)
{
  return m_rlcSapUser;
}


std::string
LteTestPdcp::GetDataReceived (void)
{
  NS_LOG_FUNCTION (this);

  return m_receivedData;
}


/**
 * RLC SAP
 */

void
LteTestPdcp::DoReceivePdcpPdu (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p->GetSize ());
  NS_LOG_LOGIC ("Data = " << (*p));

  uint32_t dataLen = p->GetSize ();
  uint8_t *buf = new uint8_t[dataLen];
  p->CopyData (buf, dataLen);
  m_receivedData = std::string ((char *)buf, dataLen);

  NS_LOG_LOGIC (m_receivedData);

  delete [] buf;
}

/**
 * START
 */

void
LteTestPdcp::Start ()
{
  NS_LOG_FUNCTION (this);
}

void
LteTestPdcp::SendData (Time time, std::string dataToSend)
{
  NS_LOG_FUNCTION (this << time << dataToSend.length () << dataToSend);

  LteRlcSapProvider::TransmitPdcpPduParameters p;
  p.rnti = 1111;
  p.lcid = 222;

  NS_LOG_LOGIC ("Data(" << dataToSend.length () << ") = " << dataToSend.data ());
  p.pdcpPdu = Create<Packet> ((uint8_t *) dataToSend.data (), dataToSend.length ());

  NS_LOG_LOGIC ("Packet(" << p.pdcpPdu->GetSize () << ")");
  Simulator::Schedule (time, &LteRlcSapProvider::TransmitPdcpPdu, m_rlcSapProvider, p);
}

/////////////////////////////////////////////////////////////////////

TypeId
LteTestMac::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteTestMac")
    .SetParent<Object> ()
    .AddConstructor<LteTestMac> ()
    ;

  return tid;
}

LteTestMac::LteTestMac ()
{
  NS_LOG_FUNCTION (this);
  m_device = 0;
  m_macSapProvider = new EnbMacMemberLteMacSapProvider<LteTestMac> (this);
  m_macSapUser = 0;
  m_macLoopback = 0;
  m_pdcpHeaderPresent = false;
  m_rlcHeaderType = UM_RLC_HEADER;
  m_txOpportunityMode = MANUAL_MODE;
  m_txOppTime = Seconds (0.001);
  m_txOppSize = 0;

  m_txPdus = 0;
  m_txBytes = 0;
  m_rxPdus = 0;
  m_rxBytes = 0;

//   m_cmacSapProvider = new EnbMacMemberLteEnbCmacSapProvider (this);
//   m_schedSapUser = new EnbMacMemberFfMacSchedSapUser (this);
//   m_cschedSapUser = new EnbMacMemberFfMacCschedSapUser (this);
//   m_enbPhySapUser = new EnbMacMemberLteEnbPhySapUser (this);
}

LteTestMac::~LteTestMac ()
{
  NS_LOG_FUNCTION (this);
}

void
LteTestMac::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_macSapProvider;
//   delete m_cmacSapProvider;
//   delete m_schedSapUser;
//   delete m_cschedSapUser;
//   delete m_enbPhySapUser;

  m_device = 0;
}

void
LteTestMac::SetDevice (Ptr<NetDevice> device)
{
  m_device = device;
}

void
LteTestMac::SetLteMacSapUser (LteMacSapUser* s)
{
  m_macSapUser = s;
}

LteMacSapProvider*
LteTestMac::GetLteMacSapProvider (void)
{
  return m_macSapProvider;
}

void
LteTestMac::SetLteMacLoopback (Ptr<LteTestMac> s)
{
  m_macLoopback = s;
}

std::string
LteTestMac::GetDataReceived (void)
{
  NS_LOG_FUNCTION (this);
  return m_receivedData;
}

// Stats
uint32_t
LteTestMac::GetTxPdus (void)
{
  NS_LOG_FUNCTION (this << m_txPdus);
  return m_txPdus;
}

uint32_t
LteTestMac::GetTxBytes (void)
{
  NS_LOG_FUNCTION (this << m_txBytes);
  return m_txBytes;
}

uint32_t
LteTestMac::GetRxPdus (void)
{
  NS_LOG_FUNCTION (this << m_rxPdus);
  return m_rxPdus;
}

uint32_t
LteTestMac::GetRxBytes (void)
{
  NS_LOG_FUNCTION (this << m_rxBytes);
  return m_rxBytes;
}


void
LteTestMac::SendTxOpportunity (Time time, uint32_t bytes)
{
  NS_LOG_FUNCTION (this << time << bytes);
  bool haveContext = false;
  Ptr<Node> node;
  if (m_device != 0)
    {
      node = m_device->GetNode ();
      if (node != 0)
        {                    
          haveContext = true;
        }
    }
  LteMacSapUser::TxOpportunityParameters txOpParmas;
  txOpParmas.bytes = bytes;
  txOpParmas.layer = 0;
  txOpParmas.componentCarrierId = 0;
  txOpParmas.harqId = 0;
  txOpParmas.rnti = 0;
  txOpParmas.lcid = 0;

  if (haveContext)
    {
      Simulator::ScheduleWithContext (node->GetId (), time, &LteMacSapUser::NotifyTxOpportunity, m_macSapUser, txOpParmas);
    }
  else
    {
      Simulator::Schedule (time, &LteMacSapUser::NotifyTxOpportunity, m_macSapUser, txOpParmas);
    }
    
  if (m_txOpportunityMode == RANDOM_MODE)
  {
    if (m_txOppTime != Seconds (0))
    {
      Simulator::Schedule (m_txOppTime, &LteTestMac::SendTxOpportunity, this, m_txOppTime, m_txOppSize);
    }
  }
}

void
LteTestMac::SetPdcpHeaderPresent (bool present)
{
  NS_LOG_FUNCTION (this << present);
  m_pdcpHeaderPresent = present;
}

void
LteTestMac::SetRlcHeaderType (uint8_t rlcHeaderType)
{
  NS_LOG_FUNCTION (this << rlcHeaderType);
  m_rlcHeaderType = rlcHeaderType;
}

void
LteTestMac::SetTxOpportunityMode (uint8_t mode)
{
  NS_LOG_FUNCTION (this << (uint32_t)mode);
  m_txOpportunityMode = mode;

  if (m_txOpportunityMode == RANDOM_MODE)
    {
      if (m_txOppTime != Seconds (0.0))
        {
          SendTxOpportunity (m_txOppTime, m_txOppSize);
        }
    }
}

void
LteTestMac::SetTxOppTime (Time txOppTime)
{
  NS_LOG_FUNCTION (this << txOppTime);
  m_txOppTime = txOppTime;
}

void
LteTestMac::SetTxOppSize (uint32_t txOppSize)
{
  NS_LOG_FUNCTION (this << txOppSize);
  m_txOppSize = txOppSize;
}


/**
 * MAC SAP
 */

void
LteTestMac::DoTransmitPdu (LteMacSapProvider::TransmitPduParameters params)
{
  NS_LOG_FUNCTION (this << params.pdu->GetSize ());

  m_txPdus++;
  m_txBytes += params.pdu->GetSize ();

  LteMacSapUser::ReceivePduParameters rxPduParams;
  rxPduParams.p = params.pdu;
  rxPduParams.rnti = params.rnti;
  rxPduParams.lcid = params.lcid;

  if (m_device)
    {
      m_device->Send (params.pdu, m_device->GetBroadcast (), 0);
    }
  else if (m_macLoopback)
    {
      Simulator::Schedule (Seconds (0.1), &LteMacSapUser::ReceivePdu,
                           m_macLoopback->m_macSapUser, rxPduParams);
    }
  else
    {
      LtePdcpHeader pdcpHeader;

      if (m_rlcHeaderType == AM_RLC_HEADER)
        {
          // Remove AM RLC header
          LteRlcAmHeader rlcAmHeader;
          params.pdu->RemoveHeader (rlcAmHeader);
          NS_LOG_LOGIC ("AM RLC header: " << rlcAmHeader);
        }
      else // if (m_rlcHeaderType == UM_RLC_HEADER)
        {
          // Remove UM RLC header
          LteRlcHeader rlcHeader;
          params.pdu->RemoveHeader (rlcHeader);
          NS_LOG_LOGIC ("UM RLC header: " << rlcHeader);
        }

      // Remove PDCP header, if present
      if (m_pdcpHeaderPresent)
        {
          params.pdu->RemoveHeader (pdcpHeader);
          NS_LOG_LOGIC ("PDCP header: " << pdcpHeader);
        }

      // Copy data to a string
      uint32_t dataLen = params.pdu->GetSize ();
      uint8_t *buf = new uint8_t[dataLen];
      params.pdu->CopyData (buf, dataLen);
      m_receivedData = std::string ((char *)buf, dataLen);

      NS_LOG_LOGIC ("Data (" << dataLen << ") = " << m_receivedData);
      delete [] buf;
    }
}

void
LteTestMac::DoReportBufferStatus (LteMacSapProvider::ReportBufferStatusParameters params)
{
  NS_LOG_FUNCTION (this << params.txQueueSize << params.retxQueueSize << params.statusPduSize);

  if (m_txOpportunityMode == AUTOMATIC_MODE)
    {
      // cancel all previously scheduled TxOpps
      for (std::list<EventId>::iterator it = m_nextTxOppList.begin ();
           it != m_nextTxOppList.end ();
           ++it)
        {          
          it->Cancel ();
        }
      m_nextTxOppList.clear ();

      int32_t size = params.statusPduSize + params.txQueueSize  + params.retxQueueSize;
      Time time = m_txOppTime;
      LteMacSapUser::TxOpportunityParameters txOpParmas;
      txOpParmas.bytes = m_txOppSize;
      txOpParmas.layer = 0;
      txOpParmas.componentCarrierId = 0;
      txOpParmas.harqId = 0;
      txOpParmas.rnti = params.rnti;
      txOpParmas.lcid = params.lcid;
      while (size > 0)
        {
          EventId e = Simulator::Schedule (time, 
                                           &LteMacSapUser::NotifyTxOpportunity,
                                           m_macSapUser, txOpParmas);
          m_nextTxOppList.push_back (e);
          size -= m_txOppSize;
          time += m_txOppTime;
        }
    }
}


bool
LteTestMac::Receive (Ptr<NetDevice> nd, Ptr<const Packet> p, uint16_t protocol, const Address& addr)
{
  NS_LOG_FUNCTION (this << addr << protocol << p->GetSize ());

  m_rxPdus++;
  m_rxBytes += p->GetSize ();

  Ptr<Packet> packet = p->Copy ();
  LteMacSapUser::ReceivePduParameters rxPduParams;
  rxPduParams.p = packet;
  rxPduParams.rnti = 0;
  rxPduParams.lcid = 0;
  m_macSapUser->ReceivePdu (rxPduParams);
  return true;
}








NS_OBJECT_ENSURE_REGISTERED (EpcTestRrc);

EpcTestRrc::EpcTestRrc ()
  : m_s1SapProvider (0)
{
  NS_LOG_FUNCTION (this);
  m_s1SapUser = new MemberEpcEnbS1SapUser<EpcTestRrc> (this);
}


EpcTestRrc::~EpcTestRrc ()
{
  NS_LOG_FUNCTION (this);
}


void
EpcTestRrc::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  delete m_s1SapUser;
}

TypeId
EpcTestRrc::GetTypeId (void)
{
  NS_LOG_FUNCTION ("EpcTestRrc::GetTypeId");
  static TypeId tid = TypeId ("ns3::EpcTestRrc")
    .SetParent<Object> ()
    .AddConstructor<EpcTestRrc> ()
  ;
  return tid;
}
void 
EpcTestRrc::SetS1SapProvider (EpcEnbS1SapProvider * s)
{
  m_s1SapProvider = s;
}

  
EpcEnbS1SapUser* 
EpcTestRrc::GetS1SapUser ()
{
  return m_s1SapUser;
}

void
EpcTestRrc::DoInitialContextSetupRequest (EpcEnbS1SapUser::InitialContextSetupRequestParameters request)
{

}

void 
EpcTestRrc::DoDataRadioBearerSetupRequest (EpcEnbS1SapUser::DataRadioBearerSetupRequestParameters request)
{

}
  
void 
EpcTestRrc::DoPathSwitchRequestAcknowledge (EpcEnbS1SapUser::PathSwitchRequestAcknowledgeParameters params)
{

}


} // namespace ns3

