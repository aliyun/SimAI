/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017-2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#include "ns3/log.h"
#include "ns3/epc-gtpu-header.h"
#include "ns3/epc-sgw-application.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EpcSgwApplication");

NS_OBJECT_ENSURE_REGISTERED (EpcSgwApplication);

EpcSgwApplication::EpcSgwApplication (const Ptr<Socket> s1uSocket, Ipv4Address s5Addr,
                                      const Ptr<Socket> s5uSocket, const Ptr<Socket> s5cSocket)
  : m_s5Addr (s5Addr),
    m_s5uSocket (s5uSocket),
    m_s5cSocket (s5cSocket),
    m_s1uSocket (s1uSocket),
    m_gtpuUdpPort (2152), // fixed by the standard
    m_gtpcUdpPort (2123), // fixed by the standard
    m_teidCount (0)
{
  NS_LOG_FUNCTION (this << s1uSocket << s5Addr << s5uSocket << s5cSocket);
  m_s1uSocket->SetRecvCallback (MakeCallback (&EpcSgwApplication::RecvFromS1uSocket, this));
  m_s5uSocket->SetRecvCallback (MakeCallback (&EpcSgwApplication::RecvFromS5uSocket, this));
  m_s5cSocket->SetRecvCallback (MakeCallback (&EpcSgwApplication::RecvFromS5cSocket, this));
}

EpcSgwApplication::~EpcSgwApplication ()
{
  NS_LOG_FUNCTION (this);
}

void
EpcSgwApplication::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_s1uSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_s1uSocket = 0;
  m_s5uSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_s5uSocket = 0;
  m_s5cSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_s5cSocket = 0;
}

TypeId
EpcSgwApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcSgwApplication")
    .SetParent<Object> ()
    .SetGroupName("Lte");
  return tid;
}


void
EpcSgwApplication::AddMme (Ipv4Address mmeS11Addr, Ptr<Socket> s11Socket)
{
  NS_LOG_FUNCTION (this << mmeS11Addr << s11Socket);
  m_mmeS11Addr = mmeS11Addr;
  m_s11Socket = s11Socket;
  m_s11Socket->SetRecvCallback (MakeCallback (&EpcSgwApplication::RecvFromS11Socket, this));
}

void
EpcSgwApplication::AddPgw (Ipv4Address pgwAddr)
{
  NS_LOG_FUNCTION (this << pgwAddr);
  m_pgwAddr = pgwAddr;
}

void
EpcSgwApplication::AddEnb (uint16_t cellId, Ipv4Address enbAddr, Ipv4Address sgwAddr)
{
  NS_LOG_FUNCTION (this << cellId << enbAddr << sgwAddr);
  EnbInfo enbInfo;
  enbInfo.enbAddr = enbAddr;
  enbInfo.sgwAddr = sgwAddr;
  m_enbInfoByCellId[cellId] = enbInfo;
}


void
EpcSgwApplication::RecvFromS11Socket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s11Socket);
  Ptr<Packet> packet = socket->Recv ();
  GtpcHeader header;
  packet->PeekHeader (header);
  uint16_t msgType = header.GetMessageType ();

  switch (msgType)
    {
    case GtpcHeader::CreateSessionRequest:
      DoRecvCreateSessionRequest (packet);
      break;

    case GtpcHeader::ModifyBearerRequest:
      DoRecvModifyBearerRequest (packet);
      break;

    case GtpcHeader::DeleteBearerCommand:
      DoRecvDeleteBearerCommand (packet);
      break;

    case GtpcHeader::DeleteBearerResponse:
      DoRecvDeleteBearerResponse (packet);
      break;

    default:
      NS_FATAL_ERROR ("GTP-C message not supported");
      break;
    }
}


void
EpcSgwApplication::RecvFromS5uSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s5uSocket);
  Ptr<Packet> packet = socket->Recv ();
  GtpuHeader gtpu;
  packet->RemoveHeader (gtpu);
  uint32_t teid = gtpu.GetTeid ();

  Ipv4Address enbAddr = m_enbByTeidMap[teid];
  NS_LOG_DEBUG ("eNB " << enbAddr << " TEID " << teid);
  SendToS1uSocket (packet, enbAddr, teid);
}

void
EpcSgwApplication::RecvFromS5cSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s5cSocket);
  Ptr<Packet> packet = socket->Recv ();
  GtpcHeader header;
  packet->PeekHeader (header);
  uint16_t msgType = header.GetMessageType ();

  switch (msgType)
    {
    case GtpcHeader::CreateSessionResponse:
      DoRecvCreateSessionResponse (packet);
      break;

    case GtpcHeader::ModifyBearerResponse:
      DoRecvModifyBearerResponse (packet);
      break;

    case GtpcHeader::DeleteBearerRequest:
      DoRecvDeleteBearerRequest (packet);
      break;

    default:
      NS_FATAL_ERROR ("GTP-C message not supported");
      break;
    }
}

void
EpcSgwApplication::RecvFromS1uSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s1uSocket);
  Ptr<Packet> packet = socket->Recv ();
  GtpuHeader gtpu;
  packet->RemoveHeader (gtpu);
  uint32_t teid = gtpu.GetTeid ();

  SendToS5uSocket (packet, m_pgwAddr, teid);
}

void
EpcSgwApplication::SendToS1uSocket (Ptr<Packet> packet, Ipv4Address enbAddr, uint32_t teid)
{
  NS_LOG_FUNCTION (this << packet << enbAddr << teid);

  GtpuHeader gtpu;
  gtpu.SetTeid (teid);
  // From 3GPP TS 29.281 v10.0.0 Section 5.1
  // Length of the payload + the non obligatory GTP-U header
  gtpu.SetLength (packet->GetSize () + gtpu.GetSerializedSize () - 8);
  packet->AddHeader (gtpu);
  m_s1uSocket->SendTo (packet, 0, InetSocketAddress (enbAddr, m_gtpuUdpPort));
}

void
EpcSgwApplication::SendToS5uSocket (Ptr<Packet> packet, Ipv4Address pgwAddr, uint32_t teid)
{
  NS_LOG_FUNCTION (this << packet << pgwAddr << teid);

  GtpuHeader gtpu;
  gtpu.SetTeid (teid);
  // From 3GPP TS 29.281 v10.0.0 Section 5.1
  // Length of the payload + the non obligatory GTP-U header
  gtpu.SetLength (packet->GetSize () + gtpu.GetSerializedSize () - 8);
  packet->AddHeader (gtpu);
  m_s5uSocket->SendTo (packet, 0, InetSocketAddress (pgwAddr, m_gtpuUdpPort));
}


///////////////////////////////////
// Process messages from the MME
///////////////////////////////////

void
EpcSgwApplication::DoRecvCreateSessionRequest (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcCreateSessionRequestMessage msg;
  packet->RemoveHeader (msg);
  uint64_t imsi = msg.GetImsi ();
  uint16_t cellId = msg.GetUliEcgi ();
  NS_LOG_DEBUG ("cellId " << cellId << " IMSI " << imsi);

  std::map<uint16_t, EnbInfo>::iterator enbit = m_enbInfoByCellId.find (cellId);
  NS_ASSERT_MSG (enbit != m_enbInfoByCellId.end (), "unknown CellId " << cellId); 
  Ipv4Address enbAddr = enbit->second.enbAddr;
  NS_LOG_DEBUG ("eNB " << enbAddr);

  GtpcHeader::Fteid_t mmeS11Fteid = msg.GetSenderCpFteid ();
  NS_ASSERT_MSG (mmeS11Fteid.interfaceType == GtpcHeader::S11_MME_GTPC,
                 "wrong interface type");

  GtpcCreateSessionRequestMessage msgOut;
  msgOut.SetImsi (imsi);
  msgOut.SetUliEcgi (cellId);

  GtpcHeader::Fteid_t sgwS5cFteid;
  sgwS5cFteid.interfaceType = GtpcHeader::S5_SGW_GTPC;
  sgwS5cFteid.teid = imsi;
  m_mmeS11FteidBySgwS5cTeid[sgwS5cFteid.teid] = mmeS11Fteid;
  sgwS5cFteid.addr = m_s5Addr;
  msgOut.SetSenderCpFteid (sgwS5cFteid); // S5 SGW GTP-C TEID

  std::list<GtpcCreateSessionRequestMessage::BearerContextToBeCreated> bearerContexts =
      msg.GetBearerContextsToBeCreated ();
  NS_LOG_DEBUG ("BearerContextToBeCreated size = " << bearerContexts.size ());
  std::list<GtpcCreateSessionRequestMessage::BearerContextToBeCreated> bearerContextsOut;
  for (auto &bearerContext : bearerContexts)
    {
      // simple sanity check. If you ever need more than 4M teids
      // throughout your simulation, you'll need to implement a smarter teid
      // management algorithm.
      NS_ABORT_IF (m_teidCount == 0xFFFFFFFF);
      uint32_t teid = ++m_teidCount;

      NS_LOG_DEBUG ("  TEID " << teid);
      m_enbByTeidMap[teid] = enbAddr;

      GtpcCreateSessionRequestMessage::BearerContextToBeCreated bearerContextOut;
      bearerContextOut.sgwS5uFteid.interfaceType = GtpcHeader::S5_SGW_GTPU;
      bearerContextOut.sgwS5uFteid.teid = teid; // S5U SGW FTEID
      bearerContextOut.sgwS5uFteid.addr = enbit->second.sgwAddr;
      bearerContextOut.epsBearerId = bearerContext.epsBearerId;
      bearerContextOut.bearerLevelQos = bearerContext.bearerLevelQos;
      bearerContextOut.tft = bearerContext.tft;
      bearerContextsOut.push_back (bearerContextOut);
    }

  msgOut.SetBearerContextsToBeCreated (bearerContextsOut);

  msgOut.SetTeid (0);
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send CreateSessionRequest to PGW " << m_pgwAddr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_pgwAddr, m_gtpcUdpPort));
}

void
EpcSgwApplication::DoRecvModifyBearerRequest (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcModifyBearerRequestMessage msg;
  packet->RemoveHeader (msg);
  uint64_t imsi = msg.GetImsi ();
  uint16_t cellId = msg.GetUliEcgi ();
  NS_LOG_DEBUG ("cellId " << cellId << " IMSI " << imsi);

  std::map<uint16_t, EnbInfo>::iterator enbit = m_enbInfoByCellId.find (cellId);
  NS_ASSERT_MSG (enbit != m_enbInfoByCellId.end (), "unknown CellId " << cellId); 
  Ipv4Address enbAddr = enbit->second.enbAddr;
  NS_LOG_DEBUG ("eNB " << enbAddr);

  GtpcModifyBearerRequestMessage msgOut;
  msgOut.SetImsi (imsi);
  msgOut.SetUliEcgi (cellId);

  std::list<GtpcModifyBearerRequestMessage::BearerContextToBeModified> bearerContextsOut;
  std::list<GtpcModifyBearerRequestMessage::BearerContextToBeModified> bearerContexts
      = msg.GetBearerContextsToBeModified ();
  NS_LOG_DEBUG ("BearerContextsToBeModified size = " << bearerContexts.size ());
  for (auto &bearerContext : bearerContexts)
    {
      NS_ASSERT_MSG (bearerContext.fteid.interfaceType == GtpcHeader::S1U_ENB_GTPU,
                     "Wrong FTEID in ModifyBearerRequest msg");
      uint32_t teid = bearerContext.fteid.teid;
      Ipv4Address enbAddr = bearerContext.fteid.addr;
      NS_LOG_DEBUG ("bearerId " << (uint16_t)bearerContext.epsBearerId <<
                    " TEID " << teid);
      std::map<uint32_t, Ipv4Address>::iterator addrit = m_enbByTeidMap.find (teid);
      NS_ASSERT_MSG (addrit != m_enbByTeidMap.end (), "unknown TEID " << teid);
      addrit->second = enbAddr;
      GtpcModifyBearerRequestMessage::BearerContextToBeModified bearerContextOut;
      bearerContextOut.epsBearerId = bearerContext.epsBearerId;
      bearerContextOut.fteid.interfaceType = GtpcHeader::S5_SGW_GTPU;
      bearerContextOut.fteid.addr = m_s5Addr;
      bearerContextOut.fteid.teid = bearerContext.fteid.teid;

      bearerContextsOut.push_back (bearerContextOut);
    }

  msgOut.SetTeid (imsi);
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send ModifyBearerRequest to PGW " << m_pgwAddr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_pgwAddr, m_gtpcUdpPort));
}

void
EpcSgwApplication::DoRecvDeleteBearerCommand (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcDeleteBearerCommandMessage msg;
  packet->RemoveHeader (msg);

  std::list<GtpcDeleteBearerCommandMessage::BearerContext> bearerContextsOut;
  for (auto &bearerContext : msg.GetBearerContexts ())
    {
      NS_LOG_DEBUG ("ebid " << (uint16_t) bearerContext.m_epsBearerId);
      GtpcDeleteBearerCommandMessage::BearerContext bearerContextOut;
      bearerContextOut.m_epsBearerId = bearerContext.m_epsBearerId;
      bearerContextsOut.push_back (bearerContextOut);
    }

  GtpcDeleteBearerCommandMessage msgOut;
  msgOut.SetBearerContexts (bearerContextsOut);
  msgOut.SetTeid (msg.GetTeid ());
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send DeleteBearerCommand to PGW " << m_pgwAddr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_pgwAddr, m_gtpcUdpPort));
}

void
EpcSgwApplication::DoRecvDeleteBearerResponse (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcDeleteBearerResponseMessage msg;
  packet->RemoveHeader (msg);
  GtpcDeleteBearerResponseMessage msgOut;
  msgOut.SetEpsBearerIds (msg.GetEpsBearerIds ());
  msgOut.SetTeid (msg.GetTeid ());
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send DeleteBearerResponse to PGW " << m_pgwAddr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_pgwAddr, m_gtpcUdpPort));
}


////////////////////////////////////////////
// Process messages received from the PGW
////////////////////////////////////////////

void
EpcSgwApplication::DoRecvCreateSessionResponse (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcCreateSessionResponseMessage msg;
  packet->RemoveHeader (msg);

  GtpcHeader::Fteid_t pgwS5cFteid = msg.GetSenderCpFteid ();
  NS_ASSERT_MSG (pgwS5cFteid.interfaceType == GtpcHeader::S5_PGW_GTPC,
                 "wrong interface type");

  GtpcCreateSessionResponseMessage msgOut;
  msgOut.SetCause (GtpcCreateSessionResponseMessage::REQUEST_ACCEPTED);

  uint32_t teid = msg.GetTeid ();
  GtpcHeader::Fteid_t mmeS11Fteid = m_mmeS11FteidBySgwS5cTeid[teid];

  std::list<GtpcCreateSessionResponseMessage::BearerContextCreated> bearerContexts =
      msg.GetBearerContextsCreated ();
  NS_LOG_DEBUG ("BearerContextsCreated size = " << bearerContexts.size ());
  std::list<GtpcCreateSessionResponseMessage::BearerContextCreated> bearerContextsOut;
  for (auto &bearerContext : bearerContexts)
    {
      GtpcCreateSessionResponseMessage::BearerContextCreated bearerContextOut;
      bearerContextOut.fteid.interfaceType = GtpcHeader::S5_SGW_GTPU;
      bearerContextOut.fteid.teid = bearerContext.fteid.teid;
      bearerContextOut.fteid.addr = m_s5Addr;
      bearerContextOut.epsBearerId =  bearerContext.epsBearerId;
      bearerContextOut.bearerLevelQos = bearerContext.bearerLevelQos;
      bearerContextOut.tft = bearerContext.tft;
      bearerContextsOut.push_back (bearerContext);
    }
    msgOut.SetBearerContextsCreated (bearerContextsOut);

    msgOut.SetTeid (mmeS11Fteid.teid);
    msgOut.ComputeMessageLength ();

    Ptr<Packet> packetOut = Create <Packet> ();
    packetOut->AddHeader (msgOut);
    NS_LOG_DEBUG ("Send CreateSessionResponse to MME " << mmeS11Fteid.addr);
    m_s11Socket->SendTo (packetOut, 0, InetSocketAddress (mmeS11Fteid.addr, m_gtpcUdpPort));
}

void
EpcSgwApplication::DoRecvModifyBearerResponse (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcModifyBearerResponseMessage msg;
  packet->RemoveHeader (msg);

  GtpcModifyBearerResponseMessage msgOut;
  msgOut.SetCause (GtpcIes::REQUEST_ACCEPTED);
  msgOut.SetTeid (msg.GetTeid ());
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send ModifyBearerResponse to MME " << m_mmeS11Addr);
  m_s11Socket->SendTo (packetOut, 0, InetSocketAddress (m_mmeS11Addr, m_gtpcUdpPort));
}

void
EpcSgwApplication::DoRecvDeleteBearerRequest (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcDeleteBearerRequestMessage msg;
  packet->RemoveHeader (msg);

  GtpcDeleteBearerRequestMessage msgOut;
  msgOut.SetEpsBearerIds (msg.GetEpsBearerIds ());
  msgOut.SetTeid (msg.GetTeid ());
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send DeleteBearerRequest to MME " << m_mmeS11Addr);
  m_s11Socket->SendTo (packetOut, 0, InetSocketAddress (m_mmeS11Addr, m_gtpcUdpPort));
}

}  // namespace ns3
