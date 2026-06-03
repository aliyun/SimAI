/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 *         (based on epc-sgw-pgw-application.cc)
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/mac48-address.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-header.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/inet-socket-address.h"
#include "ns3/epc-gtpu-header.h"
#include "ns3/epc-pgw-application.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EpcPgwApplication");


/////////////////////////
// UeInfo
/////////////////////////

EpcPgwApplication::UeInfo::UeInfo ()
{
  NS_LOG_FUNCTION (this);
}

void
EpcPgwApplication::UeInfo::AddBearer (uint8_t bearerId, uint32_t teid, Ptr<EpcTft> tft)
{
  NS_LOG_FUNCTION (this << (uint16_t) bearerId << teid << tft);
  m_teidByBearerIdMap[bearerId] = teid;
  return m_tftClassifier.Add (tft, teid);
}

void
EpcPgwApplication::UeInfo::RemoveBearer (uint8_t bearerId)
{
  NS_LOG_FUNCTION (this << (uint16_t) bearerId);
  std::map<uint8_t, uint32_t >::iterator it = m_teidByBearerIdMap.find (bearerId);
  m_tftClassifier.Delete (it->second); //delete tft
  m_teidByBearerIdMap.erase (bearerId);
}

uint32_t
EpcPgwApplication::UeInfo::Classify (Ptr<Packet> p, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << p);
  // we hardcode DOWNLINK direction since the PGW is expected to
  // classify only downlink packets (uplink packets will go to the
  // internet without any classification).
  return m_tftClassifier.Classify (p, EpcTft::DOWNLINK, protocolNumber);
}

Ipv4Address
EpcPgwApplication::UeInfo::GetSgwAddr ()
{
  return m_sgwAddr;
}

void
EpcPgwApplication::UeInfo::SetSgwAddr (Ipv4Address sgwAddr)
{
  m_sgwAddr = sgwAddr;
}

Ipv4Address 
EpcPgwApplication::UeInfo::GetUeAddr ()
{
  return m_ueAddr;
}

void
EpcPgwApplication::UeInfo::SetUeAddr (Ipv4Address ueAddr)
{
  m_ueAddr = ueAddr;
}

Ipv6Address 
EpcPgwApplication::UeInfo::GetUeAddr6 ()
{
  return m_ueAddr6;
}

void
EpcPgwApplication::UeInfo::SetUeAddr6 (Ipv6Address ueAddr)
{
  m_ueAddr6 = ueAddr;
}

/////////////////////////
// EpcPgwApplication
/////////////////////////

TypeId
EpcPgwApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::EpcPgwApplication")
    .SetParent<Object> ()
    .SetGroupName ("Lte")
    .AddTraceSource ("RxFromTun",
                     "Receive data packets from internet in Tunnel NetDevice",
                     MakeTraceSourceAccessor (&EpcPgwApplication::m_rxTunPktTrace),
                     "ns3::EpcPgwApplication::RxTracedCallback")
    .AddTraceSource ("RxFromS1u",
                     "Receive data packets from S5 Socket",
                     MakeTraceSourceAccessor (&EpcPgwApplication::m_rxS5PktTrace),
                     "ns3::EpcPgwApplication::RxTracedCallback")
    ;
  return tid;
}

void
EpcPgwApplication::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_s5uSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_s5uSocket = 0;
  m_s5cSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_s5cSocket = 0;
}

EpcPgwApplication::EpcPgwApplication (const Ptr<VirtualNetDevice> tunDevice, Ipv4Address s5Addr,
                                      const Ptr<Socket> s5uSocket, const Ptr<Socket> s5cSocket)
  : m_pgwS5Addr (s5Addr),
    m_s5uSocket (s5uSocket),
    m_s5cSocket (s5cSocket),
    m_tunDevice (tunDevice),
    m_gtpuUdpPort (2152), // fixed by the standard
    m_gtpcUdpPort (2123)  // fixed by the standard
{
  NS_LOG_FUNCTION (this << tunDevice << s5Addr << s5uSocket << s5cSocket);
  m_s5uSocket->SetRecvCallback (MakeCallback (&EpcPgwApplication::RecvFromS5uSocket, this));
  m_s5cSocket->SetRecvCallback (MakeCallback (&EpcPgwApplication::RecvFromS5cSocket, this));
}

EpcPgwApplication::~EpcPgwApplication ()
{
  NS_LOG_FUNCTION (this);
}

bool
EpcPgwApplication::RecvFromTunDevice (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << source << dest << protocolNumber << packet << packet->GetSize ());
  m_rxTunPktTrace (packet->Copy ());

  // get IP address of UE
  if (protocolNumber == Ipv4L3Protocol::PROT_NUMBER)
    {
      Ipv4Header ipv4Header;
      packet->PeekHeader (ipv4Header);
      Ipv4Address ueAddr = ipv4Header.GetDestination ();
      NS_LOG_LOGIC ("packet addressed to UE " << ueAddr);

      // find corresponding UeInfo address
      std::map<Ipv4Address, Ptr<UeInfo> >::iterator it = m_ueInfoByAddrMap.find (ueAddr);
      if (it == m_ueInfoByAddrMap.end ())
        {
          NS_LOG_WARN ("unknown UE address " << ueAddr);
        }
      else
        {
          Ipv4Address sgwAddr = it->second->GetSgwAddr ();
          uint32_t teid = it->second->Classify (packet, protocolNumber);
          if (teid == 0)
            {
              NS_LOG_WARN ("no matching bearer for this packet");
            }
          else
            {
              SendToS5uSocket (packet, sgwAddr, teid);
            }
        }
    }
  else if (protocolNumber == Ipv6L3Protocol::PROT_NUMBER)
    {
      Ipv6Header ipv6Header;
      packet->PeekHeader (ipv6Header);
      Ipv6Address ueAddr =  ipv6Header.GetDestination ();
      NS_LOG_LOGIC ("packet addressed to UE " << ueAddr);

      // find corresponding UeInfo address
      std::map<Ipv6Address, Ptr<UeInfo> >::iterator it = m_ueInfoByAddrMap6.find (ueAddr);
      if (it == m_ueInfoByAddrMap6.end ())
        {
          NS_LOG_WARN ("unknown UE address " << ueAddr);
        }
      else
        {
          Ipv4Address sgwAddr = it->second->GetSgwAddr ();
          uint32_t teid = it->second->Classify (packet, protocolNumber);
          if (teid == 0)
            {
              NS_LOG_WARN ("no matching bearer for this packet");
            }
          else
            {
              SendToS5uSocket (packet, sgwAddr, teid);
            }
        }
    }
  else
    {
      NS_ABORT_MSG ("Unknown IP type");
    }

  // there is no reason why we should notify the TUN
  // VirtualNetDevice that he failed to send the packet: if we receive
  // any bogus packet, it will just be silently discarded.
  const bool succeeded = true;
  return succeeded;
}

void
EpcPgwApplication::RecvFromS5uSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s5uSocket);
  Ptr<Packet> packet = socket->Recv ();
  m_rxS5PktTrace (packet->Copy ());

  GtpuHeader gtpu;
  packet->RemoveHeader (gtpu);
  uint32_t teid = gtpu.GetTeid ();

  SendToTunDevice (packet, teid);
}

void
EpcPgwApplication::RecvFromS5cSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  NS_ASSERT (socket == m_s5cSocket);
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
EpcPgwApplication::DoRecvCreateSessionRequest (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcCreateSessionRequestMessage msg;
  packet->RemoveHeader (msg);
  uint64_t imsi = msg.GetImsi ();
  uint16_t cellId = msg.GetUliEcgi ();
  NS_LOG_DEBUG ("cellId " << cellId << " IMSI " << imsi);

  std::map<uint64_t, Ptr<UeInfo> >::iterator ueit = m_ueInfoByImsiMap.find (imsi);
  NS_ASSERT_MSG (ueit != m_ueInfoByImsiMap.end (), "unknown IMSI " << imsi);
  ueit->second->SetSgwAddr (m_sgwS5Addr);

  GtpcHeader::Fteid_t sgwS5cFteid = msg.GetSenderCpFteid ();
  NS_ASSERT_MSG (sgwS5cFteid.interfaceType == GtpcHeader::S5_SGW_GTPC,
                 "Wrong interface type");

  GtpcCreateSessionResponseMessage msgOut;
  msgOut.SetTeid (sgwS5cFteid.teid);
  msgOut.SetCause (GtpcCreateSessionResponseMessage::REQUEST_ACCEPTED);

  GtpcHeader::Fteid_t pgwS5cFteid;
  pgwS5cFteid.interfaceType = GtpcHeader::S5_PGW_GTPC;
  pgwS5cFteid.teid = sgwS5cFteid.teid;
  pgwS5cFteid.addr = m_pgwS5Addr;
  msgOut.SetSenderCpFteid (pgwS5cFteid);

  std::list<GtpcCreateSessionRequestMessage::BearerContextToBeCreated> bearerContexts =
      msg.GetBearerContextsToBeCreated ();
  NS_LOG_DEBUG ("BearerContextsToBeCreated size = " << bearerContexts.size ());

  std::list<GtpcCreateSessionResponseMessage::BearerContextCreated> bearerContextsCreated;
  for (auto &bearerContext : bearerContexts)
    {
      uint32_t teid = bearerContext.sgwS5uFteid.teid;
      NS_LOG_DEBUG ("bearerId " << (uint16_t)bearerContext.epsBearerId <<
                    " SGW " << bearerContext.sgwS5uFteid.addr << " TEID " << teid);

      ueit->second->AddBearer (bearerContext.epsBearerId, teid, bearerContext.tft);

      GtpcCreateSessionResponseMessage::BearerContextCreated bearerContextOut;
      bearerContextOut.fteid.interfaceType = GtpcHeader::S5_PGW_GTPU;
      bearerContextOut.fteid.teid = teid;
      bearerContextOut.fteid.addr = m_pgwS5Addr;
      bearerContextOut.epsBearerId = bearerContext.epsBearerId;
      bearerContextOut.bearerLevelQos = bearerContext.bearerLevelQos;
      bearerContextOut.tft = bearerContext.tft;
      bearerContextsCreated.push_back (bearerContextOut);
    }

  NS_LOG_DEBUG ("BearerContextsCreated size = " << bearerContextsCreated.size ());
  msgOut.SetBearerContextsCreated (bearerContextsCreated);
  msgOut.SetTeid (sgwS5cFteid.teid);
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send CreateSessionResponse to SGW " << sgwS5cFteid.addr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (sgwS5cFteid.addr, m_gtpcUdpPort));
}

void
EpcPgwApplication::DoRecvModifyBearerRequest (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcModifyBearerRequestMessage msg;
  packet->RemoveHeader (msg);
  uint64_t imsi = msg.GetImsi ();
  uint16_t cellId = msg.GetUliEcgi ();
  NS_LOG_DEBUG ("cellId " << cellId << " IMSI " << imsi);

  std::map<uint64_t, Ptr<UeInfo> >::iterator ueit = m_ueInfoByImsiMap.find (imsi);
  NS_ASSERT_MSG (ueit != m_ueInfoByImsiMap.end (), "unknown IMSI " << imsi); 
  ueit->second->SetSgwAddr (m_sgwS5Addr);

  std::list<GtpcModifyBearerRequestMessage::BearerContextToBeModified> bearerContexts =
      msg.GetBearerContextsToBeModified ();
  NS_LOG_DEBUG ("BearerContextsToBeModified size = " << bearerContexts.size ());

  for (auto &bearerContext : bearerContexts)
    {
      Ipv4Address sgwAddr = bearerContext.fteid.addr;
      uint32_t teid = bearerContext.fteid.teid;
      NS_LOG_DEBUG ("bearerId " << (uint16_t)bearerContext.epsBearerId <<
                    " SGW " << sgwAddr << " TEID " << teid);
    }

  GtpcModifyBearerResponseMessage msgOut;
  msgOut.SetCause (GtpcIes::REQUEST_ACCEPTED);
  msgOut.SetTeid (imsi);
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send ModifyBearerResponse to SGW " << m_sgwS5Addr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_sgwS5Addr, m_gtpcUdpPort));
}

void
EpcPgwApplication::DoRecvDeleteBearerCommand (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcDeleteBearerCommandMessage msg;
  packet->RemoveHeader (msg);

  std::list<uint8_t> epsBearerIds;
  for (auto &bearerContext : msg.GetBearerContexts ())
    {
      NS_LOG_DEBUG ("ebid " << (uint16_t) bearerContext.m_epsBearerId);
      epsBearerIds.push_back (bearerContext.m_epsBearerId);
    }

  GtpcDeleteBearerRequestMessage msgOut;
  msgOut.SetEpsBearerIds (epsBearerIds);
  msgOut.SetTeid (msg.GetTeid ());
  msgOut.ComputeMessageLength ();

  Ptr<Packet> packetOut = Create <Packet> ();
  packetOut->AddHeader (msgOut);
  NS_LOG_DEBUG ("Send DeleteBearerRequest to SGW " << m_sgwS5Addr);
  m_s5cSocket->SendTo (packetOut, 0, InetSocketAddress (m_sgwS5Addr, m_gtpcUdpPort));
}

void
EpcPgwApplication::DoRecvDeleteBearerResponse (Ptr<Packet> packet)
{
  NS_LOG_FUNCTION (this);

  GtpcDeleteBearerResponseMessage msg;
  packet->RemoveHeader (msg);

  uint64_t imsi = msg.GetTeid ();
  std::map<uint64_t, Ptr<UeInfo> >::iterator ueit = m_ueInfoByImsiMap.find (imsi);
  NS_ASSERT_MSG (ueit != m_ueInfoByImsiMap.end (), "unknown IMSI " << imsi);

  for (auto &epsBearerId : msg.GetEpsBearerIds ())
    {
      // Remove de-activated bearer contexts from PGW side
      NS_LOG_INFO ("PGW removing bearer " << (uint16_t) epsBearerId << " of IMSI " << imsi);
      ueit->second->RemoveBearer (epsBearerId);
    }
}


void
EpcPgwApplication::SendToTunDevice (Ptr<Packet> packet, uint32_t teid)
{
  NS_LOG_FUNCTION (this << packet << teid);
  NS_LOG_LOGIC ("packet size: " << packet->GetSize () << " bytes");

  uint8_t ipType;
  packet->CopyData (&ipType, 1);
  ipType = (ipType>>4) & 0x0f;

  uint16_t protocol = 0;
  if (ipType == 0x04)
    {
      protocol = 0x0800;
    }
  else if (ipType == 0x06)
    {
      protocol = 0x86DD;
    }
  else
    {
      NS_ABORT_MSG ("Unknown IP type");
    }

  m_tunDevice->Receive (packet, protocol, m_tunDevice->GetAddress (), m_tunDevice->GetAddress (), NetDevice::PACKET_HOST);
}

void
EpcPgwApplication::SendToS5uSocket (Ptr<Packet> packet, Ipv4Address sgwAddr, uint32_t teid)
{
  NS_LOG_FUNCTION (this << packet << sgwAddr << teid);

  GtpuHeader gtpu;
  gtpu.SetTeid (teid);
  // From 3GPP TS 29.281 v10.0.0 Section 5.1
  // Length of the payload + the non obligatory GTP-U header
  gtpu.SetLength (packet->GetSize () + gtpu.GetSerializedSize () - 8);
  packet->AddHeader (gtpu);
  uint32_t flags = 0;
  m_s5uSocket->SendTo (packet, flags, InetSocketAddress (sgwAddr, m_gtpuUdpPort));
}


void
EpcPgwApplication::AddSgw (Ipv4Address sgwS5Addr)
{
    NS_LOG_FUNCTION (this << sgwS5Addr);
    m_sgwS5Addr = sgwS5Addr;
}

void
EpcPgwApplication::AddUe (uint64_t imsi)
{
  NS_LOG_FUNCTION (this << imsi);
  Ptr<UeInfo> ueInfo = Create<UeInfo> ();
  m_ueInfoByImsiMap[imsi] = ueInfo;
}

void
EpcPgwApplication::SetUeAddress (uint64_t imsi, Ipv4Address ueAddr)
{
  NS_LOG_FUNCTION (this << imsi << ueAddr);
  std::map<uint64_t, Ptr<UeInfo> >::iterator ueit = m_ueInfoByImsiMap.find (imsi);
  NS_ASSERT_MSG (ueit != m_ueInfoByImsiMap.end (), "unknown IMSI" << imsi); 
  ueit->second->SetUeAddr (ueAddr);
  m_ueInfoByAddrMap[ueAddr] = ueit->second;
}

void
EpcPgwApplication::SetUeAddress6 (uint64_t imsi, Ipv6Address ueAddr)
{
  NS_LOG_FUNCTION (this << imsi << ueAddr);
  std::map<uint64_t, Ptr<UeInfo> >::iterator ueit = m_ueInfoByImsiMap.find (imsi);
  NS_ASSERT_MSG (ueit != m_ueInfoByImsiMap.end (), "unknown IMSI " << imsi); 
  m_ueInfoByAddrMap6[ueAddr] = ueit->second;
  ueit->second->SetUeAddr6 (ueAddr);
}

}  // namespace ns3
