/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDCAST
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
 * Author: Amine Ismail <amine.ismail@sophia.inria.fr>
 *                      <amine.ismail@udcast.com>
 */
#include "rdma-client.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-end-point.h"
#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/packet.h"
#include "ns3/qbb-net-device.h"
#include "ns3/random-variable.h"
#include "ns3/seq-ts-header.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/socket.h"
#include "ns3/uinteger.h"
#include <ns3/rdma-driver.h>
#include <stdio.h>
#include <stdlib.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RdmaClient");
NS_OBJECT_ENSURE_REGISTERED(RdmaClient);

TypeId RdmaClient::GetTypeId(void) {
  static TypeId tid =
      TypeId("ns3::RdmaClient")
          .SetParent<Application>()
          .AddConstructor<RdmaClient>()
          .AddAttribute("WriteSize", "The number of bytes to write",
                        UintegerValue(10000),
                        MakeUintegerAccessor(&RdmaClient::m_size),
                        MakeUintegerChecker<uint64_t>())
          .AddAttribute("SourceIP", "Source IP", Ipv4AddressValue("0.0.0.0"),
                        MakeIpv4AddressAccessor(&RdmaClient::m_sip),
                        MakeIpv4AddressChecker())
          .AddAttribute("DestIP", "Dest IP", Ipv4AddressValue("0.0.0.0"),
                        MakeIpv4AddressAccessor(&RdmaClient::m_dip),
                        MakeIpv4AddressChecker())
          .AddAttribute("SourcePort", "Source Port", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::m_sport),
                        MakeUintegerChecker<uint16_t>())
          .AddAttribute("DestPort", "Dest Port", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::m_dport),
                        MakeUintegerChecker<uint16_t>())
          .AddAttribute("PriorityGroup", "The priority group of this flow",
                        UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::m_pg),
                        MakeUintegerChecker<uint16_t>())
          .AddAttribute("Window", "Bound of on-the-fly packets",
                        UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::m_win),
                        MakeUintegerChecker<uint32_t>())
          .AddAttribute("BaseRtt", "Base Rtt", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::m_baseRtt),
                        MakeUintegerChecker<uint64_t>())
          .AddAttribute("Tag", "Tag", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::tag),
                        MakeUintegerChecker<uint64_t>())
          .AddAttribute("Src", "Src", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::src),
                        MakeUintegerChecker<uint64_t>())
          .AddAttribute("Dest", "Dest", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::dest),
                        MakeUintegerChecker<uint64_t>())
          .AddAttribute("NVLS_enable", "NVLS enable info", UintegerValue(0),
                        MakeUintegerAccessor(&RdmaClient::nvls_enable),
                        MakeUintegerChecker<uint32_t>());
  return tid;
}

RdmaClient::RdmaClient() { NS_LOG_FUNCTION_NOARGS(); }

RdmaClient::~RdmaClient() { NS_LOG_FUNCTION_NOARGS(); }

void RdmaClient::SetRemote(Ipv4Address ip, uint16_t port) {
  m_dip = ip;
  m_dport = port;
}

void RdmaClient::SetLocal(Ipv4Address ip, uint16_t port) {
  m_sip = ip;
  m_sport = port;
}

void RdmaClient::SetPG(uint16_t pg) { m_pg = pg; }

void RdmaClient::SetSize(uint64_t size) { m_size = size; }

void RdmaClient::Sent() {
}

void RdmaClient::Finish() {
}

void RdmaClient::SetFn(void (*msg_handler)(void *fun_arg), void *fun_arg) {
  msg_handler = msg_handler;
  fun_arg = fun_arg;
}

void RdmaClient::DoDispose(void) {
  NS_LOG_FUNCTION_NOARGS();
  Application::DoDispose();
}

void RdmaClient::StartApplication(void) {
  NS_LOG_FUNCTION_NOARGS();
  // get RDMA driver and add up queue pair
  Ptr<Node> node = GetNode();
  Ptr<RdmaDriver> rdma = node->GetObject<RdmaDriver>();
  // setup NVLS
  if(nvls_enable) rdma->EnbaleNVLS();
  else rdma->DisableNVLS();
  rdma->AddQueuePair(src, dest, tag, m_size, m_pg, m_sip, m_dip, m_sport,
                     m_dport, m_win, m_baseRtt,
                     MakeCallback(&RdmaClient::Finish, this),
                     MakeCallback(&RdmaClient::Sent, this));
}

void RdmaClient::StopApplication() {
  NS_LOG_FUNCTION_NOARGS();
  // TODO stop the queue pair
}

} // Namespace ns3
