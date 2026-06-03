/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universita' di Firenze
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/packet-socket-address.h"
#include "ns3/packet-socket.h"
#include "ns3/packet-socket-factory.h"
#include "ns3/socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/abort.h"
#include "packet-socket-client.h"
#include <cstdlib>
#include <cstdio>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketSocketClient");

NS_OBJECT_ENSURE_REGISTERED (PacketSocketClient);

TypeId
PacketSocketClient::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketSocketClient")
    .SetParent<Application> ()
    .SetGroupName("Network")
    .AddConstructor<PacketSocketClient> ()
    .AddAttribute ("MaxPackets",
                   "The maximum number of packets the application will send (zero means infinite)",
                   UintegerValue (100),
                   MakeUintegerAccessor (&PacketSocketClient::m_maxPackets),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Interval",
                   "The time to wait between packets", TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&PacketSocketClient::m_interval),
                   MakeTimeChecker ())
    .AddAttribute ("PacketSize",
                   "Size of packets generated (bytes).",
                   UintegerValue (1024),
                   MakeUintegerAccessor (&PacketSocketClient::m_size),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Priority",
                   "Priority assigned to the packets generated.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&PacketSocketClient::SetPriority,
                                         &PacketSocketClient::GetPriority),
                   MakeUintegerChecker<uint8_t> ())
    .AddTraceSource ("Tx", "A packet has been sent",
                     MakeTraceSourceAccessor (&PacketSocketClient::m_txTrace),
                     "ns3::Packet::AddressTracedCallback")
  ;
  return tid;
}

PacketSocketClient::PacketSocketClient ()
{
  NS_LOG_FUNCTION (this);
  m_sent = 0;
  m_socket = 0;
  m_sendEvent = EventId ();
  m_peerAddressSet = false;
}

PacketSocketClient::~PacketSocketClient ()
{
  NS_LOG_FUNCTION (this);
}

void
PacketSocketClient::SetRemote (PacketSocketAddress addr)
{
  NS_LOG_FUNCTION (this << addr);
  m_peerAddress = addr;
  m_peerAddressSet = true;
}

void
PacketSocketClient::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Application::DoDispose ();
}

void
PacketSocketClient::SetPriority (uint8_t priority)
{
  m_priority = priority;
  if (m_socket)
    {
      m_socket->SetPriority (priority);
    }
}

uint8_t
PacketSocketClient::GetPriority (void) const
{
  return m_priority;
}

void
PacketSocketClient::StartApplication (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_peerAddressSet, "Peer address not set");

  if (m_socket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::PacketSocketFactory");
      m_socket = Socket::CreateSocket (GetNode (), tid);

      m_socket->Bind (m_peerAddress);
      m_socket->Connect (m_peerAddress);

      if (m_priority)
        {
          m_socket->SetPriority (m_priority);
        }
    }

  m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_sendEvent = Simulator::ScheduleNow (&PacketSocketClient::Send, this);
}

void
PacketSocketClient::StopApplication (void)
{
  NS_LOG_FUNCTION (this);
  Simulator::Cancel (m_sendEvent);
  m_socket->Close ();
}

void
PacketSocketClient::Send (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_sendEvent.IsExpired ());

  Ptr<Packet> p = Create<Packet> (m_size);

  std::stringstream peerAddressStringStream;
  peerAddressStringStream << PacketSocketAddress::ConvertFrom (m_peerAddress);

  if ((m_socket->Send (p)) >= 0)
    {
      m_txTrace (p, m_peerAddress);
      NS_LOG_INFO ("TraceDelay TX " << m_size << " bytes to "
                                    << peerAddressStringStream.str () << " Uid: "
                                    << p->GetUid () << " Time: "
                                    << (Simulator::Now ()).GetSeconds ());
    }
  else
    {
      NS_LOG_INFO ("Error while sending " << m_size << " bytes to "
                                          << peerAddressStringStream.str ());
    }
  m_sent++;

  if ((m_sent < m_maxPackets) || (m_maxPackets == 0))
    {
      m_sendEvent = Simulator::Schedule (m_interval, &PacketSocketClient::Send, this);
    }
}

} // Namespace ns3
