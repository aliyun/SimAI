/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 UPB
 * Copyright (c) 2017 NITK Surathkal
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
 * Author: Radu Lupu <rlupu@elcom.pub.ro>
 *         Ankit Deepak <adadeepak8@gmail.com>
 *         Deepti Rajagopal <deeptir96@gmail.com>
 *
 *
 */

#include "ns3/ipv4.h"
#include "ns3/log.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/random-variable-stream.h"
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "dhcp-client.h"
#include "dhcp-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DhcpClient");
NS_OBJECT_ENSURE_REGISTERED (DhcpClient);

TypeId
DhcpClient::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DhcpClient")
    .SetParent<Application> ()
    .AddConstructor<DhcpClient> ()
    .SetGroupName ("Internet-Apps")
    .AddAttribute ("RTRS", "Time for retransmission of Discover message",
                   TimeValue (Seconds (5)),
                   MakeTimeAccessor (&DhcpClient::m_rtrs),
                   MakeTimeChecker ())
    .AddAttribute ("Collect", "Time for which offer collection starts",
                   TimeValue (Seconds (5)),
                   MakeTimeAccessor (&DhcpClient::m_collect),
                   MakeTimeChecker ())
    .AddAttribute ("ReRequest", "Time after which request will be resent to next server",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&DhcpClient::m_nextoffer),
                   MakeTimeChecker ())
    .AddAttribute ("Transactions",
                   "The possible value of transaction numbers ",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1000000.0]"),
                   MakePointerAccessor (&DhcpClient::m_ran),
                   MakePointerChecker<RandomVariableStream> ())
    .AddTraceSource ("NewLease",
                     "Get a NewLease",
                     MakeTraceSourceAccessor (&DhcpClient::m_newLease),
                     "ns3::Ipv4Address::TracedCallback")
    .AddTraceSource ("ExpireLease",
                     "A lease expires",
                     MakeTraceSourceAccessor (&DhcpClient::m_expiry),
                     "ns3::Ipv4Address::TracedCallback");
  return tid;
}

DhcpClient::DhcpClient ()
{
  NS_LOG_FUNCTION (this);
  m_server = Ipv4Address::GetAny ();
  m_socket = 0;
  m_refreshEvent = EventId ();
  m_requestEvent = EventId ();
  m_discoverEvent = EventId ();
  m_rebindEvent = EventId ();
  m_nextOfferEvent = EventId ();
  m_timeout = EventId ();
  m_collectEvent = EventId ();
  m_firstBoot = true;
}

DhcpClient::DhcpClient (Ptr<NetDevice> netDevice)
{
  NS_LOG_FUNCTION (this << netDevice);
  m_device = netDevice;
  m_server = Ipv4Address::GetAny ();
  m_socket = 0;
  m_refreshEvent = EventId ();
  m_requestEvent = EventId ();
  m_discoverEvent = EventId ();
  m_rebindEvent = EventId ();
  m_nextOfferEvent = EventId ();
  m_timeout = EventId ();
  m_collectEvent = EventId ();
  m_firstBoot = true;
}

DhcpClient::~DhcpClient ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<NetDevice> DhcpClient::GetDhcpClientNetDevice (void)
{
  return m_device;
}


void DhcpClient::SetDhcpClientNetDevice (Ptr<NetDevice> netDevice)
{
  m_device = netDevice;
}

Ipv4Address DhcpClient::GetDhcpServer (void)
{
  return m_server;
}

void
DhcpClient::DoDispose (void)
{
  NS_LOG_FUNCTION (this);

  m_device = 0;

  // Stop all the timers
  m_refreshEvent.Cancel ();
  m_requestEvent.Cancel ();
  m_discoverEvent.Cancel ();
  m_rebindEvent.Cancel ();
  m_nextOfferEvent.Cancel ();
  m_timeout.Cancel ();
  m_collectEvent.Cancel ();

  Application::DoDispose ();
}

int64_t
DhcpClient::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_ran->SetStream (stream);
  return 1;
}

void
DhcpClient::StartApplication (void)
{
  NS_LOG_FUNCTION (this);

  m_remoteAddress = Ipv4Address ("255.255.255.255");
  m_myAddress = Ipv4Address ("0.0.0.0");
  m_gateway = Ipv4Address ("0.0.0.0");
  Ptr<Ipv4> ipv4 = GetNode ()->GetObject<Ipv4> ();
  uint32_t ifIndex = ipv4->GetInterfaceForDevice (m_device);

  // We need to cleanup the type from the stored chaddr, or later we'll fail to compare it.
  // Moreover, the length is always 16, because chaddr is 16 bytes.
  Address myAddress = m_device->GetAddress ();
  NS_LOG_INFO ("My address is " << myAddress);
  uint8_t addr[Address::MAX_SIZE];
  std::memset (addr, 0, Address::MAX_SIZE);
  uint32_t len = myAddress.CopyTo (addr);
  NS_ASSERT_MSG (len <= 16, "DHCP client can not handle a chaddr larger than 16 bytes");
  m_chaddr.CopyFrom (addr, 16);
  NS_LOG_INFO ("My m_chaddr is " << m_chaddr);

  bool found = false;
  for (uint32_t i = 0; i < ipv4->GetNAddresses (ifIndex); i++)
    {
      if (ipv4->GetAddress (ifIndex, i).GetLocal () == m_myAddress)
        {
          found = true;
        }
    }
  if (!found)
    {
      ipv4->AddAddress (ifIndex, Ipv4InterfaceAddress (Ipv4Address ("0.0.0.0"),Ipv4Mask ("/0")));
    }
  if (m_socket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      m_socket = Socket::CreateSocket (GetNode (), tid);
      InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 68);
      m_socket->SetAllowBroadcast (true);
      m_socket->BindToNetDevice (m_device);
      m_socket->Bind (local);
    }
  m_socket->SetRecvCallback (MakeCallback (&DhcpClient::NetHandler, this));

  if (m_firstBoot)
    {
      m_device->AddLinkChangeCallback (MakeCallback (&DhcpClient::LinkStateHandler, this));
      m_firstBoot = false;
    }
  Boot ();

}

void
DhcpClient::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  // Stop all the timers
  m_refreshEvent.Cancel ();
  m_requestEvent.Cancel ();
  m_discoverEvent.Cancel ();
  m_rebindEvent.Cancel ();
  m_nextOfferEvent.Cancel ();
  m_timeout.Cancel ();
  m_collectEvent.Cancel ();

  Ptr<Ipv4> ipv4 = GetNode ()->GetObject<Ipv4> ();

  int32_t ifIndex = ipv4->GetInterfaceForDevice (m_device);
  for (uint32_t i = 0; i < ipv4->GetNAddresses (ifIndex); i++)
    {
      if (ipv4->GetAddress (ifIndex,i).GetLocal () == m_myAddress)
        {
          ipv4->RemoveAddress (ifIndex, i);
          break;
        }
    }

  m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
  m_socket->Close ();
}

void DhcpClient::LinkStateHandler (void)
{
  NS_LOG_FUNCTION (this);

  if (m_device->IsLinkUp ())
    {
      NS_LOG_INFO ("Link up at " << Simulator::Now ().As (Time::S));
      m_socket->SetRecvCallback (MakeCallback (&DhcpClient::NetHandler, this));
      StartApplication ();
    }
  else
    {
      NS_LOG_INFO ("Link down at " << Simulator::Now ().As (Time::S)); //reinitialization

      // Stop all the timers
      m_refreshEvent.Cancel ();
      m_requestEvent.Cancel ();
      m_discoverEvent.Cancel ();
      m_rebindEvent.Cancel ();
      m_nextOfferEvent.Cancel ();
      m_timeout.Cancel ();
      m_collectEvent.Cancel ();

      m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());  //stop receiving on this socket !!!

      Ptr<Ipv4> ipv4MN = GetNode ()->GetObject<Ipv4> ();
      int32_t ifIndex = ipv4MN->GetInterfaceForDevice (m_device);

      for (uint32_t i = 0; i < ipv4MN->GetNAddresses (ifIndex); i++)
        {
          if (ipv4MN->GetAddress (ifIndex,i).GetLocal () == m_myAddress)
            {
              ipv4MN->RemoveAddress (ifIndex, i);
              break;
            }
        }

      Ipv4StaticRoutingHelper ipv4RoutingHelper;
      Ptr<Ipv4StaticRouting> staticRouting = ipv4RoutingHelper.GetStaticRouting (ipv4MN);
      uint32_t i;
      for (i = 0; i < staticRouting->GetNRoutes (); i++)
        {
          if (staticRouting->GetRoute (i).GetGateway () == m_gateway)
            {
              staticRouting->RemoveRoute (i);
              break;
            }
        }

      m_state = 0;
      m_myAddress = Ipv4Address ("0.0.0.0");
      m_gateway = Ipv4Address ("0.0.0.0");
    }
}

void DhcpClient::NetHandler (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  Address from;
  Ptr<Packet> packet = m_socket->RecvFrom (from);
  DhcpHeader header;
  if (packet->RemoveHeader (header) == 0)
    {
      return;
    }
  if (header.GetChaddr () != m_chaddr)
    {
      return;
    }
  if (m_state == WAIT_OFFER && header.GetType () == DhcpHeader::DHCPOFFER)
    {
      OfferHandler (header);
    }
  if (m_state == WAIT_ACK && header.GetType () == DhcpHeader::DHCPACK)
    {
      m_nextOfferEvent.Cancel ();
      AcceptAck (header,from);
    }
  if (m_state == WAIT_ACK && header.GetType () == DhcpHeader::DHCPNACK)
    {
      m_nextOfferEvent.Cancel ();
      Boot ();
    }
}

void DhcpClient::Boot (void)
{
  NS_LOG_FUNCTION (this);

  DhcpHeader header;
  Ptr<Packet> packet;
  packet = Create<Packet> ();
  header.ResetOpt ();
  m_tran = (uint32_t) (m_ran->GetValue ());
  header.SetTran (m_tran);
  header.SetType (DhcpHeader::DHCPDISCOVER);
  header.SetTime ();
  header.SetChaddr (m_chaddr);
  packet->AddHeader (header);

  if ((m_socket->SendTo (packet, 0, InetSocketAddress (Ipv4Address ("255.255.255.255"), DHCP_PEER_PORT))) >= 0)
    {
      NS_LOG_INFO ("DHCP DISCOVER sent" );
    }
  else
    {
      NS_LOG_INFO ("Error while sending DHCP DISCOVER to " << m_remoteAddress);
    }
  m_state = WAIT_OFFER;
  m_offered = false;
  m_discoverEvent = Simulator::Schedule (m_rtrs, &DhcpClient::Boot, this);
}

void DhcpClient::OfferHandler (DhcpHeader header)
{
  NS_LOG_FUNCTION (this << header);

  m_offerList.push_back (header);
  if (m_offered == false)
    {
      m_discoverEvent.Cancel ();
      m_offered = true;
      m_collectEvent = Simulator::Schedule (m_collect, &DhcpClient::Select, this);
    }
}

void DhcpClient::Select (void)
{
  NS_LOG_FUNCTION (this);

  if (m_offerList.empty ())
    {
      Boot ();
      return;
    }

  DhcpHeader header = m_offerList.front ();
  m_offerList.pop_front ();
  m_lease = Time (Seconds (header.GetLease ()));
  m_renew = Time (Seconds (header.GetRenew ()));
  m_rebind = Time (Seconds (header.GetRebind ()));
  m_offeredAddress = header.GetYiaddr ();
  m_myMask = Ipv4Mask (header.GetMask ());
  m_server = header.GetDhcps ();
  m_gateway = header.GetRouter ();
  m_offerList.clear ();
  m_offered = false;
  Request ();
}

void DhcpClient::Request (void)
{
  NS_LOG_FUNCTION (this);

  DhcpHeader header;
  Ptr<Packet> packet;
  if (m_state != REFRESH_LEASE)
    {
      packet = Create<Packet> ();
      header.ResetOpt ();
      header.SetType (DhcpHeader::DHCPREQ);
      header.SetTime ();
      header.SetTran (m_tran);
      header.SetReq (m_offeredAddress);
      header.SetChaddr (m_chaddr);
      packet->AddHeader (header);
      m_socket->SendTo (packet, 0, InetSocketAddress (Ipv4Address ("255.255.255.255"), DHCP_PEER_PORT));
      m_state = WAIT_ACK;
      m_nextOfferEvent = Simulator::Schedule (m_nextoffer, &DhcpClient::Select, this);
    }
  else
    {
      uint32_t addr = m_myAddress.Get ();
      packet = Create<Packet> ((uint8_t*)&addr, sizeof(addr));
      header.ResetOpt ();
      m_tran = (uint32_t) (m_ran->GetValue ());
      header.SetTran (m_tran);
      header.SetTime ();
      header.SetType (DhcpHeader::DHCPREQ);
      header.SetReq (m_myAddress);
      m_offeredAddress = m_myAddress;
      header.SetChaddr (m_chaddr);
      packet->AddHeader (header);
      if ((m_socket->SendTo (packet, 0, InetSocketAddress (m_remoteAddress, DHCP_PEER_PORT))) >= 0)
        {
          NS_LOG_INFO ("DHCP REQUEST sent");
        }
      else
        {
          NS_LOG_INFO ("Error while sending DHCP REQ to " << m_remoteAddress);
        }
      m_state = WAIT_ACK;
    }
}

void DhcpClient::AcceptAck (DhcpHeader header, Address from)
{
  NS_LOG_FUNCTION (this << header << from);

  m_rebindEvent.Cancel ();
  m_refreshEvent.Cancel ();
  m_timeout.Cancel ();

  NS_LOG_INFO ("DHCP ACK received");
  Ptr<Ipv4> ipv4 = GetNode ()->GetObject<Ipv4> ();
  int32_t ifIndex = ipv4->GetInterfaceForDevice (m_device);

  if (m_myAddress != m_offeredAddress)
    {
      for (uint32_t i = 0; i < ipv4->GetNAddresses (ifIndex); i++)
        {
          if (ipv4->GetAddress (ifIndex,i).GetLocal () == m_myAddress)
            {
              NS_LOG_LOGIC ("Got a new address (" << m_offeredAddress << "), removing old one: " << m_myAddress);
              ipv4->RemoveAddress (ifIndex, i);
              break;
            }
        }

      ipv4->AddAddress (ifIndex, Ipv4InterfaceAddress (m_offeredAddress, m_myMask));
      ipv4->SetUp (ifIndex);
    }

  InetSocketAddress remote = InetSocketAddress (InetSocketAddress::ConvertFrom (from).GetIpv4 (), DHCP_PEER_PORT);
  m_socket->Connect (remote);
  if (m_myAddress != m_offeredAddress)
    {
      m_newLease (m_offeredAddress);
      if (m_myAddress != Ipv4Address ("0.0.0.0"))
        {
          m_expiry (m_myAddress);
        }
    }
  m_myAddress = m_offeredAddress;
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> staticRouting = ipv4RoutingHelper.GetStaticRouting (ipv4);
  if (m_gateway == Ipv4Address ("0.0.0.0"))
    {
      m_gateway = InetSocketAddress::ConvertFrom (from).GetIpv4 ();
    }

  staticRouting->SetDefaultRoute (m_gateway, ifIndex, 0);

  m_remoteAddress = InetSocketAddress::ConvertFrom (from).GetIpv4 ();
  NS_LOG_INFO ("Current DHCP Server is " << m_remoteAddress);

  m_offerList.clear ();
  m_refreshEvent = Simulator::Schedule (m_renew, &DhcpClient::Request, this);
  m_rebindEvent = Simulator::Schedule (m_rebind, &DhcpClient::Request, this);
  m_timeout =  Simulator::Schedule (m_lease, &DhcpClient::RemoveAndStart, this);
  m_state = REFRESH_LEASE;
}

void DhcpClient::RemoveAndStart ()
{
  NS_LOG_FUNCTION (this);

  m_nextOfferEvent.Cancel ();
  m_refreshEvent.Cancel ();
  m_rebindEvent.Cancel ();
  m_timeout.Cancel ();

  Ptr<Ipv4> ipv4MN = GetNode ()->GetObject<Ipv4> ();
  int32_t ifIndex = ipv4MN->GetInterfaceForDevice (m_device);

  for (uint32_t i = 0; i < ipv4MN->GetNAddresses (ifIndex); i++)
    {
      if (ipv4MN->GetAddress (ifIndex,i).GetLocal () == m_myAddress)
        {
          ipv4MN->RemoveAddress (ifIndex, i);
          break;
        }
    }
  m_expiry (m_myAddress);
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> staticRouting = ipv4RoutingHelper.GetStaticRouting (ipv4MN);
  uint32_t i;
  for (i = 0; i < staticRouting->GetNRoutes (); i++)
    {
      if (staticRouting->GetRoute (i).GetGateway () == m_gateway)
        {
          staticRouting->RemoveRoute (i);
          break;
        }
    }
  StartApplication ();
}

} // Namespace ns3
