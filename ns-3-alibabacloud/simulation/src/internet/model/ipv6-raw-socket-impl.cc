/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include "ns3/inet6-socket-address.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv6-packet-info-tag.h"

#include "ipv6-l3-protocol.h"
#include "ipv6-raw-socket-impl.h"
#include "icmpv6-header.h"
#include "icmpv6-l4-protocol.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("Ipv6RawSocketImpl");

NS_OBJECT_ENSURE_REGISTERED (Ipv6RawSocketImpl);

TypeId Ipv6RawSocketImpl::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6RawSocketImpl")
    .SetParent<Socket> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Protocol", "Protocol number to match.", 
                   UintegerValue (0),
                   MakeUintegerAccessor (&Ipv6RawSocketImpl::m_protocol),
                   MakeUintegerChecker<uint16_t> ())
  ;
  return tid;
}

Ipv6RawSocketImpl::Ipv6RawSocketImpl ()
{
  NS_LOG_FUNCTION (this);
  m_err = Socket::ERROR_NOTERROR;
  m_node = 0;
  m_src = Ipv6Address::GetAny ();
  m_dst = Ipv6Address::GetAny ();
  m_protocol = 0;
  m_shutdownSend = false;
  m_shutdownRecv = false;
  Icmpv6FilterSetPassAll();
}

Ipv6RawSocketImpl::~Ipv6RawSocketImpl ()
{
}

void Ipv6RawSocketImpl::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  Socket::DoDispose ();
}

void Ipv6RawSocketImpl::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

Ptr<Node> Ipv6RawSocketImpl::GetNode () const
{
  return m_node;
}

enum Socket::SocketErrno Ipv6RawSocketImpl::GetErrno () const
{
  NS_LOG_FUNCTION (this);
  return m_err;
}

enum Socket::SocketType Ipv6RawSocketImpl::GetSocketType () const
{
  return NS3_SOCK_RAW;
}

int Ipv6RawSocketImpl::Bind (const Address& address)
{
  NS_LOG_FUNCTION (this << address);

  if (!Inet6SocketAddress::IsMatchingType (address))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }
  Inet6SocketAddress ad = Inet6SocketAddress::ConvertFrom (address);
  m_src = ad.GetIpv6 ();
  return 0;
}

int Ipv6RawSocketImpl::Bind ()
{
  NS_LOG_FUNCTION (this);
  m_src = Ipv6Address::GetAny ();
  return 0;
}

int Ipv6RawSocketImpl::Bind6 ()
{
  return(Bind());
}

int Ipv6RawSocketImpl::GetSockName (Address& address) const
{
  NS_LOG_FUNCTION (this);
  address = Inet6SocketAddress (m_src, 0);
  return 0;
}

int
Ipv6RawSocketImpl::GetPeerName (Address& address) const
{
  NS_LOG_FUNCTION (this << address);

  if (m_dst.IsAny ())
    {
      m_err = ERROR_NOTCONN;
      return -1;
    }

  address = Inet6SocketAddress (m_dst, 0);

  return 0;
}

int Ipv6RawSocketImpl::Close ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();

  Ipv6LeaveGroup ();
  if (ipv6)
    {
      ipv6->DeleteRawSocket (this);
    }
  return 0;
}

int Ipv6RawSocketImpl::ShutdownSend ()
{
  NS_LOG_FUNCTION (this);
  m_shutdownSend = true;
  return 0;
}

int Ipv6RawSocketImpl::ShutdownRecv ()
{
  NS_LOG_FUNCTION (this);
  m_shutdownRecv = true;
  return 0;
}

int Ipv6RawSocketImpl::Connect (const Address& address)
{
  NS_LOG_FUNCTION (this << address);

  if (!Inet6SocketAddress::IsMatchingType (address))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }
  Inet6SocketAddress ad = Inet6SocketAddress::ConvertFrom (address);
  m_dst = ad.GetIpv6 ();

  return 0;
}

int Ipv6RawSocketImpl::Listen ()
{
  NS_LOG_FUNCTION (this);
  m_err = Socket::ERROR_OPNOTSUPP;
  return -1;
}

int Ipv6RawSocketImpl::Send (Ptr<Packet> p, uint32_t flags)
{
  NS_LOG_FUNCTION (this << p << flags);
  Inet6SocketAddress to = Inet6SocketAddress (m_dst, m_protocol);
  return SendTo (p, flags, to);
}

int Ipv6RawSocketImpl::SendTo (Ptr<Packet> p, uint32_t flags, const Address& toAddress)
{
  NS_LOG_FUNCTION (this << p << flags << toAddress);

  if (!Inet6SocketAddress::IsMatchingType (toAddress))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }

  if (m_shutdownSend)
    {
      return 0;
    }

  Inet6SocketAddress ad = Inet6SocketAddress::ConvertFrom (toAddress);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  Ipv6Address dst = ad.GetIpv6 ();

  if (IsManualIpv6Tclass ())
    {
      SocketIpv6TclassTag ipTclassTag;
      ipTclassTag.SetTclass (GetIpv6Tclass ());
      p->AddPacketTag (ipTclassTag);
    }

  if (IsManualIpv6HopLimit () && GetIpv6HopLimit () != 0 && !dst.IsMulticast ())
    {
      SocketIpv6HopLimitTag tag;
      tag.SetHopLimit (GetIpv6HopLimit ());
      p->AddPacketTag (tag);
    }

  if (ipv6->GetRoutingProtocol ())
    {
      Ipv6Header hdr;
      hdr.SetDestination (dst);
      SocketErrno err = ERROR_NOTERROR;
      Ptr<Ipv6Route> route = 0;
      Ptr<NetDevice> oif = m_boundnetdevice; //specify non-zero if bound to a specific device

      if (!m_src.IsAny ())
        {
          int32_t index = ipv6->GetInterfaceForAddress (m_src);
          NS_ASSERT (index >= 0);
          oif = ipv6->GetNetDevice (index);
        }

      route = ipv6->GetRoutingProtocol ()->RouteOutput (p, hdr, oif, err);

      if (route)
        {
          NS_LOG_LOGIC ("Route exists");
          if (m_protocol == Icmpv6L4Protocol::GetStaticProtocolNumber ())
            {
              /* calculate checksum here for ICMPv6 echo request (sent by ping6) 
               * as we cannot determine source IPv6 address at application level 
               */
              uint8_t type;
              p->CopyData (&type, sizeof(type));
              if (type == Icmpv6Header::ICMPV6_ECHO_REQUEST)
                {
                  Icmpv6Echo hdr (1);
                  p->RemoveHeader (hdr);
                  hdr.CalculatePseudoHeaderChecksum (route->GetSource (), dst, p->GetSize () + hdr.GetSerializedSize (), Icmpv6L4Protocol::GetStaticProtocolNumber ());
                  p->AddHeader (hdr);
                }
            }

          uint32_t pktSize = p->GetSize ();
          if (m_src.IsAny ())
            {
              ipv6->Send (p, route->GetSource (), dst, m_protocol, route);
            }
          else
            {
              ipv6->Send (p, m_src, dst, m_protocol, route);
            }
          // Return only payload size (as Linux does).
          NotifyDataSent (pktSize);
          NotifySend (GetTxAvailable ());
          return pktSize;
        }
      else
        {
          NS_LOG_DEBUG ("No route, dropped!");
        }
    }
  return 0;
}

Ptr<Packet> Ipv6RawSocketImpl::Recv (uint32_t maxSize, uint32_t flags)
{
  NS_LOG_FUNCTION (this << maxSize << flags);
  Address tmp;
  return RecvFrom (maxSize, flags, tmp);
}

Ptr<Packet> Ipv6RawSocketImpl::RecvFrom (uint32_t maxSize, uint32_t flags, Address& fromAddress)
{
  NS_LOG_FUNCTION (this << maxSize << flags << fromAddress);

  if (m_data.empty ())
    {
      return 0;
    }

  /* get packet */
  Data data = m_data.front ();
  m_data.pop_front ();
  fromAddress = Inet6SocketAddress (data.fromIp, data.fromProtocol);
  if (data.packet->GetSize () > maxSize)
    {
      Ptr<Packet> first = data.packet->CreateFragment (0, maxSize);
      if (!(flags & MSG_PEEK))
        {
          data.packet->RemoveAtStart (maxSize);
        }
      m_data.push_front (data);
      return first;
    }

  return data.packet;
}

void
Ipv6RawSocketImpl::Ipv6JoinGroup (Ipv6Address address, Socket::Ipv6MulticastFilterMode filterMode, std::vector<Ipv6Address> sourceAddresses)
{
  NS_LOG_FUNCTION (this << address << &filterMode << &sourceAddresses);

  // We can join only one multicast group (or change its params)
  NS_ASSERT_MSG ((m_ipv6MulticastGroupAddress == address || m_ipv6MulticastGroupAddress.IsAny ()), "Can join only one IPv6 multicast group.");

  m_ipv6MulticastGroupAddress = address;

  Ptr<Ipv6L3Protocol> ipv6l3 = m_node->GetObject <Ipv6L3Protocol> ();
  if (ipv6l3)
    {
      if (filterMode == INCLUDE && sourceAddresses.empty ())
        {
          // it is a leave
          if (m_boundnetdevice)
            {
              int32_t index = ipv6l3->GetInterfaceForDevice (m_boundnetdevice);
              NS_ASSERT_MSG (index >= 0, "Interface without a valid index");
              ipv6l3->RemoveMulticastAddress (address, index);
            }
          else
            {
              ipv6l3->RemoveMulticastAddress (address);
            }
        }
      else
        {
          // it is a join or a modification
          if (m_boundnetdevice)
            {
              int32_t index = ipv6l3->GetInterfaceForDevice (m_boundnetdevice);
              NS_ASSERT_MSG (index >= 0, "Interface without a valid index");
              ipv6l3->AddMulticastAddress (address, index);
            }
          else
            {
              ipv6l3->AddMulticastAddress (address);
            }
        }
    }
}

uint32_t Ipv6RawSocketImpl::GetTxAvailable () const
{
  NS_LOG_FUNCTION (this);
  return 0xffffffff;
}

uint32_t Ipv6RawSocketImpl::GetRxAvailable () const
{
  NS_LOG_FUNCTION (this);
  uint32_t rx = 0;

  for (std::list<Data>::const_iterator it = m_data.begin (); it != m_data.end (); ++it)
    {
      rx+= (it->packet)->GetSize ();
    }

  return rx;
}

bool Ipv6RawSocketImpl::ForwardUp (Ptr<const Packet> p, Ipv6Header hdr, Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << *p << hdr << device);

  if (m_shutdownRecv)
    {
      return false;
    }

  Ptr<NetDevice> boundNetDevice = Socket::GetBoundNetDevice();
  if (boundNetDevice)
    {
      if (boundNetDevice != device)
        {
          return false;
        }
    }

  if ((m_src == Ipv6Address::GetAny () || hdr.GetDestination () == m_src) && 
      (m_dst == Ipv6Address::GetAny () || hdr.GetSource () == m_dst) &&
      hdr.GetNextHeader () == m_protocol)
    {
      Ptr<Packet> copy = p->Copy ();

      if (m_protocol == Icmpv6L4Protocol::GetStaticProtocolNumber ())
        {
          /* filter */
          Icmpv6Header icmpHeader;
          copy->PeekHeader (icmpHeader);
          uint8_t type = icmpHeader.GetType ();

          if (Icmpv6FilterWillBlock(type))
            {
              /* packet filtered */
              return false;
            }
        }

      // Should check via getsockopt ().
      if (IsRecvPktInfo ())
        {
          Ipv6PacketInfoTag tag;
          copy->RemovePacketTag (tag);
          tag.SetAddress (hdr.GetDestination ());
          tag.SetHoplimit (hdr.GetHopLimit ());
          tag.SetTrafficClass (hdr.GetTrafficClass ());
          tag.SetRecvIf (device->GetIfIndex ());
          copy->AddPacketTag (tag);
        }

      // Check only version 6 options
      if (IsIpv6RecvTclass ())
        {
          SocketIpv6TclassTag ipTclassTag;
          ipTclassTag.SetTclass (hdr.GetTrafficClass ());
          copy->AddPacketTag (ipTclassTag);
        }

      if (IsIpv6RecvHopLimit ())
        {
          SocketIpv6HopLimitTag ipHopLimitTag;
          ipHopLimitTag.SetHopLimit (hdr.GetHopLimit ());
          copy->AddPacketTag (ipHopLimitTag);
        }

      copy->AddHeader (hdr);
      Data data;
      data.packet = copy;
      data.fromIp = hdr.GetSource ();
      data.fromProtocol = hdr.GetNextHeader ();
      m_data.push_back (data);
      NotifyDataRecv ();
      return true;
    }
  return false;
}

bool
Ipv6RawSocketImpl::SetAllowBroadcast (bool allowBroadcast)
{
  if (!allowBroadcast)
    {
      return false;
    }
  return true;
}

bool
Ipv6RawSocketImpl::GetAllowBroadcast () const
{
  return true;
}

void
Ipv6RawSocketImpl::Icmpv6FilterSetPassAll()
{
  memset(&m_icmpFilter, 0xff, sizeof(Icmpv6Filter));
}

void
Ipv6RawSocketImpl::Icmpv6FilterSetBlockAll()
{
  memset(&m_icmpFilter, 0x00, sizeof(Icmpv6Filter));
}

void
Ipv6RawSocketImpl::Icmpv6FilterSetPass(uint8_t type)
{
  (m_icmpFilter.icmpv6Filt[(type) >> 5]) |= (uint32_t(1) << ((type) & 31));
}

void
Ipv6RawSocketImpl::Icmpv6FilterSetBlock(uint8_t type)
{
  (m_icmpFilter.icmpv6Filt[(type) >> 5]) &= ~(uint32_t(1) << ((type) & 31));
}

bool
Ipv6RawSocketImpl::Icmpv6FilterWillPass(uint8_t type)
{
  return (((m_icmpFilter.icmpv6Filt[(type) >> 5]) & (uint32_t(1) << ((type) & 31))) != 0);
}

bool
Ipv6RawSocketImpl::Icmpv6FilterWillBlock(uint8_t type)
{
  return (((m_icmpFilter.icmpv6Filt[(type) >> 5]) & (uint32_t(1) << ((type) & 31))) == 0);
}

} /* namespace ns3 */

