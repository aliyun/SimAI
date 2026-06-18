/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@cutebugs.net>
 */

#include "icmpv4-l4-protocol.h"
#include "ipv4-raw-socket-factory-impl.h"
#include "ipv4-interface.h"
#include "ipv4-l3-protocol.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/boolean.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv6-interface.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Icmpv4L4Protocol");

NS_OBJECT_ENSURE_REGISTERED (Icmpv4L4Protocol);

// see rfc 792
const uint8_t Icmpv4L4Protocol::PROT_NUMBER = 1;

TypeId 
Icmpv4L4Protocol::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Icmpv4L4Protocol")
    .SetParent<IpL4Protocol> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv4L4Protocol> ()
  ;
  return tid;
}

Icmpv4L4Protocol::Icmpv4L4Protocol ()
  : m_node (0)
{
  NS_LOG_FUNCTION (this);
}
Icmpv4L4Protocol::~Icmpv4L4Protocol ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_node == 0);
}

void
Icmpv4L4Protocol::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

/*
 * This method is called by AggregateObject and completes the aggregation
 * by setting the node in the ICMP stack and adding ICMP factory to
 * IPv4 stack connected to the node
 */
void
Icmpv4L4Protocol::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);
  if (m_node == 0)
    {
      Ptr<Node> node = this->GetObject<Node> ();
      if (node != 0)
        {
          Ptr<Ipv4> ipv4 = this->GetObject<Ipv4> ();
          if (ipv4 != 0 && m_downTarget.IsNull ())
            {
              this->SetNode (node);
              ipv4->Insert (this);
              Ptr<Ipv4RawSocketFactoryImpl> rawFactory = CreateObject<Ipv4RawSocketFactoryImpl> ();
              ipv4->AggregateObject (rawFactory);
              this->SetDownTarget (MakeCallback (&Ipv4::Send, ipv4));
            }
        }
    }
  IpL4Protocol::NotifyNewAggregate ();
}

uint16_t 
Icmpv4L4Protocol::GetStaticProtocolNumber (void)
{
  NS_LOG_FUNCTION_NOARGS ();
  return PROT_NUMBER;
}

int 
Icmpv4L4Protocol::GetProtocolNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return PROT_NUMBER;
}
void
Icmpv4L4Protocol::SendMessage (Ptr<Packet> packet, Ipv4Address dest, uint8_t type, uint8_t code)
{
  NS_LOG_FUNCTION (this << packet << dest << static_cast<uint32_t> (type) << static_cast<uint32_t> (code));
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();
  NS_ASSERT (ipv4 != 0 && ipv4->GetRoutingProtocol () != 0);
  Ipv4Header header;
  header.SetDestination (dest);
  header.SetProtocol (PROT_NUMBER);
  Socket::SocketErrno errno_;
  Ptr<Ipv4Route> route;
  Ptr<NetDevice> oif (0); //specify non-zero if bound to a source address
  route = ipv4->GetRoutingProtocol ()->RouteOutput (packet, header, oif, errno_);
  if (route != 0)
    {
      NS_LOG_LOGIC ("Route exists");
      Ipv4Address source = route->GetSource ();
      SendMessage (packet, source, dest, type, code, route);
    }
  else
    {
      NS_LOG_WARN ("drop icmp message");
    }
}

void
Icmpv4L4Protocol::SendMessage (Ptr<Packet> packet, Ipv4Address source, Ipv4Address dest, uint8_t type, uint8_t code, Ptr<Ipv4Route> route)
{
  NS_LOG_FUNCTION (this << packet << source << dest << static_cast<uint32_t> (type) << static_cast<uint32_t> (code) << route);
  Icmpv4Header icmp;
  icmp.SetType (type);
  icmp.SetCode (code);
  if (Node::ChecksumEnabled ())
    {
      icmp.EnableChecksum ();
    }
  packet->AddHeader (icmp);

  m_downTarget (packet, source, dest, PROT_NUMBER, route);
}
void 
Icmpv4L4Protocol::SendDestUnreachFragNeeded (Ipv4Header header, 
                                             Ptr<const Packet> orgData,
                                             uint16_t nextHopMtu)
{
  NS_LOG_FUNCTION (this << header << *orgData << nextHopMtu);
  SendDestUnreach (header, orgData, Icmpv4DestinationUnreachable::ICMPV4_FRAG_NEEDED, nextHopMtu);
}
void 
Icmpv4L4Protocol::SendDestUnreachPort (Ipv4Header header, 
                                       Ptr<const Packet> orgData)
{
  NS_LOG_FUNCTION (this << header << *orgData);
  SendDestUnreach (header, orgData, Icmpv4DestinationUnreachable::ICMPV4_PORT_UNREACHABLE, 0);
}
void 
Icmpv4L4Protocol::SendDestUnreach (Ipv4Header header, Ptr<const Packet> orgData, 
                                   uint8_t code, uint16_t nextHopMtu)
{
  NS_LOG_FUNCTION (this << header << *orgData << (uint32_t) code << nextHopMtu);
  Ptr<Packet> p = Create<Packet> ();
  Icmpv4DestinationUnreachable unreach;
  unreach.SetNextHopMtu (nextHopMtu);
  unreach.SetHeader (header);
  unreach.SetData (orgData);
  p->AddHeader (unreach);
  SendMessage (p, header.GetSource (), Icmpv4Header::ICMPV4_DEST_UNREACH, code);
}

void 
Icmpv4L4Protocol::SendTimeExceededTtl (Ipv4Header header, Ptr<const Packet> orgData, bool isFragment)
{
  NS_LOG_FUNCTION (this << header << *orgData);
  Ptr<Packet> p = Create<Packet> ();
  Icmpv4TimeExceeded time;
  time.SetHeader (header);
  time.SetData (orgData);
  p->AddHeader (time);
  if (!isFragment)
    {
      SendMessage (p, header.GetSource (), Icmpv4Header::ICMPV4_TIME_EXCEEDED, Icmpv4TimeExceeded::ICMPV4_TIME_TO_LIVE);
    }
  else
    {
      SendMessage (p, header.GetSource (), Icmpv4Header::ICMPV4_TIME_EXCEEDED, Icmpv4TimeExceeded::ICMPV4_FRAGMENT_REASSEMBLY);
    }
}

void
Icmpv4L4Protocol::HandleEcho (Ptr<Packet> p,
                              Icmpv4Header header,
                              Ipv4Address source,
                              Ipv4Address destination)
{
  NS_LOG_FUNCTION (this << p << header << source << destination);

  Ptr<Packet> reply = Create<Packet> ();
  Icmpv4Echo echo;
  p->RemoveHeader (echo);
  reply->AddHeader (echo);
  SendMessage (reply, destination, source, Icmpv4Header::ICMPV4_ECHO_REPLY, 0, 0);
}
void
Icmpv4L4Protocol::Forward (Ipv4Address source, Icmpv4Header icmp,
                           uint32_t info, Ipv4Header ipHeader,
                           const uint8_t payload[8])
{
  NS_LOG_FUNCTION (this << source << icmp << info << ipHeader << payload);

  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();
  Ptr<IpL4Protocol> l4 = ipv4->GetProtocol (ipHeader.GetProtocol ());
  if (l4 != 0)
    {
      l4->ReceiveIcmp (source, ipHeader.GetTtl (), icmp.GetType (), icmp.GetCode (),
                       info, ipHeader.GetSource (), ipHeader.GetDestination (), payload);
    }
}
void
Icmpv4L4Protocol::HandleDestUnreach (Ptr<Packet> p,
                                     Icmpv4Header icmp,
                                     Ipv4Address source,
                                     Ipv4Address destination)
{
  NS_LOG_FUNCTION (this << p << icmp << source << destination);

  Icmpv4DestinationUnreachable unreach;
  p->PeekHeader (unreach);
  uint8_t payload[8];
  unreach.GetData (payload);
  Ipv4Header ipHeader = unreach.GetHeader ();
  Forward (source, icmp, unreach.GetNextHopMtu (), ipHeader, payload);
}
void
Icmpv4L4Protocol::HandleTimeExceeded (Ptr<Packet> p,
                                      Icmpv4Header icmp,
                                      Ipv4Address source,
                                      Ipv4Address destination)
{
  NS_LOG_FUNCTION (this << p << icmp << source << destination);

  Icmpv4TimeExceeded time;
  p->PeekHeader (time);
  uint8_t payload[8];
  time.GetData (payload);
  Ipv4Header ipHeader = time.GetHeader ();
  // info field is zero for TimeExceeded on linux
  Forward (source, icmp, 0, ipHeader, payload);
}

enum IpL4Protocol::RxStatus
Icmpv4L4Protocol::Receive (Ptr<Packet> p,
                           Ipv4Header const &header,
                           Ptr<Ipv4Interface> incomingInterface)
{
  NS_LOG_FUNCTION (this << p << header << incomingInterface);

  Icmpv4Header icmp;
  p->RemoveHeader (icmp);
  switch (icmp.GetType ()) {
    case Icmpv4Header::ICMPV4_ECHO:
      HandleEcho (p, icmp, header.GetSource (), header.GetDestination ());
      break;
    case Icmpv4Header::ICMPV4_DEST_UNREACH:
      HandleDestUnreach (p, icmp, header.GetSource (), header.GetDestination ());
      break;
    case Icmpv4Header::ICMPV4_TIME_EXCEEDED:
      HandleTimeExceeded (p, icmp, header.GetSource (), header.GetDestination ());
      break;
    default:
      NS_LOG_DEBUG (icmp << " " << *p);
      break;
    }
  return IpL4Protocol::RX_OK;
}
enum IpL4Protocol::RxStatus
Icmpv4L4Protocol::Receive (Ptr<Packet> p,
                           Ipv6Header const &header,
                           Ptr<Ipv6Interface> incomingInterface)
{
  NS_LOG_FUNCTION (this << p << header.GetSource () << header.GetDestination () << incomingInterface);
  return IpL4Protocol::RX_ENDPOINT_UNREACH;
}
void 
Icmpv4L4Protocol::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  m_downTarget.Nullify ();
  IpL4Protocol::DoDispose ();
}

void
Icmpv4L4Protocol::SetDownTarget (IpL4Protocol::DownTargetCallback callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_downTarget = callback;
}

void
Icmpv4L4Protocol::SetDownTarget6 (IpL4Protocol::DownTargetCallback6 callback)
{
  NS_LOG_FUNCTION (this << &callback);
}

IpL4Protocol::DownTargetCallback
Icmpv4L4Protocol::GetDownTarget (void) const
{
  NS_LOG_FUNCTION (this);
  return m_downTarget;
}

IpL4Protocol::DownTargetCallback6
Icmpv4L4Protocol::GetDownTarget6 (void) const
{
  NS_LOG_FUNCTION (this);
  return (IpL4Protocol::DownTargetCallback6)NULL;
}

} // namespace ns3
