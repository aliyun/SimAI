/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include "ipv4-raw-socket-impl.h"
#include "ipv4-l3-protocol.h"
#include "icmpv4.h"
#include "ns3/ipv4-packet-info-tag.h"
#include "ns3/inet-socket-address.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4RawSocketImpl");

NS_OBJECT_ENSURE_REGISTERED (Ipv4RawSocketImpl);

TypeId 
Ipv4RawSocketImpl::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4RawSocketImpl")
    .SetParent<Socket> ()
    .SetGroupName ("Internet")
    .AddAttribute ("Protocol", "Protocol number to match.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&Ipv4RawSocketImpl::m_protocol),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("IcmpFilter", 
                   "Any icmp header whose type field matches a bit in this filter is dropped. Type must be less than 32.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&Ipv4RawSocketImpl::m_icmpFilter),
                   MakeUintegerChecker<uint32_t> ())
    // 
    //  from raw (7), linux, returned length of Send/Recv should be
    // 
    //            | IP_HDRINC on  |      off    |
    //  ----------+---------------+-------------+-
    //  Send(Ipv4)| hdr + payload | payload     |
    //  Recv(Ipv4)| hdr + payload | hdr+payload |
    //  ----------+---------------+-------------+-
    .AddAttribute ("IpHeaderInclude", 
                   "Include IP Header information (a.k.a setsockopt (IP_HDRINCL)).",
                   BooleanValue (false),
                   MakeBooleanAccessor (&Ipv4RawSocketImpl::m_iphdrincl),
                   MakeBooleanChecker ())
  ;
  return tid;
}

Ipv4RawSocketImpl::Ipv4RawSocketImpl ()
{
  NS_LOG_FUNCTION (this);
  m_err = Socket::ERROR_NOTERROR;
  m_node = 0;
  m_src = Ipv4Address::GetAny ();
  m_dst = Ipv4Address::GetAny ();
  m_protocol = 0;
  m_shutdownSend = false;
  m_shutdownRecv = false;
}

void 
Ipv4RawSocketImpl::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

void
Ipv4RawSocketImpl::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_node = 0;
  Socket::DoDispose ();
}

enum Socket::SocketErrno 
Ipv4RawSocketImpl::GetErrno (void) const
{
  NS_LOG_FUNCTION (this);
  return m_err;
}

enum Socket::SocketType
Ipv4RawSocketImpl::GetSocketType (void) const
{
  NS_LOG_FUNCTION (this);
  return NS3_SOCK_RAW;
}

Ptr<Node> 
Ipv4RawSocketImpl::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}
int 
Ipv4RawSocketImpl::Bind (const Address &address)
{
  NS_LOG_FUNCTION (this << address);
  if (!InetSocketAddress::IsMatchingType (address))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }
  InetSocketAddress ad = InetSocketAddress::ConvertFrom (address);
  m_src = ad.GetIpv4 ();
  return 0;
}
int 
Ipv4RawSocketImpl::Bind (void)
{
  NS_LOG_FUNCTION (this);
  m_src = Ipv4Address::GetAny ();
  return 0;
}
int 
Ipv4RawSocketImpl::Bind6 (void)
{
  NS_LOG_FUNCTION (this);
  return (-1);
}
int 
Ipv4RawSocketImpl::GetSockName (Address &address) const
{
  NS_LOG_FUNCTION (this << address);
  address = InetSocketAddress (m_src, 0);
  return 0;
}
int
Ipv4RawSocketImpl::GetPeerName (Address &address) const
{
  NS_LOG_FUNCTION (this << address);

  if (m_dst == Ipv4Address::GetAny ())
    {
      m_err = ERROR_NOTCONN;
      return -1;
    }

  address = InetSocketAddress (m_dst, 0);

  return 0;
}
int 
Ipv4RawSocketImpl::Close (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();
  if (ipv4 != 0)
    {
      ipv4->DeleteRawSocket (this);
    }
  return 0;
}
int 
Ipv4RawSocketImpl::ShutdownSend (void)
{
  NS_LOG_FUNCTION (this);
  m_shutdownSend = true;
  return 0;
}
int 
Ipv4RawSocketImpl::ShutdownRecv (void)
{
  NS_LOG_FUNCTION (this);
  m_shutdownRecv = true;
  return 0;
}
int 
Ipv4RawSocketImpl::Connect (const Address &address)
{
  NS_LOG_FUNCTION (this << address);
  if (!InetSocketAddress::IsMatchingType (address))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }
  InetSocketAddress ad = InetSocketAddress::ConvertFrom (address);
  m_dst = ad.GetIpv4 ();
  SetIpTos (ad.GetTos ());

  return 0;
}
int 
Ipv4RawSocketImpl::Listen (void)
{
  NS_LOG_FUNCTION (this);
  m_err = Socket::ERROR_OPNOTSUPP;
  return -1;
}
uint32_t 
Ipv4RawSocketImpl::GetTxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  return 0xffffffff;
}
int 
Ipv4RawSocketImpl::Send (Ptr<Packet> p, uint32_t flags)
{
  NS_LOG_FUNCTION (this << p << flags);
  InetSocketAddress to = InetSocketAddress (m_dst, m_protocol);
  to.SetTos (GetIpTos ());
  return SendTo (p, flags, to);
}
int 
Ipv4RawSocketImpl::SendTo (Ptr<Packet> p, uint32_t flags, 
                           const Address &toAddress)
{
  NS_LOG_FUNCTION (this << p << flags << toAddress);
  if (!InetSocketAddress::IsMatchingType (toAddress))
    {
      m_err = Socket::ERROR_INVAL;
      return -1;
    }
  if (m_shutdownSend)
    {
      return 0;
    }

  InetSocketAddress ad = InetSocketAddress::ConvertFrom (toAddress);
  Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4> ();
  Ipv4Address dst = ad.GetIpv4 ();
  Ipv4Address src = m_src;
  uint8_t tos = ad.GetTos ();

  uint8_t priority = GetPriority ();
  if (tos)
    {
      SocketIpTosTag ipTosTag;
      ipTosTag.SetTos (tos);
      // This packet may already have a SocketIpTosTag (see BUG 2440)
      p->ReplacePacketTag (ipTosTag);
      priority = IpTos2Priority (tos);
    }
  if (priority)
    {
      SocketPriorityTag priorityTag;
      priorityTag.SetPriority (priority);
      p->ReplacePacketTag (priorityTag);
    }

  if (IsManualIpTtl () && GetIpTtl () != 0 && !dst.IsMulticast () && !dst.IsBroadcast ())
    {
      SocketIpTtlTag tag;
      tag.SetTtl (GetIpTtl ());
      p->AddPacketTag (tag);
    }

  bool subnetDirectedBroadcast = false;
  if (m_boundnetdevice)
    {
      uint32_t iif = ipv4->GetInterfaceForDevice (m_boundnetdevice);
      for (uint32_t j = 0; j < ipv4->GetNAddresses (iif); j++)
        {
          Ipv4InterfaceAddress ifAddr = ipv4->GetAddress (iif, j);
          if (dst.IsSubnetDirectedBroadcast (ifAddr.GetMask ()))
            {
              subnetDirectedBroadcast = true;
            }
        }
    }

  if (dst.IsBroadcast () || subnetDirectedBroadcast)
    {
      Ptr <NetDevice> boundNetDevice = m_boundnetdevice;
      if (ipv4->GetNInterfaces () == 1)
        {
          boundNetDevice = ipv4->GetNetDevice (0);
        }
      if (boundNetDevice == 0)
        {
          NS_LOG_DEBUG ("dropped because no outgoing route.");
          return -1;
        }

      Ipv4Header header;
      uint32_t pktSize = p->GetSize ();
      if (!m_iphdrincl)
        {
          header.SetDestination (dst);
          header.SetProtocol (m_protocol);
          Ptr<Ipv4Route> route = Create <Ipv4Route> ();
          route->SetSource (src);
          route->SetDestination (dst);
          route->SetOutputDevice (boundNetDevice);
          ipv4->Send (p, route->GetSource (), dst, m_protocol, route);
        }
      else
        {
          p->RemoveHeader (header);
          dst = header.GetDestination ();
          src = header.GetSource ();
          pktSize += header.GetSerializedSize ();
          Ptr<Ipv4Route> route = Create <Ipv4Route> ();
          route->SetSource (src);
          route->SetDestination (dst);
          route->SetOutputDevice (boundNetDevice);
          ipv4->SendWithHeader (p, header, route);
        }
      NotifyDataSent (pktSize);
      NotifySend (GetTxAvailable ());
      return pktSize;
    }


  if (ipv4->GetRoutingProtocol ())
    {
      Ipv4Header header;
      if (!m_iphdrincl)
        {
          header.SetDestination (dst);
          header.SetProtocol (m_protocol);
        }
      else
        {
          p->RemoveHeader (header);
          dst = header.GetDestination ();
          src = header.GetSource ();
        }
      SocketErrno errno_ = ERROR_NOTERROR; //do not use errno as it is the standard C last error number
      Ptr<Ipv4Route> route;
      Ptr<NetDevice> oif = m_boundnetdevice; //specify non-zero if bound to a source address
      if (!oif && src != Ipv4Address::GetAny ())
        {
          int32_t index = ipv4->GetInterfaceForAddress (src);
          NS_ASSERT (index >= 0);
          oif = ipv4->GetNetDevice (index);
          NS_LOG_LOGIC ("Set index " << oif << "from source " << src);
        }

      // TBD-- we could cache the route and just check its validity
      route = ipv4->GetRoutingProtocol ()->RouteOutput (p, header, oif, errno_);
      if (route != 0)
        {
          NS_LOG_LOGIC ("Route exists");
          uint32_t pktSize = p->GetSize ();
          if (!m_iphdrincl)
            {
              ipv4->Send (p, route->GetSource (), dst, m_protocol, route);
            }
          else
            {
              pktSize += header.GetSerializedSize ();
              ipv4->SendWithHeader (p, header, route);
            }
          NotifyDataSent (pktSize);
          NotifySend (GetTxAvailable ());
          return pktSize;
        }
      else
        {
          NS_LOG_DEBUG ("dropped because no outgoing route.");
          return -1;
        }
    }
  return 0;
}
uint32_t 
Ipv4RawSocketImpl::GetRxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  uint32_t rx = 0;
  for (std::list<Data>::const_iterator i = m_recv.begin (); i != m_recv.end (); ++i)
    {
      rx += (i->packet)->GetSize ();
    }
  return rx;
}
Ptr<Packet> 
Ipv4RawSocketImpl::Recv (uint32_t maxSize, uint32_t flags)
{
  NS_LOG_FUNCTION (this << maxSize << flags);
  Address tmp;
  return RecvFrom (maxSize, flags, tmp);
}
Ptr<Packet> 
Ipv4RawSocketImpl::RecvFrom (uint32_t maxSize, uint32_t flags,
                             Address &fromAddress)
{
  NS_LOG_FUNCTION (this << maxSize << flags << fromAddress);
  if (m_recv.empty ())
    {
      return 0;
    }
  struct Data data = m_recv.front ();
  m_recv.pop_front ();
  InetSocketAddress inet = InetSocketAddress (data.fromIp, data.fromProtocol);
  fromAddress = inet;
  if (data.packet->GetSize () > maxSize)
    {
      Ptr<Packet> first = data.packet->CreateFragment (0, maxSize);
      if (!(flags & MSG_PEEK))
        {
          data.packet->RemoveAtStart (maxSize);
        }
      m_recv.push_front (data);
      return first;
    }
  return data.packet;
}

void 
Ipv4RawSocketImpl::SetProtocol (uint16_t protocol)
{
  NS_LOG_FUNCTION (this << protocol);
  m_protocol = protocol;
}

bool 
Ipv4RawSocketImpl::ForwardUp (Ptr<const Packet> p, Ipv4Header ipHeader, Ptr<Ipv4Interface> incomingInterface)
{
  NS_LOG_FUNCTION (this << *p << ipHeader << incomingInterface);
  if (m_shutdownRecv)
    {
      return false;
    }

  Ptr<NetDevice> boundNetDevice = Socket::GetBoundNetDevice();
  if (boundNetDevice)
    {
      if (boundNetDevice != incomingInterface->GetDevice())
        {
          return false;
        }
    }

  NS_LOG_LOGIC ("src = " << m_src << " dst = " << m_dst);
  if ((m_src == Ipv4Address::GetAny () || ipHeader.GetDestination () == m_src) &&
      (m_dst == Ipv4Address::GetAny () || ipHeader.GetSource () == m_dst) &&
      ipHeader.GetProtocol () == m_protocol)
    {
      Ptr<Packet> copy = p->Copy ();
      // Should check via getsockopt ()..
      if (IsRecvPktInfo ())
        {
          Ipv4PacketInfoTag tag;
          copy->RemovePacketTag (tag);
          tag.SetAddress (ipHeader.GetDestination ());
          tag.SetTtl (ipHeader.GetTtl ());
          tag.SetRecvIf (incomingInterface->GetDevice ()->GetIfIndex ());
          copy->AddPacketTag (tag);
        }

      //Check only version 4 options
      if (IsIpRecvTos ())
        {
          SocketIpTosTag ipTosTag;
          ipTosTag.SetTos (ipHeader.GetTos ());
          copy->AddPacketTag (ipTosTag);
        }

      if (IsIpRecvTtl ())
        {
          SocketIpTtlTag ipTtlTag;
          ipTtlTag.SetTtl (ipHeader.GetTtl ());
          copy->AddPacketTag (ipTtlTag);
        }

     if (m_protocol == 1)
        {
          Icmpv4Header icmpHeader;
          copy->PeekHeader (icmpHeader);
          uint8_t type = icmpHeader.GetType ();
          if (type < 32 &&
              ((uint32_t(1) << type) & m_icmpFilter))
            {
              // filter out icmp packet.
              return false;
            }
        }
      copy->AddHeader (ipHeader);
      struct Data data;
      data.packet = copy;
      data.fromIp = ipHeader.GetSource ();
      data.fromProtocol = ipHeader.GetProtocol ();
      m_recv.push_back (data);
      NotifyDataRecv ();
      return true;
    }
  return false;
}

bool
Ipv4RawSocketImpl::SetAllowBroadcast (bool allowBroadcast)
{
  NS_LOG_FUNCTION (this << allowBroadcast);
  if (!allowBroadcast)
    {
      return false;
    }
  return true;
}

bool
Ipv4RawSocketImpl::GetAllowBroadcast () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

} // namespace ns3
