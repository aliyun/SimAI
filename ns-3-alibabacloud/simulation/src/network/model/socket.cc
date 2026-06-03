/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 Georgia Tech Research Corporation
 *               2007 INRIA
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
 * Authors: George F. Riley<riley@ece.gatech.edu>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "node.h"
#include "socket.h"
#include "socket-factory.h"
#include <limits>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Socket");

NS_OBJECT_ENSURE_REGISTERED (Socket);

TypeId
Socket::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Socket")
    .SetParent<Object> ()
    .SetGroupName("Network");
  return tid;
}

Socket::Socket (void)
  : m_manualIpTtl (false),
    m_ipRecvTos (false),
    m_ipRecvTtl (false),
    m_manualIpv6Tclass (false),
    m_manualIpv6HopLimit (false),
    m_ipv6RecvTclass (false),
    m_ipv6RecvHopLimit (false)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_boundnetdevice = 0;
  m_recvPktInfo = false;

  m_priority = 0;
  m_ipTos = 0;
  m_ipTtl = 0;
  m_ipv6Tclass = 0;
  m_ipv6HopLimit = 0;
}

Socket::~Socket ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<Socket>
Socket::CreateSocket (Ptr<Node> node, TypeId tid)
{
  NS_LOG_FUNCTION (node << tid);
  Ptr<Socket> s;
  NS_ASSERT (node != 0);
  Ptr<SocketFactory> socketFactory = node->GetObject<SocketFactory> (tid);
  NS_ASSERT (socketFactory != 0);
  s = socketFactory->CreateSocket ();
  NS_ASSERT (s != 0);
  return s;
}

void
Socket::SetConnectCallback (
  Callback<void, Ptr<Socket> > connectionSucceeded,
  Callback<void, Ptr<Socket> > connectionFailed)
{
  NS_LOG_FUNCTION (this << &connectionSucceeded << &connectionFailed);
  m_connectionSucceeded = connectionSucceeded;
  m_connectionFailed = connectionFailed;
}

void
Socket::SetCloseCallbacks (
  Callback<void, Ptr<Socket> > normalClose,
  Callback<void, Ptr<Socket> > errorClose)
{
  NS_LOG_FUNCTION (this << &normalClose << &errorClose);
  m_normalClose = normalClose;
  m_errorClose = errorClose;
}

void
Socket::SetAcceptCallback (
  Callback<bool, Ptr<Socket>, const Address &> connectionRequest,
  Callback<void, Ptr<Socket>, const Address&> newConnectionCreated)
{
  NS_LOG_FUNCTION (this << &connectionRequest << &newConnectionCreated);
  m_connectionRequest = connectionRequest;
  m_newConnectionCreated = newConnectionCreated;
}

void
Socket::SetDataSentCallback (Callback<void, Ptr<Socket>, uint32_t> dataSent)
{
  NS_LOG_FUNCTION (this << &dataSent);
  m_dataSent = dataSent;
}

void
Socket::SetSendCallback (Callback<void, Ptr<Socket>, uint32_t> sendCb)
{
  NS_LOG_FUNCTION (this << &sendCb);
  m_sendCb = sendCb;
}

void
Socket::SetRecvCallback (Callback<void, Ptr<Socket> > receivedData)
{
  NS_LOG_FUNCTION (this << &receivedData);
  m_receivedData = receivedData;
}

int
Socket::Send (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p);
  return Send (p, 0);
}

int
Socket::Send (const uint8_t* buf, uint32_t size, uint32_t flags)
{
  NS_LOG_FUNCTION (this << &buf << size << flags);
  Ptr<Packet> p;
  if (buf)
    {
      p = Create<Packet> (buf, size);
    }
  else
    {
      p = Create<Packet> (size);
    }
  return Send (p, flags);
}

int
Socket::SendTo (const uint8_t* buf, uint32_t size, uint32_t flags,
                const Address &toAddress)
{
  NS_LOG_FUNCTION (this << &buf << size << flags << &toAddress);
  Ptr<Packet> p;
  if(buf)
    {
      p = Create<Packet> (buf, size);
    }
  else
    {
      p = Create<Packet> (size);
    }
  return SendTo (p, flags, toAddress);
}

Ptr<Packet>
Socket::Recv (void)
{
  NS_LOG_FUNCTION (this);
  return Recv (std::numeric_limits<uint32_t>::max (), 0);
}

int
Socket::Recv (uint8_t* buf, uint32_t size, uint32_t flags)
{
  NS_LOG_FUNCTION (this << &buf << size << flags);
  Ptr<Packet> p = Recv (size, flags); // read up to "size" bytes
  if (p == 0)
    {
      return 0;
    }
  p->CopyData (buf, p->GetSize ());
  return p->GetSize ();
}

Ptr<Packet>
Socket::RecvFrom (Address &fromAddress)
{
  NS_LOG_FUNCTION (this << &fromAddress);
  return RecvFrom (std::numeric_limits<uint32_t>::max (), 0, fromAddress);
}

int
Socket::RecvFrom (uint8_t* buf, uint32_t size, uint32_t flags,
                  Address &fromAddress)
{
  NS_LOG_FUNCTION (this << &buf << size << flags << &fromAddress);
  Ptr<Packet> p = RecvFrom (size, flags, fromAddress);
  if (p == 0)
    {
      return 0;
    }
  p->CopyData (buf, p->GetSize ());
  return p->GetSize ();
}


void
Socket::NotifyConnectionSucceeded (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_connectionSucceeded.IsNull ())
    {
      m_connectionSucceeded (this);
    }
}

void
Socket::NotifyConnectionFailed (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_connectionFailed.IsNull ())
    {
      m_connectionFailed (this);
    }
}

void
Socket::NotifyNormalClose (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_normalClose.IsNull ())
    {
      m_normalClose (this);
    }
}

void
Socket::NotifyErrorClose (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_errorClose.IsNull ())
    {
      m_errorClose (this);
    }
}

bool
Socket::NotifyConnectionRequest (const Address &from)
{
  NS_LOG_FUNCTION (this << &from);
  if (!m_connectionRequest.IsNull ())
    {
      return m_connectionRequest (this, from);
    }
  else
    {
      // accept all incoming connections by default.
      // this way people writing code don't have to do anything
      // special like register a callback that returns true
      // just to get incoming connections
      return true;
    }
}

void
Socket::NotifyNewConnectionCreated (Ptr<Socket> socket, const Address &from)
{
  NS_LOG_FUNCTION (this << socket << from);
  if (!m_newConnectionCreated.IsNull ())
    {
      m_newConnectionCreated (socket, from);
    }
}

void
Socket::NotifyDataSent (uint32_t size)
{
  NS_LOG_FUNCTION (this << size);
  if (!m_dataSent.IsNull ())
    {
      m_dataSent (this, size);
    }
}

void
Socket::NotifySend (uint32_t spaceAvailable)
{
  NS_LOG_FUNCTION (this << spaceAvailable);
  if (!m_sendCb.IsNull ())
    {
      m_sendCb (this, spaceAvailable);
    }
}

void
Socket::NotifyDataRecv (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_receivedData.IsNull ())
    {
      m_receivedData (this);
    }
}

void
Socket::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_connectionSucceeded = MakeNullCallback<void,Ptr<Socket> > ();
  m_connectionFailed = MakeNullCallback<void,Ptr<Socket> > ();
  m_normalClose = MakeNullCallback<void,Ptr<Socket> > ();
  m_errorClose = MakeNullCallback<void,Ptr<Socket> > ();
  m_connectionRequest = MakeNullCallback<bool,Ptr<Socket>, const Address &> ();
  m_newConnectionCreated = MakeNullCallback<void,Ptr<Socket>, const Address &> ();
  m_dataSent = MakeNullCallback<void,Ptr<Socket>, uint32_t> ();
  m_sendCb = MakeNullCallback<void,Ptr<Socket>, uint32_t> ();
  m_receivedData = MakeNullCallback<void,Ptr<Socket> > ();
}

void
Socket::BindToNetDevice (Ptr<NetDevice> netdevice)
{
  NS_LOG_FUNCTION (this << netdevice);
  if (netdevice != 0)
    {
      bool found = false;
      for (uint32_t i = 0; i < GetNode ()->GetNDevices (); i++)
        {
          if (GetNode ()->GetDevice (i) == netdevice)
            {
              found = true;
              break;
            }
        }
      NS_ASSERT_MSG (found, "Socket cannot be bound to a NetDevice not existing on the Node");
    }
  m_boundnetdevice = netdevice;
  return;
}

Ptr<NetDevice>
Socket::GetBoundNetDevice ()
{
  NS_LOG_FUNCTION (this);
  return m_boundnetdevice;
}

void
Socket::SetRecvPktInfo (bool flag)
{
  NS_LOG_FUNCTION (this << flag);
  m_recvPktInfo = flag;
}

bool Socket::IsRecvPktInfo () const
{
  NS_LOG_FUNCTION (this);
  return m_recvPktInfo;
}

bool
Socket::IsManualIpv6Tclass (void) const
{
  return m_manualIpv6Tclass;
}

bool
Socket::IsManualIpTtl (void) const
{
  return m_manualIpTtl;
}

bool
Socket::IsManualIpv6HopLimit (void) const
{
  return m_manualIpv6HopLimit;
}

void
Socket::SetPriority (uint8_t priority)
{
  NS_LOG_FUNCTION (this << priority);
  m_priority = priority;
}

uint8_t
Socket::GetPriority (void) const
{
  return m_priority;
}

uint8_t
Socket::IpTos2Priority (uint8_t ipTos)
{
  uint8_t prio = NS3_PRIO_BESTEFFORT;
  ipTos &= 0x1e;
  switch (ipTos >> 1)
    {
    case 0:
    case 1:
    case 2:
    case 3:
      prio = NS3_PRIO_BESTEFFORT;
      break;
    case 4:
    case 5:
    case 6:
    case 7:
      prio = NS3_PRIO_BULK;
      break;
    case 8:
    case 9:
    case 10:
    case 11:
      prio = NS3_PRIO_INTERACTIVE;
      break;
    case 12:
    case 13:
    case 14:
    case 15:
      prio = NS3_PRIO_INTERACTIVE_BULK;
      break;
    }
  return prio;
}

void
Socket::SetIpTos (uint8_t tos)
{
  Address address;
  GetSockName (address);
  if (GetSocketType () == NS3_SOCK_STREAM)
    {
      // preserve the least two significant bits of the current TOS
      // value, which are used for ECN
      tos &= 0xfc;
      tos |= m_ipTos & 0x3;
    }
  m_ipTos = tos;
  m_priority = IpTos2Priority (tos);
}

uint8_t
Socket::GetIpTos (void) const
{
  return m_ipTos;
}

void
Socket::SetIpRecvTos (bool ipv4RecvTos)
{
  m_ipRecvTos = ipv4RecvTos;
}

bool
Socket::IsIpRecvTos (void) const
{
  return m_ipRecvTos;
}

void
Socket::SetIpv6Tclass (int tclass)
{
  Address address;
  GetSockName (address);

  //If -1 or invalid values, use default
  if (tclass == -1 || tclass < -1 || tclass > 0xff)
    {
      //Print a warning
      if (tclass < -1 || tclass > 0xff)
        {
          NS_LOG_WARN ("Invalid IPV6_TCLASS value. Using default.");
        }
      m_manualIpv6Tclass = false;
      m_ipv6Tclass = 0;
    }
  else
    {
      m_manualIpv6Tclass = true;
      m_ipv6Tclass = tclass;
    }
}

uint8_t
Socket::GetIpv6Tclass (void) const
{
  return m_ipv6Tclass;
}

void
Socket::SetIpv6RecvTclass (bool ipv6RecvTclass)
{
  m_ipv6RecvTclass = ipv6RecvTclass;
}

bool
Socket::IsIpv6RecvTclass (void) const
{
  return m_ipv6RecvTclass;
}

void
Socket::SetIpTtl (uint8_t ttl)
{
  m_manualIpTtl = true;
  m_ipTtl = ttl;
}

uint8_t
Socket::GetIpTtl (void) const
{
  return m_ipTtl;
}

void
Socket::SetIpRecvTtl (bool ipv4RecvTtl)
{
  m_ipRecvTtl = ipv4RecvTtl;
}

bool
Socket::IsIpRecvTtl (void) const
{
  return m_ipRecvTtl;
}

void
Socket::SetIpv6HopLimit (uint8_t ipHopLimit)
{
  m_manualIpv6HopLimit = true;
  m_ipv6HopLimit = ipHopLimit;
}

uint8_t
Socket::GetIpv6HopLimit (void) const
{
  return m_ipv6HopLimit;
}

void
Socket::SetIpv6RecvHopLimit (bool ipv6RecvHopLimit)
{
  m_ipv6RecvHopLimit = ipv6RecvHopLimit;
}

bool
Socket::IsIpv6RecvHopLimit (void) const
{
  return m_ipv6RecvHopLimit;
}

void
Socket::Ipv6JoinGroup (Ipv6Address address, Ipv6MulticastFilterMode filterMode, std::vector<Ipv6Address> sourceAddresses)
{
  NS_LOG_FUNCTION (this<<address<<&filterMode<<&sourceAddresses);
  NS_ASSERT_MSG (false,"Ipv6JoinGroup not implemented on this socket");
}

void
Socket::Ipv6JoinGroup (Ipv6Address address)
{
  NS_LOG_FUNCTION (this<<address);

  // Join Group. Note that joining a group with no sources means joining without source restrictions.
  std::vector<Ipv6Address> sourceAddresses;
  Ipv6JoinGroup (address, EXCLUDE, sourceAddresses);
}

void
Socket::Ipv6LeaveGroup (void)
{
  NS_LOG_FUNCTION (this);
  if(m_ipv6MulticastGroupAddress.IsAny () )
    {
      NS_LOG_INFO (" The socket was not bound to any group.");
      return;
    }
  // Leave Group. Note that joining a group with no sources means leaving it.
  std::vector<Ipv6Address> sourceAddresses;
  Ipv6JoinGroup (m_ipv6MulticastGroupAddress, INCLUDE, sourceAddresses);
  m_ipv6MulticastGroupAddress = Ipv6Address::GetAny ();
}

/***************************************************************
 *           Socket Tags
 ***************************************************************/

SocketIpTtlTag::SocketIpTtlTag ()
{
  NS_LOG_FUNCTION (this);
}

void
SocketIpTtlTag::SetTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (ttl));
  m_ttl = ttl;
}

uint8_t
SocketIpTtlTag::GetTtl (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ttl;
}

NS_OBJECT_ENSURE_REGISTERED (SocketIpTtlTag);

TypeId
SocketIpTtlTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketIpTtlTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketIpTtlTag> ()
  ;
  return tid;
}
TypeId
SocketIpTtlTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
SocketIpTtlTag::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 1;
}
void
SocketIpTtlTag::Serialize (TagBuffer i) const
{
  NS_LOG_FUNCTION (this << &i);
  i.WriteU8 (m_ttl);
}
void
SocketIpTtlTag::Deserialize (TagBuffer i)
{
  NS_LOG_FUNCTION (this << &i);
  m_ttl = i.ReadU8 ();
}
void
SocketIpTtlTag::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "Ttl=" << (uint32_t) m_ttl;
}

SocketIpv6HopLimitTag::SocketIpv6HopLimitTag ()
{
}

void
SocketIpv6HopLimitTag::SetHopLimit (uint8_t hopLimit)
{
  m_hopLimit = hopLimit;
}

uint8_t
SocketIpv6HopLimitTag::GetHopLimit (void) const
{
  return m_hopLimit;
}

NS_OBJECT_ENSURE_REGISTERED (SocketIpv6HopLimitTag);

TypeId
SocketIpv6HopLimitTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketIpv6HopLimitTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketIpv6HopLimitTag> ()
  ;
  return tid;
}
TypeId
SocketIpv6HopLimitTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
SocketIpv6HopLimitTag::GetSerializedSize (void) const
{
  return 1;
}
void
SocketIpv6HopLimitTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_hopLimit);
}
void
SocketIpv6HopLimitTag::Deserialize (TagBuffer i)
{
  m_hopLimit = i.ReadU8 ();
}
void
SocketIpv6HopLimitTag::Print (std::ostream &os) const
{
  os << "HopLimit=" << (uint32_t) m_hopLimit;
}

SocketSetDontFragmentTag::SocketSetDontFragmentTag ()
{
  NS_LOG_FUNCTION (this);
}
void
SocketSetDontFragmentTag::Enable (void)
{
  NS_LOG_FUNCTION (this);
  m_dontFragment = true;
}
void
SocketSetDontFragmentTag::Disable (void)
{
  NS_LOG_FUNCTION (this);
  m_dontFragment = false;
}
bool
SocketSetDontFragmentTag::IsEnabled (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dontFragment;
}

NS_OBJECT_ENSURE_REGISTERED (SocketSetDontFragmentTag);

TypeId
SocketSetDontFragmentTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketSetDontFragmentTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketSetDontFragmentTag> ();
  return tid;
}
TypeId
SocketSetDontFragmentTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t
SocketSetDontFragmentTag::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 1;
}
void
SocketSetDontFragmentTag::Serialize (TagBuffer i) const
{
  NS_LOG_FUNCTION (this << &i);
  i.WriteU8 (m_dontFragment ? 1 : 0);
}
void
SocketSetDontFragmentTag::Deserialize (TagBuffer i)
{
  NS_LOG_FUNCTION (this << &i);
  m_dontFragment = (i.ReadU8 () == 1);
}
void
SocketSetDontFragmentTag::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << (m_dontFragment ? "true" : "false");
}


SocketIpTosTag::SocketIpTosTag ()
{
}

void
SocketIpTosTag::SetTos (uint8_t ipTos)
{
  m_ipTos = ipTos;
}

uint8_t
SocketIpTosTag::GetTos (void) const
{
  return m_ipTos;
}

NS_OBJECT_ENSURE_REGISTERED (SocketIpTosTag);

TypeId
SocketIpTosTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketIpTosTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketIpTosTag> ()
    ;
  return tid;
}

TypeId
SocketIpTosTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
SocketIpTosTag::GetSerializedSize (void) const
{
  return sizeof (uint8_t);
}

void
SocketIpTosTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_ipTos);
}

void
SocketIpTosTag::Deserialize (TagBuffer i)
{
  m_ipTos = i.ReadU8();
}
void
SocketIpTosTag::Print (std::ostream &os) const
{
  os << "IP_TOS = " << m_ipTos;
}


SocketPriorityTag::SocketPriorityTag ()
{
}

void
SocketPriorityTag::SetPriority (uint8_t priority)
{
  m_priority = priority;
}

uint8_t
SocketPriorityTag::GetPriority (void) const
{
  return m_priority;
}

NS_OBJECT_ENSURE_REGISTERED (SocketPriorityTag);

TypeId
SocketPriorityTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketPriorityTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketPriorityTag> ()
    ;
  return tid;
}

TypeId
SocketPriorityTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
SocketPriorityTag::GetSerializedSize (void) const
{
  return sizeof (uint8_t);
}

void
SocketPriorityTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_priority);
}

void
SocketPriorityTag::Deserialize (TagBuffer i)
{
  m_priority = i.ReadU8();
}

void
SocketPriorityTag::Print (std::ostream &os) const
{
  os << "SO_PRIORITY = " << m_priority;
}


SocketIpv6TclassTag::SocketIpv6TclassTag ()
{
}

void
SocketIpv6TclassTag::SetTclass (uint8_t tclass)
{
  m_ipv6Tclass = tclass;
}

uint8_t
SocketIpv6TclassTag::GetTclass (void) const
{
  return m_ipv6Tclass;
}

NS_OBJECT_ENSURE_REGISTERED (SocketIpv6TclassTag);

TypeId
SocketIpv6TclassTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SocketIpv6TclassTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<SocketIpv6TclassTag> ()
    ;
  return tid;
}

TypeId
SocketIpv6TclassTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
SocketIpv6TclassTag::GetSerializedSize (void) const
{
  return sizeof (uint8_t);
}

void
SocketIpv6TclassTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_ipv6Tclass);
}

void
SocketIpv6TclassTag::Deserialize (TagBuffer i)
{
  m_ipv6Tclass = i.ReadU8();
}
void
SocketIpv6TclassTag::Print (std::ostream &os) const
{
  os << "IPV6_TCLASS = " << m_ipv6Tclass;
}

} // namespace ns3
