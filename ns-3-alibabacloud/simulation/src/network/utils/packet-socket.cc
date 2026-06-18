/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 Emmanuelle Laprise, INRIA
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
 * Authors: Emmanuelle Laprise <emmanuelle.laprise@bluekazoo.ca>
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "packet-socket.h"
#include "packet-socket-address.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"

#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketSocket");

NS_OBJECT_ENSURE_REGISTERED (PacketSocket);

TypeId
PacketSocket::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketSocket")
    .SetParent<Socket> ()
    .SetGroupName("Network")
    .AddConstructor<PacketSocket> ()
    .AddTraceSource ("Drop", "Drop packet due to receive buffer overflow",
                     MakeTraceSourceAccessor (&PacketSocket::m_dropTrace),
                     "ns3::Packet::TracedCallback")
    .AddAttribute ("RcvBufSize",
                   "PacketSocket maximum receive buffer size (bytes)",
                   UintegerValue (131072),
                   MakeUintegerAccessor (&PacketSocket::m_rcvBufSize),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}

PacketSocket::PacketSocket () : m_rxAvailable (0)
{
  NS_LOG_FUNCTION (this);
  m_state = STATE_OPEN;
  m_shutdownSend = false;
  m_shutdownRecv = false;
  m_errno = ERROR_NOTERROR;
  m_isSingleDevice = false;
  m_device = 0;
}

void 
PacketSocket::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

PacketSocket::~PacketSocket ()
{
  NS_LOG_FUNCTION (this);
}

void 
PacketSocket::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_device = 0;
}

enum Socket::SocketErrno
PacketSocket::GetErrno (void) const
{
  NS_LOG_FUNCTION (this);
  return m_errno;
}

enum Socket::SocketType
PacketSocket::GetSocketType (void) const
{
  NS_LOG_FUNCTION (this);
  return NS3_SOCK_RAW;
}

Ptr<Node>
PacketSocket::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

int
PacketSocket::Bind (void)
{
  NS_LOG_FUNCTION (this);
  PacketSocketAddress address;
  address.SetProtocol (0);
  address.SetAllDevices ();
  return DoBind (address);
}

int
PacketSocket::Bind6 (void)
{
  NS_LOG_FUNCTION (this);
  return(Bind());
}

int
PacketSocket::Bind (const Address &address)
{ 
  NS_LOG_FUNCTION (this << address);
  if (!PacketSocketAddress::IsMatchingType (address))
    {
      m_errno = ERROR_INVAL;
      return -1;
    }
  PacketSocketAddress ad = PacketSocketAddress::ConvertFrom (address);
  return DoBind (ad);
}

int
PacketSocket::DoBind (const PacketSocketAddress &address)
{
  NS_LOG_FUNCTION (this << address);
  if (m_state == STATE_BOUND ||
      m_state == STATE_CONNECTED)
    {
      m_errno = ERROR_INVAL;
      return -1;
    }
  if (m_state == STATE_CLOSED)
    {
      m_errno = ERROR_BADF;
      return -1;
    }
  Ptr<NetDevice> dev;
  if (address.IsSingleDevice ())
    {
      dev = m_node->GetDevice (address.GetSingleDevice ());
    }
  else
    {
      dev = 0;
    }
  m_node->RegisterProtocolHandler (MakeCallback (&PacketSocket::ForwardUp, this),
                                   address.GetProtocol (), dev);
  m_state = STATE_BOUND;
  m_protocol = address.GetProtocol ();
  m_isSingleDevice = address.IsSingleDevice ();
  m_device = address.GetSingleDevice ();
  m_boundnetdevice = dev;
  return 0;
}

int
PacketSocket::ShutdownSend (void)
{
  NS_LOG_FUNCTION (this);
  if (m_state == STATE_CLOSED)
    {
      m_errno = ERROR_BADF;
      return -1;
    }
  m_shutdownSend = true;
  return 0;
}

int
PacketSocket::ShutdownRecv (void)
{
  NS_LOG_FUNCTION (this);
  if (m_state == STATE_CLOSED)
    {
      m_errno = ERROR_BADF;
      return -1;
    }
  m_shutdownRecv = true;
  return 0;
}

int
PacketSocket::Close (void)
{
  NS_LOG_FUNCTION (this);
  if (m_state == STATE_CLOSED)
    {
      m_errno = ERROR_BADF;
      return -1;
    }
  else if (m_state == STATE_BOUND || m_state == STATE_CONNECTED)
    {
      m_node->UnregisterProtocolHandler (MakeCallback (&PacketSocket::ForwardUp, this));
    }
  m_state = STATE_CLOSED;
  m_shutdownSend = true;
  m_shutdownRecv = true;
  return 0;
}

int
PacketSocket::Connect (const Address &ad)
{
  NS_LOG_FUNCTION (this << ad);
  PacketSocketAddress address;
  if (m_state == STATE_CLOSED)
    {
      m_errno = ERROR_BADF;
      goto error;
    }
  if (m_state == STATE_OPEN)
    {
      // connect should happen _after_ bind.
      m_errno = ERROR_INVAL; // generic error condition.
      goto error;
    }
  if (m_state == STATE_CONNECTED)
    {
      m_errno = ERROR_ISCONN;
      goto error;
    }
  if (!PacketSocketAddress::IsMatchingType (ad))
    {
      m_errno = ERROR_AFNOSUPPORT;
      goto error;
    }
  m_destAddr = ad;
  m_state = STATE_CONNECTED;
  NotifyConnectionSucceeded ();
  return 0;
error:
  NotifyConnectionFailed ();
  return -1;
}
int 
PacketSocket::Listen (void)
{
  NS_LOG_FUNCTION (this);
  m_errno = Socket::ERROR_OPNOTSUPP;
  return -1;
}

int
PacketSocket::Send (Ptr<Packet> p, uint32_t flags)
{
  NS_LOG_FUNCTION (this << p << flags);
  if (m_state == STATE_OPEN ||
      m_state == STATE_BOUND)
    {
      m_errno = ERROR_NOTCONN;
      return -1;
    }
  return SendTo (p, flags, m_destAddr);
}

uint32_t
PacketSocket::GetMinMtu (PacketSocketAddress ad) const
{
  NS_LOG_FUNCTION (this << ad);
  if (ad.IsSingleDevice ())
    {
      Ptr<NetDevice> device = m_node->GetDevice (ad.GetSingleDevice ());
      return device->GetMtu ();
    }
  else
    {
      uint32_t minMtu = 0xffff;
      for (uint32_t i = 0; i < m_node->GetNDevices (); i++)
        {
          Ptr<NetDevice> device = m_node->GetDevice (i);
          minMtu = std::min (minMtu, (uint32_t)device->GetMtu ());
        }
      return minMtu;
    }
}

uint32_t 
PacketSocket::GetTxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  if (m_state == STATE_CONNECTED)
    {
      PacketSocketAddress ad = PacketSocketAddress::ConvertFrom (m_destAddr);
      return GetMinMtu (ad);
    }
  // If we are not connected, we return a 'safe' value by default.
  return 0xffff;
}

int
PacketSocket::SendTo (Ptr<Packet> p, uint32_t flags, const Address &address)
{
  NS_LOG_FUNCTION (this << p << flags << address);
  PacketSocketAddress ad;
  if (m_state == STATE_CLOSED)
    {
      NS_LOG_LOGIC ("ERROR_BADF");
      m_errno = ERROR_BADF;
      return -1;
    }
  if (m_shutdownSend)
    {
      NS_LOG_LOGIC ("ERROR_SHUTDOWN");
      m_errno = ERROR_SHUTDOWN;
      return -1;
    }
  if (!PacketSocketAddress::IsMatchingType (address))
    {
      NS_LOG_LOGIC ("ERROR_AFNOSUPPORT");
      m_errno = ERROR_AFNOSUPPORT;
      return -1;
    }
  ad = PacketSocketAddress::ConvertFrom (address);
  if (p->GetSize () > GetMinMtu (ad))
    {
      m_errno = ERROR_MSGSIZE;
      return -1;
    }

  uint8_t priority = GetPriority ();
  if (priority)
    {
      SocketPriorityTag priorityTag;
      priorityTag.SetPriority (priority);
      p->ReplacePacketTag (priorityTag);
    }

  bool error = false;
  Address dest = ad.GetPhysicalAddress ();
  uint32_t pktSize = p->GetSize (); // device->Send() may modify the packet
  if (ad.IsSingleDevice ())
    {
      Ptr<NetDevice> device = m_node->GetDevice (ad.GetSingleDevice ());
      if (!device->Send (p, dest, ad.GetProtocol ()))
        {
          NS_LOG_LOGIC ("error: NetDevice::Send error");
          error = true;
        }
    }
  else
    {
      for (uint32_t i = 0; i < m_node->GetNDevices (); i++)
        {
          Ptr<NetDevice> device = m_node->GetDevice (i);
          if (!device->Send (p, dest, ad.GetProtocol ()))
            {
              NS_LOG_LOGIC ("error: NetDevice::Send error");
              error = true;
            }
        }
    }
  if (!error)
    {
      NotifyDataSent (pktSize);
      NotifySend (GetTxAvailable ());
    }

  if (error)
    {
      NS_LOG_LOGIC ("ERROR_INVAL 2");
      m_errno = ERROR_INVAL;
      return -1;
    }
  else
    {
      return pktSize;
    }
}

void 
PacketSocket::ForwardUp (Ptr<NetDevice> device, Ptr<const Packet> packet, 
                         uint16_t protocol, const Address &from,
                         const Address &to, NetDevice::PacketType packetType)
{
  NS_LOG_FUNCTION (this << device << packet << protocol << from << to << packetType);
  if (m_shutdownRecv)
    {
      return;
    }
  PacketSocketAddress address;
  address.SetPhysicalAddress (from);
  address.SetSingleDevice (device->GetIfIndex ());
  address.SetProtocol (protocol);

  if ((m_rxAvailable + packet->GetSize ()) <= m_rcvBufSize)
    {
      Ptr<Packet> copy = packet->Copy ();
      DeviceNameTag dnt;
      dnt.SetDeviceName (device->GetTypeId ().GetName ());
      PacketSocketTag pst;
      pst.SetPacketType (packetType);
      pst.SetDestAddress (to);
      copy->AddPacketTag (pst); // Attach Packet Type and Dest Address
      copy->AddPacketTag (dnt); // Attach device source name
      // in case the packet still has a priority tag, remove it
      SocketPriorityTag priorityTag;
      copy->RemovePacketTag (priorityTag);
      m_deliveryQueue.push (std::make_pair (copy, address));
      m_rxAvailable += packet->GetSize ();
      NS_LOG_LOGIC ("UID is " << packet->GetUid () << " PacketSocket " << this);
      NotifyDataRecv ();
    }
  else
    {
      // In general, this case should not occur unless the
      // receiving application reads data from this socket slowly
      // in comparison to the arrival rate
      //
      // drop and trace packet
      NS_LOG_WARN ("No receive buffer space available.  Drop.");
      m_dropTrace (packet);
    }
}

uint32_t
PacketSocket::GetRxAvailable (void) const
{
  NS_LOG_FUNCTION (this);
  // We separately maintain this state to avoid walking the queue 
  // every time this might be called
  return m_rxAvailable;
}

Ptr<Packet> 
PacketSocket::Recv (uint32_t maxSize, uint32_t flags)
{
  NS_LOG_FUNCTION (this << maxSize << flags);

  Address fromAddress;
  Ptr<Packet> packet = RecvFrom (maxSize, flags, fromAddress);
  return packet;
}

Ptr<Packet>
PacketSocket::RecvFrom (uint32_t maxSize, uint32_t flags, Address &fromAddress)
{
  NS_LOG_FUNCTION (this << maxSize << flags);

  if (m_deliveryQueue.empty () )
    {
      return 0;
    }
  Ptr<Packet> p = m_deliveryQueue.front ().first;
  fromAddress = m_deliveryQueue.front ().second;

  if (p->GetSize () <= maxSize)
    {
      m_deliveryQueue.pop ();
      m_rxAvailable -= p->GetSize ();
    }
  else
    {
      p = 0;
    }
  return p;
}

int
PacketSocket::GetSockName (Address &address) const
{
  NS_LOG_FUNCTION (this << address);
  PacketSocketAddress ad;

  ad.SetProtocol (m_protocol);
  if (m_isSingleDevice)
    {
      Ptr<NetDevice> device = m_node->GetDevice (m_device);
      ad.SetPhysicalAddress (device->GetAddress ());
      ad.SetSingleDevice (m_device);
    }
  else
    {
      ad.SetPhysicalAddress (Address ());
      ad.SetAllDevices ();
    }
  address = ad;

  return 0;
}

int
PacketSocket::GetPeerName (Address &address) const
{
  NS_LOG_FUNCTION (this << address);

  if (m_state != STATE_CONNECTED)
    {
      m_errno = ERROR_NOTCONN;
      return -1;
    }

  address = m_destAddr;

  return 0;
}

bool
PacketSocket::SetAllowBroadcast (bool allowBroadcast)
{
  NS_LOG_FUNCTION (this << allowBroadcast);
  if (allowBroadcast)
    {
      return false;
    }
  return true;
}

bool
PacketSocket::GetAllowBroadcast () const
{
  NS_LOG_FUNCTION (this);
  return false;
}

/***************************************************************
 *           PacketSocket Tags
 ***************************************************************/

PacketSocketTag::PacketSocketTag ()
{
}

void
PacketSocketTag::SetPacketType(NetDevice::PacketType t)
{
  m_packetType = t;
}

NetDevice::PacketType
PacketSocketTag::GetPacketType (void) const
{
  return m_packetType;
}

void
PacketSocketTag::SetDestAddress(Address a)
{
  m_destAddr = a;
}

Address
PacketSocketTag::GetDestAddress (void) const
{
  return m_destAddr;
}

NS_OBJECT_ENSURE_REGISTERED (PacketSocketTag);

TypeId
PacketSocketTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketSocketTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<PacketSocketTag> ()
  ;
  return tid;
}
TypeId
PacketSocketTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t
PacketSocketTag::GetSerializedSize (void) const
{
  return  1 + m_destAddr.GetSerializedSize();
}
void
PacketSocketTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_packetType);
  m_destAddr.Serialize (i);
}
void
PacketSocketTag::Deserialize (TagBuffer i)
{
  m_packetType = (NetDevice::PacketType) i.ReadU8 ();
  m_destAddr.Deserialize (i);
}
void
PacketSocketTag::Print (std::ostream &os) const
{
  os << "packetType=" << m_packetType;
}

/***************************************************************
 *           DeviceName Tags
 ***************************************************************/

DeviceNameTag::DeviceNameTag ()
{
}

void
DeviceNameTag::SetDeviceName (std::string n)
{
  if ( n.substr(0,5) == "ns3::" )
    {
      n = n.substr (5);
    }
  m_deviceName = n;
}

std::string
DeviceNameTag::GetDeviceName (void) const
{
  return m_deviceName;
}

NS_OBJECT_ENSURE_REGISTERED (DeviceNameTag);

TypeId
DeviceNameTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DeviceNameTag")
    .SetParent<Tag> ()
    .SetGroupName("Network")
    .AddConstructor<DeviceNameTag> ();
  return tid;
}
TypeId
DeviceNameTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t
DeviceNameTag::GetSerializedSize (void) const
{
  uint32_t s = 1 + m_deviceName.size();  // +1 for name length field
  return s;
}
void
DeviceNameTag::Serialize (TagBuffer i) const
{
  const char *n = m_deviceName.c_str();
  uint8_t l = (uint8_t) m_deviceName.size ();

  i.WriteU8 (l);
  i.Write ( (uint8_t*) n , (uint32_t) l);
}
void
DeviceNameTag::Deserialize (TagBuffer i)
{
  uint8_t l = i.ReadU8();
  char buf[256];

  i.Read ( (uint8_t* ) buf, (uint32_t) l);
  m_deviceName = std::string (buf, l);
}
void
DeviceNameTag::Print (std::ostream &os) const
{
  os << "DeviceName=" << m_deviceName;
}


} // namespace ns3
