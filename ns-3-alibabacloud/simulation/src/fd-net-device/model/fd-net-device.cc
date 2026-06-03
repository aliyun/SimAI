/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 INRIA, 2012 University of Washington
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
 * Author: Alina Quereilhac <alina.quereilhac@inria.fr>
 *         Claudio Freire <klaussfreire@sourceforge.net>
 */

#include "fd-net-device.h"

#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/channel.h"
#include "ns3/enum.h"
#include "ns3/ethernet-header.h"
#include "ns3/ethernet-trailer.h"
#include "ns3/log.h"
#include "ns3/llc-snap-header.h"
#include "ns3/mac48-address.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"

#include <unistd.h>
#include <arpa/inet.h>
#include <net/ethernet.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FdNetDevice");

FdNetDeviceFdReader::FdNetDeviceFdReader ()
  : m_bufferSize (65536) // Defaults to maximum TCP window size
{
}

void
FdNetDeviceFdReader::SetBufferSize (uint32_t bufferSize)
{
  NS_LOG_FUNCTION (this << bufferSize);
  m_bufferSize = bufferSize;
}

FdReader::Data FdNetDeviceFdReader::DoRead (void)
{
  NS_LOG_FUNCTION (this);

  uint8_t *buf = (uint8_t *)malloc (m_bufferSize);
  NS_ABORT_MSG_IF (buf == 0, "malloc() failed");

  NS_LOG_LOGIC ("Calling read on fd " << m_fd);
  ssize_t len = read (m_fd, buf, m_bufferSize);
  if (len <= 0)
    {
      free (buf);
      buf = 0;
      len = 0;
    }
  NS_LOG_LOGIC ("Read " << len << " bytes on fd " << m_fd);
  return FdReader::Data (buf, len);
}

NS_OBJECT_ENSURE_REGISTERED (FdNetDevice);

TypeId
FdNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FdNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("FdNetDevice")
    .AddConstructor<FdNetDevice> ()
    .AddAttribute ("Address",
                   "The MAC address of this device.",
                   Mac48AddressValue (Mac48Address ("ff:ff:ff:ff:ff:ff")),
                   MakeMac48AddressAccessor (&FdNetDevice::m_address),
                   MakeMac48AddressChecker ())
    .AddAttribute ("Start",
                   "The simulation time at which to spin up "
                   "the device thread.",
                   TimeValue (Seconds (0.)),
                   MakeTimeAccessor (&FdNetDevice::m_tStart),
                   MakeTimeChecker ())
    .AddAttribute ("Stop",
                   "The simulation time at which to tear down "
                   "the device thread.",
                   TimeValue (Seconds (0.)),
                   MakeTimeAccessor (&FdNetDevice::m_tStop),
                   MakeTimeChecker ())
    .AddAttribute ("EncapsulationMode",
                   "The link-layer encapsulation type to use.",
                   EnumValue (DIX),
                   MakeEnumAccessor (&FdNetDevice::m_encapMode),
                   MakeEnumChecker (DIX, "Dix",
                                    LLC, "Llc",
                                    DIXPI, "DixPi"))
    .AddAttribute ("RxQueueSize", "Maximum size of the read queue.  "
                   "This value limits number of packets that have been read "
                   "from the network into a memory buffer but have not yet "
                   "been processed by the simulator.",
                   UintegerValue (1000),
                   MakeUintegerAccessor (&FdNetDevice::m_maxPendingReads),
                   MakeUintegerChecker<uint32_t> ())
    //
    // Trace sources at the "top" of the net device, where packets transition
    // to/from higher layers.  These points do not really correspond to the
    // MAC layer of the underlying operating system, but exist to provide
    // a consistent tracing environment.  These trace hooks should really be
    // interpreted as the points at which a packet leaves the ns-3 environment
    // destined for the underlying operating system or vice-versa.
    //
    .AddTraceSource ("MacTx",
                     "Trace source indicating a packet has "
                     "arrived for transmission by this device",
                     MakeTraceSourceAccessor (&FdNetDevice::m_macTxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacTxDrop",
                     "Trace source indicating a packet has "
                     "been dropped by the device before transmission",
                     MakeTraceSourceAccessor (&FdNetDevice::m_macTxDropTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacPromiscRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a promiscuous trace,",
                     MakeTraceSourceAccessor (&FdNetDevice::m_macPromiscRxTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("MacRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a non-promiscuous trace,",
                     MakeTraceSourceAccessor (&FdNetDevice::m_macRxTrace),
                     "ns3::Packet::TracedCallback")

    //
    // Trace sources designed to simulate a packet sniffer facility (tcpdump).
    //
    .AddTraceSource ("Sniffer",
                     "Trace source simulating a non-promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&FdNetDevice::m_snifferTrace),
                     "ns3::Packet::TracedCallback")
    .AddTraceSource ("PromiscSniffer",
                     "Trace source simulating a promiscuous "
                     "packet sniffer attached to the device",
                     MakeTraceSourceAccessor (&FdNetDevice::m_promiscSnifferTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

FdNetDevice::FdNetDevice ()
  : m_node (0),
    m_ifIndex (0),
    // Defaults to Ethernet v2 MTU
    m_mtu (1500),
    m_fd (-1),
    m_fdReader (0),
    m_isBroadcast (true),
    m_isMulticast (false),
    m_startEvent (),
    m_stopEvent ()
{
  NS_LOG_FUNCTION (this);
}

FdNetDevice::~FdNetDevice ()
{
  NS_LOG_FUNCTION (this);
}

void
FdNetDevice::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  Start (m_tStart);
  if (m_tStop != Seconds (0))
    {
        Stop (m_tStop);
    }

  NetDevice::DoInitialize ();
}

void
FdNetDevice::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  StopDevice ();
  NetDevice::DoDispose ();
}

void
FdNetDevice::SetEncapsulationMode (enum EncapsulationMode mode)
{
  NS_LOG_FUNCTION (this << mode);
  m_encapMode = mode;
  NS_LOG_LOGIC ("m_encapMode = " << m_encapMode);
}

FdNetDevice::EncapsulationMode
FdNetDevice::GetEncapsulationMode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_encapMode;
}

void
FdNetDevice::Start (Time tStart)
{
  NS_LOG_FUNCTION (this << tStart);
  Simulator::Cancel (m_startEvent);
  m_startEvent = Simulator::Schedule (tStart, &FdNetDevice::StartDevice, this);
}

void
FdNetDevice::Stop (Time tStop)
{
  NS_LOG_FUNCTION (this << tStop);
  Simulator::Cancel (m_stopEvent);
  m_stopEvent = Simulator::Schedule (tStop, &FdNetDevice::StopDevice, this);
}

void
FdNetDevice::StartDevice (void)
{
  NS_LOG_FUNCTION (this);

  if (m_fd == -1)
    {
      NS_LOG_DEBUG ("FdNetDevice::Start(): Failure, invalid file descriptor.");
      return;
    }

  m_fdReader = DoCreateFdReader ();
  m_fdReader->Start (m_fd, MakeCallback (&FdNetDevice::ReceiveCallback, this));

  DoFinishStartingDevice ();

  NotifyLinkUp ();
}

Ptr<FdReader>
FdNetDevice::DoCreateFdReader (void)
{
  NS_LOG_FUNCTION (this);

  Ptr<FdNetDeviceFdReader> fdReader = Create<FdNetDeviceFdReader> ();
  // 22 bytes covers 14 bytes Ethernet header with possible 8 bytes LLC/SNAP
  fdReader->SetBufferSize (m_mtu + 22);
  return fdReader;
}

void
FdNetDevice::DoFinishStartingDevice (void)
{
  NS_LOG_FUNCTION (this);
}

void
FdNetDevice::DoFinishStoppingDevice (void)
{
  NS_LOG_FUNCTION (this);
}

void
FdNetDevice::StopDevice (void)
{
  NS_LOG_FUNCTION (this);

  if (m_fdReader != 0)
    {
      m_fdReader->Stop ();
      m_fdReader = 0;
    }

  if (m_fd != -1)
    {
      close (m_fd);
      m_fd = -1;
    }

  while (!m_pendingQueue.empty ())
    {
      std::pair<uint8_t *, ssize_t> next = m_pendingQueue.front ();
      m_pendingQueue.pop ();
      FreeBuffer (next.first);
    }

  DoFinishStoppingDevice ();
}

void
FdNetDevice::ReceiveCallback (uint8_t *buf, ssize_t len)
{
  NS_LOG_FUNCTION (this << static_cast<void *> (buf) << len);
  bool skip = false;

  {
    std::unique_lock lock {m_pendingReadMutex};
    if (m_pendingQueue.size () >= m_maxPendingReads)
      {
        NS_LOG_WARN ("Packet dropped");
        skip = true;
      }
    else
      {
        m_pendingQueue.push (std::make_pair (buf, len));
      }
  }

  if (skip)
    {
      struct timespec time = {
        0, 100000000L
      };                                        // 100 ms
      nanosleep (&time, NULL);
    }
  else
    {
      Simulator::ScheduleWithContext (m_nodeId, Time (0), MakeEvent (&FdNetDevice::ForwardUp, this));
    }
}

/**
 * \ingroup fd-net-device
 * \brief Synthesize PI header for the kernel
 * \param buf the buffer to add the header to
 * \param len the buffer length
 *
 * \todo Consider having a instance member m_packetBuffer and using memmove
 * instead of memcpy to add the PI header. It might be faster in this case
 * to use memmove and avoid the extra mallocs.
 */
static void
AddPIHeader (uint8_t *&buf, size_t &len)
{
  // Synthesize PI header for our friend the kernel
  uint8_t *buf2 = (uint8_t*)malloc (len + 4);
  memcpy (buf2 + 4, buf, len);
  len += 4;

  // PI = 16 bits flags (0) + 16 bits proto
  // NOTE: be careful to interpret buffer data explicitly as
  //  little-endian to be insensible to native byte ordering.
  uint16_t flags = 0;
  uint16_t proto = 0x0008; // default to IPv4
  if (len > 14)
    {
      if (buf[12] == 0x81 && buf[13] == 0x00 && len > 18)
        {
          // tagged ethernet packet
          proto = buf[16] | (buf[17] << 8);
        }
      else
        {
          // untagged ethernet packet
          proto = buf[12] | (buf[13] << 8);
        }
    }
  buf2[0] = (uint8_t)flags;
  buf2[1] = (uint8_t)(flags >> 8);
  buf2[2] = (uint8_t)proto;
  buf2[3] = (uint8_t)(proto >> 8);

  // swap buffer
  free (buf);
  buf = buf2;
}

/**
 * \ingroup fd-net-device
 * \brief Removes PI header
 * \param buf the buffer to add the header to
 * \param len the buffer length
 */
static void
RemovePIHeader (uint8_t *&buf, ssize_t &len)
{
  // strip PI header if present, shrink buffer
  if (len >= 4)
    {
      len -= 4;
      memmove (buf, buf + 4, len);
      buf = (uint8_t*)realloc (buf, len);
    }
}

uint8_t *
FdNetDevice::AllocateBuffer(size_t len)
{
  return (uint8_t*) malloc(len);
}

void
FdNetDevice::FreeBuffer (uint8_t *buf)
{
  free (buf);
}

void
FdNetDevice::ForwardUp (void)
{
  NS_LOG_FUNCTION (this);

  uint8_t *buf = 0;
  ssize_t len = 0;

  if (m_pendingQueue.empty())
  {
    NS_LOG_LOGIC ("buffer is empty, probably the device is stopped.");
    return;
  }

  {
    std::unique_lock lock {m_pendingReadMutex};
    std::pair<uint8_t *, ssize_t> next = m_pendingQueue.front ();
    m_pendingQueue.pop ();

    buf = next.first;
    len = next.second;
  }

  NS_LOG_LOGIC ("buffer: " << static_cast<void *> (buf) << " length: " << len);

  // We need to remove the PI header and ignore it
  if (m_encapMode == DIXPI)
    {
      RemovePIHeader (buf, len);
    }

  //
  // Create a packet out of the buffer we received and free that buffer.
  //
  Ptr<Packet> packet = Create<Packet> (reinterpret_cast<const uint8_t *> (buf), len);
  FreeBuffer (buf);
  buf = 0;

  //
  // Trace sinks will expect complete packets, not packets without some of the
  // headers
  //
  Ptr<Packet> originalPacket = packet->Copy ();

  Mac48Address destination;
  Mac48Address source;
  uint16_t protocol;
  bool isBroadcast = false;
  bool isMulticast = false;

  EthernetHeader header (false);

  //
  // This device could be running in an environment where completely unexpected
  // kinds of packets are flying around, so we need to harden things a bit and
  // filter out packets we think are completely bogus, so we always check to see
  // that the packet is long enough to contain the header we want to remove.
  //
  if (packet->GetSize () < header.GetSerializedSize ())
    {
      m_phyRxDropTrace (originalPacket);
      return;
    }

  packet->RemoveHeader (header);
  destination = header.GetDestination ();
  source = header.GetSource ();
  isBroadcast = header.GetDestination ().IsBroadcast ();
  isMulticast = header.GetDestination ().IsGroup ();
  protocol = header.GetLengthType ();

  //
  // If the length/type is less than 1500, it corresponds to a length
  // interpretation packet.  In this case, it is an 802.3 packet and
  // will also have an 802.2 LLC header.  If greater than 1500, we
  // find the protocol number (Ethernet type) directly.
  //
  if (m_encapMode == LLC and header.GetLengthType () <= 1500)
    {
      LlcSnapHeader llc;
      //
      // Check to see that the packet is long enough to possibly contain the
      // header we want to remove before just naively calling.
      //
      if (packet->GetSize () < llc.GetSerializedSize ())
        {
          m_phyRxDropTrace (originalPacket);
          return;
        }

      packet->RemoveHeader (llc);
      protocol = llc.GetType ();
    }

  NS_LOG_LOGIC ("Pkt source is " << source);
  NS_LOG_LOGIC ("Pkt destination is " << destination);

  PacketType packetType;

  if (isBroadcast)
    {
      packetType = NS3_PACKET_BROADCAST;
    }
  else if (isMulticast)
    {
      packetType = NS3_PACKET_MULTICAST;
    }
  else if (destination == m_address)
    {
      packetType = NS3_PACKET_HOST;
    }
  else
    {
      packetType = NS3_PACKET_OTHERHOST;
    }

  //
  // For all kinds of packetType we receive, we hit the promiscuous sniffer
  // hook and pass a copy up to the promiscuous callback.  Pass a copy to
  // make sure that nobody messes with our packet.
  //
  m_promiscSnifferTrace (originalPacket);

  if (!m_promiscRxCallback.IsNull ())
    {
      m_macPromiscRxTrace (originalPacket);
      m_promiscRxCallback (this, packet, protocol, source, destination,
                           packetType);
    }

  //
  // If this packet is not destined for some other host, it must be for us
  // as either a broadcast, multicast or unicast.  We need to hit the mac
  // packet received trace hook and forward the packet up the stack.
  //
  if (packetType != NS3_PACKET_OTHERHOST)
    {
      m_snifferTrace (originalPacket);
      m_macRxTrace (originalPacket);
      m_rxCallback (this, packet, protocol, source);
    }
}

bool
FdNetDevice::Send (Ptr<Packet> packet, const Address& destination, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << packet << destination << protocolNumber);
  return SendFrom (packet, m_address, destination, protocolNumber);
}

bool
FdNetDevice::SendFrom (Ptr<Packet> packet, const Address& src, const Address& dest, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << packet << src << dest << protocolNumber);
  NS_LOG_LOGIC ("packet: " << packet << " UID: " << packet->GetUid ());

  if (IsLinkUp () == false)
    {
      m_macTxDropTrace (packet);
      return false;
    }

  Mac48Address destination = Mac48Address::ConvertFrom (dest);
  Mac48Address source = Mac48Address::ConvertFrom (src);

  NS_LOG_LOGIC ("Transmit packet with UID " << packet->GetUid ());
  NS_LOG_LOGIC ("Transmit packet from " << source);
  NS_LOG_LOGIC ("Transmit packet to " << destination);

  EthernetHeader header (false);
  header.SetSource (source);
  header.SetDestination (destination);

  NS_ASSERT_MSG (packet->GetSize () <= m_mtu, "FdNetDevice::SendFrom(): Packet too big " << packet->GetSize ());

  if (m_encapMode == LLC)
    {
      LlcSnapHeader llc;
      llc.SetType (protocolNumber);
      packet->AddHeader (llc);

      header.SetLengthType (packet->GetSize ());
    }
  else
    {
      header.SetLengthType (protocolNumber);
    }

  packet->AddHeader (header);

  //
  // there's not much meaning associated with the different layers in this
  // device, so don't be surprised when they're all stacked together in
  // essentially one place.  We do this for trace consistency across devices.
  //
  m_macTxTrace (packet);

  m_promiscSnifferTrace (packet);
  m_snifferTrace (packet);

  NS_LOG_LOGIC ("calling write");


  size_t len =  (size_t) packet->GetSize ();
  uint8_t *buffer = AllocateBuffer (len);
  if(!buffer)
  {
    m_macTxDropTrace(packet);
    return false;
  }

  packet->CopyData (buffer, len);

  // We need to add the PI header
  if (m_encapMode == DIXPI)
    {
      AddPIHeader (buffer, len);
    }

  ssize_t written = Write(buffer, len);
  FreeBuffer (buffer);

  if (written == -1 || (size_t) written != len)
    {
      m_macTxDropTrace (packet);
      return false;
    }

  return true;
}

ssize_t
FdNetDevice::Write (uint8_t *buffer, size_t length)
{
  NS_LOG_FUNCTION (this << static_cast<void *> (buffer) << length);

  uint32_t ret = write (m_fd, buffer, length);
  return ret;
}

void
FdNetDevice::SetFileDescriptor (int fd)
{
  if (m_fd == -1 and fd > 0)
    {
      m_fd = fd;
    }
}

int
FdNetDevice::GetFileDescriptor (void) const
{
  return m_fd;
}

void
FdNetDevice::SetAddress (Address address)
{
  m_address = Mac48Address::ConvertFrom (address);
}

Address
FdNetDevice::GetAddress (void) const
{
  return m_address;
}

void
FdNetDevice::NotifyLinkUp (void)
{
  m_linkUp = true;
  m_linkChangeCallbacks ();
}

void
FdNetDevice::SetIfIndex (const uint32_t index)
{
  m_ifIndex = index;
}

uint32_t
FdNetDevice::GetIfIndex (void) const
{
  return m_ifIndex;
}

Ptr<Channel>
FdNetDevice::GetChannel (void) const
{
  return NULL;
}

bool
FdNetDevice::SetMtu (const uint16_t mtu)
{
  // The MTU depends on the technology associated to
  // the file descriptor. The user is responsible of
  // setting the correct value of the MTU.
  // If the file descriptor is created using a helper,
  // then is the responsibility of the helper to set
  // the correct MTU value.
  m_mtu = mtu;
  return true;
}

uint16_t
FdNetDevice::GetMtu (void) const
{
  return m_mtu;
}

bool
FdNetDevice::IsLinkUp (void) const
{
  return m_linkUp;
}

void
FdNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  m_linkChangeCallbacks.ConnectWithoutContext (callback);
}

bool
FdNetDevice::IsBroadcast (void) const
{
  return m_isBroadcast;
}

void
FdNetDevice::SetIsBroadcast (bool broadcast)
{
  m_isBroadcast = broadcast;
}

Address
FdNetDevice::GetBroadcast (void) const
{
  return Mac48Address ("ff:ff:ff:ff:ff:ff");
}

bool
FdNetDevice::IsMulticast (void) const
{
  return m_isMulticast;
}

void
FdNetDevice::SetIsMulticast (bool multicast)
{
  m_isMulticast = multicast;
}

Address
FdNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  return Mac48Address::GetMulticast (multicastGroup);
}

Address
FdNetDevice::GetMulticast (Ipv6Address addr) const
{
  return Mac48Address::GetMulticast (addr);
}

bool
FdNetDevice::IsBridge (void) const
{
  return false;
}

bool
FdNetDevice::IsPointToPoint (void) const
{
  return false;
}

Ptr<Node>
FdNetDevice::GetNode (void) const
{
  return m_node;
}

void
FdNetDevice::SetNode (Ptr<Node> node)
{
  m_node = node;

  // Save the node ID for use in the read thread, to avoid having
  // to make a call to GetNode ()->GetId () that increments
  // Ptr<Node>'s reference count.
  m_nodeId = node->GetId ();
}

bool
FdNetDevice::NeedsArp (void) const
{
  return true;
}

void
FdNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  m_rxCallback = cb;
}

void
FdNetDevice::SetPromiscReceiveCallback (PromiscReceiveCallback cb)
{
  m_promiscRxCallback = cb;
}

bool
FdNetDevice::SupportsSendFrom (void) const
{
  return true;
}

} // namespace ns3
