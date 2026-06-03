/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze, Italy
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
 *         Michele Muccio <michelemuccio@virgilio.it>
 */

#include <algorithm>

#include "ns3/node.h"
#include "ns3/channel.h"
#include "ns3/packet.h"
#include "ns3/log.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"
#include "ns3/mac16-address.h"
#include "ns3/mac48-address.h"
#include "ns3/mac64-address.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/ipv6-extension-header.h"
#include "ns3/udp-header.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "sixlowpan-net-device.h"
#include "sixlowpan-header.h"

NS_LOG_COMPONENT_DEFINE ("SixLowPanNetDevice");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (SixLowPanNetDevice);

TypeId SixLowPanNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SixLowPanNetDevice")
    .SetParent<NetDevice> ()
    .SetGroupName ("SixLowPan")
    .AddConstructor<SixLowPanNetDevice> ()
    .AddAttribute ("Rfc6282", "Use RFC6282 (IPHC) if true, RFC4944 (HC1) otherwise.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&SixLowPanNetDevice::m_useIphc),
                   MakeBooleanChecker ())
    .AddAttribute ("OmitUdpChecksum",
                   "Omit the UDP checksum in IPHC compression.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&SixLowPanNetDevice::m_omitUdpChecksum),
                   MakeBooleanChecker ())
    .AddAttribute ("FragmentReassemblyListSize", "The maximum size of the reassembly buffer (in packets). Zero meaning infinite.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&SixLowPanNetDevice::m_fragmentReassemblyListSize),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("FragmentExpirationTimeout",
                   "When this timeout expires, the fragments will be cleared from the buffer.",
                   TimeValue (Seconds (60)),
                   MakeTimeAccessor (&SixLowPanNetDevice::m_fragmentExpirationTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("CompressionThreshold",
                   "The minimum MAC layer payload size.",
                   UintegerValue (0x0),
                   MakeUintegerAccessor (&SixLowPanNetDevice::m_compressionThreshold),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("ForceEtherType",
                   "Force a specific EtherType in L2 frames.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&SixLowPanNetDevice::m_forceEtherType),
                   MakeBooleanChecker ())
    .AddAttribute ("EtherType",
                   "The specific EtherType to be used in L2 frames.",
                   UintegerValue (0xFFFF),
                   MakeUintegerAccessor (&SixLowPanNetDevice::m_etherType),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("UseMeshUnder",
                   "Use a mesh-under routing protocol.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&SixLowPanNetDevice::m_meshUnder),
                   MakeBooleanChecker ())
    .AddAttribute ("MeshUnderRadius",
                   "Hops Left to use in mesh-under.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&SixLowPanNetDevice::m_meshUnderHopsLeft),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("MeshCacheLength",
                   "Length of the cache for each source.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&SixLowPanNetDevice::m_meshCacheLength),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("MeshUnderJitter",
                   "The jitter in ms a node uses to forward mesh-under packets - used to prevent collisions",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=10.0]"),
                   MakePointerAccessor (&SixLowPanNetDevice::m_meshUnderJitter),
                   MakePointerChecker<RandomVariableStream> ())
    .AddTraceSource ("Tx",
                     "Send - packet (including 6LoWPAN header), "
                     "SixLoWPanNetDevice Ptr, interface index.",
                     MakeTraceSourceAccessor (&SixLowPanNetDevice::m_txTrace),
                     "ns3::SixLowPanNetDevice::RxTxTracedCallback")
    .AddTraceSource ("Rx",
                     "Receive - packet (including 6LoWPAN header), "
                     "SixLoWPanNetDevice Ptr, interface index.",
                     MakeTraceSourceAccessor (&SixLowPanNetDevice::m_rxTrace),
                     "ns3::SixLowPanNetDevice::RxTxTracedCallback")
    .AddTraceSource ("Drop",
                     "Drop - DropReason, packet (including 6LoWPAN header), "
                     "SixLoWPanNetDevice Ptr, interface index.",
                     MakeTraceSourceAccessor (&SixLowPanNetDevice::m_dropTrace),
                     "ns3::SixLowPanNetDevice::DropTracedCallback")
  ;
  return tid;
}

SixLowPanNetDevice::SixLowPanNetDevice ()
  : m_node (0),
  m_netDevice (0),
  m_ifIndex (0)
{
  NS_LOG_FUNCTION (this);
  m_netDevice = 0;
  m_rng = CreateObject<UniformRandomVariable> ();
  m_bc0Serial = 0;
}

Ptr<NetDevice> SixLowPanNetDevice::GetNetDevice () const
{
  NS_LOG_FUNCTION (this);
  return m_netDevice;
}

void SixLowPanNetDevice::SetNetDevice (Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  m_netDevice = device;

  NS_LOG_DEBUG ("RegisterProtocolHandler for " << device->GetInstanceTypeId ().GetName ());

  uint16_t protocolType = 0;
  if ( m_forceEtherType )
    {
      protocolType = m_etherType;
    }
  m_node->RegisterProtocolHandler (MakeCallback (&SixLowPanNetDevice::ReceiveFromDevice,
                                                 this),
                                   protocolType, device, false);
}

int64_t SixLowPanNetDevice::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_rng->SetStream (stream);
  m_meshUnderJitter->SetStream (stream + 1);
  return 2;
}

void SixLowPanNetDevice::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  m_netDevice = 0;
  m_node = 0;

  m_timeoutEventList.clear ();
  if (m_timeoutEvent.IsRunning ())
    {
      m_timeoutEvent.Cancel ();
    }

  for (MapFragmentsI_t iter = m_fragments.begin (); iter != m_fragments.end (); iter++)
    {
      iter->second = 0;
    }
  m_fragments.clear ();

  NetDevice::DoDispose ();
}

void SixLowPanNetDevice::ReceiveFromDevice (Ptr<NetDevice> incomingPort,
                                            Ptr<const Packet> packet,
                                            uint16_t protocol,
                                            Address const &src,
                                            Address const &dst,
                                            PacketType packetType)
{
  NS_LOG_FUNCTION (this << incomingPort << packet << protocol << src << dst);

  uint8_t dispatchRawVal = 0;
  SixLowPanDispatch::Dispatch_e dispatchVal;
  Ptr<Packet> copyPkt = packet->Copy ();

  m_rxTrace (copyPkt, this, GetIfIndex ());

  copyPkt->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
  dispatchVal = SixLowPanDispatch::GetDispatchType (dispatchRawVal);
  bool isPktDecompressed = false;
  bool fragmented = false;

  NS_LOG_DEBUG ( "Packet received: " << *copyPkt );
  NS_LOG_DEBUG ( "Packet length: " << copyPkt->GetSize () );
  NS_LOG_DEBUG ( "Dispatches: " << int(dispatchRawVal) << " - " << int(dispatchVal) );

  SixLowPanMesh meshHdr;
  SixLowPanBc0 bc0Hdr;
  bool hasMesh = false;
  bool hasBc0 = false;

  if ( dispatchVal == SixLowPanDispatch::LOWPAN_MESH )
    {
      hasMesh = true;
      copyPkt->RemoveHeader (meshHdr);
      copyPkt->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
      dispatchVal = SixLowPanDispatch::GetDispatchType (dispatchRawVal);
    }
  if ( dispatchVal == SixLowPanDispatch::LOWPAN_BC0 )
    {
      hasBc0 = true;
      copyPkt->RemoveHeader (bc0Hdr);
      copyPkt->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
      dispatchVal = SixLowPanDispatch::GetDispatchType (dispatchRawVal);
    }

  if (hasMesh)
    {
      if (!hasBc0)
        {
          NS_LOG_LOGIC ("Dropped packet - we only support mesh if it is paired with a BC0");
          m_dropTrace (DROP_UNKNOWN_EXTENSION, copyPkt, this, GetIfIndex ());
          return;
        }

      if (find (m_seenPkts[meshHdr.GetOriginator ()].begin (),
                m_seenPkts[meshHdr.GetOriginator ()].end (),
                bc0Hdr.GetSequenceNumber ()) != m_seenPkts[meshHdr.GetOriginator ()].end ())
        {
          NS_LOG_LOGIC ("We have already seen this, no further processing.");
          return;
        }

      m_seenPkts[meshHdr.GetOriginator ()].push_back (bc0Hdr.GetSequenceNumber ());
      if (m_seenPkts[meshHdr.GetOriginator ()].size () > m_meshCacheLength)
        {
          m_seenPkts[meshHdr.GetOriginator ()].pop_front ();
        }

      NS_ABORT_MSG_IF (!Mac16Address::IsMatchingType (meshHdr.GetFinalDst ()), "SixLowPan mesh-under flooding can not currently handle extended address final destinations: " << meshHdr.GetFinalDst ());
      NS_ABORT_MSG_IF (!Mac48Address::IsMatchingType (m_netDevice->GetAddress ()), "SixLowPan mesh-under flooding can not currently handle devices using extended addresses: " << m_netDevice->GetAddress ());

      Mac16Address finalDst = Mac16Address::ConvertFrom (meshHdr.GetFinalDst ());

      // See if the packet is for others than me. In case forward it.
      if (meshHdr.GetFinalDst () != Get16MacFrom48Mac (m_netDevice->GetAddress ())
          || finalDst.IsBroadcast ()
          || finalDst.IsMulticast ()
          )
        {
          uint8_t hopsLeft = meshHdr.GetHopsLeft ();

          if (hopsLeft == 0)
            {
              NS_LOG_LOGIC ("Not forwarding packet -- hop limit reached");
            }
          else if (meshHdr.GetOriginator () == Get16MacFrom48Mac (m_netDevice->GetAddress ()))
            {
              NS_LOG_LOGIC ("Not forwarding packet -- I am the originator");
            }
          else
            {
              meshHdr.SetHopsLeft (hopsLeft - 1);
              Ptr<Packet> sendPkt = copyPkt->Copy ();
              sendPkt->AddHeader (bc0Hdr);
              sendPkt->AddHeader (meshHdr);
              Simulator::Schedule (Time (MilliSeconds (m_meshUnderJitter->GetValue ())), &NetDevice::Send, m_netDevice, sendPkt, m_netDevice->GetBroadcast (), protocol);
            }

          if (!finalDst.IsBroadcast () && !finalDst.IsMulticast ())
            {
              return;
            }
        }
    }

  Address realDst = dst;
  Address realSrc = src;
  if (hasMesh)
    {
      realSrc = meshHdr.GetOriginator ();
      realDst = meshHdr.GetFinalDst ();
    }

  if ( dispatchVal == SixLowPanDispatch::LOWPAN_FRAG1 )
    {
      isPktDecompressed = ProcessFragment (copyPkt, realSrc, realDst, true);
      fragmented = true;
    }
  else if ( dispatchVal == SixLowPanDispatch::LOWPAN_FRAGN )
    {
      isPktDecompressed = ProcessFragment (copyPkt, realSrc, realDst, false);
      fragmented = true;
    }
  if ( fragmented )
    {
      if ( !isPktDecompressed )
        {
          return;
        }
      else
        {
          copyPkt->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
          dispatchVal = SixLowPanDispatch::GetDispatchType (dispatchRawVal);
        }
    }

  switch ( dispatchVal )
    {
    case SixLowPanDispatch::LOWPAN_IPv6:
      NS_LOG_DEBUG ( "Packet without compression. Length: " << copyPkt->GetSize () );
      {
        SixLowPanIpv6 uncompressedHdr;
        copyPkt->RemoveHeader (uncompressedHdr);
        isPktDecompressed = true;
      }
      break;
    case SixLowPanDispatch::LOWPAN_HC1:
      if (m_useIphc)
        {
          m_dropTrace (DROP_DISALLOWED_COMPRESSION, copyPkt, this, GetIfIndex ());
          return;
        }
      DecompressLowPanHc1 (copyPkt, realSrc, realDst);
      isPktDecompressed = true;
      break;
    case SixLowPanDispatch::LOWPAN_IPHC:
      if (!m_useIphc)
        {
          m_dropTrace (DROP_DISALLOWED_COMPRESSION, copyPkt, this, GetIfIndex ());
          return;
        }
      if (DecompressLowPanIphc (copyPkt, realSrc, realDst))
        {
          m_dropTrace (DROP_SATETFUL_DECOMPRESSION_PROBLEM, copyPkt, this, GetIfIndex ());
        }
      else
        {
          isPktDecompressed = true;
        }
      break;
    default:
      NS_LOG_DEBUG ("Unsupported 6LoWPAN encoding: dropping.");
      m_dropTrace (DROP_UNKNOWN_EXTENSION, copyPkt, this, GetIfIndex ());
      break;
    }

  if ( !isPktDecompressed )
    {
      return;
    }

  NS_LOG_DEBUG ( "Packet decompressed length: " << copyPkt->GetSize () );
  NS_LOG_DEBUG ( "Packet decompressed received: " << *copyPkt );

  if (!m_promiscRxCallback.IsNull ())
    {
      m_promiscRxCallback (this, copyPkt, Ipv6L3Protocol::PROT_NUMBER, realSrc, realDst, packetType);
    }

  m_rxCallback (this, copyPkt, Ipv6L3Protocol::PROT_NUMBER, realSrc);

  return;
}

void SixLowPanNetDevice::SetIfIndex (const uint32_t index)
{
  NS_LOG_FUNCTION (this << index);
  m_ifIndex = index;
}

uint32_t SixLowPanNetDevice::GetIfIndex (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ifIndex;
}

Ptr<Channel> SixLowPanNetDevice::GetChannel (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->GetChannel ();
}

void SixLowPanNetDevice::SetAddress (Address address)
{
  NS_LOG_FUNCTION (this << address);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  m_netDevice->SetAddress (address);
}

Address SixLowPanNetDevice::GetAddress (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->GetAddress ();
}

bool SixLowPanNetDevice::SetMtu (const uint16_t mtu)
{
  NS_LOG_FUNCTION (this << mtu);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->SetMtu (mtu);
}

uint16_t SixLowPanNetDevice::GetMtu (void) const
{
  NS_LOG_FUNCTION (this);

  uint16_t mtu = m_netDevice->GetMtu ();

  // RFC 4944, section 4.
  if (mtu < 1280)
    {
      mtu = 1280;
    }
  return mtu;
}

bool SixLowPanNetDevice::IsLinkUp (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->IsLinkUp ();
}

void SixLowPanNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->AddLinkChangeCallback (callback);
}

bool SixLowPanNetDevice::IsBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->IsBroadcast ();
}

Address SixLowPanNetDevice::GetBroadcast (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->GetBroadcast ();
}

bool SixLowPanNetDevice::IsMulticast (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->IsMulticast ();
}

Address SixLowPanNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (this << multicastGroup);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->GetMulticast (multicastGroup);
}

Address SixLowPanNetDevice::GetMulticast (Ipv6Address addr) const
{
  NS_LOG_FUNCTION (this << addr);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->GetMulticast (addr);
}

bool SixLowPanNetDevice::IsPointToPoint (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->IsPointToPoint ();
}

bool SixLowPanNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->IsBridge ();
}

bool SixLowPanNetDevice::Send (Ptr<Packet> packet,
                               const Address& dest,
                               uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << *packet << dest << protocolNumber);
  bool ret = false;
  Address src;

  ret = DoSend (packet, src, dest, protocolNumber, false);
  return ret;
}

bool SixLowPanNetDevice::SendFrom (Ptr<Packet> packet,
                                   const Address& src,
                                   const Address& dest,
                                   uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << *packet << src << dest << protocolNumber);
  bool ret = false;

  ret = DoSend (packet, src, dest, protocolNumber, true);
  return ret;
}

bool SixLowPanNetDevice::DoSend (Ptr<Packet> packet,
                                 const Address& src,
                                 const Address& dest,
                                 uint16_t protocolNumber,
                                 bool doSendFrom)
{
  NS_LOG_FUNCTION (this << *packet << src << dest << protocolNumber << doSendFrom);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  Ptr<Packet> origPacket = packet->Copy ();
  uint32_t origHdrSize = 0;
  uint32_t origPacketSize = packet->GetSize ();
  bool ret = false;

  Address destination = dest;

  bool useMesh = m_meshUnder;

  if (m_forceEtherType)
    {
      protocolNumber = m_etherType;
    }

  if (m_useIphc)
    {
      NS_LOG_LOGIC ("Compressing packet using IPHC");
      origHdrSize += CompressLowPanIphc (packet, m_netDevice->GetAddress (), destination);
    }
  else
    {
      NS_LOG_LOGIC ("Compressing packet using HC1");
      origHdrSize += CompressLowPanHc1 (packet, m_netDevice->GetAddress (), destination);
    }

  uint16_t pktSize = packet->GetSize ();

  SixLowPanMesh meshHdr;
  SixLowPanBc0 bc0Hdr;
  uint32_t extraHdrSize = 0;

  if (useMesh)
    {
      Address source = src;
      if (!doSendFrom)
        {
          source = m_netDevice->GetAddress ();
        }

      if (Mac48Address::IsMatchingType (source))
        {
          // We got a Mac48 pseudo-MAC. We need its original Mac16 here.
          source = Get16MacFrom48Mac (source);
        }
      if (Mac48Address::IsMatchingType (destination))
        {
          // We got a Mac48 pseudo-MAC. We need its original Mac16 here.
          destination = Get16MacFrom48Mac (destination);
        }

      meshHdr.SetOriginator (source);
      meshHdr.SetFinalDst (destination);
      meshHdr.SetHopsLeft (m_meshUnderHopsLeft);
      destination = m_netDevice->GetBroadcast ();
      // We are storing sum of mesh and bc0 header sizes. We will need it if packet is fragmented.
      extraHdrSize = meshHdr.GetSerializedSize () + bc0Hdr.GetSerializedSize ();
      pktSize +=  extraHdrSize;
    }

  if (pktSize < m_compressionThreshold)
    {
      NS_LOG_LOGIC ("Compressed packet too short, using uncompressed one");
      packet = origPacket;
      SixLowPanIpv6 ipv6UncompressedHdr;
      packet->AddHeader (ipv6UncompressedHdr);
      pktSize = packet->GetSize ();
      if (useMesh)
        {
          pktSize += meshHdr.GetSerializedSize () + bc0Hdr.GetSerializedSize ();
        }
    }


  if (pktSize > m_netDevice->GetMtu ())
    {
      NS_LOG_LOGIC ("Fragmentation: Packet size " << packet->GetSize () << " - Mtu " << m_netDevice->GetMtu () );
      // fragment
      std::list<Ptr<Packet> > fragmentList;
      DoFragmentation (packet, origPacketSize, origHdrSize, extraHdrSize, fragmentList);
      std::list<Ptr<Packet> >::iterator it;
      bool success = true;
      for ( it = fragmentList.begin (); it != fragmentList.end (); it++ )
        {
          NS_LOG_DEBUG ( "SixLowPanNetDevice::Send (Fragment) " << **it );
          m_txTrace (*it, this, GetIfIndex ());

          if (useMesh)
            {
              bc0Hdr.SetSequenceNumber (m_bc0Serial++);
              (*it)->AddHeader (bc0Hdr);
              (*it)->AddHeader (meshHdr);
            }
          if (doSendFrom)
            {
              success &= m_netDevice->SendFrom (*it, src, destination, protocolNumber);
            }
          else
            {
              success &= m_netDevice->Send (*it, destination, protocolNumber);
            }
        }
      ret = success;
    }
  else
    {
      m_txTrace (packet, this, GetIfIndex ());

      if (useMesh)
        {
          bc0Hdr.SetSequenceNumber (m_bc0Serial++);
          packet->AddHeader (bc0Hdr);
          packet->AddHeader (meshHdr);
        }

      if (doSendFrom)
        {
          NS_LOG_DEBUG ( "SixLowPanNetDevice::SendFrom " << m_node->GetId () << " " << *packet );
          ret = m_netDevice->SendFrom (packet, src, destination, protocolNumber);
        }
      else
        {
          NS_LOG_DEBUG ( "SixLowPanNetDevice::Send " << m_node->GetId () << " " << *packet );
          ret = m_netDevice->Send (packet, destination, protocolNumber);
        }
    }

  return ret;
}

Ptr<Node> SixLowPanNetDevice::GetNode (void) const
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

void SixLowPanNetDevice::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

bool SixLowPanNetDevice::NeedsArp (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG ( m_netDevice != 0, "Sixlowpan: can't find any lower-layer protocol " << m_netDevice );

  return m_netDevice->NeedsArp ();
}

void SixLowPanNetDevice::SetReceiveCallback (NetDevice::ReceiveCallback cb)
{
  NS_LOG_FUNCTION (this << &cb);
  m_rxCallback = cb;
}

void SixLowPanNetDevice::SetPromiscReceiveCallback (NetDevice::PromiscReceiveCallback cb)
{
  NS_LOG_FUNCTION (this << &cb);
  m_promiscRxCallback = cb;
}

bool SixLowPanNetDevice::SupportsSendFrom () const
{
  NS_LOG_FUNCTION (this);
  return true;
}

uint32_t
SixLowPanNetDevice::CompressLowPanHc1 (Ptr<Packet> packet, Address const &src, Address const &dst)
{
  NS_LOG_FUNCTION (this << *packet << src << dst);

  Ipv6Header ipHeader;
  SixLowPanHc1 hc1Header;
  uint32_t size = 0;

  NS_LOG_DEBUG ( "Original packet: " << *packet << " Size " << packet->GetSize () );

  if ( packet->PeekHeader (ipHeader) != 0 )
    {
      packet->RemoveHeader (ipHeader);
      size += ipHeader.GetSerializedSize ();

      hc1Header.SetHopLimit (ipHeader.GetHopLimit ());

      uint8_t bufOne[16];
      uint8_t bufTwo[16];
      Ipv6Address srcAddr = ipHeader.GetSource ();
      srcAddr.GetBytes (bufOne);
      Ipv6Address mySrcAddr = Ipv6Address::MakeAutoconfiguredLinkLocalAddress (src);

      NS_LOG_LOGIC ("Checking source compression: " << mySrcAddr << " - " << srcAddr );

      mySrcAddr.GetBytes (bufTwo);
      bool isSrcSrc = (memcmp (bufOne + 8, bufTwo + 8, 8) == 0);

      if (srcAddr.IsLinkLocal () && isSrcSrc )
        {
          hc1Header.SetSrcCompression (SixLowPanHc1::HC1_PCIC);
        }
      else if (srcAddr.IsLinkLocal () )
        {
          hc1Header.SetSrcCompression (SixLowPanHc1::HC1_PCII);
          hc1Header.SetSrcInterface (bufOne + 8);
        }
      else if ( isSrcSrc )
        {
          hc1Header.SetSrcCompression (SixLowPanHc1::HC1_PIIC);
          hc1Header.SetSrcPrefix (bufOne);
        }
      else
        {
          hc1Header.SetSrcCompression (SixLowPanHc1::HC1_PIII);
          hc1Header.SetSrcInterface (bufOne + 8);
          hc1Header.SetSrcPrefix (bufOne);
        }

      Ipv6Address dstAddr = ipHeader.GetDestination ();
      dstAddr.GetBytes (bufOne);
      Ipv6Address myDstAddr = Ipv6Address::MakeAutoconfiguredLinkLocalAddress (dst);

      NS_LOG_LOGIC ("Checking destination compression: " << myDstAddr << " - " << dstAddr );

      myDstAddr.GetBytes (bufTwo);
      bool isDstDst = (memcmp (bufOne + 8, bufTwo + 8, 8) == 0);

      if (dstAddr.IsLinkLocal () && isDstDst )
        {
          hc1Header.SetDstCompression (SixLowPanHc1::HC1_PCIC);
        }
      else if (dstAddr.IsLinkLocal () )
        {
          hc1Header.SetDstCompression (SixLowPanHc1::HC1_PCII);
          hc1Header.SetDstInterface (bufOne + 8);
        }
      else if ( isDstDst )
        {
          hc1Header.SetDstCompression (SixLowPanHc1::HC1_PIIC);
          hc1Header.SetDstPrefix (bufOne);
        }
      else
        {
          hc1Header.SetDstCompression (SixLowPanHc1::HC1_PIII);
          hc1Header.SetDstInterface (bufOne + 8);
          hc1Header.SetDstPrefix (bufOne);
        }

      if ( (ipHeader.GetFlowLabel () == 0) && (ipHeader.GetTrafficClass () == 0) )
        {
          hc1Header.SetTcflCompression (true);
        }
      else
        {
          hc1Header.SetTcflCompression (false);
          hc1Header.SetTrafficClass (ipHeader.GetTrafficClass ());
          hc1Header.SetFlowLabel (ipHeader.GetFlowLabel ());
        }

      uint8_t nextHeader = ipHeader.GetNextHeader ();
      hc1Header.SetNextHeader (nextHeader);

      // \todo implement HC2 compression
      hc1Header.SetHc2HeaderPresent (false);

      NS_LOG_DEBUG ("HC1 Compression - HC1 header size = " << hc1Header.GetSerializedSize () );
      NS_LOG_DEBUG ("HC1 Compression - packet size = " << packet->GetSize () );

      packet->AddHeader (hc1Header);

      return size;
    }

  return 0;
}

void
SixLowPanNetDevice::DecompressLowPanHc1 (Ptr<Packet> packet, Address const &src, Address const &dst)
{
  NS_LOG_FUNCTION (this << *packet << src << dst);

  Ipv6Header ipHeader;
  SixLowPanHc1 encoding;

  [[maybe_unused]] uint32_t ret = packet->RemoveHeader (encoding);
  NS_LOG_DEBUG ("removed " << ret << " bytes - pkt is " << *packet);

  ipHeader.SetHopLimit (encoding.GetHopLimit ());

  switch (encoding.GetSrcCompression ())
    {
      const uint8_t* interface;
      const uint8_t* prefix;
      uint8_t address[16];

    case SixLowPanHc1::HC1_PIII:
      prefix = encoding.GetSrcPrefix ();
      interface = encoding.GetSrcInterface ();
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = interface[j];
          address[j] = prefix[j];
        }
      ipHeader.SetSource ( Ipv6Address (address) );
      break;
    case SixLowPanHc1::HC1_PIIC:
      prefix = encoding.GetSrcPrefix ();
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = 0;
          address[j] = prefix[j];
        }
      ipHeader.SetSource (Ipv6Address::MakeAutoconfiguredAddress (src, Ipv6Address (address)));
      break;
    case SixLowPanHc1::HC1_PCII:
      interface = encoding.GetSrcInterface ();
      address[0] = 0xfe;
      address[1] = 0x80;
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = interface[j];
        }
      ipHeader.SetSource ( Ipv6Address (address) );
      break;
    case SixLowPanHc1::HC1_PCIC:
      ipHeader.SetSource (Ipv6Address::MakeAutoconfiguredLinkLocalAddress (src));
      break;
    }

  switch (encoding.GetDstCompression ())
    {
      const uint8_t* interface;
      const uint8_t* prefix;
      uint8_t address[16];

    case SixLowPanHc1::HC1_PIII:
      prefix = encoding.GetDstPrefix ();
      interface = encoding.GetDstInterface ();
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = interface[j];
          address[j] = prefix[j];
        }
      ipHeader.SetDestination ( Ipv6Address (address) );
      break;
    case SixLowPanHc1::HC1_PIIC:
      prefix = encoding.GetDstPrefix ();
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = 0;
          address[j] = prefix[j];
        }
      ipHeader.SetDestination (Ipv6Address::MakeAutoconfiguredAddress (dst, Ipv6Address (address)));
      break;
    case SixLowPanHc1::HC1_PCII:
      interface = encoding.GetDstInterface ();
      address[0] = 0xfe;
      address[1] = 0x80;
      for (int j = 0; j < 8; j++)
        {
          address[j + 8] = interface[j];
        }
      ipHeader.SetDestination ( Ipv6Address (address) );
      break;
    case SixLowPanHc1::HC1_PCIC:
      ipHeader.SetDestination (Ipv6Address::MakeAutoconfiguredLinkLocalAddress (dst));
      break;
    }

  if ( !encoding.IsTcflCompression () )
    {
      ipHeader.SetFlowLabel (encoding.GetFlowLabel ());
      ipHeader.SetTrafficClass (encoding.GetTrafficClass ());
    }
  else
    {
      ipHeader.SetFlowLabel (0);
      ipHeader.SetTrafficClass (0);
    }

  ipHeader.SetNextHeader (encoding.GetNextHeader ());

  ipHeader.SetPayloadLength (packet->GetSize ());

  NS_ASSERT_MSG (encoding.IsHc2HeaderPresent () == false,
                 "6LoWPAN: error in decompressing HC1 encoding, unsupported L4 compressed header present.");

  packet->AddHeader (ipHeader);

  NS_LOG_DEBUG ( "Rebuilt packet:  " << *packet << " Size " << packet->GetSize () );
}

uint32_t
SixLowPanNetDevice::CompressLowPanIphc (Ptr<Packet> packet, Address const &src, Address const &dst)
{
  NS_LOG_FUNCTION (this << *packet << src << dst);

  Ipv6Header ipHeader;
  SixLowPanIphc iphcHeader;
  uint32_t size = 0;

  NS_LOG_DEBUG ( "Original packet: " << *packet << " Size " << packet->GetSize () << " src: " << src << " dst: " << dst);

  if ( packet->PeekHeader (ipHeader) != 0 )
    {
      packet->RemoveHeader (ipHeader);
      size += ipHeader.GetSerializedSize ();

      // Set the TF field
      if ( (ipHeader.GetFlowLabel () == 0) && (ipHeader.GetTrafficClass () == 0) )
        {
          iphcHeader.SetTf (SixLowPanIphc::TF_ELIDED);
        }
      else if ( (ipHeader.GetFlowLabel () != 0) && (ipHeader.GetTrafficClass () != 0) )
        {
          iphcHeader.SetTf (SixLowPanIphc::TF_FULL);
          iphcHeader.SetEcn ( (ipHeader.GetTrafficClass () & 0xC0) >> 6);
          iphcHeader.SetDscp ( ipHeader.GetTrafficClass () & 0x3F );
          iphcHeader.SetFlowLabel (ipHeader.GetFlowLabel ());
        }
      else if ( (ipHeader.GetFlowLabel () == 0) && (ipHeader.GetTrafficClass () != 0) )
        {
          iphcHeader.SetTf (SixLowPanIphc::TF_FL_ELIDED);
          iphcHeader.SetEcn ( (ipHeader.GetTrafficClass () & 0xC0) >> 6);
          iphcHeader.SetDscp ( ipHeader.GetTrafficClass () & 0x3F );
        }
      else
        {
          iphcHeader.SetTf (SixLowPanIphc::TF_DSCP_ELIDED);
          iphcHeader.SetEcn ( (ipHeader.GetTrafficClass () & 0xC0) >> 6);
          iphcHeader.SetFlowLabel (ipHeader.GetFlowLabel ());
        }

      // Set the NH field and NextHeader

      uint8_t nextHeader = ipHeader.GetNextHeader ();
      if (CanCompressLowPanNhc (nextHeader))
        {
          if (nextHeader == Ipv6Header::IPV6_UDP)
            {
              iphcHeader.SetNh (true);
              size += CompressLowPanUdpNhc (packet, m_omitUdpChecksum);
            }
          else if (nextHeader == Ipv6Header::IPV6_IPV6)
            {
              iphcHeader.SetNh (true);
              size += CompressLowPanIphc (packet, src, dst);
            }
          else
            {
              uint32_t sizeNhc = CompressLowPanNhc (packet, nextHeader, src, dst);
              // the compression might fail due to Extension header size.
              if (sizeNhc)
                {
                  iphcHeader.SetNh (true);
                  size += sizeNhc;
                }
              else
                {
                  iphcHeader.SetNh (false);
                  iphcHeader.SetNextHeader (nextHeader);
                }
            }
        }
      else
        {
          iphcHeader.SetNh (false);
          iphcHeader.SetNextHeader (nextHeader);
        }

      // Set the HLIM field
      if (ipHeader.GetHopLimit () == 1)
        {
          iphcHeader.SetHlim (SixLowPanIphc::HLIM_COMPR_1);
        }
      else if (ipHeader.GetHopLimit () == 0x40)
        {
          iphcHeader.SetHlim (SixLowPanIphc::HLIM_COMPR_64);
        }
      else if (ipHeader.GetHopLimit () == 0xFF)
        {
          iphcHeader.SetHlim (SixLowPanIphc::HLIM_COMPR_255);
        }
      else
        {
          iphcHeader.SetHlim (SixLowPanIphc::HLIM_INLINE);
          // Set the HopLimit
          iphcHeader.SetHopLimit (ipHeader.GetHopLimit ());
        }

      // Set the CID + SAC + DAC fields to their default value
      iphcHeader.SetCid (false);
      iphcHeader.SetSac (false);
      iphcHeader.SetDac (false);


      Ipv6Address checker = Ipv6Address ("fe80:0000:0000:0000:0000:00ff:fe00:1");
      uint8_t unicastAddrCheckerBuf[16];
      checker.GetBytes (unicastAddrCheckerBuf);
      uint8_t addressBuf[16];

      // This is just to limit the scope of some variables.
      if (true)
        {
          Ipv6Address srcAddr = ipHeader.GetSource ();
          uint8_t srcContextId;

          // The "::" address is compressed as a fake stateful compression.
          if (srcAddr == Ipv6Address::GetAny ())
            {
              // No context information is needed.
              iphcHeader.SetSam (SixLowPanIphc::HC_INLINE);
              iphcHeader.SetSac (true);
            }
          // Check if the address can be compressed with stateful compression
          else if ( FindUnicastCompressionContext (srcAddr, srcContextId) )
            {
              // We can do stateful compression.
              NS_LOG_LOGIC ("Checking stateful source compression: " << srcAddr );

              iphcHeader.SetSac (true);
              if (srcContextId != 0)
                {
                  // the default context is zero, no need to explicit it if it's zero
                  iphcHeader.SetSrcContextId (srcContextId);
                  iphcHeader.SetCid (true);
                }

              // Note that a context might include parts of the EUI-64 (i.e., be as long as 128 bits).

              if (Ipv6Address::MakeAutoconfiguredAddress (src, m_contextTable[srcContextId].contextPrefix) == srcAddr)
                {
                  iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_0);
                }
              else
                {
                  Ipv6Address cleanedAddr = CleanPrefix (srcAddr, m_contextTable[srcContextId].contextPrefix);
                  uint8_t serializedCleanedAddress[16];
                  cleanedAddr.Serialize (serializedCleanedAddress);

                  if ( serializedCleanedAddress[8] == 0x00 && serializedCleanedAddress[9] == 0x00 &&
                      serializedCleanedAddress[10] == 0x00 && serializedCleanedAddress[11] == 0xff &&
                      serializedCleanedAddress[12] == 0xfe && serializedCleanedAddress[13] == 0x00 )
                    {
                      iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_16);
                      iphcHeader.SetSrcInlinePart (serializedCleanedAddress+14, 2);
                    }
                  else
                    {
                      iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_64);
                      iphcHeader.SetSrcInlinePart (serializedCleanedAddress+8, 8);

                    }
                }
            }
          else
            {
              // We must do stateless compression.
              NS_LOG_LOGIC ("Checking stateless source compression: " << srcAddr );

              srcAddr.GetBytes (addressBuf);

              uint8_t serializedSrcAddress[16];
              srcAddr.Serialize (serializedSrcAddress);

              if ( srcAddr == Ipv6Address::MakeAutoconfiguredLinkLocalAddress (src) )
                {
                  iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_0);
                }
              else if (memcmp (addressBuf, unicastAddrCheckerBuf, 14) == 0)
                {
                  iphcHeader.SetSrcInlinePart (serializedSrcAddress+14, 2);
                  iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_16);
                }
              else if ( srcAddr.IsLinkLocal () )
                {
                  iphcHeader.SetSrcInlinePart (serializedSrcAddress+8, 8);
                  iphcHeader.SetSam (SixLowPanIphc::HC_COMPR_64);
                }
              else
                {
                  iphcHeader.SetSrcInlinePart (serializedSrcAddress, 16);
                  iphcHeader.SetSam (SixLowPanIphc::HC_INLINE);
                }
            }
        }

      // Set the M field
      if (ipHeader.GetDestination ().IsMulticast ())
        {
          iphcHeader.SetM (true);
        }
      else
        {
          iphcHeader.SetM (false);
        }

      // This is just to limit the scope of some variables.
      if (true)
        {
          Ipv6Address dstAddr = ipHeader.GetDestination ();
          dstAddr.GetBytes (addressBuf);

          NS_LOG_LOGIC ("Checking destination compression: " << dstAddr );

          uint8_t serializedDstAddress[16];
          dstAddr.Serialize (serializedDstAddress);

          if ( !iphcHeader.GetM () )
            {
              // Unicast address

              uint8_t dstContextId;
              if ( FindUnicastCompressionContext (dstAddr, dstContextId) )
                {
                  // We can do stateful compression.
                  NS_LOG_LOGIC ("Checking stateful destination compression: " << dstAddr );

                  iphcHeader.SetDac (true);
                  if (dstContextId != 0)
                    {
                      // the default context is zero, no need to explicit it if it's zero
                      iphcHeader.SetDstContextId (dstContextId);
                      iphcHeader.SetCid (true);
                    }

                  // Note that a context might include parts of the EUI-64 (i.e., be as long as 128 bits).
                  if (Ipv6Address::MakeAutoconfiguredAddress (dst, m_contextTable[dstContextId].contextPrefix) == dstAddr)
                    {
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_0);
                    }
                  else
                    {
                      Ipv6Address cleanedAddr = CleanPrefix (dstAddr, m_contextTable[dstContextId].contextPrefix);

                      uint8_t serializedCleanedAddress[16];
                      cleanedAddr.Serialize (serializedCleanedAddress);

                      if ( serializedCleanedAddress[8] == 0x00 && serializedCleanedAddress[9] == 0x00 &&
                          serializedCleanedAddress[10] == 0x00 && serializedCleanedAddress[11] == 0xff &&
                          serializedCleanedAddress[12] == 0xfe && serializedCleanedAddress[13] == 0x00 )
                        {
                          iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_16);
                          iphcHeader.SetDstInlinePart (serializedCleanedAddress+14, 2);
                        }
                      else
                        {
                          iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_64);
                          iphcHeader.SetDstInlinePart (serializedCleanedAddress+8, 8);
                        }
                    }
                }
              else
                {
                  NS_LOG_LOGIC ("Checking stateless destination compression: " << dstAddr );

                  if ( dstAddr == Ipv6Address::MakeAutoconfiguredLinkLocalAddress (dst) )
                    {
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_0);
                    }
                  else if (memcmp (addressBuf, unicastAddrCheckerBuf, 14) == 0)
                    {
                      iphcHeader.SetDstInlinePart (serializedDstAddress+14, 2);
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_16);
                    }
                  else if ( dstAddr.IsLinkLocal () )
                    {
                      iphcHeader.SetDstInlinePart (serializedDstAddress+8, 8);
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_64);
                    }
                  else
                    {
                      iphcHeader.SetDstInlinePart (serializedDstAddress, 16);
                      iphcHeader.SetDam (SixLowPanIphc::HC_INLINE);
                    }
                }
            }
          else
            {
              // Multicast address

              uint8_t dstContextId;
              if ( FindMulticastCompressionContext (dstAddr, dstContextId) )
                {
                  // Stateful compression (only one possible case)

                  // ffXX:XXLL:PPPP:PPPP:PPPP:PPPP:XXXX:XXXX
                  uint8_t dstInlinePart[6] = {};
                  dstInlinePart[0] = serializedDstAddress[1];
                  dstInlinePart[1] = serializedDstAddress[2];
                  dstInlinePart[2] = serializedDstAddress[12];
                  dstInlinePart[3] = serializedDstAddress[13];
                  dstInlinePart[4] = serializedDstAddress[14];
                  dstInlinePart[5] = serializedDstAddress[15];

                  iphcHeader.SetDac (true);
                  if (dstContextId != 0)
                    {
                      // the default context is zero, no need to explicit it if it's zero
                      iphcHeader.SetDstContextId (dstContextId);
                      iphcHeader.SetCid (true);
                    }
                  iphcHeader.SetDstInlinePart (dstInlinePart, 6);
                  iphcHeader.SetDam (SixLowPanIphc::HC_INLINE);
                }
              else
                {
                  // Stateless compression

                  uint8_t multicastAddrCheckerBuf[16];
                  Ipv6Address multicastCheckAddress = Ipv6Address ("ff02::1");
                  multicastCheckAddress.GetBytes (multicastAddrCheckerBuf);

                  // The address takes the form ff02::00XX.
                  if ( memcmp (addressBuf, multicastAddrCheckerBuf, 15) == 0 )
                    {
                      iphcHeader.SetDstInlinePart (serializedDstAddress+15, 1);
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_0);
                    }
                  // The address takes the form ffXX::00XX:XXXX.
                  //                            ffXX:0000:0000:0000:0000:0000:00XX:XXXX.
                  else if ( (addressBuf[0] == multicastAddrCheckerBuf[0])
                      && (memcmp (addressBuf + 2, multicastAddrCheckerBuf + 2, 11) == 0) )
                    {
                      uint8_t dstInlinePart[4] = {};
                      memcpy (dstInlinePart, serializedDstAddress+1, 1);
                      memcpy (dstInlinePart+1, serializedDstAddress+13, 3);
                      iphcHeader.SetDstInlinePart (dstInlinePart, 4);
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_16);
                    }
                  // The address takes the form ffXX::00XX:XXXX:XXXX.
                  //                            ffXX:0000:0000:0000:0000:00XX:XXXX:XXXX.
                  else if ( (addressBuf[0] == multicastAddrCheckerBuf[0])
                      && (memcmp (addressBuf + 2, multicastAddrCheckerBuf + 2, 9) == 0) )
                    {
                      uint8_t dstInlinePart[6] = {};
                      memcpy (dstInlinePart, serializedDstAddress+1, 1);
                      memcpy (dstInlinePart+1, serializedDstAddress+11, 5);
                      iphcHeader.SetDstInlinePart (dstInlinePart, 6);
                      iphcHeader.SetDam (SixLowPanIphc::HC_COMPR_64);
                    }
                  else
                    {
                      iphcHeader.SetDstInlinePart (serializedDstAddress, 16);
                      iphcHeader.SetDam (SixLowPanIphc::HC_INLINE);
                    }
                }
            }
        }

      NS_LOG_DEBUG ("IPHC Compression - IPHC header size = " << iphcHeader.GetSerializedSize () );
      NS_LOG_DEBUG ("IPHC Compression - packet size = " << packet->GetSize () );

      packet->AddHeader (iphcHeader);

      NS_LOG_DEBUG ("Packet after IPHC compression: " << *packet);

      return size;
    }
  return 0;
}

bool
SixLowPanNetDevice::CanCompressLowPanNhc (uint8_t nextHeader)
{
  bool ret = false;

  switch (nextHeader)
    {
    case Ipv6Header::IPV6_UDP:
    case Ipv6Header::IPV6_EXT_HOP_BY_HOP:
    case Ipv6Header::IPV6_EXT_ROUTING:
    case Ipv6Header::IPV6_EXT_FRAGMENTATION:
    case Ipv6Header::IPV6_IPV6:
      ret = true;
      break;
    case Ipv6Header::IPV6_EXT_MOBILITY:
    default:
      ret = false;
    }
  return ret;
}

bool
SixLowPanNetDevice::DecompressLowPanIphc (Ptr<Packet> packet, Address const &src, Address const &dst)
{
  NS_LOG_FUNCTION (this << *packet << src << dst);

  Ipv6Header ipHeader;
  SixLowPanIphc encoding;

  [[maybe_unused]] uint32_t ret = packet->RemoveHeader (encoding);
  NS_LOG_DEBUG ("removed " << ret << " bytes - pkt is " << *packet);

  // Hop Limit
  ipHeader.SetHopLimit (encoding.GetHopLimit ());

  // Source address
  if ( encoding.GetSac () )
    {
      // Source address compression uses stateful, context-based compression.
      if ( encoding.GetSam () == SixLowPanIphc::HC_INLINE )
        {
          ipHeader.SetSource ( Ipv6Address::GetAny () );
        }
      else
        {
          uint8_t contextId = encoding.GetSrcContextId ();
          if (m_contextTable.find (contextId) == m_contextTable.end ())
            {
              NS_LOG_LOGIC ("Unknown Source compression context (" << +contextId << "), dropping packet");
              return true;
            }
          if (m_contextTable[contextId].validLifetime < Simulator::Now ())
            {
              NS_LOG_LOGIC ("Expired Source compression context (" << +contextId << "), dropping packet");
              return true;
           }

          uint8_t contexPrefix[16];
          m_contextTable[contextId].contextPrefix.GetBytes(contexPrefix);
          uint8_t contextLength = m_contextTable[contextId].contextPrefix.GetPrefixLength ();

          uint8_t srcAddress[16] = { };
          if ( encoding.GetSam () == SixLowPanIphc::HC_COMPR_64 )
            {
              memcpy (srcAddress +8, encoding.GetSrcInlinePart (), 8);
            }
          else if ( encoding.GetSam () == SixLowPanIphc::HC_COMPR_16 )
            {
              srcAddress[11] = 0xff;
              srcAddress[12] = 0xfe;
              memcpy (srcAddress +14, encoding.GetSrcInlinePart (), 2);
            }
          else // SixLowPanIphc::HC_COMPR_0
            {
              Ipv6Address::MakeAutoconfiguredLinkLocalAddress (src).GetBytes (srcAddress);
            }

          uint8_t bytesToCopy = contextLength / 8;
          uint8_t bitsToCopy = contextLength % 8;

          // Do not combine the prefix - we want to override the bytes.
          for (uint8_t i=0; i<bytesToCopy; i++)
            {
              srcAddress[i] = contexPrefix[i];
            }
          if (bitsToCopy)
            {
              uint8_t addressBitMask = (1<<(8-bitsToCopy))-1;
              uint8_t prefixBitMask = ~addressBitMask;
              srcAddress[bytesToCopy] = (contexPrefix[bytesToCopy] & prefixBitMask) | (srcAddress[bytesToCopy] & addressBitMask);
            }
          ipHeader.SetSource ( Ipv6Address::Deserialize (srcAddress) );
        }
    }
  else
    {
      // Source address compression uses stateless compression.

      if ( encoding.GetSam () == SixLowPanIphc::HC_INLINE )
        {
          uint8_t srcAddress[16] = { };
          memcpy (srcAddress, encoding.GetSrcInlinePart (), 16);
          ipHeader.SetSource ( Ipv6Address::Deserialize (srcAddress) );
        }
      else if ( encoding.GetSam () == SixLowPanIphc::HC_COMPR_64 )
        {
          uint8_t srcAddress[16] = { };
          memcpy (srcAddress +8, encoding.GetSrcInlinePart (), 8);
          srcAddress[0] = 0xfe;
          srcAddress[1] = 0x80;
          ipHeader.SetSource ( Ipv6Address::Deserialize (srcAddress) );
        }
      else if ( encoding.GetSam () == SixLowPanIphc::HC_COMPR_16 )
        {
          uint8_t srcAddress[16] = { };
          memcpy (srcAddress +14, encoding.GetSrcInlinePart (), 2);
          srcAddress[0] = 0xfe;
          srcAddress[1] = 0x80;
          srcAddress[11] = 0xff;
          srcAddress[12] = 0xfe;
          ipHeader.SetSource ( Ipv6Address::Deserialize (srcAddress) );
        }
      else // SixLowPanIphc::HC_COMPR_0
        {
          ipHeader.SetSource (Ipv6Address::MakeAutoconfiguredLinkLocalAddress (src));
        }
    }
  // Destination address
  if ( encoding.GetDac () )
    {
      // Destination address compression uses stateful, context-based compression.
      if ((encoding.GetDam () == SixLowPanIphc::HC_INLINE  && !encoding.GetM ())
          || (encoding.GetDam () == SixLowPanIphc::HC_COMPR_64  && encoding.GetM ())
          || (encoding.GetDam () == SixLowPanIphc::HC_COMPR_16  && encoding.GetM ())
          || (encoding.GetDam () == SixLowPanIphc::HC_COMPR_0  && encoding.GetM ()) )
        {
          NS_ABORT_MSG ("Reserved code found");
        }

      uint8_t contextId = encoding.GetDstContextId ();
      if (m_contextTable.find (contextId) == m_contextTable.end ())
        {
          NS_LOG_LOGIC ("Unknown Destination compression context (" << +contextId << "), dropping packet");
          return true;
        }
      if (m_contextTable[contextId].validLifetime < Simulator::Now ())
        {
          NS_LOG_LOGIC ("Expired Destination compression context (" << +contextId << "), dropping packet");
          return true;
       }

      uint8_t contexPrefix[16];
      m_contextTable[contextId].contextPrefix.GetBytes(contexPrefix);
      uint8_t contextLength = m_contextTable[contextId].contextPrefix.GetPrefixLength ();

      if (encoding.GetM () == false)
        {
          // unicast
          uint8_t dstAddress[16] = { };
          if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_64 )
            {
              memcpy (dstAddress +8, encoding.GetDstInlinePart (), 8);
            }
          else if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_16 )
            {
              dstAddress[11] = 0xff;
              dstAddress[12] = 0xfe;
              memcpy (dstAddress +14, encoding.GetDstInlinePart (), 2);
            }
          else // SixLowPanIphc::HC_COMPR_0
            {
              Ipv6Address::MakeAutoconfiguredLinkLocalAddress (dst).GetBytes (dstAddress);
            }

          uint8_t bytesToCopy = m_contextTable[contextId].contextPrefix.GetPrefixLength () / 8;
          uint8_t bitsToCopy = contextLength % 8;

          // Do not combine the prefix - we want to override the bytes.
          for (uint8_t i=0; i<bytesToCopy; i++)
            {
              dstAddress[i] = contexPrefix[i];
            }
          if (bitsToCopy)
            {
              uint8_t addressBitMask = (1<<(8-bitsToCopy))-1;
              uint8_t prefixBitMask = ~addressBitMask;
              dstAddress[bytesToCopy] = (contexPrefix[bytesToCopy] & prefixBitMask) | (dstAddress[bytesToCopy] & addressBitMask);
            }
          ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
        }
      else
        {
          // multicast
          // Just one possibility: ffXX:XXLL:PPPP:PPPP:PPPP:PPPP:XXXX:XXXX
          uint8_t dstAddress[16] = { };
          dstAddress[0] = 0xff;
          memcpy (dstAddress +1, encoding.GetDstInlinePart (), 2);
          dstAddress[3] = contextLength;
          memcpy (dstAddress +4, contexPrefix, 8);
          memcpy (dstAddress +12, encoding.GetDstInlinePart ()+2, 4);
          ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
        }
    }
  else
    {
      // Destination address compression uses stateless compression.
      if (encoding.GetM () == false)
        {
          // unicast
          if ( encoding.GetDam () == SixLowPanIphc::HC_INLINE )
             {
               uint8_t dstAddress[16] = { };
               memcpy (dstAddress, encoding.GetDstInlinePart (), 16);
               ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
             }
          else if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_64 )
            {
              uint8_t dstAddress[16] = { };
              memcpy (dstAddress +8, encoding.GetDstInlinePart (), 8);
              dstAddress[0] = 0xfe;
              dstAddress[1] = 0x80;
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
          else if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_16 )
            {
              uint8_t dstAddress[16] = { };
              memcpy (dstAddress +14, encoding.GetDstInlinePart (), 2);
              dstAddress[0] = 0xfe;
              dstAddress[1] = 0x80;
              dstAddress[11] = 0xff;
              dstAddress[12] = 0xfe;
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
          else // SixLowPanIphc::HC_COMPR_0
            {
              ipHeader.SetDestination (Ipv6Address::MakeAutoconfiguredLinkLocalAddress (dst));
            }
         }
      else
        {
          // multicast
          if ( encoding.GetDam () == SixLowPanIphc::HC_INLINE )
            {
              uint8_t dstAddress[16] = { };
              memcpy (dstAddress, encoding.GetDstInlinePart (), 16);
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
          else if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_64 )
            {
              uint8_t dstAddress[16] = { };
              dstAddress[0] = 0xff;
              memcpy (dstAddress +1, encoding.GetDstInlinePart (), 1);
              memcpy (dstAddress +11, encoding.GetDstInlinePart ()+1, 5);
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
          else if ( encoding.GetDam () == SixLowPanIphc::HC_COMPR_16 )
            {
              uint8_t dstAddress[16] = { };
              dstAddress[0] = 0xff;
              memcpy (dstAddress +1, encoding.GetDstInlinePart (), 1);
              memcpy (dstAddress +13, encoding.GetDstInlinePart ()+1, 3);
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
          else // SixLowPanIphc::HC_COMPR_0
            {
              uint8_t dstAddress[16] = { };
              dstAddress[0] = 0xff;
              dstAddress[1] = 0x02;
              memcpy (dstAddress+15, encoding.GetDstInlinePart (), 1);
              ipHeader.SetDestination ( Ipv6Address::Deserialize (dstAddress) );
            }
        }
    }

  // Traffic class and Flow Label
  uint8_t traf = 0x00;
  switch (encoding.GetTf ())
    {
    case SixLowPanIphc::TF_FULL:
      traf |= encoding.GetEcn ();
      traf = ( traf << 6 ) | encoding.GetDscp ();
      ipHeader.SetTrafficClass (traf);
      ipHeader.SetFlowLabel ( encoding.GetFlowLabel () & 0xfff ); //Add 4-bit pad
      break;
    case SixLowPanIphc::TF_DSCP_ELIDED:
      traf |= encoding.GetEcn ();
      traf <<= 2;   // Add 2-bit pad
      ipHeader.SetTrafficClass (traf);
      ipHeader.SetFlowLabel (encoding.GetFlowLabel ());
      break;
    case SixLowPanIphc::TF_FL_ELIDED:
      traf |= encoding.GetEcn ();
      traf = ( traf << 6 ) | encoding.GetDscp ();
      ipHeader.SetTrafficClass (traf);
      ipHeader.SetFlowLabel (0);
      break;
    case SixLowPanIphc::TF_ELIDED:
      ipHeader.SetFlowLabel (0);
      ipHeader.SetTrafficClass (0);
      break;
    }

  if ( encoding.GetNh () )
    {
      // Next Header
      uint8_t dispatchRawVal = 0;
      SixLowPanDispatch::NhcDispatch_e dispatchVal;

      packet->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
      dispatchVal = SixLowPanDispatch::GetNhcDispatchType (dispatchRawVal);

      if (dispatchVal == SixLowPanDispatch::LOWPAN_UDPNHC)
        {
          ipHeader.SetNextHeader (Ipv6Header::IPV6_UDP);
          DecompressLowPanUdpNhc (packet, ipHeader.GetSource (), ipHeader.GetDestination ());
        }
      else
        {
          std::pair <uint8_t, bool> retval = DecompressLowPanNhc (packet, src, dst, ipHeader.GetSource (), ipHeader.GetDestination ());
          if ( retval.second == true )
            {
              return true;
            }
          else
            {
              ipHeader.SetNextHeader (retval.first);
            }
        }
    }
  else
    {
      ipHeader.SetNextHeader (encoding.GetNextHeader ());
    }

  ipHeader.SetPayloadLength (packet->GetSize ());

  packet->AddHeader (ipHeader);

  NS_LOG_DEBUG ( "Rebuilt packet:  " << *packet << " Size " << packet->GetSize () );

  return false;
}

uint32_t
SixLowPanNetDevice::CompressLowPanNhc (Ptr<Packet> packet, uint8_t headerType, Address const &src, Address const &dst)
{
  NS_LOG_FUNCTION (this << *packet << int(headerType));

  SixLowPanNhcExtension nhcHeader;
  uint32_t size = 0;
  Buffer blob;

  if (headerType == Ipv6Header::IPV6_EXT_HOP_BY_HOP)
    {
      Ipv6ExtensionHopByHopHeader hopHeader;
      packet->PeekHeader (hopHeader);
      if (hopHeader.GetLength () >= 0xff)
        {
          NS_LOG_DEBUG ("LOWPAN_NHC MUST NOT be used to encode IPv6 Extension Headers "
                        "that have more than 255 octets following the Length field after compression. "
                        "Packet uncompressed.");
          return 0;
        }

      size += packet->RemoveHeader (hopHeader);
      nhcHeader.SetEid (SixLowPanNhcExtension::EID_HOPBYHOP_OPTIONS_H);

      // recursively compress other headers
      uint8_t nextHeader = hopHeader.GetNextHeader ();
      if (CanCompressLowPanNhc (nextHeader))
        {
          if (nextHeader == Ipv6Header::IPV6_UDP)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanUdpNhc (packet, m_omitUdpChecksum);
            }
          else if (nextHeader == Ipv6Header::IPV6_IPV6)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanIphc (packet, src, dst);
            }
          else
            {
              uint32_t sizeNhc = CompressLowPanNhc (packet, nextHeader, src, dst);
              // the compression might fail due to Extension header size.
              if (sizeNhc)
                {
                  nhcHeader.SetNh (true);
                  size += sizeNhc;
                }
              else
                {
                  nhcHeader.SetNh (false);
                  nhcHeader.SetNextHeader (nextHeader);
                }
            }
        }
      else
        {
          nhcHeader.SetNh (false);
          nhcHeader.SetNextHeader (nextHeader);
        }

      uint32_t blobSize = hopHeader.GetSerializedSize ();
      blob.AddAtStart (blobSize);
      hopHeader.Serialize (blob.Begin ());
      blob.RemoveAtStart (2);
      blobSize = blob.GetSize ();
      nhcHeader.SetBlob (blob.PeekData (), blobSize);
    }
  else if (headerType == Ipv6Header::IPV6_EXT_ROUTING)
    {
      Ipv6ExtensionRoutingHeader routingHeader;
      packet->PeekHeader (routingHeader);
      if (routingHeader.GetLength () >= 0xff)
        {
          NS_LOG_DEBUG ("LOWPAN_NHC MUST NOT be used to encode IPv6 Extension Headers "
                        "that have more than 255 octets following the Length field after compression. "
                        "Packet uncompressed.");
          return 0;
        }

      size += packet->RemoveHeader (routingHeader);
      nhcHeader.SetEid (SixLowPanNhcExtension::EID_ROUTING_H);

      // recursively compress other headers
      uint8_t nextHeader = routingHeader.GetNextHeader ();
      if (CanCompressLowPanNhc (nextHeader))
        {
          if (nextHeader == Ipv6Header::IPV6_UDP)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanUdpNhc (packet, m_omitUdpChecksum);
            }
          else if (nextHeader == Ipv6Header::IPV6_IPV6)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanIphc (packet, src, dst);
            }
          else
            {
              uint32_t sizeNhc = CompressLowPanNhc (packet, nextHeader, src, dst);
              // the compression might fail due to Extension header size.
              if (sizeNhc)
                {
                  nhcHeader.SetNh (true);
                  size += sizeNhc;
                }
              else
                {
                  nhcHeader.SetNh (false);
                  nhcHeader.SetNextHeader (nextHeader);
                }
            }
        }
      else
        {
          nhcHeader.SetNh (false);
          nhcHeader.SetNextHeader (nextHeader);
        }

      uint32_t blobSize = routingHeader.GetSerializedSize ();
      blob.AddAtStart (blobSize);
      routingHeader.Serialize (blob.Begin ());
      blob.RemoveAtStart (2);
      blobSize = blob.GetSize ();
      nhcHeader.SetBlob (blob.PeekData (), blobSize);
    }
  else if (headerType == Ipv6Header::IPV6_EXT_FRAGMENTATION)
    {
      Ipv6ExtensionFragmentHeader fragHeader;
      packet->PeekHeader (fragHeader);
      if (fragHeader.GetLength () >= 0xff)
        {
          NS_LOG_DEBUG ("LOWPAN_NHC MUST NOT be used to encode IPv6 Extension Headers "
                        "that have more than 255 octets following the Length field after compression. "
                        "Packet uncompressed.");
          return 0;
        }
      size += packet->RemoveHeader (fragHeader);
      nhcHeader.SetEid (SixLowPanNhcExtension::EID_FRAGMENTATION_H);

      // recursively compress other headers
      uint8_t nextHeader = fragHeader.GetNextHeader ();
      if (CanCompressLowPanNhc (nextHeader))
        {
          if (nextHeader == Ipv6Header::IPV6_UDP)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanUdpNhc (packet, m_omitUdpChecksum);
            }
          else if (nextHeader == Ipv6Header::IPV6_IPV6)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanIphc (packet, src, dst);
            }
          else
            {
              uint32_t sizeNhc = CompressLowPanNhc (packet, nextHeader, src, dst);
              // the compression might fail due to Extension header size.
              if (sizeNhc)
                {
                  nhcHeader.SetNh (true);
                  size += sizeNhc;
                }
              else
                {
                  nhcHeader.SetNh (false);
                  nhcHeader.SetNextHeader (nextHeader);
                }
            }
        }
      else
        {
          nhcHeader.SetNh (false);
          nhcHeader.SetNextHeader (nextHeader);
        }

      uint32_t blobSize = fragHeader.GetSerializedSize ();
      blob.AddAtStart (blobSize);
      fragHeader.Serialize (blob.Begin ());
      blob.RemoveAtStart (2);
      blobSize = blob.GetSize ();
      nhcHeader.SetBlob (blob.PeekData (), blobSize);
    }
  else if (headerType == Ipv6Header::IPV6_EXT_DESTINATION)
    {
      Ipv6ExtensionDestinationHeader destHeader;
      packet->PeekHeader (destHeader);
      if (destHeader.GetLength () >= 0xff)
        {
          NS_LOG_DEBUG ("LOWPAN_NHC MUST NOT be used to encode IPv6 Extension Headers "
                        "that have more than 255 octets following the Length field after compression. "
                        "Packet uncompressed.");
          return 0;
        }
      size += packet->RemoveHeader (destHeader);
      nhcHeader.SetEid (SixLowPanNhcExtension::EID_DESTINATION_OPTIONS_H);

      // recursively compress other headers
      uint8_t nextHeader = destHeader.GetNextHeader ();
      if (CanCompressLowPanNhc (nextHeader))
        {
          if (nextHeader == Ipv6Header::IPV6_UDP)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanUdpNhc (packet, m_omitUdpChecksum);
            }
          else if (nextHeader == Ipv6Header::IPV6_IPV6)
            {
              nhcHeader.SetNh (true);
              size += CompressLowPanIphc (packet, src, dst);
            }
          else
            {
              uint32_t sizeNhc = CompressLowPanNhc (packet, nextHeader, src, dst);
              // the compression might fail due to Extension header size.
              if (sizeNhc)
                {
                  nhcHeader.SetNh (true);
                  size += sizeNhc;
                }
              else
                {
                  nhcHeader.SetNh (false);
                  nhcHeader.SetNextHeader (nextHeader);
                }
            }
        }
      else
        {
          nhcHeader.SetNh (false);
          nhcHeader.SetNextHeader (nextHeader);
        }

      uint32_t blobSize = destHeader.GetSerializedSize ();
      blob.AddAtStart (blobSize);
      destHeader.Serialize (blob.Begin ());
      blob.RemoveAtStart (2);
      blobSize = blob.GetSize ();
      nhcHeader.SetBlob (blob.PeekData (), blobSize);
    }
  else if (headerType == Ipv6Header::IPV6_EXT_MOBILITY)
    {
      // \todo: IPv6 Mobility Header is not supported in ns-3
      NS_ABORT_MSG ("IPv6 Mobility Header is not supported in ns-3 yet");
      return 0;
    }
  else
    {
      NS_ABORT_MSG ("Unexpected Extension Header");
    }

  NS_LOG_DEBUG ("NHC Compression - NHC header size = " << nhcHeader.GetSerializedSize () );
  NS_LOG_DEBUG ("NHC Compression - packet size = " << packet->GetSize () );

  packet->AddHeader (nhcHeader);

  NS_LOG_DEBUG ("Packet after NHC compression: " << *packet);
  return size;
}

std::pair <uint8_t, bool>
SixLowPanNetDevice::DecompressLowPanNhc (Ptr<Packet> packet, Address const &src, Address const &dst, Ipv6Address srcAddress, Ipv6Address dstAddress)
{
  NS_LOG_FUNCTION (this << *packet);

  SixLowPanNhcExtension encoding;

  [[maybe_unused]] uint32_t ret = packet->RemoveHeader (encoding);
  NS_LOG_DEBUG ("removed " << ret << " bytes - pkt is " << *packet);

  Ipv6ExtensionHopByHopHeader hopHeader;
  Ipv6ExtensionRoutingHeader routingHeader;
  Ipv6ExtensionFragmentHeader fragHeader;
  Ipv6ExtensionDestinationHeader destHeader;

  uint32_t blobSize;
  uint8_t blobData[260];
  blobSize = encoding.CopyBlob (blobData + 2, 260 - 2);
  uint8_t paddingSize = 0;

  uint8_t actualEncodedHeaderType = encoding.GetEid ();
  uint8_t actualHeaderType;
  Buffer blob;

  switch (actualEncodedHeaderType)
    {
    case SixLowPanNhcExtension::EID_HOPBYHOP_OPTIONS_H:
      actualHeaderType = Ipv6Header::IPV6_EXT_HOP_BY_HOP;
      if ( encoding.GetNh () )
        {
          // Next Header
          uint8_t dispatchRawVal = 0;
          SixLowPanDispatch::NhcDispatch_e dispatchVal;

          packet->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
          dispatchVal = SixLowPanDispatch::GetNhcDispatchType (dispatchRawVal);

          if (dispatchVal == SixLowPanDispatch::LOWPAN_UDPNHC)
            {
              blobData [0] = Ipv6Header::IPV6_UDP;
              DecompressLowPanUdpNhc (packet, srcAddress, dstAddress);
            }
          else
            {
              blobData [0] = DecompressLowPanNhc (packet, src, dst, srcAddress, dstAddress).first;
            }
        }
      else
        {
          blobData [0] = encoding.GetNextHeader ();
        }

      // manually add some padding if needed
      if ((blobSize + 2) % 8 > 0)
        {
          paddingSize = 8 - (blobSize + 2) % 8;
        }
      if (paddingSize == 1)
        {
          blobData[blobSize + 2] = 0;
        }
      else if (paddingSize > 1)
        {
          blobData[blobSize + 2] = 1;
          blobData[blobSize + 2 + 1] = paddingSize - 2;
          for (uint8_t i = 0; i < paddingSize - 2; i++)
            {
              blobData[blobSize + 2 + 2 + i] = 0;
            }
        }
      blobData [1] = ((blobSize + 2 + paddingSize) >> 3) - 1;
      blob.AddAtStart (blobSize + 2 + paddingSize);
      blob.Begin ().Write (blobData, blobSize + 2 + paddingSize);
      hopHeader.Deserialize (blob.Begin ());

      packet->AddHeader (hopHeader);
      break;

    case SixLowPanNhcExtension::EID_ROUTING_H:
      actualHeaderType = Ipv6Header::IPV6_EXT_ROUTING;
      if ( encoding.GetNh () )
        {
          // Next Header
          uint8_t dispatchRawVal = 0;
          SixLowPanDispatch::NhcDispatch_e dispatchVal;

          packet->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
          dispatchVal = SixLowPanDispatch::GetNhcDispatchType (dispatchRawVal);

          if (dispatchVal == SixLowPanDispatch::LOWPAN_UDPNHC)
            {
              blobData [0] = Ipv6Header::IPV6_UDP;
              DecompressLowPanUdpNhc (packet, srcAddress, dstAddress);
            }
          else
            {
              blobData [0] = DecompressLowPanNhc (packet, src, dst, srcAddress, dstAddress).first;
            }
        }
      else
        {
          blobData [0] = encoding.GetNextHeader ();
        }
      blobData [1] = ((blobSize + 2) >> 3) - 1;
      blob.AddAtStart (blobSize + 2);
      blob.Begin ().Write (blobData, blobSize + 2);
      routingHeader.Deserialize (blob.Begin ());
      packet->AddHeader (routingHeader);
      break;

    case SixLowPanNhcExtension::EID_FRAGMENTATION_H:
      actualHeaderType = Ipv6Header::IPV6_EXT_FRAGMENTATION;
      if ( encoding.GetNh () )
        {
          // Next Header
          uint8_t dispatchRawVal = 0;
          SixLowPanDispatch::NhcDispatch_e dispatchVal;

          packet->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
          dispatchVal = SixLowPanDispatch::GetNhcDispatchType (dispatchRawVal);

          if (dispatchVal == SixLowPanDispatch::LOWPAN_UDPNHC)
            {
              blobData [0] = Ipv6Header::IPV6_UDP;
              DecompressLowPanUdpNhc (packet, srcAddress, dstAddress);
            }
          else
            {
              blobData [0] = DecompressLowPanNhc (packet, src, dst, srcAddress, dstAddress).first;
            }
        }
      else
        {
          blobData [0] = encoding.GetNextHeader ();
        }
      blobData [1] = 0;

      blob.AddAtStart (blobSize + 2);
      blob.Begin ().Write (blobData, blobSize + 2);

      fragHeader.Deserialize (blob.Begin ());
      packet->AddHeader (fragHeader);
      break;

    case SixLowPanNhcExtension::EID_DESTINATION_OPTIONS_H:
      actualHeaderType = Ipv6Header::IPV6_EXT_DESTINATION;
      if ( encoding.GetNh () )
        {
          // Next Header
          uint8_t dispatchRawVal = 0;
          SixLowPanDispatch::NhcDispatch_e dispatchVal;

          packet->CopyData (&dispatchRawVal, sizeof(dispatchRawVal));
          dispatchVal = SixLowPanDispatch::GetNhcDispatchType (dispatchRawVal);

          if (dispatchVal == SixLowPanDispatch::LOWPAN_UDPNHC)
            {
              blobData [0] = Ipv6Header::IPV6_UDP;
              DecompressLowPanUdpNhc (packet, srcAddress, dstAddress);
            }
          else
            {
              blobData [0] = DecompressLowPanNhc (packet, src, dst, srcAddress, dstAddress).first;
            }
        }
      else
        {
          blobData [0] = encoding.GetNextHeader ();
        }

      // manually add some padding if needed
      if ((blobSize + 2) % 8 > 0)
        {
          paddingSize = 8 - (blobSize + 2) % 8;
        }
      if (paddingSize == 1)
        {
          blobData[blobSize + 2] = 0;
        }
      else if (paddingSize > 1)
        {
          blobData[blobSize + 2] = 1;
          blobData[blobSize + 2 + 1] = paddingSize - 2;
          for (uint8_t i = 0; i < paddingSize - 2; i++)
            {
              blobData[blobSize + 2 + 2 + i] = 0;
            }
        }
      blobData [1] = ((blobSize + 2 + paddingSize) >> 3) - 1;
      blob.AddAtStart (blobSize + 2 + paddingSize);
      blob.Begin ().Write (blobData, blobSize + 2 + paddingSize);
      destHeader.Deserialize (blob.Begin ());

      packet->AddHeader (destHeader);
      break;
    case SixLowPanNhcExtension::EID_MOBILITY_H:
      // \todo: IPv6 Mobility Header is not supported in ns-3
      NS_ABORT_MSG ("IPv6 Mobility Header is not supported in ns-3 yet");
      break;
    case SixLowPanNhcExtension::EID_IPv6_H:
      actualHeaderType = Ipv6Header::IPV6_IPV6;
      if (DecompressLowPanIphc (packet, src, dst))
        {
          m_dropTrace (DROP_SATETFUL_DECOMPRESSION_PROBLEM, packet, this, GetIfIndex ());
          return std::pair<uint8_t, bool> (0, true);
        }
      break;
    default:
      NS_ABORT_MSG ("Trying to decode unknown Extension Header");
      break;
    }

  NS_LOG_DEBUG ( "Rebuilt packet: " << *packet << " Size " << packet->GetSize () );
  return std::pair<uint8_t, bool> (actualHeaderType, false);
}

uint32_t
SixLowPanNetDevice::CompressLowPanUdpNhc (Ptr<Packet> packet, bool omitChecksum)
{
  NS_LOG_FUNCTION (this << *packet << int(omitChecksum));

  UdpHeader udpHeader;
  SixLowPanUdpNhcExtension udpNhcHeader;
  uint32_t size = 0;

  NS_ASSERT_MSG (packet->PeekHeader (udpHeader) != 0, "UDP header not found, abort");

  size += packet->RemoveHeader (udpHeader);

  // Set the C field and checksum
  udpNhcHeader.SetC (false);
  uint16_t checksum = udpHeader.GetChecksum ();
  udpNhcHeader.SetChecksum (checksum);

  if (omitChecksum && udpHeader.IsChecksumOk ())
    {
      udpNhcHeader.SetC (true);
    }

  // Set the value of the ports
  udpNhcHeader.SetSrcPort (udpHeader.GetSourcePort ());
  udpNhcHeader.SetDstPort (udpHeader.GetDestinationPort ());

  //Set the P field
  if ( (udpHeader.GetSourcePort () >> 4 ) == 0xf0b && (udpHeader.GetDestinationPort () >> 4 ) == 0xf0b )
    {
      udpNhcHeader.SetPorts (SixLowPanUdpNhcExtension::PORTS_LAST_SRC_LAST_DST);
    }
  else if ( (udpHeader.GetSourcePort () >> 8 ) == 0xf0 && (udpHeader.GetDestinationPort () >> 8 ) != 0xf0 )
    {
      udpNhcHeader.SetPorts (SixLowPanUdpNhcExtension::PORTS_LAST_SRC_ALL_DST);
    }
  else if ( (udpHeader.GetSourcePort () >> 8 ) != 0xf0 && (udpHeader.GetDestinationPort () >> 8 ) == 0xf0 )
    {
      udpNhcHeader.SetPorts (SixLowPanUdpNhcExtension::PORTS_ALL_SRC_LAST_DST);
    }
  else
    {
      udpNhcHeader.SetPorts (SixLowPanUdpNhcExtension::PORTS_INLINE);
    }

  NS_LOG_DEBUG ("UDP_NHC Compression - UDP_NHC header size = " << udpNhcHeader.GetSerializedSize () );
  NS_LOG_DEBUG ("UDP_NHC Compression - packet size = " << packet->GetSize () );

  packet->AddHeader (udpNhcHeader);

  NS_LOG_DEBUG ("Packet after UDP_NHC compression: " << *packet);

  return size;
}

void
SixLowPanNetDevice::DecompressLowPanUdpNhc (Ptr<Packet> packet, Ipv6Address saddr, Ipv6Address daddr)
{
  NS_LOG_FUNCTION (this << *packet);

  UdpHeader udpHeader;
  SixLowPanUdpNhcExtension encoding;

  [[maybe_unused]] uint32_t ret = packet->RemoveHeader (encoding);
  NS_LOG_DEBUG ("removed " << ret << " bytes - pkt is " << *packet);

  // Set the value of the ports
  switch ( encoding.GetPorts () )
    {
      uint16_t temp;
    case SixLowPanUdpNhcExtension::PORTS_INLINE:
      udpHeader.SetSourcePort (encoding.GetSrcPort ());
      udpHeader.SetDestinationPort (encoding.GetDstPort ());
      break;
    case SixLowPanUdpNhcExtension::PORTS_ALL_SRC_LAST_DST:
      udpHeader.SetSourcePort (encoding.GetSrcPort ());
      temp = (0xf0 << 8) | encoding.GetDstPort ();
      udpHeader.SetDestinationPort (temp);
      break;
    case SixLowPanUdpNhcExtension::PORTS_LAST_SRC_ALL_DST:
      temp = (0xf0 << 8) | encoding.GetSrcPort ();
      udpHeader.SetSourcePort (temp);
      udpHeader.SetDestinationPort (encoding.GetDstPort ());
      break;
    case SixLowPanUdpNhcExtension::PORTS_LAST_SRC_LAST_DST:
      temp = (0xf0b << 4) | encoding.GetSrcPort ();
      udpHeader.SetSourcePort (temp);
      temp = (0xf0b << 4) | encoding.GetDstPort ();
      udpHeader.SetDestinationPort (temp);
      break;
    }

  // Get the C field and checksum
  if (Node::ChecksumEnabled ())
    {
      if ( encoding.GetC () )
        {
          NS_LOG_LOGIC ("Recalculating UDP Checksum");
          udpHeader.EnableChecksums ();
          udpHeader.InitializeChecksum (saddr,
                                        daddr,
                                        UdpL4Protocol::PROT_NUMBER);
          packet->AddHeader (udpHeader);
        }
      else
        {
          NS_LOG_LOGIC ("Forcing UDP Checksum to " << encoding.GetChecksum ());
          udpHeader.ForceChecksum (encoding.GetChecksum ());
          packet->AddHeader (udpHeader);
          NS_LOG_LOGIC ("UDP checksum is ok ? " << udpHeader.IsChecksumOk ());
        }
    }
  else
    {
      packet->AddHeader (udpHeader);
    }

  NS_LOG_DEBUG ( "Rebuilt packet: " << *packet << " Size " << packet->GetSize () );
}

void SixLowPanNetDevice::DoFragmentation (Ptr<Packet> packet,
                                          uint32_t origPacketSize,
                                          uint32_t origHdrSize,
                                          uint32_t extraHdrSize,
                                          std::list<Ptr<Packet> >& listFragments)
{
  NS_LOG_FUNCTION (this << *packet);

  Ptr<Packet> p = packet->Copy ();

  uint16_t offsetData = 0;
  uint16_t offset = 0;
  uint16_t l2Mtu = m_netDevice->GetMtu ();
  uint32_t packetSize = packet->GetSize ();
  uint32_t compressedHeaderSize = packetSize - (origPacketSize - origHdrSize);

  uint16_t tag = uint16_t (m_rng->GetValue (0, 65535));
  NS_LOG_LOGIC ("random tag " << tag << " - test " << packetSize );

  // first fragment
  SixLowPanFrag1 frag1Hdr;
  frag1Hdr.SetDatagramTag (tag);

  uint32_t size;
  NS_ASSERT_MSG ( l2Mtu > frag1Hdr.GetSerializedSize (),
                  "6LoWPAN: can not fragment, 6LoWPAN headers are bigger than MTU");
  
  // All the headers are substracted to get remaining units for data
  size = l2Mtu - frag1Hdr.GetSerializedSize () - compressedHeaderSize - extraHdrSize;
  size -= size % 8;
  size += compressedHeaderSize;

  frag1Hdr.SetDatagramSize (origPacketSize);

  Ptr<Packet> fragment1 = p->CreateFragment (offsetData, size);
  offset += size + origHdrSize - compressedHeaderSize;
  offsetData += size;

  fragment1->AddHeader (frag1Hdr);
  listFragments.push_back (fragment1);

  bool moreFrag = true;
  do
    {
      SixLowPanFragN fragNHdr;
      fragNHdr.SetDatagramTag (tag);
      fragNHdr.SetDatagramSize (origPacketSize);
      fragNHdr.SetDatagramOffset ((offset) >> 3);

      size = l2Mtu - fragNHdr.GetSerializedSize () - extraHdrSize;
      size -= size % 8;

      if ( (offsetData + size) > packetSize )
        {
          size = packetSize - offsetData;
          moreFrag = false;
        }

      if (size > 0)
        {
          NS_LOG_LOGIC ("Fragment creation - " << offset << ", " << offset );
          Ptr<Packet> fragment = p->CreateFragment (offsetData, size);
          NS_LOG_LOGIC ("Fragment created - " << offset << ", " << fragment->GetSize () );

          offset += size;
          offsetData += size;

          fragment->AddHeader (fragNHdr);
          listFragments.push_back (fragment);
        }
    }
  while (moreFrag);

  return;
}

bool SixLowPanNetDevice::ProcessFragment (Ptr<Packet>& packet, Address const &src, Address const &dst, bool isFirst)
{
  NS_LOG_FUNCTION ( this << *packet );
  SixLowPanFrag1 frag1Header;
  SixLowPanFragN fragNHeader;
  FragmentKey_t key;
  uint16_t packetSize;
  key.first = std::pair<Address, Address> (src, dst);

  Ptr<Packet> p = packet->Copy ();
  uint16_t offset = 0;

  /* Implementation note:
   *
   * The fragment offset is relative to the *uncompressed* packet.
   * On the other hand, the packet can not be uncompressed correctly without all
   * its fragments, as the UDP checksum can not be computed otherwise.
   *
   * As a consequence we must uncompress the packet twice, and save its first
   * fragment for the final one.
   */

  if ( isFirst )
    {
      uint8_t dispatchRawValFrag1 = 0;
      SixLowPanDispatch::Dispatch_e dispatchValFrag1;

      p->RemoveHeader (frag1Header);
      packetSize = frag1Header.GetDatagramSize ();
      p->CopyData (&dispatchRawValFrag1, sizeof(dispatchRawValFrag1));
      dispatchValFrag1 = SixLowPanDispatch::GetDispatchType (dispatchRawValFrag1);
      NS_LOG_DEBUG ( "Dispatches: " << int(dispatchRawValFrag1) << " - " << int(dispatchValFrag1) );
      NS_LOG_DEBUG ( "Packet: " << *p );

      switch ( dispatchValFrag1 )
        {
        case SixLowPanDispatch::LOWPAN_IPv6:
          {
            SixLowPanIpv6 uncompressedHdr;
            p->RemoveHeader (uncompressedHdr);
          }
          break;
        case SixLowPanDispatch::LOWPAN_HC1:
          DecompressLowPanHc1 (p, src, dst);
          break;
        case SixLowPanDispatch::LOWPAN_IPHC:
          if (DecompressLowPanIphc (p, src, dst))
            {
              m_dropTrace (DROP_SATETFUL_DECOMPRESSION_PROBLEM, p, this, GetIfIndex ());
              return false;
            }
          break;
        default:
          NS_FATAL_ERROR ("Unsupported 6LoWPAN encoding, exiting.");
          break;
        }

      key.second = std::pair<uint16_t, uint16_t> (frag1Header.GetDatagramSize (), frag1Header.GetDatagramTag ());
    }
  else
    {
      p->RemoveHeader (fragNHeader);
      packetSize = fragNHeader.GetDatagramSize ();
      offset = fragNHeader.GetDatagramOffset () << 3;
      key.second = std::pair<uint16_t, uint16_t> (fragNHeader.GetDatagramSize (), fragNHeader.GetDatagramTag ());
    }

  Ptr<Fragments> fragments;

  MapFragments_t::iterator it = m_fragments.find (key);
  if (it == m_fragments.end ())
    {
      // erase the oldest packet.
      if ( m_fragmentReassemblyListSize && (m_fragments.size () >= m_fragmentReassemblyListSize) )
        {
          FragmentsTimeoutsListI_t iter = m_timeoutEventList.begin ();
          FragmentKey_t oldestKey = std::get<1> (*iter);

          std::list< Ptr<Packet> > storedFragments = m_fragments[oldestKey]->GetFraments ();
          for (std::list< Ptr<Packet> >::iterator fragIter = storedFragments.begin ();
               fragIter != storedFragments.end (); fragIter++)
            {
              m_dropTrace (DROP_FRAGMENT_BUFFER_FULL, *fragIter, this, GetIfIndex ());
            }

          m_timeoutEventList.erase (m_fragments[oldestKey]->GetTimeoutIter ());
          m_fragments[oldestKey] = 0;
          m_fragments.erase (oldestKey);

        }
      fragments = Create<Fragments> ();
      fragments->SetPacketSize (packetSize);
      m_fragments.insert (std::make_pair (key, fragments));
      uint32_t ifIndex = GetIfIndex ();

      FragmentsTimeoutsListI_t iter = SetTimeout (key, ifIndex);
      fragments->SetTimeoutIter (iter);
    }
  else
    {
      fragments = it->second;
    }

  fragments->AddFragment (p, offset);

  // add the very first fragment so we can correctly decode the packet once is rebuilt.
  // this is needed because otherwise the UDP header length and checksum can not be calculated.
  if ( isFirst )
    {
      fragments->AddFirstFragment (packet);
    }

  if ( fragments->IsEntire () )
    {
      packet = fragments->GetPacket ();
      NS_LOG_LOGIC ("Reconstructed packet: " << *packet);

      SixLowPanFrag1 frag1Header;
      packet->RemoveHeader (frag1Header);

      NS_LOG_LOGIC ("Rebuilt packet. Size " << packet->GetSize () << " - " << *packet);
      m_timeoutEventList.erase (fragments->GetTimeoutIter ());
      fragments = 0;
      m_fragments.erase (key);
      return true;
    }

  return false;
}

SixLowPanNetDevice::Fragments::Fragments ()
{
  NS_LOG_FUNCTION (this);
  m_packetSize = 0;
}

SixLowPanNetDevice::Fragments::~Fragments ()
{
  NS_LOG_FUNCTION (this);
}

void SixLowPanNetDevice::Fragments::AddFragment (Ptr<Packet> fragment, uint16_t fragmentOffset)
{
  NS_LOG_FUNCTION (this << fragmentOffset << *fragment);

  std::list<std::pair<Ptr<Packet>, uint16_t> >::iterator it;
  bool duplicate = false;

  for (it = m_fragments.begin (); it != m_fragments.end (); it++)
    {
      if (it->second > fragmentOffset)
        {
          break;
        }
      if (it->second == fragmentOffset)
        {
          duplicate = true;
          NS_ASSERT_MSG (fragment->GetSize () == it->first->GetSize (), "Duplicate fragment size differs. Aborting.");
          break;
        }
    }
  if (!duplicate)
    {
      m_fragments.insert (it, std::make_pair (fragment, fragmentOffset));
    }
}

void SixLowPanNetDevice::Fragments::AddFirstFragment (Ptr<Packet> fragment)
{
  NS_LOG_FUNCTION (this << *fragment);

  m_firstFragment = fragment;
}

bool SixLowPanNetDevice::Fragments::IsEntire () const
{
  NS_LOG_FUNCTION (this);

  bool ret = m_fragments.size () > 0;
  uint16_t lastEndOffset = 0;

  if (ret)
    {
      for (std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator it = m_fragments.begin (); it != m_fragments.end (); it++)
        {
          // overlapping fragments should not exist
          NS_LOG_LOGIC ("Checking overlaps " << lastEndOffset << " - " << it->second );

          if (lastEndOffset < it->second)
            {
              ret = false;
              break;
            }
          // fragments might overlap in strange ways
          uint16_t fragmentEnd = it->first->GetSize () + it->second;
          lastEndOffset = std::max ( lastEndOffset, fragmentEnd );
        }
    }

  if ( ret && (lastEndOffset == m_packetSize))
    {
      return true;
    }
  return false;
}

Ptr<Packet> SixLowPanNetDevice::Fragments::GetPacket () const
{
  NS_LOG_FUNCTION (this);

  std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator it = m_fragments.begin ();

  Ptr<Packet> p = Create<Packet> ();
  uint16_t lastEndOffset = 0;

  p->AddAtEnd (m_firstFragment);
  it = m_fragments.begin ();
  lastEndOffset = it->first->GetSize ();

  for ( it++; it != m_fragments.end (); it++)
    {
      if ( lastEndOffset > it->second )
        {
          NS_ABORT_MSG ("Overlapping fragments found, forbidden condition");
        }
      else
        {
          NS_LOG_LOGIC ("Adding: " << *(it->first) );
          p->AddAtEnd (it->first);
        }
      lastEndOffset += it->first->GetSize ();
    }

  return p;
}

void SixLowPanNetDevice::Fragments::SetPacketSize (uint32_t packetSize)
{
  NS_LOG_FUNCTION (this << packetSize);
  m_packetSize = packetSize;
}

std::list< Ptr<Packet> > SixLowPanNetDevice::Fragments::GetFraments () const
{
  std::list< Ptr<Packet> > fragments;
  std::list<std::pair<Ptr<Packet>, uint16_t> >::const_iterator iter;
  for ( iter = m_fragments.begin (); iter != m_fragments.end (); iter++)
    {
      fragments.push_back (iter->first);
    }
  return fragments;
}

void
SixLowPanNetDevice::Fragments::SetTimeoutIter (FragmentsTimeoutsListI_t iter)
{
  m_timeoutIter = iter;
  return;
}

SixLowPanNetDevice::FragmentsTimeoutsListI_t
SixLowPanNetDevice::Fragments::GetTimeoutIter ()
{
  return m_timeoutIter;
}

void SixLowPanNetDevice::HandleFragmentsTimeout (FragmentKey_t key, uint32_t iif)
{
  NS_LOG_FUNCTION (this);

  MapFragments_t::iterator it = m_fragments.find (key);
  std::list< Ptr<Packet> > storedFragments = it->second->GetFraments ();
  for (std::list< Ptr<Packet> >::iterator fragIter = storedFragments.begin ();
       fragIter != storedFragments.end (); fragIter++)
    {
      m_dropTrace (DROP_FRAGMENT_TIMEOUT, *fragIter, this, iif);
    }
  // clear the buffers
  it->second = 0;

  m_fragments.erase (key);
}

Address SixLowPanNetDevice::Get16MacFrom48Mac (Address addr)
{
  NS_ASSERT_MSG (Mac48Address::IsMatchingType (addr), "Need a Mac48Address" << addr);

  uint8_t buf[6];
  addr.CopyTo (buf);

  Mac16Address shortAddr;
  shortAddr.CopyFrom (buf + 4);

  return shortAddr;
}

SixLowPanNetDevice::FragmentsTimeoutsListI_t SixLowPanNetDevice::SetTimeout (FragmentKey_t key, uint32_t iif)
{
  if (m_timeoutEventList.empty ())
    {
      m_timeoutEvent = Simulator::Schedule (m_fragmentExpirationTimeout, &SixLowPanNetDevice::HandleTimeout, this);
    }
  m_timeoutEventList.emplace_back (Simulator::Now () + m_fragmentExpirationTimeout, key, iif);

  SixLowPanNetDevice::FragmentsTimeoutsListI_t iter = --m_timeoutEventList.end();

  return (iter);
}

void SixLowPanNetDevice::HandleTimeout (void)
{
  Time now = Simulator::Now ();

  while (!m_timeoutEventList.empty () && std::get<0> (*m_timeoutEventList.begin ()) == now)
    {
      HandleFragmentsTimeout (std::get<1> (*m_timeoutEventList.begin ()),
                              std::get<2> (*m_timeoutEventList.begin ()));
      m_timeoutEventList.pop_front ();
    }

  if (m_timeoutEventList.empty ())
    {
      return;
    }

  Time difference = std::get<0> (*m_timeoutEventList.begin ()) - now;
  m_timeoutEvent = Simulator::Schedule (difference, &SixLowPanNetDevice::HandleTimeout, this);

  return;
}

void SixLowPanNetDevice::AddContext (uint8_t contextId, Ipv6Prefix contextPrefix, bool compressionAllowed, Time validLifetime)
{
  NS_LOG_FUNCTION (this << +contextId << Ipv6Address::GetOnes ().CombinePrefix (contextPrefix) << contextPrefix << compressionAllowed << validLifetime.As (Time::S));

  if (contextId > 15)
    {
      NS_LOG_LOGIC ("Invalid context ID (" << +contextId << "), ignoring");
      return;
    }

  if (validLifetime == Time(0))
    {
      NS_LOG_LOGIC ("Context (" << +contextId << "), removed (validity time is zero)");
      m_contextTable.erase (contextId);
      return;
    }

  m_contextTable[contextId].contextPrefix = contextPrefix;
  m_contextTable[contextId].compressionAllowed = compressionAllowed;
  m_contextTable[contextId].validLifetime = Simulator::Now () + validLifetime;

  return;
}

bool SixLowPanNetDevice::GetContext (uint8_t contextId, Ipv6Prefix& contextPrefix, bool& compressionAllowed, Time& validLifetime)
{
  NS_LOG_FUNCTION (this << +contextId);

  if (contextId > 15)
    {
      NS_LOG_LOGIC ("Invalid context ID (" << +contextId << "), ignoring");
      return false;
    }

  if (m_contextTable.find (contextId) == m_contextTable.end ())
    {
      NS_LOG_LOGIC ("Context not found (" << +contextId << "), ignoring");
      return false;
    }

  contextPrefix = m_contextTable[contextId].contextPrefix;
  compressionAllowed = m_contextTable[contextId].compressionAllowed;
  validLifetime = m_contextTable[contextId].validLifetime;

  return true;
}

void SixLowPanNetDevice::RenewContext (uint8_t contextId, Time validLifetime)
{
  NS_LOG_FUNCTION (this << +contextId << validLifetime.As (Time::S));

  if (contextId > 15)
    {
      NS_LOG_LOGIC ("Invalid context ID (" << +contextId << "), ignoring");
      return;
    }

  if (m_contextTable.find (contextId) == m_contextTable.end ())
    {
      NS_LOG_LOGIC ("Context not found (" << +contextId << "), ignoring");
      return;
    }
  m_contextTable[contextId].compressionAllowed = true;
  m_contextTable[contextId].validLifetime = Simulator::Now () + validLifetime;
  return;
}


void SixLowPanNetDevice::InvalidateContext (uint8_t contextId)
{
  NS_LOG_FUNCTION (this << +contextId);

  if (contextId > 15)
    {
      NS_LOG_LOGIC ("Invalid context ID (" << +contextId << "), ignoring");
      return;
    }

  if (m_contextTable.find (contextId) == m_contextTable.end ())
    {
      NS_LOG_LOGIC ("Context not found (" << +contextId << "), ignoring");
      return;
    }
  m_contextTable[contextId].compressionAllowed = false;
  return;
}

void SixLowPanNetDevice::RemoveContext (uint8_t contextId)
{
  NS_LOG_FUNCTION (this << +contextId);

  if (contextId > 15)
    {
      NS_LOG_LOGIC ("Invalid context ID (" << +contextId << "), ignoring");
      return;
    }

  if (m_contextTable.find (contextId) == m_contextTable.end ())
    {
      NS_LOG_LOGIC ("Context not found (" << +contextId << "), ignoring");
      return;
    }

  m_contextTable.erase (contextId);
  return;
}

bool SixLowPanNetDevice::FindUnicastCompressionContext (Ipv6Address address, uint8_t& contextId)
{
  NS_LOG_FUNCTION (this << address);

  for (const auto& iter: m_contextTable)
    {
      ContextEntry context = iter.second;

      if ( (context.compressionAllowed == true) && (context.validLifetime > Simulator::Now ()) )
        {

          if (address.HasPrefix (context.contextPrefix))
            {
              NS_LOG_LOGIC ("Fount context " << +contextId << " " <<
                            Ipv6Address::GetOnes ().CombinePrefix (context.contextPrefix) << context.contextPrefix << " matching");

              contextId = iter.first;
              return true;
            }
        }
    }
  return false;
}

bool SixLowPanNetDevice::FindMulticastCompressionContext (Ipv6Address address, uint8_t& contextId)
{
  NS_LOG_FUNCTION (this << address);

  // The only allowed context-based compressed multicast address is in the form
  // ffXX:XXLL:PPPP:PPPP:PPPP:PPPP:XXXX:XXXX

  for (const auto& iter: m_contextTable)
    {
      ContextEntry context = iter.second;

      if ( (context.compressionAllowed == true) && (context.validLifetime > Simulator::Now ()) )
        {
          uint8_t contextLength = context.contextPrefix.GetPrefixLength ();

          if (contextLength <= 64) // only 64-bit prefixes or less are allowed.
            {
              uint8_t contextBytes[16];
              uint8_t addressBytes[16];

              context.contextPrefix.GetBytes (contextBytes);
              address.GetBytes (addressBytes);

              if (addressBytes[3] == contextLength &&
                  addressBytes[4] == contextBytes[0] &&
                  addressBytes[5] == contextBytes[1] &&
                  addressBytes[6] == contextBytes[2] &&
                  addressBytes[7] == contextBytes[3] &&
                  addressBytes[8] == contextBytes[4] &&
                  addressBytes[9] == contextBytes[5] &&
                  addressBytes[10] == contextBytes[6] &&
                  addressBytes[11] == contextBytes[7])
                {
                  NS_LOG_LOGIC ("Fount context " << +contextId << " " <<
                                Ipv6Address::GetOnes ().CombinePrefix (context.contextPrefix) << context.contextPrefix << " matching");

                  contextId = iter.first;
                  return true;
                }
            }
        }
    }
  return false;
}

Ipv6Address SixLowPanNetDevice::CleanPrefix (Ipv6Address address, Ipv6Prefix prefix)
{
  uint8_t addressBytes[16];
  address.GetBytes (addressBytes);
  uint8_t prefixLength = prefix.GetPrefixLength ();

  uint8_t bytesToClean = prefixLength / 8;
  uint8_t bitsToClean = prefixLength % 8;
  for (uint8_t i=0; i<bytesToClean; i++)
    {
      addressBytes[i] = 0;
    }
  if (bitsToClean)
    {
      uint8_t cleanupMask = (1<<bitsToClean)-1;
      addressBytes[bytesToClean] &= cleanupMask;
    }

  Ipv6Address cleanedAddress = Ipv6Address::Deserialize (addressBytes);

  return cleanedAddress;
}

}

// namespace ns3
