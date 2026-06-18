/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#include "ns3/llc-snap-header.h"
#include "ns3/simulator.h"
#include "ns3/callback.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "wimax-net-device.h"
#include "wimax-channel.h"
#include "ns3/packet-burst.h"
#include "burst-profile-manager.h"
#include <list>
#include "send-params.h"
#include "ns3/uinteger.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "service-flow-manager.h"
#include "connection-manager.h"
#include "bandwidth-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WimaxNetDevice");

NS_OBJECT_ENSURE_REGISTERED (WimaxNetDevice);

uint32_t WimaxNetDevice::m_nrFrames = 0;
uint8_t WimaxNetDevice::m_direction = ~0;
Time WimaxNetDevice::m_frameStartTime = Seconds (0);

TypeId WimaxNetDevice::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::WimaxNetDevice")

    .SetParent<NetDevice> ()
    .SetGroupName ("Wimax")

    // No AddConstructor because this is an abstract class.

    .AddAttribute ("Mtu", "The MAC-level Maximum Transmission Unit",
                   UintegerValue (DEFAULT_MSDU_SIZE),
                   MakeUintegerAccessor (&WimaxNetDevice::SetMtu,
                                         &WimaxNetDevice::GetMtu),
                   MakeUintegerChecker<uint16_t> (0,MAX_MSDU_SIZE))

    .AddAttribute ("Phy",
                   "The PHY layer attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::GetPhy, &WimaxNetDevice::SetPhy),
                   MakePointerChecker<WimaxPhy> ())

    .AddAttribute ("Channel",
                   "The channel attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::GetPhyChannel, &WimaxNetDevice::SetChannel),
                   MakePointerChecker<WimaxChannel> ())

    .AddAttribute ("RTG",
                   "receive/transmit transition gap.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WimaxNetDevice::GetRtg, &WimaxNetDevice::SetRtg),
                   MakeUintegerChecker<uint16_t> (0, 120))

    .AddAttribute ("TTG",
                   "transmit/receive transition gap.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&WimaxNetDevice::GetTtg, &WimaxNetDevice::SetTtg),
                   MakeUintegerChecker<uint16_t> (0, 120))

    .AddAttribute ("ConnectionManager",
                   "The connection manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::GetConnectionManager,
                                        &WimaxNetDevice::SetConnectionManager),
                   MakePointerChecker<ConnectionManager> ())

    .AddAttribute ("BurstProfileManager",
                   "The burst profile manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::GetBurstProfileManager,
                                        &WimaxNetDevice::SetBurstProfileManager),
                   MakePointerChecker<BurstProfileManager> ())

    .AddAttribute ("BandwidthManager",
                   "The bandwidth manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::GetBandwidthManager,
                                        &WimaxNetDevice::SetBandwidthManager),
                   MakePointerChecker<BandwidthManager> ())

    .AddAttribute ("InitialRangingConnection",
                   "Initial ranging connection",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::m_initialRangingConnection),
                   MakePointerChecker<WimaxConnection> ())

    .AddAttribute ("BroadcastConnection",
                   "Broadcast connection",
                   PointerValue (),
                   MakePointerAccessor (&WimaxNetDevice::m_broadcastConnection),
                   MakePointerChecker<WimaxConnection> ())

    .AddTraceSource ("Rx",
                     "Receive trace",
                     MakeTraceSourceAccessor (&WimaxNetDevice::m_traceRx),
                     "ns3::WimaxNetDevice::TxRxTracedCallback")

    .AddTraceSource ("Tx",
                     "Transmit trace",
                     MakeTraceSourceAccessor (&WimaxNetDevice::m_traceTx),
                     "ns3::WimaxNetDevice::TxRxTracedCallback");
  return tid;
}

WimaxNetDevice::WimaxNetDevice (void)
  : m_state (0),
    m_symbolIndex (0),
    m_ttg (0),
    m_rtg (0)
{
  InitializeChannels ();
  m_connectionManager = CreateObject<ConnectionManager> ();
  m_burstProfileManager = CreateObject<BurstProfileManager> (this);
  m_bandwidthManager = CreateObject<BandwidthManager> (this);
  m_nrFrames = 0;
  m_direction = ~0;
  m_frameStartTime = Seconds (0);
}

WimaxNetDevice::~WimaxNetDevice (void)
{
}

void
WimaxNetDevice::DoDispose (void)
{

  m_phy->Dispose ();
  m_phy = 0;
  m_node = 0;
  m_initialRangingConnection = 0;
  m_broadcastConnection = 0;
  m_connectionManager = 0;
  m_burstProfileManager = 0;
  m_bandwidthManager = 0;
  m_connectionManager = 0;
  m_bandwidthManager = 0;

  NetDevice::DoDispose ();
}


void
WimaxNetDevice::SetTtg (uint16_t ttg)
{
  m_ttg = ttg;
}

uint16_t
WimaxNetDevice::GetTtg (void) const
{
  return m_ttg;
}

void
WimaxNetDevice::SetRtg (uint16_t rtg)
{
  m_rtg = rtg;
}

uint16_t
WimaxNetDevice::GetRtg (void) const
{
  return m_rtg;
}

void
WimaxNetDevice::SetName (const std::string name)
{
  m_name = name;
}

std::string
WimaxNetDevice::GetName (void) const
{
  return m_name;
}

void
WimaxNetDevice::SetIfIndex (const uint32_t index)
{
  m_ifIndex = index;
}

uint32_t
WimaxNetDevice::GetIfIndex (void) const
{
  return m_ifIndex;
}

Ptr<Channel>
WimaxNetDevice::GetChannel (void) const
{
  return DoGetChannel ();
}

Ptr<Channel>
WimaxNetDevice::GetPhyChannel (void) const
{
  return DoGetChannel ();
}

bool
WimaxNetDevice::SetMtu (const uint16_t mtu)
{
  if (mtu > MAX_MSDU_SIZE)
    {
      return false;
    }
  m_mtu = mtu;
  return true;
}

uint16_t
WimaxNetDevice::GetMtu (void) const
{
  return m_mtu;
}

bool
WimaxNetDevice::IsLinkUp (void) const
{

  return m_phy != 0 && m_linkUp;

}

void
WimaxNetDevice::SetLinkChangeCallback (Callback<void> callback)
{
  m_linkChange = callback;
}

bool
WimaxNetDevice::IsBroadcast (void) const
{
  return true;
}

Address
WimaxNetDevice::GetBroadcast (void) const
{
  return Mac48Address::GetBroadcast ();
}

bool
WimaxNetDevice::IsMulticast (void) const
{
  return false;
}

Address
WimaxNetDevice::GetMulticast (void) const
{
  return Mac48Address ("01:00:5e:00:00:00");
}

Address
WimaxNetDevice::MakeMulticastAddress (Ipv4Address multicastGroup) const
{
  return GetMulticast ();
}

bool
WimaxNetDevice::IsPointToPoint (void) const
{
  return false;
}

bool
WimaxNetDevice::Send (Ptr<Packet> packet, const Address& dest, uint16_t protocolNumber)
{

  Mac48Address to = Mac48Address::ConvertFrom (dest);
  LlcSnapHeader llcHdr;
  llcHdr.SetType (protocolNumber);
  packet->AddHeader (llcHdr);

  m_traceTx (packet, to);

  return DoSend (packet, Mac48Address::ConvertFrom (GetAddress ()), to, protocolNumber);
}

void
WimaxNetDevice::SetNode (Ptr<Node> node)
{
  m_node = node;
}

Ptr<Node>
WimaxNetDevice::GetNode (void) const
{
  return m_node;
}

bool
WimaxNetDevice::NeedsArp (void) const
{
  return false;
  /*
   * Modified by Mohamed Amine ISMAIL.
   * see "Transmission of IPv4 packets over IEEE 802.16's IP Convergence
   *      Sublayer draft-ietf-16ng-ipv4-over-802-dot-16-ipcs-04.txt" section
   * 5.2
   */
}

void
WimaxNetDevice::SetReceiveCallback (ReceiveCallback cb)
{
  m_forwardUp = cb;
}

void
WimaxNetDevice::ForwardUp (Ptr<Packet> packet, const Mac48Address &source, const Mac48Address &dest)
{
  m_traceRx (packet, source);
  LlcSnapHeader llc;
  packet->RemoveHeader (llc);
  m_forwardUp (this, packet, llc.GetType (), source);
}

void
WimaxNetDevice::Attach (Ptr<WimaxChannel> channel)
{
  m_phy->Attach (channel);
}

void
WimaxNetDevice::SetPhy (Ptr<WimaxPhy> phy)
{
  m_phy = phy;
}

Ptr<WimaxPhy>
WimaxNetDevice::GetPhy (void) const
{
  return m_phy;
}

void
WimaxNetDevice::SetChannel (Ptr<WimaxChannel> channel)
{
  if (m_phy != 0)
    {
      m_phy->Attach (channel);
    }

}

uint64_t
WimaxNetDevice::GetChannel (uint8_t index) const
{
  return m_dlChannels.at (index);
}

void
WimaxNetDevice::SetNrFrames (uint32_t nrFrames)
{
  m_nrFrames = nrFrames;
}

uint32_t WimaxNetDevice::GetNrFrames (void) const
{

  return m_nrFrames;
}

void
WimaxNetDevice::SetAddress (Address address)
{
  m_address = Mac48Address::ConvertFrom (address);
}

void
WimaxNetDevice::SetMacAddress (Mac48Address address)
{
  m_address = address;
}

Address
WimaxNetDevice::GetAddress (void) const
{
  return m_address;
}

Mac48Address
WimaxNetDevice::GetMacAddress (void) const
{
  return m_address;
}

void
WimaxNetDevice::SetState (uint8_t state)
{
  m_state = state;
}

uint8_t
WimaxNetDevice::GetState (void) const
{
  return m_state;
}

Ptr<WimaxConnection>
WimaxNetDevice::GetInitialRangingConnection (void) const
{
  return m_initialRangingConnection;
}

Ptr<WimaxConnection>
WimaxNetDevice::GetBroadcastConnection (void) const
{
  return m_broadcastConnection;
}

void
WimaxNetDevice::SetCurrentDcd (Dcd dcd)
{
  m_currentDcd = dcd;
}

Dcd
WimaxNetDevice::GetCurrentDcd (void) const
{
  return m_currentDcd;
}

void
WimaxNetDevice::SetCurrentUcd (Ucd ucd)
{
  m_currentUcd = ucd;
}

Ucd
WimaxNetDevice::GetCurrentUcd (void) const
{
  return m_currentUcd;
}

Ptr<ConnectionManager>
WimaxNetDevice::GetConnectionManager (void) const
{
  return m_connectionManager;
}

void
WimaxNetDevice::SetConnectionManager (Ptr<ConnectionManager> cm)
{
  m_connectionManager = cm;
}

Ptr<BurstProfileManager>
WimaxNetDevice::GetBurstProfileManager (void) const
{
  return m_burstProfileManager;
}

void
WimaxNetDevice::SetBurstProfileManager (Ptr<BurstProfileManager> bpm)
{
  m_burstProfileManager = bpm;
}

Ptr<BandwidthManager>
WimaxNetDevice::GetBandwidthManager (void) const
{
  return m_bandwidthManager;
}

void
WimaxNetDevice::SetBandwidthManager (Ptr<BandwidthManager> bwm)
{
  m_bandwidthManager = bwm;
}

void
WimaxNetDevice::CreateDefaultConnections (void)
{
  m_initialRangingConnection = CreateObject<WimaxConnection> (Cid::InitialRanging (), Cid::INITIAL_RANGING);
  m_broadcastConnection = CreateObject<WimaxConnection> (Cid::Broadcast (), Cid::BROADCAST);
}

void
WimaxNetDevice::Receive (Ptr<const PacketBurst> burst)
{

  NS_LOG_DEBUG ("WimaxNetDevice::Receive, station = " << GetMacAddress ());

  Ptr<PacketBurst> b = burst->Copy ();
  for (std::list<Ptr<Packet> >::const_iterator iter = b->Begin (); iter != b->End (); ++iter)
    {
      Ptr<Packet> packet = *iter;
      DoReceive (packet);
    }
}

Ptr<WimaxChannel>
WimaxNetDevice::DoGetChannel (void) const
{
  return m_phy->GetChannel ();
}

void
WimaxNetDevice::SetReceiveCallback (void)
{
  m_phy->SetReceiveCallback (MakeCallback (&WimaxNetDevice::Receive, this));
}

bool
WimaxNetDevice::SendFrom (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
{

  Mac48Address from = Mac48Address::ConvertFrom (source);
  Mac48Address to = Mac48Address::ConvertFrom (dest);

  LlcSnapHeader llcHdr;
  llcHdr.SetType (protocolNumber);
  packet->AddHeader (llcHdr);

  m_traceTx (packet, to);
  return DoSend (packet, from, to, protocolNumber);
}

void
WimaxNetDevice::SetPromiscReceiveCallback (PromiscReceiveCallback cb)
{
  m_promiscRx = cb;
}

bool
WimaxNetDevice::IsPromisc (void)
{
  return !(m_promiscRx.IsNull ());
}

void
WimaxNetDevice::NotifyPromiscTrace (Ptr<Packet> p)
{
  // m_promiscRx(p);
}

bool
WimaxNetDevice::SupportsSendFrom (void) const
{
  return false;
}

void
WimaxNetDevice::ForwardDown (Ptr<PacketBurst> burst, WimaxPhy::ModulationType modulationType)
{
  SendParams * params = new OfdmSendParams (burst, modulationType, m_direction);
  m_phy->Send (params);
  delete params;
}

void
WimaxNetDevice::InitializeChannels (void)
{

  // initializing vector of channels (or frequencies)
  // Values according to WirelessMAN-OFDM RF profile for 10 MHz channelization
  // Section 12.3.3.1 from IEEE 802.16-2004 standard
  // profR10_3 :
  // channels: 5000 + n ⋅ 5 MHz, ∀n ∈ { 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167 }
  // from a range 5GHz to 6GHz, according to Section 8.5.1.
  uint64_t frequency = 5000;

  for (uint8_t i = 0; i < 200; i++)
    {
      m_dlChannels.push_back (frequency);
      frequency += 5;
    }
}

bool
WimaxNetDevice::IsBridge (void) const
{
  NS_LOG_FUNCTION_NOARGS ();
  return false;
}

Address
WimaxNetDevice::GetMulticast (Ipv4Address multicastGroup) const
{
  NS_LOG_FUNCTION (multicastGroup);

  Mac48Address ad = Mac48Address::GetMulticast (multicastGroup);

  //
  // Implicit conversion (operator Address ()) is defined for Mac48Address, so
  // use it by just returning the EUI-48 address which is automagically converted
  // to an Address.
  //
  NS_LOG_LOGIC ("multicast address is " << ad);

  return ad;
}

Address
WimaxNetDevice::GetMulticast (Ipv6Address addr) const
{
  Mac48Address ad = Mac48Address::GetMulticast (addr);

  NS_LOG_LOGIC ("MAC IPv6 multicast address is " << ad);
  return ad;
}

void
WimaxNetDevice::AddLinkChangeCallback (Callback<void> callback)
{
  /* \todo Add a callback invoked whenever the link
   * status changes to UP. This callback is typically used
   * by the IP/ARP layer to flush the ARP cache and by IPv6 stack
   * to flush NDISC cache whenever the link goes up.
   */
  NS_FATAL_ERROR ("Not implemented-- please implement and contribute a patch");
}
} // namespace ns3
