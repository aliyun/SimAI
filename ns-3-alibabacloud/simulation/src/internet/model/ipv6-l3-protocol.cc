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

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/uinteger.h"
#include "ns3/vector.h"
#include "ns3/boolean.h"
#include "ns3/callback.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/object-vector.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv6-route.h"
#include "ns3/mac16-address.h"
#include "ns3/mac64-address.h"
#include "ns3/traffic-control-layer.h"

#include "loopback-net-device.h"
#include "ipv6-l3-protocol.h"
#include "ipv6-interface.h"
#include "ipv6-raw-socket-impl.h"
#include "ipv6-autoconfigured-prefix.h"
#include "ipv6-extension-demux.h"
#include "ipv6-extension.h"
#include "ipv6-extension-header.h"
#include "ipv6-option-demux.h"
#include "ipv6-option.h"
#include "icmpv6-l4-protocol.h"
#include "ndisc-cache.h"
#include "ipv6-raw-socket-factory-impl.h"

/// Minimum IPv6 MTU, as defined by \RFC{2460}
#define IPV6_MIN_MTU 1280

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6L3Protocol");

NS_OBJECT_ENSURE_REGISTERED (Ipv6L3Protocol);

const uint16_t Ipv6L3Protocol::PROT_NUMBER = 0x86DD;

TypeId Ipv6L3Protocol::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6L3Protocol")
    .SetParent<Ipv6> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv6L3Protocol> ()
    .AddAttribute ("DefaultTtl",
                   "The TTL value set by default on all "
                   "outgoing packets generated on this node.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&Ipv6L3Protocol::m_defaultTtl),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("DefaultTclass",
                   "The TCLASS value set by default on all "
                   "outgoing packets generated on this node.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&Ipv6L3Protocol::m_defaultTclass),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("InterfaceList",
                   "The set of IPv6 interfaces associated to this IPv6 stack.",
                   ObjectVectorValue (),
                   MakeObjectVectorAccessor (&Ipv6L3Protocol::m_interfaces),
                   MakeObjectVectorChecker<Ipv6Interface> ())
    .AddAttribute ("SendIcmpv6Redirect",
                   "Send the ICMPv6 Redirect when appropriate.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&Ipv6L3Protocol::SetSendIcmpv6Redirect,
                                        &Ipv6L3Protocol::GetSendIcmpv6Redirect),
                   MakeBooleanChecker ())
    .AddAttribute ("StrongEndSystemModel",
                   "Reject packets for an address not configured on the interface they're coming from (RFC1222).",
                   BooleanValue (true),
                   MakeBooleanAccessor (&Ipv6L3Protocol::m_strongEndSystemModel),
                   MakeBooleanChecker ())
    .AddTraceSource ("Tx",
                     "Send IPv6 packet to outgoing interface.",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_txTrace),
                     "ns3::Ipv6L3Protocol::TxRxTracedCallback")
    .AddTraceSource ("Rx",
                     "Receive IPv6 packet from incoming interface.",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_rxTrace),
                     "ns3::Ipv6L3Protocol::TxRxTracedCallback")
    .AddTraceSource ("Drop",
                     "Drop IPv6 packet",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_dropTrace),
                     "ns3::Ipv6L3Protocol::DropTracedCallback")

    .AddTraceSource ("SendOutgoing",
                     "A newly-generated packet by this node is "
                     "about to be queued for transmission",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_sendOutgoingTrace),
                     "ns3::Ipv6L3Protocol::SentTracedCallback")
    .AddTraceSource ("UnicastForward",
                     "A unicast IPv6 packet was received by this node "
                     "and is being forwarded to another node",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_unicastForwardTrace),
                     "ns3::Ipv6L3Protocol::SentTracedCallback")
    .AddTraceSource ("LocalDeliver",
                     "An IPv6 packet was received by/for this node, "
                     "and it is being forward up the stack",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_localDeliverTrace),
                     "ns3::Ipv6L3Protocol::SentTracedCallback")
  ;
  return tid;
}

Ipv6L3Protocol::Ipv6L3Protocol ()
  : m_nInterfaces (0)
{
  NS_LOG_FUNCTION (this);
  m_pmtuCache = CreateObject<Ipv6PmtuCache> ();
  
  Ptr<Ipv6RawSocketFactoryImpl> rawFactoryImpl = CreateObject<Ipv6RawSocketFactoryImpl> ();
  AggregateObject (rawFactoryImpl);
}

Ipv6L3Protocol::~Ipv6L3Protocol ()
{
  NS_LOG_FUNCTION (this);
}

void Ipv6L3Protocol::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  /* clear protocol and interface list */
  for (L4List_t::iterator it = m_protocols.begin (); it != m_protocols.end (); ++it)
    {
      it->second = 0;
    }
  m_protocols.clear ();

  /* remove interfaces */
  for (Ipv6InterfaceList::iterator it = m_interfaces.begin (); it != m_interfaces.end (); ++it)
    {
      *it = 0;
    }
  m_interfaces.clear ();
  m_reverseInterfacesContainer.clear ();

  /* remove raw sockets */
  for (SocketList::iterator it = m_sockets.begin (); it != m_sockets.end (); ++it)
    {
      *it = 0;
    }
  m_sockets.clear ();

  /* remove list of prefix */
  for (Ipv6AutoconfiguredPrefixListI it = m_prefixes.begin (); it != m_prefixes.end (); ++it)
    {
      (*it)->StopValidTimer ();
      (*it)->StopPreferredTimer ();
      (*it) = 0;
    }
  m_prefixes.clear ();

  m_node = 0;
  m_routingProtocol = 0;
  m_pmtuCache = 0;
  Object::DoDispose ();
}

void Ipv6L3Protocol::SetRoutingProtocol (Ptr<Ipv6RoutingProtocol> routingProtocol)
{
  NS_LOG_FUNCTION (this << routingProtocol);
  m_routingProtocol = routingProtocol;
  m_routingProtocol->SetIpv6 (this);
}

Ptr<Ipv6RoutingProtocol> Ipv6L3Protocol::GetRoutingProtocol () const
{
  NS_LOG_FUNCTION (this);
  return m_routingProtocol;
}

uint32_t Ipv6L3Protocol::AddInterface (Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << device);
  Ptr<Ipv6Interface> interface = CreateObject<Ipv6Interface> ();

  Ptr<TrafficControlLayer> tc = m_node->GetObject<TrafficControlLayer> ();

  NS_ASSERT (tc != 0);

  m_node->RegisterProtocolHandler (MakeCallback (&TrafficControlLayer::Receive, tc),
                                   Ipv6L3Protocol::PROT_NUMBER, device);

  tc->RegisterProtocolHandler (MakeCallback (&Ipv6L3Protocol::Receive, this),
                               Ipv6L3Protocol::PROT_NUMBER, device);

  interface->SetNode (m_node);
  interface->SetDevice (device);
  interface->SetTrafficControl (tc);
  interface->SetForwarding (m_ipForward);
  return AddIpv6Interface (interface);
}

uint32_t Ipv6L3Protocol::AddIpv6Interface (Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << interface);
  uint32_t index = m_nInterfaces;

  m_interfaces.push_back (interface);
  m_reverseInterfacesContainer[interface->GetDevice ()] = index;
  m_nInterfaces++;
  return index;
}

Ptr<Ipv6Interface> Ipv6L3Protocol::GetInterface (uint32_t index) const
{
  NS_LOG_FUNCTION (this << index);

  if (index < m_interfaces.size ())
    {
      return m_interfaces[index];
    }
  return 0;
}

uint32_t Ipv6L3Protocol::GetNInterfaces () const
{
  NS_LOG_FUNCTION (this);
  return m_nInterfaces;
}

int32_t Ipv6L3Protocol::GetInterfaceForAddress (Ipv6Address address) const
{
  NS_LOG_FUNCTION (this << address);
  int32_t index = 0;

  for (Ipv6InterfaceList::const_iterator it = m_interfaces.begin (); it != m_interfaces.end (); it++)
    {
      uint32_t j = 0;
      uint32_t max = (*it)->GetNAddresses ();

      for (j = 0; j < max; j++)
        {
          if ((*it)->GetAddress (j).GetAddress () == address)
            {
              return index;
            }
        }
      index++;
    }
  return -1;
}

int32_t Ipv6L3Protocol::GetInterfaceForPrefix (Ipv6Address address, Ipv6Prefix mask) const
{
  NS_LOG_FUNCTION (this << address << mask);
  int32_t index = 0;

  for (Ipv6InterfaceList::const_iterator it = m_interfaces.begin (); it != m_interfaces.end (); it++)
    {
      uint32_t j = 0;
      for (j = 0; j < (*it)->GetNAddresses (); j++)
        {
          if ((*it)->GetAddress (j).GetAddress ().CombinePrefix (mask) == address.CombinePrefix (mask))
            {
              return index;
            }
        }
      index++;
    }
  return -1;
}

Ptr<NetDevice> Ipv6L3Protocol::GetNetDevice (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  return GetInterface (i)->GetDevice ();
}

int32_t Ipv6L3Protocol::GetInterfaceForDevice (Ptr<const NetDevice> device) const
{
  NS_LOG_FUNCTION (this << device);

  Ipv6InterfaceReverseContainer::const_iterator iter = m_reverseInterfacesContainer.find (device);
  if (iter != m_reverseInterfacesContainer.end ())
    {
      return (*iter).second;
    }

  return -1;
}

void Ipv6L3Protocol::AddAutoconfiguredAddress (uint32_t interface, Ipv6Address network, Ipv6Prefix mask, uint8_t flags, uint32_t validTime, uint32_t preferredTime, Ipv6Address defaultRouter)
{
  NS_LOG_FUNCTION (this << interface << network << mask << (uint32_t)flags << validTime << preferredTime);
  Ipv6InterfaceAddress address;

  Address addr = GetInterface (interface)->GetDevice ()->GetAddress ();

  if (!defaultRouter.IsAny())
    {
      GetRoutingProtocol ()->NotifyAddRoute (Ipv6Address::GetAny (), Ipv6Prefix ((uint8_t)0), defaultRouter, interface, network);
    }

  bool onLink = false;
  if (flags & Icmpv6OptionPrefixInformation::ONLINK)
    {
      onLink = true;
    }

  if (flags & Icmpv6OptionPrefixInformation::AUTADDRCONF) /* auto flag */
    {
      address = Ipv6Address::MakeAutoconfiguredAddress (addr, network);
      address.SetOnLink (onLink);

      /* see if we have already the prefix */
      for (Ipv6AutoconfiguredPrefixListI it = m_prefixes.begin (); it != m_prefixes.end (); ++it)
        {
          if ((*it)->GetInterface () == interface && (*it)->GetPrefix () == network && (*it)->GetMask () == mask)
            {
              (*it)->StopPreferredTimer ();
              (*it)->StopValidTimer ();
              (*it)->StartPreferredTimer ();
              return;
            }
        }

      /* no prefix found, add autoconfigured address and the prefix */
      NS_LOG_INFO ("Autoconfigured address is :" << address.GetAddress ());
      AddAddress (interface, address, onLink);

      Ptr<Ipv6AutoconfiguredPrefix> aPrefix = CreateObject<Ipv6AutoconfiguredPrefix> (m_node, interface, network, mask, preferredTime, validTime, defaultRouter);
      aPrefix->StartPreferredTimer ();

      m_prefixes.push_back (aPrefix);
    }

  if (onLink) /* on-link flag */
    {
      /* add default router
       * if a previous default route exists, the new ones is simply added
       */
      m_routingProtocol->NotifyAddRoute (network, mask, Ipv6Address::GetAny (), interface);
    }
}

void Ipv6L3Protocol::RemoveAutoconfiguredAddress (uint32_t interface, Ipv6Address network, Ipv6Prefix mask, Ipv6Address defaultRouter)
{
  NS_LOG_FUNCTION (this << interface << network << mask);
  Ptr<Ipv6Interface> iface = GetInterface (interface);
  Address addr = iface->GetDevice ()->GetAddress ();
  
  Ipv6Address addressToFind = Ipv6Address::MakeAutoconfiguredAddress (addr, network);

  for (uint32_t i = 0; i < iface->GetNAddresses (); i++)
    {
      if (iface->GetAddress (i).GetAddress () == addressToFind)
        {
          RemoveAddress (interface, i);
          break;
        }
    }
  
  /* remove from list of autoconfigured address */
  for (Ipv6AutoconfiguredPrefixListI it = m_prefixes.begin (); it != m_prefixes.end (); ++it)
    {
      if ((*it)->GetInterface () == interface && (*it)->GetPrefix () == network && (*it)->GetMask () == mask)
        {
          *it = 0;
          m_prefixes.erase (it);
          break;
        }
    }

  GetRoutingProtocol ()->NotifyRemoveRoute (Ipv6Address::GetAny (), Ipv6Prefix ((uint8_t)0), defaultRouter, interface, network);
}

bool Ipv6L3Protocol::AddAddress (uint32_t i, Ipv6InterfaceAddress address, bool addOnLinkRoute)
{
  NS_LOG_FUNCTION (this << i << address);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  address.SetOnLink (addOnLinkRoute);
  bool ret = interface->AddAddress (address);

  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyAddAddress (i, address);
    }

  if (addOnLinkRoute)
    {
      Ipv6Address networkAddress = address.GetAddress ().CombinePrefix (address.GetPrefix ());
      Ipv6Prefix networkMask = address.GetPrefix ();
      GetRoutingProtocol ()->NotifyAddRoute (networkAddress, networkMask, Ipv6Address::GetZero (), i);
    }
  return ret;
}

uint32_t Ipv6L3Protocol::GetNAddresses (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  return interface->GetNAddresses ();
}

Ipv6InterfaceAddress Ipv6L3Protocol::GetAddress (uint32_t i, uint32_t addressIndex) const
{
  NS_LOG_FUNCTION (this << i << addressIndex);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  return interface->GetAddress (addressIndex);
}

bool Ipv6L3Protocol::RemoveAddress (uint32_t i, uint32_t addressIndex)
{
  NS_LOG_FUNCTION (this << i << addressIndex);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  Ipv6InterfaceAddress address = interface->RemoveAddress (addressIndex);

  if (address != Ipv6InterfaceAddress ())
    {
      if (m_routingProtocol != 0)
        {
          m_routingProtocol->NotifyRemoveAddress (i, address);
        }
      return true;
    }
  return false;
}

bool 
Ipv6L3Protocol::RemoveAddress (uint32_t i, Ipv6Address address)
{
  NS_LOG_FUNCTION (this << i << address);

  if (address == Ipv6Address::GetLoopback())
    {
      NS_LOG_WARN ("Cannot remove loopback address.");
      return false;
    }
  Ptr<Ipv6Interface> interface = GetInterface (i);
  Ipv6InterfaceAddress ifAddr = interface->RemoveAddress (address);
  if (ifAddr != Ipv6InterfaceAddress ())
  {
    if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyRemoveAddress (i, ifAddr);
    }
    return true;
  }
  return false;
}

void Ipv6L3Protocol::SetMetric (uint32_t i, uint16_t metric)
{
  NS_LOG_FUNCTION (this << i << metric);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  interface->SetMetric (metric);
}

uint16_t Ipv6L3Protocol::GetMetric (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  return interface->GetMetric ();
}

uint16_t Ipv6L3Protocol::GetMtu (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);

  // RFC 1981, if PMTU is disabled, return the minimum MTU
  if (!m_mtuDiscover)
    {
      return IPV6_MIN_MTU;
    }

  Ptr<Ipv6Interface> interface = GetInterface (i);
  return interface->GetDevice ()->GetMtu ();
}

void Ipv6L3Protocol::SetPmtu (Ipv6Address dst, uint32_t pmtu)
{
  NS_LOG_FUNCTION (this << dst << int(pmtu));
  m_pmtuCache->SetPmtu (dst, pmtu);
}


bool Ipv6L3Protocol::IsUp (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  return interface->IsUp ();
}

void Ipv6L3Protocol::SetUp (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);

  // RFC 2460, Section 5, pg. 24:
  //  IPv6 requires that every link in the internet have an MTU of 1280
  //  octets or greater.  On any link that cannot convey a 1280-octet
  //  packet in one piece, link-specific fragmentation and reassembly must
  //  be provided at a layer below IPv6.
  if (interface->GetDevice ()->GetMtu () >= 1280)
    {
      interface->SetUp ();

      if (m_routingProtocol != 0)
        {
          m_routingProtocol->NotifyInterfaceUp (i);
        }
    }
  else
    {
      NS_LOG_LOGIC ("Interface " << int(i) << " is set to be down for IPv6. Reason: not respecting minimum IPv6 MTU (1280 octets)");
    }
}

void Ipv6L3Protocol::SetDown (uint32_t i)
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);

  interface->SetDown ();

  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceDown (i);
    }
}

void Ipv6L3Protocol::SetupLoopback ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv6Interface> interface = CreateObject<Ipv6Interface> ();
  Ptr<LoopbackNetDevice> device = 0;
  uint32_t i = 0;

  /* see if we have already an loopback NetDevice */
  for (i = 0; i < m_node->GetNDevices (); i++)
    {
      if ((device = DynamicCast<LoopbackNetDevice> (m_node->GetDevice (i))))
        {
          break;
        }
    }

  if (device == 0)
    {
      device = CreateObject<LoopbackNetDevice> ();
      m_node->AddDevice (device);
    }

  interface->SetDevice (device);
  interface->SetNode (m_node);
  Ipv6InterfaceAddress ifaceAddr = Ipv6InterfaceAddress (Ipv6Address::GetLoopback (), Ipv6Prefix (128));
  interface->AddAddress (ifaceAddr);
  uint32_t index = AddIpv6Interface (interface);
  Ptr<Node> node = GetObject<Node> ();
  node->RegisterProtocolHandler (MakeCallback (&Ipv6L3Protocol::Receive, this), Ipv6L3Protocol::PROT_NUMBER, device);
  interface->SetUp ();

  if (m_routingProtocol != 0)
    {
      m_routingProtocol->NotifyInterfaceUp (index);
    }
}

bool Ipv6L3Protocol::IsForwarding (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  Ptr<Ipv6Interface> interface = GetInterface (i);

  NS_LOG_LOGIC ("Forwarding state: " << interface->IsForwarding ());
  return interface->IsForwarding ();
}

void Ipv6L3Protocol::SetForwarding (uint32_t i, bool val)
{
  NS_LOG_FUNCTION (this << i << val);
  Ptr<Ipv6Interface> interface = GetInterface (i);
  interface->SetForwarding (val);
}

Ipv6Address Ipv6L3Protocol::SourceAddressSelection (uint32_t interface, Ipv6Address dest)
{
  NS_LOG_FUNCTION (this << interface << dest);
  Ipv6Address ret;

  if (dest.IsLocalhost ())
    {
      return Ipv6Address::GetLoopback ();
    }
  
  if (dest.IsLinkLocal () || dest.IsLinkLocalMulticast ())
    {
      for (uint32_t i = 0; i < GetNAddresses (interface); i++)
        {
          Ipv6InterfaceAddress test = GetAddress (interface, i);
          if (test.GetScope () == Ipv6InterfaceAddress::LINKLOCAL)
            {
              return test.GetAddress ();
            }
        }
      NS_ASSERT_MSG (false, "No link-local address found on interface " << interface);
    }

  for (uint32_t i = 0; i < GetNAddresses (interface); i++)
    {
      Ipv6InterfaceAddress test = GetAddress (interface, i);

      if (test.GetScope () == Ipv6InterfaceAddress::GLOBAL)
        {
          if (test.IsInSameSubnet (dest))
            {
              return test.GetAddress ();
            }
          else
            {
              ret = test.GetAddress ();
            }
        }
    }

  // no specific match found. Use a global address (any useful is fine).
  NS_ASSERT_MSG (!ret.IsAny (),
                 "Could not find any address for " << dest << " on interface " << interface);
  return ret;
}

void Ipv6L3Protocol::SetIpForward (bool forward)
{
  NS_LOG_FUNCTION (this << forward);
  m_ipForward = forward;

  for (Ipv6InterfaceList::const_iterator it = m_interfaces.begin (); it != m_interfaces.end (); it++)
    {
      (*it)->SetForwarding (forward);
    }
}

bool Ipv6L3Protocol::GetIpForward () const
{
  NS_LOG_FUNCTION (this);
  return m_ipForward;
}

void Ipv6L3Protocol::SetMtuDiscover (bool mtuDiscover)
{
  NS_LOG_FUNCTION (this << int(mtuDiscover));
  m_mtuDiscover = mtuDiscover;
}

bool Ipv6L3Protocol::GetMtuDiscover () const
{
  NS_LOG_FUNCTION (this);
  return m_mtuDiscover;
}

void Ipv6L3Protocol::SetSendIcmpv6Redirect (bool sendIcmpv6Redirect)
{
  NS_LOG_FUNCTION (this << sendIcmpv6Redirect);
  m_sendIcmpv6Redirect = sendIcmpv6Redirect;
}

bool Ipv6L3Protocol::GetSendIcmpv6Redirect () const
{
  NS_LOG_FUNCTION (this);
  return m_sendIcmpv6Redirect;
}

void Ipv6L3Protocol::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);

  if (m_node == 0)
    {
      Ptr<Node> node = this->GetObject<Node> ();
      // verify that it's a valid node and that
      // the node has not been set before
      if (node != 0)
        {
          this->SetNode (node);
        }
    }
  
  Ipv6::NotifyNewAggregate ();
}

void Ipv6L3Protocol::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
  /* add LoopbackNetDevice if needed, and an Ipv6Interface on top of it */
  SetupLoopback ();
}

void Ipv6L3Protocol::Insert (Ptr<IpL4Protocol> protocol)
{
  NS_LOG_FUNCTION (this << protocol);
  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), -1);
  if (m_protocols.find (key) != m_protocols.end ())
    {
      NS_LOG_WARN ("Overwriting default protocol " << int(protocol->GetProtocolNumber ()));
    }
  m_protocols[key] = protocol;
}

void Ipv6L3Protocol::Insert (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
{
  NS_LOG_FUNCTION (this << protocol << interfaceIndex);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), interfaceIndex);
  if (m_protocols.find (key) != m_protocols.end ())
    {
      NS_LOG_WARN ("Overwriting protocol " << int(protocol->GetProtocolNumber ()) << " on interface " << int(interfaceIndex));
    }
  m_protocols[key] = protocol;
}

void Ipv6L3Protocol::Remove (Ptr<IpL4Protocol> protocol)
{
  NS_LOG_FUNCTION (this << protocol);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), -1);
  L4List_t::iterator iter = m_protocols.find (key);
  if (iter == m_protocols.end ())
    {
      NS_LOG_WARN ("Trying to remove an non-existent default protocol " << int(protocol->GetProtocolNumber ()));
    }
  else
    {
      m_protocols.erase (key);
    }
}

void Ipv6L3Protocol::Remove (Ptr<IpL4Protocol> protocol, uint32_t interfaceIndex)
{
  NS_LOG_FUNCTION (this << protocol << interfaceIndex);

  L4ListKey_t key = std::make_pair (protocol->GetProtocolNumber (), interfaceIndex);
  L4List_t::iterator iter = m_protocols.find (key);
  if (iter == m_protocols.end ())
    {
      NS_LOG_WARN ("Trying to remove an non-existent protocol " << int(protocol->GetProtocolNumber ()) << " on interface " << int(interfaceIndex));
    }
  else
    {
      m_protocols.erase (key);
    }
}

Ptr<IpL4Protocol> Ipv6L3Protocol::GetProtocol (int protocolNumber) const
{
  NS_LOG_FUNCTION (this << protocolNumber);

  return GetProtocol (protocolNumber, -1);
}

Ptr<IpL4Protocol> Ipv6L3Protocol::GetProtocol (int protocolNumber, int32_t interfaceIndex) const
{
  NS_LOG_FUNCTION (this << protocolNumber << interfaceIndex);

  L4ListKey_t key;
  L4List_t::const_iterator i;
  if (interfaceIndex >= 0)
    {
      // try the interface-specific protocol.
      key = std::make_pair (protocolNumber, interfaceIndex);
      i = m_protocols.find (key);
      if (i != m_protocols.end ())
        {
          return i->second;
        }
    }
  // try the generic protocol.
  key = std::make_pair (protocolNumber, -1);
  i = m_protocols.find (key);
  if (i != m_protocols.end ())
    {
      return i->second;
    }

  return 0;
}

Ptr<Socket> Ipv6L3Protocol::CreateRawSocket ()
{
  NS_LOG_FUNCTION (this);
  Ptr<Ipv6RawSocketImpl> sock = CreateObject<Ipv6RawSocketImpl> ();
  sock->SetNode (m_node);
  m_sockets.push_back (sock);
  return sock;
}

void Ipv6L3Protocol::DeleteRawSocket (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  for (SocketList::iterator it = m_sockets.begin (); it != m_sockets.end (); ++it)
    {
      if ((*it) == socket)
        {
          m_sockets.erase (it);
          return;
        }
    }
}

Ptr<Icmpv6L4Protocol> Ipv6L3Protocol::GetIcmpv6 () const
{
  NS_LOG_FUNCTION (this);
  Ptr<IpL4Protocol> protocol = GetProtocol (Icmpv6L4Protocol::GetStaticProtocolNumber ());

  if (protocol)
    {
      return protocol->GetObject<Icmpv6L4Protocol> ();
    }
  else
    {
      return 0;
    }
}

void Ipv6L3Protocol::SetDefaultTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION (this << ttl);
  m_defaultTtl = ttl;
}

void Ipv6L3Protocol::SetDefaultTclass (uint8_t tclass)
{
  NS_LOG_FUNCTION (this << tclass);
  m_defaultTclass = tclass;
}

void Ipv6L3Protocol::Send (Ptr<Packet> packet, Ipv6Address source, Ipv6Address destination, uint8_t protocol, Ptr<Ipv6Route> route)
{
  NS_LOG_FUNCTION (this << packet << source << destination << (uint32_t)protocol << route);
  Ipv6Header hdr;
  uint8_t ttl = m_defaultTtl;
  SocketIpv6HopLimitTag tag;
  bool found = packet->RemovePacketTag (tag);

  if (found)
    {
      ttl = tag.GetHopLimit ();
    }

  SocketIpv6TclassTag tclassTag;
  uint8_t tclass = m_defaultTclass;
  found = packet->RemovePacketTag (tclassTag);
  
  if (found)
    {
      tclass = tclassTag.GetTclass ();
    }

  /* Handle 3 cases:
   * 1) Packet is passed in with a route entry
   * 2) Packet is passed in with a route entry but route->GetGateway is not set (e.g., same network)
   * 3) route is NULL (e.g., a raw socket call or ICMPv6)
   */

  /* 1) */
  if (route && route->GetGateway () != Ipv6Address::GetZero ())
    {
      NS_LOG_LOGIC ("Ipv6L3Protocol::Send case 1: passed in with a route");
      hdr = BuildHeader (source, destination, protocol, packet->GetSize (), ttl, tclass);
      int32_t interface = GetInterfaceForDevice (route->GetOutputDevice ());
      m_sendOutgoingTrace (hdr, packet, interface);
      SendRealOut (route, packet, hdr);
      return;
    }

  /* 2) */
  if (route && route->GetGateway () == Ipv6Address::GetZero ())
    {
      NS_LOG_LOGIC ("Ipv6L3Protocol::Send case 2: probably sent to machine on same IPv6 network");
      hdr = BuildHeader (source, destination, protocol, packet->GetSize (), ttl, tclass);
      int32_t interface = GetInterfaceForDevice (route->GetOutputDevice ());
      m_sendOutgoingTrace (hdr, packet, interface);
      SendRealOut (route, packet, hdr);
      return;
    }

  /* 3) */
  NS_LOG_LOGIC ("Ipv6L3Protocol::Send case 3: passed in with no route " << destination);
  Socket::SocketErrno err;
  Ptr<NetDevice> oif (0);
  Ptr<Ipv6Route> newRoute = 0;

  hdr = BuildHeader (source, destination, protocol, packet->GetSize (), ttl, tclass);

  //for link-local traffic, we need to determine the interface
  if (source.IsLinkLocal ()
      || destination.IsLinkLocal ()
      || destination.IsLinkLocalMulticast ())
    {
      int32_t index = GetInterfaceForAddress (source);
      NS_ASSERT_MSG (index >= 0, "Can not find an outgoing interface for a packet with src " << source << " and dst " << destination);
      oif = GetNetDevice (index);
    }

  newRoute = m_routingProtocol->RouteOutput (packet, hdr, oif, err);

  if (newRoute)
    {
      int32_t interface = GetInterfaceForDevice (newRoute->GetOutputDevice ());
      m_sendOutgoingTrace (hdr, packet, interface);
      SendRealOut (newRoute, packet, hdr);
    }
  else
    {
      NS_LOG_WARN ("No route to host, drop!");
      m_dropTrace (hdr, packet, DROP_NO_ROUTE, this, GetInterfaceForDevice (oif));
    }
}

void Ipv6L3Protocol::Receive (Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address &from, const Address &to, NetDevice::PacketType packetType)
{
  NS_LOG_FUNCTION (this << device << p << protocol << from << to << packetType);
  NS_LOG_LOGIC ("Packet from " << from << " received on node " << m_node->GetId ());

  NS_ASSERT_MSG (GetInterfaceForDevice(device) != -1, "Received a packet from an interface that is not known to IPv6");
  uint32_t interface = GetInterfaceForDevice(device);

  Ptr<Ipv6Interface> ipv6Interface = m_interfaces[interface];
  Ptr<Packet> packet = p->Copy ();

  if (ipv6Interface->IsUp ())
    {
      m_rxTrace (packet, this, interface);
    }
  else
    {
      NS_LOG_LOGIC ("Dropping received packet-- interface is down");
      Ipv6Header hdr;
      packet->RemoveHeader (hdr);
      m_dropTrace (hdr, packet, DROP_INTERFACE_DOWN, this, interface);
      return;
    }

  Ipv6Header hdr;
  packet->RemoveHeader (hdr);

  // Trim any residual frame padding from underlying devices
  if (hdr.GetPayloadLength () < packet->GetSize ())
    {
      packet->RemoveAtEnd (packet->GetSize () - hdr.GetPayloadLength ());
    }

  // the packet is valid, we update the NDISC cache entry (if present)
  Ptr<NdiscCache> ndiscCache = ipv6Interface->GetNdiscCache ();
  if (ndiscCache)
    {
      // case one, it's a a direct routing.
      NdiscCache::Entry *entry = ndiscCache->Lookup (hdr.GetSource ());
      if (entry)
        {
          entry->UpdateReachableTimer ();
        }
      else
        {
          // It's not in the direct routing, so it's the router, and it could have multiple IP addresses.
          // In doubt, update all of them.
          // Note: it's a confirmed behavior for Linux routers.
          std::list<NdiscCache::Entry *> entryList = ndiscCache->LookupInverse (from);
          std::list<NdiscCache::Entry *>::iterator iter;
          for (iter = entryList.begin (); iter != entryList.end (); iter ++)
            {
              (*iter)->UpdateReachableTimer ();
            }
        }
    }



  /* forward up to IPv6 raw sockets */
  for (SocketList::iterator it = m_sockets.begin (); it != m_sockets.end (); ++it)
    {
      Ptr<Ipv6RawSocketImpl> socket = *it;
      socket->ForwardUp (packet, hdr, device);
    }

  Ptr<Ipv6ExtensionDemux> ipv6ExtensionDemux = m_node->GetObject<Ipv6ExtensionDemux> ();
  Ptr<Ipv6Extension> ipv6Extension = 0;
  uint8_t nextHeader = hdr.GetNextHeader ();
  bool stopProcessing = false;
  bool isDropped = false;
  DropReason dropReason;

  if (nextHeader == Ipv6Header::IPV6_EXT_HOP_BY_HOP)
    {
      ipv6Extension = ipv6ExtensionDemux->GetExtension (nextHeader);

      if (ipv6Extension)
        {
          ipv6Extension->Process (packet, 0, hdr, hdr.GetDestination (), (uint8_t *)0, stopProcessing, isDropped, dropReason);
        }

      if (isDropped)
        {
          m_dropTrace (hdr, packet, dropReason, this, interface);
        }

      if (stopProcessing)
        {
          return;
        }
    }

  if (hdr.GetDestination ().IsAllNodesMulticast ())
    {
      LocalDeliver (packet, hdr, interface);
      return;
    }
  else if (hdr.GetDestination ().IsAllRoutersMulticast() && ipv6Interface->IsForwarding ())
    {
      LocalDeliver (packet, hdr, interface);
      return;
    }
  else if (hdr.GetDestination ().IsMulticast ())
    {
      bool isSolicited = ipv6Interface->IsSolicitedMulticastAddress (hdr.GetDestination ());
      bool isRegisteredOnInterface = IsRegisteredMulticastAddress (hdr.GetDestination (), interface);
      bool isRegisteredGlobally = IsRegisteredMulticastAddress (hdr.GetDestination ());
      if (isSolicited || isRegisteredGlobally || isRegisteredOnInterface)
        {
          LocalDeliver (packet, hdr, interface);
          // do not return, the packet could be handled by a routing protocol
        }
    }


  for (uint32_t j = 0; j < GetNInterfaces (); j++)
    {
      if (j == interface || !m_strongEndSystemModel)
        {
          for (uint32_t i = 0; i < GetNAddresses (j); i++)
            {
              Ipv6InterfaceAddress iaddr = GetAddress (j, i);
              Ipv6Address addr = iaddr.GetAddress ();
              if (addr == hdr.GetDestination ())
                {
                  if (j == interface)
                    {
                      NS_LOG_LOGIC ("For me (destination " << addr << " match)");
                    }
                  else
                    {
                      NS_LOG_LOGIC ("For me (destination " << addr << " match) on another interface " << hdr.GetDestination ());
                    }
                  LocalDeliver (packet, hdr, interface);
                  return;
                }
              NS_LOG_LOGIC ("Address " << addr << " not a match");
            }
        }
    }

  if (!m_routingProtocol->RouteInput (packet, hdr, device,
                                      MakeCallback (&Ipv6L3Protocol::IpForward, this),
                                      MakeCallback (&Ipv6L3Protocol::IpMulticastForward, this),
                                      MakeCallback (&Ipv6L3Protocol::LocalDeliver, this),
                                      MakeCallback (&Ipv6L3Protocol::RouteInputError, this)))
    {
      NS_LOG_WARN ("No route found for forwarding packet.  Drop.");
      // Drop trace and ICMPs are courtesy of RouteInputError
    }
}

void
Ipv6L3Protocol::CallTxTrace (const Ipv6Header & ipHeader, Ptr<Packet> packet,
                                    Ptr<Ipv6> ipv6, uint32_t interface)
{
  if (!m_txTrace.IsEmpty ())
  {
    Ptr<Packet> packetCopy = packet->Copy ();
    packetCopy->AddHeader (ipHeader);
    m_txTrace (packetCopy, ipv6, interface);
  }
}

void Ipv6L3Protocol::SendRealOut (Ptr<Ipv6Route> route, Ptr<Packet> packet, Ipv6Header const& ipHeader)
{
  NS_LOG_FUNCTION (this << route << packet << ipHeader);

  if (!route)
    {
      NS_LOG_LOGIC ("No route to host, drop!.");
      return;
    }

  Ptr<NetDevice> dev = route->GetOutputDevice ();
  int32_t interface = GetInterfaceForDevice (dev);
  NS_ASSERT (interface >= 0);

  Ptr<Ipv6Interface> outInterface = GetInterface (interface);
  NS_LOG_LOGIC ("Send via NetDevice ifIndex " << dev->GetIfIndex () << " Ipv6InterfaceIndex " << interface);

  // Check packet size
  std::list<Ipv6ExtensionFragment::Ipv6PayloadHeaderPair> fragments;

  // Check if this is the source of the packet
  bool fromMe = false;
  for (uint32_t i = 0; i < GetNInterfaces (); i++ )
    {
      for (uint32_t j = 0; j < GetNAddresses (i); j++ )
        {
          if (GetAddress (i,j).GetAddress () == ipHeader.GetSource ())
            {
              fromMe = true;
              break;
            }
        }
    }

  size_t targetMtu = 0;

  // Check if we have a Path MTU stored. If so, use it. Else, use the link MTU.
  // Note: PMTU must not be cached in intermediate nodes, and must be checked only by the source node
  if (fromMe)
    {
      targetMtu = (size_t)(m_pmtuCache->GetPmtu (ipHeader.GetDestination ()));
    }
  if (targetMtu == 0)
    {
      targetMtu = dev->GetMtu ();
    }

  if (packet->GetSize () + ipHeader.GetSerializedSize () > targetMtu)
    {
      // Router => drop
      if (!fromMe)
        {
          Ptr<Icmpv6L4Protocol> icmpv6 = GetIcmpv6 ();
          if ( icmpv6 )
            {
              packet->AddHeader (ipHeader);
              icmpv6->SendErrorTooBig (packet, ipHeader.GetSource (), dev->GetMtu ());
            }
          return;
        }

      Ptr<Ipv6ExtensionDemux> ipv6ExtensionDemux = m_node->GetObject<Ipv6ExtensionDemux> ();

      // To get specific method GetFragments from Ipv6ExtensionFragmentation
      Ipv6ExtensionFragment *ipv6Fragment = dynamic_cast<Ipv6ExtensionFragment *> (PeekPointer (ipv6ExtensionDemux->GetExtension (Ipv6Header::IPV6_EXT_FRAGMENTATION)));
      NS_ASSERT (ipv6Fragment != 0);
      ipv6Fragment->GetFragments (packet, ipHeader, targetMtu, fragments);
    }

  if (route->GetGateway () != Ipv6Address::GetAny ())
    {
      if (outInterface->IsUp ())
        {
          NS_LOG_LOGIC ("Send to gateway " << route->GetGateway ());

          if (fragments.size () != 0)
            {
              std::ostringstream oss;

              for (std::list<Ipv6ExtensionFragment::Ipv6PayloadHeaderPair>::const_iterator it = fragments.begin (); it != fragments.end (); it++)
                {
                  CallTxTrace (it->second, it->first, this, interface);
                  outInterface->Send (it->first, it->second, route->GetGateway ());
                }
            }
          else
            {
              CallTxTrace (ipHeader, packet, this, interface);
              outInterface->Send (packet, ipHeader, route->GetGateway ());
            }
        }
      else
        {
          NS_LOG_LOGIC ("Dropping-- outgoing interface is down: " << route->GetGateway ());
          m_dropTrace (ipHeader, packet, DROP_INTERFACE_DOWN, this, interface);
        }
    }
  else
    {
      if (outInterface->IsUp ())
        {
          NS_LOG_LOGIC ("Send to destination " << ipHeader.GetDestination ());

          if (fragments.size () != 0)
            {
              std::ostringstream oss;

              for (std::list<Ipv6ExtensionFragment::Ipv6PayloadHeaderPair>::const_iterator it = fragments.begin (); it != fragments.end (); it++)
                {
                  CallTxTrace (it->second, it->first, this, interface);
                  outInterface->Send (it->first, it->second, ipHeader.GetDestination ());
                }
            }
          else
            {
              CallTxTrace (ipHeader, packet, this, interface);
              outInterface->Send (packet, ipHeader, ipHeader.GetDestination ());
            }
        }
      else
        {
          NS_LOG_LOGIC ("Dropping-- outgoing interface is down: " << ipHeader.GetDestination ());
          m_dropTrace (ipHeader, packet, DROP_INTERFACE_DOWN, this, interface);
        }
    }
}

void Ipv6L3Protocol::IpForward (Ptr<const NetDevice> idev, Ptr<Ipv6Route> rtentry, Ptr<const Packet> p, const Ipv6Header& header)
{
  NS_LOG_FUNCTION (this << rtentry << p << header);
  NS_LOG_LOGIC ("Forwarding logic for node: " << m_node->GetId ());

  // Drop RFC 3849 packets: 2001:db8::/32
  if (header.GetDestination().IsDocumentation ())
    {
      NS_LOG_WARN ("Received a packet for 2001:db8::/32 (documentation class).  Drop.");
      m_dropTrace (header, p, DROP_ROUTE_ERROR, this, 0);
      return;
    }

  // Forwarding
  Ipv6Header ipHeader = header;
  Ptr<Packet> packet = p->Copy ();
  ipHeader.SetHopLimit (ipHeader.GetHopLimit () - 1);

  if (ipHeader.GetSource ().IsLinkLocal ())
    {
      /* no forward for link-local address */
      return;
    }

  if (ipHeader.GetHopLimit () == 0)
    {
      NS_LOG_WARN ("TTL exceeded.  Drop.");
      m_dropTrace (ipHeader, packet, DROP_TTL_EXPIRED, this, 0);
      // Do not reply to multicast IPv6 address
      if (ipHeader.GetDestination ().IsMulticast () == false)
        {
          packet->AddHeader (ipHeader);
          GetIcmpv6 ()->SendErrorTimeExceeded (packet, ipHeader.GetSource (), Icmpv6Header::ICMPV6_HOPLIMIT);
        }
      return;
    }

  /* ICMPv6 Redirect */

  /* if we forward to a machine on the same network as the source,
   * we send him an ICMPv6 redirect message to notify him that a short route
   * exists.
   */

  /* Theoretically we should also check if the redirect target is on the same network
   * as the source node. On the other hand, we are sure that the router we're redirecting to
   * used a link-local address. As a consequence, they MUST be on the same network, the link-local net.
   */

  if (m_sendIcmpv6Redirect && (rtentry->GetOutputDevice ()==idev))
    {
      NS_LOG_LOGIC ("ICMPv6 redirect!");
      Ptr<Icmpv6L4Protocol> icmpv6 = GetIcmpv6 ();
      Address hardwareTarget;
      Ipv6Address dst = header.GetDestination ();
      Ipv6Address src = header.GetSource ();
      Ipv6Address target = rtentry->GetGateway ();
      Ptr<Packet> copy = p->Copy ();

      if (target.IsAny ())
        {
          target = dst;
        }

      copy->AddHeader (header);
      Ipv6Address linkLocal = GetInterface (GetInterfaceForDevice (rtentry->GetOutputDevice ()))->GetLinkLocalAddress ().GetAddress ();

      if (icmpv6->Lookup (target, rtentry->GetOutputDevice (), 0, &hardwareTarget))
        {
          icmpv6->SendRedirection (copy, linkLocal, src, target, dst, hardwareTarget);
        }
      else
        {
          icmpv6->SendRedirection (copy, linkLocal, src, target, dst, Address ());
        }
    }
  // in case the packet still has a priority tag attached, remove it
  SocketPriorityTag priorityTag;
  packet->RemovePacketTag (priorityTag);
  int32_t interface = GetInterfaceForDevice (rtentry->GetOutputDevice ());
  m_unicastForwardTrace (ipHeader, packet, interface);
  SendRealOut (rtentry, packet, ipHeader);
}

void Ipv6L3Protocol::IpMulticastForward (Ptr<const NetDevice> idev, Ptr<Ipv6MulticastRoute> mrtentry, Ptr<const Packet> p, const Ipv6Header& header)
{
  NS_LOG_FUNCTION (this << mrtentry << p << header);
  NS_LOG_LOGIC ("Multicast forwarding logic for node: " << m_node->GetId ());

  std::map<uint32_t, uint32_t> ttlMap = mrtentry->GetOutputTtlMap ();
  std::map<uint32_t, uint32_t>::iterator mapIter;

  for (mapIter = ttlMap.begin (); mapIter != ttlMap.end (); mapIter++)
    {
      uint32_t interfaceId = mapIter->first;
      //uint32_t outputTtl = mapIter->second;  // Unused for now
      Ptr<Packet> packet = p->Copy ();
      Ipv6Header h = header;
      h.SetHopLimit (header.GetHopLimit () - 1);
      if (h.GetHopLimit () == 0)
        {
          NS_LOG_WARN ("TTL exceeded.  Drop.");
          m_dropTrace (header, packet, DROP_TTL_EXPIRED, this, interfaceId);
          return;
        }
      NS_LOG_LOGIC ("Forward multicast via interface " << interfaceId);
      Ptr<Ipv6Route> rtentry = Create<Ipv6Route> ();
      rtentry->SetSource (h.GetSource ());
      rtentry->SetDestination (h.GetDestination ());
      rtentry->SetGateway (Ipv6Address::GetAny ());
      rtentry->SetOutputDevice (GetNetDevice (interfaceId));
      SendRealOut (rtentry, packet, h);
      continue;
    }
}

void Ipv6L3Protocol::LocalDeliver (Ptr<const Packet> packet, Ipv6Header const& ip, uint32_t iif)
{
  NS_LOG_FUNCTION (this << packet << ip << iif);
  Ptr<Packet> p = packet->Copy ();
  Ptr<IpL4Protocol> protocol = 0;
  Ptr<Ipv6ExtensionDemux> ipv6ExtensionDemux = m_node->GetObject<Ipv6ExtensionDemux> ();
  Ptr<Ipv6Extension> ipv6Extension = 0;
  Ipv6Address src = ip.GetSource ();
  Ipv6Address dst = ip.GetDestination ();
  uint8_t nextHeader = ip.GetNextHeader ();
  uint8_t nextHeaderPosition = 0;
  bool isDropped = false;
  bool stopProcessing = false;
  DropReason dropReason;

  // check for a malformed hop-by-hop extension
  // this is a common case when forging IPv6 raw packets
  if (nextHeader == Ipv6Header::IPV6_EXT_HOP_BY_HOP)
    {
      uint8_t buf;
      p->CopyData (&buf, 1);
      if (buf == Ipv6Header::IPV6_EXT_HOP_BY_HOP)
        {
          NS_LOG_WARN("Double Ipv6Header::IPV6_EXT_HOP_BY_HOP in packet, dropping packet");
          return;
        }
    }

  /* process all the extensions found and the layer 4 protocol */
  do
    {
      /* it return 0 for non-extension (i.e. layer 4 protocol) */
      ipv6Extension = ipv6ExtensionDemux->GetExtension (nextHeader);

      if (ipv6Extension)
        {
          uint8_t nextHeaderStep = 0;
          uint8_t curHeader = nextHeader;
          nextHeaderStep = ipv6Extension->Process (p, nextHeaderPosition, ip, dst, &nextHeader, stopProcessing, isDropped, dropReason);
          nextHeaderPosition += nextHeaderStep;

          if (isDropped)
            {
              m_dropTrace (ip, packet, dropReason, this, iif);
            }

          if (stopProcessing)
            {
              return;
            }
          NS_ASSERT_MSG (nextHeaderStep != 0 || curHeader == Ipv6Header::IPV6_EXT_FRAGMENTATION,
                         "Zero-size IPv6 Option Header, aborting" << *packet );
        }
      else
        {
          protocol = GetProtocol (nextHeader, iif);

          if (!protocol)
            {
              NS_LOG_LOGIC ("Unknown Next Header. Drop!");

              // For ICMPv6 Error packets
              Ptr<Packet> malformedPacket  = packet->Copy ();
              malformedPacket->AddHeader (ip);

              if (nextHeaderPosition == 0)
                {
                  GetIcmpv6 ()->SendErrorParameterError (malformedPacket, dst, Icmpv6Header::ICMPV6_UNKNOWN_NEXT_HEADER, 40);
                }
              else
                {
                  GetIcmpv6 ()->SendErrorParameterError (malformedPacket, dst, Icmpv6Header::ICMPV6_UNKNOWN_NEXT_HEADER, ip.GetSerializedSize () + nextHeaderPosition);
                }
              m_dropTrace (ip, p, DROP_UNKNOWN_PROTOCOL, this, iif);
              break;
            }
          else
            {
              p->RemoveAtStart (nextHeaderPosition);
              /* protocol->Receive (p, src, dst, incomingInterface); */

              /* L4 protocol */
              Ptr<Packet> copy = p->Copy ();

              m_localDeliverTrace (ip, p, iif);

              enum IpL4Protocol::RxStatus status = protocol->Receive (p, ip, GetInterface (iif));

              switch (status)
                {
                case IpL4Protocol::RX_OK:
                  break;
                case IpL4Protocol::RX_CSUM_FAILED:
                  break;
                case IpL4Protocol::RX_ENDPOINT_CLOSED:
                  break;
                case IpL4Protocol::RX_ENDPOINT_UNREACH:
                  if (ip.GetDestination ().IsMulticast ())
                    {
                      /* do not rely on multicast address */
                      break;
                    }

                  copy->AddHeader (ip);
                  GetIcmpv6 ()->SendErrorDestinationUnreachable (copy, ip.GetSource (), Icmpv6Header::ICMPV6_PORT_UNREACHABLE);
                }
            }
        }
    }
  while (ipv6Extension);
}

void Ipv6L3Protocol::RouteInputError (Ptr<const Packet> p, const Ipv6Header& ipHeader, Socket::SocketErrno sockErrno)
{
  NS_LOG_FUNCTION (this << p << ipHeader << sockErrno);
  NS_LOG_LOGIC ("Route input failure-- dropping packet to " << ipHeader << " with errno " << sockErrno);

  m_dropTrace (ipHeader, p, DROP_ROUTE_ERROR, this, 0);

  if (!ipHeader.GetDestination ().IsMulticast ())
    {
      Ptr<Packet> packet = p->Copy ();
      packet->AddHeader (ipHeader);
      GetIcmpv6 ()->SendErrorDestinationUnreachable (packet, ipHeader.GetSource (), Icmpv6Header::ICMPV6_NO_ROUTE);
    }
}

Ipv6Header Ipv6L3Protocol::BuildHeader (Ipv6Address src, Ipv6Address dst, uint8_t protocol, uint16_t payloadSize, uint8_t ttl, uint8_t tclass)
{
  NS_LOG_FUNCTION (this << src << dst << (uint32_t)protocol << (uint32_t)payloadSize << (uint32_t)ttl << (uint32_t)tclass);
  Ipv6Header hdr;

  hdr.SetSource (src);
  hdr.SetDestination (dst);
  hdr.SetNextHeader (protocol);
  hdr.SetPayloadLength (payloadSize);
  hdr.SetHopLimit (ttl);
  hdr.SetTrafficClass (tclass);
  return hdr;
}

void Ipv6L3Protocol::RegisterExtensions ()
{
  Ptr<Ipv6ExtensionDemux> ipv6ExtensionDemux = CreateObject<Ipv6ExtensionDemux> ();
  ipv6ExtensionDemux->SetNode (m_node);

  Ptr<Ipv6ExtensionHopByHop> hopbyhopExtension = CreateObject<Ipv6ExtensionHopByHop> ();
  hopbyhopExtension->SetNode (m_node);
  Ptr<Ipv6ExtensionDestination> destinationExtension = CreateObject<Ipv6ExtensionDestination> ();
  destinationExtension->SetNode (m_node);
  Ptr<Ipv6ExtensionFragment> fragmentExtension = CreateObject<Ipv6ExtensionFragment> ();
  fragmentExtension->SetNode (m_node);
  Ptr<Ipv6ExtensionRouting> routingExtension = CreateObject<Ipv6ExtensionRouting> ();
  routingExtension->SetNode (m_node);
  // Ptr<Ipv6ExtensionESP> espExtension = CreateObject<Ipv6ExtensionESP> ();
  // Ptr<Ipv6ExtensionAH> ahExtension = CreateObject<Ipv6ExtensionAH> ();

  ipv6ExtensionDemux->Insert (hopbyhopExtension);
  ipv6ExtensionDemux->Insert (destinationExtension);
  ipv6ExtensionDemux->Insert (fragmentExtension);
  ipv6ExtensionDemux->Insert (routingExtension);
  // ipv6ExtensionDemux->Insert (espExtension);
  // ipv6ExtensionDemux->Insert (ahExtension);

  Ptr<Ipv6ExtensionRoutingDemux> routingExtensionDemux = CreateObject<Ipv6ExtensionRoutingDemux> ();
  routingExtensionDemux->SetNode (m_node);
  Ptr<Ipv6ExtensionLooseRouting> looseRoutingExtension = CreateObject<Ipv6ExtensionLooseRouting> ();
  looseRoutingExtension->SetNode (m_node);
  routingExtensionDemux->Insert (looseRoutingExtension);

  m_node->AggregateObject (routingExtensionDemux);
  m_node->AggregateObject (ipv6ExtensionDemux);
}

void Ipv6L3Protocol::RegisterOptions ()
{
  Ptr<Ipv6OptionDemux> ipv6OptionDemux = CreateObject<Ipv6OptionDemux> ();
  ipv6OptionDemux->SetNode (m_node);

  Ptr<Ipv6OptionPad1> pad1Option = CreateObject<Ipv6OptionPad1> ();
  pad1Option->SetNode (m_node);
  Ptr<Ipv6OptionPadn> padnOption = CreateObject<Ipv6OptionPadn> ();
  padnOption->SetNode (m_node);
  Ptr<Ipv6OptionJumbogram> jumbogramOption = CreateObject<Ipv6OptionJumbogram> ();
  jumbogramOption->SetNode (m_node);
  Ptr<Ipv6OptionRouterAlert> routerAlertOption = CreateObject<Ipv6OptionRouterAlert> ();
  routerAlertOption->SetNode (m_node);

  ipv6OptionDemux->Insert (pad1Option);
  ipv6OptionDemux->Insert (padnOption);
  ipv6OptionDemux->Insert (jumbogramOption);
  ipv6OptionDemux->Insert (routerAlertOption);

  m_node->AggregateObject (ipv6OptionDemux);
}

void Ipv6L3Protocol::ReportDrop (Ipv6Header ipHeader, Ptr<Packet> p, DropReason dropReason)
{
  m_dropTrace (ipHeader, p, dropReason, this, 0);
}

void Ipv6L3Protocol::AddMulticastAddress (Ipv6Address address, uint32_t interface)
{
  NS_LOG_FUNCTION (address << interface);

  if (!address.IsMulticast ())
    {
      NS_LOG_WARN ("Not adding a non-multicast address " << address);
      return;
    }

  Ipv6RegisteredMulticastAddressKey_t key = std::make_pair (address, interface);
  m_multicastAddresses[key]++;
}

void Ipv6L3Protocol::AddMulticastAddress (Ipv6Address address)
{
  NS_LOG_FUNCTION (address);

  if (!address.IsMulticast ())
    {
      NS_LOG_WARN ("Not adding a non-multicast address " << address);
      return;
    }

  m_multicastAddressesNoInterface[address]++;
}

void Ipv6L3Protocol::RemoveMulticastAddress (Ipv6Address address, uint32_t interface)
{
  NS_LOG_FUNCTION (address << interface);

  Ipv6RegisteredMulticastAddressKey_t key = std::make_pair (address, interface);

  m_multicastAddresses[key]--;
  if (m_multicastAddresses[key] == 0)
    {
      m_multicastAddresses.erase (key);
    }
}

void Ipv6L3Protocol::RemoveMulticastAddress (Ipv6Address address)
{
  NS_LOG_FUNCTION (address);

  m_multicastAddressesNoInterface[address]--;
  if (m_multicastAddressesNoInterface[address] == 0)
    {
      m_multicastAddressesNoInterface.erase (address);
    }
}

bool Ipv6L3Protocol::IsRegisteredMulticastAddress (Ipv6Address address, uint32_t interface) const
{
  NS_LOG_FUNCTION (address << interface);

  Ipv6RegisteredMulticastAddressKey_t key = std::make_pair (address, interface);
  Ipv6RegisteredMulticastAddressCIter_t iter = m_multicastAddresses.find (key);

  if (iter == m_multicastAddresses.end ())
    {
      return false;
    }
  return true;
}

bool Ipv6L3Protocol::IsRegisteredMulticastAddress (Ipv6Address address) const
{
  NS_LOG_FUNCTION (address);

  Ipv6RegisteredMulticastAddressNoInterfaceCIter_t iter = m_multicastAddressesNoInterface.find (address);

  if (iter == m_multicastAddressesNoInterface.end ())
    {
      return false;
    }
  return true;
}

bool Ipv6L3Protocol::ReachabilityHint (uint32_t ipInterfaceIndex, Ipv6Address address)
{
  if (ipInterfaceIndex >= m_interfaces.size ())
    {
      return false;
    }

  Ptr<NdiscCache> ndiscCache = m_interfaces[ipInterfaceIndex]->GetNdiscCache ();
  if (!ndiscCache)
    {
      return false;
    }

  NdiscCache::Entry* entry = ndiscCache->Lookup (address);
  if (!entry || entry->IsIncomplete ())
    {
      return false;
    }


  if (entry->IsReachable ())
    {
      entry->UpdateReachableTimer ();
    }
  else if (entry->IsPermanent ())
    {
      return true;
    }
  else if (entry->IsProbe ())
    {
      // we just confirm the entry's MAC address to get the waiting packets (if any)
      std::list<NdiscCache::Ipv6PayloadHeaderPair> waiting = entry->MarkReachable (entry->GetMacAddress ());
      for (std::list<NdiscCache::Ipv6PayloadHeaderPair>::const_iterator it = waiting.begin (); it != waiting.end (); it++)
        {
          ndiscCache->GetInterface ()->Send (it->first, it->second, it->second.GetSource ());
        }
      entry->ClearWaitingPacket ();
      entry->StartReachableTimer ();
    }
  else // STALE OR DELAY
    {
      entry->MarkReachable ();
      entry->StartReachableTimer ();
    }


  return true;
}

} /* namespace ns3 */

