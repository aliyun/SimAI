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
 *         David Gross <gdavid.devel@gmail.com>
 *         Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 *         Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/log.h"
#include "ns3/assert.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "ns3/boolean.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv6-route.h"
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/integer.h"

#include "ipv6-l3-protocol.h"
#include "ipv6-interface.h"
#include "icmpv6-l4-protocol.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Icmpv6L4Protocol");

NS_OBJECT_ENSURE_REGISTERED (Icmpv6L4Protocol);

const uint8_t Icmpv6L4Protocol::PROT_NUMBER = 58;

//const uint8_t Icmpv6L4Protocol::MAX_INITIAL_RTR_ADVERT_INTERVAL = 16; // max initial RA initial interval.
//const uint8_t Icmpv6L4Protocol::MAX_INITIAL_RTR_ADVERTISEMENTS = 3;   // max initial RA transmission.
//const uint8_t Icmpv6L4Protocol::MAX_FINAL_RTR_ADVERTISEMENTS = 3;     // max final RA transmission.
//const uint8_t Icmpv6L4Protocol::MIN_DELAY_BETWEEN_RAS = 3;            // min delay between RA.
//const uint32_t Icmpv6L4Protocol::MAX_RA_DELAY_TIME = 500;             // millisecond - max delay between RA.

//const uint8_t Icmpv6L4Protocol::MAX_RTR_SOLICITATION_DELAY = 1;       // max RS delay.
//const uint8_t Icmpv6L4Protocol::RTR_SOLICITATION_INTERVAL = 4;        // RS interval.
//const uint8_t Icmpv6L4Protocol::MAX_RTR_SOLICITATIONS = 3;            // max RS transmission.

//const uint8_t Icmpv6L4Protocol::MAX_ANYCAST_DELAY_TIME = 1;           // max anycast delay.
//const uint8_t Icmpv6L4Protocol::MAX_NEIGHBOR_ADVERTISEMENT = 3;       // max NA transmission.

//const double Icmpv6L4Protocol::MIN_RANDOM_FACTOR = 0.5;               // min random factor.
//const double Icmpv6L4Protocol::MAX_RANDOM_FACTOR = 1.5;               // max random factor.

TypeId Icmpv6L4Protocol::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Icmpv6L4Protocol")
    .SetParent<IpL4Protocol> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv6L4Protocol> ()
    .AddAttribute ("DAD", "Always do DAD check.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&Icmpv6L4Protocol::m_alwaysDad),
                   MakeBooleanChecker ())
    .AddAttribute ("SolicitationJitter", "The jitter in ms a node is allowed to wait before sending any solicitation. Some jitter aims to prevent collisions. By default, the model will wait for a duration in ms defined by a uniform random-variable between 0 and SolicitationJitter",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=10.0]"),
                   MakePointerAccessor (&Icmpv6L4Protocol::m_solicitationJitter),
                   MakePointerChecker<RandomVariableStream> ())
    .AddAttribute ("MaxMulticastSolicit", "Neighbor Discovery node constants: max multicast solicitations.",
                   IntegerValue (3),
                   MakeIntegerAccessor (&Icmpv6L4Protocol::m_maxMulticastSolicit),
                   MakeIntegerChecker<uint8_t> ())
    .AddAttribute ("MaxUnicastSolicit", "Neighbor Discovery node constants: max unicast solicitations.",
                   IntegerValue (3),
                   MakeIntegerAccessor (&Icmpv6L4Protocol::m_maxUnicastSolicit),
                   MakeIntegerChecker<uint8_t> ())
    .AddAttribute ("ReachableTime", "Neighbor Discovery node constants: reachable time.",
                   TimeValue (Seconds (30)),
                   MakeTimeAccessor (&Icmpv6L4Protocol::m_reachableTime),
                   MakeTimeChecker ())
    .AddAttribute ("RetransmissionTime", "Neighbor Discovery node constants: retransmission timer.",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&Icmpv6L4Protocol::m_retransmissionTime),
                   MakeTimeChecker ())
    .AddAttribute ("DelayFirstProbe", "Neighbor Discovery node constants: delay for the first probe.",
                   TimeValue (Seconds (5)),
                   MakeTimeAccessor (&Icmpv6L4Protocol::m_delayFirstProbe),
                   MakeTimeChecker ())
    ;
  return tid;
}

TypeId Icmpv6L4Protocol::GetInstanceTypeId () const
{
  NS_LOG_FUNCTION (this);
  return Icmpv6L4Protocol::GetTypeId ();
}

Icmpv6L4Protocol::Icmpv6L4Protocol ()
  : m_node (0)
{
  NS_LOG_FUNCTION (this);
}

Icmpv6L4Protocol::~Icmpv6L4Protocol ()
{
  NS_LOG_FUNCTION (this);
}

void Icmpv6L4Protocol::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  for (CacheList::const_iterator it = m_cacheList.begin (); it != m_cacheList.end (); it++)
    {
      Ptr<NdiscCache> cache = *it;
      cache->Dispose ();
      cache = 0;
    }
  m_cacheList.clear ();
  m_downTarget.Nullify ();

  m_node = 0;
  IpL4Protocol::DoDispose ();
}

int64_t Icmpv6L4Protocol::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_solicitationJitter->SetStream (stream);
  return 1;
}

void Icmpv6L4Protocol::NotifyNewAggregate ()
{
  NS_LOG_FUNCTION (this);
  if (m_node == 0)
    {
      Ptr<Node> node = this->GetObject<Node> ();
      if (node != 0)
        {
          Ptr<Ipv6> ipv6 = this->GetObject<Ipv6> ();
          if (ipv6 != 0 && m_downTarget.IsNull ())
            {
              SetNode (node);
              ipv6->Insert (this);
              SetDownTarget6 (MakeCallback (&Ipv6::Send, ipv6));
            }
        }
    }
  IpL4Protocol::NotifyNewAggregate ();
}

void Icmpv6L4Protocol::SetNode (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this << node);
  m_node = node;
}

Ptr<Node> Icmpv6L4Protocol::GetNode ()
{
  NS_LOG_FUNCTION (this);
  return m_node;
}

uint16_t Icmpv6L4Protocol::GetStaticProtocolNumber ()
{
  NS_LOG_FUNCTION_NOARGS ();
  return PROT_NUMBER;
}

int Icmpv6L4Protocol::GetProtocolNumber () const
{
  NS_LOG_FUNCTION (this);
  return PROT_NUMBER;
}

int Icmpv6L4Protocol::GetVersion () const
{
  NS_LOG_FUNCTION (this);
  return 1;
}

bool Icmpv6L4Protocol::IsAlwaysDad () const
{
  NS_LOG_FUNCTION (this);
  return m_alwaysDad;
}

void Icmpv6L4Protocol::DoDAD (Ipv6Address target, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << target << interface);
  Ipv6Address addr;
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();

  NS_ASSERT (ipv6);

  if (!m_alwaysDad)
    {
      return;
    }

  /** \todo disable multicast loopback to prevent NS probing to be received by the sender */
  
  NdiscCache::Ipv6PayloadHeaderPair p = ForgeNS ("::",Ipv6Address::MakeSolicitedAddress (target), target, interface->GetDevice ()->GetAddress ());

  /* update last packet UID */
  interface->SetNsDadUid (target, p.first->GetUid ());
  Simulator::Schedule (Time (MilliSeconds (m_solicitationJitter->GetValue ())), &Ipv6Interface::Send, interface, p.first, p.second, Ipv6Address::MakeSolicitedAddress (target));
}

enum IpL4Protocol::RxStatus Icmpv6L4Protocol::Receive (Ptr<Packet> packet, Ipv4Header const &header,  Ptr<Ipv4Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << header);
  return IpL4Protocol::RX_ENDPOINT_UNREACH;
}

enum IpL4Protocol::RxStatus Icmpv6L4Protocol::Receive (Ptr<Packet> packet, Ipv6Header const &header, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << header.GetSource () << header.GetDestination () << interface);
  Ptr<Packet> p = packet->Copy ();
  Ptr<Ipv6> ipv6 = m_node->GetObject<Ipv6> ();

  /* very ugly! try to find something better in the future */
  uint8_t type;
  p->CopyData (&type, sizeof(type));

  switch (type)
    {
    case Icmpv6Header::ICMPV6_ND_ROUTER_SOLICITATION:
      if (ipv6->IsForwarding (ipv6->GetInterfaceForDevice (interface->GetDevice ())))
        {
          HandleRS (p, header.GetSource (), header.GetDestination (), interface);
        }
      break;
    case Icmpv6Header::ICMPV6_ND_ROUTER_ADVERTISEMENT:
      if (!ipv6->IsForwarding (ipv6->GetInterfaceForDevice (interface->GetDevice ())))
        {
          HandleRA (p, header.GetSource (), header.GetDestination (), interface);
        }
      break;
    case Icmpv6Header::ICMPV6_ND_NEIGHBOR_SOLICITATION:
      HandleNS (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ND_NEIGHBOR_ADVERTISEMENT:
      HandleNA (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ND_REDIRECTION:
      HandleRedirection (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ECHO_REQUEST:
      HandleEchoRequest (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ECHO_REPLY:
      // EchoReply does not contain any info about L4
      // so we can not forward it up.
      /// \todo implement request / reply consistency check.
      break;
    case Icmpv6Header::ICMPV6_ERROR_DESTINATION_UNREACHABLE:
      HandleDestinationUnreachable (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ERROR_PACKET_TOO_BIG:
      HandlePacketTooBig (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ERROR_TIME_EXCEEDED:
      HandleTimeExceeded (p, header.GetSource (), header.GetDestination (), interface);
      break;
    case Icmpv6Header::ICMPV6_ERROR_PARAMETER_ERROR:
      HandleParameterError (p, header.GetSource (), header.GetDestination (), interface);
      break;
    default:
      NS_LOG_LOGIC ("Unknown ICMPv6 message type=" << type);
      break;
    }

  return IpL4Protocol::RX_OK;
}

void Icmpv6L4Protocol::Forward (Ipv6Address source, Icmpv6Header icmp,
                                uint32_t info, Ipv6Header ipHeader,
                                const uint8_t payload[8])
{
  NS_LOG_FUNCTION (this << source << icmp << info << ipHeader << payload);

  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();

  /// \todo assuming the ICMP is carrying a extensionless IP packet

  uint8_t nextHeader = ipHeader.GetNextHeader ();

  if (nextHeader != Icmpv6L4Protocol::PROT_NUMBER)
    {
      Ptr<IpL4Protocol> l4 = ipv6->GetProtocol (nextHeader);
      if (l4 != 0)
        {
          l4->ReceiveIcmp (source, ipHeader.GetHopLimit (), icmp.GetType (), icmp.GetCode (),
                           info, ipHeader.GetSource (), ipHeader.GetDestination (), payload);
        }
    }
}

void Icmpv6L4Protocol::HandleEchoRequest (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  Icmpv6Echo request;
  uint8_t* buf = new uint8_t[packet->GetSize ()];

  packet->RemoveHeader (request);
  /* XXX IPv6 extension: obtain a fresh copy of data otherwise it crash... */
  packet->CopyData (buf, packet->GetSize ());
  Ptr<Packet> p = Create<Packet> (buf, packet->GetSize ());

  /* if we send message from ff02::* (link-local multicast), we use our link-local address */
  SendEchoReply (dst.IsMulticast () ? interface->GetLinkLocalAddress ().GetAddress () : dst, src, request.GetId (), request.GetSeq (), p);
  delete[] buf;
}

void Icmpv6L4Protocol::HandleRA (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  Ptr<Packet> p = packet->Copy ();
  Icmpv6RA raHeader;
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  Icmpv6OptionPrefixInformation prefixHdr;
  Icmpv6OptionMtu mtuHdr;
  Icmpv6OptionLinkLayerAddress llaHdr;
  bool next = true;
  bool hasLla = false;
  bool hasMtu = false;
  Ipv6Address defaultRouter = Ipv6Address::GetZero ();

  p->RemoveHeader (raHeader);

  if (raHeader.GetLifeTime())
    {
      defaultRouter = src;
    }

  while (next == true)
    {
      uint8_t type = 0;
      p->CopyData (&type, sizeof(type));

      switch (type)
        {
        case Icmpv6Header::ICMPV6_OPT_PREFIX:
          p->RemoveHeader (prefixHdr);
          ipv6->AddAutoconfiguredAddress (ipv6->GetInterfaceForDevice (interface->GetDevice ()), prefixHdr.GetPrefix (), prefixHdr.GetPrefixLength (),
                                          prefixHdr.GetFlags (), prefixHdr.GetValidTime (), prefixHdr.GetPreferredTime (), defaultRouter);
          break;
        case Icmpv6Header::ICMPV6_OPT_MTU:
          /* take in account the first MTU option */
          if (!hasMtu)
            {
              p->RemoveHeader (mtuHdr);
              hasMtu = true;
              /** \todo case of multiple prefix on single interface */
              /* interface->GetDevice ()->SetMtu (m.GetMtu ()); */
            }
          break;
        case Icmpv6Header::ICMPV6_OPT_LINK_LAYER_SOURCE:
          /* take in account the first LLA option */
          if (!hasLla)
            {
              p->RemoveHeader (llaHdr);
              ReceiveLLA (llaHdr, src, dst, interface);
              hasLla = true;
            }
          break;
        default:
          /* unknown option, quit */
          next = false;
        }
    }
}

void Icmpv6L4Protocol::ReceiveLLA (Icmpv6OptionLinkLayerAddress lla, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << lla << src << dst << interface);
  Address hardwareAddress;
  NdiscCache::Entry* entry = 0;
  Ptr<NdiscCache> cache = FindCache (interface->GetDevice ());

  /* check if we have this address in our cache */
  entry = cache->Lookup (src);

  if (!entry)
    {
      entry = cache->Add (src);
      entry->SetRouter (true);
      entry->SetMacAddress (lla.GetAddress ());
      entry->MarkReachable ();
      entry->StartReachableTimer ();
    }
  else
    {
      std::list<NdiscCache::Ipv6PayloadHeaderPair> waiting;
      if (entry->IsIncomplete ())
        {
          entry->StopNudTimer ();
          // mark it to reachable
          waiting = entry->MarkReachable (lla.GetAddress ());
          entry->StartReachableTimer ();
          // send out waiting packet
          for (std::list<NdiscCache::Ipv6PayloadHeaderPair>::const_iterator it = waiting.begin (); it != waiting.end (); it++)
            {
              cache->GetInterface ()->Send (it->first, it->second, src);
            }
          entry->ClearWaitingPacket ();
        }
      else
        {
          if (entry->GetMacAddress () != lla.GetAddress ())
            {
              entry->SetMacAddress (lla.GetAddress ());
              entry->MarkStale ();
              entry->SetRouter (true);
            }
          else
            {
              if (!entry->IsReachable () || !entry->IsPermanent ())
                {
                  entry->StopNudTimer ();
                  waiting = entry->MarkReachable (lla.GetAddress ());
                  if (entry->IsProbe ())
                    {
                      for (std::list<NdiscCache::Ipv6PayloadHeaderPair>::const_iterator it = waiting.begin (); it != waiting.end (); it++)
                        {
                          cache->GetInterface ()->Send (it->first, it->second, src);
                        }
                    }
                  if (!entry->IsPermanent ())
                    {
                      entry->StartReachableTimer ();
                    }
                }
            }
        }
    }
}

void Icmpv6L4Protocol::HandleRS (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  Icmpv6RS rsHeader;
  packet->RemoveHeader (rsHeader);
  Address hardwareAddress;
  Icmpv6OptionLinkLayerAddress lla (1);
  NdiscCache::Entry* entry = 0;
  Ptr<NdiscCache> cache = FindCache (interface->GetDevice ());

  if (src != Ipv6Address::GetAny ())
    {
      /* XXX search all options following the RS header */
      /* test if the next option is SourceLinkLayerAddress */
      uint8_t type;
      packet->CopyData (&type, sizeof(type));

      if (type != Icmpv6Header::ICMPV6_OPT_LINK_LAYER_SOURCE)
        {
          return;
        }
      packet->RemoveHeader (lla);
      NS_LOG_LOGIC ("Cache updated by RS");

      entry = cache->Lookup (src);
      if (!entry)
        {
          entry = cache->Add (src);
          entry->SetRouter (false);
          entry->MarkStale (lla.GetAddress ());
        }
      else if (entry->GetMacAddress () != lla.GetAddress ())
        {
          entry->MarkStale (lla.GetAddress ());
        }
    }
}

void Icmpv6L4Protocol::HandleNS (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  Icmpv6NS nsHeader ("::");
  Ipv6InterfaceAddress ifaddr;
  uint32_t nb = interface->GetNAddresses ();
  uint32_t i = 0;
  bool found = false;

  packet->RemoveHeader (nsHeader);

  Ipv6Address target = nsHeader.GetIpv6Target ();

  for (i = 0; i < nb; i++)
    {
      ifaddr = interface->GetAddress (i);

      if (ifaddr.GetAddress () == target)
        {
          found = true;
          break;
        }
    }

  if (!found)
    {
      NS_LOG_LOGIC ("Not a NS for us");
      return;
    }

  if (packet->GetUid () == ifaddr.GetNsDadUid ())
    {
      /* don't process our own DAD probe */
      NS_LOG_LOGIC ("Hey we receive our DAD probe!");
      return;
    }

  NdiscCache::Entry* entry = 0;
  Ptr<NdiscCache> cache = FindCache (interface->GetDevice ());
  uint8_t flags = 0;

  /* search all options following the NS header */
  Icmpv6OptionLinkLayerAddress sllaoHdr (true);

  bool next = true;
  bool hasSllao = false;

  while (next == true)
    {
      uint8_t type;
      packet->CopyData (&type, sizeof (type));

      switch (type)
        {
          case Icmpv6Header::ICMPV6_OPT_LINK_LAYER_SOURCE:
            if (!hasSllao)
              {
                packet->RemoveHeader (sllaoHdr);
                hasSllao = true;
              }
            break;
          default:
            /* unknow option, quit */
            next = false;
        }
      if (packet->GetSize () == 0)
        {
          next = false;
        }
    }

  Address replyMacAddress;

  if (src != Ipv6Address::GetAny ())
    {
      entry = cache->Lookup (src);
      if (!entry)
        {
          if (!hasSllao)
            {
              NS_LOG_LOGIC ("Icmpv6L4Protocol::HandleNS: NS without SLLAO and we do not have a NCE, discarding.");
              return;
            }
          entry = cache->Add (src);
          entry->SetRouter (false);
          entry->MarkStale (sllaoHdr.GetAddress ());
          replyMacAddress = sllaoHdr.GetAddress ();
        }
      else if (hasSllao && (entry->GetMacAddress () != sllaoHdr.GetAddress ()))
        {
          entry->MarkStale (sllaoHdr.GetAddress ());
          replyMacAddress = sllaoHdr.GetAddress ();
        }
      else
        {
          replyMacAddress = entry->GetMacAddress ();
        }

      flags = 3; /* S + O flags */
    }
  else
    {
      /* it's a DAD */
      flags = 1; /* O flag */
      replyMacAddress = interface->GetDevice ()->GetMulticast (dst);
    }

  /* send a NA to src */
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();

  if (ipv6->IsForwarding (ipv6->GetInterfaceForDevice (interface->GetDevice ())))
    {
      flags += 4; /* R flag */
    }

  Address hardwareAddress = interface->GetDevice ()->GetAddress ();
  NdiscCache::Ipv6PayloadHeaderPair p = ForgeNA (target.IsLinkLocal () ? interface->GetLinkLocalAddress ().GetAddress () : ifaddr.GetAddress (),
                                                 src.IsAny () ? dst : src, // DAD replies must go to the multicast group it was sent to.
                                                 &hardwareAddress,
                                                 flags );

  // We must bypass the IPv6 layer, as a NA must be sent regardless of the NCE status (and not change it beyond what we did already).
  Ptr<Packet> pkt = p.first;
  pkt->AddHeader (p.second);
  interface->GetDevice ()->Send (pkt, replyMacAddress, Ipv6L3Protocol::PROT_NUMBER);
}

NdiscCache::Ipv6PayloadHeaderPair Icmpv6L4Protocol::ForgeRS (Ipv6Address src, Ipv6Address dst, Address hardwareAddress)
{
  NS_LOG_FUNCTION (this << src << dst << hardwareAddress);
  Ptr<Packet> p = Create<Packet> ();
  Ipv6Header ipHeader;
  Icmpv6RS rs;

  NS_LOG_LOGIC ("Forge RS (from " << src << " to " << dst << ")");
  // RFC 4861:
  // The link-layer address of the sender MUST NOT be included if the Source Address is the unspecified address.
  // Otherwise, it SHOULD be included on link layers that have addresses.
  if (!src.IsAny ())
    {
      Icmpv6OptionLinkLayerAddress llOption (1, hardwareAddress);  /* we give our mac address in response */
      p->AddHeader (llOption);
    }

  rs.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + rs.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (rs);

  ipHeader.SetSource (src);
  ipHeader.SetDestination (dst);
  ipHeader.SetNextHeader (PROT_NUMBER);
  ipHeader.SetPayloadLength (p->GetSize ());
  ipHeader.SetHopLimit (255);

  return NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader);
}

NdiscCache::Ipv6PayloadHeaderPair Icmpv6L4Protocol::ForgeEchoRequest (Ipv6Address src, Ipv6Address dst, uint16_t id, uint16_t seq, Ptr<Packet> data)
{
  NS_LOG_FUNCTION (this << src << dst << id << seq << data);
  Ptr<Packet> p = data->Copy ();
  Ipv6Header ipHeader;
  Icmpv6Echo req (1);

  req.SetId (id);
  req.SetSeq (seq);

  req.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + req.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (req);

  ipHeader.SetSource (src);
  ipHeader.SetDestination (dst);
  ipHeader.SetNextHeader (PROT_NUMBER);
  ipHeader.SetPayloadLength (p->GetSize ());
  ipHeader.SetHopLimit (255);

  return NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader);
}

void Icmpv6L4Protocol::HandleNA (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  Icmpv6NA naHeader;
  Icmpv6OptionLinkLayerAddress lla (1);

  packet->RemoveHeader (naHeader);
  Ipv6Address target = naHeader.GetIpv6Target ();

  Address hardwareAddress;
  NdiscCache::Entry* entry = 0;
  Ptr<NdiscCache> cache = FindCache (interface->GetDevice ());
  std::list<NdiscCache::Ipv6PayloadHeaderPair> waiting;

  /* check if we have something in our cache */
  entry = cache->Lookup (target);

  if (!entry)
    {
      /* ouch!! we might be victim of a DAD */
      
      Ipv6InterfaceAddress ifaddr;
      bool found = false;
      uint32_t i = 0;
      uint32_t nb = interface->GetNAddresses ();

      for (i = 0; i < nb; i++)
        {
          ifaddr = interface->GetAddress (i);
          if (ifaddr.GetAddress () == target)
            {
              found = true;
              break;
            }
        }

      if (found)
        {
          if (ifaddr.GetState () == Ipv6InterfaceAddress::TENTATIVE || ifaddr.GetState () == Ipv6InterfaceAddress::TENTATIVE_OPTIMISTIC)
            {
              interface->SetState (ifaddr.GetAddress (), Ipv6InterfaceAddress::INVALID);
            }
        }

      /* we have not initiated any communication with the target so... discard the NA */
      return;
    }

  /* XXX search all options following the NA header */
  /* Get LLA */
  uint8_t type;
  packet->CopyData (&type, sizeof(type));

  if (type != Icmpv6Header::ICMPV6_OPT_LINK_LAYER_TARGET)
    {
      return;
    }
  packet->RemoveHeader (lla);

  if (entry->IsIncomplete ())
    {
      /* we receive a NA so stop the retransmission timer */
      entry->StopNudTimer ();

      if (naHeader.GetFlagS ())
        {
          /* mark it to reachable */
          waiting = entry->MarkReachable (lla.GetAddress ());
          entry->StartReachableTimer ();
          /* send out waiting packet */
          for (std::list<NdiscCache::Ipv6PayloadHeaderPair>::const_iterator it = waiting.begin (); it != waiting.end (); it++)
            {
              cache->GetInterface ()->Send (it->first, it->second, src);
            }
          entry->ClearWaitingPacket ();
        }
      else
        {
          entry->MarkStale (lla.GetAddress ());
        }

      if (naHeader.GetFlagR ())
        {
          entry->SetRouter (true);
        }
    }
  else
    {
      /* we receive a NA so stop the probe timer or delay timer if any */
      entry->StopNudTimer ();

      /* if the Flag O is clear and mac address differs from the cache */
      if (!naHeader.GetFlagO () && lla.GetAddress () != entry->GetMacAddress ())
        {
          if (entry->IsReachable ())
            {
              entry->MarkStale ();
            }
          return;
        }
      else
        {
          if ((!naHeader.GetFlagO () && lla.GetAddress () == entry->GetMacAddress ()) || naHeader.GetFlagO ()) /* XXX lake "no target link-layer address option supplied" */
            {
              entry->SetMacAddress (lla.GetAddress ());

              if (naHeader.GetFlagS ())
                {
                  if (!entry->IsReachable () || !entry->IsPermanent ())
                    {
                      if (entry->IsProbe ())
                        {
                          waiting = entry->MarkReachable (lla.GetAddress ());
                          for (std::list<NdiscCache::Ipv6PayloadHeaderPair>::const_iterator it = waiting.begin (); it != waiting.end (); it++)
                            {
                              cache->GetInterface ()->Send (it->first, it->second, src);
                            }
                          entry->ClearWaitingPacket ();
                        }
                      else
                        {
                          entry->MarkReachable (lla.GetAddress ());
                        }
                    }
                  if (!entry->IsPermanent ())
                    {
                      entry->StartReachableTimer ();
                    }
                }
              else if (lla.GetAddress () != entry->GetMacAddress ())
                {
                  entry->MarkStale ();
                }
              entry->SetRouter (naHeader.GetFlagR ());
            }
        }
    }
}

void Icmpv6L4Protocol::HandleRedirection (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << packet << src << dst << interface);
  bool hasLla = false;
  Ptr<Packet> p = packet->Copy ();
  Icmpv6OptionLinkLayerAddress llOptionHeader (0);

  Icmpv6Redirection redirectionHeader;
  p->RemoveHeader (redirectionHeader);

  /* little ugly try to find a better way */
  uint8_t type;
  p->CopyData (&type, sizeof(type));
  if (type == Icmpv6Header::ICMPV6_OPT_LINK_LAYER_TARGET)
    {
      hasLla = true;
      p->RemoveHeader (llOptionHeader);
    }

  Icmpv6OptionRedirected redirectedOptionHeader;
  p->RemoveHeader (redirectedOptionHeader);

  Ipv6Address redirTarget = redirectionHeader.GetTarget ();
  Ipv6Address redirDestination = redirectionHeader.GetDestination ();

  if (hasLla)
    {
      /* update the cache if needed */
      NdiscCache::Entry* entry = 0;
      Ptr<NdiscCache> cache = FindCache (interface->GetDevice ());

      entry = cache->Lookup (redirTarget);
      if (!entry)
        {
          entry = cache->Add (redirTarget);
          /* destination and target different => necessarily a router */
          entry->SetRouter (redirTarget != redirDestination);
          entry->SetMacAddress (llOptionHeader.GetAddress ());
          entry->MarkStale ();
        }
      else
        {
          if (entry->IsIncomplete () || entry->GetMacAddress () != llOptionHeader.GetAddress ())
            {
              /* update entry to STALE */
              if (entry->GetMacAddress () != llOptionHeader.GetAddress ())
                {
                  entry->SetMacAddress (llOptionHeader.GetAddress ());
                  entry->MarkStale ();
                }
            }
          else
            {
              /* stay unchanged */
            }
        }
    }

  /* add redirection in routing table */
  Ptr<Ipv6> ipv6 = m_node->GetObject<Ipv6> ();

  if (redirTarget == redirDestination)
    {
      ipv6->GetRoutingProtocol ()->NotifyAddRoute (redirDestination, Ipv6Prefix (128), Ipv6Address ("::"), ipv6->GetInterfaceForAddress (dst));
    }
  else
    {
      uint32_t ifIndex = ipv6->GetInterfaceForAddress (dst);
      ipv6->GetRoutingProtocol ()->NotifyAddRoute (redirDestination, Ipv6Prefix (128), redirTarget, ifIndex);
    }
}

void Icmpv6L4Protocol::HandleDestinationUnreachable (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << *p << src << dst << interface);
  Ptr<Packet> pkt = p->Copy ();

  Icmpv6DestinationUnreachable unreach;
  pkt->RemoveHeader (unreach);

  Ipv6Header ipHeader;
  if (pkt->GetSize () > ipHeader.GetSerializedSize ())
    {
      pkt->RemoveHeader (ipHeader);
      uint8_t payload[8];
      pkt->CopyData (payload, 8);
      Forward (src, unreach, unreach.GetCode (), ipHeader, payload);
    }
}

void Icmpv6L4Protocol::HandleTimeExceeded (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << *p << src << dst << interface);
  Ptr<Packet> pkt = p->Copy ();

  Icmpv6TimeExceeded timeexceeded;
  pkt->RemoveHeader (timeexceeded);

  Ipv6Header ipHeader;
  if (pkt->GetSize () > ipHeader.GetSerializedSize ())
    {
      Ipv6Header ipHeader;
      pkt->RemoveHeader (ipHeader);
      uint8_t payload[8];
      pkt->CopyData (payload, 8);
      Forward (src, timeexceeded, timeexceeded.GetCode (), ipHeader, payload);
    }
}

void Icmpv6L4Protocol::HandlePacketTooBig (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << *p << src << dst << interface);
  Ptr<Packet> pkt = p->Copy ();

  Icmpv6TooBig tooBig;
  pkt->RemoveHeader (tooBig);

  Ipv6Header ipHeader;
  if (pkt->GetSize () > ipHeader.GetSerializedSize ())
    {
      pkt->RemoveHeader (ipHeader);
      uint8_t payload[8];
      pkt->CopyData (payload, 8);

      Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
      ipv6->SetPmtu (ipHeader.GetDestination (), tooBig.GetMtu ());

      Forward (src, tooBig, tooBig.GetMtu (), ipHeader, payload);
    }
}

void Icmpv6L4Protocol::HandleParameterError (Ptr<Packet> p, Ipv6Address const &src, Ipv6Address const &dst, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << *p << src << dst << interface);
  Ptr<Packet> pkt = p->Copy ();

  Icmpv6ParameterError paramErr;
  pkt->RemoveHeader (paramErr);

  Ipv6Header ipHeader;
  if (pkt->GetSize () > ipHeader.GetSerializedSize ())
    {
      pkt->RemoveHeader (ipHeader);
      uint8_t payload[8];
      pkt->CopyData (payload, 8);
      Forward (src, paramErr, paramErr.GetCode (), ipHeader, payload);
    }
}

void Icmpv6L4Protocol::SendMessage (Ptr<Packet> packet, Ipv6Address src, Ipv6Address dst, uint8_t ttl)
{
  NS_LOG_FUNCTION (this << packet << src << dst << (uint32_t)ttl);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  SocketIpv6HopLimitTag tag;
  NS_ASSERT (ipv6 != 0);

  tag.SetHopLimit (ttl);
  packet->AddPacketTag (tag);
  m_downTarget (packet, src, dst, PROT_NUMBER, 0);
}

void Icmpv6L4Protocol::DelayedSendMessage (Ptr<Packet> packet, Ipv6Address src, Ipv6Address dst, uint8_t ttl)
{
  NS_LOG_FUNCTION (this << packet << src << dst << (uint32_t)ttl);
  SendMessage (packet, src, dst, ttl);
}

void Icmpv6L4Protocol::SendMessage (Ptr<Packet> packet, Ipv6Address dst, Icmpv6Header& icmpv6Hdr, uint8_t ttl)
{
  NS_LOG_FUNCTION (this << packet << dst << icmpv6Hdr << (uint32_t)ttl);
  Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol> ();
  NS_ASSERT (ipv6 != 0 && ipv6->GetRoutingProtocol () != 0);
  Ipv6Header header;
  SocketIpv6HopLimitTag tag;
  Socket::SocketErrno err;
  Ptr<Ipv6Route> route;
  Ptr<NetDevice> oif (0); //specify non-zero if bound to a source address

  header.SetDestination (dst);
  route = ipv6->GetRoutingProtocol ()->RouteOutput (packet, header, oif, err);

  if (route != 0)
    {
      NS_LOG_LOGIC ("Route exists");
      tag.SetHopLimit (ttl);
      packet->AddPacketTag (tag);
      Ipv6Address src = route->GetSource ();

      icmpv6Hdr.CalculatePseudoHeaderChecksum (src, dst, packet->GetSize () + icmpv6Hdr.GetSerializedSize (), PROT_NUMBER);
      packet->AddHeader (icmpv6Hdr);
      m_downTarget (packet, src, dst, PROT_NUMBER, route);
    }
  else
    {
      NS_LOG_WARN ("drop icmp message");
    }
}

void Icmpv6L4Protocol::SendNA (Ipv6Address src, Ipv6Address dst, Address* hardwareAddress, uint8_t flags)
{
  NS_LOG_FUNCTION (this << src << dst << hardwareAddress << static_cast<uint32_t> (flags));
  Ptr<Packet> p = Create<Packet> ();
  Icmpv6NA na;
  Icmpv6OptionLinkLayerAddress llOption (0, *hardwareAddress); /* not a source link layer */

  NS_LOG_LOGIC ("Send NA ( from " << src << " to " << dst << " target " << src << ")");
  na.SetIpv6Target (src);

  if ((flags & 1))
    {
      na.SetFlagO (true);
    }
  if ((flags & 2) && src != Ipv6Address::GetAny ())
    {
      na.SetFlagS (true);
    }
  if ((flags & 4))
    {
      na.SetFlagR (true);
    }

  p->AddHeader (llOption);
  na.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + na.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (na);

  SendMessage (p, src, dst, 255);
}

void Icmpv6L4Protocol::SendEchoReply (Ipv6Address src, Ipv6Address dst, uint16_t id, uint16_t seq, Ptr<Packet> data)
{
  NS_LOG_FUNCTION (this << src << dst << id << seq << data);
  Ptr<Packet> p = data->Copy ();
  Icmpv6Echo reply (0); /* echo reply */

  reply.SetId (id);
  reply.SetSeq (seq);

  reply.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + reply.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (reply);
  SendMessage (p, src, dst, 64);
}

void Icmpv6L4Protocol::SendNS (Ipv6Address src, Ipv6Address dst, Ipv6Address target, Address hardwareAddress)
{
  NS_LOG_FUNCTION (this << src << dst << target << hardwareAddress);
  Ptr<Packet> p = Create<Packet> ();
  /* Ipv6Header ipHeader; */
  Icmpv6NS ns (target);
  Icmpv6OptionLinkLayerAddress llOption (1, hardwareAddress);  /* we give our mac address in response */

  /* if the source is unspec, multicast the NA to all-nodes multicast */
  if (src == Ipv6Address::GetAny ())
    {
      dst = Ipv6Address::GetAllNodesMulticast ();
    }

  NS_LOG_LOGIC ("Send NS ( from " << src << " to " << dst << " target " << target << ")");

  p->AddHeader (llOption);
  ns.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + ns.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (ns);
  if (!dst.IsMulticast ())
    {
      SendMessage (p, src, dst, 255);
    }
  else
    {
      NS_LOG_LOGIC ("Destination is Multicast, using DelayedSendMessage");
      Simulator::Schedule (Time (MilliSeconds (m_solicitationJitter->GetValue ())), &Icmpv6L4Protocol::DelayedSendMessage, this, p, src, dst, 255);
    }
}

void Icmpv6L4Protocol::SendRS (Ipv6Address src, Ipv6Address dst,  Address hardwareAddress)
{
  NS_LOG_FUNCTION (this << src << dst << hardwareAddress);
  Ptr<Packet> p = Create<Packet> ();
  Icmpv6RS rs;

  // RFC 4861:
  // The link-layer address of the sender MUST NOT be included if the Source Address is the unspecified address.
  // Otherwise, it SHOULD be included on link layers that have addresses.
  if (!src.IsAny ())
    {
      Icmpv6OptionLinkLayerAddress llOption (1, hardwareAddress);
      p->AddHeader (llOption);
    }

  NS_LOG_LOGIC ("Send RS (from " << src << " to " << dst << ")");

  rs.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + rs.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (rs);
  if (!dst.IsMulticast ())
    {
      SendMessage (p, src, dst, 255);
    }
  else
    {
      NS_LOG_LOGIC ("Destination is Multicast, using DelayedSendMessage");
      Simulator::Schedule (Time (MilliSeconds (m_solicitationJitter->GetValue ())), &Icmpv6L4Protocol::DelayedSendMessage, this, p, src, dst, 255);
    }
}

void Icmpv6L4Protocol::SendErrorDestinationUnreachable (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code)
{
  NS_LOG_FUNCTION (this << malformedPacket << dst << (uint32_t)code);
  uint32_t malformedPacketSize = malformedPacket->GetSize ();
  Icmpv6DestinationUnreachable header;
  header.SetCode (code);

  NS_LOG_LOGIC ("Send Destination Unreachable ( to " << dst << " code " << (uint32_t)code << " )");

  /* 48 = sizeof IPv6 header + sizeof ICMPv6 error header */
  if (malformedPacketSize <= 1280 - 48)
    {
      header.SetPacket (malformedPacket);
      SendMessage (malformedPacket, dst, header, 255);
    }
  else
    {
      Ptr<Packet> fragment = malformedPacket->CreateFragment (0, 1280 - 48);
      header.SetPacket (fragment);
      SendMessage (fragment, dst, header, 255);
    }
}

void Icmpv6L4Protocol::SendErrorTooBig (Ptr<Packet> malformedPacket, Ipv6Address dst, uint32_t mtu)
{
  NS_LOG_FUNCTION (this << malformedPacket << dst << mtu);
  uint32_t malformedPacketSize = malformedPacket->GetSize ();
  Icmpv6TooBig header;
  header.SetCode (0);
  header.SetMtu (mtu);

  NS_LOG_LOGIC ("Send Too Big ( to " << dst << " )");

  /* 48 = sizeof IPv6 header + sizeof ICMPv6 error header */
  if (malformedPacketSize <= 1280 - 48)
    {
      header.SetPacket (malformedPacket);
      SendMessage (malformedPacket, dst, header, 255);
    }
  else
    {
      Ptr<Packet> fragment = malformedPacket->CreateFragment (0, 1280 - 48);
      header.SetPacket (fragment);
      SendMessage (fragment, dst, header, 255);
    }
}

void Icmpv6L4Protocol::SendErrorTimeExceeded (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code)
{
  NS_LOG_FUNCTION (this << malformedPacket << dst << static_cast<uint32_t> (code));
  uint32_t malformedPacketSize = malformedPacket->GetSize ();
  Icmpv6TimeExceeded header;
  header.SetCode (code);

  NS_LOG_LOGIC ("Send Time Exceeded ( to " << dst << " code " << (uint32_t)code << " )");

  /* 48 = sizeof IPv6 header + sizeof ICMPv6 error header */
  if (malformedPacketSize <= 1280 - 48)
    {
      header.SetPacket (malformedPacket);
      SendMessage (malformedPacket, dst, header, 255);
    }
  else
    {
      Ptr<Packet> fragment = malformedPacket->CreateFragment (0, 1280 - 48);
      header.SetPacket (fragment);
      SendMessage (fragment, dst, header, 255);
    }
}

void Icmpv6L4Protocol::SendErrorParameterError (Ptr<Packet> malformedPacket, Ipv6Address dst, uint8_t code, uint32_t ptr)
{
  NS_LOG_FUNCTION (this << malformedPacket << dst << static_cast<uint32_t> (code) << ptr);
  uint32_t malformedPacketSize = malformedPacket->GetSize ();
  Icmpv6ParameterError header;
  header.SetCode (code);
  header.SetPtr (ptr);

  NS_LOG_LOGIC ("Send Parameter Error ( to " << dst << " code " << (uint32_t)code << " )");

  /* 48 = sizeof IPv6 header + sizeof ICMPv6 error header */
  if (malformedPacketSize <= 1280 - 48 )
    {
      header.SetPacket (malformedPacket);
      SendMessage (malformedPacket, dst, header, 255);
    }
  else
    {
      Ptr<Packet> fragment = malformedPacket->CreateFragment (0, 1280 - 48);
      header.SetPacket (fragment);
      SendMessage (fragment, dst, header, 255);
    }
}

void Icmpv6L4Protocol::SendRedirection (Ptr<Packet> redirectedPacket, Ipv6Address src, Ipv6Address dst, Ipv6Address redirTarget, Ipv6Address redirDestination, Address redirHardwareTarget)
{
  NS_LOG_FUNCTION (this << redirectedPacket << dst << redirTarget << redirDestination << redirHardwareTarget);
  uint32_t llaSize = 0;
  Ptr<Packet> p = Create<Packet> ();
  uint32_t redirectedPacketSize = redirectedPacket->GetSize ();
  Icmpv6OptionLinkLayerAddress llOption (0);

  NS_LOG_LOGIC ("Send Redirection ( to " << dst << " target " << redirTarget << " destination " << redirDestination << " )");

  Icmpv6OptionRedirected redirectedOptionHeader;

  if ((redirectedPacketSize % 8) != 0)
    {
      Ptr<Packet> pad = Create<Packet> (8 - (redirectedPacketSize % 8));
      redirectedPacket->AddAtEnd (pad);
    }

  if (redirHardwareTarget.GetLength ())
    {
      llOption.SetAddress (redirHardwareTarget);
      llaSize = llOption.GetSerializedSize ();
    }

  /* 56 = sizeof IPv6 header + sizeof ICMPv6 error header + sizeof redirected option */
  if (redirectedPacketSize <= (1280 - 56 - llaSize))
    {
      redirectedOptionHeader.SetPacket (redirectedPacket);
    }
  else
    {
      Ptr<Packet> fragment = redirectedPacket->CreateFragment (0, 1280 - 56 - llaSize);
      redirectedOptionHeader.SetPacket (fragment);
    }

  p->AddHeader (redirectedOptionHeader);

  if (llaSize)
    {
      p->AddHeader (llOption);
    }

  Icmpv6Redirection redirectionHeader;
  redirectionHeader.SetTarget (redirTarget);
  redirectionHeader.SetDestination (redirDestination);
  redirectionHeader.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + redirectionHeader.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (redirectionHeader);

  SendMessage (p, src, dst, 64);
}

NdiscCache::Ipv6PayloadHeaderPair Icmpv6L4Protocol::ForgeNA (Ipv6Address src, Ipv6Address dst, Address* hardwareAddress, uint8_t flags)
{
  NS_LOG_FUNCTION (this << src << dst << hardwareAddress << (uint32_t)flags);
  Ptr<Packet> p = Create<Packet> ();
  Ipv6Header ipHeader;
  Icmpv6NA na;
  Icmpv6OptionLinkLayerAddress llOption (0, *hardwareAddress);  /* we give our mac address in response */

  NS_LOG_LOGIC ("Send NA ( from " << src << " to " << dst << ")");

  /* forge the entire NA packet from IPv6 header to ICMPv6 link-layer option, so that the packet does not pass by Icmpv6L4Protocol::Lookup again */

  p->AddHeader (llOption);
  na.SetIpv6Target (src);

  if ((flags & 1))
    {
      na.SetFlagO (true);
    }
  if ((flags & 2) && src != Ipv6Address::GetAny ())
    {
      na.SetFlagS (true);
    }
  if ((flags & 4))
    {
      na.SetFlagR (true);
    }

  na.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + na.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (na);

  ipHeader.SetSource (src);
  ipHeader.SetDestination (dst);
  ipHeader.SetNextHeader (PROT_NUMBER);
  ipHeader.SetPayloadLength (p->GetSize ());
  ipHeader.SetHopLimit (255);

  return NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader);
}

NdiscCache::Ipv6PayloadHeaderPair Icmpv6L4Protocol::ForgeNS (Ipv6Address src, Ipv6Address dst, Ipv6Address target, Address hardwareAddress)
{
  NS_LOG_FUNCTION (this << src << dst << target << hardwareAddress);
  Ptr<Packet> p = Create<Packet> ();
  Ipv6Header ipHeader;
  Icmpv6NS ns (target);
  Icmpv6OptionLinkLayerAddress llOption (1, hardwareAddress);  /* we give our mac address in response */

  NS_LOG_LOGIC ("Send NS ( from " << src << " to " << dst << " target " << target << ")");

  p->AddHeader (llOption);
  ns.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + ns.GetSerializedSize (), PROT_NUMBER);
  p->AddHeader (ns);

  ipHeader.SetSource (src);
  ipHeader.SetDestination (dst);
  ipHeader.SetNextHeader (PROT_NUMBER);
  ipHeader.SetPayloadLength (p->GetSize ());
  ipHeader.SetHopLimit (255);

  return NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader);
}

Ptr<NdiscCache> Icmpv6L4Protocol::FindCache (Ptr<NetDevice> device)
{
  NS_LOG_FUNCTION (this << device);

  for (CacheList::const_iterator i = m_cacheList.begin (); i != m_cacheList.end (); i++)
    {
      if ((*i)->GetDevice () == device)
        {
          return *i;
        }
    }

  NS_ASSERT_MSG (false, "Icmpv6L4Protocol can not find a NDIS Cache for device " << device);
  /* quiet compiler */
  return 0;
}

Ptr<NdiscCache> Icmpv6L4Protocol::CreateCache (Ptr<NetDevice> device, Ptr<Ipv6Interface> interface)
{
  NS_LOG_FUNCTION (this << device << interface);

  Ptr<NdiscCache> cache = CreateObject<NdiscCache> ();

  cache->SetDevice (device, interface, this);
  device->AddLinkChangeCallback (MakeCallback (&NdiscCache::Flush, cache));
  m_cacheList.push_back (cache);
  return cache;
}

bool Icmpv6L4Protocol::Lookup (Ipv6Address dst, Ptr<NetDevice> device, Ptr<NdiscCache> cache, Address* hardwareDestination)
{
  NS_LOG_FUNCTION (this << dst << device << cache << hardwareDestination);

  if (!cache)
    {
      /* try to find the cache */
      cache = FindCache (device);
    }
  if (cache)
    {
      NdiscCache::Entry* entry = cache->Lookup (dst);
      if (entry)
        {
          if (entry->IsReachable () || entry->IsDelay () || entry->IsPermanent ())
            {
              *hardwareDestination = entry->GetMacAddress ();
              return true;
            }
          else if (entry->IsStale ())
            {
              entry->StartDelayTimer ();
              entry->MarkDelay ();
              *hardwareDestination = entry->GetMacAddress ();
              return true;
            }
        }
    }
  return false;
}

bool Icmpv6L4Protocol::Lookup (Ptr<Packet> p, const Ipv6Header & ipHeader, Ipv6Address dst, Ptr<NetDevice> device, Ptr<NdiscCache> cache, Address* hardwareDestination)
{
  NS_LOG_FUNCTION (this << p << ipHeader << dst << device << cache << hardwareDestination);

  if (!cache)
    {
      /* try to find the cache */
      cache = FindCache (device);
    }
  if (!cache)
    {
      return false;
    }

  NdiscCache::Entry* entry = cache->Lookup (dst);
  if (entry)
    {
      if (entry->IsReachable () || entry->IsDelay () || entry->IsPermanent ())
        {
          /* XXX check reachability time */
          /* send packet */
          *hardwareDestination = entry->GetMacAddress ();
          return true;
        }
      else if (entry->IsStale ())
        {
          /* start delay timer */
          entry->StartDelayTimer ();
          entry->MarkDelay ();
          *hardwareDestination = entry->GetMacAddress ();
          return true;
        }
      else /* INCOMPLETE or PROBE */
        {
          /* queue packet */
          entry->AddWaitingPacket (NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader));
          return false;
        }
    }
  else
    {
      /* we contact this node for the first time
       * add it to the cache and send an NS
       */
      Ipv6Address addr;
      NdiscCache::Entry* entry = cache->Add (dst);
      entry->MarkIncomplete (NdiscCache::Ipv6PayloadHeaderPair (p, ipHeader));
      entry->SetRouter (false);

      if (dst.IsLinkLocal ())
        {
          addr = cache->GetInterface ()->GetLinkLocalAddress ().GetAddress ();
        }
      else if (cache->GetInterface ()->GetNAddresses () == 1) /* an interface have at least one address (link-local) */
        {
          /* try to resolve global address without having global address so return! */
          cache->Remove (entry);
          return false;
        }
      else
        {
          /* find source address that match destination */
          addr = cache->GetInterface ()->GetAddressMatchingDestination (dst).GetAddress ();
        }

      SendNS (addr, Ipv6Address::MakeSolicitedAddress (dst), dst, cache->GetDevice ()->GetAddress ());

      /* start retransmit timer */
      entry->StartRetransmitTimer ();
      return false;
    }

  return false;
}

void Icmpv6L4Protocol::FunctionDadTimeout (Ipv6Interface* interface, Ipv6Address addr)
{
  NS_LOG_FUNCTION  (this << interface << addr);

  Ipv6InterfaceAddress ifaddr;
  bool found = false;
  uint32_t i = 0;
  uint32_t nb = interface->GetNAddresses ();

  for (i = 0; i < nb; i++)
    {
      ifaddr = interface->GetAddress (i);

      if (ifaddr.GetAddress () == addr)
        {
          found = true;
          break;
        }
    }

  if (!found)
    {
      NS_LOG_LOGIC ("Can not find the address in the interface.");
    }

  /* for the moment, this function is always called, if we was victim of a DAD the address is INVALID
   * and we do not set it to PREFERRED
   */
  if (found && ifaddr.GetState () != Ipv6InterfaceAddress::INVALID)
    {
      interface->SetState (ifaddr.GetAddress (), Ipv6InterfaceAddress::PREFERRED);
      NS_LOG_LOGIC ("DAD OK, interface in state PREFERRED");

      /* send an RS if our interface is not forwarding (router) and if address is a link-local ones
       * (because we will send RS with it)
       */
      Ptr<Ipv6> ipv6 = m_node->GetObject<Ipv6> ();

      if (!ipv6->IsForwarding (ipv6->GetInterfaceForDevice (interface->GetDevice ())) && addr.IsLinkLocal ())
        {
          /* \todo Add random delays before sending RS
           * because all nodes start at the same time, there will be many of RS around 1 second of simulation time
           */
          NS_LOG_LOGIC ("Scheduled a Router Solicitation");
          Simulator::Schedule (Seconds (0.0), &Icmpv6L4Protocol::SendRS, this, ifaddr.GetAddress (), Ipv6Address::GetAllRoutersMulticast (), interface->GetDevice ()->GetAddress ());
        }
      else
        {
          NS_LOG_LOGIC ("Did not schedule a Router Solicitation because the interface is in forwarding mode");
        }
    }
}

void
Icmpv6L4Protocol::SetDownTarget (IpL4Protocol::DownTargetCallback callback)
{
  NS_LOG_FUNCTION (this << &callback);
}

void
Icmpv6L4Protocol::SetDownTarget6 (IpL4Protocol::DownTargetCallback6 callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_downTarget = callback;
}

IpL4Protocol::DownTargetCallback
Icmpv6L4Protocol::GetDownTarget (void) const
{
  NS_LOG_FUNCTION (this);
  return (IpL4Protocol::DownTargetCallback)NULL;
}

IpL4Protocol::DownTargetCallback6
Icmpv6L4Protocol::GetDownTarget6 (void) const
{
  NS_LOG_FUNCTION (this);
  return m_downTarget;
}

uint8_t
Icmpv6L4Protocol::GetMaxMulticastSolicit () const
{
  return m_maxMulticastSolicit;
}

uint8_t
Icmpv6L4Protocol::GetMaxUnicastSolicit () const
{
  return m_maxUnicastSolicit;
}

Time
Icmpv6L4Protocol::GetReachableTime () const
{
  return m_reachableTime;
}

Time
Icmpv6L4Protocol::GetRetransmissionTime () const
{
  return m_retransmissionTime;
}

Time
Icmpv6L4Protocol::GetDelayFirstProbe () const
{
  return m_delayFirstProbe;
}


} /* namespace ns3 */

