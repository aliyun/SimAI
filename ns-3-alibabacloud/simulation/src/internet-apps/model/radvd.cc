/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 Telecom Bretagne
 * Copyright (c) 2009 Strasbourg University
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
 *         Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 */

#include "radvd.h"
#include "ns3/log.h"
#include "ns3/abort.h"
#include "ns3/ipv6-address.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/net-device.h"
#include "ns3/uinteger.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/ipv6.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/ipv6-interface.h"
#include "ns3/ipv6-raw-socket-factory.h"
#include "ns3/ipv6-packet-info-tag.h"
#include "ns3/ipv6-header.h"
#include "ns3/icmpv6-header.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/socket.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("RadvdApplication");

NS_OBJECT_ENSURE_REGISTERED (Radvd);

TypeId Radvd::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Radvd")
    .SetParent<Application> ()
    .SetGroupName("Internet-Apps")
    .AddConstructor<Radvd> ()
    .AddAttribute ("AdvertisementJitter",
                   "Uniform variable to provide jitter between min and max values of AdvInterval",
                   StringValue("ns3::UniformRandomVariable"),
                   MakePointerAccessor (&Radvd::m_jitter),
                   MakePointerChecker<UniformRandomVariable> ());
  ;
  return tid;
}

Radvd::Radvd ()
{
  NS_LOG_FUNCTION (this);
}

Radvd::~Radvd ()
{
  NS_LOG_FUNCTION (this);
  for (RadvdInterfaceListI it = m_configurations.begin (); it != m_configurations.end (); ++it)
    {
      *it = 0;
    }
  m_configurations.clear ();
  m_recvSocket = 0;
}

void Radvd::DoDispose ()
{
  NS_LOG_FUNCTION (this);

  m_recvSocket->Close ();
  m_recvSocket = 0;

  for (SocketMapI it = m_sendSockets.begin (); it != m_sendSockets.end (); ++it)
    {
      it->second->Close ();
      it->second = 0;
    }

  Application::DoDispose ();
}

void Radvd::StartApplication ()
{
  NS_LOG_FUNCTION (this);

  TypeId tid = TypeId::LookupByName ("ns3::Ipv6RawSocketFactory");

  if (!m_recvSocket)
    {
      m_recvSocket = Socket::CreateSocket (GetNode (), tid);

      NS_ASSERT (m_recvSocket);

      m_recvSocket->Bind (Inet6SocketAddress (Ipv6Address::GetAllRoutersMulticast (), 0));
      m_recvSocket->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
      m_recvSocket->SetRecvCallback (MakeCallback (&Radvd::HandleRead, this));
      m_recvSocket->ShutdownSend ();
      m_recvSocket->SetRecvPktInfo (true);
    }

  for (RadvdInterfaceListCI it = m_configurations.begin (); it != m_configurations.end (); it++)
    {
      if ((*it)->IsSendAdvert ())
        {
          m_unsolicitedEventIds[(*it)->GetInterface ()] = Simulator::Schedule (Seconds (0.), &Radvd::Send,
                                                                               this, (*it), Ipv6Address::GetAllNodesMulticast (), true);
        }

      if (m_sendSockets.find ((*it)->GetInterface ()) == m_sendSockets.end ())
        {
          Ptr<Ipv6L3Protocol> ipv6 = GetNode ()->GetObject<Ipv6L3Protocol> ();
          Ptr<Ipv6Interface> iFace = ipv6->GetInterface ((*it)->GetInterface ());

          m_sendSockets[(*it)->GetInterface ()] = Socket::CreateSocket (GetNode (), tid);
          m_sendSockets[(*it)->GetInterface ()]->Bind (Inet6SocketAddress (iFace->GetLinkLocalAddress ().GetAddress (), 0));
          m_sendSockets[(*it)->GetInterface ()]->SetAttribute ("Protocol", UintegerValue (Ipv6Header::IPV6_ICMPV6));
          m_sendSockets[(*it)->GetInterface ()]->ShutdownRecv ();
        }
    }
}

void Radvd::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  if (m_recvSocket)
    {
      m_recvSocket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }

  for (EventIdMapI it = m_unsolicitedEventIds.begin (); it != m_unsolicitedEventIds.end (); ++it)
    {
      Simulator::Cancel ((*it).second);
    }
  m_unsolicitedEventIds.clear ();

  for (EventIdMapI it = m_solicitedEventIds.begin (); it != m_solicitedEventIds.end (); ++it)
    {
      Simulator::Cancel ((*it).second);
    }
  m_solicitedEventIds.clear ();
}

void Radvd::AddConfiguration (Ptr<RadvdInterface> routerInterface)
{
  NS_LOG_FUNCTION (this << routerInterface);
  m_configurations.push_back (routerInterface);
}

int64_t 
Radvd:: AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_jitter->SetStream (stream);
  return 1;
}

void Radvd::Send (Ptr<RadvdInterface> config, Ipv6Address dst, bool reschedule)
{
  NS_LOG_FUNCTION (this << dst << reschedule);

  if (reschedule == true)
    {
      config->SetLastRaTxTime (Simulator::Now ());
    }

  Icmpv6RA raHdr;
  Icmpv6OptionLinkLayerAddress llaHdr;
  Icmpv6OptionMtu mtuHdr;
  Icmpv6OptionPrefixInformation prefixHdr;

  std::list<Ptr<RadvdPrefix> > prefixes = config->GetPrefixes ();
  Ptr<Packet> p = Create<Packet> ();
  Ptr<Ipv6> ipv6 = GetNode ()->GetObject<Ipv6> ();

  /* set RA header information */
  raHdr.SetFlagM (config->IsManagedFlag ());
  raHdr.SetFlagO (config->IsOtherConfigFlag ());
  raHdr.SetFlagH (config->IsHomeAgentFlag ());
  raHdr.SetCurHopLimit (config->GetCurHopLimit ());
  raHdr.SetLifeTime (config->GetDefaultLifeTime ());
  raHdr.SetReachableTime (config->GetReachableTime ());
  raHdr.SetRetransmissionTime (config->GetRetransTimer ());

  if (config->IsSourceLLAddress ())
    {
      /* Get L2 address from NetDevice */
      Address addr = ipv6->GetNetDevice (config->GetInterface ())->GetAddress ();
      llaHdr = Icmpv6OptionLinkLayerAddress (true, addr);
      p->AddHeader (llaHdr);
    }

  if (config->GetLinkMtu ())
    {
      NS_ASSERT (config->GetLinkMtu () >= 1280);
      mtuHdr = Icmpv6OptionMtu (config->GetLinkMtu ());
      p->AddHeader (mtuHdr);
    }

  /* add list of prefixes */
  for (std::list<Ptr<RadvdPrefix> >::const_iterator jt = prefixes.begin (); jt != prefixes.end (); jt++)
    {
      uint8_t flags = 0;
      prefixHdr = Icmpv6OptionPrefixInformation ();
      prefixHdr.SetPrefix ((*jt)->GetNetwork ());
      prefixHdr.SetPrefixLength ((*jt)->GetPrefixLength ());
      prefixHdr.SetValidTime ((*jt)->GetValidLifeTime ());
      prefixHdr.SetPreferredTime ((*jt)->GetPreferredLifeTime ());

      if ((*jt)->IsOnLinkFlag ())
        {
          flags |= Icmpv6OptionPrefixInformation::ONLINK;
        }

      if ((*jt)->IsAutonomousFlag ())
        {
          flags |= Icmpv6OptionPrefixInformation::AUTADDRCONF;
        }

      if ((*jt)->IsRouterAddrFlag ())
        {
          flags |= Icmpv6OptionPrefixInformation::ROUTERADDR;
        }

      prefixHdr.SetFlags (flags);

      p->AddHeader (prefixHdr);
    }

  Address sockAddr;
  m_sendSockets[config->GetInterface ()]->GetSockName (sockAddr);
  Ipv6Address src = Inet6SocketAddress::ConvertFrom (sockAddr).GetIpv6 ();

  /* as we know interface index that will be used to send RA and 
   * we always send RA with router's link-local address, we can 
   * calculate checksum here.
   */
  raHdr.CalculatePseudoHeaderChecksum (src, dst, p->GetSize () + raHdr.GetSerializedSize (), 58 /* ICMPv6 */);
  p->AddHeader (raHdr);

  /* Router advertisements MUST always have a ttl of 255
   * The ttl value should be set as a socket option, but this is not yet implemented
   */
  SocketIpTtlTag ttl;
  ttl.SetTtl (255);
  p->AddPacketTag (ttl);

  /* send RA */
  NS_LOG_LOGIC ("Send RA to " << dst);
  m_sendSockets[config->GetInterface ()]->SendTo (p, 0, Inet6SocketAddress (dst, 0));

  if (reschedule)
    {
      uint64_t delay = static_cast<uint64_t> (m_jitter->GetValue (config->GetMinRtrAdvInterval (), config->GetMaxRtrAdvInterval ()) + 0.5);
      if (config->IsInitialRtrAdv ())
        {
          if (delay > MAX_INITIAL_RTR_ADVERT_INTERVAL)
            delay = MAX_INITIAL_RTR_ADVERT_INTERVAL;
        }

      NS_LOG_INFO ("Reschedule in " << delay << " milliseconds");
      Time t = MilliSeconds (delay);
      m_unsolicitedEventIds[config->GetInterface ()] = Simulator::Schedule (t, &Radvd::Send, this, config, Ipv6Address::GetAllNodesMulticast (), true);
    }
}

void Radvd::HandleRead (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);
  Ptr<Packet> packet = 0;
  Address from;

  while ((packet = socket->RecvFrom (from)))
    {
      if (Inet6SocketAddress::IsMatchingType (from))
        {
          Ipv6PacketInfoTag interfaceInfo;
          if (!packet->RemovePacketTag (interfaceInfo))
            {
              NS_ABORT_MSG ("No incoming interface on RADVD message, aborting.");
            }
          uint32_t incomingIf = interfaceInfo.GetRecvIf ();
          Ptr<NetDevice> dev = GetNode ()->GetDevice (incomingIf);
          Ptr<Ipv6> ipv6 = GetNode ()->GetObject<Ipv6> ();
          uint32_t ipInterfaceIndex = ipv6->GetInterfaceForDevice (dev);

          Ipv6Header hdr;
          Icmpv6RS rsHdr;
          uint64_t delay = 0;
          Time t;

          packet->RemoveHeader (hdr);
          uint8_t type;
          packet->CopyData (&type, sizeof(type));

          switch (type)
            {
            case Icmpv6Header::ICMPV6_ND_ROUTER_SOLICITATION:
              packet->RemoveHeader (rsHdr);
              NS_LOG_INFO ("Received ICMPv6 Router Solicitation from " << hdr.GetSource () << " code = " << (uint32_t)rsHdr.GetCode ());

              for (RadvdInterfaceListCI it = m_configurations.begin (); it != m_configurations.end (); it++)
                {
                  if (ipInterfaceIndex == (*it)->GetInterface ())
                    {
                      /* calculate minimum delay between RA */
                      delay = static_cast<uint64_t> (m_jitter->GetValue (0, MAX_RA_DELAY_TIME) + 0.5);
                      t = Simulator::Now () + MilliSeconds (delay); /* absolute time of solicited RA */

                      if (Simulator::Now () < (*it)->GetLastRaTxTime () + MilliSeconds (MIN_DELAY_BETWEEN_RAS) )
                        {
                          t += MilliSeconds (MIN_DELAY_BETWEEN_RAS);
                        }

                      /* if our solicited RA is before the next periodic RA, we schedule it */
                      bool scheduleSingle = true;

                      if (m_solicitedEventIds.find ((*it)->GetInterface ()) != m_solicitedEventIds.end ())
                        {
                          if (m_solicitedEventIds[(*it)->GetInterface ()].IsRunning ())
                            {
                              scheduleSingle = false;
                            }
                        }

                      if (m_unsolicitedEventIds.find ((*it)->GetInterface ()) != m_unsolicitedEventIds.end ())
                        {
                          if (t.GetTimeStep () > static_cast<int64_t> (m_unsolicitedEventIds[(*it)->GetInterface ()].GetTs ()))
                            {
                              scheduleSingle = false;
                            }
                        }

                      if (scheduleSingle)
                        {
                          NS_LOG_INFO ("schedule new RA");
                          m_solicitedEventIds[(*it)->GetInterface ()] = Simulator::Schedule (MilliSeconds (delay), &Radvd::Send,
                                                                                             this, (*it), Ipv6Address::GetAllNodesMulticast (), false);
                        }
                    }
                }
              break;
            default:
              break;
            }
        }
    }
}

} /* namespace ns3 */

