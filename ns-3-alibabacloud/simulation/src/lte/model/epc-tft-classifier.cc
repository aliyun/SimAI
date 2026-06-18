/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
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
 * Authors: 
 *   Nicola Baldo <nbaldo@cttc.es> (the EpcTftClassifier class)
 *   Giuseppe Piro <g.piro@poliba.it> (part of the code in EpcTftClassifier::Classify () 
 *       which comes from RrcEntity::Classify of the GSoC 2010 LTE module)
 *
 */

#include "epc-tft-classifier.h"
#include "epc-tft.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv6-header.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/icmpv4-l4-protocol.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("EpcTftClassifier");

EpcTftClassifier::EpcTftClassifier ()
{
  NS_LOG_FUNCTION (this);
}

void
EpcTftClassifier::Add (Ptr<EpcTft> tft, uint32_t id)
{
  NS_LOG_FUNCTION (this << tft << id);
  m_tftMap[id] = tft;

  // simple sanity check: there shouldn't be more than 16 bearers (hence TFTs) per UE
  NS_ASSERT (m_tftMap.size () <= 16);
}

void
EpcTftClassifier::Delete (uint32_t id)
{
  NS_LOG_FUNCTION (this << id);
  m_tftMap.erase (id);
}

uint32_t 
EpcTftClassifier::Classify (Ptr<Packet> p, EpcTft::Direction direction, uint16_t protocolNumber)
{
  NS_LOG_FUNCTION (this << p << p->GetSize () << direction);

  Ptr<Packet> pCopy = p->Copy ();

  Ipv4Address localAddressIpv4;
  Ipv4Address remoteAddressIpv4;

  Ipv6Address localAddressIpv6;
  Ipv6Address remoteAddressIpv6;

  uint8_t protocol;
  uint8_t tos;

  uint16_t localPort = 0;
  uint16_t remotePort = 0;

  if (protocolNumber == Ipv4L3Protocol::PROT_NUMBER)
    {
      Ipv4Header ipv4Header;
      pCopy->RemoveHeader (ipv4Header);

      if (direction ==  EpcTft::UPLINK)
        {
          localAddressIpv4 = ipv4Header.GetSource ();
          remoteAddressIpv4 = ipv4Header.GetDestination ();
        }
      else
        {
          NS_ASSERT (direction ==  EpcTft::DOWNLINK);
          remoteAddressIpv4 = ipv4Header.GetSource ();
          localAddressIpv4 = ipv4Header.GetDestination ();
        }
      NS_LOG_INFO ("local address: " << localAddressIpv4 << " remote address: " << remoteAddressIpv4);

      uint16_t payloadSize = ipv4Header.GetPayloadSize ();
      uint16_t fragmentOffset = ipv4Header.GetFragmentOffset ();
      bool isLastFragment = ipv4Header.IsLastFragment ();

      // NS_LOG_DEBUG ("PayloadSize = " << payloadSize);
      // NS_LOG_DEBUG ("fragmentOffset " << fragmentOffset << " isLastFragment " << isLastFragment);

      protocol = ipv4Header.GetProtocol ();
      tos = ipv4Header.GetTos ();

      // Port info only can be get if it is the first fragment and
      // there is enough data in the payload
      // We keep the port info for fragmented packets,
      // i.e. it is the first one but it is not the last one
      if (fragmentOffset == 0)
        {
          if (protocol == UdpL4Protocol::PROT_NUMBER && payloadSize >= 8)
            {
              UdpHeader udpHeader;
              pCopy->RemoveHeader (udpHeader);
              if (direction ==  EpcTft::UPLINK)
                {
                  localPort = udpHeader.GetSourcePort ();
                  remotePort = udpHeader.GetDestinationPort ();
                }
              else
                {
                  remotePort = udpHeader.GetSourcePort ();
                  localPort = udpHeader.GetDestinationPort ();
                }
              if (!isLastFragment)
                {
                  std::tuple<uint32_t, uint32_t, uint8_t, uint16_t> fragmentKey =
                      std::make_tuple (ipv4Header.GetSource ().Get (),
                                       ipv4Header.GetDestination ().Get (),
                                       protocol,
                                       ipv4Header.GetIdentification ());

                  m_classifiedIpv4Fragments[fragmentKey] = std::make_pair (localPort, remotePort);
                }
            }
          else if (protocol == TcpL4Protocol::PROT_NUMBER && payloadSize >= 20)
            {
              TcpHeader tcpHeader;
              pCopy->RemoveHeader (tcpHeader);
              if (direction ==  EpcTft::UPLINK)
                {
                  localPort = tcpHeader.GetSourcePort ();
                  remotePort = tcpHeader.GetDestinationPort ();
                }
              else
                {
                  remotePort = tcpHeader.GetSourcePort ();
                  localPort = tcpHeader.GetDestinationPort ();
                }

              if (!isLastFragment)
                {
                  std::tuple<uint32_t, uint32_t, uint8_t, uint16_t> fragmentKey =
                      std::make_tuple (ipv4Header.GetSource ().Get (),
                                       ipv4Header.GetDestination ().Get (),
                                       protocol,
                                       ipv4Header.GetIdentification ());

                  m_classifiedIpv4Fragments[fragmentKey] = std::make_pair (localPort, remotePort);
                }
            }

          // else
          //   First fragment but not enough data for port info or not UDP/TCP protocol.
          //   Nothing can be done, i.e. we cannot get port info from packet.
        }
      else
        {
          // Not first fragment, so port info is not available but
          // port info should already be known (if there is not fragment reordering)
          std::tuple<uint32_t, uint32_t, uint8_t, uint16_t> fragmentKey =
              std::make_tuple (ipv4Header.GetSource ().Get (),
                               ipv4Header.GetDestination ().Get (),
                               protocol,
                               ipv4Header.GetIdentification ());

          std::map< std::tuple<uint32_t, uint32_t, uint8_t, uint16_t>,
                    std::pair<uint32_t, uint32_t> >::iterator it =
              m_classifiedIpv4Fragments.find (fragmentKey);

          if (it != m_classifiedIpv4Fragments.end ())
            {
              localPort = it->second.first;
              remotePort = it->second.second;

              if (isLastFragment)
                {
                  m_classifiedIpv4Fragments.erase (fragmentKey);
                }
            }
        }
    }
  else if (protocolNumber == Ipv6L3Protocol::PROT_NUMBER)
    {
      Ipv6Header ipv6Header;
      pCopy->RemoveHeader (ipv6Header);

      if (direction ==  EpcTft::UPLINK)
        {
          localAddressIpv6 = ipv6Header.GetSource ();
          remoteAddressIpv6 = ipv6Header.GetDestination ();
        }
      else
        {
          NS_ASSERT (direction ==  EpcTft::DOWNLINK);
          remoteAddressIpv6 = ipv6Header.GetSource ();
          localAddressIpv6 = ipv6Header.GetDestination ();
        }
      NS_LOG_INFO ("local address: " << localAddressIpv6 << " remote address: " << remoteAddressIpv6);

      protocol = ipv6Header.GetNextHeader ();
      tos = ipv6Header.GetTrafficClass ();

      if (protocol == UdpL4Protocol::PROT_NUMBER)
        {
          UdpHeader udpHeader;
          pCopy->RemoveHeader (udpHeader);

          if (direction ==  EpcTft::UPLINK)
            {
              localPort = udpHeader.GetSourcePort ();
              remotePort = udpHeader.GetDestinationPort ();
            }
          else
            {
              remotePort = udpHeader.GetSourcePort ();
              localPort = udpHeader.GetDestinationPort ();
            }
        }
      else if (protocol == TcpL4Protocol::PROT_NUMBER)
        {
          TcpHeader tcpHeader;
          pCopy->RemoveHeader (tcpHeader);
          if (direction ==  EpcTft::UPLINK)
            {
              localPort = tcpHeader.GetSourcePort ();
              remotePort = tcpHeader.GetDestinationPort ();
            }
          else
            {
              remotePort = tcpHeader.GetSourcePort ();
              localPort = tcpHeader.GetDestinationPort ();
            }
        }
    }
  else
    {
      NS_ABORT_MSG ("EpcTftClassifier::Classify - Unknown IP type...");
    }


  if (protocolNumber == Ipv4L3Protocol::PROT_NUMBER)
    {
      NS_LOG_INFO ("Classifying packet:"
          << " localAddr="  << localAddressIpv4
          << " remoteAddr=" << remoteAddressIpv4
          << " localPort="  << localPort
          << " remotePort=" << remotePort
          << " tos=0x" << (uint16_t) tos );

      // now it is possible to classify the packet!
      // we use a reverse iterator since filter priority is not implemented properly.
      // This way, since the default bearer is expected to be added first, it will be evaluated last.
      std::map <uint32_t, Ptr<EpcTft> >::const_reverse_iterator it;
      NS_LOG_LOGIC ("TFT MAP size: " << m_tftMap.size ());

      for (it = m_tftMap.rbegin (); it != m_tftMap.rend (); ++it)
        {
          NS_LOG_LOGIC ("TFT id: " << it->first );
          NS_LOG_LOGIC (" Ptr<EpcTft>: " << it->second);
          Ptr<EpcTft> tft = it->second;
          if (tft->Matches (direction, remoteAddressIpv4, localAddressIpv4, remotePort, localPort, tos))
            {
              NS_LOG_LOGIC ("matches with TFT ID = " << it->first);
              return it->first; // the id of the matching TFT
            }
        }
    }
  else if (protocolNumber == Ipv6L3Protocol::PROT_NUMBER)
    {
      NS_LOG_INFO ("Classifying packet:"
          << " localAddr="  << localAddressIpv6
          << " remoteAddr=" << remoteAddressIpv6
          << " localPort="  << localPort
          << " remotePort=" << remotePort
          << " tos=0x" << (uint16_t) tos );

      // now it is possible to classify the packet!
      // we use a reverse iterator since filter priority is not implemented properly.
      // This way, since the default bearer is expected to be added first, it will be evaluated last.
      std::map <uint32_t, Ptr<EpcTft> >::const_reverse_iterator it;
      NS_LOG_LOGIC ("TFT MAP size: " << m_tftMap.size ());

      for (it = m_tftMap.rbegin (); it != m_tftMap.rend (); ++it)
        {
          NS_LOG_LOGIC ("TFT id: " << it->first );
          NS_LOG_LOGIC (" Ptr<EpcTft>: " << it->second);
          Ptr<EpcTft> tft = it->second;
          if (tft->Matches (direction, remoteAddressIpv6, localAddressIpv6, remotePort, localPort, tos))
            {
              NS_LOG_LOGIC ("matches with TFT ID = " << it->first);
              return it->first; // the id of the matching TFT
            }
        }
    }
  NS_LOG_LOGIC ("no match");
  return 0;  // no match
}


} // namespace ns3
