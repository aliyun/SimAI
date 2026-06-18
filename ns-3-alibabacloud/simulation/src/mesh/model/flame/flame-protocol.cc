/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#include "flame-protocol.h"
#include "flame-protocol-mac.h"
#include "flame-header.h"
#include "flame-rtable.h"
#include "ns3/llc-snap-header.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/mesh-point-device.h"
#include "ns3/wifi-net-device.h"
#include "ns3/mesh-wifi-interface-mac.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FlameProtocol");
  
namespace flame {
  
//-----------------------------------------------------------------------------
// FlameTag
//-----------------------------------------------------------------------------
NS_OBJECT_ENSURE_REGISTERED (FlameTag);
NS_OBJECT_ENSURE_REGISTERED (FlameProtocol);

TypeId
FlameTag::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::flame::FlameTag")
    .SetParent<Tag> ()
    .SetGroupName ("Mesh")
    .AddConstructor<FlameTag> ()
    ;
  return tid;
}

TypeId
FlameTag::GetInstanceTypeId () const
{
  return GetTypeId ();
}

uint32_t
FlameTag::GetSerializedSize () const
{
  return 12;
}

void
FlameTag::Serialize (TagBuffer i) const
{
  uint8_t buf[6];
  receiver.CopyTo (buf);
  for (int j = 0; j < 6; j++)
    {
      i.WriteU8 (buf[j]);
    }
  transmitter.CopyTo (buf);
  for (int j = 0; j < 6; j++)
    {
      i.WriteU8 (buf[j]);
    }

}

void
FlameTag::Deserialize (TagBuffer i)
{
  uint8_t buf[6];
  for (int j = 0; j < 6; j++)
    {
      buf[j] = i.ReadU8 ();
    }
  receiver.CopyFrom (buf);
  for (int j = 0; j < 6; j++)
    {
      buf[j] = i.ReadU8 ();
    }
  transmitter.CopyFrom (buf);

}

void
FlameTag::Print (std::ostream &os) const
{
  os << "receiver = " << receiver << ", transmitter = " << transmitter;
}

//-----------------------------------------------------------------------------
// FlameProtocol
//-----------------------------------------------------------------------------
TypeId
FlameProtocol::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::flame::FlameProtocol")
    .SetParent<MeshL2RoutingProtocol> ()
    .SetGroupName ("Mesh")
    .AddConstructor<FlameProtocol> ()
    .AddAttribute ( "BroadcastInterval",
                    "How often we must send broadcast packets",
                    TimeValue (Seconds (5)),
                    MakeTimeAccessor (
                      &FlameProtocol::m_broadcastInterval),
                    MakeTimeChecker ()
                    )
    .AddAttribute ( "MaxCost",
                    "Cost threshold after which packet will be dropped",
                    UintegerValue (32),
                    MakeUintegerAccessor (
                      &FlameProtocol::m_maxCost),
                    MakeUintegerChecker<uint8_t> (3)
                    )
  ;
  return tid;
}
FlameProtocol::FlameProtocol () :
  m_address (Mac48Address ()), m_broadcastInterval (Seconds (5)), m_lastBroadcast (Seconds (0)),
  m_maxCost (32), m_myLastSeqno (1), m_rtable (CreateObject<FlameRtable> ())
{
}
FlameProtocol::~FlameProtocol ()
{
}
void
FlameProtocol::DoDispose ()
{
  m_interfaces.clear ();
  m_rtable = 0;
  m_mp = 0;
}
bool
FlameProtocol::RequestRoute (uint32_t sourceIface, const Mac48Address source, const Mac48Address destination,
                             Ptr<const Packet> const_packet, uint16_t protocolType, RouteReplyCallback routeReply)
{
  Ptr<Packet> packet = const_packet->Copy ();
  if (sourceIface == m_mp->GetIfIndex ())
    {
      //Packet from upper layer!
      FlameTag tag;
      if (packet->PeekPacketTag (tag))
        {
          NS_FATAL_ERROR ("FLAME tag is not supposed to be received from upper layers");
        }
      FlameRtable::LookupResult result = m_rtable->Lookup (destination);
      if (result.retransmitter == Mac48Address::GetBroadcast ())
        {
          m_lastBroadcast = Simulator::Now ();
        }
      if (m_lastBroadcast + m_broadcastInterval < Simulator::Now ())
        {
          result.retransmitter = Mac48Address::GetBroadcast ();
          result.ifIndex = FlameRtable::INTERFACE_ANY;
          m_lastBroadcast = Simulator::Now ();
        }
      FlameHeader flameHdr;
      flameHdr.AddCost (0);
      flameHdr.SetSeqno (m_myLastSeqno++);
      flameHdr.SetProtocol (protocolType);
      flameHdr.SetOrigDst (destination);
      flameHdr.SetOrigSrc (source);
      m_stats.txBytes += packet->GetSize ();
      packet->AddHeader (flameHdr);
      tag.receiver = result.retransmitter;
      if (result.retransmitter == Mac48Address::GetBroadcast ())
        {
          m_stats.txBroadcast++;
        }
      else
        {
          m_stats.txUnicast++;
        }
      NS_LOG_DEBUG ("Source: send packet with RA = " << tag.receiver);
      packet->AddPacketTag (tag);
      routeReply (true, packet, source, destination, FLAME_PROTOCOL, result.ifIndex);
    }
  else
    {
      FlameHeader flameHdr;
      packet->RemoveHeader (flameHdr);
      FlameTag tag;

      if (!packet->RemovePacketTag (tag))
        {
          NS_FATAL_ERROR ("FLAME tag must exist here");
        }
      if (destination == Mac48Address::GetBroadcast ())
        {
          //Broadcast always is forwarded as broadcast!
          NS_ASSERT (HandleDataFrame (flameHdr.GetSeqno (), source, flameHdr, tag.transmitter, sourceIface));
          FlameTag tag (Mac48Address::GetBroadcast ());
          flameHdr.AddCost (1);
          m_stats.txBytes += packet->GetSize ();
          packet->AddHeader (flameHdr);
          packet->AddPacketTag (tag);
          routeReply (true, packet, source, destination, FLAME_PROTOCOL, FlameRtable::INTERFACE_ANY);
          m_stats.txBroadcast++;
          return true;
        }
      else
        {
          // We check sequence only when forward unicast, because broadcast-checks were done
          // inside remove routing stuff.
          if (HandleDataFrame (flameHdr.GetSeqno (), source, flameHdr, tag.transmitter, sourceIface))
            {
              return false;
            }
          FlameRtable::LookupResult result = m_rtable->Lookup (destination);
          if (tag.receiver != Mac48Address::GetBroadcast ())
            {
              if (result.retransmitter == Mac48Address::GetBroadcast ())
                {
                  NS_LOG_DEBUG ("unicast packet dropped, because no route! I am " << GetAddress ()
                                                                                  << ", RA = " << tag.receiver << ", TA = " << tag.transmitter);
                  m_stats.totalDropped++;
                  return false;
                }
              tag.receiver = result.retransmitter;
            }
          else
            {
              tag.receiver = Mac48Address::GetBroadcast ();
            }
          if (result.retransmitter == Mac48Address::GetBroadcast ())
            {
              m_stats.txBroadcast++;
            }
          else
            {
              m_stats.txUnicast++;
            }
          m_stats.txBytes += packet->GetSize ();
          flameHdr.AddCost (1);
          packet->AddHeader (flameHdr);
          packet->AddPacketTag (tag);
          routeReply (true, packet, source, destination, FLAME_PROTOCOL, result.ifIndex);
          return true;
        }
      return true;
    }
  return false;
}
bool
FlameProtocol::RemoveRoutingStuff (uint32_t fromIface, const Mac48Address source,
                                   const Mac48Address destination, Ptr<Packet> packet, uint16_t& protocolType)
{
  //Filter seqno:
  if (source == GetAddress ())
    {
      NS_LOG_DEBUG ("Dropped my own frame!");
      return false;
    }
  FlameTag tag;
  if (!packet->RemovePacketTag (tag))
    {
      NS_FATAL_ERROR ("FLAME tag must exist when packet is coming to protocol");
    }
  FlameHeader flameHdr;
  packet->RemoveHeader (flameHdr);
  if (HandleDataFrame (flameHdr.GetSeqno (), source, flameHdr, tag.transmitter, fromIface))
    {
      return false;
    }
  // Start PATH_UPDATE procedure if destination is our own address and last broadcast was sent more
  // than broadcast interval ago or was not sent at all
  if ((destination == GetAddress ()) && ((m_lastBroadcast + m_broadcastInterval < Simulator::Now ())
                                         || (m_lastBroadcast == Seconds (0))))
    {
      Ptr<Packet> packet = Create<Packet> ();
      m_mp->Send (packet, Mac48Address::GetBroadcast (), 0);
      m_lastBroadcast = Simulator::Now ();
    }
  NS_ASSERT (protocolType == FLAME_PROTOCOL);
  protocolType = flameHdr.GetProtocol ();
  return true;
}
bool
FlameProtocol::Install (Ptr<MeshPointDevice> mp)
{
  m_mp = mp;
  std::vector<Ptr<NetDevice> > interfaces = mp->GetInterfaces ();
  for (std::vector<Ptr<NetDevice> >::const_iterator i = interfaces.begin (); i != interfaces.end (); i++)
    {
      // Checking for compatible net device
      Ptr<WifiNetDevice> wifiNetDev = (*i)->GetObject<WifiNetDevice> ();
      if (wifiNetDev == 0)
        {
          return false;
        }
      Ptr<MeshWifiInterfaceMac> mac = wifiNetDev->GetMac ()->GetObject<MeshWifiInterfaceMac> ();
      if (mac == 0)
        {
          return false;
        }
      // Installing plugins:
      Ptr<FlameProtocolMac> flameMac = Create<FlameProtocolMac> (this);
      m_interfaces[wifiNetDev->GetIfIndex ()] = flameMac;
      mac->SetBeaconGeneration (false);
      mac->InstallPlugin (flameMac);
    }
  mp->SetRoutingProtocol (this);
  // Mesh point aggregates all installed protocols
  mp->AggregateObject (this);
  m_address = Mac48Address::ConvertFrom (mp->GetAddress ()); //* address;
  return true;
}
Mac48Address
FlameProtocol::GetAddress ()
{
  return m_address;
}
bool
FlameProtocol::HandleDataFrame (uint16_t seqno, Mac48Address source, const FlameHeader flameHdr,
                                Mac48Address receiver, uint32_t fromInterface)
{
  if (source == GetAddress ())
    {
      m_stats.totalDropped++;
      return true;
    }
  FlameRtable::LookupResult result = m_rtable->Lookup (source);
  if ((result.retransmitter != Mac48Address::GetBroadcast ()) && ((int16_t)(result.seqnum - seqno) >= 0))
    {
      return true;
    }
  if (flameHdr.GetCost () > m_maxCost)
    {
      m_stats.droppedTtl++;
      return true;
    }
  m_rtable->AddPath (source, receiver, fromInterface, flameHdr.GetCost (), flameHdr.GetSeqno ());
  return false;
}
//Statistics:
FlameProtocol::Statistics::Statistics () :
  txUnicast (0), txBroadcast (0), txBytes (0), droppedTtl (0), totalDropped (0)
{
}
void
FlameProtocol::Statistics::Print (std::ostream & os) const
{
  os << "<Statistics "
  "txUnicast=\"" << txUnicast << "\" "
  "txBroadcast=\"" << txBroadcast << "\" "
  "txBytes=\"" << txBytes << "\" "
  "droppedTtl=\"" << droppedTtl << "\" "
  "totalDropped=\"" << totalDropped << "\"/>" << std::endl;
}
void
FlameProtocol::Report (std::ostream & os) const
{
  os << "<Flame "
  "address=\"" << m_address << "\"" << std::endl <<
  "broadcastInterval=\"" << m_broadcastInterval.GetSeconds () << "\"" << std::endl <<
  "maxCost=\"" << (uint16_t) m_maxCost << "\">" << std::endl;
  m_stats.Print (os);
  for (FlamePluginMap::const_iterator plugin = m_interfaces.begin (); plugin != m_interfaces.end (); plugin++)
    {
      plugin->second->Report (os);
    }
  os << "</Flame>" << std::endl;
}
void
FlameProtocol::ResetStats ()
{
  m_stats = Statistics ();
  for (FlamePluginMap::const_iterator plugin = m_interfaces.begin (); plugin != m_interfaces.end (); plugin++)
    {
      plugin->second->ResetStats ();
    }
}

} // namespace flame
} // namespace ns3
