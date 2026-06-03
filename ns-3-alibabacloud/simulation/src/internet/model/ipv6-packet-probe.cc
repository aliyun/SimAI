/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Bucknell University
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
 * Authors: L. Felipe Perrone (perrone@bucknell.edu)
 *          Tiago G. Rodrigues (tgr002@bucknell.edu)
 *
 * Modified by: Mitch Watrous (watrous@u.washington.edu)
 * Adapted to Ipv6 by: Tommaso Pecorella (tommaso.pecorella@unifi.it)
 */

#include "ns3/ipv6-packet-probe.h"
#include "ns3/object.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6PacketProbe");

NS_OBJECT_ENSURE_REGISTERED (Ipv6PacketProbe);

TypeId
Ipv6PacketProbe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6PacketProbe")
    .SetParent<Probe> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv6PacketProbe> ()
    .AddTraceSource ( "Output",
                      "The packet plus its IPv6 object and interface "
                      "that serve as the output for this probe",
                      MakeTraceSourceAccessor (&Ipv6PacketProbe::m_output),
                      "ns3::Ipv6L3Protocol::TxRxTracedCallback")
    .AddTraceSource ( "OutputBytes",
                      "The number of bytes in the packet",
                      MakeTraceSourceAccessor (&Ipv6PacketProbe::m_outputBytes),
                      "ns3::Packet::SizeTracedCallback")
  ;
  return tid;
}

Ipv6PacketProbe::Ipv6PacketProbe ()
{
  NS_LOG_FUNCTION (this);
  m_packet    = 0;
  m_packetSizeOld = 0;
  m_ipv6      = 0;
  m_interface = 0;
}

Ipv6PacketProbe::~Ipv6PacketProbe ()
{
  NS_LOG_FUNCTION (this);
}

void
Ipv6PacketProbe::SetValue (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface)
{
  NS_LOG_FUNCTION (this << packet << ipv6 << interface);
  m_packet    = packet;
  m_ipv6      = ipv6;
  m_interface = interface;
  m_output (packet, ipv6, interface);

  uint32_t packetSizeNew = packet->GetSize ();
  m_outputBytes (m_packetSizeOld, packetSizeNew);
  m_packetSizeOld = packetSizeNew;
}

void
Ipv6PacketProbe::SetValueByPath (std::string path, Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface)
{
  NS_LOG_FUNCTION (path << packet << ipv6 << interface);
  Ptr<Ipv6PacketProbe> probe = Names::Find<Ipv6PacketProbe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (packet, ipv6, interface);
}

bool
Ipv6PacketProbe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of probe (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::Ipv6PacketProbe::TraceSink, this));
  return connected;
}

void
Ipv6PacketProbe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of probe to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::Ipv6PacketProbe::TraceSink, this));
}

void
Ipv6PacketProbe::TraceSink (Ptr<const Packet> packet, Ptr<Ipv6> ipv6, uint32_t interface)
{
  NS_LOG_FUNCTION (this << packet << ipv6 << interface);
  if (IsEnabled ())
    {
      m_packet    = packet;
      m_ipv6      = ipv6;
      m_interface = interface;
      m_output (packet, ipv6, interface);

      uint32_t packetSizeNew = packet->GetSize ();
      m_outputBytes (m_packetSizeOld, packetSizeNew);
      m_packetSizeOld = packetSizeNew;
    }
}

} // namespace ns3
