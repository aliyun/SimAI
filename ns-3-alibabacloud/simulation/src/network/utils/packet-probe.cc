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
 */

#include "ns3/packet-probe.h"
#include "ns3/object.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketProbe");

NS_OBJECT_ENSURE_REGISTERED (PacketProbe);

TypeId
PacketProbe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::PacketProbe")
    .SetParent<Probe> ()
    .SetGroupName("Network")
    .AddConstructor<PacketProbe> ()
    .AddTraceSource ( "Output",
                      "The packet that serve as the output for this probe",
                      MakeTraceSourceAccessor (&PacketProbe::m_output),
                      "ns3::Packet::TracedCallback")
    .AddTraceSource ( "OutputBytes",
                      "The number of bytes in the packet",
                      MakeTraceSourceAccessor (&PacketProbe::m_outputBytes),
                      "ns3::Packet::SizeTracedCallback")
  ;
  return tid;
}

PacketProbe::PacketProbe ()
{
  NS_LOG_FUNCTION (this);
  m_packet = 0;
}

PacketProbe::~PacketProbe ()
{
  NS_LOG_FUNCTION (this);
}

void
PacketProbe::SetValue (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this << packet);
  m_packet = packet;
  m_output (packet);

  uint32_t packetSizeNew = packet->GetSize ();
  m_outputBytes (m_packetSizeOld, packetSizeNew);
  m_packetSizeOld = packetSizeNew;
}

void
PacketProbe::SetValueByPath (std::string path, Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (path << packet);
  Ptr<PacketProbe> probe = Names::Find<PacketProbe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (packet);
}

bool
PacketProbe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of probe (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::PacketProbe::TraceSink, this));
  return connected;
}

void
PacketProbe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of probe to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::PacketProbe::TraceSink, this));
}

void
PacketProbe::TraceSink (Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (this << packet);
  if (IsEnabled ())
    {
      m_packet = packet;
      m_output (packet);

      uint32_t packetSizeNew = packet->GetSize ();
      m_outputBytes (m_packetSizeOld, packetSizeNew);
      m_packetSizeOld = packetSizeNew;
    }
}

} // namespace ns3
