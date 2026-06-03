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

#include "ns3/application-packet-probe.h"
#include "ns3/object.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/config.h"
#include "ns3/trace-source-accessor.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ApplicationPacketProbe");

NS_OBJECT_ENSURE_REGISTERED (ApplicationPacketProbe);

TypeId
ApplicationPacketProbe::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::ApplicationPacketProbe")
    .SetParent<Probe> ()
    .SetGroupName("Applications")
    .AddConstructor<ApplicationPacketProbe> ()
    .AddTraceSource ( "Output",
                      "The packet plus its socket address that serve "
                      "as the output for this probe",
                      MakeTraceSourceAccessor (&ApplicationPacketProbe::m_output),
                      "ns3::Packet::AddressTracedCallback")
    .AddTraceSource ( "OutputBytes",
                      "The number of bytes in the packet",
                      MakeTraceSourceAccessor (&ApplicationPacketProbe::m_outputBytes),
                      "ns3::Packet::SizeTracedCallback")
  ;
  return tid;
}

ApplicationPacketProbe::ApplicationPacketProbe ()
{
  NS_LOG_FUNCTION (this);
  m_packet = 0;
}

ApplicationPacketProbe::~ApplicationPacketProbe ()
{
  NS_LOG_FUNCTION (this);
}

void
ApplicationPacketProbe::SetValue (Ptr<const Packet> packet, const Address& address)
{
  NS_LOG_FUNCTION (this << packet << address);
  m_packet  = packet;
  m_address = address;
  m_output (packet, address);

  uint32_t packetSizeNew = packet->GetSize ();
  m_outputBytes (m_packetSizeOld, packetSizeNew);
  m_packetSizeOld = packetSizeNew;
}

void
ApplicationPacketProbe::SetValueByPath (std::string path, Ptr<const Packet> packet, const Address& address)
{
  NS_LOG_FUNCTION (path << packet << address);
  Ptr<ApplicationPacketProbe> probe = Names::Find<ApplicationPacketProbe> (path);
  NS_ASSERT_MSG (probe, "Error:  Can't find probe for path " << path);
  probe->SetValue (packet, address);
}

bool
ApplicationPacketProbe::ConnectByObject (std::string traceSource, Ptr<Object> obj)
{
  NS_LOG_FUNCTION (this << traceSource << obj);
  NS_LOG_DEBUG ("Name of probe (if any) in names database: " << Names::FindPath (obj));
  bool connected = obj->TraceConnectWithoutContext (traceSource, MakeCallback (&ns3::ApplicationPacketProbe::TraceSink, this));
  return connected;
}

void
ApplicationPacketProbe::ConnectByPath (std::string path)
{
  NS_LOG_FUNCTION (this << path);
  NS_LOG_DEBUG ("Name of probe to search for in config database: " << path);
  Config::ConnectWithoutContext (path, MakeCallback (&ns3::ApplicationPacketProbe::TraceSink, this));
}

void
ApplicationPacketProbe::TraceSink (Ptr<const Packet> packet, const Address& address)
{
  NS_LOG_FUNCTION (this << packet << address);
  if (IsEnabled ())
    {
      m_packet  = packet;
      m_address = address;
      m_output (packet, address);

      uint32_t packetSizeNew = packet->GetSize ();
      m_outputBytes (m_packetSizeOld, packetSizeNew);
      m_packetSizeOld = packetSizeNew;
    }
}

} // namespace ns3
