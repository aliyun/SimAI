/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2009 INESC Porto
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: Gustavo J. A. M. Carneiro  <gjc@inescporto.pt> <gjcarneiro@gmail.com>
//

#include "ns3/flow-probe.h"
#include "ns3/flow-monitor.h"

namespace ns3 {

/* static */
TypeId FlowProbe::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FlowProbe")
    .SetParent<Object> ()
    .SetGroupName ("FlowMonitor")
    // No AddConstructor because this class has no default constructor.
    ;

  return tid;
}

FlowProbe::~FlowProbe ()
{
}


FlowProbe::FlowProbe (Ptr<FlowMonitor> flowMonitor)
  : m_flowMonitor (flowMonitor)
{
  m_flowMonitor->AddProbe (this);
}

void
FlowProbe::DoDispose (void)
{
  m_flowMonitor = 0;
  Object::DoDispose ();
}

void
FlowProbe::AddPacketStats (FlowId flowId, uint32_t packetSize, Time delayFromFirstProbe)
{
  FlowStats &flow = m_stats[flowId];
  flow.delayFromFirstProbeSum += delayFromFirstProbe;
  flow.bytes += packetSize;
  ++flow.packets;
}

void
FlowProbe::AddPacketDropStats (FlowId flowId, uint32_t packetSize, uint32_t reasonCode)
{
  FlowStats &flow = m_stats[flowId];

  if (flow.packetsDropped.size () < reasonCode + 1)
    {
      flow.packetsDropped.resize (reasonCode + 1, 0);
      flow.bytesDropped.resize (reasonCode + 1, 0);
    }
  ++flow.packetsDropped[reasonCode];
  flow.bytesDropped[reasonCode] += packetSize;
}
 
FlowProbe::Stats
FlowProbe::GetStats () const 
{
  return m_stats;
}

void
FlowProbe::SerializeToXmlStream (std::ostream &os, uint16_t indent, uint32_t index) const
{
  os << std::string ( indent, ' ' ) << "<FlowProbe index=\"" << index << "\">\n";

  indent += 2;

  for (Stats::const_iterator iter = m_stats.begin (); iter != m_stats.end (); iter++)
    {
      os << std::string ( indent, ' ' );
      os << "<FlowStats "
         << " flowId=\"" << iter->first << "\""
         << " packets=\"" << iter->second.packets << "\""
         << " bytes=\"" << iter->second.bytes << "\""
         << " delayFromFirstProbeSum=\"" << iter->second.delayFromFirstProbeSum << "\""
         << " >\n";
      indent += 2;
      for (uint32_t reasonCode = 0; reasonCode < iter->second.packetsDropped.size (); reasonCode++)
        {
          os << std::string ( indent, ' ' );
          os << "<packetsDropped reasonCode=\"" << reasonCode << "\""
             << " number=\"" << iter->second.packetsDropped[reasonCode]
             << "\" />\n";
        }
      for (uint32_t reasonCode = 0; reasonCode < iter->second.bytesDropped.size (); reasonCode++)
        {
          os << std::string ( indent, ' ' );
          os << "<bytesDropped reasonCode=\"" << reasonCode << "\""
             << " bytes=\"" << iter->second.bytesDropped[reasonCode]
             << "\" />\n";
        }
      indent -= 2;
      os << std::string ( indent, ' ' ) << "</FlowStats>\n";
    }
  indent -= 2;
  os << std::string ( indent, ' ' ) << "</FlowProbe>\n";
}



} // namespace ns3


