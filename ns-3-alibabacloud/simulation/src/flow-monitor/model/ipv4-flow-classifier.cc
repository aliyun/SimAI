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

#include "ns3/packet.h"

#include "ipv4-flow-classifier.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include <algorithm>

namespace ns3 {

/* see http://www.iana.org/assignments/protocol-numbers */
const uint8_t TCP_PROT_NUMBER = 6;  //!< TCP Protocol number
const uint8_t UDP_PROT_NUMBER = 17; //!< UDP Protocol number



bool operator < (const Ipv4FlowClassifier::FiveTuple &t1,
                 const Ipv4FlowClassifier::FiveTuple &t2)
{
  if (t1.sourceAddress < t2.sourceAddress)
    {
      return true;
    }
  if (t1.sourceAddress != t2.sourceAddress)
    {
      return false;
    }

  if (t1.destinationAddress < t2.destinationAddress)
    {
      return true;
    }
  if (t1.destinationAddress != t2.destinationAddress)
    {
      return false;
    }

  if (t1.protocol < t2.protocol)
    {
      return true;
    }
  if (t1.protocol != t2.protocol)
    {
      return false;
    }

  if (t1.sourcePort < t2.sourcePort)
    {
      return true;
    }
  if (t1.sourcePort != t2.sourcePort)
    {
      return false;
    }

  if (t1.destinationPort < t2.destinationPort)
    {
      return true;
    }
  if (t1.destinationPort != t2.destinationPort)
    {
      return false;
    }

  return false;
}

bool operator == (const Ipv4FlowClassifier::FiveTuple &t1,
                  const Ipv4FlowClassifier::FiveTuple &t2)
{
  return (t1.sourceAddress      == t2.sourceAddress &&
          t1.destinationAddress == t2.destinationAddress &&
          t1.protocol           == t2.protocol &&
          t1.sourcePort         == t2.sourcePort &&
          t1.destinationPort    == t2.destinationPort);
}



Ipv4FlowClassifier::Ipv4FlowClassifier ()
{
#ifdef NS3_MTP
  m_lock.store (false, std::memory_order_relaxed);
#endif
}

bool
Ipv4FlowClassifier::Classify (const Ipv4Header &ipHeader, Ptr<const Packet> ipPayload,
                              uint32_t *out_flowId, uint32_t *out_packetId)
{
  if (ipHeader.GetFragmentOffset () > 0 )
    {
      // Ignore fragments: they don't carry a valid L4 header
      return false;
    }

  FiveTuple tuple;
  tuple.sourceAddress = ipHeader.GetSource ();
  tuple.destinationAddress = ipHeader.GetDestination ();
  tuple.protocol = ipHeader.GetProtocol ();

  if ((tuple.protocol != UDP_PROT_NUMBER) && (tuple.protocol != TCP_PROT_NUMBER))
    {
      return false;
    }

  if (ipPayload->GetSize () < 4)
    {
      // the packet doesn't carry enough bytes
      return false;
    }

  // we rely on the fact that for both TCP and UDP the ports are
  // carried in the first 4 octects.
  // This allows to read the ports even on fragmented packets
  // not carrying a full TCP or UDP header.

  uint8_t data[4];
  ipPayload->CopyData (data, 4);

  uint16_t srcPort = 0;
  srcPort |= data[0];
  srcPort <<= 8;
  srcPort |= data[1];

  uint16_t dstPort = 0;
  dstPort |= data[2];
  dstPort <<= 8;
  dstPort |= data[3];

  tuple.sourcePort = srcPort;
  tuple.destinationPort = dstPort;

#ifdef NS3_MTP
  while (m_lock.exchange (true, std::memory_order_acquire))
    ;
#endif

  // try to insert the tuple, but check if it already exists
  std::pair<std::map<FiveTuple, FlowId>::iterator, bool> insert
    = m_flowMap.insert (std::pair<FiveTuple, FlowId> (tuple, 0));

  // if the insertion succeeded, we need to assign this tuple a new flow identifier
  if (insert.second)
    {
      FlowId newFlowId = GetNewFlowId ();
      insert.first->second = newFlowId;
      m_flowPktIdMap[newFlowId] = 0;
      m_flowDscpMap[newFlowId];
    }
  else
    {
      m_flowPktIdMap[insert.first->second] ++;
    }

  // increment the counter of packets with the same DSCP value
  Ipv4Header::DscpType dscp = ipHeader.GetDscp ();
  std::pair<std::map<Ipv4Header::DscpType, uint32_t>::iterator, bool> dscpInserter
    = m_flowDscpMap[insert.first->second].insert (std::pair<Ipv4Header::DscpType, uint32_t> (dscp, 1));

  // if the insertion did not succeed, we need to increment the counter
  if (!dscpInserter.second)
    {
      m_flowDscpMap[insert.first->second][dscp] ++;
    }

  *out_flowId = insert.first->second;
  *out_packetId = m_flowPktIdMap[*out_flowId];

#ifdef NS3_MTP
  m_lock.store (false, std::memory_order_release);
#endif

  return true;
}


Ipv4FlowClassifier::FiveTuple
Ipv4FlowClassifier::FindFlow (FlowId flowId) const
{
  for (std::map<FiveTuple, FlowId>::const_iterator
       iter = m_flowMap.begin (); iter != m_flowMap.end (); iter++)
    {
      if (iter->second == flowId)
        {
          return iter->first;
        }
    }
  NS_FATAL_ERROR ("Could not find the flow with ID " << flowId);
  FiveTuple retval = { Ipv4Address::GetZero (), Ipv4Address::GetZero (), 0, 0, 0 };
  return retval;
}

bool
Ipv4FlowClassifier::SortByCount::operator() (std::pair<Ipv4Header::DscpType, uint32_t> left,
                                             std::pair<Ipv4Header::DscpType, uint32_t> right)
{
  return left.second > right.second;
}

std::vector<std::pair<Ipv4Header::DscpType, uint32_t> >
Ipv4FlowClassifier::GetDscpCounts (FlowId flowId) const
{
  std::map<FlowId, std::map<Ipv4Header::DscpType, uint32_t> >::const_iterator flow
    = m_flowDscpMap.find (flowId);

  if (flow == m_flowDscpMap.end ())
    {
      NS_FATAL_ERROR ("Could not find the flow with ID " << flowId);
    }

  std::vector<std::pair<Ipv4Header::DscpType, uint32_t> > v (flow->second.begin (), flow->second.end ());
  std::sort (v.begin (), v.end (), SortByCount ());
  return v;
}

void
Ipv4FlowClassifier::SerializeToXmlStream (std::ostream &os, uint16_t indent) const
{
  Indent (os, indent); os << "<Ipv4FlowClassifier>\n";

  indent += 2;
  for (std::map<FiveTuple, FlowId>::const_iterator
       iter = m_flowMap.begin (); iter != m_flowMap.end (); iter++)
    {
      Indent (os, indent);
      os << "<Flow flowId=\"" << iter->second << "\""
         << " sourceAddress=\"" << iter->first.sourceAddress << "\""
         << " destinationAddress=\"" << iter->first.destinationAddress << "\""
         << " protocol=\"" << int(iter->first.protocol) << "\""
         << " sourcePort=\"" << iter->first.sourcePort << "\""
         << " destinationPort=\"" << iter->first.destinationPort << "\">\n";

      indent += 2;
      std::map<FlowId, std::map<Ipv4Header::DscpType, uint32_t> >::const_iterator flow
        = m_flowDscpMap.find (iter->second);

      if (flow != m_flowDscpMap.end ())
        {
          for (std::map<Ipv4Header::DscpType, uint32_t>::const_iterator i = flow->second.begin (); i != flow->second.end (); i++)
            {
              Indent (os, indent);
              os << "<Dscp value=\"0x" << std::hex << static_cast<uint32_t> (i->first) << "\""
                 << " packets=\"" << std::dec << i->second << "\" />\n";
            }
        }

      indent -= 2;
      Indent (os, indent); os << "</Flow>\n";
    }

  indent -= 2;
  Indent (os, indent); os << "</Ipv4FlowClassifier>\n";
}


} // namespace ns3

