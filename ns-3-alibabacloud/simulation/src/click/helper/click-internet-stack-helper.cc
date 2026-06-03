/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 * Author: Faker Moatamri <faker.moatamri@sophia.inria.fr>
 * Author: Lalith Suresh  <suresh.lalith@gmail.com>
 */

#ifdef NS3_CLICK

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/names.h"
#include "ns3/ipv4.h"
#include "ns3/packet-socket-factory.h"
#include "ns3/config.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/net-device.h"
#include "ns3/callback.h"
#include "ns3/node.h"
#include "ns3/core-config.h"
#include "ns3/arp-l3-protocol.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/ipv4-l3-click-protocol.h"
#include "ns3/trace-helper.h"
#include "click-internet-stack-helper.h"
#include <limits>
#include <map>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ClickInternetStackHelper");

#define INTERFACE_CONTEXT

typedef std::pair<Ptr<Ipv4>, uint32_t> InterfacePairIpv4;
typedef std::map<InterfacePairIpv4, Ptr<PcapFileWrapper> > InterfaceFileMapIpv4;
typedef std::map<InterfacePairIpv4, Ptr<OutputStreamWrapper> > InterfaceStreamMapIpv4;

static InterfaceFileMapIpv4 g_interfaceFileMapIpv4; /**< A mapping of Ipv4/interface pairs to pcap files */
static InterfaceStreamMapIpv4 g_interfaceStreamMapIpv4; /**< A mapping of Ipv4/interface pairs to ascii streams */

ClickInternetStackHelper::ClickInternetStackHelper ()
  : m_ipv4Enabled (true)
{
  Initialize ();
}

// private method called by both constructor and Reset ()
void
ClickInternetStackHelper::Initialize ()
{
  SetTcp ("ns3::TcpL4Protocol");
}

ClickInternetStackHelper::~ClickInternetStackHelper ()
{
}

ClickInternetStackHelper::ClickInternetStackHelper (const ClickInternetStackHelper &o)
{
  m_ipv4Enabled = o.m_ipv4Enabled;
  m_tcpFactory = o.m_tcpFactory;
}

ClickInternetStackHelper &
ClickInternetStackHelper::operator = (const ClickInternetStackHelper &o)
{
  if (this == &o)
    {
      return *this;
    }
  return *this;
}

void
ClickInternetStackHelper::Reset (void)
{
  m_ipv4Enabled = true;
  Initialize ();
}

void
ClickInternetStackHelper::SetTcp (const std::string tid)
{
  m_tcpFactory.SetTypeId (tid);
}

void
ClickInternetStackHelper::SetTcp (std::string tid, std::string n0, const AttributeValue &v0)
{
  m_tcpFactory.SetTypeId (tid);
  m_tcpFactory.Set (n0,v0);
}

void
ClickInternetStackHelper::SetClickFile (NodeContainer c, std::string clickfile)
{
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      SetClickFile (*i, clickfile);
    }
}

void
ClickInternetStackHelper::SetClickFile (Ptr<Node> node, std::string clickfile)
{
  m_nodeToClickFileMap.insert (std::make_pair (node, clickfile));
}

void
ClickInternetStackHelper::SetDefines (NodeContainer c, std::map<std::string, std::string> defines)
{
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
	  SetDefines (*i, defines);
    }
}

void
ClickInternetStackHelper::SetDefines (Ptr<Node> node, std::map<std::string, std::string> defines)
{
  m_nodeToDefinesMap.insert (std::make_pair (node, defines));
}

void
ClickInternetStackHelper::SetRoutingTableElement (NodeContainer c, std::string rt)
{
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      SetRoutingTableElement (*i, rt);
    }
}

void
ClickInternetStackHelper::SetRoutingTableElement (Ptr<Node> node, std::string rt)
{
  m_nodeToRoutingTableElementMap.insert (std::make_pair (node, rt));
}

void
ClickInternetStackHelper::Install (NodeContainer c) const
{
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Install (*i);
    }
}

void
ClickInternetStackHelper::InstallAll (void) const
{
  Install (NodeContainer::GetGlobal ());
}

void
ClickInternetStackHelper::CreateAndAggregateObjectFromTypeId (Ptr<Node> node, const std::string typeId)
{
  ObjectFactory factory;
  factory.SetTypeId (typeId);
  Ptr<Object> protocol = factory.Create <Object> ();
  node->AggregateObject (protocol);
}

void
ClickInternetStackHelper::Install (Ptr<Node> node) const
{
  if (m_ipv4Enabled)
    {
      if (node->GetObject<Ipv4> () != 0)
        {
          NS_FATAL_ERROR ("ClickInternetStackHelper::Install (): Aggregating "
                          "an InternetStack to a node with an existing Ipv4 object");
          return;
        }

      CreateAndAggregateObjectFromTypeId (node, "ns3::ArpL3Protocol");
      CreateAndAggregateObjectFromTypeId (node, "ns3::Ipv4L3ClickProtocol");
      CreateAndAggregateObjectFromTypeId (node, "ns3::Icmpv4L4Protocol");
      CreateAndAggregateObjectFromTypeId (node, "ns3::UdpL4Protocol");
      node->AggregateObject (m_tcpFactory.Create<Object> ());
      Ptr<PacketSocketFactory> factory = CreateObject<PacketSocketFactory> ();
      node->AggregateObject (factory);
      // Set routing
      Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
      Ptr<Ipv4ClickRouting> ipv4Routing = CreateObject<Ipv4ClickRouting> ();
      std::map< Ptr<Node>, std::string >::const_iterator it;
      it = m_nodeToClickFileMap.find (node);

      if (it != m_nodeToClickFileMap.end ())
        {
          ipv4Routing->SetClickFile (it->second);
        }

      std::map<Ptr<Node>, std::map<std::string, std::string> >::const_iterator definesIt;
      definesIt = m_nodeToDefinesMap.find (node);
      if (definesIt != m_nodeToDefinesMap.end ())
        {
          ipv4Routing->SetDefines (definesIt->second);
        }

      it = m_nodeToRoutingTableElementMap.find (node);
      if (it != m_nodeToRoutingTableElementMap.end ())
        {
          ipv4Routing->SetClickRoutingTableElement (it->second);
        }
      ipv4->SetRoutingProtocol (ipv4Routing);
      node->AggregateObject (ipv4Routing);
    }
}

void
ClickInternetStackHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  Install (node);
}

static void
Ipv4L3ProtocolRxTxSink (Ptr<const Packet> p, Ptr<Ipv4> ipv4, uint32_t interface)
{
  NS_LOG_FUNCTION (p << ipv4 << interface);

  //
  // Since trace sources are independent of interface, if we hook a source
  // on a particular protocol we will get traces for all of its interfaces.
  // We need to filter this to only report interfaces for which the user
  // has expressed interest.
  //
  InterfacePairIpv4 pair = std::make_pair (ipv4, interface);
  if (g_interfaceFileMapIpv4.find (pair) == g_interfaceFileMapIpv4.end ())
    {
      NS_LOG_INFO ("Ignoring packet to/from interface " << interface);
      return;
    }

  Ptr<PcapFileWrapper> file = g_interfaceFileMapIpv4[pair];
  file->Write (Simulator::Now (), p);
}

bool
ClickInternetStackHelper::PcapHooked (Ptr<Ipv4> ipv4)
{
  for (  InterfaceFileMapIpv4::const_iterator i = g_interfaceFileMapIpv4.begin ();
         i != g_interfaceFileMapIpv4.end ();
         ++i)
    {
      if ((*i).first.first == ipv4)
        {
          return true;
        }
    }
  return false;
}

void
ClickInternetStackHelper::EnablePcapIpv4Internal (std::string prefix, Ptr<Ipv4> ipv4, uint32_t interface, bool explicitFilename)
{
  NS_LOG_FUNCTION (prefix << ipv4 << interface);

  if (!m_ipv4Enabled)
    {
      NS_LOG_INFO ("Call to enable Ipv4 pcap tracing but Ipv4 not enabled");
      return;
    }

  //
  // We have to create a file and a mapping from protocol/interface to file
  // irrespective of how many times we want to trace a particular protocol.
  //
  PcapHelper pcapHelper;

  std::string filename;
  if (explicitFilename)
    {
      filename = prefix;
    }
  else
    {
      filename = pcapHelper.GetFilenameFromInterfacePair (prefix, ipv4, interface);
    }

  Ptr<PcapFileWrapper> file = pcapHelper.CreateFile (filename, std::ios::out, PcapHelper::DLT_RAW);

  //
  // However, we only hook the trace source once to avoid multiple trace sink
  // calls per event (connect is independent of interface).
  //
  if (!PcapHooked (ipv4))
    {
      //
      // Ptr<Ipv4> is aggregated to node and Ipv4L3Protocol is aggregated to
      // node so we can get to Ipv4L3Protocol through Ipv4.
      //
      Ptr<Ipv4L3Protocol> ipv4L3Protocol = ipv4->GetObject<Ipv4L3Protocol> ();
      NS_ASSERT_MSG (ipv4L3Protocol, "ClickInternetStackHelper::EnablePcapIpv4Internal(): "
                     "m_ipv4Enabled and ipv4L3Protocol inconsistent");

      bool result = ipv4L3Protocol->TraceConnectWithoutContext ("Tx", MakeCallback (&Ipv4L3ProtocolRxTxSink));
      NS_ASSERT_MSG (result == true, "ClickInternetStackHelper::EnablePcapIpv4Internal():  "
                     "Unable to connect ipv4L3Protocol \"Tx\"");

      result = ipv4L3Protocol->TraceConnectWithoutContext ("Rx", MakeCallback (&Ipv4L3ProtocolRxTxSink));
      NS_ASSERT_MSG (result == true, "ClickInternetStackHelper::EnablePcapIpv4Internal():  "
                     "Unable to connect ipv4L3Protocol \"Rx\"");
    }

  g_interfaceFileMapIpv4[std::make_pair (ipv4, interface)] = file;
}

static void
Ipv4L3ProtocolDropSinkWithoutContext (
  Ptr<OutputStreamWrapper> stream,
  Ipv4Header const &header,
  Ptr<const Packet> packet,
  Ipv4L3Protocol::DropReason reason,
  Ptr<Ipv4> ipv4,
  uint32_t interface)
{
  //
  // Since trace sources are independent of interface, if we hook a source
  // on a particular protocol we will get traces for all of its interfaces.
  // We need to filter this to only report interfaces for which the user
  // has expressed interest.
  //
  InterfacePairIpv4 pair = std::make_pair (ipv4, interface);
  if (g_interfaceStreamMapIpv4.find (pair) == g_interfaceStreamMapIpv4.end ())
    {
      NS_LOG_INFO ("Ignoring packet to/from interface " << interface);
      return;
    }

  Ptr<Packet> p = packet->Copy ();
  p->AddHeader (header);
  *stream->GetStream () << "d " << Simulator::Now ().GetSeconds () << " " << *p << std::endl;
}

static void
Ipv4L3ProtocolDropSinkWithContext (
  Ptr<OutputStreamWrapper> stream,
  std::string context,
  Ipv4Header const &header,
  Ptr<const Packet> packet,
  Ipv4L3Protocol::DropReason reason,
  Ptr<Ipv4> ipv4,
  uint32_t interface)
{
  //
  // Since trace sources are independent of interface, if we hook a source
  // on a particular protocol we will get traces for all of its interfaces.
  // We need to filter this to only report interfaces for which the user
  // has expressed interest.
  //
  InterfacePairIpv4 pair = std::make_pair (ipv4, interface);
  if (g_interfaceStreamMapIpv4.find (pair) == g_interfaceStreamMapIpv4.end ())
    {
      NS_LOG_INFO ("Ignoring packet to/from interface " << interface);
      return;
    }

  Ptr<Packet> p = packet->Copy ();
  p->AddHeader (header);
#ifdef INTERFACE_CONTEXT
  *stream->GetStream () << "d " << Simulator::Now ().GetSeconds () << " " << context << "(" << interface << ") "
                        << *p << std::endl;
#else
  *stream->GetStream () << "d " << Simulator::Now ().GetSeconds () << " " << context << " "  << *p << std::endl;
#endif
}

bool
ClickInternetStackHelper::AsciiHooked (Ptr<Ipv4> ipv4)
{
  for (  InterfaceStreamMapIpv4::const_iterator i = g_interfaceStreamMapIpv4.begin ();
         i != g_interfaceStreamMapIpv4.end ();
         ++i)
    {
      if ((*i).first.first == ipv4)
        {
          return true;
        }
    }
  return false;
}

void
ClickInternetStackHelper::EnableAsciiIpv4Internal (
  Ptr<OutputStreamWrapper> stream,
  std::string prefix,
  Ptr<Ipv4> ipv4,
  uint32_t interface,
  bool explicitFilename)
{
  if (!m_ipv4Enabled)
    {
      NS_LOG_INFO ("Call to enable Ipv4 ascii tracing but Ipv4 not enabled");
      return;
    }

  //
  // Our trace sinks are going to use packet printing, so we have to
  // make sure that is turned on.
  //
  Packet::EnablePrinting ();

  //
  // If we are not provided an OutputStreamWrapper, we are expected to create
  // one using the usual trace filename conventions and hook WithoutContext
  // since there will be one file per context and therefore the context would
  // be redundant.
  //
  if (stream == 0)
    {
      //
      // Set up an output stream object to deal with private ofstream copy
      // constructor and lifetime issues.  Let the helper decide the actual
      // name of the file given the prefix.
      //
      // We have to create a stream and a mapping from protocol/interface to
      // stream irrespective of how many times we want to trace a particular
      // protocol.
      //
      AsciiTraceHelper asciiTraceHelper;

      std::string filename;
      if (explicitFilename)
        {
          filename = prefix;
        }
      else
        {
          filename = asciiTraceHelper.GetFilenameFromInterfacePair (prefix, ipv4, interface);
        }

      Ptr<OutputStreamWrapper> theStream = asciiTraceHelper.CreateFileStream (filename);

      //
      // However, we only hook the trace sources once to avoid multiple trace sink
      // calls per event (connect is independent of interface).
      //
      if (!AsciiHooked (ipv4))
        {
          //
          // We can use the default drop sink for the ArpL3Protocol since it has
          // the usual signature.  We can get to the Ptr<ArpL3Protocol> through
          // our Ptr<Ipv4> since they must both be aggregated to the same node.
          //
          Ptr<ArpL3Protocol> arpL3Protocol = ipv4->GetObject<ArpL3Protocol> ();
          asciiTraceHelper.HookDefaultDropSinkWithoutContext<ArpL3Protocol> (arpL3Protocol, "Drop", theStream);

          //
          // The drop sink for the Ipv4L3Protocol uses a different signature than
          // the default sink, so we have to cook one up for ourselves.  We can get
          // to the Ptr<Ipv4L3Protocol> through our Ptr<Ipv4> since they must both
          // be aggregated to the same node.
          //
          Ptr<Ipv4L3Protocol> ipv4L3Protocol = ipv4->GetObject<Ipv4L3Protocol> ();
          bool result = ipv4L3Protocol->TraceConnectWithoutContext ("Drop",
                                                                    MakeBoundCallback (&Ipv4L3ProtocolDropSinkWithoutContext,
                                                                                       theStream));
          NS_ASSERT_MSG (result == true, "ClickInternetStackHelper::EanableAsciiIpv4Internal():  "
                         "Unable to connect ipv4L3Protocol \"Drop\"");
        }

      g_interfaceStreamMapIpv4[std::make_pair (ipv4, interface)] = theStream;
      return;
    }

  //
  // If we are provided an OutputStreamWrapper, we are expected to use it, and
  // to provide a context.  We are free to come up with our own context if we
  // want, and use the AsciiTraceHelper Hook*WithContext functions, but for
  // compatibility and simplicity, we just use Config::Connect and let it deal
  // with the context.
  //
  // We need to associate the ipv4/interface with a stream to express interest
  // in tracing events on that pair, however, we only hook the trace sources
  // once to avoid multiple trace sink calls per event (connect is independent
  // of interface).
  //
  if (!AsciiHooked (ipv4))
    {
      Ptr<Node> node = ipv4->GetObject<Node> ();
      std::ostringstream oss;

      //
      // For the ARP Drop, we are going to use the default trace sink provided by
      // the ascii trace helper.  There is actually no AsciiTraceHelper in sight
      // here, but the default trace sinks are actually publicly available static
      // functions that are always there waiting for just such a case.
      //
      oss << "/NodeList/" << node->GetId () << "/$ns3::ArpL3Protocol/Drop";
      Config::Connect (oss.str (), MakeBoundCallback (&AsciiTraceHelper::DefaultDropSinkWithContext, stream));

      //
      // This has all kinds of parameters coming with, so we have to cook up our
      // own sink.
      //
      oss.str ("");
      oss << "/NodeList/" << node->GetId () << "/$ns3::Ipv4L3Protocol/Drop";
      Config::Connect (oss.str (), MakeBoundCallback (&Ipv4L3ProtocolDropSinkWithContext, stream));
    }

  g_interfaceStreamMapIpv4[std::make_pair (ipv4, interface)] = stream;
}

} // namespace ns3

#endif // NS3_CLICK
