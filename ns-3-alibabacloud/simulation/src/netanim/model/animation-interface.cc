/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: George F. Riley<riley@ece.gatech.edu>
 * Modified by: John Abraham <john.abraham@gatech.edu>
 * Contributions: Eugene Kalishenko <ydginster@gmail.com> (Open Source and Linux Laboratory http://dev.osll.ru/)
 *                Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *                Pavel Vasilyev <pavel.vasilyev@sredasolutions.com>
 */

// Interface between ns-3 and the network animator



#include <cstdio>
#ifndef WIN32
#include <unistd.h>
#endif
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <map>

// ns3 includes
#include "ns3/animation-interface.h"
#include "ns3/channel.h"
#include "ns3/config.h"
#include "ns3/node.h"
#include "ns3/mobility-model.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wimax-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-psdu.h"
#include "ns3/lr-wpan-mac-header.h"
#include "ns3/lr-wpan-net-device.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/lte-ue-phy.h"
#include "ns3/lte-enb-phy.h"
#include "ns3/uan-net-device.h"
#include "ns3/uan-mac.h"
#include "ns3/double.h"
#include "ns3/ipv4.h"
#include "ns3/ipv6.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/energy-source-container.h"
#include "animation-interface.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AnimationInterface");

// Globals

static bool initialized = false; //!< Initialization flag


// Public methods

AnimationInterface::AnimationInterface (const std::string fn)
  : m_f (0),
    m_routingF (0),
    m_mobilityPollInterval (Seconds (0.25)),
    m_outputFileName (fn),
    gAnimUid (0),
    m_writeCallback (0),
    m_started (false),
    m_enablePacketMetadata (false),
    m_startTime (Seconds (0)),
    m_stopTime (Seconds (3600 * 1000)),
    m_maxPktsPerFile (MAX_PKTS_PER_TRACE_FILE),
    m_originalFileName (fn),
    m_routingStopTime (Seconds (0)),
    m_routingFileName (""),
    m_routingPollInterval (Seconds (5)),
    m_trackPackets (true)
{
  initialized = true;
  StartAnimation ();
}

AnimationInterface::~AnimationInterface ()
{
  StopAnimation ();
}

void
AnimationInterface::SkipPacketTracing ()
{
  m_trackPackets = false;
}

void
AnimationInterface::EnableWifiPhyCounters (Time startTime, Time stopTime, Time pollInterval)
{
  m_wifiPhyCountersStopTime = stopTime;
  m_wifiPhyCountersPollInterval = pollInterval;
  m_wifiPhyTxDropCounterId = AddNodeCounter ("WifiPhy TxDrop", AnimationInterface::DOUBLE_COUNTER);
  m_wifiPhyRxDropCounterId = AddNodeCounter ("WifiPhy RxDrop", AnimationInterface::DOUBLE_COUNTER);
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      m_nodeWifiPhyTxDrop[n->GetId ()] = 0;
      m_nodeWifiPhyRxDrop[n->GetId ()] = 0;
      UpdateNodeCounter (m_wifiPhyTxDropCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_wifiPhyRxDropCounterId, n->GetId (), 0);
    }
  Simulator::Schedule (startTime, &AnimationInterface::TrackWifiPhyCounters, this);

}

void
AnimationInterface::EnableWifiMacCounters (Time startTime, Time stopTime, Time pollInterval)
{
  m_wifiMacCountersStopTime = stopTime;
  m_wifiMacCountersPollInterval = pollInterval;
  m_wifiMacTxCounterId = AddNodeCounter ("WifiMac Tx", AnimationInterface::DOUBLE_COUNTER);
  m_wifiMacTxDropCounterId = AddNodeCounter ("WifiMac TxDrop", AnimationInterface::DOUBLE_COUNTER);
  m_wifiMacRxCounterId = AddNodeCounter ("WifiMac Rx", AnimationInterface::DOUBLE_COUNTER);
  m_wifiMacRxDropCounterId = AddNodeCounter ("WifiMac RxDrop", AnimationInterface::DOUBLE_COUNTER);
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      m_nodeWifiMacTx[n->GetId ()] = 0;
      m_nodeWifiMacTxDrop[n->GetId ()] = 0;
      m_nodeWifiMacRx[n->GetId ()] = 0;
      m_nodeWifiMacRxDrop[n->GetId ()] = 0;
      UpdateNodeCounter (m_wifiMacTxCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_wifiMacTxDropCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_wifiMacRxCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_wifiMacRxDropCounterId, n->GetId (), 0);
    }
  Simulator::Schedule (startTime, &AnimationInterface::TrackWifiMacCounters, this);
}

void
AnimationInterface::EnableQueueCounters (Time startTime, Time stopTime, Time pollInterval)
{
  m_queueCountersStopTime = stopTime;
  m_queueCountersPollInterval = pollInterval;
  m_queueEnqueueCounterId = AddNodeCounter ("Enqueue", AnimationInterface::DOUBLE_COUNTER);
  m_queueDequeueCounterId = AddNodeCounter ("Dequeue", AnimationInterface::DOUBLE_COUNTER);
  m_queueDropCounterId = AddNodeCounter ("Queue Drop", AnimationInterface::DOUBLE_COUNTER);
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      m_nodeQueueEnqueue[n->GetId ()] = 0;
      m_nodeQueueDequeue[n->GetId ()] = 0;
      m_nodeQueueDrop[n->GetId ()] = 0;
      UpdateNodeCounter (m_queueEnqueueCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_queueDequeueCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_queueDropCounterId, n->GetId (), 0);
    }
  Simulator::Schedule (startTime, &AnimationInterface::TrackQueueCounters, this);
}

void
AnimationInterface::EnableIpv4L3ProtocolCounters (Time startTime, Time stopTime, Time pollInterval)
{
  m_ipv4L3ProtocolCountersStopTime = stopTime;
  m_ipv4L3ProtocolCountersPollInterval = pollInterval;
  m_ipv4L3ProtocolTxCounterId = AddNodeCounter ("Ipv4 Tx", AnimationInterface::DOUBLE_COUNTER);
  m_ipv4L3ProtocolRxCounterId = AddNodeCounter ("Ipv4 Rx", AnimationInterface::DOUBLE_COUNTER);
  m_ipv4L3ProtocolDropCounterId = AddNodeCounter ("Ipv4 Drop", AnimationInterface::DOUBLE_COUNTER);
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      m_nodeIpv4Tx[n->GetId ()] = 0;
      m_nodeIpv4Rx[n->GetId ()] = 0;
      m_nodeIpv4Drop[n->GetId ()] = 0;
      UpdateNodeCounter (m_ipv4L3ProtocolTxCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_ipv4L3ProtocolRxCounterId, n->GetId (), 0);
      UpdateNodeCounter (m_ipv4L3ProtocolDropCounterId, n->GetId (), 0);
    }
  Simulator::Schedule (startTime, &AnimationInterface::TrackIpv4L3ProtocolCounters, this);
}

AnimationInterface &
AnimationInterface::EnableIpv4RouteTracking (std::string fileName, Time startTime, Time stopTime, Time pollInterval)
{
  SetOutputFile (fileName, true);
  m_routingStopTime = stopTime;
  m_routingPollInterval = pollInterval;
  WriteXmlAnim (true);
  Simulator::Schedule (startTime, &AnimationInterface::TrackIpv4Route, this);
  return *this;
}

AnimationInterface &
AnimationInterface::EnableIpv4RouteTracking (std::string fileName, Time startTime, Time stopTime, NodeContainer nc, Time pollInterval)
{
  m_routingNc = nc;
  return EnableIpv4RouteTracking (fileName, startTime, stopTime, pollInterval);
}

AnimationInterface &
AnimationInterface::AddSourceDestination (uint32_t fromNodeId, std::string ipv4Address)
{
  Ipv4RouteTrackElement element = { ipv4Address, fromNodeId };
  m_ipv4RouteTrackElements.push_back (element);
  return *this;
}

void
AnimationInterface::SetStartTime (Time t)
{
  m_startTime = t;
}

void
AnimationInterface::SetStopTime (Time t)
{
  m_stopTime = t;
}

void
AnimationInterface::SetMaxPktsPerTraceFile (uint64_t maxPacketsPerFile)
{
  m_maxPktsPerFile = maxPacketsPerFile;
}

uint32_t
AnimationInterface::AddNodeCounter (std::string counterName, CounterType counterType)
{
  m_nodeCounters.push_back (counterName);
  uint32_t counterId = m_nodeCounters.size () - 1; // counter ID is zero-indexed
  WriteXmlAddNodeCounter (counterId, counterName, counterType);
  return counterId;
}

uint32_t
AnimationInterface::AddResource (std::string resourcePath)
{
  m_resources.push_back (resourcePath);
  uint32_t resourceId = m_resources.size () -  1; // resource ID is zero-indexed
  WriteXmlAddResource (resourceId, resourcePath);
  return resourceId;
}

void
AnimationInterface::EnablePacketMetadata (bool enable)
{
  m_enablePacketMetadata = enable;
  if (enable)
    {
      Packet::EnablePrinting ();
    }
}

bool
AnimationInterface::IsInitialized ()
{
  return initialized;
}

bool
AnimationInterface::IsStarted ()
{
  return m_started;
}

void
AnimationInterface::SetAnimWriteCallback (AnimWriteCallback cb)
{
  m_writeCallback = cb;
}

void
AnimationInterface::ResetAnimWriteCallback ()
{
  m_writeCallback = 0;
}

void
AnimationInterface::SetMobilityPollInterval (Time t)
{
  m_mobilityPollInterval = t;
}


void
AnimationInterface::SetConstantPosition (Ptr <Node> n, double x, double y, double z)
{
  NS_ASSERT (n);
  Ptr<ConstantPositionMobilityModel> loc =  n->GetObject<ConstantPositionMobilityModel> ();
  if (loc == 0)
    {
      loc = CreateObject<ConstantPositionMobilityModel> ();
      n->AggregateObject (loc);
    }
  Vector hubVec (x, y, z);
  loc->SetPosition (hubVec);
  NS_LOG_INFO ("Node:" << n->GetId () << " Position set to:(" << x << "," << y << "," << z << ")");

}

void
AnimationInterface::UpdateNodeImage (uint32_t nodeId, uint32_t resourceId)
{
  NS_LOG_INFO ("Setting node image for Node Id:" << nodeId);
  if (resourceId > (m_resources.size () - 1))
    {
      NS_FATAL_ERROR ("Resource Id:" << resourceId << " not found. Did you use AddResource?");
    }
  WriteXmlUpdateNodeImage (nodeId, resourceId);
}

void
AnimationInterface::UpdateNodeCounter (uint32_t nodeCounterId, uint32_t nodeId, double counter)
{
  if (nodeCounterId > (m_nodeCounters.size () - 1))
    {
      NS_FATAL_ERROR ("NodeCounter Id:" << nodeCounterId << " not found. Did you use AddNodeCounter?");
    }
  WriteXmlUpdateNodeCounter (nodeCounterId, nodeId, counter);
}

void
AnimationInterface::SetBackgroundImage (std::string fileName, double x, double y, double scaleX, double scaleY, double opacity)
{
  if ((opacity < 0) || (opacity > 1))
    {
      NS_FATAL_ERROR ("Opacity must be between 0.0 and 1.0");
    }
  WriteXmlUpdateBackground (fileName, x, y, scaleX, scaleY, opacity);
}

void
AnimationInterface::UpdateNodeSize (uint32_t nodeId, double width, double height)
{
  AnimationInterface::NodeSize s = { width, height };
  m_nodeSizes[nodeId] = s;
  WriteXmlUpdateNodeSize (nodeId, s.width, s.height);
}

void
AnimationInterface::UpdateNodeColor (Ptr <Node> n, uint8_t r, uint8_t g, uint8_t b)
{
  UpdateNodeColor (n->GetId (), r, g, b);
}

void
AnimationInterface::UpdateNodeColor (uint32_t nodeId, uint8_t r, uint8_t g, uint8_t b)
{
  NS_ASSERT (NodeList::GetNode (nodeId));
  NS_LOG_INFO ("Setting node color for Node Id:" << nodeId);
  Rgb rgb = {r, g, b};
  m_nodeColors[nodeId] = rgb;
  WriteXmlUpdateNodeColor (nodeId, r, g, b);
}

void
AnimationInterface::UpdateLinkDescription (uint32_t fromNode, uint32_t toNode,
                                           std::string linkDescription)
{
  WriteXmlUpdateLink (fromNode, toNode, linkDescription);
}

void
AnimationInterface::UpdateLinkDescription (Ptr <Node> fromNode, Ptr <Node> toNode,
                                           std::string linkDescription)
{
  NS_ASSERT (fromNode);
  NS_ASSERT (toNode);
  WriteXmlUpdateLink (fromNode->GetId (), toNode->GetId (), linkDescription);
}

void
AnimationInterface::UpdateNodeDescription (Ptr <Node> n, std::string descr)
{
  UpdateNodeDescription (n->GetId (), descr);
}

void
AnimationInterface::UpdateNodeDescription (uint32_t nodeId, std::string descr)
{
  NS_ASSERT (NodeList::GetNode (nodeId));
  m_nodeDescriptions[nodeId] = descr;
  WriteXmlUpdateNodeDescription (nodeId);
}

// Private methods


double
AnimationInterface::GetNodeEnergyFraction (Ptr <const Node> node) const
{
  const EnergyFractionMap::const_iterator fractionIter = m_nodeEnergyFraction.find (node->GetId ());
  NS_ASSERT (fractionIter != m_nodeEnergyFraction.end ());
  return fractionIter->second;
}

void
AnimationInterface::MobilityCourseChangeTrace (Ptr <const MobilityModel> mobility)
{
  CHECK_STARTED_INTIMEWINDOW;
  Ptr <Node> n = mobility->GetObject <Node> ();
  NS_ASSERT (n);
  Vector v;
  if (!mobility)
    {
      v = GetPosition (n);
    }
  else
    {
      v = mobility->GetPosition ();
    }
  UpdatePosition (n, v);
  WriteXmlUpdateNodePosition (n->GetId (), v.x, v.y);
}

bool
AnimationInterface::NodeHasMoved (Ptr <Node> n, Vector newLocation)
{
  Vector oldLocation = GetPosition (n);
  bool moved = true;
  if ((ceil (oldLocation.x) == ceil (newLocation.x))
      && (ceil (oldLocation.y) == ceil (newLocation.y)))
    {
      moved = false;
    }
  else
    {
      moved = true;
    }
  return moved;
}

void
AnimationInterface::MobilityAutoCheck ()
{
  CHECK_STARTED_INTIMEWINDOW;
  std::vector <Ptr <Node> > MovedNodes = GetMovedNodes ();
  for (uint32_t i = 0; i < MovedNodes.size (); i++)
    {
      Ptr <Node> n = MovedNodes [i];
      NS_ASSERT (n);
      Vector v = GetPosition (n);
      WriteXmlUpdateNodePosition (n->GetId (), v.x, v.y);
    }
  if (!Simulator::IsFinished ())
    {
      PurgePendingPackets (AnimationInterface::WIFI);
      PurgePendingPackets (AnimationInterface::WIMAX);
      PurgePendingPackets (AnimationInterface::LTE);
      PurgePendingPackets (AnimationInterface::CSMA);
      PurgePendingPackets (AnimationInterface::LRWPAN);
      PurgePendingPackets (AnimationInterface::WAVE);
      Simulator::Schedule (m_mobilityPollInterval, &AnimationInterface::MobilityAutoCheck, this);
    }
}

std::vector <Ptr <Node> >
AnimationInterface::GetMovedNodes ()
{
  std::vector < Ptr <Node> > movedNodes;
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      NS_ASSERT (n);
      Ptr <MobilityModel> mobility = n->GetObject <MobilityModel> ();
      Vector newLocation;
      if (!mobility)
        {
          newLocation = GetPosition (n);
        }
      else
        {
          newLocation = mobility->GetPosition ();
        }
      if (!NodeHasMoved (n, newLocation))
        {
          continue; //Location has not changed
        }
      else
        {
          UpdatePosition (n, newLocation);
          movedNodes.push_back (n);
        }
    }
  return movedNodes;
}

int
AnimationInterface::WriteN (const std::string& st, FILE * f)
{
  if (!f)
    {
      return 0;
    }
  if (m_writeCallback)
    {
      m_writeCallback (st.c_str ());
    }
  return WriteN (st.c_str (), st.length (), f);
}

int
AnimationInterface::WriteN (const char* data, uint32_t count, FILE * f)
{
  if (!f)
    {
      return 0;
    }
  // Write count bytes to h from data
  uint32_t    nLeft   = count;
  const char* p       = data;
  uint32_t    written = 0;
  while (nLeft)
    {
      int n = std::fwrite (p, 1,  nLeft, f);
      if (n <= 0)
        {
          return written;
        }
      written += n;
      nLeft -= n;
      p += n;
    }
  return written;
}

void
AnimationInterface::WriteRoutePath (uint32_t nodeId, std::string destination, Ipv4RoutePathElements rpElements)
{
  NS_LOG_INFO ("Writing Route Path From :" << nodeId << " To: " << destination.c_str ());
  WriteXmlRp (nodeId, destination, rpElements);
  /*for (Ipv4RoutePathElements::const_iterator i = rpElements.begin ();
       i != rpElements.end ();
       ++i)
    {
      Ipv4RoutePathElement rpElement = *i;
      NS_LOG_INFO ("Node:" << rpElement.nodeId << "-->" << rpElement.nextHop.c_str ());
      WriteN (GetXmlRp (rpElement.node, GetIpv4RoutingTable (n)), m_routingF);

    }
  */
}

void
AnimationInterface::WriteNonP2pLinkProperties (uint32_t id, std::string ipv4Address, std::string channelType)
{
  WriteXmlNonP2pLinkProperties (id, ipv4Address, channelType);
}

const std::vector<std::string>
AnimationInterface::GetElementsFromContext (const std::string& context) const
{
  std::vector <std::string> elements;
  std::size_t pos1 = 0, pos2;
  while (pos1 != context.npos)
    {
      pos1 = context.find ("/",pos1);
      pos2 = context.find ("/",pos1 + 1);
      elements.push_back (context.substr (pos1 + 1,pos2 - (pos1 + 1)));
      pos1 = pos2;
      pos2 = context.npos;
    }
  return elements;
}

Ptr <Node>
AnimationInterface::GetNodeFromContext (const std::string& context) const
{
  // Use "NodeList/*/ as reference
  // where element [1] is the Node Id

  std::vector <std::string> elements = GetElementsFromContext (context);
  Ptr <Node> n = NodeList::GetNode (atoi (elements.at (1).c_str ()));
  NS_ASSERT (n);

  return n;
}

Ptr <NetDevice>
AnimationInterface::GetNetDeviceFromContext (std::string context)
{
  // Use "NodeList/*/DeviceList/*/ as reference
  // where element [1] is the Node Id
  // element [2] is the NetDevice Id

  std::vector <std::string> elements = GetElementsFromContext (context);
  Ptr <Node> n = GetNodeFromContext (context);

  return n->GetDevice (atoi (elements.at (3).c_str ()));
}

uint64_t
AnimationInterface::GetAnimUidFromPacket (Ptr <const Packet> p)
{
  AnimByteTag tag;
  TypeId tid = tag.GetInstanceTypeId ();
  ByteTagIterator i = p->GetByteTagIterator ();
  bool found = false;
  while (i.HasNext ())
    {
      ByteTagIterator::Item item = i.Next ();
      if (tid == item.GetTypeId ())
        {
          item.GetTag (tag);
          found = true;
        }
    }
  if (found)
    {
      return tag.Get ();
    }
  else
    {
      return 0;
    }
}

void
AnimationInterface::AddByteTag (uint64_t animUid, Ptr<const Packet> p)
{
  AnimByteTag tag;
  tag.Set (animUid);
  p->AddByteTag (tag);
}

void
AnimationInterface::RemainingEnergyTrace (std::string context, double previousEnergy, double currentEnergy)
{
  CHECK_STARTED_INTIMEWINDOW;
  const Ptr <const Node> node = GetNodeFromContext (context);
  const uint32_t nodeId = node->GetId ();

  NS_LOG_INFO ("Remaining energy on one of sources on node " << nodeId << ": " << currentEnergy);

  const Ptr<EnergySource> energySource = node->GetObject<EnergySource> ();

  NS_ASSERT (energySource);
  // Don't call GetEnergyFraction () because of recursion
  const double energyFraction = currentEnergy / energySource->GetInitialEnergy ();

  NS_LOG_INFO ("Total energy fraction on node " << nodeId << ": " << energyFraction);

  m_nodeEnergyFraction[nodeId] = energyFraction;
  UpdateNodeCounter (m_remainingEnergyCounterId, nodeId, energyFraction);
}

void
AnimationInterface::WifiPhyTxDropTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiPhyTxDrop[node->GetId ()];
}

void
AnimationInterface::WifiPhyRxDropTrace (std::string context, Ptr<const Packet> p, WifiPhyRxfailureReason reason)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiPhyRxDrop[node->GetId ()];
}

void
AnimationInterface::WifiMacTxTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiMacTx[node->GetId ()];
}

void
AnimationInterface::WifiMacTxDropTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiMacTxDrop[node->GetId ()];
}

void
AnimationInterface::WifiMacRxTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiMacRx[node->GetId ()];
}

void
AnimationInterface::WifiMacRxDropTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeWifiMacRxDrop[node->GetId ()];
}

void
AnimationInterface::LrWpanMacTxTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeLrWpanMacTx[node->GetId ()];
}

void
AnimationInterface::LrWpanMacTxDropTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeLrWpanMacTxDrop[node->GetId ()];
}

void
AnimationInterface::LrWpanMacRxTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeLrWpanMacRx[node->GetId ()];
}

void
AnimationInterface::LrWpanMacRxDropTrace (std::string context, Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeLrWpanMacRxDrop[node->GetId ()];
}

void
AnimationInterface::Ipv4TxTrace (std::string context, Ptr<const Packet> p, Ptr<Ipv4> ipv4, uint32_t interfaceIndex)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeIpv4Tx[node->GetId ()];
}

void
AnimationInterface::Ipv4RxTrace (std::string context, Ptr<const Packet> p, Ptr<Ipv4> ipv4, uint32_t interfaceIndex)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeIpv4Rx[node->GetId ()];
}

void
AnimationInterface::Ipv4DropTrace (std::string context,
                                   const Ipv4Header & ipv4Header,
                                   Ptr<const Packet> p,
                                   Ipv4L3Protocol::DropReason dropReason,
                                   Ptr<Ipv4> ipv4,
                                   uint32_t)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeIpv4Drop[node->GetId ()];
}

void
AnimationInterface::EnqueueTrace (std::string context,
                                  Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeQueueEnqueue[node->GetId ()];
}

void
AnimationInterface::DequeueTrace (std::string context,
                                  Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeQueueDequeue[node->GetId ()];
}

void
AnimationInterface::QueueDropTrace (std::string context,
                                    Ptr<const Packet> p)
{
  const Ptr <const Node> node = GetNodeFromContext (context);
  ++m_nodeQueueDrop[node->GetId ()];
}

void
AnimationInterface::DevTxTrace (std::string context,
                                Ptr<const Packet> p,
                                Ptr<NetDevice> tx,
                                Ptr<NetDevice> rx,
                                Time txTime,
                                Time rxTime)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  NS_ASSERT (tx);
  NS_ASSERT (rx);
  Time now = Simulator::Now ();
  double fbTx = now.GetSeconds ();
  double lbTx = (now + txTime).GetSeconds ();
  double fbRx = (now + rxTime - txTime).GetSeconds ();
  double lbRx = (now + rxTime).GetSeconds ();
  CheckMaxPktsPerTraceFile ();
  WriteXmlP ("p",
             tx->GetNode ()->GetId (),
             fbTx,
             lbTx,
             rx->GetNode ()->GetId (),
             fbRx,
             lbRx,
             m_enablePacketMetadata ? GetPacketMetadata (p) : "");
}

void
AnimationInterface::GenericWirelessTxTrace (std::string context, Ptr<const Packet> p, ProtocolType protocolType)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);

  ++gAnimUid;
  NS_LOG_INFO (ProtocolTypeToString (protocolType).c_str () << " GenericWirelessTxTrace for packet:" << gAnimUid);
  AddByteTag (gAnimUid, p);
  AnimPacketInfo pktInfo (ndev, Simulator::Now ());
  AddPendingPacket (protocolType, gAnimUid, pktInfo);

  Ptr<WifiNetDevice> netDevice = DynamicCast<WifiNetDevice> (ndev);
  if (netDevice)
    {
      Mac48Address nodeAddr = netDevice->GetMac ()->GetAddress ();
      std::ostringstream oss;
      oss << nodeAddr;
      Ptr <Node> n = netDevice->GetNode ();
      NS_ASSERT (n);
      m_macToNodeIdMap[oss.str ()] = n->GetId ();
      NS_LOG_INFO ("Added Mac" << oss.str () << " node:" << m_macToNodeIdMap[oss.str ()]);
    }
  AnimUidPacketInfoMap * pendingPackets =  ProtocolTypeToPendingPackets (protocolType);
  OutputWirelessPacketTxInfo (p, pendingPackets->at (gAnimUid), gAnimUid);
}

void
AnimationInterface::GenericWirelessRxTrace (std::string context, Ptr<const Packet> p, ProtocolType protocolType)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  NS_LOG_INFO (ProtocolTypeToString (protocolType).c_str () << " for packet:" << animUid);
  if (!IsPacketPending (animUid, protocolType))
    {
      NS_LOG_WARN (ProtocolTypeToString (protocolType).c_str () << " GenericWirelessRxTrace: unknown Uid");
      return;
    }
  AnimUidPacketInfoMap * pendingPackets =  ProtocolTypeToPendingPackets (protocolType);
  pendingPackets->at (animUid).ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
  OutputWirelessPacketRxInfo (p, pendingPackets->at (animUid), animUid);
}

void
AnimationInterface::UanPhyGenTxTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessTxTrace (context, p, AnimationInterface::UAN);
}

void
AnimationInterface::UanPhyGenRxTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessRxTrace (context, p, AnimationInterface::UAN);
}

void
AnimationInterface::WifiPhyTxBeginTrace (std::string context, WifiConstPsduMap psduMap, [[maybe_unused]] WifiTxVector txVector, double txPowerW)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr<NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);

  AnimPacketInfo pktInfo (ndev, Simulator::Now ());
  AnimUidPacketInfoMap * pendingPackets =  ProtocolTypeToPendingPackets (WIFI);
  for (auto& psdu : psduMap)
    {
      for (auto& mpdu : *PeekPointer (psdu.second))
        {
          ++gAnimUid;
          NS_LOG_INFO ("WifiPhyTxTrace for MPDU:" << gAnimUid);
          AddByteTag (gAnimUid, mpdu->GetPacket ()); //the underlying MSDU/A-MSDU should be handed off
          AddPendingPacket (WIFI, gAnimUid, pktInfo);
          OutputWirelessPacketTxInfo (mpdu->GetProtocolDataUnit (), pendingPackets->at (gAnimUid), gAnimUid); //PDU should be considered in order to have header
        }
    }

  Ptr<WifiNetDevice> netDevice = DynamicCast<WifiNetDevice> (ndev);
  if (netDevice)
    {
      Mac48Address nodeAddr = netDevice->GetMac ()->GetAddress ();
      std::ostringstream oss;
      oss << nodeAddr;
      Ptr<Node> n = netDevice->GetNode ();
      NS_ASSERT (n);
      m_macToNodeIdMap[oss.str ()] = n->GetId ();
      NS_LOG_INFO ("Added Mac" << oss.str () << " node:" << m_macToNodeIdMap[oss.str ()]);
    }
  else
    {
      NS_ABORT_MSG ("This NetDevice should be a Wi-Fi network device");
    }
}

void
AnimationInterface::WifiPhyRxBeginTrace (std::string context, Ptr<const Packet> p, RxPowerWattPerChannelBand rxPowersW)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  NS_LOG_INFO ("Wifi RxBeginTrace for packet: " << animUid);
  if (!IsPacketPending (animUid, AnimationInterface::WIFI))
    {
      NS_ASSERT_MSG (false, "WifiPhyRxBeginTrace: unknown Uid");
      std::ostringstream oss;
      WifiMacHeader hdr;
      if (!p->PeekHeader (hdr))
        {
          NS_LOG_WARN ("WifiMacHeader not present");
          return;
        }
      oss << hdr.GetAddr2 ();
      if (m_macToNodeIdMap.find (oss.str ()) == m_macToNodeIdMap.end ())
        {
          NS_LOG_WARN ("Transmitter Mac address " << oss.str () << " never seen before. Skipping");
          return;
        }
      Ptr <Node> txNode = NodeList::GetNode (m_macToNodeIdMap[oss.str ()]);
      UpdatePosition (txNode);
      AnimPacketInfo pktInfo (0, Simulator::Now (), m_macToNodeIdMap[oss.str ()]);
      AddPendingPacket (AnimationInterface::WIFI, animUid, pktInfo);
      NS_LOG_WARN ("WifiPhyRxBegin: unknown Uid, but we are adding a wifi packet");
    }
  /// \todo NS_ASSERT (WifiPacketIsPending (animUid) == true);
  m_pendingWifiPackets[animUid].ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
  OutputWirelessPacketRxInfo (p, m_pendingWifiPackets[animUid], animUid);
}

void
AnimationInterface::LrWpanPhyTxBeginTrace (std::string context,
                                           Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;

  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  Ptr<LrWpanNetDevice> netDevice = DynamicCast<LrWpanNetDevice> (ndev);

  Ptr <Node> n = ndev->GetNode ();
  NS_ASSERT (n);

  UpdatePosition (n);

  LrWpanMacHeader hdr;
  if (!p->PeekHeader (hdr))
    {
      NS_LOG_WARN ("LrWpanMacHeader not present");
      return;
    }

  std::ostringstream oss;
  if (hdr.GetSrcAddrMode () == 2)
    {
      Mac16Address nodeAddr = netDevice->GetMac ()->GetShortAddress ();
      oss << nodeAddr;
    }
  else if (hdr.GetSrcAddrMode () == 3)
    {
      Mac64Address nodeAddr = netDevice->GetMac ()->GetExtendedAddress ();
      oss << nodeAddr;
    }
  else
    {
      NS_LOG_WARN ("LrWpanMacHeader without source address");
      return;
    }
  m_macToNodeIdMap[oss.str ()] = n->GetId ();
  NS_LOG_INFO ("Added Mac" << oss.str () << " node:" << m_macToNodeIdMap[oss.str ()]);

  ++gAnimUid;
  NS_LOG_INFO ("LrWpan TxBeginTrace for packet:" << gAnimUid);
  AddByteTag (gAnimUid, p);

  AnimPacketInfo pktInfo (ndev, Simulator::Now ());
  AddPendingPacket (AnimationInterface::LRWPAN, gAnimUid, pktInfo);

  OutputWirelessPacketTxInfo (p, m_pendingLrWpanPackets[gAnimUid], gAnimUid);
}

void
AnimationInterface::LrWpanPhyRxBeginTrace (std::string context,
                                           Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  Ptr <Node> n = ndev->GetNode ();
  NS_ASSERT (n);

  AnimByteTag tag;
  if (!p->FindFirstMatchingByteTag (tag))
    {
      return;
    }

  uint64_t animUid = GetAnimUidFromPacket (p);
  NS_LOG_INFO ("LrWpan RxBeginTrace for packet:" << animUid);
  if (!IsPacketPending (animUid, AnimationInterface::LRWPAN))
    {
      NS_LOG_WARN ("LrWpanPhyRxBeginTrace: unknown Uid - most probably it's an ACK.");
    }

  UpdatePosition (n);
  m_pendingLrWpanPackets[animUid].ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
  OutputWirelessPacketRxInfo (p, m_pendingLrWpanPackets[animUid], animUid);
}

void
AnimationInterface::WavePhyTxBeginTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessTxTrace (context, p, AnimationInterface::WAVE);
}

void
AnimationInterface::WavePhyRxBeginTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  NS_LOG_INFO ("Wave RxBeginTrace for packet:" << animUid);
  if (!IsPacketPending (animUid, AnimationInterface::WAVE))
    {
      NS_ASSERT_MSG (false, "WavePhyRxBeginTrace: unknown Uid");
      std::ostringstream oss;
      WifiMacHeader hdr;
      if (!p->PeekHeader (hdr))
        {
          NS_LOG_WARN ("WaveMacHeader not present");
          return;
        }
      oss << hdr.GetAddr2 ();
      if (m_macToNodeIdMap.find (oss.str ()) == m_macToNodeIdMap.end ())
        {
          NS_LOG_WARN ("Transmitter Mac address " << oss.str () << " never seen before. Skipping");
          return;
        }
      Ptr <Node> txNode = NodeList::GetNode (m_macToNodeIdMap[oss.str ()]);
      UpdatePosition (txNode);
      AnimPacketInfo pktInfo (0, Simulator::Now (), m_macToNodeIdMap[oss.str ()]);
      AddPendingPacket (AnimationInterface::WAVE, animUid, pktInfo);
      NS_LOG_WARN ("WavePhyRxBegin: unknown Uid, but we are adding a wave packet");
    }
  /// \todo NS_ASSERT (WavePacketIsPending (animUid) == true);
  m_pendingWavePackets[animUid].ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
  OutputWirelessPacketRxInfo (p, m_pendingWavePackets[animUid], animUid);
}


void
AnimationInterface::WimaxTxTrace (std::string context, Ptr<const Packet> p, const Mac48Address & m)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessTxTrace (context, p, AnimationInterface::WIMAX);
}


void
AnimationInterface::WimaxRxTrace (std::string context, Ptr<const Packet> p, const Mac48Address & m)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessRxTrace (context, p, AnimationInterface::WIMAX);
}

void
AnimationInterface::LteTxTrace (std::string context, Ptr<const Packet> p, const Mac48Address & m)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessTxTrace (context, p, AnimationInterface::LTE);
}

void
AnimationInterface::LteRxTrace (std::string context, Ptr<const Packet> p, const Mac48Address & m)
{
  NS_LOG_FUNCTION (this);
  return GenericWirelessRxTrace (context, p, AnimationInterface::LTE);
}

void
AnimationInterface::LteSpectrumPhyTxStart (std::string context, Ptr<const PacketBurst> pb)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  if (!pb)
    {
      NS_LOG_WARN ("pb == 0. Not yet supported");
      return;
    }
  context = "/" + context;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);

  std::list <Ptr <Packet> > pbList = pb->GetPackets ();
  for (std::list <Ptr <Packet> >::iterator i  = pbList.begin ();
       i != pbList.end ();
       ++i)
    {
      Ptr <Packet> p = *i;
      ++gAnimUid;
      NS_LOG_INFO ("LteSpectrumPhyTxTrace for packet:" << gAnimUid);
      AnimPacketInfo pktInfo (ndev, Simulator::Now ());
      AddByteTag (gAnimUid, p);
      AddPendingPacket (AnimationInterface::LTE, gAnimUid, pktInfo);
      OutputWirelessPacketTxInfo (p, pktInfo, gAnimUid);
    }
}

void
AnimationInterface::LteSpectrumPhyRxStart (std::string context, Ptr<const PacketBurst> pb)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  if (!pb)
    {
      NS_LOG_WARN ("pb == 0. Not yet supported");
      return;
    }
  context = "/" + context;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);

  std::list <Ptr <Packet> > pbList = pb->GetPackets ();
  for (std::list <Ptr <Packet> >::iterator i  = pbList.begin ();
       i != pbList.end ();
       ++i)
    {
      Ptr <Packet> p = *i;
      uint64_t animUid = GetAnimUidFromPacket (p);
      NS_LOG_INFO ("LteSpectrumPhyRxTrace for packet:" << gAnimUid);
      if (!IsPacketPending (animUid, AnimationInterface::LTE))
        {
          NS_LOG_WARN ("LteSpectrumPhyRxTrace: unknown Uid");
          return;
        }
      AnimPacketInfo& pktInfo = m_pendingLtePackets[animUid];
      pktInfo.ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
      OutputWirelessPacketRxInfo (p, pktInfo, animUid);
    }
}

void
AnimationInterface::CsmaPhyTxBeginTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  ++gAnimUid;
  NS_LOG_INFO ("CsmaPhyTxBeginTrace for packet:" << gAnimUid);
  AddByteTag (gAnimUid, p);
  UpdatePosition (ndev);
  AnimPacketInfo pktInfo (ndev, Simulator::Now ());
  AddPendingPacket (AnimationInterface::CSMA, gAnimUid, pktInfo);


}

void
AnimationInterface::CsmaPhyTxEndTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  NS_LOG_INFO ("CsmaPhyTxEndTrace for packet:" << animUid);
  if (!IsPacketPending (animUid, AnimationInterface::CSMA))
    {
      NS_LOG_WARN ("CsmaPhyTxEndTrace: unknown Uid");
      NS_FATAL_ERROR ("CsmaPhyTxEndTrace: unknown Uid");
      AnimPacketInfo pktInfo (ndev, Simulator::Now ());
      AddPendingPacket (AnimationInterface::CSMA, animUid, pktInfo);
      NS_LOG_WARN ("Unknown Uid, but adding Csma Packet anyway");
    }
  /// \todo NS_ASSERT (IsPacketPending (AnimUid) == true);
  AnimPacketInfo& pktInfo = m_pendingCsmaPackets[animUid];
  pktInfo.m_lbTx = Simulator::Now ().GetSeconds ();
}

void
AnimationInterface::CsmaPhyRxEndTrace (std::string context, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  UpdatePosition (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  if (!IsPacketPending (animUid, AnimationInterface::CSMA))
    {
      NS_LOG_WARN ("CsmaPhyRxEndTrace: unknown Uid");
      return;
    }
  /// \todo NS_ASSERT (CsmaPacketIsPending (AnimUid) == true);
  AnimPacketInfo& pktInfo = m_pendingCsmaPackets[animUid];
  pktInfo.ProcessRxBegin (ndev, Simulator::Now ().GetSeconds ());
  NS_LOG_INFO ("CsmaPhyRxEndTrace for packet:" << animUid);
  NS_LOG_INFO ("CsmaPhyRxEndTrace for packet:" << animUid << " complete");
  OutputCsmaPacket (p, pktInfo);
}

void
AnimationInterface::CsmaMacRxTrace (std::string context,
                                    Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this);
  CHECK_STARTED_INTIMEWINDOW_TRACKPACKETS;
  Ptr <NetDevice> ndev = GetNetDeviceFromContext (context);
  NS_ASSERT (ndev);
  uint64_t animUid = GetAnimUidFromPacket (p);
  if (!IsPacketPending (animUid, AnimationInterface::CSMA))
    {
      NS_LOG_WARN ("CsmaMacRxTrace: unknown Uid");
      return;
    }
  /// \todo NS_ASSERT (CsmaPacketIsPending (AnimUid) == true);
  AnimPacketInfo& pktInfo = m_pendingCsmaPackets[animUid];
  NS_LOG_INFO ("MacRxTrace for packet:" << animUid << " complete");
  OutputCsmaPacket (p, pktInfo);
}

void
AnimationInterface::OutputWirelessPacketTxInfo (Ptr<const Packet> p, AnimPacketInfo &pktInfo, uint64_t animUid)
{
  CheckMaxPktsPerTraceFile ();
  uint32_t nodeId = 0;
  if (pktInfo.m_txnd)
    {
      nodeId = pktInfo.m_txnd->GetNode ()->GetId ();
    }
  else
    {
      nodeId = pktInfo.m_txNodeId;
    }
  WriteXmlPRef (animUid, nodeId, pktInfo.m_fbTx, m_enablePacketMetadata ? GetPacketMetadata (p) : "");
}

void
AnimationInterface::OutputWirelessPacketRxInfo (Ptr<const Packet> p, AnimPacketInfo & pktInfo, uint64_t animUid)
{
  CheckMaxPktsPerTraceFile ();
  uint32_t rxId = pktInfo.m_rxnd->GetNode ()->GetId ();
  WriteXmlP (animUid, "wpr", rxId, pktInfo.m_fbRx, pktInfo.m_lbRx);
}

void
AnimationInterface::OutputCsmaPacket (Ptr<const Packet> p, AnimPacketInfo &pktInfo)
{
  CheckMaxPktsPerTraceFile ();
  NS_ASSERT (pktInfo.m_txnd);
  uint32_t nodeId = pktInfo.m_txnd->GetNode ()->GetId ();
  uint32_t rxId = pktInfo.m_rxnd->GetNode ()->GetId ();

  WriteXmlP ("p",
             nodeId,
             pktInfo.m_fbTx,
             pktInfo.m_lbTx,
             rxId,
             pktInfo.m_fbRx,
             pktInfo.m_lbRx,
             m_enablePacketMetadata ? GetPacketMetadata (p) : "");
}

void
AnimationInterface::AddPendingPacket (ProtocolType protocolType, uint64_t animUid, AnimPacketInfo pktInfo)
{
  AnimUidPacketInfoMap * pendingPackets = ProtocolTypeToPendingPackets (protocolType);
  NS_ASSERT (pendingPackets);
  pendingPackets->insert (AnimUidPacketInfoMap::value_type (animUid, pktInfo));
}

bool
AnimationInterface::IsPacketPending (uint64_t animUid, AnimationInterface::ProtocolType protocolType)
{
  AnimUidPacketInfoMap * pendingPackets = ProtocolTypeToPendingPackets (protocolType);
  NS_ASSERT (pendingPackets);
  return (pendingPackets->find (animUid) != pendingPackets->end ());
}

void
AnimationInterface::PurgePendingPackets (AnimationInterface::ProtocolType protocolType)
{
  AnimUidPacketInfoMap * pendingPackets = ProtocolTypeToPendingPackets (protocolType);
  NS_ASSERT (pendingPackets);
  if (pendingPackets->empty ())
    {
      return;
    }
  std::vector <uint64_t> purgeList;
  for (AnimUidPacketInfoMap::iterator i = pendingPackets->begin ();
       i != pendingPackets->end ();
       ++i)
    {

      AnimPacketInfo pktInfo = i->second;
      double delta = (Simulator::Now ().GetSeconds () - pktInfo.m_fbTx);
      if (delta > PURGE_INTERVAL)
        {
          purgeList.push_back (i->first);
        }
    }
  for (std::vector <uint64_t>::iterator i = purgeList.begin ();
       i != purgeList.end ();
       ++i)
    {
      pendingPackets->erase (*i);
    }
}

AnimationInterface::AnimUidPacketInfoMap *
AnimationInterface::ProtocolTypeToPendingPackets (AnimationInterface::ProtocolType protocolType)
{
  AnimUidPacketInfoMap * pendingPackets = 0;
  switch (protocolType)
    {
    case AnimationInterface::WIFI:
      {
        pendingPackets = &m_pendingWifiPackets;
        break;
      }
    case AnimationInterface::UAN:
      {
        pendingPackets = &m_pendingUanPackets;
        break;
      }
    case AnimationInterface::CSMA:
      {
        pendingPackets = &m_pendingCsmaPackets;
        break;
      }
    case AnimationInterface::WIMAX:
      {
        pendingPackets = &m_pendingWimaxPackets;
        break;
      }
    case AnimationInterface::LTE:
      {
        pendingPackets = &m_pendingLtePackets;
        break;
      }
    case AnimationInterface::LRWPAN:
      {
        pendingPackets = &m_pendingLrWpanPackets;
        break;
      }
    case AnimationInterface::WAVE:
      {
        pendingPackets = &m_pendingWavePackets;
        break;
      }
    }
  return pendingPackets;

}

std::string
AnimationInterface::ProtocolTypeToString (AnimationInterface::ProtocolType protocolType)
{
  std::string result = "Unknown";
  switch (protocolType)
    {
    case AnimationInterface::WIFI:
      {
        result = "WIFI";
        break;
      }
    case AnimationInterface::UAN:
      {
        result = "UAN";
        break;
      }
    case AnimationInterface::CSMA:
      {
        result = "CSMA";
        break;
      }
    case AnimationInterface::WIMAX:
      {
        result = "WIMAX";
        break;
      }
    case AnimationInterface::LTE:
      {
        result = "LTE";
        break;
      }
    case AnimationInterface::LRWPAN:
      {
        result = "LRWPAN";
        break;
      }
    case AnimationInterface::WAVE:
      {
        result = "WAVE";
        break;
      }
    }
  return result;
}

// Counters

std::string
AnimationInterface::CounterTypeToString (CounterType counterType)
{
  std::string typeString = "unknown";
  switch (counterType)
    {
    case UINT32_COUNTER:
      {
        typeString = "UINT32";
        break;
      }
    case DOUBLE_COUNTER:
      {
        typeString = "DOUBLE";
        break;
      }
    }
  return typeString;
}

// General

std::string
AnimationInterface::GetPacketMetadata (Ptr<const Packet> p)
{
  std::ostringstream oss;
  p->Print (oss);
  return oss.str ();
}

uint64_t
AnimationInterface::GetTracePktCount ()
{
  return m_currentPktCount;
}

void
AnimationInterface::StopAnimation (bool onlyAnimation)
{
  m_started = false;
  NS_LOG_INFO ("Stopping Animation");
  ResetAnimWriteCallback ();
  if (m_f)
    {
      // Terminate the anim element
      WriteXmlClose ("anim");
      std::fclose (m_f);
      m_f = 0;
    }
  if (onlyAnimation)
    {
      return;
    }
  if (m_routingF)
    {
      WriteXmlClose ("anim", true);
      std::fclose (m_routingF);
      m_routingF = 0;
    }
}

void
AnimationInterface::StartAnimation (bool restart)
{
  m_currentPktCount = 0;
  m_started = true;
  SetOutputFile (m_outputFileName);
  WriteXmlAnim ();
  WriteNodes ();
  WriteNodeColors ();
  WriteLinkProperties ();
  WriteIpv4Addresses ();
  WriteIpv6Addresses ();
  WriteNodeSizes ();
  WriteNodeEnergies ();
  if (!restart)
    {
      Simulator::Schedule (m_mobilityPollInterval, &AnimationInterface::MobilityAutoCheck, this);
      ConnectCallbacks ();
    }
}

void
AnimationInterface::AddToIpv4AddressNodeIdTable (std::string ipv4Address, uint32_t nodeId)
{
  m_ipv4ToNodeIdMap[ipv4Address] = nodeId;
  m_nodeIdIpv4Map.insert (NodeIdIpv4Pair (nodeId, ipv4Address));
}

void
AnimationInterface::AddToIpv4AddressNodeIdTable (std::vector<std::string> ipv4Addresses, uint32_t nodeId)
{
  for (std::vector<std::string>::const_iterator i = ipv4Addresses.begin ();
       i != ipv4Addresses.end ();
       ++i)
    {
      AddToIpv4AddressNodeIdTable (*i, nodeId);
    }
}

void
AnimationInterface::AddToIpv6AddressNodeIdTable (std::string ipv6Address, uint32_t nodeId)
{
  m_ipv6ToNodeIdMap[ipv6Address] = nodeId;
  m_nodeIdIpv6Map.insert (NodeIdIpv6Pair (nodeId, ipv6Address));
}

void
AnimationInterface::AddToIpv6AddressNodeIdTable (std::vector<std::string> ipv6Addresses, uint32_t nodeId)
{
  for (std::vector<std::string>::const_iterator i = ipv6Addresses.begin ();
       i != ipv6Addresses.end ();
       ++i)
    {
      AddToIpv6AddressNodeIdTable (*i, nodeId);
    }
}

// Callbacks
void
AnimationInterface::ConnectLteEnb (Ptr <Node> n, Ptr <LteEnbNetDevice> nd, uint32_t devIndex)
{

  Ptr<LteEnbPhy> lteEnbPhy = nd->GetPhy ();
  Ptr<LteSpectrumPhy> dlPhy = lteEnbPhy->GetDownlinkSpectrumPhy ();
  Ptr<LteSpectrumPhy> ulPhy = lteEnbPhy->GetUplinkSpectrumPhy ();
  std::ostringstream oss;
  //NodeList/*/DeviceList/*/
  oss << "NodeList/" << n->GetId () << "/DeviceList/" << devIndex << "/";
  if (dlPhy)
    {
      dlPhy->TraceConnect ("TxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyTxStart, this));
      dlPhy->TraceConnect ("RxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyRxStart, this));
    }
  if (ulPhy)
    {
      ulPhy->TraceConnect ("TxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyTxStart, this));
      ulPhy->TraceConnect ("RxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyRxStart, this));
    }
}



void
AnimationInterface::ConnectLteUe (Ptr <Node> n, Ptr <LteUeNetDevice> nd, uint32_t devIndex)
{

  Ptr<LteUePhy> lteUePhy = nd->GetPhy ();
  Ptr<LteSpectrumPhy> dlPhy = lteUePhy->GetDownlinkSpectrumPhy ();
  Ptr<LteSpectrumPhy> ulPhy = lteUePhy->GetUplinkSpectrumPhy ();
  std::ostringstream oss;
  //NodeList/*/DeviceList/*/
  oss << "NodeList/" << n->GetId () << "/DeviceList/" << devIndex << "/";
  if (dlPhy)
    {
      dlPhy->TraceConnect ("TxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyTxStart, this));
      dlPhy->TraceConnect ("RxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyRxStart, this));
    }
  if (ulPhy)
    {
      ulPhy->TraceConnect ("TxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyTxStart, this));
      ulPhy->TraceConnect ("RxStart", oss.str (), MakeCallback (&AnimationInterface::LteSpectrumPhyRxStart, this));
    }
}

void
AnimationInterface::ConnectLte ()
{

  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      NS_ASSERT (n);
      uint32_t nDevices = n->GetNDevices ();
      for (uint32_t devIndex = 0; devIndex < nDevices; ++devIndex)
        {
          Ptr <NetDevice> nd = n->GetDevice (devIndex);
          if (!nd)
            {
              continue;
            }
          Ptr<LteUeNetDevice> lteUeNetDevice = DynamicCast<LteUeNetDevice> (nd);
          if (lteUeNetDevice)
            {
              ConnectLteUe (n, lteUeNetDevice, devIndex);
              continue;
            }
          Ptr<LteEnbNetDevice> lteEnbNetDevice = DynamicCast<LteEnbNetDevice> (nd);
          if (lteEnbNetDevice)
            {
              ConnectLteEnb (n, lteEnbNetDevice, devIndex);
            }
        }

    }
}

void
AnimationInterface::ConnectCallbacks ()
{
  // Connect the callbacks
  Config::ConnectFailSafe ("/ChannelList/*/TxRxPointToPoint",
                           MakeCallback (&AnimationInterface::DevTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                           MakeCallback (&AnimationInterface::WifiPhyTxBeginTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxBegin",
                           MakeCallback (&AnimationInterface::WifiPhyRxBeginTrace, this));
  Config::ConnectWithoutContextFailSafe ("/NodeList/*/$ns3::MobilityModel/CourseChange",
                                         MakeCallback (&AnimationInterface::MobilityCourseChangeTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WimaxNetDevice/Tx",
                           MakeCallback (&AnimationInterface::WimaxTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WimaxNetDevice/Rx",
                           MakeCallback (&AnimationInterface::WimaxRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/Tx",
                           MakeCallback (&AnimationInterface::LteTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/Rx",
                           MakeCallback (&AnimationInterface::LteRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/PhyTxBegin",
                           MakeCallback (&AnimationInterface::CsmaPhyTxBeginTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/PhyTxEnd",
                           MakeCallback (&AnimationInterface::CsmaPhyTxEndTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/PhyRxEnd",
                           MakeCallback (&AnimationInterface::CsmaPhyRxEndTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/MacRx",
                           MakeCallback (&AnimationInterface::CsmaMacRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::UanNetDevice/Phy/PhyTxBegin",
                           MakeCallback (&AnimationInterface::UanPhyGenTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::UanNetDevice/Phy/PhyRxBegin",
                           MakeCallback (&AnimationInterface::UanPhyGenRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/$ns3::BasicEnergySource/RemainingEnergy",
                           MakeCallback (&AnimationInterface::RemainingEnergyTrace, this));
  
  ConnectLte ();
  
  Config::ConnectFailSafe ("/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                           MakeCallback (&AnimationInterface::Ipv4TxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
                           MakeCallback (&AnimationInterface::Ipv4RxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/$ns3::Ipv4L3Protocol/Drop",
                           MakeCallback (&AnimationInterface::Ipv4DropTrace, this));
  
  // Queue Enqueues
  
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::AlohaNoackNetDevice/Queue/Enqueue",
                           MakeCallback (&AnimationInterface::EnqueueTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/TxQueue/Enqueue",
                           MakeCallback (&AnimationInterface::EnqueueTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/TxQueue/Enqueue",
                           MakeCallback (&AnimationInterface::EnqueueTrace, this));
  
  // Queue Dequeues
  
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::AlohaNoackNetDevice/Queue/Dequeue",
                           MakeCallback (&AnimationInterface::DequeueTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/TxQueue/Dequeue",
                           MakeCallback (&AnimationInterface::DequeueTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/TxQueue/Dequeue",
                           MakeCallback (&AnimationInterface::DequeueTrace, this));
  
  // Queue Drops
  
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::AlohaNoackNetDevice/Queue/Drop",
                           MakeCallback (&AnimationInterface::QueueDropTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/TxQueue/Drop",
                           MakeCallback (&AnimationInterface::QueueDropTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/TxQueue/Drop",
                           MakeCallback (&AnimationInterface::QueueDropTrace, this));
  
  
  // Wifi Mac
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                           MakeCallback (&AnimationInterface::WifiMacTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTxDrop",
                           MakeCallback (&AnimationInterface::WifiMacTxDropTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
                           MakeCallback (&AnimationInterface::WifiMacRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRxDrop",
                           MakeCallback (&AnimationInterface::WifiMacRxDropTrace, this));
  
  // Wifi Phy
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxDrop",
                           MakeCallback (&AnimationInterface::WifiPhyTxDropTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop",
                           MakeCallback (&AnimationInterface::WifiPhyRxDropTrace, this));
  
  // LrWpan
  Config::ConnectFailSafe ("NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Phy/PhyTxBegin",
                           MakeCallback (&AnimationInterface::LrWpanPhyTxBeginTrace, this));
  Config::ConnectFailSafe ("NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Phy/PhyRxBegin",
                           MakeCallback (&AnimationInterface::LrWpanPhyRxBeginTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Mac/MacTx",
                           MakeCallback (&AnimationInterface::LrWpanMacTxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Mac/MacTxDrop",
                           MakeCallback (&AnimationInterface::LrWpanMacTxDropTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Mac/MacRx",
                           MakeCallback (&AnimationInterface::LrWpanMacRxTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LrWpanNetDevice/Mac/MacRxDrop",
                           MakeCallback (&AnimationInterface::LrWpanMacRxDropTrace, this));
  
  // Wave
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/PhyTxBegin",
                           MakeCallback (&AnimationInterface::WavePhyTxBeginTrace, this));
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WaveNetDevice/PhyEntities/*/$ns3::WifiPhy/PhyRxBegin",
                           MakeCallback (&AnimationInterface::WavePhyRxBeginTrace, this));
}

Vector
AnimationInterface::UpdatePosition (Ptr <Node> n)
{
  Ptr<MobilityModel> loc = n->GetObject<MobilityModel> ();
  if (loc)
    {
      m_nodeLocation[n->GetId ()] = loc->GetPosition ();
    }
  else
    {
      NS_LOG_UNCOND ( "AnimationInterface WARNING:Node:" << n->GetId () << " Does not have a mobility model. Use SetConstantPosition if it is stationary");
      Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
      x->SetAttribute ("Min", DoubleValue (0));
      x->SetAttribute ("Max", DoubleValue (100));
      Ptr<UniformRandomVariable> y = CreateObject<UniformRandomVariable> ();
      y->SetAttribute ("Min", DoubleValue (0));
      y->SetAttribute ("Max", DoubleValue (100));
      m_nodeLocation[n->GetId ()] = Vector (int (x->GetValue ()), int (y->GetValue ()), 0);
    }
  return m_nodeLocation[n->GetId ()];
}

Vector
AnimationInterface::UpdatePosition (Ptr <Node> n, Vector v)
{
  m_nodeLocation[n->GetId ()] = v;
  return v;
}

Vector
AnimationInterface::UpdatePosition (Ptr <NetDevice> ndev)
{
  Ptr <Node> n = ndev->GetNode ();
  NS_ASSERT (n);
  return UpdatePosition (n);
}

Vector
AnimationInterface::GetPosition (Ptr <Node> n)
{
  if (m_nodeLocation.find (n->GetId ()) == m_nodeLocation.end ())
    {
      NS_FATAL_ERROR ("Node:" << n->GetId () << " not found in Location table");
    }
  return m_nodeLocation[n->GetId ()];
}


std::string
AnimationInterface::GetMacAddress (Ptr <NetDevice> nd)
{
  Address nodeAddr = nd->GetAddress ();
  std::ostringstream oss;
  oss << nodeAddr;
  return oss.str ().substr (6); // Skip the first 6 chars to get the Mac
}

std::string
AnimationInterface::GetIpv4Address (Ptr <NetDevice> nd)
{
  Ptr<Ipv4> ipv4 = NodeList::GetNode (nd->GetNode ()->GetId ())->GetObject <Ipv4> ();
  if (!ipv4)
    {
      NS_LOG_WARN ("Node: " << nd->GetNode ()->GetId () << " No ipv4 object found");
      return "0.0.0.0";
    }
  int32_t ifIndex = ipv4->GetInterfaceForDevice (nd);
  if (ifIndex == -1)
    {
      NS_LOG_WARN ("Node :" << nd->GetNode ()->GetId () << " Could not find index of NetDevice");
      return "0.0.0.0";
    }
  Ipv4InterfaceAddress addr = ipv4->GetAddress (ifIndex, 0);
  std::ostringstream oss;
  oss << addr.GetLocal ();
  return oss.str ();
}

std::string
AnimationInterface::GetIpv6Address (Ptr <NetDevice> nd)
{
  Ptr<Ipv6> ipv6 = NodeList::GetNode (nd->GetNode ()->GetId ())->GetObject <Ipv6> ();
  if (!ipv6)
    {
      NS_LOG_WARN ("Node: " << nd->GetNode ()->GetId () << " No ipv4 object found");
      return "::";
    }
  int32_t ifIndex = ipv6->GetInterfaceForDevice (nd);
  if (ifIndex == -1)
    {
      NS_LOG_WARN ("Node :" << nd->GetNode ()->GetId () << " Could not find index of NetDevice");
      return "::";
    }
  bool nonLinkLocalFound = false;
  uint32_t nAddresses = ipv6->GetNAddresses (ifIndex);
  Ipv6InterfaceAddress addr;
  for (uint32_t addressIndex = 0; addressIndex < nAddresses; ++addressIndex)
    {
      addr = ipv6->GetAddress (ifIndex, addressIndex);
      if (!addr.GetAddress ().IsLinkLocal ())
        {
          nonLinkLocalFound = true;
          break;
        }
    }
  if (!nonLinkLocalFound)
    {
      addr = ipv6->GetAddress (ifIndex, 0);
    }
  std::ostringstream oss;
  oss << addr.GetAddress ();
  return oss.str ();
}



std::vector<std::string>
AnimationInterface::GetIpv4Addresses (Ptr <NetDevice> nd)
{
  std::vector<std::string> ipv4Addresses;
  Ptr<Ipv4> ipv4 = NodeList::GetNode (nd->GetNode ()->GetId ())->GetObject <Ipv4> ();
  if (!ipv4)
    {
      NS_LOG_WARN ("Node: " << nd->GetNode ()->GetId () << " No ipv4 object found");
      return ipv4Addresses;
    }
  int32_t ifIndex = ipv4->GetInterfaceForDevice (nd);
  if (ifIndex == -1)
    {
      NS_LOG_WARN ("Node :" << nd->GetNode ()->GetId () << " Could not find index of NetDevice");
      return ipv4Addresses;
    }
  for (uint32_t index = 0; index < ipv4->GetNAddresses (ifIndex); ++index)
    {
      Ipv4InterfaceAddress addr = ipv4->GetAddress (ifIndex, index);
      std::ostringstream oss;
      oss << addr.GetLocal ();
      ipv4Addresses.push_back (oss.str ());
    }
  return ipv4Addresses;
}

std::vector<std::string>
AnimationInterface::GetIpv6Addresses (Ptr <NetDevice> nd)
{
  std::vector<std::string> ipv6Addresses;
  Ptr<Ipv6> ipv6 = NodeList::GetNode (nd->GetNode ()->GetId ())->GetObject <Ipv6> ();
  if (!ipv6)
    {
      NS_LOG_WARN ("Node: " << nd->GetNode ()->GetId () << " No ipv6 object found");
      return ipv6Addresses;
    }
  int32_t ifIndex = ipv6->GetInterfaceForDevice (nd);
  if (ifIndex == -1)
    {
      NS_LOG_WARN ("Node :" << nd->GetNode ()->GetId () << " Could not find index of NetDevice");
      return ipv6Addresses;
    }
  for (uint32_t index = 0; index < ipv6->GetNAddresses (ifIndex); ++index)
    {
      Ipv6InterfaceAddress addr = ipv6->GetAddress (ifIndex, index);
      std::ostringstream oss;
      oss << addr.GetAddress ();
      ipv6Addresses.push_back (oss.str ());
    }
  return ipv6Addresses;
}


void
AnimationInterface::WriteIpv4Addresses ()
{
  for (NodeIdIpv4Map::const_iterator i = m_nodeIdIpv4Map.begin ();
       i != m_nodeIdIpv4Map.end ();
       ++i)
    {
      std::vector <std::string> ipv4Addresses;
      std::pair<NodeIdIpv4Map::const_iterator, NodeIdIpv4Map::const_iterator> iterPair = m_nodeIdIpv4Map.equal_range (i->first);
      for (NodeIdIpv4Map::const_iterator it = iterPair.first;
           it != iterPair.second;
           ++it)
        {
          ipv4Addresses.push_back (it->second);
        }
      WriteXmlIpv4Addresses (i->first, ipv4Addresses);
    }
}

void
AnimationInterface::WriteIpv6Addresses ()
{
  for (NodeIdIpv6Map::const_iterator i = m_nodeIdIpv6Map.begin ();
       i != m_nodeIdIpv6Map.end ();
       i = m_nodeIdIpv6Map.upper_bound (i->first))
    {
      std::vector <std::string> ipv6Addresses;
      std::pair<NodeIdIpv6Map::const_iterator, NodeIdIpv6Map::const_iterator> iterPair = m_nodeIdIpv6Map.equal_range (i->first);
      for (NodeIdIpv6Map::const_iterator it = iterPair.first;
           it != iterPair.second;
           ++it)
        {
          ipv6Addresses.push_back (it->second);
        }
      WriteXmlIpv6Addresses (i->first, ipv6Addresses);
    }
}

void
AnimationInterface::WriteLinkProperties ()
{
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      UpdatePosition (n);
      uint32_t n1Id = n->GetId ();
      uint32_t nDev = n->GetNDevices ();  // Number of devices
      for (uint32_t i = 0; i < nDev; ++i)
        {
          Ptr<NetDevice> dev = n->GetDevice (i);
          NS_ASSERT (dev);
          Ptr<Channel>   ch = dev->GetChannel ();
          std::string channelType = "Unknown channel";
          if (ch)
            {
              channelType = ch->GetInstanceTypeId ().GetName ();
            }
          NS_LOG_DEBUG ("Got ChannelType" << channelType);

          if (!ch || (channelType != std::string ("ns3::PointToPointChannel")))
            {
              NS_LOG_DEBUG ("No channel can't be a p2p device");
              /*
              // Try to see if it is an LTE NetDevice, which does not return a channel
              if ((dev->GetInstanceTypeId ().GetName () == "ns3::LteUeNetDevice") ||
                  (dev->GetInstanceTypeId ().GetName () == "ns3::LteEnbNetDevice")||
                  (dev->GetInstanceTypeId ().GetName () == "ns3::VirtualNetDevice"))
                {
                  WriteNonP2pLinkProperties (n->GetId (), GetIpv4Address (dev) + "~" + GetMacAddress (dev), channelType);
                  AddToIpv4AddressNodeIdTable (GetIpv4Address (dev), n->GetId ());
                }
               */
              std::vector<std::string> ipv4Addresses = GetIpv4Addresses (dev);
              AddToIpv4AddressNodeIdTable (ipv4Addresses, n->GetId ());
              std::vector<std::string> ipv6Addresses = GetIpv6Addresses (dev);
              AddToIpv6AddressNodeIdTable (ipv6Addresses, n->GetId ());
              if (!ipv4Addresses.empty ())
                {
                  NS_LOG_INFO ("Writing Ipv4 link");
                  WriteNonP2pLinkProperties (n->GetId (), GetIpv4Address (dev) + "~" + GetMacAddress (dev), channelType);
                }
              else if (!ipv6Addresses.empty ())
                {
                  NS_LOG_INFO ("Writing Ipv6 link");
                  WriteNonP2pLinkProperties (n->GetId (), GetIpv6Address (dev) + "~" + GetMacAddress (dev), channelType);
                }
              continue;
            }

          else if (channelType == std::string ("ns3::PointToPointChannel"))
            { // Since these are duplex links, we only need to dump
              // if srcid < dstid
              std::size_t nChDev = ch->GetNDevices ();
              for (std::size_t j = 0; j < nChDev; ++j)
                {
                  Ptr<NetDevice> chDev = ch->GetDevice (j);
                  uint32_t n2Id = chDev->GetNode ()->GetId ();
                  if (n1Id < n2Id)
                    {

                      std::vector<std::string> ipv4Addresses = GetIpv4Addresses (dev);
                      AddToIpv4AddressNodeIdTable (ipv4Addresses, n1Id);
                      ipv4Addresses = GetIpv4Addresses (chDev);
                      AddToIpv4AddressNodeIdTable (ipv4Addresses, n2Id);
                      std::vector<std::string> ipv6Addresses = GetIpv6Addresses (dev);
                      AddToIpv6AddressNodeIdTable (ipv6Addresses, n1Id);
                      ipv6Addresses = GetIpv6Addresses (chDev);
                      AddToIpv6AddressNodeIdTable (ipv6Addresses, n2Id);

                      P2pLinkNodeIdPair p2pPair;
                      p2pPair.fromNode = n1Id;
                      p2pPair.toNode = n2Id;
                      if (!ipv4Addresses.empty ())
                        {
                          LinkProperties lp = { GetIpv4Address (dev) + "~" + GetMacAddress (dev), GetIpv4Address (chDev) + "~" + GetMacAddress (chDev), "" };
                          m_linkProperties[p2pPair] = lp;
                        }
                      else if (!ipv6Addresses.empty ())
                        {
                          LinkProperties lp = { GetIpv6Address (dev) + "~" + GetMacAddress (dev), GetIpv6Address (chDev) + "~" + GetMacAddress (chDev), "" };
                          m_linkProperties[p2pPair] = lp;
                        }
                      WriteXmlLink (n1Id, 0, n2Id);
                    }
                }
            }
        }
    }
  m_linkProperties.clear ();
}

void
AnimationInterface::WriteNodes ()
{
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      NS_LOG_INFO ("Update Position for Node: " << n->GetId ());
      Vector v = UpdatePosition (n);
      WriteXmlNode (n->GetId (), n->GetSystemId (), v.x, v.y);
    }
}

void
AnimationInterface::WriteNodeColors ()
{
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      Rgb rgb = {255, 0, 0};
      if (m_nodeColors.find (n->GetId ()) == m_nodeColors.end ())
        {
          m_nodeColors[n->GetId ()] = rgb;
        }
      UpdateNodeColor (n, rgb.r, rgb.g, rgb.b);
    }
}

void
AnimationInterface::WriteNodeSizes ()
{
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      NS_LOG_INFO ("Update Size for Node: " << n->GetId ());
      AnimationInterface::NodeSize s = { 1, 1 };
      m_nodeSizes[n->GetId ()] = s;
      UpdateNodeSize (n->GetId (), s.width, s.height);
    }
}

void
AnimationInterface::WriteNodeEnergies ()
{
  m_remainingEnergyCounterId = AddNodeCounter ("RemainingEnergy", AnimationInterface::DOUBLE_COUNTER);
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      Ptr<Node> n = *i;
      if (NodeList::GetNode (n->GetId ())->GetObject<EnergySource> ())
        {
          UpdateNodeCounter (m_remainingEnergyCounterId, n->GetId (), 1);
        }
    }
}

bool
AnimationInterface::IsInTimeWindow ()
{
  if ((Simulator::Now () >= m_startTime)
      && (Simulator::Now () <= m_stopTime))
    {
      return true;
    }
  else
    {
      return false;
    }
}

void
AnimationInterface::SetOutputFile (const std::string& fn, bool routing)
{
  if (!routing && m_f)
    {
      return;
    }
  if (routing && m_routingF)
    {
      NS_FATAL_ERROR ("SetRoutingOutputFile already used once");
      return;
    }

  NS_LOG_INFO ("Creating new trace file:" << fn.c_str ());
  FILE * f = 0;
  f = std::fopen (fn.c_str (), "w");
  if (!f)
    {
      NS_FATAL_ERROR ("Unable to open output file:" << fn.c_str ());
      return; // Can't open output file
    }
  if (routing)
    {
      m_routingF = f;
      m_routingFileName = fn;
    }
  else
    {
      m_f = f;
      m_outputFileName = fn;
    }
  return;
}

void
AnimationInterface::CheckMaxPktsPerTraceFile ()
{
  // Start a new trace file if the current packet count exceeded nax packets per file
  ++m_currentPktCount;
  if (m_currentPktCount <= m_maxPktsPerFile)
    {
      return;
    }
  NS_LOG_UNCOND ("Max Packets per trace file exceeded");
  StopAnimation (true);
}

std::string
AnimationInterface::GetNetAnimVersion ()
{
  return NETANIM_VERSION;
}


void
AnimationInterface::TrackQueueCounters ()
{
  if (Simulator::Now () > m_queueCountersStopTime)
    {
      NS_LOG_INFO ("TrackQueueCounters Completed");
      return;
    }
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      uint32_t nodeId = Ptr <Node> (*i)->GetId ();
      UpdateNodeCounter (m_queueEnqueueCounterId, nodeId, m_nodeQueueEnqueue[nodeId]);
      UpdateNodeCounter (m_queueDequeueCounterId, nodeId, m_nodeQueueDequeue[nodeId]);
      UpdateNodeCounter (m_queueDropCounterId, nodeId, m_nodeQueueDrop[nodeId]);
    }
  Simulator::Schedule (m_queueCountersPollInterval, &AnimationInterface::TrackQueueCounters, this);
}

void
AnimationInterface::TrackWifiMacCounters ()
{
  if (Simulator::Now () > m_wifiMacCountersStopTime)
    {
      NS_LOG_INFO ("TrackWifiMacCounters Completed");
      return;
    }
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      uint32_t nodeId = Ptr <Node> (*i)->GetId ();
      UpdateNodeCounter (m_wifiMacTxCounterId, nodeId, m_nodeWifiMacTx[nodeId]);
      UpdateNodeCounter (m_wifiMacTxDropCounterId, nodeId, m_nodeWifiMacTxDrop[nodeId]);
      UpdateNodeCounter (m_wifiMacRxCounterId, nodeId, m_nodeWifiMacRx[nodeId]);
      UpdateNodeCounter (m_wifiMacRxDropCounterId, nodeId, m_nodeWifiMacRxDrop[nodeId]);
    }
  Simulator::Schedule (m_wifiMacCountersPollInterval, &AnimationInterface::TrackWifiMacCounters, this);
}

void
AnimationInterface::TrackWifiPhyCounters ()
{
  if (Simulator::Now () > m_wifiPhyCountersStopTime)
    {
      NS_LOG_INFO ("TrackWifiPhyCounters Completed");
      return;
    }
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      uint32_t nodeId = Ptr <Node> (*i)->GetId ();
      UpdateNodeCounter (m_wifiPhyTxDropCounterId, nodeId, m_nodeWifiPhyTxDrop[nodeId]);
      UpdateNodeCounter (m_wifiPhyRxDropCounterId, nodeId, m_nodeWifiPhyRxDrop[nodeId]);
    }
  Simulator::Schedule (m_wifiPhyCountersPollInterval, &AnimationInterface::TrackWifiPhyCounters, this);
}

void
AnimationInterface::TrackIpv4L3ProtocolCounters ()
{
  if (Simulator::Now () > m_ipv4L3ProtocolCountersStopTime)
    {
      NS_LOG_INFO ("TrackIpv4L3ProtocolCounters Completed");
      return;
    }
  for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
    {
      uint32_t nodeId = Ptr <Node> (*i)->GetId ();
      UpdateNodeCounter (m_ipv4L3ProtocolTxCounterId, nodeId, m_nodeIpv4Tx[nodeId]);
      UpdateNodeCounter (m_ipv4L3ProtocolRxCounterId, nodeId, m_nodeIpv4Rx[nodeId]);
      UpdateNodeCounter (m_ipv4L3ProtocolDropCounterId, nodeId, m_nodeIpv4Drop[nodeId]);
    }
  Simulator::Schedule (m_ipv4L3ProtocolCountersPollInterval, &AnimationInterface::TrackIpv4L3ProtocolCounters, this);
}





/***** Routing-related *****/

void
AnimationInterface::TrackIpv4RoutePaths ()
{
  if (m_ipv4RouteTrackElements.empty ())
    {
      return;
    }
  for (std::vector <Ipv4RouteTrackElement>::const_iterator i = m_ipv4RouteTrackElements.begin ();
       i != m_ipv4RouteTrackElements.end ();
       ++i)
    {
      Ipv4RouteTrackElement trackElement = *i;
      Ptr <Node> fromNode = NodeList::GetNode (trackElement.fromNodeId);
      if (!fromNode)
        {
          NS_FATAL_ERROR ("Node: " << trackElement.fromNodeId << " Not found");
          continue;
        }
      Ptr <ns3::Ipv4> ipv4 = fromNode->GetObject <ns3::Ipv4> ();
      if (!ipv4)
        {
          NS_LOG_WARN ("ipv4 object not found");
          continue;
        }
      Ptr <Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol ();
      if (!rp)
        {
          NS_LOG_WARN ("Routing protocol object not found");
          continue;
        }
      NS_LOG_INFO ("Begin Track Route for: " << trackElement.destination.c_str () << " From:" << trackElement.fromNodeId);
      Ptr<Packet> pkt = Create<Packet> ();
      Ipv4Header header;
      header.SetDestination (Ipv4Address (trackElement.destination.c_str ()));
      Socket::SocketErrno sockerr;
      Ptr <Ipv4Route> rt = rp->RouteOutput (pkt, header, 0, sockerr);
      Ipv4RoutePathElements rpElements;
      if (!rt)
        {
          NS_LOG_INFO ("No route to :" << trackElement.destination.c_str ());
          Ipv4RoutePathElement elem = { trackElement.fromNodeId, "-1" };
          rpElements.push_back (elem);
          WriteRoutePath (trackElement.fromNodeId, trackElement.destination, rpElements);
          continue;
        }
      std::ostringstream oss;
      oss << rt->GetGateway ();
      NS_LOG_INFO ("Node:" << trackElement.fromNodeId << "-->" << rt->GetGateway ());
      if (rt->GetGateway () == "0.0.0.0")
        {
          Ipv4RoutePathElement elem = { trackElement.fromNodeId, "C" };
          rpElements.push_back (elem);
          if ( m_ipv4ToNodeIdMap.find (trackElement.destination) != m_ipv4ToNodeIdMap.end ())
            {
              Ipv4RoutePathElement elem2 = { m_ipv4ToNodeIdMap[trackElement.destination], "L" };
              rpElements.push_back (elem2);
            }
        }
      else if (rt->GetGateway () == "127.0.0.1")
        {
          Ipv4RoutePathElement elem = { trackElement.fromNodeId, "-1" };
          rpElements.push_back (elem);
        }
      else
        {
          Ipv4RoutePathElement elem = { trackElement.fromNodeId, oss.str () };
          rpElements.push_back (elem);
        }
      RecursiveIpv4RoutePathSearch (oss.str (), trackElement.destination, rpElements);
      WriteRoutePath (trackElement.fromNodeId, trackElement.destination, rpElements);
    }

}

void
AnimationInterface::TrackIpv4Route ()
{
  if (Simulator::Now () > m_routingStopTime)
    {
      NS_LOG_INFO ("TrackIpv4Route completed");
      return;
    }
  if (m_routingNc.GetN ())
    {
      for (NodeContainer::Iterator i = m_routingNc.Begin (); i != m_routingNc.End (); ++i)
        {
          Ptr <Node> n = *i;
          WriteXmlRouting (n->GetId (), GetIpv4RoutingTable (n));
        }
    }
  else
    {
      for (NodeList::Iterator i = NodeList::Begin (); i != NodeList::End (); ++i)
        {
          Ptr <Node> n = *i;
          WriteXmlRouting (n->GetId (), GetIpv4RoutingTable (n));
        }
    }
  TrackIpv4RoutePaths ();
  Simulator::Schedule (m_routingPollInterval, &AnimationInterface::TrackIpv4Route, this);
}

std::string
AnimationInterface::GetIpv4RoutingTable (Ptr <Node> n)
{

  NS_ASSERT (n);
  Ptr <ns3::Ipv4> ipv4 = n->GetObject <ns3::Ipv4> ();
  if (!ipv4)
    {
      NS_LOG_WARN ("Node " << n->GetId () << " Does not have an Ipv4 object");
      return "";
    }
  std::stringstream stream;
  Ptr<OutputStreamWrapper> routingstream = Create<OutputStreamWrapper> (&stream);
  ipv4->GetRoutingProtocol ()->PrintRoutingTable (routingstream);
  return stream.str ();

}

void
AnimationInterface::RecursiveIpv4RoutePathSearch (std::string from, std::string to, Ipv4RoutePathElements & rpElements)
{
  NS_LOG_INFO ("RecursiveIpv4RoutePathSearch from:" << from.c_str () << " to:" << to.c_str ());
  if ((from == "0.0.0.0") || (from == "127.0.0.1"))
    {
      NS_LOG_INFO ("Got " << from.c_str () << " End recursion");
      return;
    }
  Ptr <Node> fromNode = NodeList::GetNode (m_ipv4ToNodeIdMap[from]);
  Ptr <Node> toNode = NodeList::GetNode (m_ipv4ToNodeIdMap[to]);
  if (fromNode->GetId () == toNode->GetId ())
    {
      Ipv4RoutePathElement elem = { fromNode->GetId (), "L" };
      rpElements.push_back (elem);
      return;
    }
  if (!fromNode)
    {
      NS_FATAL_ERROR ("Node: " << m_ipv4ToNodeIdMap[from] << " Not found");
      return;
    }
  if (!toNode)
    {
      NS_FATAL_ERROR ("Node: " << m_ipv4ToNodeIdMap[to] << " Not found");
      return;
    }
  Ptr <ns3::Ipv4> ipv4 = fromNode->GetObject <ns3::Ipv4> ();
  if (!ipv4)
    {
      NS_LOG_WARN ("ipv4 object not found");
      return;
    }
  Ptr <Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol ();
  if (!rp)
    {
      NS_LOG_WARN ("Routing protocol object not found");
      return;
    }
  Ptr<Packet> pkt = Create<Packet> ();
  Ipv4Header header;
  header.SetDestination (Ipv4Address (to.c_str ()));
  Socket::SocketErrno sockerr;
  Ptr <Ipv4Route> rt = rp->RouteOutput (pkt, header, 0, sockerr);
  if (!rt)
    {
      return;
    }
  NS_LOG_DEBUG ("Node: " << fromNode->GetId () << " G:" << rt->GetGateway ());
  std::ostringstream oss;
  oss << rt->GetGateway ();
  if (oss.str () == "0.0.0.0" && (sockerr != Socket::ERROR_NOROUTETOHOST))
    {
      NS_LOG_INFO ("Null gw");
      Ipv4RoutePathElement elem = { fromNode->GetId (), "C" };
      rpElements.push_back (elem);
      if ( m_ipv4ToNodeIdMap.find (to) != m_ipv4ToNodeIdMap.end ())
        {
          Ipv4RoutePathElement elem2 = { m_ipv4ToNodeIdMap[to], "L" };
          rpElements.push_back (elem2);
        }
      return;
    }
  NS_LOG_INFO ("Node:" << fromNode->GetId () << "-->" << rt->GetGateway ());
  Ipv4RoutePathElement elem = { fromNode->GetId (), oss.str () };
  rpElements.push_back (elem);
  RecursiveIpv4RoutePathSearch (oss.str (), to, rpElements);

}



/***** WriteXml *****/

void
AnimationInterface::WriteXmlAnim (bool routing)
{
  AnimXmlElement element ("anim");
  element.AddAttribute ("ver", GetNetAnimVersion ());
  FILE * f = m_f;
  if (!routing)
    {
      element.AddAttribute ("filetype", "animation");
    }
  else
    {
      element.AddAttribute ("filetype", "routing");
      f = m_routingF;
    }
  WriteN (element.ToString (false) + ">\n", f);
}

void
AnimationInterface::WriteXmlClose (std::string name, bool routing)
{
  std::string closeString = "</" + name + ">\n";
  if (!routing)
    {
      WriteN (closeString, m_f);
    }
  else
    {
      WriteN (closeString, m_routingF);
    }
}

void
AnimationInterface::WriteXmlNode (uint32_t id, uint32_t sysId, double locX, double locY)
{
  AnimXmlElement element ("node");
  element.AddAttribute ("id", id);
  element.AddAttribute ("sysId", sysId);
  element.AddAttribute ("locX", locX);
  element.AddAttribute ("locY", locY);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateLink (uint32_t fromId, uint32_t toId, std::string linkDescription)
{
  AnimXmlElement element ("linkupdate");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("fromId", fromId);
  element.AddAttribute ("toId", toId);
  element.AddAttribute ("ld", linkDescription, true);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlLink (uint32_t fromId, uint32_t toLp, uint32_t toId)
{
  AnimXmlElement element ("link");
  element.AddAttribute ("fromId", fromId);
  element.AddAttribute ("toId", toId);

  LinkProperties lprop;
  lprop.fromNodeDescription = "";
  lprop.toNodeDescription = "";
  lprop.linkDescription = "";

  P2pLinkNodeIdPair p1 = { fromId, toId };
  P2pLinkNodeIdPair p2 = { toId, fromId };
  if (m_linkProperties.find (p1) != m_linkProperties.end ())
    {
      lprop = m_linkProperties[p1];
    }
  else if (m_linkProperties.find (p2) != m_linkProperties.end ())
    {
      lprop = m_linkProperties[p2];
    }

  element.AddAttribute ("fd", lprop.fromNodeDescription, true);
  element.AddAttribute ("td", lprop.toNodeDescription, true);
  element.AddAttribute ("ld", lprop.linkDescription, true);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlIpv4Addresses (uint32_t nodeId, std::vector<std::string> ipv4Addresses)
{
  AnimXmlElement element ("ip");
  element.AddAttribute ("n", nodeId);
  for (std::vector<std::string>::const_iterator i = ipv4Addresses.begin ();
       i != ipv4Addresses.end ();
       ++i)
    {
      AnimXmlElement valueElement ("address");
      valueElement.SetText (*i);
      element.AppendChild (valueElement);
    }
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlIpv6Addresses (uint32_t nodeId, std::vector<std::string> ipv6Addresses)
{
  AnimXmlElement element ("ipv6");
  element.AddAttribute ("n", nodeId);
  for (std::vector<std::string>::const_iterator i = ipv6Addresses.begin ();
       i != ipv6Addresses.end ();
       ++i)
    {
      AnimXmlElement valueElement ("address");
      valueElement.SetText (*i);
      element.AppendChild (valueElement);
    }
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlRouting (uint32_t nodeId, std::string routingInfo)
{
  AnimXmlElement element ("rt");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("info", routingInfo.c_str (), true);
  WriteN (element.ToString (), m_routingF);
}

void
AnimationInterface::WriteXmlRp (uint32_t nodeId, std::string destination, Ipv4RoutePathElements rpElements)
{
  std::string tagName = "rp";
  AnimXmlElement element (tagName, false);
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("d", destination.c_str ());
  element.AddAttribute ("c", rpElements.size ());
  for (Ipv4RoutePathElements::const_iterator i = rpElements.begin ();
       i != rpElements.end ();
       ++i)
    {
      Ipv4RoutePathElement rpElement = *i;
      AnimXmlElement rpeElement ("rpe");
      rpeElement.AddAttribute ("n", rpElement.nodeId);
      rpeElement.AddAttribute ("nH", rpElement.nextHop.c_str ());
      element.AppendChild (rpeElement);
    }
  WriteN (element.ToString (),  m_routingF);
}


void
AnimationInterface::WriteXmlPRef (uint64_t animUid, uint32_t fId, double fbTx, std::string metaInfo)
{
  AnimXmlElement element ("pr");
  element.AddAttribute ("uId", animUid);
  element.AddAttribute ("fId", fId);
  element.AddAttribute ("fbTx", fbTx);
  if (!metaInfo.empty ())
    {
      element.AddAttribute ("meta-info", metaInfo.c_str (), true);
    }
  WriteN (element.ToString (),  m_f);
}

void
AnimationInterface::WriteXmlP (uint64_t animUid, std::string pktType, uint32_t tId, double fbRx, double lbRx)
{
  AnimXmlElement element (pktType);
  element.AddAttribute ("uId", animUid);
  element.AddAttribute ("tId", tId);
  element.AddAttribute ("fbRx", fbRx);
  element.AddAttribute ("lbRx", lbRx);
  WriteN (element.ToString (),  m_f);
}

void
AnimationInterface::WriteXmlP (std::string pktType, uint32_t fId, double fbTx, double lbTx,
                               uint32_t tId, double fbRx, double lbRx, std::string metaInfo)
{
  AnimXmlElement element (pktType);
  element.AddAttribute ("fId", fId);
  element.AddAttribute ("fbTx", fbTx);
  element.AddAttribute ("lbTx", lbTx);
  if (!metaInfo.empty ())
    {
      element.AddAttribute ("meta-info", metaInfo.c_str (), true);
    }
  element.AddAttribute ("tId", tId);
  element.AddAttribute ("fbRx", fbRx);
  element.AddAttribute ("lbRx", lbRx);
  WriteN (element.ToString (),  m_f);
}

void
AnimationInterface::WriteXmlAddNodeCounter (uint32_t nodeCounterId, std::string counterName, CounterType counterType)
{
  AnimXmlElement element ("ncs");
  element.AddAttribute ("ncId", nodeCounterId);
  element.AddAttribute ("n", counterName);
  element.AddAttribute ("t", CounterTypeToString (counterType));
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlAddResource (uint32_t resourceId, std::string resourcePath)
{
  AnimXmlElement element ("res");
  element.AddAttribute ("rid", resourceId);
  element.AddAttribute ("p", resourcePath);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateNodeImage (uint32_t nodeId, uint32_t resourceId)
{
  AnimXmlElement element ("nu");
  element.AddAttribute ("p", "i");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("rid", resourceId);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateNodeSize (uint32_t nodeId, double width, double height)
{
  AnimXmlElement element ("nu");
  element.AddAttribute ("p", "s");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("w", width);
  element.AddAttribute ("h", height);
  WriteN (element.ToString (),  m_f);
}

void
AnimationInterface::WriteXmlUpdateNodePosition (uint32_t nodeId, double x, double y)
{
  AnimXmlElement element ("nu");
  element.AddAttribute ("p", "p");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("x", x);
  element.AddAttribute ("y", y);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateNodeColor (uint32_t nodeId, uint8_t r, uint8_t g, uint8_t b)
{
  AnimXmlElement element ("nu");
  element.AddAttribute ("p", "c");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  element.AddAttribute ("r", (uint32_t) r);
  element.AddAttribute ("g", (uint32_t) g);
  element.AddAttribute ("b", (uint32_t) b);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateNodeDescription (uint32_t nodeId)
{
  AnimXmlElement element ("nu");
  element.AddAttribute ("p", "d");
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("id", nodeId);
  if (m_nodeDescriptions.find (nodeId) != m_nodeDescriptions.end ())
    {
      element.AddAttribute ("descr", m_nodeDescriptions[nodeId], true);
    }
  WriteN (element.ToString (), m_f);
}


void
AnimationInterface::WriteXmlUpdateNodeCounter (uint32_t nodeCounterId, uint32_t nodeId, double counterValue)
{
  AnimXmlElement element ("nc");
  element.AddAttribute ("c", nodeCounterId);
  element.AddAttribute ("i", nodeId);
  element.AddAttribute ("t", Simulator::Now ().GetSeconds ());
  element.AddAttribute ("v", counterValue);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlUpdateBackground (std::string fileName, double x, double y, double scaleX, double scaleY, double opacity)
{
  AnimXmlElement element ("bg");
  element.AddAttribute ("f", fileName);
  element.AddAttribute ("x", x);
  element.AddAttribute ("y", y);
  element.AddAttribute ("sx", scaleX);
  element.AddAttribute ("sy", scaleY);
  element.AddAttribute ("o", opacity);
  WriteN (element.ToString (), m_f);
}

void
AnimationInterface::WriteXmlNonP2pLinkProperties (uint32_t id, std::string ipAddress, std::string channelType)
{
  AnimXmlElement element ("nonp2plinkproperties");
  element.AddAttribute ("id", id);
  element.AddAttribute ("ipAddress", ipAddress);
  element.AddAttribute ("channelType", channelType);
  WriteN (element.ToString (), m_f);
}



/***** AnimXmlElement  *****/

AnimationInterface::AnimXmlElement::AnimXmlElement (std::string tagName, bool emptyElement)
  : m_tagName (tagName),
    m_text ("")
{
}

template <typename T>
void
AnimationInterface::AnimXmlElement::AddAttribute (std::string attribute, T value, bool xmlEscape)
{
  std::ostringstream oss;
  oss << std::setprecision (10);
  oss << value;
  std::string attributeString = attribute.c_str ();
  if (xmlEscape)
    {
      attributeString += "=\"";
      std::string valueStr = oss.str ();
      for (std::string::iterator it = valueStr.begin (); it != valueStr.end (); ++it)
        {
          switch (*it)
            {
            case '&':
              attributeString += "&amp;";
              break;
            case '\"':
              attributeString += "&quot;";
              break;
            case '\'':
              attributeString += "&apos;";
              break;
            case '<':
              attributeString += "&lt;";
              break;
            case '>':
              attributeString += "&gt;";
              break;
            default:
              attributeString += *it;
              break;
            }
        }
      attributeString += "\" ";
    }
  else
    {
      attributeString += "=\"" + oss.str () + "\" ";
    }
  m_attributes.push_back (attributeString);
}

void
AnimationInterface::AnimXmlElement::AppendChild (AnimXmlElement e)
{
  m_children.push_back (e.ToString ());
}

void
AnimationInterface::AnimXmlElement::SetText (std::string text)
{
  m_text = text;
}

std::string
AnimationInterface::AnimXmlElement::ToString (bool autoClose)
{
  std::string elementString = "<" + m_tagName + " ";


  for (std::vector<std::string>::const_iterator i = m_attributes.begin ();
       i != m_attributes.end ();
       ++i)
    {
      elementString += *i;
    }
  if (m_children.empty () && m_text.empty ())
    {
      if (autoClose)
        {
          elementString += "/>";
        }
    }
  else
    {
      elementString += ">";
      if (!m_text.empty ())
        {
          elementString += m_text;
        }
      if (!m_children.empty ())
        {
          elementString += "\n";
          for (std::vector<std::string>::const_iterator i = m_children.begin ();
               i != m_children.end ();
               ++i)
            {
              elementString += *i + "\n";
            }

        }
      if (autoClose)
        {
          elementString += "</" + m_tagName + ">";
        }
    }


  return elementString + ((autoClose) ? "\n" : "");
}





/***** AnimByteTag *****/

TypeId
AnimByteTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AnimByteTag")
    .SetParent<Tag> ()
    .SetGroupName ("NetAnim")
    .AddConstructor<AnimByteTag> ()
  ;
  return tid;
}

TypeId
AnimByteTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
AnimByteTag::GetSerializedSize (void) const
{
  return sizeof (uint64_t);
}

void
AnimByteTag::Serialize (TagBuffer i) const
{
  i.WriteU64 (m_AnimUid);
}

void
AnimByteTag::Deserialize (TagBuffer i)
{
  m_AnimUid = i.ReadU64 ();
}

void
AnimByteTag::Print (std::ostream &os) const
{
  os << "AnimUid=" << m_AnimUid;
}

void
AnimByteTag::Set (uint64_t AnimUid)
{
  m_AnimUid = AnimUid;
}

uint64_t
AnimByteTag::Get (void) const
{
  return m_AnimUid;
}

AnimationInterface::AnimPacketInfo::AnimPacketInfo ()
  : m_txnd (0),
    m_txNodeId (0),
    m_fbTx (0),
    m_lbTx (0),
    m_lbRx (0)
{
}

AnimationInterface::AnimPacketInfo::AnimPacketInfo (const AnimPacketInfo & pInfo)
{
  m_txnd = pInfo.m_txnd;
  m_txNodeId = pInfo.m_txNodeId;
  m_fbTx = pInfo.m_fbTx;
  m_lbTx = pInfo.m_lbTx;
  m_lbRx = pInfo.m_lbRx;
}

AnimationInterface::AnimPacketInfo::AnimPacketInfo (Ptr <const NetDevice> txnd,
                                                    const Time fbTx,
                                                    uint32_t txNodeId)
  : m_txnd (txnd),
    m_txNodeId (0),
    m_fbTx (fbTx.GetSeconds ()),
    m_lbTx (0),
    m_lbRx (0)
{
  if (!m_txnd)
    {
      m_txNodeId = txNodeId;
    }
}

void
AnimationInterface::AnimPacketInfo::ProcessRxBegin (Ptr<const NetDevice> nd, const double fbRx)
{
  Ptr <Node> n = nd->GetNode ();
  m_fbRx = fbRx;
  m_rxnd = nd;
}

} // namespace ns3
