/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INESC Porto
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
 * Author: Gustavo Carneiro  <gjc@inescporto.pt>
 */

#include <cstdlib>
#include "pyviz.h"
#include "ns3/simulator.h"
#include "ns3/config.h"
#include "ns3/node-list.h"
#include "ns3/wifi-net-device.h"
#include "ns3/ppp-header.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/ethernet-header.h"
#include "ns3/log.h"
#include "ns3/abort.h"

#include "visual-simulator-impl.h"

#include <sstream>

NS_LOG_COMPONENT_DEFINE ("PyViz");

#define NUM_LAST_PACKETS 10

static
std::vector<std::string>
PathSplit (std::string str)
{
  std::vector<std::string> results;
  size_t cutAt;
  while ((cutAt = str.find_first_of ('/')) != str.npos)
    {
      if(cutAt > 0)
        {
          results.push_back (str.substr (0,cutAt));
        }
      str = str.substr (cutAt+1);
    }
  if (str.length () > 0)
    {
      results.push_back (str);
    }
  return results;
}


namespace ns3 {

static PyViz* g_visualizer = NULL; ///< the visualizer


/**
 * PyVizPacketTag structure
 */
struct PyVizPacketTag : public Tag
{
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer buf) const;
  virtual void Deserialize (TagBuffer buf);
  virtual void Print (std::ostream &os) const;
  PyVizPacketTag ();

  uint32_t m_packetId; ///< packet id
};


/**
 * \brief Get the type ID.
 * \return the object TypeId
 */
TypeId 
PyVizPacketTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PyVizPacketTag")
    .SetParent<Tag> ()
    .SetGroupName ("Visualizer")
    .AddConstructor<PyVizPacketTag> ()
  ;
  return tid;
}
TypeId 
PyVizPacketTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t 
PyVizPacketTag::GetSerializedSize (void) const
{
  return 4;
}
void 
PyVizPacketTag::Serialize (TagBuffer buf) const
{
  buf.WriteU32 (m_packetId);
}
void 
PyVizPacketTag::Deserialize (TagBuffer buf)
{
  m_packetId = buf.ReadU32 ();
}
void 
PyVizPacketTag::Print (std::ostream &os) const
{
  os << "PacketId=" << m_packetId;
}
PyVizPacketTag::PyVizPacketTag ()
  : Tag () 
{
}



PyViz::PyViz ()
{
  NS_LOG_FUNCTION_NOARGS ();
  NS_ASSERT (g_visualizer == NULL);
  g_visualizer = this;

  // WiFi
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacTx",
                           MakeCallback (&PyViz::TraceNetDevTxWifi, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/MacRx",
                           MakeCallback (&PyViz::TraceNetDevRxWifi, this));

  // CSMA
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/MacTx",
                           MakeCallback (&PyViz::TraceNetDevTxCsma, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/MacRx",
                           MakeCallback (&PyViz::TraceNetDevRxCsma, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::CsmaNetDevice/MacPromiscRx",
                           MakeCallback (&PyViz::TraceNetDevPromiscRxCsma, this));

  // Generic queue drop
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/TxQueue/Drop",
                           MakeCallback (&PyViz::TraceDevQueueDrop, this));
  // IPv4 drop
  Config::ConnectFailSafe ("/NodeList/*/$ns3::Ipv4L3Protocol/Drop",
                           MakeCallback (&PyViz::TraceIpv4Drop, this));

  // Point-to-Point
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/MacTx",
                           MakeCallback (&PyViz::TraceNetDevTxPointToPoint, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::PointToPointNetDevice/MacRx",
                           MakeCallback (&PyViz::TraceNetDevRxPointToPoint, this));

  // WiMax
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WimaxNetDevice/Tx",
                           MakeCallback (&PyViz::TraceNetDevTxWimax, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::WimaxNetDevice/Rx",
                           MakeCallback (&PyViz::TraceNetDevRxWimax, this));

  // LTE
  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/Tx",
                           MakeCallback (&PyViz::TraceNetDevTxLte, this));

  Config::ConnectFailSafe ("/NodeList/*/DeviceList/*/$ns3::LteNetDevice/Rx",
                           MakeCallback (&PyViz::TraceNetDevRxLte, this));
}

void
PyViz::RegisterCsmaLikeDevice (std::string const &deviceTypeName)
{
  TypeId::LookupByName (deviceTypeName); // this will assert if the type name is invalid

  std::ostringstream sstream;
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/MacTx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevTxCsma, this));

  sstream.str ("");
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/Rx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevRxCsma, this));

  sstream.str ("");
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/PromiscRx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevPromiscRxCsma, this));
}

void
PyViz::RegisterWifiLikeDevice (std::string const &deviceTypeName)
{
  TypeId::LookupByName (deviceTypeName); // this will assert if the type name is invalid

  std::ostringstream sstream;
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/Tx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevTxWifi, this));

  sstream.str ("");
  sstream <<"/NodeList/*/DeviceList/*/$" << deviceTypeName << "/Rx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevRxWifi, this));
}

void
PyViz::RegisterPointToPointLikeDevice (std::string const &deviceTypeName)
{
  TypeId::LookupByName (deviceTypeName); // this will assert if the type name is invalid

  std::ostringstream sstream;
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/TxQueue/Dequeue";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevTxPointToPoint, this));

  sstream.str ("");
  sstream << "/NodeList/*/DeviceList/*/$" << deviceTypeName << "/Rx";
  Config::Connect (sstream.str (), MakeCallback (&PyViz::TraceNetDevRxPointToPoint, this));
}

void
PyViz::SetPacketCaptureOptions (uint32_t nodeId, PacketCaptureOptions options)
{
  NS_LOG_DEBUG ("  SetPacketCaptureOptions " << nodeId
                                             << " PacketCaptureOptions (headers size = " << options.headers.size ()
                                             << " mode = " << options.mode << " numLastPackets = " << options.numLastPackets
                                             << ")");
  m_packetCaptureOptions[nodeId] = options;
}

void
PyViz::RegisterDropTracePath (std::string const &tracePath)
{
  Config::Connect (tracePath, MakeCallback (&PyViz::TraceDevQueueDrop, this));
}

PyViz::~PyViz ()
{
  NS_LOG_FUNCTION_NOARGS ();

  NS_ASSERT (g_visualizer == this);
  g_visualizer = NULL;
}

void PyViz::DoPause (std::string const &message)
{
  m_pauseMessages.push_back (message);
  m_stop = true;
  NS_LOG_LOGIC (Simulator::Now ().As (Time::S) << ": Have "
                                                << g_visualizer->m_pauseMessages.size () << " pause messages");
}

void PyViz::Pause (std::string const &message)
{
  NS_ASSERT (g_visualizer);
  g_visualizer->DoPause (message);
}

std::vector<std::string>
PyViz::GetPauseMessages () const
{
  NS_LOG_LOGIC (Simulator::Now ().As (Time::S) << ": GetPauseMessages: have "
                                                << g_visualizer->m_pauseMessages.size () << " pause messages");
  return m_pauseMessages;
}


void
PyViz::CallbackStopSimulation ()
{
  NS_LOG_FUNCTION_NOARGS ();
  if (m_runUntil <= Simulator::Now ())
    {
      Simulator::Stop (Seconds (0)); // Stop right now
      m_stop = true;
    }
}

void
PyViz::SimulatorRunUntil (Time time)
{
  NS_LOG_LOGIC ("SimulatorRunUntil " << time << " (now is " << Simulator::Now () << ")");

  m_pauseMessages.clear ();
  m_transmissionSamples.clear ();
  m_packetDrops.clear ();

  Time expirationTime = Simulator::Now () - Seconds (10);

  // Clear very old transmission records
  for (std::map<TxRecordKey, TxRecordValue>::iterator iter = m_txRecords.begin ();
       iter != m_txRecords.end ();)
    {
      if (iter->second.time < expirationTime)
        {
          m_txRecords.erase (iter++);
        }
      else
        {
          iter++;
        }
    }

  // Clear very old packets of interest
  for (std::map<uint32_t, Time>::iterator iter = m_packetsOfInterest.begin ();
       iter != m_packetsOfInterest.end ();)
    {
      if (iter->second < expirationTime)
        {
          m_packetsOfInterest.erase (iter++);
        }
      else
        {
          iter++;
        }
    }

  if (Simulator::Now () >= time)
    {
      return;
    }
  // Schedule a dummy callback function for the target time, to make
  // sure we stop at the right time.  Otherwise, simulations with few
  // events just appear to "jump" big chunks of time.
  NS_LOG_LOGIC ("Schedule dummy callback to be called in " << (time - Simulator::Now ()));
  m_runUntil = time;
  m_stop = false;
  Simulator::ScheduleWithContext (Simulator::NO_CONTEXT, time - Simulator::Now (), &PyViz::CallbackStopSimulation, this);

  Ptr<SimulatorImpl> impl = Simulator::GetImplementation ();
  Ptr<VisualSimulatorImpl> visualImpl = DynamicCast<VisualSimulatorImpl> (impl);
  if (visualImpl)
    {
      visualImpl->RunRealSimulator ();
    }
  else
    {
      impl->Run ();
    }
}

bool PyViz::TransmissionSampleKey::operator < (PyViz::TransmissionSampleKey const &other) const
{
  if (this->transmitter < other.transmitter)
    {
      return true;
    }
  if (this->transmitter != other.transmitter)
    {
      return false;
    }
  if (this->receiver < other.receiver)
    {
      return true;
    }
  if (this->receiver != other.receiver)
    {
      return false;
    }
  if (this->channel < other.channel)
    {
      return true;
    }
  else
    {
      return false;
    }
}

bool PyViz::TransmissionSampleKey::operator == (PyViz::TransmissionSampleKey const &other) const
{
  bool retval = (transmitter == other.transmitter) &&
    (receiver == other.receiver) &&
    (channel == other.channel);
  return retval;
}


PyViz::NetDeviceStatistics &
PyViz::FindNetDeviceStatistics (int node, int interface)
{
  std::map<uint32_t, std::vector<NetDeviceStatistics> >::iterator nodeStatsIter = m_nodesStatistics.find (node);
  std::vector<NetDeviceStatistics> *stats;
  if (nodeStatsIter == m_nodesStatistics.end ())
    {
      stats = &m_nodesStatistics[node];
      stats->resize (NodeList::GetNode (node)->GetNDevices ());
    }
  else
    {
      stats = &(nodeStatsIter->second);
    }
  NetDeviceStatistics &devStats = (*stats)[interface];
  return devStats;
}

bool PyViz::GetPacketCaptureOptions (uint32_t nodeId, const PacketCaptureOptions **outOptions) const
{
  std::map<uint32_t, PacketCaptureOptions>::const_iterator iter = m_packetCaptureOptions.find (nodeId);
  if (iter == m_packetCaptureOptions.end ())
    {
      return false;
    }
  else
    {
      *outOptions = &iter->second;
      return true;
    }
}

bool PyViz::FilterPacket (Ptr<const Packet> packet, const PacketCaptureOptions &options)
{
  switch (options.mode)
    {
    case PACKET_CAPTURE_DISABLED:
      return false;

    case PACKET_CAPTURE_FILTER_HEADERS_OR:
      {
        PacketMetadata::ItemIterator metadataIterator = packet->BeginItem ();
        while (metadataIterator.HasNext ())
          {
            PacketMetadata::Item item = metadataIterator.Next ();
            if (options.headers.find (item.tid) != options.headers.end ())
              {
                return true;
              }
          }
        return false;
      }

    case PACKET_CAPTURE_FILTER_HEADERS_AND:
      {
        std::set<TypeId> missingHeaders (options.headers);
        PacketMetadata::ItemIterator metadataIterator = packet->BeginItem ();
        while (metadataIterator.HasNext ())
          {
            PacketMetadata::Item item = metadataIterator.Next ();
            std::set<TypeId>::iterator missingIter = missingHeaders.find (item.tid);
            if (missingIter != missingHeaders.end ())
              {
                missingHeaders.erase (missingIter);
              }
          }
        if (missingHeaders.size () == 0)
          {
            return true;
          }
        else
          {
            return false;
          }
      }

    default:
      NS_FATAL_ERROR ("should not be reached");
      return false;
    }
}

void
PyViz::TraceDevQueueDrop (std::string context, Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (context << packet->GetUid ());
  std::vector<std::string> splitPath = PathSplit (context);
  int nodeIndex = std::atoi (splitPath[1].c_str ());
  Ptr<Node> node = NodeList::GetNode (nodeIndex);

  if (m_nodesOfInterest.find (nodeIndex) == m_nodesOfInterest.end ())
    {
      // if the transmitting node is not "of interest", we still
      // record the transmission if it is a packet of interest.
      if (m_packetsOfInterest.find (packet->GetUid ()) == m_packetsOfInterest.end ())
        {
          NS_LOG_DEBUG ("Packet " << packet->GetUid () << " is not of interest");
          return;
        }
    }

  // ---- "last packets"
  const PacketCaptureOptions *captureOptions;
  if (GetPacketCaptureOptions (nodeIndex, &captureOptions) && FilterPacket (packet, *captureOptions))
    {
      LastPacketsSample &last = m_lastPackets[nodeIndex];
      PacketSample lastPacket;
      lastPacket.time = Simulator::Now ();
      lastPacket.packet = packet->Copy ();
      lastPacket.device = NULL;
      last.lastDroppedPackets.push_back (lastPacket);
      while (last.lastDroppedPackets.size () > captureOptions->numLastPackets)
        {
          last.lastDroppedPackets.erase (last.lastDroppedPackets.begin ());
        }
    }

  std::map<Ptr<Node>, uint32_t>::iterator iter = m_packetDrops.find (node);
  if (iter == m_packetDrops.end ())
    {
      m_packetDrops[node] = packet->GetSize ();
    }
  else
    {
      iter->second += packet->GetSize ();
    }
}

void
PyViz::TraceIpv4Drop (std::string context, ns3::Ipv4Header const &hdr, Ptr<const Packet> packet,
                      ns3::Ipv4L3Protocol::DropReason reason, Ptr<Ipv4> dummy_ipv4, uint32_t interface)
{
  Ptr<Packet> packetCopy = packet->Copy ();
  packetCopy->AddHeader (hdr);
  TraceDevQueueDrop (context, packetCopy);
}


// --------- TX device tracing -------------------

void
PyViz::TraceNetDevTxCommon (std::string const &context, Ptr<const Packet> packet,
                            Mac48Address const &destinationAddress)
{
  NS_LOG_FUNCTION (context << packet->GetUid () << *packet);

  std::vector<std::string> splitPath = PathSplit (context);
  int nodeIndex = std::atoi (splitPath[1].c_str ());
  int devIndex = std::atoi (splitPath[3].c_str ());
  Ptr<Node> node = NodeList::GetNode (nodeIndex);
  Ptr<NetDevice> device = node->GetDevice (devIndex);

  // ---- statistics
  NetDeviceStatistics &stats = FindNetDeviceStatistics (nodeIndex, devIndex);
  ++stats.transmittedPackets;
  stats.transmittedBytes += packet->GetSize ();

  // ---- "last packets"
  const PacketCaptureOptions *captureOptions;
  if (GetPacketCaptureOptions (nodeIndex, &captureOptions) && FilterPacket (packet, *captureOptions))
    {
      LastPacketsSample &last = m_lastPackets[nodeIndex];
      TxPacketSample lastPacket;
      lastPacket.time = Simulator::Now ();
      lastPacket.packet = packet->Copy ();
      lastPacket.device = device;
      lastPacket.to = destinationAddress;
      last.lastTransmittedPackets.push_back (lastPacket);
      while (last.lastTransmittedPackets.size () > captureOptions->numLastPackets)
        {
          last.lastTransmittedPackets.erase (last.lastTransmittedPackets.begin ());
        }
    }

  // ---- transmissions records

  if (m_nodesOfInterest.find (nodeIndex) == m_nodesOfInterest.end ())
    {
      // if the transmitting node is not "of interest", we still
      // record the transmission if it is a packet of interest.
      if (m_packetsOfInterest.find (packet->GetUid ()) == m_packetsOfInterest.end ())
        {
          NS_LOG_DEBUG ("Packet " << packet->GetUid () << " is not of interest");
          return;
        }
    }
  else
    {
      // We will follow this packet throughout the network.
      m_packetsOfInterest[packet->GetUid ()] = Simulator::Now ();
    }

  TxRecordValue record = { Simulator::Now (), node, false };
  if (destinationAddress == device->GetBroadcast ())
    {
      record.isBroadcast = true;
    }

  m_txRecords[TxRecordKey (device->GetChannel (), packet->GetUid ())] = record;

  PyVizPacketTag tag;
  //packet->RemovePacketTag (tag);
  tag.m_packetId = packet->GetUid ();
  packet->AddByteTag (tag);
}

void
PyViz::TraceNetDevTxWifi (std::string context, Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (context << packet->GetUid () << *packet);

  /*
   *  To DS    From DS   Address 1    Address 2    Address 3    Address 4 
   *----------------------------------------------------------------------
   *    0        0       Destination  Source        BSSID         N/A
   *    0        1       Destination  BSSID         Source        N/A
   *    1        0       BSSID        Source        Destination   N/A
   *    1        1       Receiver     Transmitter   Destination   Source
   */
  WifiMacHeader hdr;
  NS_ABORT_IF (packet->PeekHeader (hdr) == 0);
  Mac48Address destinationAddress;
  if (hdr.IsToDs () && !hdr.IsFromDs ())
    {
      destinationAddress = hdr.GetAddr3 ();
    }
  else if (!hdr.IsToDs () && hdr.IsFromDs ())
    {
      destinationAddress = hdr.GetAddr1 ();
    }
  else if (!hdr.IsToDs () && !hdr.IsFromDs ())
    {
      destinationAddress = hdr.GetAddr1 ();
    }
  else
    {
      destinationAddress = hdr.GetAddr3 ();
    }
  TraceNetDevTxCommon (context, packet, destinationAddress);
}


void
PyViz::TraceNetDevTxCsma (std::string context, Ptr<const Packet> packet)
{
  EthernetHeader ethernetHeader;
  NS_ABORT_IF (packet->PeekHeader (ethernetHeader) == 0);
  TraceNetDevTxCommon (context, packet, ethernetHeader.GetDestination ());
}

void
PyViz::TraceNetDevTxPointToPoint (std::string context, Ptr<const Packet> packet)
{
  TraceNetDevTxCommon (context, packet, Mac48Address ());
}




// --------- RX device tracing -------------------

void
PyViz::TraceNetDevRxCommon (std::string const &context, Ptr<const Packet> packet, Mac48Address const &from)
{
  uint32_t uid;
  PyVizPacketTag tag;
  if (packet->FindFirstMatchingByteTag (tag))
    {
      uid = tag.m_packetId;
    }
  else
    {
      //NS_ASSERT (0);
      NS_LOG_WARN ("Packet has no byte tag; wimax link?");
      uid = packet->GetUid ();
    }

  NS_LOG_FUNCTION (context << uid);
  std::vector<std::string> splitPath = PathSplit (context);
  int nodeIndex = std::atoi (splitPath[1].c_str ());
  int devIndex = std::atoi (splitPath[3].c_str ());

  // ---- statistics
  NetDeviceStatistics &stats = FindNetDeviceStatistics (nodeIndex, devIndex);
  ++stats.receivedPackets;
  stats.receivedBytes += packet->GetSize ();

  Ptr<Node> node = NodeList::GetNode (nodeIndex);
  Ptr<NetDevice> device = node->GetDevice (devIndex);

  // ---- "last packets"
  const PacketCaptureOptions *captureOptions;
  if (GetPacketCaptureOptions (nodeIndex, &captureOptions) && FilterPacket (packet, *captureOptions))
    {
      LastPacketsSample &last = m_lastPackets[nodeIndex];
      RxPacketSample lastPacket;
      lastPacket.time = Simulator::Now ();
      lastPacket.packet = packet->Copy ();
      lastPacket.device = device;
      lastPacket.from = from;
      last.lastReceivedPackets.push_back (lastPacket);
      while (last.lastReceivedPackets.size () > captureOptions->numLastPackets)
        {
          last.lastReceivedPackets.erase (last.lastReceivedPackets.begin ());
        }
    }

  // ---- transmissions
  if (m_packetsOfInterest.find (uid) == m_packetsOfInterest.end ())
    {
      NS_LOG_DEBUG ("RX Packet " << uid << " is not of interest");
      return;
    }

  Ptr<Channel> channel = device->GetChannel ();

  std::map<TxRecordKey, TxRecordValue>::iterator recordIter = 
    m_txRecords.find (TxRecordKey (channel, uid));

  if (recordIter == m_txRecords.end ())
    {
      NS_LOG_DEBUG ("RX Packet " << uid << " was not transmitted?!");
      return;
    }

  TxRecordValue &record = recordIter->second;

  if (record.srcNode == node)
    {
      NS_LOG_WARN ("Node " << node->GetId () << " receiving back the same packet (UID=" << uid
                           << ") it had previously transmitted, on the same channel!");
      return;
    }

  TransmissionSampleKey key = { record.srcNode, node, channel };

#ifdef  NS3_LOG_ENABLE
  NS_LOG_DEBUG ("m_transmissionSamples begin:");
  if (g_log.IsEnabled (ns3::LOG_DEBUG))
    {
      for (std::map<TransmissionSampleKey,TransmissionSampleValue>::const_iterator iter
             = m_transmissionSamples.begin (); iter != m_transmissionSamples.end (); iter++)
        {
          NS_LOG_DEBUG (iter->first.transmitter<<"/"<<iter->first.transmitter->GetId () << ", "
                                               << iter->first.receiver<<"/"<<iter->first.receiver->GetId ()
                                               << ", " << iter->first.channel << " => " << iter->second.bytes << " (@ " << &iter->second << ")");
        }
    }
  NS_LOG_DEBUG ("m_transmissionSamples end.");
#endif

  std::map<TransmissionSampleKey,TransmissionSampleValue>::iterator
    iter = m_transmissionSamples.find (key);

  if (iter == m_transmissionSamples.end ())
    {
      TransmissionSampleValue sample = { packet->GetSize () };
      NS_LOG_DEBUG ("RX: from " << key.transmitter<<"/"<<key.transmitter->GetId () << " to "
                                << key.receiver<<"/"<<key.receiver->GetId ()
                                << " channel " << channel << ": " << packet->GetSize ()
                                << " bytes more. => new sample with " << packet->GetSize () << " bytes.");
      m_transmissionSamples[key] = sample;
    }
  else
    {
      TransmissionSampleValue &sample = iter->second;
      NS_LOG_DEBUG ("RX: from " << key.transmitter<<"/"<<key.transmitter->GetId () << " to "
                                << key.receiver<<"/"<<key.receiver->GetId ()
                                << " channel " << channel << ": " << packet->GetSize ()
                                << " bytes more. => sample " << &sample << " with bytes " << sample.bytes);

      sample.bytes += packet->GetSize ();
    }
}

void
PyViz::TraceNetDevRxWifi (std::string context, Ptr<const Packet> packet)
{
  NS_LOG_FUNCTION (context << packet->GetUid ());


  /*
   *  To DS    From DS   Address 1    Address 2    Address 3    Address 4 
   *----------------------------------------------------------------------
   *    0        0       Destination  Source        BSSID         N/A
   *    0        1       Destination  BSSID         Source        N/A
   *    1        0       BSSID        Source        Destination   N/A
   *    1        1       Receiver     Transmitter   Destination   Source
   */
  WifiMacHeader hdr;
  NS_ABORT_IF (packet->PeekHeader (hdr) == 0);
  Mac48Address sourceAddress;
  if (hdr.IsToDs () && !hdr.IsFromDs ())
    {
      sourceAddress = hdr.GetAddr2 ();
    }
  else if (!hdr.IsToDs () && hdr.IsFromDs ())
    {
      sourceAddress = hdr.GetAddr3 ();
    }
  else if (!hdr.IsToDs () && !hdr.IsFromDs ())
    {
      sourceAddress = hdr.GetAddr2 ();
    }
  else
    {
      sourceAddress = hdr.GetAddr4 ();
    }

  TraceNetDevRxCommon (context, packet, sourceAddress);
}



void
PyViz::TraceNetDevRxCsma (std::string context, Ptr<const Packet> packet)
{
  EthernetHeader ethernetHeader;
  NS_ABORT_IF (packet->PeekHeader (ethernetHeader) == 0);
  TraceNetDevRxCommon (context, packet, ethernetHeader.GetSource ());
}

void
PyViz::TraceNetDevRxPointToPoint (std::string context, Ptr<const Packet> packet)
{
  TraceNetDevRxCommon (context, packet, Mac48Address ());
}

void
PyViz::TraceNetDevPromiscRxCsma (std::string context, Ptr<const Packet> packet)
{
  EthernetHeader ethernetHeader;
  NS_ABORT_IF (packet->PeekHeader (ethernetHeader) == 0);

  NetDevice::PacketType packetType = NetDevice::PACKET_OTHERHOST; // FIXME

  // Other packet types are already being received by
  // TraceNetDevRxCsma; we don't want to receive them twice.
  if (packetType == NetDevice::PACKET_OTHERHOST)
    {
      TraceNetDevRxCommon (context, packet, ethernetHeader.GetDestination ());
    }
}

void
PyViz::TraceNetDevTxWimax (std::string context, Ptr<const Packet> packet, Mac48Address const &destination)
{
  NS_LOG_FUNCTION (context);
  TraceNetDevTxCommon (context, packet, destination);
}

void
PyViz::TraceNetDevRxWimax (std::string context, Ptr<const Packet> packet, Mac48Address const &source)
{
  NS_LOG_FUNCTION (context);
  TraceNetDevRxCommon (context, packet, source);
}

void
PyViz::TraceNetDevTxLte (std::string context, Ptr<const Packet> packet, Mac48Address const &destination)
{
  NS_LOG_FUNCTION (context);
  TraceNetDevTxCommon (context, packet, destination);
}

void
PyViz::TraceNetDevRxLte (std::string context, Ptr<const Packet> packet, Mac48Address const &source)
{
  NS_LOG_FUNCTION (context);
  TraceNetDevRxCommon (context, packet, source);
}

// ---------------------

PyViz::TransmissionSampleList
PyViz::GetTransmissionSamples () const
{
  NS_LOG_DEBUG ("GetTransmissionSamples BEGIN");
  TransmissionSampleList list;
  for (std::map<TransmissionSampleKey, TransmissionSampleValue>::const_iterator
       iter = m_transmissionSamples.begin ();
       iter !=  m_transmissionSamples.end ();
       iter++)
    {
      TransmissionSample sample;
      sample.transmitter = iter->first.transmitter;
      sample.receiver = iter->first.receiver;
      sample.channel = iter->first.channel;
      sample.bytes = iter->second.bytes;
      NS_LOG_DEBUG ("from " << sample.transmitter->GetId () << " to " << sample.receiver->GetId ()
                            << ": " << sample.bytes << " bytes.");
      list.push_back (sample);
    }
  NS_LOG_DEBUG ("GetTransmissionSamples END");
  return list;
}

PyViz::PacketDropSampleList
PyViz::GetPacketDropSamples () const
{
  NS_LOG_DEBUG ("GetPacketDropSamples BEGIN");
  PacketDropSampleList list;
  for (std::map<Ptr<Node>, uint32_t>::const_iterator
       iter = m_packetDrops.begin ();
       iter !=  m_packetDrops.end ();
       iter++)
    {
      PacketDropSample sample;
      sample.transmitter = iter->first;
      sample.bytes = iter->second;
      NS_LOG_DEBUG ("in " << sample.transmitter->GetId ()
                          << ": " << sample.bytes << " bytes dropped.");
      list.push_back (sample);
    }
  NS_LOG_DEBUG ("GetPacketDropSamples END");
  return list;
}

void
PyViz::SetNodesOfInterest (std::set<uint32_t> nodes)
{
  m_nodesOfInterest = nodes;
}

std::vector<PyViz::NodeStatistics>
PyViz::GetNodesStatistics () const
{
  std::vector<PyViz::NodeStatistics> retval;
  for (std::map<uint32_t, std::vector<NetDeviceStatistics> >::const_iterator iter = m_nodesStatistics.begin ();
       iter != m_nodesStatistics.end (); iter++)
    {
      NodeStatistics stats = { iter->first, iter->second };
      retval.push_back (stats);
    }
  return retval;
}


PyViz::LastPacketsSample
PyViz::GetLastPackets (uint32_t nodeId) const
{
  NS_LOG_DEBUG ("GetLastPackets: " << nodeId);

  std::map<uint32_t, LastPacketsSample>::const_iterator
    iter = m_lastPackets.find (nodeId);
  if (iter != m_lastPackets.end ())
    {
      return iter->second;
    }
  else
    {
      return LastPacketsSample ();
    }
}






namespace
{
/// Adapted from http://en.wikipedia.org/w/index.php?title=Line_clipping&oldid=248609574
class FastClipping
{
public:
  /// Vector2 structure
  struct Vector2
  {
    double x; ///< X
    double y; ///< Y
  };

  Vector2 m_clipMin; ///< clip minimum 
  Vector2 m_clipMax; ///< clip maximum

  /// Line structure
  struct Line
  {
    Vector2 start; ///< start
    Vector2 end; ///<  end
    double dx; ///< dX 
    double dy; ///< dY
  };

private:

  /**
   * Clip start top function
   * \param line the clip line
   */
  void ClipStartTop (Line &line)
  {
    line.start.x += line.dx * (m_clipMin.y - line.start.y) / line.dy;
    line.start.y = m_clipMin.y;
  }
 
  /**
   * Clip start bottom function
   * \param line the clip line
   */
  void ClipStartBottom (Line &line)
  {
    line.start.x += line.dx * (m_clipMax.y - line.start.y) / line.dy;
    line.start.y = m_clipMax.y;
  }
 
  /**
   * Clip start right function
   * \param line the clip line
   */
  void ClipStartRight (Line &line)
  {
    line.start.y += line.dy * (m_clipMax.x - line.start.x) / line.dx;
    line.start.x = m_clipMax.x;
  }
 
  /**
   * Clip start left function
   * \param line the clip line
   */
  void ClipStartLeft (Line &line)
  {
    line.start.y += line.dy * (m_clipMin.x - line.start.x) / line.dx;
    line.start.x = m_clipMin.x;
  }
 
  /**
   * Clip end top function
   * \param line the clip line
   */
  void ClipEndTop (Line &line)
  {
    line.end.x += line.dx * (m_clipMin.y - line.end.y) / line.dy;
    line.end.y = m_clipMin.y;
  }

  /**
   * Clip end bottom function
   * \param line the clip line
   */
  void ClipEndBottom (Line &line) {
    line.end.x += line.dx * (m_clipMax.y - line.end.y) / line.dy;
    line.end.y = m_clipMax.y;
  }
 
  /**
   * Clip end right function
   * \param line the clip line
   */
  void ClipEndRight (Line &line)
  {
    line.end.y += line.dy * (m_clipMax.x - line.end.x) / line.dx;
    line.end.x = m_clipMax.x;
  }
 
  /**
   * Clip end left function
   * \param line the clip line
   */
  void ClipEndLeft (Line &line)
  {
    line.end.y += line.dy * (m_clipMin.x - line.end.x) / line.dx;
    line.end.x = m_clipMin.x;
  }

public:
  /**
   * Constructor
   *
   * \param clipMin minimum clipping vector
   * \param clipMax maximum clipping vector
   */
  FastClipping (Vector2 clipMin, Vector2 clipMax)
    : m_clipMin (clipMin), m_clipMax (clipMax)
  {
  }

 
  /**
   * Clip line function
   * \param line the clip line
   * \returns true if clipped
   */
  bool ClipLine (Line &line)
  {
    uint8_t lineCode = 0;
 
    if (line.end.y < m_clipMin.y)
      lineCode |= 8;
    else if (line.end.y > m_clipMax.y)
      lineCode |= 4;
 
    if (line.end.x > m_clipMax.x)
      lineCode |= 2;
    else if (line.end.x < m_clipMin.x)
      lineCode |= 1;
 
    if (line.start.y < m_clipMin.y)
      lineCode |= 128;
    else if (line.start.y > m_clipMax.y)
      lineCode |= 64;
 
    if (line.start.x > m_clipMax.x)
      lineCode |= 32;
    else if (line.start.x < m_clipMin.x)
      lineCode |= 16;
 
    // 9 - 8 - A
    // |   |   |
    // 1 - 0 - 2
    // |   |   |
    // 5 - 4 - 6
    switch (lineCode)
      {
      // center
      case 0x00:
        return true;
 
      case 0x01:
        ClipEndLeft (line);
        return true;
 
      case 0x02:
        ClipEndRight (line);
        return true;
 
      case 0x04:
        ClipEndBottom (line);
        return true;
 
      case 0x05:
        ClipEndLeft (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      case 0x06:
        ClipEndRight (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      case 0x08:
        ClipEndTop (line);
        return true;
 
      case 0x09:
        ClipEndLeft (line);
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        return true;
 
      case 0x0A:
        ClipEndRight (line);
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        return true;
 
      // left
      case 0x10:
        ClipStartLeft (line);
        return true;
 
      case 0x12:
        ClipStartLeft (line);
        ClipEndRight (line);
        return true;
 
      case 0x14:
        ClipStartLeft (line);
        if (line.start.y > m_clipMax.y)
          return false;
        ClipEndBottom (line);
        return true;
 
      case 0x16:
        ClipStartLeft (line);
        if (line.start.y > m_clipMax.y)
          return false;
        ClipEndBottom (line);
        if (line.end.x > m_clipMax.x)
          ClipEndRight (line);
        return true;
 
      case 0x18:
        ClipStartLeft (line);
        if (line.start.y < m_clipMin.y)
          return false;
        ClipEndTop (line);
        return true;
 
      case 0x1A:
        ClipStartLeft (line);
        if (line.start.y < m_clipMin.y)
          return false;
        ClipEndTop (line);
        if (line.end.x > m_clipMax.x)
          ClipEndRight (line);
        return true;
 
      // right
      case 0x20:
        ClipStartRight (line);
        return true;
 
      case 0x21:
        ClipStartRight (line);
        ClipEndLeft (line);
        return true;
 
      case 0x24:
        ClipStartRight (line);
        if (line.start.y > m_clipMax.y)
          return false;
        ClipEndBottom (line);
        return true;
 
      case 0x25:
        ClipStartRight (line);
        if (line.start.y > m_clipMax.y)
          return false;
        ClipEndBottom (line);
        if (line.end.x < m_clipMin.x)
          ClipEndLeft (line);
        return true;
 
      case 0x28:
        ClipStartRight (line);
        if (line.start.y < m_clipMin.y)
          return false;
        ClipEndTop (line);
        return true;
 
      case 0x29:
        ClipStartRight (line);
        if (line.start.y < m_clipMin.y)
          return false;
        ClipEndTop (line);
        if (line.end.x < m_clipMin.x)
          ClipEndLeft (line);
        return true;
 
      // bottom
      case 0x40:
        ClipStartBottom (line);
        return true;
 
      case 0x41:
        ClipStartBottom (line);
        if (line.start.x < m_clipMin.x)
          return false;
        ClipEndLeft (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      case 0x42:
        ClipStartBottom (line);
        if (line.start.x > m_clipMax.x)
          return false;
        ClipEndRight (line);
        return true;
 
      case 0x48:
        ClipStartBottom (line);
        ClipEndTop (line);
        return true;
 
      case 0x49:
        ClipStartBottom (line);
        if (line.start.x < m_clipMin.x)
          return false;
        ClipEndLeft (line);
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        return true;
 
      case 0x4A:
        ClipStartBottom (line);
        if (line.start.x > m_clipMax.x)
          return false;
        ClipEndRight (line);
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        return true;
 
      // bottom-left
      case 0x50:
        ClipStartLeft (line);
        if (line.start.y > m_clipMax.y)
          ClipStartBottom (line);
        return true;
 
      case 0x52:
        ClipEndRight (line);
        if (line.end.y > m_clipMax.y)
          return false;
        ClipStartBottom (line);
        if (line.start.x < m_clipMin.x)
          ClipStartLeft (line);
        return true;
 
      case 0x58:
        ClipEndTop (line);
        if (line.end.x < m_clipMin.x)
          return false;
        ClipStartBottom (line);
        if (line.start.x < m_clipMin.x)
          ClipStartLeft (line);
        return true;
 
      case 0x5A:
        ClipStartLeft (line);
        if (line.start.y < m_clipMin.y)
          return false;
        ClipEndRight (line);
        if (line.end.y > m_clipMax.y)
          return false;
        if (line.start.y > m_clipMax.y)
          ClipStartBottom (line);
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        return true;
 
      // bottom-right
      case 0x60:
        ClipStartRight (line);
        if (line.start.y > m_clipMax.y)
          ClipStartBottom (line);
        return true;
 
      case 0x61:
        ClipEndLeft (line);
        if (line.end.y > m_clipMax.y)
          return false;
        ClipStartBottom (line);
        if (line.start.x > m_clipMax.x)
          ClipStartRight (line);
        return true;
 
      case 0x68:
        ClipEndTop (line);
        if (line.end.x > m_clipMax.x)
          return false;
        ClipStartRight (line);
        if (line.start.y > m_clipMax.y)
          ClipStartBottom (line);
        return true;
 
      case 0x69:
        ClipEndLeft (line);
        if (line.end.y > m_clipMax.y)
          return false;
        ClipStartRight (line);
        if (line.start.y < m_clipMin.y)
          return false;
        if (line.end.y < m_clipMin.y)
          ClipEndTop (line);
        if (line.start.y > m_clipMax.y)
          ClipStartBottom (line);
        return true;
 
      // top
      case 0x80:
        ClipStartTop (line);
        return true;
 
      case 0x81:
        ClipStartTop (line);
        if (line.start.x < m_clipMin.x)
          return false;
        ClipEndLeft (line);
        return true;
 
      case 0x82:
        ClipStartTop (line);
        if (line.start.x > m_clipMax.x)
          return false;
        ClipEndRight (line);
        return true;
 
      case 0x84:
        ClipStartTop (line);
        ClipEndBottom (line);
        return true;
 
      case 0x85:
        ClipStartTop (line);
        if (line.start.x < m_clipMin.x)
          return false;
        ClipEndLeft (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      case 0x86:
        ClipStartTop (line);
        if (line.start.x > m_clipMax.x)
          return false;
        ClipEndRight (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      // top-left
      case 0x90:
        ClipStartLeft (line);
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        return true;
 
      case 0x92:
        ClipEndRight (line);
        if (line.end.y < m_clipMin.y)
          return false;
        ClipStartTop (line);
        if (line.start.x < m_clipMin.x)
          ClipStartLeft (line);
        return true;
 
      case 0x94:
        ClipEndBottom (line);
        if (line.end.x < m_clipMin.x)
          return false;
        ClipStartLeft (line);
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        return true;
 
      case 0x96:
        ClipStartLeft (line);
        if (line.start.y > m_clipMax.y)
          return false;
        ClipEndRight (line);
        if (line.end.y < m_clipMin.y)
          return false;
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        return true;
 
      // top-right
      case 0xA0:
        ClipStartRight (line);
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        return true;
 
      case 0xA1:
        ClipEndLeft (line);
        if (line.end.y < m_clipMin.y)
          return false;
        ClipStartTop (line);
        if (line.start.x > m_clipMax.x)
          ClipStartRight (line);
        return true;
 
      case 0xA4:
        ClipEndBottom (line);
        if (line.end.x > m_clipMax.x)
          return false;
        ClipStartRight (line);
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        return true;
 
      case 0xA5:
        ClipEndLeft (line);
        if (line.end.y < m_clipMin.y)
          return false;
        ClipStartRight (line);
        if (line.start.y > m_clipMax.y)
          return false;
        if (line.end.y > m_clipMax.y)
          ClipEndBottom (line);
        if (line.start.y < m_clipMin.y)
          ClipStartTop (line);
        return true;
      }
 
    return false;
  }
};
}

void
PyViz::LineClipping (double boundsX1, double boundsY1, double boundsX2, double boundsY2,
                     double &lineX1, double &lineY1, double &lineX2, double &lineY2)
{
  FastClipping::Vector2 clipMin = { boundsX1, boundsY1}, clipMax = { boundsX2, boundsY2};
  FastClipping::Line line = { { lineX1, lineY1 }, { lineX2, lineY2 }, (lineX2-lineX1), (lineY2-lineY1) };

  FastClipping clipper (clipMin, clipMax);
  clipper.ClipLine (line);
  lineX1 = line.start.x;
  lineX2 = line.end.x;
  lineY1 = line.start.y;
  lineY2 = line.end.y;
}


}

