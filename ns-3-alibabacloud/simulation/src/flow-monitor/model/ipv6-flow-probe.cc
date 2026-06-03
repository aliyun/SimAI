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
// Modifications: Tommaso Pecorella <tommaso.pecorella@unifi.it>
//

#include "ns3/ipv6-flow-probe.h"
#include "ns3/ipv6-flow-classifier.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/flow-monitor.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/config.h"
#include "ns3/flow-id-tag.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6FlowProbe");

//////////////////////////////////////
// Ipv6FlowProbeTag class implementation //
//////////////////////////////////////

/**
 * \ingroup flow-monitor
 *
 * \brief Tag used to allow a fast identification of the packet
 *
 * This tag is added by FlowMonitor when a packet is seen for
 * the first time, and it is then used to classify the packet in
 * the following hops.
 */
class Ipv6FlowProbeTag : public Tag
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer buf) const;
  virtual void Deserialize (TagBuffer buf);
  virtual void Print (std::ostream &os) const;
  Ipv6FlowProbeTag ();
  /**
   * \brief Consructor
   * \param flowId the flow identifier
   * \param packetId the packet identifier
   * \param packetSize the packet size
   */
  Ipv6FlowProbeTag (uint32_t flowId, uint32_t packetId, uint32_t packetSize);
  /**
   * \brief Set the flow identifier
   * \param flowId the flow identifier
   */
  void SetFlowId (uint32_t flowId);
  /**
   * \brief Set the packet identifier
   * \param packetId the packet identifier
   */
  void SetPacketId (uint32_t packetId);
  /**
   * \brief Set the packet size
   * \param packetSize the packet size
   */
  void SetPacketSize (uint32_t packetSize);
  /**
   * \brief Set the flow identifier
   * \returns the flow identifier
   */
  uint32_t GetFlowId (void) const;
  /**
   * \brief Set the packet identifier
   * \returns the packet identifier
   */
  uint32_t GetPacketId (void) const;
  /**
   * \brief Get the packet size
   * \returns the packet size
   */
  uint32_t GetPacketSize (void) const;
private:
  uint32_t m_flowId;      //!< flow identifier
  uint32_t m_packetId;    //!< packet identifier
  uint32_t m_packetSize;  //!< packet size

};

TypeId 
Ipv6FlowProbeTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv6FlowProbeTag")
    .SetParent<Tag> ()
    .SetGroupName ("FlowMonitor")
    .AddConstructor<Ipv6FlowProbeTag> ()
  ;
  return tid;
}
TypeId 
Ipv6FlowProbeTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}
uint32_t 
Ipv6FlowProbeTag::GetSerializedSize (void) const
{
  return 4 + 4 + 4;
}
void 
Ipv6FlowProbeTag::Serialize (TagBuffer buf) const
{
  buf.WriteU32 (m_flowId);
  buf.WriteU32 (m_packetId);
  buf.WriteU32 (m_packetSize);
}
void 
Ipv6FlowProbeTag::Deserialize (TagBuffer buf)
{
  m_flowId = buf.ReadU32 ();
  m_packetId = buf.ReadU32 ();
  m_packetSize = buf.ReadU32 ();
}
void 
Ipv6FlowProbeTag::Print (std::ostream &os) const
{
  os << "FlowId=" << m_flowId;
  os << "PacketId=" << m_packetId;
  os << "PacketSize=" << m_packetSize;
}
Ipv6FlowProbeTag::Ipv6FlowProbeTag ()
  : Tag () 
{
}

Ipv6FlowProbeTag::Ipv6FlowProbeTag (uint32_t flowId, uint32_t packetId, uint32_t packetSize)
  : Tag (), m_flowId (flowId), m_packetId (packetId), m_packetSize (packetSize)
{
}

void
Ipv6FlowProbeTag::SetFlowId (uint32_t id)
{
  m_flowId = id;
}
void
Ipv6FlowProbeTag::SetPacketId (uint32_t id)
{
  m_packetId = id;
}
void
Ipv6FlowProbeTag::SetPacketSize (uint32_t size)
{
  m_packetSize = size;
}
uint32_t
Ipv6FlowProbeTag::GetFlowId (void) const
{
  return m_flowId;
}
uint32_t
Ipv6FlowProbeTag::GetPacketId (void) const
{
  return m_packetId;
} 
uint32_t
Ipv6FlowProbeTag::GetPacketSize (void) const
{
  return m_packetSize;
} 

////////////////////////////////////////
// Ipv6FlowProbe class implementation //
////////////////////////////////////////

Ipv6FlowProbe::Ipv6FlowProbe (Ptr<FlowMonitor> monitor,
                              Ptr<Ipv6FlowClassifier> classifier,
                              Ptr<Node> node)
  : FlowProbe (monitor),
    m_classifier (classifier)
{
  NS_LOG_FUNCTION (this << node->GetId ());

  Ptr<Ipv6L3Protocol> ipv6 = node->GetObject<Ipv6L3Protocol> ();

  if (!ipv6->TraceConnectWithoutContext ("SendOutgoing",
                                         MakeCallback (&Ipv6FlowProbe::SendOutgoingLogger, Ptr<Ipv6FlowProbe> (this))))
    {
      NS_FATAL_ERROR ("trace fail");
    }
  if (!ipv6->TraceConnectWithoutContext ("UnicastForward",
                                         MakeCallback (&Ipv6FlowProbe::ForwardLogger, Ptr<Ipv6FlowProbe> (this))))
    {
      NS_FATAL_ERROR ("trace fail");
    }
  if (!ipv6->TraceConnectWithoutContext ("LocalDeliver",
                                         MakeCallback (&Ipv6FlowProbe::ForwardUpLogger, Ptr<Ipv6FlowProbe> (this))))
    {
      NS_FATAL_ERROR ("trace fail");
    }

  if (!ipv6->TraceConnectWithoutContext ("Drop",
                                         MakeCallback (&Ipv6FlowProbe::DropLogger, Ptr<Ipv6FlowProbe> (this))))
    {
      NS_FATAL_ERROR ("trace fail");
    }

  std::ostringstream qd;
  qd << "/NodeList/" << node->GetId () << "/$ns3::TrafficControlLayer/RootQueueDiscList/*/Drop";
  Config::ConnectWithoutContextFailSafe (qd.str (), MakeCallback (&Ipv6FlowProbe::QueueDiscDropLogger, Ptr<Ipv6FlowProbe> (this)));

  // code copied from point-to-point-helper.cc
  std::ostringstream oss;
  oss << "/NodeList/" << node->GetId () << "/DeviceList/*/TxQueue/Drop";
  Config::ConnectWithoutContextFailSafe (oss.str (), MakeCallback (&Ipv6FlowProbe::QueueDropLogger, Ptr<Ipv6FlowProbe> (this)));
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6FlowProbeTag);

/* static */
TypeId
Ipv6FlowProbe::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv6FlowProbe")
    .SetParent<FlowProbe> ()
    .SetGroupName ("FlowMonitor")
    // No AddConstructor because this class has no default constructor.
    ;

  return tid;
}

Ipv6FlowProbe::~Ipv6FlowProbe ()
{
}

void
Ipv6FlowProbe::DoDispose ()
{
  FlowProbe::DoDispose ();
}

void
Ipv6FlowProbe::SendOutgoingLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface)
{
  FlowId flowId;
  FlowPacketId packetId;

  if (m_classifier->Classify (ipHeader, ipPayload, &flowId, &packetId))
    {
      uint32_t size = (ipPayload->GetSize () + ipHeader.GetSerializedSize ());
      NS_LOG_DEBUG ("ReportFirstTx ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<"); "
                                     << ipHeader << *ipPayload);
      m_flowMonitor->ReportFirstTx (this, flowId, packetId, size);

      // tag the packet with the flow id and packet id, so that the packet can be identified even
      // when Ipv6Header is not accessible at some non-IPv6 protocol layer
      Ipv6FlowProbeTag fTag (flowId, packetId, size);
      ipPayload->AddByteTag (fTag);
    }
}

void
Ipv6FlowProbe::ForwardLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface)
{
  Ipv6FlowProbeTag fTag;
  bool found = ipPayload->FindFirstMatchingByteTag (fTag);

  if (found)
    {
      FlowId flowId = fTag.GetFlowId ();
      FlowPacketId packetId = fTag.GetPacketId ();

      uint32_t size = (ipPayload->GetSize () + ipHeader.GetSerializedSize ());
      NS_LOG_DEBUG ("ReportForwarding ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<");");
      m_flowMonitor->ReportForwarding (this, flowId, packetId, size);
    }
}

void
Ipv6FlowProbe::ForwardUpLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload, uint32_t interface)
{
  Ipv6FlowProbeTag fTag;
  bool found = ipPayload->FindFirstMatchingByteTag (fTag);

  if (found)
    {
      FlowId flowId = fTag.GetFlowId ();
      FlowPacketId packetId = fTag.GetPacketId ();

      uint32_t size = (ipPayload->GetSize () + ipHeader.GetSerializedSize ());
      NS_LOG_DEBUG ("ReportLastRx ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<");");
      m_flowMonitor->ReportLastRx (this, flowId, packetId, size);
    }
}

void
Ipv6FlowProbe::DropLogger (const Ipv6Header &ipHeader, Ptr<const Packet> ipPayload,
                           Ipv6L3Protocol::DropReason reason, Ptr<Ipv6> ipv6, uint32_t ifIndex)
{
#if 0
  switch (reason)
    {
    case Ipv6L3Protocol::DROP_NO_ROUTE:
      break;

    case Ipv6L3Protocol::DROP_TTL_EXPIRED:
    case Ipv6L3Protocol::DROP_BAD_CHECKSUM:
      Ipv6Address addri = m_ipv6->GetAddress (ifIndex);
      Ipv6Mask maski = m_ipv6->GetNetworkMask (ifIndex);
      Ipv6Address bcast = addri.GetSubnetDirectedBroadcast (maski);
      if (ipHeader.GetDestination () == bcast) // we don't want broadcast packets
        {
          return;
        }
    }
#endif

  Ipv6FlowProbeTag fTag;
  bool found = ipPayload->FindFirstMatchingByteTag (fTag);

  if (found)
    {
      FlowId flowId = fTag.GetFlowId ();
      FlowPacketId packetId = fTag.GetPacketId ();

      uint32_t size = (ipPayload->GetSize () + ipHeader.GetSerializedSize ());
      NS_LOG_DEBUG ("Drop ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<", " << reason 
                            << ", destIp=" << ipHeader.GetDestination () << "); "
                            << "HDR: " << ipHeader << " PKT: " << *ipPayload);

      DropReason myReason;


      switch (reason)
        {
        case Ipv6L3Protocol::DROP_TTL_EXPIRED:
          myReason = DROP_TTL_EXPIRE;
          NS_LOG_DEBUG ("DROP_TTL_EXPIRE");
          break;
        case Ipv6L3Protocol::DROP_NO_ROUTE:
          myReason = DROP_NO_ROUTE;
          NS_LOG_DEBUG ("DROP_NO_ROUTE");
          break;
        case Ipv6L3Protocol::DROP_INTERFACE_DOWN:
          myReason = DROP_INTERFACE_DOWN;
          NS_LOG_DEBUG ("DROP_INTERFACE_DOWN");
          break;
        case Ipv6L3Protocol::DROP_ROUTE_ERROR:
           myReason = DROP_ROUTE_ERROR;
           NS_LOG_DEBUG ("DROP_ROUTE_ERROR");
           break;
        case Ipv6L3Protocol::DROP_UNKNOWN_PROTOCOL:
           myReason = DROP_UNKNOWN_PROTOCOL;
           NS_LOG_DEBUG ("DROP_UNKNOWN_PROTOCOL");
           break;
        case Ipv6L3Protocol::DROP_UNKNOWN_OPTION:
           myReason = DROP_UNKNOWN_OPTION;
           NS_LOG_DEBUG ("DROP_UNKNOWN_OPTION");
           break;
        case Ipv6L3Protocol::DROP_MALFORMED_HEADER:
           myReason = DROP_MALFORMED_HEADER;
           NS_LOG_DEBUG ("DROP_MALFORMED_HEADER");
           break;
        case Ipv6L3Protocol::DROP_FRAGMENT_TIMEOUT:
          myReason = DROP_FRAGMENT_TIMEOUT;
          NS_LOG_DEBUG ("DROP_FRAGMENT_TIMEOUT");
          break;
        default:
          myReason = DROP_INVALID_REASON;
          NS_FATAL_ERROR ("Unexpected drop reason code " << reason);
        }

      m_flowMonitor->ReportDrop (this, flowId, packetId, size, myReason);
    }
}

void 
Ipv6FlowProbe::QueueDropLogger (Ptr<const Packet> ipPayload)
{
  Ipv6FlowProbeTag fTag;
  bool tagFound = ipPayload->FindFirstMatchingByteTag (fTag);

  if (!tagFound)
    {
      return;
    }

  FlowId flowId = fTag.GetFlowId ();
  FlowPacketId packetId = fTag.GetPacketId ();
  uint32_t size = fTag.GetPacketSize ();

  NS_LOG_DEBUG ("Drop ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<", " << DROP_QUEUE 
                        << "); ");

  m_flowMonitor->ReportDrop (this, flowId, packetId, size, DROP_QUEUE);
}

void
Ipv6FlowProbe::QueueDiscDropLogger (Ptr<const QueueDiscItem> item)
{
  Ipv6FlowProbeTag fTag;
  bool tagFound = item->GetPacket ()->FindFirstMatchingByteTag (fTag);

  if (!tagFound)
    {
      return;
    }

  FlowId flowId = fTag.GetFlowId ();
  FlowPacketId packetId = fTag.GetPacketId ();
  uint32_t size = fTag.GetPacketSize ();

  NS_LOG_DEBUG ("Drop ("<<this<<", "<<flowId<<", "<<packetId<<", "<<size<<", " << DROP_QUEUE_DISC
                        << "); ");

  m_flowMonitor->ReportDrop (this, flowId, packetId, size, DROP_QUEUE_DISC);
}

} // namespace ns3


