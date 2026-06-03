/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#include "ns3/simulator.h"
#include "ns3/node.h"
#include "ns3/config.h"
#include "ss-net-device.h"
#include "wimax-phy.h"
#include "ns3/packet-burst.h"
#include <algorithm>
#include "dl-mac-messages.h"
#include "ul-mac-messages.h"
#include "ss-scheduler.h"
#include "wimax-mac-queue.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "service-flow.h"
#include "service-flow-record.h"
#include "service-flow-manager.h"
#include "connection-manager.h"
#include "burst-profile-manager.h"
#include "ss-link-manager.h"
#include "bandwidth-manager.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SubscriberStationNetDevice");

NS_OBJECT_ENSURE_REGISTERED (SubscriberStationNetDevice);

Time SubscriberStationNetDevice::GetDefaultLostDlMapInterval ()
{
  return (MicroSeconds (500000));
}

TypeId
SubscriberStationNetDevice::GetTypeId (void)
{
  static TypeId
    tid =
    TypeId ("ns3::SubscriberStationNetDevice")

    .SetParent<WimaxNetDevice> ()
    .SetGroupName ("Wimax")

    .AddConstructor<SubscriberStationNetDevice> ()

    .AddAttribute ("BasicConnection",
                   "Basic connection",
                   PointerValue (),
                   MakePointerAccessor (&SubscriberStationNetDevice::m_basicConnection),
                   MakePointerChecker<WimaxConnection> ())

    .AddAttribute ("PrimaryConnection",
                   "Primary connection",
                   PointerValue (),
                   MakePointerAccessor (&SubscriberStationNetDevice::m_primaryConnection),
                   MakePointerChecker<WimaxConnection> ())

    .AddAttribute ("LostDlMapInterval",
                   "Time since last received DL-MAP message before downlink synchronization is considered lost. Maximum is 600ms",
                   TimeValue (Seconds (0.5)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetLostDlMapInterval,
                                     &SubscriberStationNetDevice::SetLostDlMapInterval),
                   MakeTimeChecker ())

    .AddAttribute ("LostUlMapInterval",
                   "Time since last received UL-MAP before uplink synchronization is considered lost, maximum is 600.",
                   TimeValue (MilliSeconds (500)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetLostUlMapInterval,
                                     &SubscriberStationNetDevice::SetLostUlMapInterval),
                   MakeTimeChecker ())

    .AddAttribute ("MaxDcdInterval",
                   "Maximum time between transmission of DCD messages. Maximum is 10s",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetMaxDcdInterval,
                                     &SubscriberStationNetDevice::SetMaxDcdInterval),
                   MakeTimeChecker ())

    .AddAttribute ("MaxUcdInterval",
                   "Maximum time between transmission of UCD messages. Maximum is 10s",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetMaxUcdInterval,
                                     &SubscriberStationNetDevice::SetMaxUcdInterval),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT1",
                   "Wait for DCD timeout. Maximum is 5*maxDcdInterval",
                   TimeValue (Seconds (50)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT1,
                                     &SubscriberStationNetDevice::SetIntervalT1),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT2",
                   "Wait for broadcast ranging timeout, i.e., wait for initial ranging opportunity. Maximum is 5*Ranging interval",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT2,
                                     &SubscriberStationNetDevice::SetIntervalT2),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT3",
                   "ranging Response reception timeout following the transmission of a ranging request. Maximum is 200ms",
                   TimeValue (Seconds (0.2)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT3,
                                     &SubscriberStationNetDevice::SetIntervalT3),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT7",
                   "wait for DSA/DSC/DSD Response timeout. Maximum is 1s",
                   TimeValue (Seconds (0.1)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT7,
                                     &SubscriberStationNetDevice::SetIntervalT7),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT12",
                   "Wait for UCD descriptor.Maximum is 5*MaxUcdInterval",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT12,
                                     &SubscriberStationNetDevice::SetIntervalT12),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT20",
                   "Time the SS searches for preambles on a given channel. Minimum is 2 MAC frames",
                   TimeValue (Seconds (0.5)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT20,
                                     &SubscriberStationNetDevice::SetIntervalT20),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT21",
                   "time the SS searches for (decodable) DL-MAP on a given channel",
                   TimeValue (Seconds (10)),
                   MakeTimeAccessor (&SubscriberStationNetDevice::GetIntervalT21,
                                     &SubscriberStationNetDevice::SetIntervalT21),
                   MakeTimeChecker ())

    .AddAttribute ("MaxContentionRangingRetries",
                   "Number of retries on contention Ranging Requests",
                   UintegerValue (16),
                   MakeUintegerAccessor (&SubscriberStationNetDevice::GetMaxContentionRangingRetries,
                                         &SubscriberStationNetDevice::SetMaxContentionRangingRetries),
                   MakeUintegerChecker<uint8_t> (1, 16))

    .AddAttribute ("SSScheduler",
                   "The ss scheduler attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&SubscriberStationNetDevice::GetScheduler,
                                        &SubscriberStationNetDevice::SetScheduler),
                   MakePointerChecker<SSScheduler> ())

    .AddAttribute ("LinkManager",
                   "The ss link manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&SubscriberStationNetDevice::GetLinkManager,
                                        &SubscriberStationNetDevice::SetLinkManager),
                   MakePointerChecker<SSLinkManager> ())

    .AddAttribute ("Classifier",
                   "The ss classifier attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&SubscriberStationNetDevice::GetIpcsClassifier,
                                        &SubscriberStationNetDevice::SetIpcsPacketClassifier),
                   MakePointerChecker<IpcsClassifier> ())

    .AddTraceSource ("SSTxDrop",
                     "A packet has been dropped in the MAC layer before being queued for transmission.",
                     MakeTraceSourceAccessor (&SubscriberStationNetDevice::m_ssTxDropTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("SSPromiscRx",
                     "A packet has been received by this device, has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  This is a promiscuous trace,",
                     MakeTraceSourceAccessor (&SubscriberStationNetDevice::m_ssPromiscRxTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("SSRx",
                     "A packet has been received by this device, has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  This is a non-promiscuous trace,",
                     MakeTraceSourceAccessor (&SubscriberStationNetDevice::m_ssRxTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("SSRxDrop",
                     "A packet has been dropped in the MAC layer after it has been passed up from the physical "
                     "layer.",
                     MakeTraceSourceAccessor (&SubscriberStationNetDevice::m_ssRxDropTrace),
                     "ns3::Packet::TracedCallback")
    ;
  return tid;
}

SubscriberStationNetDevice::SubscriberStationNetDevice (void)
{

  InitSubscriberStationNetDevice ();

}

void
SubscriberStationNetDevice::InitSubscriberStationNetDevice (void)
{
  m_lostDlMapInterval = MilliSeconds (500);
  m_lostUlMapInterval = MilliSeconds (500);
  m_maxDcdInterval = Seconds (10);
  m_maxUcdInterval = Seconds (10);
  m_intervalT1 = Seconds (5 * m_maxDcdInterval.GetSeconds ());
  m_intervalT2 = Seconds (5 * 2); // shall be 5 * RangingInterval, if ranging interval=see T2 at page 638) means Initial Ranging Interval=see page 637)
  m_intervalT3 = MilliSeconds (200);
  m_intervalT7 = Seconds (0.1); // maximum is 1
  m_intervalT12 = Seconds (5 * m_maxUcdInterval.GetSeconds ());
  m_intervalT21 = Seconds (11);
  m_maxContentionRangingRetries = 16;
  m_dcdCount = 0;
  m_baseStationId = Mac48Address ("00:00:00:00:00:00");
  m_ucdCount = 0;
  m_allocationStartTime = 0;
  m_nrDlMapElements = 0;
  m_nrUlMapElements = 0;
  m_nrDlMapRecvd = 0;
  m_nrUlMapRecvd = 0;
  m_nrDcdRecvd = 0;
  m_nrUcdRecvd = 0;
  m_modulationType = WimaxPhy::MODULATION_TYPE_BPSK_12;
  m_areManagementConnectionsAllocated = false;
  m_areServiceFlowsAllocated = false;

  m_basicConnection = 0;
  m_primaryConnection = 0;

  m_dlBurstProfile = new OfdmDlBurstProfile ();
  m_ulBurstProfile = new OfdmUlBurstProfile ();
  m_classifier = CreateObject<IpcsClassifier> ();
  m_linkManager = CreateObject<SSLinkManager> (this);
  m_scheduler = CreateObject<SSScheduler> (this);
  m_serviceFlowManager = CreateObject<SsServiceFlowManager> (this);
}

SubscriberStationNetDevice::SubscriberStationNetDevice (Ptr<Node> node, Ptr<WimaxPhy> phy)
{
  InitSubscriberStationNetDevice ();
  this->SetNode (node);
  this->SetPhy (phy);
}

SubscriberStationNetDevice::~SubscriberStationNetDevice (void)
{
}

void
SubscriberStationNetDevice::DoDispose (void)
{
  delete m_dlBurstProfile;
  delete m_ulBurstProfile;
  m_scheduler = 0;
  m_serviceFlowManager = 0;
  m_basicConnection = 0;
  m_primaryConnection = 0;
  m_classifier = 0;
  m_dlBurstProfile = 0;
  m_ulBurstProfile = 0;

  m_linkManager = 0;

  m_asciiTxQueueEnqueueCb.Nullify ();
  m_asciiTxQueueDequeueCb.Nullify ();
  m_asciiTxQueueDropCb.Nullify ();

  WimaxNetDevice::DoDispose ();
}

void
SubscriberStationNetDevice::SetLostDlMapInterval (Time lostDlMapInterval)
{
  m_lostDlMapInterval = lostDlMapInterval;
}

Time
SubscriberStationNetDevice::GetLostDlMapInterval (void) const
{
  return m_lostDlMapInterval;
}

void
SubscriberStationNetDevice::SetLostUlMapInterval (Time lostUlMapInterval)
{
  m_lostUlMapInterval = lostUlMapInterval;
}

Time
SubscriberStationNetDevice::GetLostUlMapInterval (void) const
{
  return m_lostUlMapInterval;
}

void
SubscriberStationNetDevice::SetMaxDcdInterval (Time maxDcdInterval)
{
  m_maxDcdInterval = maxDcdInterval;
}

Time
SubscriberStationNetDevice::GetMaxDcdInterval (void) const
{
  return m_maxDcdInterval;
}

void
SubscriberStationNetDevice::SetMaxUcdInterval (Time maxUcdInterval)
{
  m_maxUcdInterval = maxUcdInterval;
}

Time
SubscriberStationNetDevice::GetMaxUcdInterval (void) const
{
  return m_maxUcdInterval;
}

void
SubscriberStationNetDevice::SetIntervalT1 (Time interval)
{
  m_intervalT1 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT1 (void) const
{
  return m_intervalT1;
}

void
SubscriberStationNetDevice::SetIntervalT2 (Time interval)
{
  m_intervalT2 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT2 (void) const
{
  return m_intervalT2;
}

void
SubscriberStationNetDevice::SetIntervalT3 (Time interval)
{
  m_intervalT3 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT3 (void) const
{
  return m_intervalT3;
}

void
SubscriberStationNetDevice::SetIntervalT7 (Time interval)
{
  m_intervalT7 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT7 (void) const
{
  return m_intervalT7;
}

void
SubscriberStationNetDevice::SetIntervalT12 (Time interval)
{
  m_intervalT12 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT12 (void) const
{
  return m_intervalT12;
}

void
SubscriberStationNetDevice::SetIntervalT20 (Time interval)
{
  m_intervalT20 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT20 (void) const
{
  return m_intervalT20;
}

void
SubscriberStationNetDevice::SetIntervalT21 (Time interval)
{
  m_intervalT21 = interval;
}

Time
SubscriberStationNetDevice::GetIntervalT21 (void) const
{
  return m_intervalT21;
}

void
SubscriberStationNetDevice::SetMaxContentionRangingRetries (uint8_t maxContentionRangingRetries)
{
  m_maxContentionRangingRetries = maxContentionRangingRetries;
}

uint8_t
SubscriberStationNetDevice::GetMaxContentionRangingRetries (void) const
{
  return m_maxContentionRangingRetries;
}

void
SubscriberStationNetDevice::SetBasicConnection (Ptr<WimaxConnection> basicConnection)
{
  m_basicConnection = basicConnection;
  uint32_t nodeId = GetNode ()->GetId ();
  uint32_t ifaceId = GetIfIndex ();

  if (!m_asciiTxQueueEnqueueCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/BasicConnection/TxQueue/Enqueue";
      Config::Connect (oss.str (), m_asciiTxQueueEnqueueCb);
    }
  if (!m_asciiTxQueueDequeueCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/BasicConnection/TxQueue/Dequeue";
      Config::Connect (oss.str (), m_asciiTxQueueDequeueCb);
    }
  if (!m_asciiTxQueueDropCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/BasicConnection/TxQueue/Drop";
      Config::Connect (oss.str (), m_asciiTxQueueDropCb);
    }
}

Ptr<WimaxConnection>
SubscriberStationNetDevice::GetBasicConnection (void) const
{
  return m_basicConnection;
}

void
SubscriberStationNetDevice::SetPrimaryConnection (Ptr<WimaxConnection> primaryConnection)
{
  m_primaryConnection = primaryConnection;

  uint32_t nodeId = GetNode ()->GetId ();
  uint32_t ifaceId = GetIfIndex ();

  if (!m_asciiTxQueueEnqueueCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/PrimaryConnection/TxQueue/Enqueue";
      Config::Connect (oss.str (), m_asciiTxQueueEnqueueCb);
    }
  if (!m_asciiTxQueueDequeueCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/PrimaryConnection/TxQueue/Dequeue";
      Config::Connect (oss.str (), m_asciiTxQueueDequeueCb);
    }
  if (!m_asciiTxQueueDropCb.IsNull ())
    {
      std::ostringstream oss;
      oss << "/NodeList/" << nodeId << "/DeviceList/" << ifaceId << "/$ns3::SubscriberStationNetDevice/PrimaryConnection/TxQueue/Drop";
      Config::Connect (oss.str (), m_asciiTxQueueDropCb);
    }
}

Ptr<WimaxConnection>
SubscriberStationNetDevice::GetPrimaryConnection (void) const
{
  return m_primaryConnection;
}

Cid
SubscriberStationNetDevice::GetBasicCid (void) const
{
  return m_basicConnection->GetCid ();
}

Cid
SubscriberStationNetDevice::GetPrimaryCid (void) const
{
  return m_primaryConnection->GetCid ();
}

void
SubscriberStationNetDevice::SetModulationType (WimaxPhy::ModulationType modulationType)
{
  m_modulationType = modulationType;
}

WimaxPhy::ModulationType
SubscriberStationNetDevice::GetModulationType (void) const
{
  return m_modulationType;
}

void
SubscriberStationNetDevice::SetAreManagementConnectionsAllocated (bool areManagementConnectionsAllocated)
{
  m_areManagementConnectionsAllocated = areManagementConnectionsAllocated;
}

bool
SubscriberStationNetDevice::GetAreManagementConnectionsAllocated (void) const
{
  return m_areManagementConnectionsAllocated;
}

void
SubscriberStationNetDevice::SetAreServiceFlowsAllocated (bool areServiceFlowsAllocated)
{
  m_areServiceFlowsAllocated = areServiceFlowsAllocated;
}

bool
SubscriberStationNetDevice::GetAreServiceFlowsAllocated (void) const
{
  return m_areServiceFlowsAllocated;
}

Ptr<SSScheduler>
SubscriberStationNetDevice::GetScheduler (void) const
{
  return m_scheduler;
}

void
SubscriberStationNetDevice::SetScheduler (Ptr<SSScheduler> scheduler)
{
  m_scheduler = scheduler;
}

bool
SubscriberStationNetDevice::HasServiceFlows (void) const
{
  return GetServiceFlowManager ()->GetServiceFlows (ServiceFlow::SF_TYPE_ALL).size () > 0;
}

Ptr<IpcsClassifier>
SubscriberStationNetDevice::GetIpcsClassifier () const
{

  return m_classifier;
}

void
SubscriberStationNetDevice::SetIpcsPacketClassifier (Ptr<IpcsClassifier> classifier)
{
  m_classifier = classifier;
}

Ptr<SSLinkManager>
SubscriberStationNetDevice::GetLinkManager (void) const
{
  return m_linkManager;
}

void
SubscriberStationNetDevice::SetLinkManager (Ptr<SSLinkManager> linkManager)
{
  m_linkManager = linkManager;
}

Ptr<SsServiceFlowManager>
SubscriberStationNetDevice::GetServiceFlowManager (void) const
{
  return m_serviceFlowManager;
}

void
SubscriberStationNetDevice::SetServiceFlowManager (Ptr<SsServiceFlowManager> sfm)
{
  m_serviceFlowManager = sfm;
}

void
SubscriberStationNetDevice::Start (void)
{
  SetReceiveCallback ();

  GetPhy ()->SetPhyParameters ();
  GetPhy ()->SetDataRates ();
  m_intervalT20 = Seconds (4 * GetPhy ()->GetFrameDuration ().GetSeconds ());

  CreateDefaultConnections ();
  Simulator::ScheduleNow (&SSLinkManager::StartScanning, m_linkManager, EVENT_NONE, false);
}

void
SubscriberStationNetDevice::Stop (void)
{
  SetState (SS_STATE_STOPPED);
}

void
SubscriberStationNetDevice::AddServiceFlow (ServiceFlow sf)
{
  GetServiceFlowManager ()->AddServiceFlow (sf);
}

void
SubscriberStationNetDevice::AddServiceFlow (ServiceFlow * sf)
{
  GetServiceFlowManager ()->AddServiceFlow (sf);
}

bool
SubscriberStationNetDevice::DoSend (Ptr<Packet> packet,
                                    const Mac48Address &source,
                                    const Mac48Address &dest,
                                    uint16_t protocolNumber)
{
  NS_LOG_INFO ("SS (" << source << "):" );
  NS_LOG_INFO ("\tSending packet..." );
  NS_LOG_INFO ("\t\tDestination: " << dest );
  NS_LOG_INFO ("\t\tPacket Size:  " << packet->GetSize () );
  NS_LOG_INFO ("\t\tProtocol:    " << protocolNumber );

  ServiceFlow *serviceFlow = 0;

  if (IsRegistered ())
    {
      NS_LOG_DEBUG ("SS (Basic CID: " << m_basicConnection->GetCid () << ")");
    }
  else
    {
      NS_LOG_DEBUG ("SS (" << GetMacAddress () << ")");
      NS_LOG_INFO ("\tCan't send packet! (NotRegitered with the network)");
      return false;
    }

  NS_LOG_DEBUG ("packet to send, size : " << packet->GetSize () << ", destination : " << dest);

  if (GetServiceFlowManager ()->GetNrServiceFlows () == 0)
    {
      NS_LOG_INFO ("\tCan't send packet! (No service Flow)");
      return false;
    }

  if (protocolNumber == 2048)
    {
      serviceFlow = m_classifier->Classify (packet, GetServiceFlowManager (), ServiceFlow::SF_DIRECTION_UP);
    }

  if ((protocolNumber != 2048) || (serviceFlow == NULL))
    {
      serviceFlow = *GetServiceFlowManager ()->GetServiceFlows (ServiceFlow::SF_TYPE_ALL).begin ();
      NS_LOG_INFO ("\tNo service flows matches...using the default one.");
    }

  NS_LOG_INFO ("\tPacket classified in the service flow SFID =  " << serviceFlow->GetSfid () << " CID = "
                                                                  << serviceFlow->GetCid ());
  if (serviceFlow->GetIsEnabled ())
    {
      if (!Enqueue (packet, MacHeaderType (), serviceFlow->GetConnection ()))
        {
          NS_LOG_INFO ("\tEnqueue ERROR!!" );
          m_ssTxDropTrace (packet);
          return false;
        }
      else
        {
          m_ssTxTrace (packet);
        }
    }
  else
    {
      NS_LOG_INFO ("Error!! The Service Flow is not enabled" );
      m_ssTxDropTrace (packet);
      return false;
    }

  return true;
}

bool
SubscriberStationNetDevice::Enqueue (Ptr<Packet> packet,
                                     const MacHeaderType &hdrType,
                                     Ptr<WimaxConnection> connection)
{
  NS_ASSERT_MSG (connection != 0, "SS: Can not enqueue the packet: the selected connection is nor initialized");

  GenericMacHeader hdr;

  if (hdrType.GetType () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      hdr.SetLen (packet->GetSize () + hdr.GetSerializedSize ());
      hdr.SetCid (connection->GetCid ());

    }

  if (connection->GetType () == Cid::TRANSPORT)
    {

      if (connection->GetSchedulingType () == ServiceFlow::SF_TYPE_UGS && m_scheduler->GetPollMe ())
        {
          NS_ASSERT_MSG (hdrType.GetType () != MacHeaderType::HEADER_TYPE_BANDWIDTH,
                         "Error while equeuing  packet: incorrect header type");

          GrantManagementSubheader grantMgmntSubhdr;
          grantMgmntSubhdr.SetPm (true);
          packet->AddHeader (grantMgmntSubhdr);
        }
    }
  NS_LOG_INFO ("ServiceFlowManager: enqueuing packet" );
  return connection->Enqueue (packet, hdrType, hdr);
}

void
SubscriberStationNetDevice::SendBurst (uint8_t uiuc,
                                       uint16_t nrSymbols,
                                       Ptr<WimaxConnection> connection,
                                       MacHeaderType::HeaderType packetType)
{
  WimaxPhy::ModulationType modulationType;

  if (uiuc == OfdmUlBurstProfile::UIUC_INITIAL_RANGING || uiuc == OfdmUlBurstProfile::UIUC_REQ_REGION_FULL)
    {
      modulationType = WimaxPhy::MODULATION_TYPE_BPSK_12;
    }
  else
    {
      modulationType = GetBurstProfileManager ()->GetModulationType (uiuc, DIRECTION_UPLINK);
    }
  Ptr<PacketBurst> burst = m_scheduler->Schedule (nrSymbols, modulationType, packetType, connection);

  if (burst->GetNPackets () == 0)
    {
      return;
    }

  if (IsRegistered ())
    {
      NS_LOG_DEBUG ("SS (Basic CID: " << m_basicConnection->GetCid () << ")");
    }
  else
    {
      NS_LOG_DEBUG ("SS (" << GetMacAddress () << ")");
    }

  if (connection->GetType () == Cid::TRANSPORT)
    {
      ServiceFlowRecord *record = connection->GetServiceFlow ()->GetRecord ();
      record->UpdatePktsSent (burst->GetNPackets ());
      record->UpdateBytesSent (burst->GetSize ());

      NS_LOG_DEBUG (" sending burst" << ", SFID: " << connection->GetServiceFlow ()->GetSfid () << ", pkts sent: "
                                     << record->GetPktsSent () << ", pkts rcvd: " << record->GetPktsRcvd () << ", bytes sent: "
                                     << record->GetBytesSent () << ", bytes rcvd: " << record->GetBytesRcvd () );

    }
  else
    {

    }
  ForwardDown (burst, modulationType);
}

void
SubscriberStationNetDevice::DoReceive (Ptr<Packet> packet)
{
  GenericMacHeader gnrcMacHdr;
  ManagementMessageType msgType;
  RngRsp rngrsp;
  Cid cid;
  uint32_t pktSize = packet->GetSize ();
  packet->RemoveHeader (gnrcMacHdr);
  FragmentationSubheader fragSubhdr;
  bool fragmentation = false;  // it becomes true when there is a fragmentation subheader

  if (gnrcMacHdr.GetHt () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      if (gnrcMacHdr.check_hcs () == false)
        {
          // The header is noisy
          NS_LOG_INFO ("Header HCS ERROR");
          m_ssRxDropTrace (packet);
          return;
        }

      cid = gnrcMacHdr.GetCid ();

      // checking for subheaders
      uint8_t type = gnrcMacHdr.GetType ();
      if (type)
        {
          // Check if there is a fragmentation Subheader
          uint8_t tmpType = type;
          if (((tmpType >> 2) & 1) == 1)
            {
              // a TRANSPORT packet with fragmentation subheader has been received!
              fragmentation = true;
              NS_LOG_INFO ("SS DoReceive -> the packet is a fragment" <<  std::endl);
            }
        }

      if (cid == GetBroadcastConnection ()->GetCid () && !fragmentation)
        {
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_DL_MAP:
              {
                if (GetState () == SS_STATE_SYNCHRONIZING)
                  {
                    Simulator::Cancel (m_linkManager->GetDlMapSyncTimeoutEvent ());
                  }

                if (m_lostDlMapEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_lostDlMapEvent);
                  }

                m_linkManager->ScheduleScanningRestart (m_lostDlMapInterval, EVENT_LOST_DL_MAP, false, m_lostDlMapEvent);

                if (m_dcdWaitTimeoutEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_dcdWaitTimeoutEvent);
                  }

                m_linkManager->ScheduleScanningRestart (m_intervalT1,
                                                        EVENT_DCD_WAIT_TIMEOUT,
                                                        false,
                                                        m_dcdWaitTimeoutEvent);

                if (m_ucdWaitTimeoutEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_ucdWaitTimeoutEvent);
                  }

                m_linkManager->ScheduleScanningRestart (m_intervalT12,
                                                        EVENT_UCD_WAIT_TIMEOUT,
                                                        true,
                                                        m_ucdWaitTimeoutEvent);

                DlMap dlmap;
                packet->RemoveHeader (dlmap);
                ProcessDlMap (dlmap);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_UL_MAP:
              {
                if (m_lostUlMapEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_lostUlMapEvent);
                    m_linkManager->ScheduleScanningRestart (m_lostUlMapInterval,
                                                            EVENT_LOST_UL_MAP,
                                                            true,
                                                            m_lostUlMapEvent);
                  }

                UlMap ulmap;
                packet->RemoveHeader (ulmap);

                ProcessUlMap (ulmap);

                if (GetState () == SS_STATE_WAITING_REG_RANG_INTRVL)
                  {
                    if (m_linkManager->GetRangingIntervalFound ())
                      {
                        if (m_rangOppWaitTimeoutEvent.IsRunning ())
                          {
                            Simulator::Cancel (m_rangOppWaitTimeoutEvent);
                          }
                        m_linkManager->PerformBackoff ();
                      }
                  }
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_DCD:
              {
                if (GetState () == SS_STATE_SYNCHRONIZING)
                  {
                    SetState (SS_STATE_ACQUIRING_PARAMETERS);
                  }

                if (m_dcdWaitTimeoutEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_dcdWaitTimeoutEvent);
                    m_linkManager->ScheduleScanningRestart (m_intervalT1,
                                                            EVENT_DCD_WAIT_TIMEOUT,
                                                            false,
                                                            m_dcdWaitTimeoutEvent);
                  }

                Dcd dcd;
                // number of burst profiles is set to number of DL-MAP IEs after processing DL-MAP, not a very good solution
                // dcd.SetNrDlBurstProfiles (m_nrDlMapElements);
                dcd.SetNrDlBurstProfiles (7);
                packet->RemoveHeader (dcd);

                ProcessDcd (dcd);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_UCD:
              {
                Ucd ucd;
                // number of burst profiles is set to number of UL-MAP IEs after processing UL-MAP, not a very good solution
                // ucd.SetNrUlBurstProfiles (m_nrUlMapElements);
                ucd.SetNrUlBurstProfiles (7);
                packet->RemoveHeader (ucd);

                ProcessUcd (ucd);

                if (m_ucdWaitTimeoutEvent.IsRunning ())
                  {
                    Simulator::Cancel (m_ucdWaitTimeoutEvent);
                    m_linkManager->ScheduleScanningRestart (m_intervalT12,
                                                            EVENT_UCD_WAIT_TIMEOUT,
                                                            true,
                                                            m_ucdWaitTimeoutEvent);
                  }

                if (GetState () == SS_STATE_ACQUIRING_PARAMETERS)
                  {
                    /*state indicating that SS has completed scanning, synchronization and parameter acquisition
                     successfully and now waiting for UL-MAP to start initial ranging.*/
                    SetState (SS_STATE_WAITING_REG_RANG_INTRVL);

                    m_linkManager->ScheduleScanningRestart (m_intervalT2,
                                                            EVENT_RANG_OPP_WAIT_TIMEOUT,
                                                            false,
                                                            m_rangOppWaitTimeoutEvent);
                    m_linkManager->ScheduleScanningRestart (m_lostUlMapInterval,
                                                            EVENT_LOST_UL_MAP,
                                                            true,
                                                            m_lostUlMapEvent);
                  }
                break;
              }
            default:
              NS_FATAL_ERROR ("Invalid management message type");
            }
        }
      else if (GetInitialRangingConnection () != 0 && cid == GetInitialRangingConnection ()->GetCid () && !fragmentation)
        {
          m_traceSSRx (packet, GetMacAddress (), cid);
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_RNG_REQ:
              // intended for base station, ignore
              break;
            case ManagementMessageType::MESSAGE_TYPE_RNG_RSP:
              NS_ASSERT_MSG (GetState () >= SS_STATE_WAITING_REG_RANG_INTRVL,
                             "SS: Error while receiving a ranging response message: SS state should be at least SS_STATE_WAITING_REG_RANG_INTRVL");
              packet->RemoveHeader (rngrsp);
              m_linkManager->PerformRanging (cid, rngrsp);
              break;
            default:
              NS_LOG_ERROR ("Invalid management message type");
            }
        }
      else if (m_basicConnection != 0 && cid == m_basicConnection->GetCid () && !fragmentation)
        {
          m_traceSSRx (packet, GetMacAddress (), cid);
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_RNG_REQ:
              // intended for base station, ignore
              break;
            case ManagementMessageType::MESSAGE_TYPE_RNG_RSP:
              NS_ASSERT_MSG (GetState () == SS_STATE_WAITING_RNG_RSP,
                             "SS: Error while receiving a ranging response message: SS state should be SS_STATE_WAITING_RNG_RSP");
              packet->RemoveHeader (rngrsp);
              m_linkManager->PerformRanging (cid, rngrsp);
              break;
            default:
              NS_LOG_ERROR ("Invalid management message type");
            }
        }
      else if (m_primaryConnection != 0 && cid == m_primaryConnection->GetCid () && !fragmentation)
        {
          m_traceSSRx (packet, GetMacAddress (), cid);
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_REG_REQ:
              // not yet implemented
              break;
            case ManagementMessageType::MESSAGE_TYPE_REG_RSP:
              // intended for base station, ignore
              break;
            case ManagementMessageType::MESSAGE_TYPE_DSA_REQ:
              /*from other station as DSA initiation
               by BS is not supported, ignore*/
              break;
            case ManagementMessageType::MESSAGE_TYPE_DSA_RSP:
              {
                Simulator::Cancel (GetServiceFlowManager ()->GetDsaRspTimeoutEvent ());
                DsaRsp dsaRsp;
                packet->RemoveHeader (dsaRsp);
                GetServiceFlowManager ()->ProcessDsaRsp (dsaRsp);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_DSA_ACK:
              /*from other station as DSA initiation
               by BS is not supported, ignore*/
              break;
            default:
              NS_LOG_ERROR ("Invalid management message type");
            }
        }
      else if (GetConnectionManager ()->GetConnection (cid)) // transport connection
        {
          ServiceFlow *serviceFlow = GetConnectionManager ()->GetConnection (cid)->GetServiceFlow ();
          ServiceFlowRecord *record = serviceFlow->GetRecord ();

          record->UpdatePktsRcvd (1);
          record->UpdateBytesRcvd (pktSize);

          // If fragmentation is true, the packet is a fragment.
          if (!fragmentation)
            {
              m_ssRxTrace (packet);
              ForwardUp (packet, m_baseStationId, GetMacAddress ()); // source shall be BS's address or sender SS's?
            }
          else
            {
              NS_LOG_INFO ( "FRAG_DEBUG: SS DoReceive, the Packet is a fragment" << std::endl);
              packet->RemoveHeader (fragSubhdr);
              uint32_t fc = fragSubhdr.GetFc ();
              NS_LOG_INFO ( "\t fragment size = " << packet->GetSize () << std::endl);

              if (fc == 2)
                {
                  // This is the latest fragment.
                  // Take the fragment queue, defragment a packet and send it to the upper layer
                  NS_LOG_INFO ( "\t Received the latest fragment" << std::endl);
                  GetConnectionManager ()->GetConnection (cid)
                  ->FragmentEnqueue (packet);

                  WimaxConnection::FragmentsQueue fragmentsQueue = GetConnectionManager ()->
                    GetConnection (cid)->GetFragmentsQueue ();

                  Ptr<Packet> fullPacket = Create<Packet> ();

                  // DEFRAGMENTATION
                  NS_LOG_INFO ( "\t SS PACKET DEFRAGMENTATION" << std::endl);
                  for (std::list<Ptr<const Packet> >::const_iterator iter = fragmentsQueue.begin ();
                       iter != fragmentsQueue.end (); ++iter)
                    {
                      // Create the whole Packet
                      fullPacket->AddAtEnd (*iter);
                    }
                  GetConnectionManager ()->GetConnection (cid)
                  ->ClearFragmentsQueue ();
                  NS_LOG_INFO ( "\t fullPacket size = " << fullPacket->GetSize () << std::endl);

                  m_ssRxTrace (fullPacket);
                  ForwardUp (fullPacket, m_baseStationId, GetMacAddress ()); // source shall be BS's address or sender SS's?
                }
              else
                {
                  // This is the first or middle fragment.
                  // Take the fragment queue, store the fragment into the queue
                  NS_LOG_INFO ( "\t Received the first or the middle fragment" << std::endl);
                  GetConnectionManager ()->GetConnection (cid)->FragmentEnqueue (packet);
                }
            }
        }
      else if (cid.IsMulticast ())
        {
          m_traceSSRx (packet, GetMacAddress (), cid);
          ForwardUp (packet, m_baseStationId, GetMacAddress ()); // source shall be BS's address or sender SS's?
        }
      else if (IsPromisc ())
        {
          NotifyPromiscTrace (packet);
          m_ssPromiscRxTrace (packet);

          // not for me, ignore
        }
      else
        {
          // not for me drop
        }
    }
  else
    {
      // from other SS, ignore
    }
}

void
SubscriberStationNetDevice::ProcessDlMap (const DlMap &dlmap)
{
  m_nrDlMapRecvd++;
  m_dcdCount = dlmap.GetDcdCount ();
  m_baseStationId = dlmap.GetBaseStationId ();
  std::list<OfdmDlMapIe> dlMapElements = dlmap.GetDlMapElements ();

  for (std::list<OfdmDlMapIe>::iterator iter = dlMapElements.begin (); iter != dlMapElements.end (); ++iter)
    {
      if (iter->GetDiuc () == OfdmDlBurstProfile::DIUC_END_OF_MAP)
        {
          break;
        }

      if (iter->GetCid () == m_basicConnection->GetCid ())
        {
          /*here the SS shall actually acquire the start time it shall start receiving the burst at. start time is used for power saving
           which is not implemented here, furthermore there is no need since the simulator architecture automatically callbacks the receive
           function. shall acquire the DIUC (burst profile) as well to decode the burst, again not required again because the callback
           mechanism automatically passes it as parameter.*/
        }

#if 0 /* a template for future implementation following */
      uint8_t temp = iter->GetDiuc ();
      temp = iter->GetPreamblePresent ();
      temp = iter->GetStartTime ();
#endif
    }
}

void
SubscriberStationNetDevice::ProcessUlMap (const UlMap &ulmap)
{
  m_nrUlMapRecvd++;
  m_ucdCount = ulmap.GetUcdCount ();
  m_allocationStartTime = ulmap.GetAllocationStartTime ();
  std::list<OfdmUlMapIe> ulMapElements = ulmap.GetUlMapElements ();
  m_linkManager->SetRangingIntervalFound (false);

  for (std::list<OfdmUlMapIe>::iterator iter = ulMapElements.begin (); iter != ulMapElements.end (); ++iter)
    {
      OfdmUlMapIe ulMapIe = *iter;

      if (ulMapIe.GetUiuc () == OfdmUlBurstProfile::UIUC_END_OF_MAP)
        {
          break;
        }

      Cid cid = ulMapIe.GetCid ();

      if (ulMapIe.GetUiuc () == OfdmUlBurstProfile::UIUC_INITIAL_RANGING && cid == GetBroadcastConnection ()->GetCid ())
        {
          m_linkManager->SetRangingIntervalFound (true);
        }

      if (m_areManagementConnectionsAllocated && cid == m_basicConnection->GetCid ())
        {

          Time timeToAllocation = GetTimeToAllocation (Seconds (ulMapIe.GetStartTime ()
                                                                * GetPhy ()->GetSymbolDuration ().GetSeconds ()));

          if (ulMapIe.GetUiuc () == OfdmUlBurstProfile::UIUC_INITIAL_RANGING) // invited ranging interval

            {

              m_linkManager->IncrementNrInvitedPollsRecvd ();
              NS_ASSERT_MSG (GetState () == SS_STATE_WAITING_INV_RANG_INTRVL,
                             "SS: Error while processing UL MAP: SS state should be SS_STATE_WAITING_INV_RANG_INTRVL");
              Simulator::Schedule (timeToAllocation,
                                   &SSLinkManager::SendRangingRequest,
                                   m_linkManager,
                                   ulMapIe.GetUiuc (),
                                   ulMapIe.GetDuration ());
            }
          else if (ulMapIe.GetUiuc () == OfdmUlBurstProfile::UIUC_REQ_REGION_FULL) // unicast poll

            {

              Simulator::Schedule (timeToAllocation,
                                   &BandwidthManager::SendBandwidthRequest,
                                   GetBandwidthManager (),
                                   ulMapIe.GetUiuc (),
                                   ulMapIe.GetDuration ());
            }
          else // regular allocation/grant for data, for UGS flows or in response of requests for non-UGS flows

            {

              Ptr<WimaxConnection> connection = NULL;
              Simulator::Schedule (timeToAllocation,
                                   &SubscriberStationNetDevice::SendBurst,
                                   this,
                                   ulMapIe.GetUiuc (),
                                   ulMapIe.GetDuration (),
                                   connection,
                                   MacHeaderType::HEADER_TYPE_GENERIC);
            }
        }
      else
        {
          if (ulMapIe.GetUiuc () == OfdmUlBurstProfile::UIUC_INITIAL_RANGING && cid
              == GetBroadcastConnection ()->GetCid ()) // regular ranging interval

            {
              if (GetCurrentUcd ().GetChannelEncodings ().GetRangReqOppSize () != 0)
                {
                  m_linkManager->SetNrRangingTransOpps ((ulMapIe.GetDuration () * GetPhy ()->GetPsPerSymbol ())
                                                        / GetCurrentUcd ().GetChannelEncodings ().GetRangReqOppSize ());

                }

              if (GetState () == SS_STATE_WAITING_REG_RANG_INTRVL || GetState () == SS_STATE_ADJUSTING_PARAMETERS)
                {
                  m_linkManager->StartContentionResolution ();
                }

            }
        }
    }
}

void
SubscriberStationNetDevice::ProcessDcd (const Dcd &dcd)
{
  m_nrDcdRecvd++;
  if (dcd.GetConfigurationChangeCount () == GetCurrentDcd ().GetConfigurationChangeCount ())
    {
      return; // nothing new in DCD so don't read

    }
  SetCurrentDcd (dcd);
  OfdmDcdChannelEncodings dcdChnlEncodings = dcd.GetChannelEncodings ();

  // parameters for initial ranging
  m_linkManager->SetBsEirp (dcdChnlEncodings.GetBsEirp ());
  m_linkManager->SetEirXPIrMax (dcdChnlEncodings.GetEirxPIrMax ());

  GetPhy ()->GetFrameDuration (dcdChnlEncodings.GetFrameDurationCode ());

  std::vector<OfdmDlBurstProfile> dlBurstProfiles = dcd.GetDlBurstProfiles ();

  for (std::vector<OfdmDlBurstProfile>::iterator iter = dlBurstProfiles.begin (); iter != dlBurstProfiles.end (); ++iter)
    {
      OfdmDlBurstProfile brstProfile = *iter;

      /*NS-2 does this, may be not correct, assumes DIUC/UIUC to
       modulation type mapping in DCD/UCD may change over time*/
      if (brstProfile.GetFecCodeType () == m_modulationType)
        {
          m_dlBurstProfile->SetFecCodeType (brstProfile.GetFecCodeType ());
          m_dlBurstProfile->SetDiuc (brstProfile.GetDiuc ());
        }
    }
}

void
SubscriberStationNetDevice::ProcessUcd (const Ucd &ucd)
{
  m_nrUcdRecvd++;
  if (!m_linkManager->IsUlChannelUsable ())
    {
      m_linkManager->StartScanning (EVENT_NONE, false);
      return;
    }

  if (ucd.GetConfigurationChangeCount () == GetCurrentUcd ().GetConfigurationChangeCount ())
    {
      return; // nothing new in UCD so don't read

    }
  SetCurrentUcd (ucd);
  m_linkManager->SetRangingCW ((uint8_t) std::pow ((double) 2, (double) ucd.GetRangingBackoffStart ()) - 1); // initializing ranging CW
  OfdmUcdChannelEncodings ucdChnlEncodings = ucd.GetChannelEncodings ();

  std::vector<OfdmUlBurstProfile> ulBurstProfiles = ucd.GetUlBurstProfiles ();

  for (std::vector<OfdmUlBurstProfile>::iterator iter = ulBurstProfiles.begin (); iter != ulBurstProfiles.end (); ++iter)
    {
      OfdmUlBurstProfile brstProfile = *iter;

      /*NS-2 does this, may be not correct, assumes DIUC/UIUC to
       modulation type mapping in DCD/UCD may change over time*/
      if (brstProfile.GetFecCodeType () == m_modulationType)
        {
          m_ulBurstProfile->SetFecCodeType (brstProfile.GetFecCodeType ());
          m_ulBurstProfile->SetUiuc (brstProfile.GetUiuc ());
        }
    }
}

/*temporarily assuming registered if ranging is complete,
 shall actually consider the registration step also */
bool
SubscriberStationNetDevice::IsRegistered (void) const
{
  return GetState () >= SS_STATE_REGISTERED;
}

Time
SubscriberStationNetDevice::GetTimeToAllocation (Time deferTime)
{
  Time timeAlreadyElapsed = Simulator::Now () - m_frameStartTime;
  Time timeToUlSubframe = Seconds (m_allocationStartTime * GetPhy ()->GetPsDuration ().GetSeconds ())
    - timeAlreadyElapsed;
  return timeToUlSubframe + deferTime;
}

void
SubscriberStationNetDevice::SetTimer (EventId eventId, EventId &event)
{
  if (GetState () == SS_STATE_STOPPED)
    {
      Simulator::Cancel (eventId); // cancelling this event (already scheduled in function call)
      return;
    }

  event = eventId;
}

void
SubscriberStationNetDevice::SetAsciiTxQueueEnqueueCallback (AsciiTraceCallback cb)
{
  m_asciiTxQueueEnqueueCb = cb;
}

void
SubscriberStationNetDevice::SetAsciiTxQueueDequeueCallback (AsciiTraceCallback cb)
{
  m_asciiTxQueueDequeueCb = cb;
}

void
SubscriberStationNetDevice::SetAsciiTxQueueDropCallback (AsciiTraceCallback cb)
{
  m_asciiTxQueueDropCb = cb;
}

} // namespace ns`
