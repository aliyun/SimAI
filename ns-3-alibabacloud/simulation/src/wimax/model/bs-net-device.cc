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

#include <cmath>

#include "ns3/simulator.h"
#include "ns3/node.h"
#include "bs-uplink-scheduler.h"
#include "bs-net-device.h"
#include "wimax-phy.h"
#include "ns3/packet-burst.h"
#include "ss-record.h"
#include "bs-scheduler.h"
#include "wimax-mac-queue.h"
#include "burst-profile-manager.h"
#include "ss-manager.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/pointer.h"
#include "ns3/enum.h"
#include "ns3/uinteger.h"
#include "service-flow.h"
#include "service-flow-manager.h"
#include "connection-manager.h"
#include "bs-link-manager.h"
#include "bandwidth-manager.h"
#include "ns3/ipv4-address.h"
#include "ns3/llc-snap-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BaseStationNetDevice");

NS_OBJECT_ENSURE_REGISTERED (BaseStationNetDevice);

TypeId BaseStationNetDevice::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BaseStationNetDevice")

    .SetParent<WimaxNetDevice> ()

    .SetGroupName("Wimax")

    .AddConstructor<BaseStationNetDevice> ()

    .AddAttribute ("BSScheduler",
                   "Downlink Scheduler for BS",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::m_scheduler),
                   MakePointerChecker<BSScheduler> ())

    .AddAttribute ("InitialRangInterval",
                   "Time between Initial Ranging regions assigned by the BS. Maximum is 2s",
                   TimeValue (Seconds (0.05)),
                   MakeTimeAccessor (&BaseStationNetDevice::GetInitialRangingInterval,
                                     &BaseStationNetDevice::SetInitialRangingInterval),
                   MakeTimeChecker ())

    .AddAttribute ("DcdInterval",
                   "Time between transmission of DCD messages. Maximum value is 10s.",
                   TimeValue (Seconds (3)),
                   MakeTimeAccessor (&BaseStationNetDevice::GetDcdInterval, &BaseStationNetDevice::SetDcdInterval),
                   MakeTimeChecker ())

    .AddAttribute ("UcdInterval",
                   "Time between transmission of UCD messages. Maximum value is 10s.",
                   TimeValue (Seconds (3)),
                   MakeTimeAccessor (&BaseStationNetDevice::GetUcdInterval, &BaseStationNetDevice::SetUcdInterval),
                   MakeTimeChecker ())

    .AddAttribute ("IntervalT8",
                   "Wait for DSA/DSC Acknowledge timeout. Maximum 300ms.",
                   TimeValue (Seconds (0.05)),
                   MakeTimeAccessor (&BaseStationNetDevice::GetIntervalT8, &BaseStationNetDevice::SetIntervalT8),
                   MakeTimeChecker ())

    .AddAttribute ("RangReqOppSize",
                   "The ranging opportunity size in symbols",
                   UintegerValue (8),
                   MakeUintegerAccessor (&BaseStationNetDevice::GetRangReqOppSize,
                                         &BaseStationNetDevice::SetRangReqOppSize),
                   MakeUintegerChecker<uint8_t> (1, 256))

    .AddAttribute ("BwReqOppSize",
                   "The bandwidth request opportunity size in symbols",
                   UintegerValue (2),
                   MakeUintegerAccessor (&BaseStationNetDevice::GetBwReqOppSize, &BaseStationNetDevice::SetBwReqOppSize),
                   MakeUintegerChecker<uint8_t> (1, 256))

    .AddAttribute ("MaxRangCorrectionRetries",
                   "Number of retries on contention Ranging Requests",
                   UintegerValue (16),
                   MakeUintegerAccessor (&BaseStationNetDevice::GetMaxRangingCorrectionRetries,
                                         &BaseStationNetDevice::SetMaxRangingCorrectionRetries),
                   MakeUintegerChecker<uint8_t> (1, 16))

    .AddAttribute ("SSManager",
                   "The ss manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetSSManager, &BaseStationNetDevice::SetSSManager),
                   MakePointerChecker<SSManager> ())

    .AddAttribute ("Scheduler",
                   "The BS scheduler attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetBSScheduler, &BaseStationNetDevice::SetBSScheduler),
                   MakePointerChecker<BSScheduler> ())

    .AddAttribute ("LinkManager",
                   "The link manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetLinkManager, &BaseStationNetDevice::SetLinkManager),
                   MakePointerChecker<BSLinkManager> ())

    .AddAttribute ("UplinkScheduler",
                   "The uplink scheduler attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetUplinkScheduler,
                                        &BaseStationNetDevice::SetUplinkScheduler),
                   MakePointerChecker<UplinkScheduler> ())

    .AddAttribute ("BsIpcsPacketClassifier",
                   "The uplink IP packet classifier attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetBsClassifier, &BaseStationNetDevice::SetBsClassifier),
                   MakePointerChecker<IpcsClassifier> ())

    .AddAttribute ("ServiceFlowManager",
                   "The service flow manager attached to this device.",
                   PointerValue (),
                   MakePointerAccessor (&BaseStationNetDevice::GetServiceFlowManager,
                                        &BaseStationNetDevice::SetServiceFlowManager),
                   MakePointerChecker<ServiceFlowManager> ())

    .AddTraceSource ("BSTx",
                     "A packet has been received from higher layers "
                     "and is being processed in preparation "
                     "for queueing for transmission.",
                     MakeTraceSourceAccessor (&BaseStationNetDevice::m_bsTxTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("BSTxDrop",
                     "A packet has been dropped in the MAC layer "
                     "before being queued for transmission.",
                     MakeTraceSourceAccessor (&BaseStationNetDevice::m_bsTxDropTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("BSPromiscRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a promiscuous trace,",
                     MakeTraceSourceAccessor (&BaseStationNetDevice::m_bsPromiscRxTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("BSRx",
                     "A packet has been received by this device, "
                     "has been passed up from the physical layer "
                     "and is being forwarded up the local protocol stack.  "
                     "This is a non-promiscuous trace,",
                     MakeTraceSourceAccessor (&BaseStationNetDevice::m_bsRxTrace),
                     "ns3::Packet::TracedCallback")

    .AddTraceSource ("BSRxDrop",
                     "A packet has been dropped in the MAC layer "
                     "after it has been passed up from the physical layer.",
                     MakeTraceSourceAccessor (&BaseStationNetDevice::m_bsRxDropTrace),
                     "ns3::Packet::TracedCallback");
  return tid;
}

BaseStationNetDevice::BaseStationNetDevice (void)
{
  InitBaseStationNetDevice ();
}

void
BaseStationNetDevice::InitBaseStationNetDevice ()
{

  m_initialRangInterval = Seconds (0.05); // maximum is 2
  m_dcdInterval = Seconds (3); // maximum is 10
  m_ucdInterval = Seconds (3); // maximum is 10
  m_intervalT8 = MilliSeconds (50); // maximum is 300 milliseconds
  m_maxRangCorrectionRetries = 16;
  m_maxInvitedRangRetries = 16;
  m_rangReqOppSize = 8; // 8 symbols = 2 (preamble) + 2 (RNG-REQ) + 4 (round-trip propagation time)
  m_bwReqOppSize = 2; // 2 symbols = 1 (preamble) + 1 (bandwidth request header)
  m_nrDlSymbols = 0;
  m_nrUlSymbols = 0;
  m_nrDlMapSent = 0;
  m_nrUlMapSent = 0;
  m_nrDcdSent = 0;
  m_nrUcdSent = 0;
  m_dcdConfigChangeCount = 0;
  m_ucdConfigChangeCount = 0;
  m_framesSinceLastDcd = 0;
  m_framesSinceLastUcd = 0;
  m_nrDlFrames = 0;
  m_nrUlFrames = 0;
  m_nrSsRegistered = 0;
  m_nrDlAllocations = 0;
  m_nrUlAllocations = 0;
  m_dlSubframeStartTime = Seconds (0);
  m_ulSubframeStartTime = Seconds (0);
  m_ulAllocationNumber = 0;
  m_rangingOppNumber = 0;
  m_allocationStartTime = 0;
  m_psDuration = Seconds (0);
  m_symbolDuration = Seconds (0);
  m_linkManager = CreateObject<BSLinkManager> (this);
  m_cidFactory = new CidFactory ();
  m_ssManager = CreateObject<SSManager> ();
  m_bsClassifier = CreateObject<IpcsClassifier> ();
  m_serviceFlowManager = CreateObject<BsServiceFlowManager> (this);

}

BaseStationNetDevice::BaseStationNetDevice (Ptr<Node> node, Ptr<WimaxPhy> phy)
{
  InitBaseStationNetDevice ();
  this->SetNode (node);
  this->SetPhy (phy);
}

BaseStationNetDevice::BaseStationNetDevice (Ptr<Node> node,
                                            Ptr<WimaxPhy> phy,
                                            Ptr<UplinkScheduler> uplinkScheduler,
                                            Ptr<BSScheduler> bsScheduler)
{
  InitBaseStationNetDevice ();
  this->SetNode (node);
  this->SetPhy (phy);
  m_uplinkScheduler = uplinkScheduler;
  m_scheduler = bsScheduler;
}

BaseStationNetDevice::~BaseStationNetDevice (void)
{
}

void
BaseStationNetDevice::DoDispose (void)
{
  delete m_cidFactory;

  m_linkManager = 0;
  m_ssManager = 0;
  m_bsClassifier = 0;
  m_serviceFlowManager = 0;
  m_uplinkScheduler = 0;
  m_cidFactory = 0;
  m_ssManager = 0;
  m_uplinkScheduler = 0;
  m_scheduler = 0;

  WimaxNetDevice::DoDispose ();
}

void
BaseStationNetDevice::SetBsClassifier (Ptr<IpcsClassifier> bsc)
{
  m_bsClassifier = bsc;
}

Ptr<IpcsClassifier>
BaseStationNetDevice::GetBsClassifier (void) const
{
  return m_bsClassifier;
}

void
BaseStationNetDevice::SetInitialRangingInterval (Time initialRangInterval)
{
  m_initialRangInterval = initialRangInterval;
}

Time
BaseStationNetDevice::GetInitialRangingInterval (void) const
{
  return m_initialRangInterval;
}

void
BaseStationNetDevice::SetDcdInterval (Time dcdInterval)
{
  m_dcdInterval = dcdInterval;
}

Time
BaseStationNetDevice::GetDcdInterval (void) const
{
  return m_dcdInterval;
}

void
BaseStationNetDevice::SetUcdInterval (Time ucdInterval)
{
  m_ucdInterval = ucdInterval;
}

Time
BaseStationNetDevice::GetUcdInterval (void) const
{
  return m_ucdInterval;
}

void
BaseStationNetDevice::SetIntervalT8 (Time interval)
{
  m_intervalT8 = interval;
}

Time
BaseStationNetDevice::GetIntervalT8 (void) const
{
  return m_intervalT8;
}

void
BaseStationNetDevice::SetMaxRangingCorrectionRetries (uint8_t maxRangCorrectionRetries)
{
  m_maxRangCorrectionRetries = maxRangCorrectionRetries;
}

uint8_t
BaseStationNetDevice::GetMaxRangingCorrectionRetries (void) const
{
  return m_maxRangCorrectionRetries;
}

void
BaseStationNetDevice::SetMaxInvitedRangRetries (uint8_t maxInvitedRangRetries)
{
  m_maxInvitedRangRetries = maxInvitedRangRetries;
}

uint8_t
BaseStationNetDevice::GetMaxInvitedRangRetries (void) const
{
  return m_maxInvitedRangRetries;
}

void
BaseStationNetDevice::SetRangReqOppSize (uint8_t rangReqOppSize)
{
  m_rangReqOppSize = rangReqOppSize;
}

uint8_t
BaseStationNetDevice::GetRangReqOppSize (void) const
{
  return m_rangReqOppSize;
}

void
BaseStationNetDevice::SetBwReqOppSize (uint8_t bwReqOppSize)
{
  m_bwReqOppSize = bwReqOppSize;
}

uint8_t
BaseStationNetDevice::GetBwReqOppSize (void) const
{
  return m_bwReqOppSize;
}

void
BaseStationNetDevice::SetNrDlSymbols (uint32_t nrDlSymbols)
{
  m_nrDlSymbols = nrDlSymbols;
}

uint32_t
BaseStationNetDevice::GetNrDlSymbols (void) const
{
  return m_nrDlSymbols;
}

void
BaseStationNetDevice::SetNrUlSymbols (uint32_t nrUlSymbols)
{
  m_nrUlSymbols = nrUlSymbols;
}

uint32_t
BaseStationNetDevice::GetNrUlSymbols (void) const
{
  return m_nrUlSymbols;
}

uint32_t
BaseStationNetDevice::GetNrDcdSent (void) const
{
  return m_nrDcdSent;
}

uint32_t
BaseStationNetDevice::GetNrUcdSent (void) const
{
  return m_nrUcdSent;
}

Time
BaseStationNetDevice::GetDlSubframeStartTime (void) const
{
  return m_dlSubframeStartTime;
}

Time
BaseStationNetDevice::GetUlSubframeStartTime (void) const
{
  return m_ulSubframeStartTime;
}

uint8_t
BaseStationNetDevice::GetRangingOppNumber (void) const
{
  return m_rangingOppNumber;
}

Ptr<SSManager>
BaseStationNetDevice::GetSSManager (void) const
{
  return m_ssManager;
}

void
BaseStationNetDevice::SetSSManager (Ptr<SSManager> ssm)
{
  m_ssManager = ssm;
}

Ptr<BsServiceFlowManager>
BaseStationNetDevice::GetServiceFlowManager (void) const
{
  return m_serviceFlowManager;
}

void
BaseStationNetDevice::SetServiceFlowManager (Ptr<BsServiceFlowManager> sfm)
{
  m_serviceFlowManager = sfm;
}

Ptr<UplinkScheduler>
BaseStationNetDevice::GetUplinkScheduler (void) const
{
  return m_uplinkScheduler;
}

void
BaseStationNetDevice::SetUplinkScheduler (Ptr<UplinkScheduler> uls)
{
  m_uplinkScheduler = uls;
}

Ptr<BSLinkManager>
BaseStationNetDevice::GetLinkManager (void) const
{
  return m_linkManager;
}

void
BaseStationNetDevice::SetLinkManager (Ptr<BSLinkManager> lm)
{
  m_linkManager = lm;
}

void
BaseStationNetDevice::SetBSScheduler (Ptr<BSScheduler> bss)
{
  m_scheduler = bss;
}
Ptr<BSScheduler>
BaseStationNetDevice::GetBSScheduler (void) const
{
  return m_scheduler;
}

Time
BaseStationNetDevice::GetPsDuration (void) const
{
  return m_psDuration;
}

Time
BaseStationNetDevice::GetSymbolDuration (void) const
{
  return m_symbolDuration;
}

void
BaseStationNetDevice::Start (void)
{
  SetReceiveCallback ();
  GetConnectionManager ()->SetCidFactory (m_cidFactory);
  GetPhy ()->SetPhyParameters ();
  GetPhy ()->SetDataRates ();
  SetTtg (GetPhy ()->GetTtg ());
  SetRtg (GetPhy ()->GetRtg ());
  m_psDuration = GetPhy ()->GetPsDuration ();
  m_symbolDuration = GetPhy ()->GetSymbolDuration ();
  GetBandwidthManager ()->SetSubframeRatio ();

  CreateDefaultConnections ();
  GetPhy ()->SetSimplex (m_linkManager->SelectDlChannel ());
  Simulator::ScheduleNow (&BaseStationNetDevice::StartFrame, this);

  /* shall actually be 2 symbols = 1 (preamble) + 1 (bandwidth request header)*/
  m_bwReqOppSize = 6;
  m_uplinkScheduler->InitOnce ();
}

void
BaseStationNetDevice::Stop (void)
{
}

void
BaseStationNetDevice::StartFrame (void)
{
  //setting DL/UL subframe allocation for this frame
  uint32_t symbolsPerFrame = GetPhy ()->GetSymbolsPerFrame ();
  SetNrDlSymbols ((symbolsPerFrame / 2) - static_cast<uint32_t> (std::ceil (GetTtg ()*m_psDuration.GetSeconds ()/m_symbolDuration.GetSeconds ())));
  SetNrUlSymbols ((symbolsPerFrame / 2) - static_cast<uint32_t> (std::ceil (GetRtg ()*m_psDuration.GetSeconds ()/m_symbolDuration.GetSeconds ())));

  m_frameStartTime = Simulator::Now ();

  NS_LOG_INFO ("----------------------frame" << GetNrFrames () + 1 << "----------------------");

  StartDlSubFrame ();
}

void
BaseStationNetDevice::StartDlSubFrame (void)
{
  m_dlSubframeStartTime = Simulator::Now (); // same as m_frameStartTime

  NS_LOG_DEBUG ("DL frame started : " << m_frameStartTime.As (Time::S));

  SetNrFrames (GetNrFrames () + 1);
  SetState (BS_STATE_DL_SUB_FRAME);
  m_direction = DIRECTION_DOWNLINK;
  m_uplinkScheduler->Schedule ();
  CreateMapMessages ();
  m_scheduler->Schedule ();
  SendBursts ();
  Simulator::Schedule (Seconds (m_nrDlSymbols * m_symbolDuration.GetSeconds ()),
                       &BaseStationNetDevice::EndDlSubFrame,
                       this);
}

void
BaseStationNetDevice::EndDlSubFrame (void)
{
  m_nrDlFrames++;
  SetState (BS_STATE_TTG);
  Simulator::Schedule (Seconds (GetTtg () * m_psDuration.GetSeconds ()), &BaseStationNetDevice::StartUlSubFrame, this);
}

void
BaseStationNetDevice::StartUlSubFrame (void)
{
  m_ulSubframeStartTime = Simulator::Now ();

  NS_LOG_INFO ("UL frame started : " << m_ulSubframeStartTime.As (Time::S));

  SetState (BS_STATE_UL_SUB_FRAME);
  m_direction = DIRECTION_UPLINK;
  MarkUplinkAllocations ();
  Simulator::Schedule (Seconds (m_nrUlSymbols * m_symbolDuration.GetSeconds ()),
                       &BaseStationNetDevice::EndUlSubFrame,
                       this);
}

void
BaseStationNetDevice::EndUlSubFrame (void)
{
  m_nrUlFrames++;
  SetState (BS_STATE_RTG);
  Simulator::Schedule (Seconds (GetRtg () * m_psDuration.GetSeconds ()), &BaseStationNetDevice::EndFrame, this);
}

void
BaseStationNetDevice::EndFrame (void)
{
  StartFrame ();
}

bool
BaseStationNetDevice::DoSend (Ptr<Packet> packet,
                              const Mac48Address &source,
                              const Mac48Address &dest,
                              uint16_t protocolNumber)
{
  Ptr<PacketBurst> burst = Create<PacketBurst> ();
  ServiceFlow *serviceFlow = 0;

  NS_LOG_INFO ("BS (" << source << "):");
  NS_LOG_INFO ("\tSending packet...");
  NS_LOG_INFO ("\t\tDestination: " << dest);
  NS_LOG_INFO ("\t\tPaket Size:  " << packet->GetSize ());
  NS_LOG_INFO ("\t\tProtocol:    " << protocolNumber);


  if (protocolNumber == 2048)
    {
      serviceFlow = m_bsClassifier->Classify (packet, GetServiceFlowManager (), ServiceFlow::SF_DIRECTION_DOWN);
    }

  if (protocolNumber != 2048 || serviceFlow == 0)
    {
      serviceFlow = *GetServiceFlowManager ()->GetServiceFlows (ServiceFlow::SF_TYPE_ALL).begin ();
    }

  if (serviceFlow == 0)
    {
      NS_LOG_INFO ("No Service Flow!!");
      m_bsTxDropTrace (packet);
      return false;
    }
  if (serviceFlow->GetIsEnabled ())
    {
      if (!Enqueue (packet, MacHeaderType (), serviceFlow->GetConnection ()))
        {
          NS_LOG_INFO ("Enqueue Error!!");
          m_bsTxDropTrace (packet);
          return false;
        }
    }
  else
    {
      m_bsTxDropTrace (packet);
      NS_LOG_INFO ("Service Flow is not enabled");
      return false;
    }
  m_bsTxTrace (packet);

  return true;
}

bool
BaseStationNetDevice::Enqueue (Ptr<Packet> packet, const MacHeaderType &hdrType, Ptr<WimaxConnection> connection)
{
  NS_ASSERT_MSG (connection != 0,
                 "BS: Can not enqueue packet on the selected connection: the connection is not initialized");

  GenericMacHeader hdr;
  hdr.SetLen (packet->GetSize () + hdr.GetSerializedSize ());

  hdr.SetCid (connection->GetCid ());

  return connection->Enqueue (packet, hdrType, hdr);
}

void
BaseStationNetDevice::DoReceive (Ptr<Packet> packet)
{
  GenericMacHeader gnrcMacHdr;
  BandwidthRequestHeader bwRequestHdr;
  ManagementMessageType msgType;
  RngReq rngReq;
  Cid cid;
  uint8_t type = 0;
  GrantManagementSubheader grantMgmntSubhdr;
  Mac48Address source;
  LlcSnapHeader llc;
  Ptr<WimaxConnection> connection = 0;
  FragmentationSubheader fragSubhdr;
  bool fragmentation = false;  // it becomes true when there is a fragmentation subheader

  packet->RemoveHeader (gnrcMacHdr);
  if (gnrcMacHdr.GetHt () == MacHeaderType::HEADER_TYPE_GENERIC)
    {
      if (gnrcMacHdr.check_hcs () == false)
        {
          // The header is noisy
          m_bsRxDropTrace (packet);
          NS_LOG_INFO ("Header HCS ERROR");
          return;
        }

      cid = gnrcMacHdr.GetCid ();

      // checking for subheaders (only grant management subheader is implemented)
      type = gnrcMacHdr.GetType ();
      if (type)
        {
          // checking 1st bit, see Table 6
          if (type & 1)
            {
              packet->RemoveHeader (grantMgmntSubhdr);
            }
          // Check if there is a fragmentation Subheader
          uint8_t tmpType = type;
          if (((tmpType >> 2) & 1) == 1)
            {
              // a TRANSPORT packet with fragmentation subheader has been received!
              NS_LOG_INFO ("FRAG_DEBUG: DoReceive -> the packet is a fragment" << std::endl);
              fragmentation = true;
            }
        }

      if (cid.IsInitialRanging ()) // initial ranging connection
        {
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_RNG_REQ:
              {
                packet->RemoveHeader (rngReq);
                m_linkManager->ProcessRangingRequest (cid, rngReq);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_RNG_RSP:
              // from other base station, ignore
              break;
            default:
              NS_FATAL_ERROR ("Invalid message type");
            }
        }
      else if (m_cidFactory->IsBasic (cid)) // basic management connection
        {
          source = m_ssManager->GetMacAddress (cid);
          m_traceBSRx (packet, source, cid);
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_RNG_REQ:
              {
                packet->RemoveHeader (rngReq);
                m_linkManager->ProcessRangingRequest (cid, rngReq);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_RNG_RSP:
              // from other base station, ignore
              break;
            default:
              NS_FATAL_ERROR ("Invalid message type");
            }
        }
      else if (m_cidFactory->IsPrimary (cid)) // primary management connection
        {
          source = m_ssManager->GetMacAddress (cid);
          m_traceBSRx (packet, source, cid);
          packet->RemoveHeader (msgType);
          switch (msgType.GetType ())
            {
            case ManagementMessageType::MESSAGE_TYPE_REG_REQ:
              // not yet implemented
              break;
            case ManagementMessageType::MESSAGE_TYPE_REG_RSP:
              // from other base station, ignore
              break;
            case ManagementMessageType::MESSAGE_TYPE_DSA_REQ:
              {
                DsaReq dsaReq;
                packet->RemoveHeader (dsaReq);
                GetServiceFlowManager ()->AllocateServiceFlows (dsaReq, cid);
                break;
              }
            case ManagementMessageType::MESSAGE_TYPE_DSA_RSP:

              /*from other base station, as DSA initiated
               from BS is not supported, ignore*/
              break;
            case ManagementMessageType::MESSAGE_TYPE_DSA_ACK:
              {
                Simulator::Cancel (GetServiceFlowManager ()->GetDsaAckTimeoutEvent ());
                DsaAck dsaAck;
                packet->RemoveHeader (dsaAck);
                GetServiceFlowManager ()->ProcessDsaAck (dsaAck, cid);
                break;
              }
            default:
              NS_FATAL_ERROR ("Invalid message type");
            }
        }
      else if (cid.IsBroadcast ()) // broadcast connection
        {
          // from other base station, ignore
          // or perhaps data packet (using other protocol) for BS, handle later
          return;
        }
      else // transport connection
        {
          // If fragmentation is true, the packet is a fragment.
          Ptr<Packet> C_Packet = packet->Copy ();
          if (!fragmentation)
            {
              C_Packet->RemoveHeader (llc);
              source = m_ssManager->GetMacAddress (cid);
              m_bsRxTrace (packet);
              ForwardUp (packet->Copy (), source, Mac48Address ("ff:ff:ff:ff:ff:ff"));
            }
          else
            {
              NS_LOG_INFO ( "FRAG_DEBUG: BS DoReceive, the Packet is a fragment" << std::endl);
              packet->RemoveHeader (fragSubhdr);
              uint32_t fc = fragSubhdr.GetFc ();
              NS_LOG_INFO ("\t fragment size = " << packet->GetSize () << std::endl);
              if (fc == 2)
                {
                  // This is the latest fragment.
                  // Take the fragment queue, defragment a packet and send it to the upper layer
                  NS_LOG_INFO ("\t Received the latest fragment" << std::endl);
                  GetConnectionManager ()->GetConnection (cid)
                  ->FragmentEnqueue (packet);
                  WimaxConnection::FragmentsQueue fragmentsQueue = GetConnectionManager ()->
                    GetConnection (cid)->GetFragmentsQueue ();
                  Ptr<Packet> fullPacket = Create<Packet> ();

                  // DEFRAGMENTATION
                  NS_LOG_INFO ("\t BS PACKET DEFRAGMENTATION" << std::endl);
                  for (std::list<Ptr<const Packet> >::const_iterator iter = fragmentsQueue.begin ();
                       iter != fragmentsQueue.end (); ++iter)
                    {
                      // Create the whole Packet
                      fullPacket->AddAtEnd (*iter);
                    }
                  GetConnectionManager ()->GetConnection (cid)
                  ->ClearFragmentsQueue ();

                  NS_LOG_INFO ("\t fullPacket size = " << fullPacket->GetSize () << std::endl);
                  source = m_ssManager->GetMacAddress (cid);
                  m_bsRxTrace (fullPacket);
                  ForwardUp (fullPacket->Copy (), source, Mac48Address ("ff:ff:ff:ff:ff:ff"));
                }
              else
                {
                  // This is the first or middle fragment.
                  // Take the fragment queue, store the fragment into the queue
                  NS_LOG_INFO ("\t Received the first or the middle fragment" << std::endl);
                  GetConnectionManager ()->GetConnection (cid)
                  ->FragmentEnqueue (packet);
                }
            }
        }
    }
  else
    {
      // bandwidth request header
      packet->AddHeader (gnrcMacHdr);
      packet->RemoveHeader (bwRequestHdr);
      NS_ASSERT_MSG (bwRequestHdr.GetHt () == MacHeaderType::HEADER_TYPE_BANDWIDTH,
                     "A bandwidth request should be carried by a bandwidth header type");
      if (bwRequestHdr.check_hcs () == false)
        {
          // The header is noisy
          NS_LOG_INFO ("BS:Header HCS ERROR");
          return;
        }
      cid = bwRequestHdr.GetCid ();
      source = m_ssManager->GetMacAddress (cid);
      m_traceBSRx (packet, source, cid);
      GetBandwidthManager ()->ProcessBandwidthRequest (bwRequestHdr);
    }

}

void
BaseStationNetDevice::CreateMapMessages (void)
{
  Ptr<Packet> dlmap, ulmap;
  bool sendDcd = false, sendUcd = false, updateDcd = false, updateUcd = false;

  uint16_t currentNrSsRegistered = m_ssManager->GetNRegisteredSSs ();

  if (m_nrSsRegistered == currentNrSsRegistered)
    {
      m_uplinkScheduler->GetChannelDescriptorsToUpdate (updateDcd, updateUcd, sendDcd, sendUcd);
    }
  else
    {
      sendDcd = sendUcd = true;
    }

  m_nrSsRegistered = currentNrSsRegistered;

  /*either DCD and UCD must be created first because CCC is set during their
   creation, or CCC must be calculated first so that it could be set during
   creation of DL-MAP and UL-MAP and then set duirng creation of DCD and UCD*/

  if (sendDcd)
    {
      m_dcdConfigChangeCount += 1 % 256;
    }

  if (sendUcd)
    {
      m_ucdConfigChangeCount += 1 % 256;
    }

  dlmap = CreateDlMap ();
  Enqueue (dlmap, MacHeaderType (), GetBroadcastConnection ());
  m_nrDlMapSent++;

  ulmap = CreateUlMap ();
  Enqueue (ulmap, MacHeaderType (), GetBroadcastConnection ());
  m_nrUlMapSent++;

  CreateDescriptorMessages (sendDcd, sendUcd);
}

void
BaseStationNetDevice::CreateDescriptorMessages (bool sendDcd, bool sendUcd)
{
  Ptr<Packet> dcd, ucd;

  if (sendDcd)
    {
      dcd = CreateDcd ();
      Enqueue (dcd, MacHeaderType (), GetBroadcastConnection ());
      m_nrDcdSent++;
      m_framesSinceLastDcd = 0;
    }
  else
    {
      m_framesSinceLastDcd++;
    }

  if (sendUcd)
    {
      ucd = CreateUcd ();
      Enqueue (ucd, MacHeaderType (), GetBroadcastConnection ());
      m_nrUcdSent++;
      m_framesSinceLastUcd = 0;
    }
  else
    {
      m_framesSinceLastUcd++;
    }
}

/*
 Sends bursts in the downlink subframe. i.e., creates the downlink subframe. The first burst
 is broadcast burst with MAC management messages. The rest of the bursts contain data packets.
 */
void
BaseStationNetDevice::SendBursts (void)
{
  Time txTime = Seconds (0);
  std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > pair;
  WimaxPhy::ModulationType modulationType = WimaxPhy::MODULATION_TYPE_BPSK_12;
  std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > *downlinkBursts = m_scheduler->GetDownlinkBursts ();
  Ptr<PacketBurst> burst;
  OfdmDlMapIe *dlMapIe;
  Cid cid;

  while (downlinkBursts->size ())
    {
      pair = downlinkBursts->front ();
      burst = pair.second;
      dlMapIe = pair.first;
      cid = dlMapIe->GetCid ();
      uint8_t diuc = dlMapIe->GetDiuc ();

      if (cid != GetInitialRangingConnection ()->GetCid () && cid != GetBroadcastConnection ()->GetCid ())
        {
          if (m_serviceFlowManager->GetServiceFlow (cid) != 0)
            {
              modulationType = GetBurstProfileManager ()->GetModulationType (diuc, WimaxNetDevice::DIRECTION_DOWNLINK);
            }
          else
            {
              modulationType = GetBurstProfileManager ()->GetModulationType (diuc, WimaxNetDevice::DIRECTION_DOWNLINK);
            }
        }
      else
        {
          modulationType = WimaxPhy::MODULATION_TYPE_BPSK_12;
        }

      Simulator::Schedule (txTime, &WimaxNetDevice::ForwardDown, this, burst, modulationType);
      txTime += GetPhy ()->GetTransmissionTime (burst->GetSize (), modulationType);
      downlinkBursts->pop_front ();
      delete dlMapIe;
    }
}

Ptr<Packet>
BaseStationNetDevice::CreateDlMap (void)
{
  m_nrDlAllocations = 0;

  DlMap dlmap;
  dlmap.SetDcdCount (m_dcdConfigChangeCount);
  dlmap.SetBaseStationId (GetMacAddress ());

  std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > > *downlinkBursts = m_scheduler->GetDownlinkBursts ();

  for (std::list<std::pair<OfdmDlMapIe*, Ptr<PacketBurst> > >::iterator iter = downlinkBursts->begin (); iter
       != downlinkBursts->end (); ++iter)
    {
      iter->first->SetPreamblePresent (0);
      iter->first->SetStartTime (0);
      dlmap.AddDlMapElement (*(iter->first));
    }

  OfdmDlMapIe dlMapIeEnd;

  dlMapIeEnd.SetCid (Cid::InitialRanging ());
  dlMapIeEnd.SetDiuc (OfdmDlBurstProfile::DIUC_END_OF_MAP);
  dlMapIeEnd.SetPreamblePresent (0);
  dlMapIeEnd.SetStartTime (0);

  dlmap.AddDlMapElement (dlMapIeEnd);
  m_nrDlAllocations = downlinkBursts->size ();

  Ptr<Packet> p = Create<Packet> ();
  p->AddHeader (dlmap);
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_DL_MAP));
  return p;
}

Ptr<Packet>
BaseStationNetDevice::CreateDcd (void)
{
  Dcd dcd;
  OfdmDcdChannelEncodings chnlEncodings;

  chnlEncodings.SetBsEirp (0);
  chnlEncodings.SetEirxPIrMax (0);
  chnlEncodings.SetFrequency (GetPhy ()->GetFrequency ());
  chnlEncodings.SetChannelNr (0);
  chnlEncodings.SetTtg (GetTtg ());
  chnlEncodings.SetRtg (GetRtg ());
  chnlEncodings.SetBaseStationId (GetMacAddress ());
  chnlEncodings.SetFrameDurationCode (GetPhy ()->GetFrameDurationCode ());
  chnlEncodings.SetFrameNumber (GetNrFrames ());

  dcd.SetConfigurationChangeCount (m_dcdConfigChangeCount);
  dcd.SetChannelEncodings (chnlEncodings);

  SetDlBurstProfiles (&dcd);
  SetCurrentDcd (dcd);

  Ptr<Packet> p = Create<Packet> ();
  p->AddHeader (dcd);
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_DCD));
  return p;
}

Ptr<Packet>
BaseStationNetDevice::CreateUlMap (void)
{
  m_ulAllocationNumber = 0;
  m_rangingOppNumber = 0;
  m_nrUlAllocations = 0;

  UlMap ulmap;
  ulmap.SetUcdCount (m_ucdConfigChangeCount);
  ulmap.SetAllocationStartTime (m_uplinkScheduler->CalculateAllocationStartTime ());

  std::list<OfdmUlMapIe> uplinkAllocations = m_uplinkScheduler->GetUplinkAllocations ();

  for (std::list<OfdmUlMapIe>::iterator iter = uplinkAllocations.begin (); iter != uplinkAllocations.end (); ++iter)
    {
      ulmap.AddUlMapElement (*iter);
    }

  m_nrUlAllocations = uplinkAllocations.size ();

  Ptr<Packet> p = Create<Packet> ();
  p->AddHeader (ulmap);
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_UL_MAP));
  return p;
}

Ptr<Packet>
BaseStationNetDevice::CreateUcd (void)
{
  Ucd ucd;
  ucd.SetConfigurationChangeCount (m_ucdConfigChangeCount);
  ucd.SetRangingBackoffStart (3); // setting to 7. i.e., 2^3 = 8 -> 0-7
  ucd.SetRangingBackoffEnd (6); // setting to 63. i.e., 2^6 = 64 -> 0-63
  ucd.SetRequestBackoffStart (3);
  ucd.SetRequestBackoffEnd (6);

  OfdmUcdChannelEncodings chnlEncodings;

  chnlEncodings.SetBwReqOppSize (m_bwReqOppSize * GetPhy ()->GetPsPerSymbol ());
  chnlEncodings.SetRangReqOppSize (m_rangReqOppSize * GetPhy ()->GetPsPerSymbol ());

  chnlEncodings.SetFrequency (GetPhy ()->GetFrequency ());
  chnlEncodings.SetSbchnlReqRegionFullParams (0);
  chnlEncodings.SetSbchnlFocContCodes (0);

  ucd.SetChannelEncodings (chnlEncodings);

  SetUlBurstProfiles (&ucd);
  SetCurrentUcd (ucd);

  Ptr<Packet> p = Create<Packet> ();
  p->AddHeader (ucd);
  p->AddHeader (ManagementMessageType (ManagementMessageType::MESSAGE_TYPE_UCD));
  return p;
}

void
BaseStationNetDevice::SetDlBurstProfiles (Dcd *dcd)
{
  for (int i = 0; i < GetBurstProfileManager ()->GetNrBurstProfilesToDefine (); ++i)
    {
      OfdmDlBurstProfile brstProfile;
      brstProfile.SetType (0);
      brstProfile.SetLength (0);
      brstProfile.SetDiuc (i + 1); // DIUC will be between 1-11, see Table 237
      brstProfile.SetFecCodeType (i);
      dcd->AddDlBurstProfile (brstProfile);
    }
}

void
BaseStationNetDevice::SetUlBurstProfiles (Ucd *ucd)
{
  for (int i = 0; i < GetBurstProfileManager ()->GetNrBurstProfilesToDefine (); ++i)
    {
      OfdmUlBurstProfile brstProfile;
      brstProfile.SetType (0);
      brstProfile.SetLength (0);
      // UIUC will be between 5-12, see Table 246. UIUC 1 (initial ranging) is not included
      brstProfile.SetUiuc (i + 5);
      brstProfile.SetFecCodeType (i);

      ucd->AddUlBurstProfile (brstProfile);
    }
}

Ptr<WimaxConnection>
BaseStationNetDevice::GetConnection (Cid cid)
{
  Ptr<WimaxConnection> connection = 0;
  if (cid.IsInitialRanging ())
    {
      return GetInitialRangingConnection ();
    }
  else if (cid.IsBroadcast ())
    {
      connection = GetBroadcastConnection ();
    }
  else
    {
      connection = GetConnectionManager ()->GetConnection (cid);
    }

  NS_ASSERT_MSG (connection != 0, "BS: Invalid connection=0");
  return connection;
}

void
BaseStationNetDevice::MarkUplinkAllocations (void)
{
  uint16_t symbolsToAllocation = 0;
  std::list<OfdmUlMapIe> uplinkAllocations = m_uplinkScheduler->GetUplinkAllocations ();
  for (std::list<OfdmUlMapIe>::iterator iter = uplinkAllocations.begin (); iter != uplinkAllocations.end (); ++iter)
    {
      OfdmUlMapIe uplinkAllocation = *iter;

      if (uplinkAllocation.GetUiuc () == OfdmUlBurstProfile::UIUC_END_OF_MAP)
        {
          break;
        }

      symbolsToAllocation = uplinkAllocation.GetStartTime ();
      MarkUplinkAllocationStart (symbolsToAllocation * m_symbolDuration);
      MarkUplinkAllocationEnd ((symbolsToAllocation + uplinkAllocation.GetDuration ())
                                        * m_symbolDuration, uplinkAllocation.GetCid (), uplinkAllocation.GetUiuc ());
    }
}

void
BaseStationNetDevice::MarkUplinkAllocationStart (Time allocationStartTime)
{
  Simulator::Schedule (allocationStartTime, &BaseStationNetDevice::UplinkAllocationStart, this);
}

void
BaseStationNetDevice::MarkUplinkAllocationEnd (Time allocationEndTime, Cid cid, uint8_t uiuc)
{
  Simulator::Schedule (allocationEndTime, &BaseStationNetDevice::UplinkAllocationEnd, this, cid, uiuc);
}

void
BaseStationNetDevice::UplinkAllocationStart (void)
{
  m_ulAllocationNumber++;

  NS_LOG_DEBUG ("--UL allocation " << (uint32_t) m_ulAllocationNumber << " started : "
                                   << Simulator::Now ().As (Time::S));

}

void
BaseStationNetDevice::UplinkAllocationEnd (Cid cid, uint8_t uiuc)
{
  NS_LOG_DEBUG ("--UL allocation " << (uint32_t) m_ulAllocationNumber << " ended : " << Simulator::Now ().As (Time::S));

  if (m_cidFactory->IsBasic (cid))
    {
      m_linkManager->VerifyInvitedRanging (cid, uiuc);
    }
}

void
BaseStationNetDevice::MarkRangingOppStart (Time rangingOppStartTime)
{
  Simulator::Schedule (rangingOppStartTime, &BaseStationNetDevice::RangingOppStart, this);
}

void
BaseStationNetDevice::RangingOppStart (void)
{
  m_rangingOppNumber++;

  NS_LOG_DEBUG ("Ranging TO " << (uint32_t) m_rangingOppNumber << ": " << Simulator::Now ().As (Time::S));
}

} // namespace ns3
