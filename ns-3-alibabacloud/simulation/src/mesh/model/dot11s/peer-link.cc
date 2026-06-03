/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Aleksey Kovalenko <kovalenko@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */

#include "peer-management-protocol-mac.h"
#include "ns3/peer-link.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/traced-value.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Dot11sPeerManagementProtocol");
  
namespace dot11s {

NS_OBJECT_ENSURE_REGISTERED ( PeerLink);

TypeId
PeerLink::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dot11s::PeerLink")
    .SetParent<Object> ()
    .SetGroupName ("Mesh")
    .AddConstructor<PeerLink> ()
    .AddAttribute ( "RetryTimeout",
                    "Retry timeout",
                    TimeValue (TimeValue (MicroSeconds (40 * 1024))),
                    MakeTimeAccessor (
                      &PeerLink::m_dot11MeshRetryTimeout),
                    MakeTimeChecker ()
                    )
    .AddAttribute ( "HoldingTimeout",
                    "Holding timeout",
                    TimeValue (TimeValue (MicroSeconds (40 * 1024))),
                    MakeTimeAccessor (
                      &PeerLink::m_dot11MeshHoldingTimeout),
                    MakeTimeChecker ()
                    )
    .AddAttribute ( "ConfirmTimeout",
                    "Confirm timeout",
                    TimeValue (TimeValue (MicroSeconds (40 * 1024))),
                    MakeTimeAccessor (
                      &PeerLink::m_dot11MeshConfirmTimeout),
                    MakeTimeChecker ()
                    )
    .AddAttribute ( "MaxRetries",
                    "Maximum number of retries",
                    UintegerValue (4),
                    MakeUintegerAccessor (
                      &PeerLink::m_dot11MeshMaxRetries),
                    MakeUintegerChecker<uint16_t> ()
                    )
    .AddAttribute ( "MaxBeaconLoss",
                    "Maximum number of lost beacons before link will be closed",
                    UintegerValue (2),
                    MakeUintegerAccessor (
                      &PeerLink::m_maxBeaconLoss),
                    MakeUintegerChecker<uint16_t> (1)
                    )
    .AddAttribute ( "MaxPacketFailure",
                    "Maximum number of failed packets before link will be closed",
                    UintegerValue (2),
                    MakeUintegerAccessor (
                      &PeerLink::m_maxPacketFail),
                    MakeUintegerChecker<uint16_t> (1)
                    )
  ;
  return tid;
}

const char* const
PeerLink::PeerStateNames[6] = { "IDLE", "OPN_SNT", "CNF_RCVD", "OPN_RCVD", "ESTAB", "HOLDING" };

//-----------------------------------------------------------------------------
// PeerLink public interface
//-----------------------------------------------------------------------------
PeerLink::PeerLink () :
  m_peerAddress (Mac48Address::GetBroadcast ()),
  m_peerMeshPointAddress (Mac48Address::GetBroadcast ()),
  m_localLinkId (0),
  m_peerLinkId (0),
  m_assocId (0),
  m_peerAssocId (0),
  m_lastBeacon (Seconds (0)),
  m_beaconInterval (Seconds (0)),
  m_packetFail (0),
  m_state (IDLE),
  m_retryCounter (0),
  m_maxPacketFail (3)
{
  NS_LOG_FUNCTION (this);
}
PeerLink::~PeerLink ()
{
}
void
PeerLink::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_retryTimer.Cancel ();
  m_holdingTimer.Cancel ();
  m_confirmTimer.Cancel ();
  m_beaconLossTimer.Cancel ();
  m_beaconTiming.ClearTimingElement ();
}
void
PeerLink::SetPeerAddress (Mac48Address macaddr)
{
  m_peerAddress = macaddr;
}
void
PeerLink::SetPeerMeshPointAddress (Mac48Address macaddr)
{
  m_peerMeshPointAddress = macaddr;
}
void
PeerLink::SetInterface (uint32_t interface)
{
  m_interface = interface;
}
void
PeerLink::SetLocalLinkId (uint16_t id)
{
  m_localLinkId = id;
}
void
PeerLink::SetLocalAid (uint16_t aid)
{
  m_assocId = aid;
}
void
PeerLink::SetBeaconInformation (Time lastBeacon, Time beaconInterval)
{
  m_lastBeacon = lastBeacon;
  m_beaconInterval = beaconInterval;
  m_beaconLossTimer.Cancel ();
  Time delay = Seconds (beaconInterval.GetSeconds () * m_maxBeaconLoss);
  NS_ASSERT (delay.GetMicroSeconds () != 0);
  m_beaconLossTimer = Simulator::Schedule (delay, &PeerLink::BeaconLoss, this);
}
void
PeerLink::MLMESetSignalStatusCallback (PeerLink::SignalStatusCallback cb)
{
  m_linkStatusCallback = cb;
}
void
PeerLink::BeaconLoss ()
{
  NS_LOG_FUNCTION (this);
  StateMachine (CNCL);
}
void
PeerLink::TransmissionSuccess ()
{
  m_packetFail = 0;
}
void
PeerLink::TransmissionFailure ()
{
  NS_LOG_FUNCTION (this);
  m_packetFail++;
  if (m_packetFail == m_maxPacketFail)
    {
      NS_LOG_DEBUG ("TransmissionFailure:: CNCL");
      StateMachine (CNCL);
      m_packetFail = 0;
    }
}

void
PeerLink::SetBeaconTimingElement (IeBeaconTiming beaconTiming)
{
  m_beaconTiming = beaconTiming;
}
Mac48Address
PeerLink::GetPeerAddress () const
{
  return m_peerAddress;
}
uint16_t
PeerLink::GetLocalAid () const
{
  return m_assocId;
}
uint16_t
PeerLink::GetPeerAid () const
{
  return m_peerAssocId;
}

Time
PeerLink::GetLastBeacon () const
{
  return m_lastBeacon;
}
Time
PeerLink::GetBeaconInterval () const
{
  return m_beaconInterval;
}
IeBeaconTiming
PeerLink::GetBeaconTimingElement () const
{
  return m_beaconTiming;
}
void
PeerLink::MLMECancelPeerLink (PmpReasonCode reason)
{
  StateMachine (CNCL, reason);
}
void
PeerLink::MLMEActivePeerLinkOpen ()
{
  StateMachine (ACTOPN);
}
void
PeerLink::MLMEPeeringRequestReject ()
{
  StateMachine (REQ_RJCT, REASON11S_PEERING_CANCELLED);
}
void
PeerLink::Close (uint16_t localLinkId, uint16_t peerLinkId, PmpReasonCode reason)
{
  NS_LOG_FUNCTION (this << localLinkId << peerLinkId << reason);
  if (peerLinkId != 0 && m_localLinkId != peerLinkId)
    {
      return;
    }
  if (m_peerLinkId == 0)
    {
      m_peerLinkId = localLinkId;
    }
  else
    {
      if (m_peerLinkId != localLinkId)
        {
          return;
        }
    }
  StateMachine (CLS_ACPT, reason);
}
void
PeerLink::OpenAccept (uint16_t localLinkId, IeConfiguration conf, Mac48Address peerMp)
{
  NS_LOG_FUNCTION (this << localLinkId << peerMp);
  m_peerLinkId = localLinkId;
  m_configuration = conf;
  if (m_peerMeshPointAddress != Mac48Address::GetBroadcast ())
    {
      NS_ASSERT (m_peerMeshPointAddress == peerMp);
    }
  else
    {
      m_peerMeshPointAddress = peerMp;
    }
  StateMachine (OPN_ACPT);
}
void
PeerLink::OpenReject (uint16_t localLinkId, IeConfiguration conf, Mac48Address peerMp, PmpReasonCode reason)
{
  NS_LOG_FUNCTION (this << localLinkId << peerMp << reason);
  if (m_peerLinkId == 0)
    {
      m_peerLinkId = localLinkId;
    }
  m_configuration = conf;
  if (m_peerMeshPointAddress != Mac48Address::GetBroadcast ())
    {
      NS_ASSERT (m_peerMeshPointAddress == peerMp);
    }
  else
    {
      m_peerMeshPointAddress = peerMp;
    }
  StateMachine (OPN_RJCT, reason);
}
void
PeerLink::ConfirmAccept (uint16_t localLinkId, uint16_t peerLinkId, uint16_t peerAid, IeConfiguration conf,
                         Mac48Address peerMp)
{
  NS_LOG_FUNCTION (this << localLinkId << peerLinkId << peerAid << peerMp);
  if (m_localLinkId != peerLinkId)
    {
      return;
    }
  if (m_peerLinkId == 0)
    {
      m_peerLinkId = localLinkId;
    }
  else
    {
      if (m_peerLinkId != localLinkId)
        {
          return;
        }
    }
  m_configuration = conf;
  m_peerAssocId = peerAid;
  if (m_peerMeshPointAddress != Mac48Address::GetBroadcast ())
    {
      NS_ASSERT (m_peerMeshPointAddress == peerMp);
    }
  else
    {
      m_peerMeshPointAddress = peerMp;
    }
  StateMachine (CNF_ACPT);
}
void
PeerLink::ConfirmReject (uint16_t localLinkId, uint16_t peerLinkId, IeConfiguration conf,
                         Mac48Address peerMp, PmpReasonCode reason)
{
  NS_LOG_FUNCTION (this << localLinkId << peerLinkId << peerMp << reason);
  if (m_localLinkId != peerLinkId)
    {
      return;
    }
  if (m_peerLinkId == 0)
    {
      m_peerLinkId = localLinkId;
    }
  else
    {
      if (m_peerLinkId != localLinkId)
        {
          return;
        }
    }
  m_configuration = conf;
  if (m_peerMeshPointAddress != Mac48Address::GetBroadcast ())
    {
      NS_ASSERT (m_peerMeshPointAddress == peerMp);
    }
  m_peerMeshPointAddress = peerMp;
  StateMachine (CNF_RJCT, reason);
}
bool
PeerLink::LinkIsEstab () const
{
  return (m_state == ESTAB);
}
bool
PeerLink::LinkIsIdle () const
{
  return (m_state == IDLE);
}
void
PeerLink::SetMacPlugin (Ptr<PeerManagementProtocolMac> plugin)
{
  m_macPlugin = plugin;
}
//-----------------------------------------------------------------------------
// Private
//-----------------------------------------------------------------------------
void
PeerLink::StateMachine (PeerEvent event, PmpReasonCode reasoncode)
{
  switch (m_state)
    {
    case IDLE:
      switch (event)
        {
        case CNCL:
        case CLS_ACPT:
          m_state = IDLE;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, IDLE, IDLE);
          break;
        case REQ_RJCT:
          SendPeerLinkClose (reasoncode);
          break;
        case ACTOPN:
          m_state = OPN_SNT;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, IDLE, OPN_SNT);
          SendPeerLinkOpen ();
          SetRetryTimer ();
          break;
        case OPN_ACPT:
          m_state = OPN_RCVD;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, IDLE, OPN_RCVD);
          SendPeerLinkConfirm ();
          SendPeerLinkOpen ();
          SetRetryTimer ();
          break;
        default:
          //11B.5.3.4 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    case OPN_SNT:
      switch (event)
        {
        case TOR1:
          SendPeerLinkOpen ();
          m_retryCounter++;
          SetRetryTimer ();
          break;
        case CNF_ACPT:
          m_state = CNF_RCVD;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, CNF_RCVD);
          ClearRetryTimer ();
          SetConfirmTimer ();
          break;
        case OPN_ACPT:
          m_state = OPN_RCVD;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, OPN_RCVD);
          SendPeerLinkConfirm ();
          break;
        case CLS_ACPT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_MESH_CLOSE_RCVD);
          SetHoldingTimer ();
          break;
        case OPN_RJCT:
        case CNF_RJCT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (reasoncode);
          SetHoldingTimer ();
          break;
        case TOR2:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_MESH_MAX_RETRIES);
          SetHoldingTimer ();
          break;
        case CNCL:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_SNT, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_PEERING_CANCELLED);
          SetHoldingTimer ();
          break;
        default:
          //11B.5.3.5 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    case CNF_RCVD:
      switch (event)
        {
        case CNF_ACPT:
          break;
        case OPN_ACPT:
          m_state = ESTAB;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, CNF_RCVD, ESTAB);
          ClearConfirmTimer ();
          SendPeerLinkConfirm ();
          NS_ASSERT (m_peerMeshPointAddress != Mac48Address::GetBroadcast ());
          break;
        case CLS_ACPT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, CNF_RCVD, HOLDING);
          ClearConfirmTimer ();
          SendPeerLinkClose (REASON11S_MESH_CLOSE_RCVD);
          SetHoldingTimer ();
          break;
        case CNF_RJCT:
        case OPN_RJCT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, CNF_RCVD, HOLDING);
          ClearConfirmTimer ();
          SendPeerLinkClose (reasoncode);
          SetHoldingTimer ();
          break;
        case CNCL:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, CNF_RCVD, HOLDING);
          ClearConfirmTimer ();
          SendPeerLinkClose (REASON11S_PEERING_CANCELLED);
          SetHoldingTimer ();
          break;
        case TOC:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, CNF_RCVD, HOLDING);
          SendPeerLinkClose (REASON11S_MESH_CONFIRM_TIMEOUT);
          SetHoldingTimer ();
          break;
        default:
          //11B.5.3.6 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    case OPN_RCVD:
      switch (event)
        {
        case TOR1:
          SendPeerLinkOpen ();
          m_retryCounter++;
          SetRetryTimer ();
          break;
        case CNF_ACPT:
          m_state = ESTAB;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_RCVD, ESTAB);
          ClearRetryTimer ();
          NS_ASSERT (m_peerMeshPointAddress != Mac48Address::GetBroadcast ());
          break;
        case CLS_ACPT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_RCVD, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_MESH_CLOSE_RCVD);
          SetHoldingTimer ();
          break;
        case OPN_RJCT:
        case CNF_RJCT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_RCVD, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (reasoncode);
          SetHoldingTimer ();
          break;
        case TOR2:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_RCVD, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_MESH_MAX_RETRIES);
          SetHoldingTimer ();
          break;
        case CNCL:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, OPN_RCVD, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (REASON11S_PEERING_CANCELLED);
          SetHoldingTimer ();
          break;
        default:
          //11B.5.3.7 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    case ESTAB:
      switch (event)
        {
        case OPN_ACPT:
          SendPeerLinkConfirm ();
          break;
        case CLS_ACPT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, ESTAB, HOLDING);
          SendPeerLinkClose (REASON11S_MESH_CLOSE_RCVD);
          SetHoldingTimer ();
          break;
        case OPN_RJCT:
        case CNF_RJCT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, ESTAB, HOLDING);
          ClearRetryTimer ();
          SendPeerLinkClose (reasoncode);
          SetHoldingTimer ();
          break;
        case CNCL:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, ESTAB, HOLDING);
          SendPeerLinkClose (REASON11S_PEERING_CANCELLED);
          SetHoldingTimer ();
          break;
        default:
          //11B.5.3.8 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    case HOLDING:
      switch (event)
        {
        case CLS_ACPT:
          ClearHoldingTimer ();
          // fall through:
        case TOH:
          m_state = IDLE;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, HOLDING, IDLE);
          break;
        case OPN_ACPT:
        case CNF_ACPT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, HOLDING, HOLDING);
          // reason not spec in D2.0
          SendPeerLinkClose (REASON11S_PEERING_CANCELLED);
          break;
        case OPN_RJCT:
        case CNF_RJCT:
          m_state = HOLDING;
          m_linkStatusCallback (m_interface, m_peerAddress, m_peerMeshPointAddress, HOLDING, HOLDING);
          SendPeerLinkClose (reasoncode);
          break;
        default:
          //11B.5.3.9 of 802.11s Draft D3.0
          //All other events shall be ignored in this state
          break;
        }
      break;
    }
}
void
PeerLink::ClearRetryTimer ()
{
  m_retryTimer.Cancel ();
}
void
PeerLink::ClearConfirmTimer ()
{
  m_confirmTimer.Cancel ();
}
void
PeerLink::ClearHoldingTimer ()
{
  m_holdingTimer.Cancel ();
}
void
PeerLink::SendPeerLinkClose (PmpReasonCode reasoncode)
{
  IePeerManagement peerElement;
  peerElement.SetPeerClose (m_localLinkId, m_peerLinkId, reasoncode);
  m_macPlugin->SendPeerLinkManagementFrame (m_peerAddress, m_peerMeshPointAddress, m_assocId, peerElement,
                                            m_configuration);
}
void
PeerLink::SendPeerLinkOpen ()
{
  IePeerManagement peerElement;
  peerElement.SetPeerOpen (m_localLinkId);
  NS_ASSERT (m_macPlugin != 0);
  m_macPlugin->SendPeerLinkManagementFrame (m_peerAddress, m_peerMeshPointAddress, m_assocId, peerElement,
                                            m_configuration);
}
void
PeerLink::SendPeerLinkConfirm ()
{
  IePeerManagement peerElement;
  peerElement.SetPeerConfirm (m_localLinkId, m_peerLinkId);
  m_macPlugin->SendPeerLinkManagementFrame (m_peerAddress, m_peerMeshPointAddress, m_assocId, peerElement,
                                            m_configuration);
}
void
PeerLink::SetHoldingTimer ()
{
  NS_ASSERT (m_dot11MeshHoldingTimeout.GetMicroSeconds () != 0);
  m_holdingTimer = Simulator::Schedule (m_dot11MeshHoldingTimeout, &PeerLink::HoldingTimeout, this);
}
void
PeerLink::HoldingTimeout ()
{
  NS_LOG_FUNCTION (this);
  StateMachine (TOH);
}
void
PeerLink::SetRetryTimer ()
{
  NS_ASSERT (m_dot11MeshRetryTimeout.GetMicroSeconds () != 0);
  m_retryTimer = Simulator::Schedule (m_dot11MeshRetryTimeout, &PeerLink::RetryTimeout, this);
}
void
PeerLink::RetryTimeout ()
{
  NS_LOG_FUNCTION (this);
  if (m_retryCounter < m_dot11MeshMaxRetries)
    {
      NS_LOG_LOGIC ("Retry timeout TOR1");
      StateMachine (TOR1);
    }
  else
    {
      NS_LOG_LOGIC ("Retry timeout TOR2");
      StateMachine (TOR2);
    }
}
void
PeerLink::SetConfirmTimer ()
{
  NS_ASSERT (m_dot11MeshConfirmTimeout.GetMicroSeconds () != 0);
  m_confirmTimer = Simulator::Schedule (m_dot11MeshConfirmTimeout, &PeerLink::ConfirmTimeout, this);
}
void
PeerLink::ConfirmTimeout ()
{
  StateMachine (TOC);
}
void
PeerLink::Report (std::ostream & os) const
{
  if (m_state != ESTAB)
    {
      return;
    }
  os << "<PeerLink" << std::endl <<
  "localAddress=\"" << m_macPlugin->GetAddress () << "\"" << std::endl <<
  "peerInterfaceAddress=\"" << m_peerAddress << "\"" << std::endl <<
  "peerMeshPointAddress=\"" << m_peerMeshPointAddress << "\"" << std::endl <<
  "metric=\"" << m_macPlugin->GetLinkMetric (m_peerAddress) << "\"" << std::endl <<
  "lastBeacon=\"" << m_lastBeacon.GetSeconds () << "\"" << std::endl <<
  "localLinkId=\"" << m_localLinkId << "\"" << std::endl <<
  "peerLinkId=\"" << m_peerLinkId << "\"" << std::endl <<
  "assocId=\"" << m_assocId << "\"" << std::endl <<
  "/>" << std::endl;
}
} // namespace dot11s
} // namespace ns3

