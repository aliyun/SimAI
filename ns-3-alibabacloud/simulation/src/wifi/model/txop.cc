/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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
 */

#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/simulator.h"
#include "ns3/random-variable-stream.h"
#include "ns3/socket.h"
#include "txop.h"
#include "channel-access-manager.h"
#include "wifi-mac.h"
#include "wifi-mac-queue.h"
#include "mac-tx-middle.h"
#include "wifi-mac-trailer.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT if (m_mac != 0) { std::clog << "[mac=" << m_mac->GetAddress () << "] "; }

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Txop");

NS_OBJECT_ENSURE_REGISTERED (Txop);

TypeId
Txop::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Txop")
    .SetParent<ns3::Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<Txop> ()
    .AddAttribute ("MinCw", "The minimum value of the contention window.",
                   UintegerValue (15),
                   MakeUintegerAccessor (&Txop::SetMinCw,
                                         &Txop::GetMinCw),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MaxCw", "The maximum value of the contention window.",
                   UintegerValue (1023),
                   MakeUintegerAccessor (&Txop::SetMaxCw,
                                         &Txop::GetMaxCw),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Aifsn", "The AIFSN: the default value conforms to non-QOS.",
                   UintegerValue (2),
                   MakeUintegerAccessor (&Txop::SetAifsn,
                                         &Txop::GetAifsn),
                   MakeUintegerChecker<uint8_t> ())
    .AddAttribute ("TxopLimit", "The TXOP limit: the default value conforms to non-QoS.",
                   TimeValue (MilliSeconds (0)),
                   MakeTimeAccessor (&Txop::SetTxopLimit,
                                     &Txop::GetTxopLimit),
                   MakeTimeChecker ())
    .AddAttribute ("Queue", "The WifiMacQueue object",
                   PointerValue (),
                   MakePointerAccessor (&Txop::GetWifiMacQueue),
                   MakePointerChecker<WifiMacQueue> ())
    .AddTraceSource ("BackoffTrace",
                     "Trace source for backoff values",
                     MakeTraceSourceAccessor (&Txop::m_backoffTrace),
                     "ns3::TracedCallback::Uint32Callback")
    .AddTraceSource ("CwTrace",
                     "Trace source for contention window values",
                     MakeTraceSourceAccessor (&Txop::m_cwTrace),
                     "ns3::TracedValueCallback::Uint32")
  ;
  return tid;
}

Txop::Txop ()
  : Txop (CreateObject<WifiMacQueue> (AC_BE_NQOS))
{
}

Txop::Txop (Ptr<WifiMacQueue> queue)
  : m_channelAccessManager (0),
    m_queue (queue),
    m_cwMin (0),
    m_cwMax (0),
    m_cw (0),
    m_backoff (0),
    m_access (NOT_REQUESTED),
    m_backoffSlots (0),
    m_backoffStart (Seconds (0.0))
{
  NS_LOG_FUNCTION (this);
  m_rng = CreateObject<UniformRandomVariable> ();
}

Txop::~Txop ()
{
  NS_LOG_FUNCTION (this);
}

void
Txop::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_queue = 0;
  m_mac = 0;
  m_rng = 0;
  m_txMiddle = 0;
  m_channelAccessManager = 0;
}

void
Txop::SetChannelAccessManager (const Ptr<ChannelAccessManager> manager)
{
  NS_LOG_FUNCTION (this << manager);
  m_channelAccessManager = manager;
  m_channelAccessManager->Add (this);
}

void Txop::SetTxMiddle (const Ptr<MacTxMiddle> txMiddle)
{
  NS_LOG_FUNCTION (this);
  m_txMiddle = txMiddle;
}

void
Txop::SetWifiMac (const Ptr<WifiMac> mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_mac = mac;
}

void
Txop::SetDroppedMpduCallback (DroppedMpdu callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_droppedMpduCallback = callback;
  m_queue->TraceConnectWithoutContext ("DropBeforeEnqueue",
                                       m_droppedMpduCallback.Bind (WIFI_MAC_DROP_FAILED_ENQUEUE));
  m_queue->TraceConnectWithoutContext ("Expired",
                                       m_droppedMpduCallback.Bind (WIFI_MAC_DROP_EXPIRED_LIFETIME));
}

Ptr<WifiMacQueue >
Txop::GetWifiMacQueue () const
{
  NS_LOG_FUNCTION (this);
  return m_queue;
}

void
Txop::SetMinCw (uint32_t minCw)
{
  NS_LOG_FUNCTION (this << minCw);
  bool changed = (m_cwMin != minCw);
  m_cwMin = minCw;
  if (changed == true)
    {
      ResetCw ();
    }
}

void
Txop::SetMaxCw (uint32_t maxCw)
{
  NS_LOG_FUNCTION (this << maxCw);
  bool changed = (m_cwMax != maxCw);
  m_cwMax = maxCw;
  if (changed == true)
    {
      ResetCw ();
    }
}

uint32_t
Txop::GetCw (void) const
{
  return m_cw;
}

void
Txop::ResetCw (void)
{
  NS_LOG_FUNCTION (this);
  m_cw = GetMinCw ();
  m_cwTrace = m_cw;
}

void
Txop::UpdateFailedCw (void)
{
  NS_LOG_FUNCTION (this);
  //see 802.11-2012, section 9.19.2.5
  m_cw = std::min ( 2 * (m_cw + 1) - 1, GetMaxCw ());
  // if the MU EDCA timer is running, CW cannot be less than MU CW min
  m_cw = std::max (m_cw, GetMinCw ());
  m_cwTrace = m_cw;
}

uint32_t
Txop::GetBackoffSlots (void) const
{
  return m_backoffSlots;
}

Time
Txop::GetBackoffStart (void) const
{
  return m_backoffStart;
}

void
Txop::UpdateBackoffSlotsNow (uint32_t nSlots, Time backoffUpdateBound)
{
  NS_LOG_FUNCTION (this << nSlots << backoffUpdateBound);
  m_backoffSlots -= nSlots;
  m_backoffStart = backoffUpdateBound;
  NS_LOG_DEBUG ("update slots=" << nSlots << " slots, backoff=" << m_backoffSlots);
}

void
Txop::StartBackoffNow (uint32_t nSlots)
{
  NS_LOG_FUNCTION (this << nSlots);
  if (m_backoffSlots != 0)
    {
      NS_LOG_DEBUG ("reset backoff from " << m_backoffSlots << " to " << nSlots << " slots");
    }
  else
    {
      NS_LOG_DEBUG ("start backoff=" << nSlots << " slots");
    }
  m_backoffSlots = nSlots;
  m_backoffStart = Simulator::Now ();
}

void
Txop::SetAifsn (uint8_t aifsn)
{
  NS_LOG_FUNCTION (this << +aifsn);
  m_aifsn = aifsn;
}

void
Txop::SetTxopLimit (Time txopLimit)
{
  NS_LOG_FUNCTION (this << txopLimit);
  NS_ASSERT_MSG ((txopLimit.GetMicroSeconds () % 32 == 0), "The TXOP limit must be expressed in multiple of 32 microseconds!");
  m_txopLimit = txopLimit;
}

uint32_t
Txop::GetMinCw (void) const
{
  return m_cwMin;
}

uint32_t
Txop::GetMaxCw (void) const
{
  return m_cwMax;
}

uint8_t
Txop::GetAifsn (void) const
{
  return m_aifsn;
}

Time
Txop::GetTxopLimit (void) const
{
  return m_txopLimit;
}

bool
Txop::HasFramesToTransmit (void)
{
  bool ret = (!m_queue->IsEmpty ());
  NS_LOG_FUNCTION (this << ret);
  return ret;
}

void
Txop::Queue (Ptr<Packet> packet, const WifiMacHeader &hdr)
{
  NS_LOG_FUNCTION (this << packet << &hdr);
  // remove the priority tag attached, if any
  SocketPriorityTag priorityTag;
  packet->RemovePacketTag (priorityTag);
  if (m_channelAccessManager->NeedBackoffUponAccess (this))
    {
      GenerateBackoff ();
    }
  m_queue->Enqueue (Create<WifiMacQueueItem> (packet, hdr));
  StartAccessIfNeeded ();
}

int64_t
Txop::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_rng->SetStream (stream);
  return 1;
}

void
Txop::StartAccessIfNeeded (void)
{
  NS_LOG_FUNCTION (this);
  if (HasFramesToTransmit () && m_access == NOT_REQUESTED)
    {
      m_channelAccessManager->RequestAccess (this);
    }
}

void
Txop::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  ResetCw ();
  GenerateBackoff ();
}

Txop::ChannelAccessStatus
Txop::GetAccessStatus (void) const
{
  return m_access;
}

void
Txop::NotifyAccessRequested (void)
{
  NS_LOG_FUNCTION (this);
  m_access = REQUESTED;
}

void
Txop::NotifyChannelAccessed (Time txopDuration)
{
  NS_LOG_FUNCTION (this << txopDuration);
  m_access = GRANTED;
}

void
Txop::NotifyChannelReleased (void)
{
  NS_LOG_FUNCTION (this);
  m_access = NOT_REQUESTED;
  GenerateBackoff ();
  if (HasFramesToTransmit ())
    {
      Simulator::ScheduleNow (&Txop::RequestAccess, this);
    }
}

void
Txop::RequestAccess (void)
{
  if (m_access == NOT_REQUESTED)
    {
      m_channelAccessManager->RequestAccess (this);
    }
}

void
Txop::GenerateBackoff (void)
{
  NS_LOG_FUNCTION (this);
  m_backoff = m_rng->GetInteger (0, GetCw ());
  m_backoffTrace (m_backoff);
  StartBackoffNow (m_backoff);
}

void
Txop::NotifySleep (void)
{
  NS_LOG_FUNCTION (this);
}

void
Txop::NotifyOff (void)
{
  NS_LOG_FUNCTION (this);
  m_queue->Flush ();
}

void
Txop::NotifyWakeUp (void)
{
  NS_LOG_FUNCTION (this);
  StartAccessIfNeeded ();
}

void
Txop::NotifyOn (void)
{
  NS_LOG_FUNCTION (this);
  StartAccessIfNeeded ();
}

bool
Txop::IsQosTxop () const
{
  return false;
}

} //namespace ns3
