/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009, 2010 MIRKO BANCHI
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
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/log.h"
#include "ns3/simulator.h"
#include "block-ack-manager.h"
#include "wifi-utils.h"
#include "ctrl-headers.h"
#include "mgt-headers.h"
#include "wifi-mac-queue.h"
#include "qos-utils.h"
#include "wifi-tx-vector.h"
#include <optional>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BlockAckManager");

Bar::Bar ()
{
  NS_LOG_FUNCTION (this);
}

Bar::Bar (Ptr<const WifiMacQueueItem> bar, uint8_t tid, bool skipIfNoDataQueued)
  : bar (bar),
    tid (tid),
    skipIfNoDataQueued (skipIfNoDataQueued)
{
  NS_LOG_FUNCTION (this << *bar << +tid << skipIfNoDataQueued);
}

NS_OBJECT_ENSURE_REGISTERED (BlockAckManager);

TypeId
BlockAckManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BlockAckManager")
    .SetParent<Object> ()
    .SetGroupName ("Wifi")
    .AddConstructor<BlockAckManager> ()
    .AddTraceSource ("AgreementState",
                     "The state of the ADDBA handshake",
                     MakeTraceSourceAccessor (&BlockAckManager::m_agreementState),
                     "ns3::BlockAckManager::AgreementStateTracedCallback")
  ;
  return tid;
}

BlockAckManager::BlockAckManager ()
{
  NS_LOG_FUNCTION (this);
}

BlockAckManager::~BlockAckManager ()
{
  NS_LOG_FUNCTION (this);
}

void
BlockAckManager::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  m_agreements.clear ();
  m_bars.clear ();
  m_queue = nullptr;
}

bool
BlockAckManager::ExistsAgreement (Mac48Address recipient, uint8_t tid) const
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  return (m_agreements.find (std::make_pair (recipient, tid)) != m_agreements.end ());
}

bool
BlockAckManager::ExistsAgreementInState (Mac48Address recipient, uint8_t tid,
                                         OriginatorBlockAckAgreement::State state) const
{
  AgreementsCI it;
  it = m_agreements.find (std::make_pair (recipient, tid));
  if (it != m_agreements.end ())
    {
      switch (state)
        {
        case OriginatorBlockAckAgreement::ESTABLISHED:
          return it->second.first.IsEstablished ();
        case OriginatorBlockAckAgreement::PENDING:
          return it->second.first.IsPending ();
        case OriginatorBlockAckAgreement::REJECTED:
          return it->second.first.IsRejected ();
        case OriginatorBlockAckAgreement::NO_REPLY:
          return it->second.first.IsNoReply ();
        case OriginatorBlockAckAgreement::RESET:
          return it->second.first.IsReset ();
        default:
          NS_FATAL_ERROR ("Invalid state for block ack agreement");
        }
    }
  return false;
}

void
BlockAckManager::CreateAgreement (const MgtAddBaRequestHeader *reqHdr, Mac48Address recipient, bool htSupported)
{
  NS_LOG_FUNCTION (this << reqHdr << recipient << htSupported);
  std::pair<Mac48Address, uint8_t> key (recipient, reqHdr->GetTid ());
  OriginatorBlockAckAgreement agreement (recipient, reqHdr->GetTid ());
  agreement.SetStartingSequence (reqHdr->GetStartingSequence ());
  /* For now we assume that originator doesn't use this field. Use of this field
     is mandatory only for recipient */
  agreement.SetBufferSize (reqHdr->GetBufferSize());
  agreement.SetTimeout (reqHdr->GetTimeout ());
  agreement.SetAmsduSupport (reqHdr->IsAmsduSupported ());
  agreement.SetHtSupported (htSupported);
  if (reqHdr->IsImmediateBlockAck ())
    {
      agreement.SetImmediateBlockAck ();
    }
  else
    {
      agreement.SetDelayedBlockAck ();
    }
  uint8_t tid = reqHdr->GetTid ();
  m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::PENDING);
  agreement.SetState (OriginatorBlockAckAgreement::PENDING);
  PacketQueue queue;
  std::pair<OriginatorBlockAckAgreement, PacketQueue> value (agreement, queue);
  if (ExistsAgreement (recipient, tid))
    {
      // Delete agreement if it exists and in RESET state
      NS_ASSERT (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::RESET));
      m_agreements.erase (key);
    }
  m_agreements.insert (std::make_pair (key, value));
  m_blockPackets (recipient, reqHdr->GetTid ());
}

void
BlockAckManager::DestroyAgreement (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  if (it != m_agreements.end ())
    {
      m_agreements.erase (it);
      //remove scheduled BAR
      for (std::list<Bar>::const_iterator i = m_bars.begin (); i != m_bars.end (); )
        {
          if (i->bar->GetHeader ().GetAddr1 () == recipient && i->tid == tid)
            {
              i = m_bars.erase (i);
            }
          else
            {
              i++;
            }
        }
    }
}

void
BlockAckManager::UpdateAgreement (const MgtAddBaResponseHeader *respHdr, Mac48Address recipient,
                                  uint16_t startingSeq)
{
  NS_LOG_FUNCTION (this << respHdr << recipient << startingSeq);
  uint8_t tid = respHdr->GetTid ();
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  if (it != m_agreements.end ())
    {
      OriginatorBlockAckAgreement& agreement = it->second.first;
      agreement.SetBufferSize (respHdr->GetBufferSize ());
      agreement.SetTimeout (respHdr->GetTimeout ());
      agreement.SetAmsduSupport (respHdr->IsAmsduSupported ());
      agreement.SetStartingSequence (startingSeq);
      agreement.InitTxWindow ();
      if (respHdr->IsImmediateBlockAck ())
        {
          agreement.SetImmediateBlockAck ();
        }
      else
        {
          agreement.SetDelayedBlockAck ();
        }
      if (!it->second.first.IsEstablished ())
      {
        m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED);
      }
      agreement.SetState (OriginatorBlockAckAgreement::ESTABLISHED);
      if (agreement.GetTimeout () != 0)
        {
          Time timeout = MicroSeconds (1024 * agreement.GetTimeout ());
          agreement.m_inactivityEvent = Simulator::Schedule (timeout,
                                                             &BlockAckManager::InactivityTimeout,
                                                             this,
                                                             recipient, tid);
        }
    }
  m_unblockPackets (recipient, tid);
}

void
BlockAckManager::StorePacket (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  NS_ASSERT (mpdu->GetHeader ().IsQosData ());

  uint8_t tid = mpdu->GetHeader ().GetQosTid ();
  Mac48Address recipient = mpdu->GetHeader ().GetAddr1 ();

  AgreementsI agreementIt = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (agreementIt != m_agreements.end ());

  uint16_t mpduDist = agreementIt->second.first.GetDistance (mpdu->GetHeader ().GetSequenceNumber ());

  if (mpduDist >= SEQNO_SPACE_HALF_SIZE)
    {
      NS_LOG_DEBUG ("Got an old packet. Do nothing");
      return;
    }

  // store the packet and keep the list sorted in increasing order of sequence number
  // with respect to the starting sequence number
  auto it = agreementIt->second.second.rbegin ();
  while (it != agreementIt->second.second.rend ())
    {
      if (mpdu->GetHeader ().GetSequenceControl () == (*it)->GetHeader ().GetSequenceControl ())
        {
          NS_LOG_DEBUG ("Packet already in the queue of the BA agreement");
          return;
        }

      uint16_t dist = agreementIt->second.first.GetDistance ((*it)->GetHeader ().GetSequenceNumber ());

      if (mpduDist > dist ||
          (mpduDist == dist && mpdu->GetHeader ().GetFragmentNumber () > (*it)->GetHeader ().GetFragmentNumber ()))
        {
          break;
        }

      it++;
    }
  agreementIt->second.second.insert (it.base (), mpdu);
  agreementIt->second.first.NotifyTransmittedMpdu (mpdu);
  mpdu->SetInFlight ();
}

Ptr<const WifiMacQueueItem>
BlockAckManager::GetBar (bool remove, uint8_t tid, Mac48Address address)
{
  Time now = Simulator::Now ();
  Ptr<const WifiMacQueueItem> bar;
  // remove all expired MPDUs from the head of the MAC queue, so that
  // BlockAckRequest frames (if needed) are scheduled
  m_queue->IsEmpty ();

  auto nextBar = m_bars.begin ();

  while (nextBar != m_bars.end ())
    {
      Mac48Address recipient = nextBar->bar->GetHeader ().GetAddr1 ();

      if (address != Mac48Address::GetBroadcast () && tid != 8
            && (!nextBar->bar->GetHeader ().IsBlockAckReq ()
                  || address != recipient || tid != nextBar->tid))
        {
          // we can only return a BAR addressed to the given station and for the given TID
          nextBar++;
          continue;
        }
      if (nextBar->bar->GetHeader ().IsBlockAckReq ())
        {
          AgreementsI it = m_agreements.find (std::make_pair (recipient, nextBar->tid));
          if (it == m_agreements.end ())
            {
              // BA agreement was torn down; remove this BAR and continue
              nextBar = m_bars.erase (nextBar);
              continue;
            }
          if (nextBar->skipIfNoDataQueued
              && m_queue->PeekByTidAndAddress (nextBar->tid, recipient) == nullptr)
            {
              // skip this BAR as there is no data queued
              nextBar++;
              continue;
            }
          // remove expired outstanding MPDUs and update the starting sequence number
          for (auto mpduIt = it->second.second.begin (); mpduIt != it->second.second.end (); )
            {
              if (!(*mpduIt)->IsQueued ())
                {
                  // the MPDU is no longer in the EDCA queue
                  mpduIt = it->second.second.erase (mpduIt);
                  continue;
                }

              if ((*mpduIt)->GetTimeStamp () + m_queue->GetMaxDelay () <= now)
                {
                  // MPDU expired
                  it->second.first.NotifyDiscardedMpdu (*mpduIt);
                  // Remove from the EDCA queue and fire the Expired trace source, but the
                  // consequent call to NotifyDiscardedMpdu does nothing (in particular,
                  // does not schedule a BAR) because we have advanced the transmit window
                  // and hence this MPDU became an old packet
                  m_queue->TtlExceeded (*mpduIt, now);
                  mpduIt = it->second.second.erase (mpduIt);
                }
              else
                {
                  // MPDUs are typically in increasing order of remaining lifetime
                  break;
                }
            }
          // update BAR if the starting sequence number changed
          CtrlBAckRequestHeader reqHdr;
          nextBar->bar->GetPacket ()->PeekHeader (reqHdr);
          if (reqHdr.GetStartingSequence () != it->second.first.GetStartingSequence ())
            {
              reqHdr.SetStartingSequence (it->second.first.GetStartingSequence ());
              Ptr<Packet> packet = Create<Packet> ();
              packet->AddHeader (reqHdr);
              nextBar->bar = Create<const WifiMacQueueItem> (packet, nextBar->bar->GetHeader ());
            }
        }

      bar = nextBar->bar;
      if (remove)
        {
          m_bars.erase (nextBar);
        }
      break;
    }
  return bar;
}

uint32_t
BlockAckManager::GetNBufferedPackets (Mac48Address recipient, uint8_t tid) const
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  if (it == m_agreements.end ())
    {
      return 0;
    }
  return it->second.second.size ();
}

void
BlockAckManager::SetBlockAckThreshold (uint8_t nPackets)
{
  NS_LOG_FUNCTION (this << +nPackets);
  m_blockAckThreshold = nPackets;
}

BlockAckManager::PacketQueueI
BlockAckManager::HandleInFlightMpdu (PacketQueueI mpduIt, MpduStatus status,
                                     const AgreementsI& it, const Time& now)
{
  NS_LOG_FUNCTION (this << **mpduIt << +static_cast<uint8_t> (status));

  if (!(*mpduIt)->IsQueued ())
    {
      // MPDU is not in the EDCA queue (e.g., its lifetime expired and it was
      // removed by another method), remove from the queue of in flight MPDUs
      NS_LOG_DEBUG ("MPDU is not stored in the EDCA queue, drop MPDU");
      return it->second.second.erase (mpduIt);
    }

  if (status == ACKNOWLEDGED)
    {
      // the MPDU has to be dequeued from the EDCA queue
      m_queue->DequeueIfQueued (*mpduIt);
      return it->second.second.erase (mpduIt);
    }

  WifiMacHeader& hdr = (*mpduIt)->GetHeader ();

  NS_ASSERT (hdr.GetAddr1 () == it->first.first);
  NS_ASSERT (hdr.IsQosData () && hdr.GetQosTid () == it->first.second);

  if (it->second.first.GetDistance (hdr.GetSequenceNumber ()) >= SEQNO_SPACE_HALF_SIZE)
    {
      NS_LOG_DEBUG ("Old packet. Remove from the EDCA queue, too");
      if (!m_droppedOldMpduCallback.IsNull ())
        {
          m_droppedOldMpduCallback (*mpduIt);
        }
      m_queue->Remove (*mpduIt, false);
      return it->second.second.erase (mpduIt);
    }

  std::optional<PacketQueueI> prevIt;
  if (mpduIt != it->second.second.begin ())
    {
      prevIt = std::prev (mpduIt);
    }

  if (m_queue->TtlExceeded (*mpduIt, now))
    {
      // WifiMacQueue::TtlExceeded() has removed the MPDU from the EDCA queue
      // and fired the Expired trace source, which called NotifyDiscardedMpdu,
      // which removed this MPDU (and possibly others) from the in flight queue as well
      NS_LOG_DEBUG ("MSDU lifetime expired, drop MPDU");
      return (prevIt.has_value () ? std::next (prevIt.value ()) : it->second.second.begin ());
    }

  if (status == STAY_INFLIGHT)
    {
      // the MPDU has to stay in flight, do nothing
      return ++mpduIt;
    }

  NS_ASSERT (status == TO_RETRANSMIT);
  (*mpduIt)->GetHeader ().SetRetry ();
  (*mpduIt)->ResetInFlight ();  // no longer in flight; will be if retransmitted

  return it->second.second.erase (mpduIt);
}

void
BlockAckManager::NotifyGotAck (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  NS_ASSERT (mpdu->GetHeader ().IsQosData ());

  Mac48Address recipient = mpdu->GetHeader ().GetAddr1 ();
  uint8_t tid = mpdu->GetHeader ().GetQosTid ();
  NS_ASSERT (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED));

  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());

  it->second.first.NotifyAckedMpdu (mpdu);

  // remove the acknowledged frame from the queue of outstanding packets
  for (auto queueIt = it->second.second.begin (); queueIt != it->second.second.end (); ++queueIt)
    {
      if ((*queueIt)->GetHeader ().GetSequenceNumber () == mpdu->GetHeader ().GetSequenceNumber ())
        {
          HandleInFlightMpdu (queueIt, ACKNOWLEDGED, it, Simulator::Now ());
          break;
        }
    }
}

void
BlockAckManager::NotifyMissedAck (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);
  NS_ASSERT (mpdu->GetHeader ().IsQosData ());

  Mac48Address recipient = mpdu->GetHeader ().GetAddr1 ();
  uint8_t tid = mpdu->GetHeader ().GetQosTid ();
  NS_ASSERT (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED));

  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());

  // remove the frame from the queue of outstanding packets (it will be re-inserted
  // if retransmitted)
  for (auto queueIt = it->second.second.begin (); queueIt != it->second.second.end (); ++queueIt)
    {
      if ((*queueIt)->GetHeader ().GetSequenceNumber () == mpdu->GetHeader ().GetSequenceNumber ())
        {
          HandleInFlightMpdu (queueIt, TO_RETRANSMIT, it, Simulator::Now ());
          break;
        }
    }
}

std::pair<uint16_t,uint16_t>
BlockAckManager::NotifyGotBlockAck (const CtrlBAckResponseHeader& blockAck, Mac48Address recipient,
                                    const std::set<uint8_t>& tids, size_t index)
{
  NS_LOG_FUNCTION (this << blockAck << recipient << index);
  uint16_t nSuccessfulMpdus = 0;
  uint16_t nFailedMpdus = 0;

  NS_ABORT_MSG_IF (blockAck.IsBasic (), "Basic Block Ack is not supported");
  NS_ABORT_MSG_IF (blockAck.IsMultiTid (), "Multi-TID Block Ack is not supported");

  uint8_t tid = blockAck.GetTidInfo (index);
  // If this is a Multi-STA Block Ack with All-ack context (TID equal to 14),
  // use the TID passed by the caller.
  if (tid == 14)
    {
      NS_ASSERT (blockAck.GetAckType (index) && tids.size () == 1);
      tid = *tids.begin ();
    }
  if (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED))
    {
      AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));

      if (it->second.first.m_inactivityEvent.IsRunning ())
        {
          /* Upon reception of a BlockAck frame, the inactivity timer at the
              originator must be reset.
              For more details see section 11.5.3 in IEEE802.11e standard */
          it->second.first.m_inactivityEvent.Cancel ();
          Time timeout = MicroSeconds (1024 * it->second.first.GetTimeout ());
          it->second.first.m_inactivityEvent = Simulator::Schedule (timeout,
                                                                    &BlockAckManager::InactivityTimeout,
                                                                    this,
                                                                    recipient, tid);
        }

      NS_ASSERT (blockAck.IsCompressed () || blockAck.IsExtendedCompressed () || blockAck.IsMultiSta ());
      Time now = Simulator::Now ();

      for (auto queueIt = it->second.second.begin (); queueIt != it->second.second.end (); )
        {
          uint16_t currentSeq = (*queueIt)->GetHeader ().GetSequenceNumber ();
          if (blockAck.IsPacketReceived (currentSeq, index))
            {
              it->second.first.NotifyAckedMpdu (*queueIt);
              nSuccessfulMpdus++;
              if (!m_txOkCallback.IsNull ())
                {
                  m_txOkCallback (*queueIt);
                }
              queueIt = HandleInFlightMpdu (queueIt, ACKNOWLEDGED, it, now);
            }
          else
            {
              nFailedMpdus++;
              if (!m_txFailedCallback.IsNull ())
                {
                  m_txFailedCallback (*queueIt);
                }
              queueIt = HandleInFlightMpdu (queueIt, TO_RETRANSMIT, it, now);
            }
        }
    }
  return {nSuccessfulMpdus, nFailedMpdus};
}

void
BlockAckManager::NotifyMissedBlockAck (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  if (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED))
    {
      AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
      Time now = Simulator::Now ();

      // remove all packets from the queue of outstanding packets (they will be
      // re-inserted if retransmitted)
      for (auto mpduIt = it->second.second.begin (); mpduIt != it->second.second.end (); )
        {
          mpduIt = HandleInFlightMpdu (mpduIt, TO_RETRANSMIT, it, now);
        }
    }
}

void
BlockAckManager::NotifyDiscardedMpdu (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  if (!mpdu->GetHeader ().IsQosData ())
    {
      NS_LOG_DEBUG ("Not a QoS Data frame");
      return;
    }

  if (!mpdu->GetHeader ().IsRetry () && !mpdu->IsInFlight ())
    {
      NS_LOG_DEBUG ("This frame has never been transmitted");
      return;
    }

  Mac48Address recipient = mpdu->GetHeader ().GetAddr1 ();
  uint8_t tid = mpdu->GetHeader ().GetQosTid ();
  if (!ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED))
    {
      NS_LOG_DEBUG ("No established Block Ack agreement");
      return;
    }

  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  uint16_t currStartingSeq = it->second.first.GetStartingSequence ();
  if (QosUtilsIsOldPacket (currStartingSeq, mpdu->GetHeader ().GetSequenceNumber ()))
    {
      NS_LOG_DEBUG ("Discarded an old frame");
      return;
    }

  // actually advance the transmit window
  it->second.first.NotifyDiscardedMpdu (mpdu);

  // remove old MPDUs from the EDCA queue and from the in flight queue
  // (including the given MPDU which became old after advancing the transmit window)
  for (auto mpduIt = it->second.second.begin (); mpduIt != it->second.second.end (); )
    {
      if (it->second.first.GetDistance ((*mpduIt)->GetHeader ().GetSequenceNumber ()) >= SEQNO_SPACE_HALF_SIZE)
        {
          NS_LOG_DEBUG ("Dropping old MPDU: " << **mpduIt);
          m_queue->DequeueIfQueued (*mpduIt);
          if (!m_droppedOldMpduCallback.IsNull ())
            {
              m_droppedOldMpduCallback (*mpduIt);
            }
          mpduIt = it->second.second.erase (mpduIt);
        }
      else
        {
          break; // MPDUs are in increasing order of sequence number in the in flight queue
        }
    }

  // schedule a BlockAckRequest
  NS_LOG_DEBUG ("Schedule a Block Ack Request for agreement (" << recipient << ", " << +tid << ")");
  Ptr<Packet> bar = Create<Packet> ();
  bar->AddHeader (GetBlockAckReqHeader (recipient, tid));

  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_BACKREQ);
  hdr.SetAddr1 (recipient);
  hdr.SetAddr2 (mpdu->GetHeader ().GetAddr2 ());
  hdr.SetAddr3 (mpdu->GetHeader ().GetAddr3 ());
  hdr.SetDsNotTo ();
  hdr.SetDsNotFrom ();
  hdr.SetNoRetry ();
  hdr.SetNoMoreFragments ();

  ScheduleBar (Create<const WifiMacQueueItem> (bar, hdr));
}

CtrlBAckRequestHeader
BlockAckManager::GetBlockAckReqHeader (Mac48Address recipient, uint8_t tid) const
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());

  CtrlBAckRequestHeader reqHdr;
  reqHdr.SetType ((*it).second.first.GetBlockAckReqType ());
  reqHdr.SetTidInfo (tid);
  reqHdr.SetStartingSequence ((*it).second.first.GetStartingSequence ());
  return reqHdr;
}

void
BlockAckManager::ScheduleBar (Ptr<const WifiMacQueueItem> bar, bool skipIfNoDataQueued)
{
  NS_LOG_FUNCTION (this << *bar);
  NS_ASSERT (bar->GetHeader ().IsBlockAckReq () || bar->GetHeader ().IsTrigger ());

  uint8_t tid = 0;
  if (bar->GetHeader ().IsBlockAckReq ())
    {
      CtrlBAckRequestHeader reqHdr;
      bar->GetPacket ()->PeekHeader (reqHdr);
      tid = reqHdr.GetTidInfo ();
    }
#ifdef NS3_BUILD_PROFILE_DEBUG
  else
    {
      CtrlTriggerHeader triggerHdr;
      bar->GetPacket ()->PeekHeader (triggerHdr);
      NS_ASSERT (triggerHdr.IsMuBar ());
    }
#endif
  Bar request (bar, tid, skipIfNoDataQueued);

  // if a BAR for the given agreement is present, replace it with the new one
  std::list<Bar>::const_iterator i = m_bars.end ();

  if (bar->GetHeader ().IsBlockAckReq ())
    {
      for (i = m_bars.begin (); i != m_bars.end (); i++)
        {
          if (i->bar->GetHeader ().IsBlockAckReq ()
              && i->bar->GetHeader ().GetAddr1 () == bar->GetHeader ().GetAddr1 () && i->tid == tid)
            {
              i = m_bars.erase (i);
              break;
            }
        }
    }

  if (bar->GetHeader ().IsRetry ())
    {
      m_bars.push_front (request);
    }
  else
    {
      m_bars.insert (i, request);
    }
}

void
BlockAckManager::InactivityTimeout (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  m_blockAckInactivityTimeout (recipient, tid, true);
}

void
BlockAckManager::NotifyAgreementEstablished (Mac48Address recipient, uint8_t tid, uint16_t startingSeq)
{
  NS_LOG_FUNCTION (this << recipient << +tid << startingSeq);
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());
  if (!it->second.first.IsEstablished ())
  {
    m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED);
  }
  it->second.first.SetState (OriginatorBlockAckAgreement::ESTABLISHED);
  it->second.first.SetStartingSequence (startingSeq);
}

void
BlockAckManager::NotifyAgreementRejected (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());
  if (!it->second.first.IsRejected ())
    {
      m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::REJECTED);
    }
  it->second.first.SetState (OriginatorBlockAckAgreement::REJECTED);
}

void
BlockAckManager::NotifyAgreementNoReply (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());
  if (!it->second.first.IsNoReply ())
    {
      m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::NO_REPLY);
    }
  it->second.first.SetState (OriginatorBlockAckAgreement::NO_REPLY);
  m_unblockPackets (recipient, tid);
}

void
BlockAckManager::NotifyAgreementReset (Mac48Address recipient, uint8_t tid)
{
  NS_LOG_FUNCTION (this << recipient << +tid);
  AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ASSERT (it != m_agreements.end ());
  if (!it->second.first.IsReset ())
    {
      m_agreementState (Simulator::Now (), recipient, tid, OriginatorBlockAckAgreement::RESET);
    }
  it->second.first.SetState (OriginatorBlockAckAgreement::RESET);
}

void
BlockAckManager::SetQueue (const Ptr<WifiMacQueue> queue)
{
  NS_LOG_FUNCTION (this << queue);
  m_queue = queue;
}

bool
BlockAckManager::SwitchToBlockAckIfNeeded (Mac48Address recipient, uint8_t tid, uint16_t startingSeq)
{
  NS_LOG_FUNCTION (this << recipient << +tid << startingSeq);
  NS_ASSERT (!ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::PENDING));
  if (!ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::REJECTED) && ExistsAgreement (recipient, tid))
    {
      uint32_t packets = m_queue->GetNPacketsByTidAndAddress (tid, recipient) +
        GetNBufferedPackets (recipient, tid);
      if (packets >= m_blockAckThreshold)
        {
          NotifyAgreementEstablished (recipient, tid, startingSeq);
          return true;
        }
    }
  return false;
}

bool BlockAckManager::NeedBarRetransmission (uint8_t tid, Mac48Address recipient)
{
  if (ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::ESTABLISHED))
    {
      AgreementsI it = m_agreements.find (std::make_pair (recipient, tid));
      NS_ASSERT (it != m_agreements.end ());

      Time now = Simulator::Now ();

      // A BAR needs to be retransmitted if there is at least a non-expired in flight MPDU
      for (auto mpduIt = it->second.second.begin (); mpduIt != it->second.second.end (); )
        {
          // remove MPDU if old or with expired lifetime
          mpduIt = HandleInFlightMpdu (mpduIt, STAY_INFLIGHT, it, now);

          if (mpduIt != it->second.second.begin ())
            {
              // the MPDU has not been removed
              return true;
            }
        }
    }

  // If the inactivity timer has expired, QosTxop::SendDelbaFrame has been called and
  // has destroyed the agreement, hence we get here and correctly return false
  return false;
}

void
BlockAckManager::SetBlockAckInactivityCallback (Callback<void, Mac48Address, uint8_t, bool> callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_blockAckInactivityTimeout = callback;
}

void
BlockAckManager::SetBlockDestinationCallback (Callback<void, Mac48Address, uint8_t> callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_blockPackets = callback;
}

void
BlockAckManager::SetUnblockDestinationCallback (Callback<void, Mac48Address, uint8_t> callback)
{
  NS_LOG_FUNCTION (this << &callback);
  m_unblockPackets = callback;
}

void
BlockAckManager::SetTxOkCallback (TxOk callback)
{
  m_txOkCallback = callback;
}

void
BlockAckManager::SetTxFailedCallback (TxFailed callback)
{
  m_txFailedCallback = callback;
}

void
BlockAckManager::SetDroppedOldMpduCallback (DroppedOldMpdu callback)
{
  m_droppedOldMpduCallback = callback;
}

uint16_t
BlockAckManager::GetRecipientBufferSize (Mac48Address recipient, uint8_t tid) const
{
  uint16_t size = 0;
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  if (it != m_agreements.end ())
    {
      size = it->second.first.GetBufferSize ();
    }
  return size;
}

BlockAckReqType
BlockAckManager::GetBlockAckReqType (Mac48Address recipient, uint8_t tid) const
{
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ABORT_MSG_IF (it == m_agreements.end (), "No established Block Ack agreement");
  return it->second.first.GetBlockAckReqType ();
}

BlockAckType
BlockAckManager::GetBlockAckType (Mac48Address recipient, uint8_t tid) const
{
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  NS_ABORT_MSG_IF (it == m_agreements.end (), "No established Block Ack agreement");
  return it->second.first.GetBlockAckType ();
}

uint16_t
BlockAckManager::GetOriginatorStartingSequence (Mac48Address recipient, uint8_t tid) const
{
  uint16_t seqNum = 0;
  AgreementsCI it = m_agreements.find (std::make_pair (recipient, tid));
  if (it != m_agreements.end ())
    {
      seqNum = it->second.first.GetStartingSequence ();
    }
  return seqNum;
}

} //namespace ns3
