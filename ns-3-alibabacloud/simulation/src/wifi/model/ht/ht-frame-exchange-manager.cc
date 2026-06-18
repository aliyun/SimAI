/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "ns3/abort.h"
#include "ht-frame-exchange-manager.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/mgt-headers.h"
#include "ns3/recipient-block-ack-agreement.h"
#include "ns3/wifi-utils.h"
#include "ns3/snr-tag.h"
#include "ns3/ctrl-headers.h"
#include <array>
#include <optional>

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT std::clog << "[mac=" << m_self << "] "

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HtFrameExchangeManager");

NS_OBJECT_ENSURE_REGISTERED (HtFrameExchangeManager);

TypeId
HtFrameExchangeManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::HtFrameExchangeManager")
    .SetParent<QosFrameExchangeManager> ()
    .AddConstructor<HtFrameExchangeManager> ()
    .SetGroupName ("Wifi")
  ;
  return tid;
}

HtFrameExchangeManager::HtFrameExchangeManager ()
{
  NS_LOG_FUNCTION (this);
  m_msduAggregator = CreateObject<MsduAggregator> ();
  m_mpduAggregator = CreateObject<MpduAggregator> ();
}

HtFrameExchangeManager::~HtFrameExchangeManager ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
HtFrameExchangeManager::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  m_agreements.clear ();
  m_msduAggregator = 0;
  m_mpduAggregator = 0;
  m_psdu = 0;
  m_txParams.Clear ();
  QosFrameExchangeManager::DoDispose ();
}

void
HtFrameExchangeManager::SetWifiMac (const Ptr<WifiMac> mac)
{
  m_msduAggregator->SetWifiMac (mac);
  m_mpduAggregator->SetWifiMac (mac);
  QosFrameExchangeManager::SetWifiMac (mac);
}

Ptr<MsduAggregator>
HtFrameExchangeManager::GetMsduAggregator (void) const
{
  return m_msduAggregator;
}

Ptr<MpduAggregator>
HtFrameExchangeManager::GetMpduAggregator (void) const
{
  return m_mpduAggregator;
}

Ptr<BlockAckManager>
HtFrameExchangeManager::GetBaManager (uint8_t tid) const
{
  return m_mac->GetQosTxop (tid)->GetBaManager ();
}

bool
HtFrameExchangeManager::NeedSetupBlockAck (Mac48Address recipient, uint8_t tid)
{
  Ptr<QosTxop> qosTxop = m_mac->GetQosTxop (tid);
  bool establish;

  if (!m_mac->GetWifiRemoteStationManager ()->GetHtSupported (recipient))
    {
      establish = false;
    }
  else if (qosTxop->GetBaManager ()->ExistsAgreement (recipient, tid)
           && !qosTxop->GetBaManager ()->ExistsAgreementInState (recipient, tid, OriginatorBlockAckAgreement::RESET))
    {
      establish = false;
    }
  else
    {
      uint32_t packets = qosTxop->GetWifiMacQueue ()->GetNPacketsByTidAndAddress (tid, recipient);
      establish = ((qosTxop->GetBlockAckThreshold () > 0 && packets >= qosTxop->GetBlockAckThreshold ())
                   || (m_mpduAggregator->GetMaxAmpduSize (recipient, tid, WIFI_MOD_CLASS_HT) > 0 && packets > 1)
                   || m_mac->GetWifiRemoteStationManager ()->GetVhtSupported ());
    }

  NS_LOG_FUNCTION (this << recipient << +tid << establish);
  return establish;
}

void
HtFrameExchangeManager::SendAddBaRequest (Mac48Address dest, uint8_t tid, uint16_t startingSeq,
                                          uint16_t timeout, bool immediateBAck)
{
  NS_LOG_FUNCTION (this << dest << +tid << startingSeq << timeout << immediateBAck);
  NS_LOG_DEBUG ("Send ADDBA request to " << dest);

  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetAddr1 (dest);
  hdr.SetAddr2 (m_self);
  hdr.SetAddr3 (m_bssid);
  hdr.SetDsNotTo ();
  hdr.SetDsNotFrom ();

  WifiActionHeader actionHdr;
  WifiActionHeader::ActionValue action;
  action.blockAck = WifiActionHeader::BLOCK_ACK_ADDBA_REQUEST;
  actionHdr.SetAction (WifiActionHeader::BLOCK_ACK, action);

  Ptr<Packet> packet = Create<Packet> ();
  // Setting ADDBARequest header
  MgtAddBaRequestHeader reqHdr;
  reqHdr.SetAmsduSupport (true);
  if (immediateBAck)
    {
      reqHdr.SetImmediateBlockAck ();
    }
  else
    {
      reqHdr.SetDelayedBlockAck ();
    }
  reqHdr.SetTid (tid);
  /* For now we don't use buffer size field in the ADDBA request frame. The recipient
   * will choose how many packets it can receive under block ack.
   */
  reqHdr.SetBufferSize (0);
  reqHdr.SetTimeout (timeout);
  // set the starting sequence number for the BA agreement
  reqHdr.SetStartingSequence (startingSeq);

  GetBaManager (tid)->CreateAgreement (&reqHdr, dest);

  packet->AddHeader (reqHdr);
  packet->AddHeader (actionHdr);

  Ptr<WifiMacQueueItem> mpdu = Create<WifiMacQueueItem> (packet, hdr);

  // get the sequence number for the ADDBA Request management frame
  uint16_t sequence = m_txMiddle->GetNextSequenceNumberFor (&mpdu->GetHeader ());
  mpdu->GetHeader ().SetSequenceNumber (sequence);

  WifiTxParameters txParams;
  txParams.m_txVector = m_mac->GetWifiRemoteStationManager ()->GetDataTxVector (mpdu->GetHeader ());
  txParams.m_protection = std::unique_ptr<WifiProtection> (new WifiNoProtection);
  txParams.m_acknowledgment = GetAckManager ()->TryAddMpdu (mpdu, txParams);

  // Push the MPDU to the front of the queue and transmit it
  auto queue = m_mac->GetQosTxop (tid)->GetWifiMacQueue ();
  if (!queue->PushFront (mpdu))
    {
      NS_LOG_DEBUG ("Queue is full, replace the oldest frame with the ADDBA Request frame");
      queue->Replace (queue->Peek (), mpdu);
    }
  SendMpduWithProtection (mpdu, txParams);
}

void
HtFrameExchangeManager::SendAddBaResponse (const MgtAddBaRequestHeader *reqHdr,
                                           Mac48Address originator)
{
  NS_LOG_FUNCTION (this << originator);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetAddr1 (originator);
  hdr.SetAddr2 (m_self);
  hdr.SetAddr3 (m_bssid);
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();

  MgtAddBaResponseHeader respHdr;
  StatusCode code;
  code.SetSuccess ();
  respHdr.SetStatusCode (code);
  //Here a control about queues type?
  respHdr.SetAmsduSupport (reqHdr->IsAmsduSupported ());

  if (reqHdr->IsImmediateBlockAck ())
    {
      respHdr.SetImmediateBlockAck ();
    }
  else
    {
      respHdr.SetDelayedBlockAck ();
    }
  respHdr.SetTid (reqHdr->GetTid ());

  respHdr.SetBufferSize (GetSupportedBaBufferSize ());
  respHdr.SetTimeout (reqHdr->GetTimeout ());

  WifiActionHeader actionHdr;
  WifiActionHeader::ActionValue action;
  action.blockAck = WifiActionHeader::BLOCK_ACK_ADDBA_RESPONSE;
  actionHdr.SetAction (WifiActionHeader::BLOCK_ACK, action);

  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (respHdr);
  packet->AddHeader (actionHdr);

  CreateBlockAckAgreement (&respHdr, originator, reqHdr->GetStartingSequence ());

  //It is unclear which queue this frame should go into. For now we
  //bung it into the queue corresponding to the TID for which we are
  //establishing an agreement, and push it to the head.
  m_mac->GetQosTxop (reqHdr->GetTid ())->PushFront (packet, hdr);
}

uint16_t
HtFrameExchangeManager::GetSupportedBaBufferSize (void) const
{
  return 64;
}

void
HtFrameExchangeManager::SendDelbaFrame (Mac48Address addr, uint8_t tid, bool byOriginator)
{
  NS_LOG_FUNCTION (this << addr << +tid << byOriginator);
  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_MGT_ACTION);
  hdr.SetAddr1 (addr);
  hdr.SetAddr2 (m_self);
  hdr.SetAddr3 (m_bssid);
  hdr.SetDsNotTo ();
  hdr.SetDsNotFrom ();

  MgtDelBaHeader delbaHdr;
  delbaHdr.SetTid (tid);
  if (byOriginator)
    {
      delbaHdr.SetByOriginator ();
      GetBaManager (tid)->DestroyAgreement (addr, tid);
    }
  else
    {
      delbaHdr.SetByRecipient ();
      DestroyBlockAckAgreement (addr, tid);
    }

  WifiActionHeader actionHdr;
  WifiActionHeader::ActionValue action;
  action.blockAck = WifiActionHeader::BLOCK_ACK_DELBA;
  actionHdr.SetAction (WifiActionHeader::BLOCK_ACK, action);

  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (delbaHdr);
  packet->AddHeader (actionHdr);

  m_mac->GetQosTxop (tid)->GetWifiMacQueue ()->PushFront (Create<WifiMacQueueItem> (packet, hdr));
}

void
HtFrameExchangeManager::CreateBlockAckAgreement (const MgtAddBaResponseHeader *respHdr, Mac48Address originator,
                                                 uint16_t startingSeq)
{
  NS_LOG_FUNCTION (this << *respHdr << originator << startingSeq);
  uint8_t tid = respHdr->GetTid ();

  RecipientBlockAckAgreement agreement (originator, respHdr->IsAmsduSupported (), tid,
                                        respHdr->GetBufferSize (), respHdr->GetTimeout (),
                                        startingSeq,
                                        m_mac->GetWifiRemoteStationManager ()->GetHtSupported ()
                                        && m_mac->GetWifiRemoteStationManager ()->GetHtSupported (originator));
  agreement.SetMacRxMiddle (m_rxMiddle);
  if (respHdr->IsImmediateBlockAck ())
    {
      agreement.SetImmediateBlockAck ();
    }
  else
    {
      agreement.SetDelayedBlockAck ();
    }

  if (respHdr->GetTimeout () != 0)
    {
      Time timeout = MicroSeconds (1024 * agreement.GetTimeout ());

      agreement.m_inactivityEvent = Simulator::Schedule (timeout, &HtFrameExchangeManager::SendDelbaFrame,
                                                         this, originator, tid, false);
    }

  m_agreements.insert ({{originator, tid}, agreement});
}

void
HtFrameExchangeManager::DestroyBlockAckAgreement  (Mac48Address originator, uint8_t tid)
{
  NS_LOG_FUNCTION (this << originator << +tid);

  auto agreementIt = m_agreements.find ({originator, tid});
  if (agreementIt != m_agreements.end ())
    {
      // forward up the buffered MPDUs before destroying the agreement
      agreementIt->second.Flush ();
      m_agreements.erase (agreementIt);
    }
}

bool
HtFrameExchangeManager::StartFrameExchange (Ptr<QosTxop> edca, Time availableTime, bool initialFrame)
{
  NS_LOG_FUNCTION (this << edca << availableTime << initialFrame);

  // First, check if there is a BAR to be transmitted
  if (SendMpduFromBaManager (edca, availableTime, initialFrame))
    {
      return true;
    }

  Ptr<const WifiMacQueueItem> peekedItem = edca->PeekNextMpdu ();

  // Even though channel access is requested when the queue is not empty, at
  // the time channel access is granted the lifetime of the packet might be
  // expired and the queue might be empty.
  if (peekedItem == 0)
    {
      NS_LOG_DEBUG ("No frames available for transmission");
      return false;
    }

  const WifiMacHeader& hdr = peekedItem->GetHeader ();
  // setup a Block Ack agreement if needed
  if (hdr.IsQosData () && !hdr.GetAddr1 ().IsGroup ()
      && NeedSetupBlockAck (hdr.GetAddr1 (), hdr.GetQosTid ()))
    {
      // if the peeked MPDU has been already transmitted, use its sequence number
      // as the starting sequence number for the BA agreement, otherwise use the
      // next available sequence number
      uint16_t startingSeq = (hdr.IsRetry () ? hdr.GetSequenceNumber ()
                                             : m_txMiddle->GetNextSeqNumberByTidAndAddress (hdr.GetQosTid (),
                                                                                            hdr.GetAddr1 ()));
      SendAddBaRequest (hdr.GetAddr1 (), hdr.GetQosTid (), startingSeq,
                        edca->GetBlockAckInactivityTimeout (), true);
      return true;
    }

  // Use SendDataFrame if we can try aggregation
  if (hdr.IsQosData () && !hdr.GetAddr1 ().IsGroup () && !peekedItem->IsFragment ()
      && !m_mac->GetWifiRemoteStationManager ()->NeedFragmentation (peekedItem))
    {
      return SendDataFrame (peekedItem, availableTime, initialFrame);
    }

  // Use the QoS FEM to transmit the frame in all the other cases, i.e.:
  // - the frame is not a QoS data frame
  // - the frame is a broadcast QoS data frame
  // - the frame is a fragment
  // - the frame must be fragmented
  return QosFrameExchangeManager::StartFrameExchange (edca, availableTime, initialFrame);
}

bool
HtFrameExchangeManager::SendMpduFromBaManager (Ptr<QosTxop> edca, Time availableTime, bool initialFrame)
{
  NS_LOG_FUNCTION (this << edca << availableTime << initialFrame);

  // First, check if there is a BAR to be transmitted
  Ptr<const WifiMacQueueItem> peekedItem = edca->GetBaManager ()->GetBar (false);

  if (peekedItem == 0)
    {
      NS_LOG_DEBUG ("Block Ack Manager returned no frame to send");
      return false;
    }

  NS_ASSERT (peekedItem->GetHeader ().IsBlockAckReq ());

  // Prepare the TX parameters. Note that the default ack manager expects the
  // data TxVector in the m_txVector field to compute the BlockAck TxVector.
  // The m_txVector field of the TX parameters is set to the BlockAckReq TxVector
  // a few lines below.
  WifiTxParameters txParams;
  txParams.m_txVector = m_mac->GetWifiRemoteStationManager ()->GetDataTxVector (peekedItem->GetHeader ());
  txParams.m_protection = std::unique_ptr<WifiProtection> (new WifiNoProtection);
  txParams.m_acknowledgment = GetAckManager ()->TryAddMpdu (peekedItem, txParams);

  NS_ABORT_IF (txParams.m_acknowledgment->method != WifiAcknowledgment::BLOCK_ACK);

  WifiBlockAck* blockAcknowledgment = static_cast<WifiBlockAck*> (txParams.m_acknowledgment.get ());
  CalculateAcknowledgmentTime (blockAcknowledgment);
  // the BlockAckReq frame is sent using the same TXVECTOR as the BlockAck frame
  txParams.m_txVector = blockAcknowledgment->blockAckTxVector;

  Time barTxDuration = m_phy->CalculateTxDuration (peekedItem->GetSize (),
                                                   blockAcknowledgment->blockAckTxVector,
                                                   m_phy->GetPhyBand ());

  // if the available time is limited and we are not transmitting the initial
  // frame of the TXOP, we have to check that this frame and its response fit
  // within the given time limits
  if (availableTime != Time::Min () && !initialFrame
      && barTxDuration + m_phy->GetSifs () + blockAcknowledgment->acknowledgmentTime > availableTime)
    {
      NS_LOG_DEBUG ("Not enough time to send the BAR frame returned by the Block Ack Manager");
      return false;
    }

  // we can transmit the BlockAckReq frame
  Ptr<const WifiMacQueueItem> mpdu = edca->GetBaManager ()->GetBar ();
  SendPsduWithProtection (GetWifiPsdu (Copy (mpdu), txParams.m_txVector), txParams);
  return true;
}

bool
HtFrameExchangeManager::SendDataFrame (Ptr<const WifiMacQueueItem> peekedItem,
                                       Time availableTime, bool initialFrame)
{
  NS_ASSERT (peekedItem != 0 && peekedItem->GetHeader ().IsQosData ()
             && !peekedItem->GetHeader ().GetAddr1 ().IsBroadcast ()
             && !peekedItem->IsFragment ());
  NS_LOG_FUNCTION (this << *peekedItem << availableTime << initialFrame);

  Ptr<QosTxop> edca = m_mac->GetQosTxop (peekedItem->GetHeader ().GetQosTid ());
  WifiTxParameters txParams;
  txParams.m_txVector = m_mac->GetWifiRemoteStationManager ()->GetDataTxVector (peekedItem->GetHeader ());
  Ptr<WifiMacQueueItem> mpdu = edca->GetNextMpdu (peekedItem, txParams, availableTime, initialFrame);

  if (mpdu == nullptr)
    {
      NS_LOG_DEBUG ("Not enough time to transmit a frame");
      return false;
    }

  // try A-MPDU aggregation
  std::vector<Ptr<WifiMacQueueItem>> mpduList = m_mpduAggregator->GetNextAmpdu (mpdu, txParams,
                                                                                availableTime);
  NS_ASSERT (txParams.m_acknowledgment);

  if (mpduList.size () > 1)
    {
      // A-MPDU aggregation succeeded
      SendPsduWithProtection (Create<WifiPsdu> (std::move (mpduList)), txParams);
    }
  else if (txParams.m_acknowledgment->method == WifiAcknowledgment::BAR_BLOCK_ACK)
    {
      // a QoS data frame using the Block Ack policy can be followed by a BlockAckReq
      // frame and a BlockAck frame. Such a sequence is handled by the HT FEM
      SendPsduWithProtection (Create<WifiPsdu> (mpdu, false), txParams);
    }
  else
    {
      // transmission can be handled by the base FEM
      SendMpduWithProtection (mpdu, txParams);
    }

  return true;
}

void
HtFrameExchangeManager::CalculateAcknowledgmentTime (WifiAcknowledgment* acknowledgment) const
{
  NS_LOG_FUNCTION (this << acknowledgment);
  NS_ASSERT (acknowledgment != nullptr);

  if (acknowledgment->method == WifiAcknowledgment::BLOCK_ACK)
    {
      WifiBlockAck* blockAcknowledgment = static_cast<WifiBlockAck*> (acknowledgment);
      Time baTxDuration = m_phy->CalculateTxDuration (GetBlockAckSize (blockAcknowledgment->baType),
                                                      blockAcknowledgment->blockAckTxVector,
                                                      m_phy->GetPhyBand ());
      blockAcknowledgment->acknowledgmentTime = m_phy->GetSifs () + baTxDuration;
    }
  else if (acknowledgment->method == WifiAcknowledgment::BAR_BLOCK_ACK)
    {
      WifiBarBlockAck* barBlockAcknowledgment = static_cast<WifiBarBlockAck*> (acknowledgment);
      Time barTxDuration = m_phy->CalculateTxDuration (GetBlockAckRequestSize (barBlockAcknowledgment->barType),
                                                      barBlockAcknowledgment->blockAckReqTxVector,
                                                      m_phy->GetPhyBand ());
      Time baTxDuration = m_phy->CalculateTxDuration (GetBlockAckSize (barBlockAcknowledgment->baType),
                                                      barBlockAcknowledgment->blockAckTxVector,
                                                      m_phy->GetPhyBand ());
      barBlockAcknowledgment->acknowledgmentTime = 2 * m_phy->GetSifs () + barTxDuration + baTxDuration;
    }
  else
    {
      QosFrameExchangeManager::CalculateAcknowledgmentTime (acknowledgment);
    }
}

void
HtFrameExchangeManager::ForwardMpduDown (Ptr<WifiMacQueueItem> mpdu, WifiTxVector& txVector)
{
  ForwardPsduDown (GetWifiPsdu (mpdu, txVector), txVector);
}

Ptr<WifiPsdu>
HtFrameExchangeManager::GetWifiPsdu (Ptr<WifiMacQueueItem> mpdu, const WifiTxVector& txVector) const
{
  return Create<WifiPsdu> (mpdu, false);
}

void
HtFrameExchangeManager::NotifyReceivedNormalAck (Ptr<WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  if (mpdu->GetHeader ().IsQosData ())
    {
      uint8_t tid = mpdu->GetHeader ().GetQosTid ();
      Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

      if (edca->GetBaAgreementEstablished (mpdu->GetHeader ().GetAddr1 (), tid))
        {
          // notify the BA manager that the MPDU was acknowledged
          edca->GetBaManager ()->NotifyGotAck (mpdu);
        }
    }
  else if (mpdu->GetHeader ().IsAction ())
    {
      WifiActionHeader actionHdr;
      Ptr<Packet> p = mpdu->GetPacket ()->Copy ();
      p->RemoveHeader (actionHdr);
      if (actionHdr.GetCategory () == WifiActionHeader::BLOCK_ACK)
        {
          if (actionHdr.GetAction ().blockAck == WifiActionHeader::BLOCK_ACK_DELBA)
            {
              MgtDelBaHeader delBa;
              p->PeekHeader (delBa);
              if (delBa.IsByOriginator ())
                {
                  GetBaManager (delBa.GetTid ())->DestroyAgreement (mpdu->GetHeader ().GetAddr1 (),
                                                                    delBa.GetTid ());
                }
              else
                {
                  DestroyBlockAckAgreement (mpdu->GetHeader ().GetAddr1 (), delBa.GetTid ());
                }
            }
          else if (actionHdr.GetAction ().blockAck == WifiActionHeader::BLOCK_ACK_ADDBA_REQUEST)
            {
              // Setup ADDBA response timeout
              MgtAddBaRequestHeader addBa;
              p->PeekHeader (addBa);
              Ptr<QosTxop> edca = m_mac->GetQosTxop (addBa.GetTid ());
              Simulator::Schedule (edca->GetAddBaResponseTimeout (),
                                   &QosTxop::AddBaResponseTimeout, edca,
                                   mpdu->GetHeader ().GetAddr1 (), addBa.GetTid ());
            }
        }
    }
  QosFrameExchangeManager::NotifyReceivedNormalAck (mpdu);
}

void
HtFrameExchangeManager::TransmissionSucceeded (void)
{
  NS_LOG_DEBUG (this);

  if (m_edca != 0 && m_edca->GetTxopLimit ().IsZero () && m_edca->GetBaManager ()->GetBar (false) != 0)
    {
      // A TXOP limit of 0 indicates that the TXOP holder may transmit or cause to
      // be transmitted (as responses) the following within the current TXOP:
      // f) Any number of BlockAckReq frames
      // (Sec. 10.22.2.8 of 802.11-2016)
      NS_LOG_DEBUG ("Schedule a transmission from Block Ack Manager in a SIFS");
      bool (HtFrameExchangeManager::*fp) (Ptr<QosTxop>, Time) = &HtFrameExchangeManager::StartTransmission;

      // TXOP limit is null, hence the txopDuration parameter is unused
      Simulator::Schedule (m_phy->GetSifs (), fp, this, m_edca, Seconds (0));
    }
  else
    {
      QosFrameExchangeManager::TransmissionSucceeded ();
    }
}

void
HtFrameExchangeManager::NotifyPacketDiscarded (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  if (mpdu->GetHeader ().IsQosData ())
    {
      GetBaManager (mpdu->GetHeader ().GetQosTid ())->NotifyDiscardedMpdu (mpdu);
    }
  else if (mpdu->GetHeader ().IsAction ())
    {
      WifiActionHeader actionHdr;
      mpdu->GetPacket ()->PeekHeader (actionHdr);
      if (actionHdr.GetCategory () == WifiActionHeader::BLOCK_ACK)
        {
          uint8_t tid = GetTid (mpdu->GetPacket (), mpdu->GetHeader ());
          if (GetBaManager (tid)->ExistsAgreementInState (mpdu->GetHeader ().GetAddr1 (), tid,
                                                          OriginatorBlockAckAgreement::PENDING))
            {
              NS_LOG_DEBUG ("No ACK after ADDBA request");
              GetBaManager (tid)->NotifyAgreementNoReply (mpdu->GetHeader ().GetAddr1 (), tid);
              Ptr<QosTxop> qosTxop = m_mac->GetQosTxop (tid);
              Simulator::Schedule (qosTxop->GetFailedAddBaTimeout (), &QosTxop::ResetBa, qosTxop,
                                   mpdu->GetHeader ().GetAddr1 (), tid);
            }
        }
    }
  QosFrameExchangeManager::NotifyPacketDiscarded (mpdu);
}

void
HtFrameExchangeManager::RetransmitMpduAfterMissedAck (Ptr<WifiMacQueueItem> mpdu) const
{
  NS_LOG_FUNCTION (this << *mpdu);

  if (mpdu->GetHeader ().IsQosData ())
    {
      uint8_t tid = mpdu->GetHeader ().GetQosTid ();
      Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

      if (edca->GetBaAgreementEstablished (mpdu->GetHeader ().GetAddr1 (), tid))
        {
          // notify the BA manager that the MPDU was not acknowledged
          edca->GetBaManager ()->NotifyMissedAck (mpdu);
          return;
        }
    }
  QosFrameExchangeManager::RetransmitMpduAfterMissedAck (mpdu);
}

void
HtFrameExchangeManager::ReleaseSequenceNumber (Ptr<WifiMacQueueItem> mpdu) const
{
  NS_LOG_FUNCTION (this << *mpdu);

  // the MPDU should be still in the queue, unless it expired.
  const WifiMacHeader& hdr = mpdu->GetHeader ();
  if (hdr.IsQosData ())
    {
      uint8_t tid = hdr.GetQosTid ();
      Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

      if (edca->GetBaAgreementEstablished (hdr.GetAddr1 (), tid)
          && !hdr.IsRetry ())
        {
          // The MPDU has never been transmitted, so we can make its sequence
          // number available again if it is lower than the sequence number
          // maintained by the MAC TX middle
          uint16_t currentNextSeq = m_txMiddle->PeekNextSequenceNumberFor (&hdr);
          uint16_t startingSeq = edca->GetBaStartingSequence (hdr.GetAddr1 (), tid);

          if (BlockAckAgreement::GetDistance (hdr.GetSequenceNumber (), startingSeq)
              < BlockAckAgreement::GetDistance (currentNextSeq, startingSeq))
            {
              m_txMiddle->SetSequenceNumberFor (&hdr);
            }

          return;
        }
    }
  QosFrameExchangeManager::ReleaseSequenceNumber (mpdu);
}

Time
HtFrameExchangeManager::GetPsduDurationId (Time txDuration, const WifiTxParameters& txParams) const
{
  NS_LOG_FUNCTION (this << txDuration << &txParams);

  NS_ASSERT (m_edca != 0);

  if (m_edca->GetTxopLimit ().IsZero ())
    {
      NS_ASSERT (txParams.m_acknowledgment && txParams.m_acknowledgment->acknowledgmentTime != Time::Min ());
      return txParams.m_acknowledgment->acknowledgmentTime;
    }

  // under multiple protection settings, if the TXOP limit is not null, Duration/ID
  // is set to cover the remaining TXOP time (Sec. 9.2.5.2 of 802.11-2016).
  // The TXOP holder may exceed the TXOP limit in some situations (Sec. 10.22.2.8
  // of 802.11-2016)
  return std::max (m_edca->GetRemainingTxop () - txDuration, Seconds (0));
}

void
HtFrameExchangeManager::SendPsduWithProtection (Ptr<WifiPsdu> psdu, WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << psdu << &txParams);

  m_psdu = psdu;
  m_txParams = std::move (txParams);

#ifdef NS3_BUILD_PROFILE_DEBUG
  // If protection is required, the MPDUs must be stored in some queue because
  // they are not put back in a queue if the RTS/CTS exchange fails
  if (m_txParams.m_protection->method != WifiProtection::NONE)
    {
      for (const auto& mpdu : *PeekPointer (m_psdu))
        {
          NS_ASSERT (mpdu->GetHeader ().IsCtl () || mpdu->IsQueued ());
        }
    }
#endif

  // Make sure that the acknowledgment time has been computed, so that SendRts()
  // and SendCtsToSelf() can reuse this value.
  NS_ASSERT (m_txParams.m_acknowledgment);

  if (m_txParams.m_acknowledgment->acknowledgmentTime == Time::Min ())
    {
      CalculateAcknowledgmentTime (m_txParams.m_acknowledgment.get ());
    }

  // Set QoS Ack policy
  WifiAckManager::SetQosAckPolicy (m_psdu, m_txParams.m_acknowledgment.get ());

  if (m_txParams.m_protection->method == WifiProtection::RTS_CTS)
    {
      SendRts (m_txParams);
    }
  else if (m_txParams.m_protection->method == WifiProtection::CTS_TO_SELF)
    {
      SendCtsToSelf (m_txParams);
    }
  else if (m_txParams.m_protection->method == WifiProtection::NONE)
    {
      SendPsdu ();
    }
  else
    {
      NS_ABORT_MSG ("Unknown protection type");
    }
}

void
HtFrameExchangeManager::CtsTimeout (Ptr<WifiMacQueueItem> rts, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << *rts << txVector);

  if (m_psdu == 0)
    {
      // A CTS Timeout occurred when protecting a single MPDU is handled by the
      // parent classes
      QosFrameExchangeManager::CtsTimeout (rts, txVector);
      return;
    }

  DoCtsTimeout (m_psdu);
  m_psdu = nullptr;
}

void
HtFrameExchangeManager::SendPsdu (void)
{
  NS_LOG_FUNCTION (this);

  Time txDuration = m_phy->CalculateTxDuration (m_psdu->GetSize (), m_txParams.m_txVector, m_phy->GetPhyBand ());

  NS_ASSERT (m_txParams.m_acknowledgment);

  if (m_txParams.m_acknowledgment->method == WifiAcknowledgment::NONE)
    {
      Simulator::Schedule (txDuration, &HtFrameExchangeManager::TransmissionSucceeded, this);

      std::set<uint8_t> tids = m_psdu->GetTids ();
      NS_ASSERT_MSG (tids.size () <= 1, "Multi-TID A-MPDUs are not supported");

      if (tids.size () == 0 || m_psdu->GetAckPolicyForTid (*tids.begin ()) == WifiMacHeader::NO_ACK)
        {
          // No acknowledgment, hence dequeue the PSDU if it is stored in a queue
          DequeuePsdu (m_psdu);
        }
    }
  else if (m_txParams.m_acknowledgment->method == WifiAcknowledgment::BLOCK_ACK)
    {
      m_psdu->SetDuration (GetPsduDurationId (txDuration, m_txParams));

      // the timeout duration is "aSIFSTime + aSlotTime + aRxPHYStartDelay, starting
      // at the PHY-TXEND.confirm primitive" (section 10.3.2.9 or 10.22.2.2 of 802.11-2016).
      // aRxPHYStartDelay equals the time to transmit the PHY header.
      WifiBlockAck* blockAcknowledgment = static_cast<WifiBlockAck*> (m_txParams.m_acknowledgment.get ());

      Time timeout = txDuration
                    + m_phy->GetSifs ()
                    + m_phy->GetSlot ()
                    + m_phy->CalculatePhyPreambleAndHeaderDuration (blockAcknowledgment->blockAckTxVector);
      NS_ASSERT (!m_txTimer.IsRunning ());
      m_txTimer.Set (WifiTxTimer::WAIT_BLOCK_ACK, timeout, &HtFrameExchangeManager::BlockAckTimeout,
                     this, m_psdu, m_txParams.m_txVector);
      m_channelAccessManager->NotifyAckTimeoutStartNow (timeout);
    }
  else if (m_txParams.m_acknowledgment->method == WifiAcknowledgment::BAR_BLOCK_ACK)
    {
      m_psdu->SetDuration (GetPsduDurationId (txDuration, m_txParams));

      // schedule the transmission of a BAR in a SIFS
      std::set<uint8_t> tids = m_psdu->GetTids ();
      NS_ABORT_MSG_IF (tids.size () > 1, "Acknowledgment method incompatible with a Multi-TID A-MPDU");
      uint8_t tid = *tids.begin ();

      Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);
      edca->ScheduleBar (edca->PrepareBlockAckRequest (m_psdu->GetAddr1 (), tid));

      Simulator::Schedule (txDuration, &HtFrameExchangeManager::TransmissionSucceeded, this);
    }
  else
    {
      NS_ABORT_MSG ("Unable to handle the selected acknowledgment method ("
                    << m_txParams.m_acknowledgment.get () << ")");
    }

  // transmit the PSDU
  if (m_psdu->GetNMpdus () > 1)
    {
      ForwardPsduDown (m_psdu, m_txParams.m_txVector);
    }
  else
    {
      ForwardMpduDown (*m_psdu->begin (), m_txParams.m_txVector);
    }

  if (m_txParams.m_acknowledgment->method == WifiAcknowledgment::NONE)
    {
      // we are done in case the A-MPDU does not require acknowledgment
      m_psdu = 0;
    }
}

void
HtFrameExchangeManager::NotifyTxToEdca (Ptr<const WifiPsdu> psdu) const
{
  NS_LOG_FUNCTION (this << psdu);

  // use an array to avoid computing the queue size for every MPDU in the PSDU
  std::array<std::optional<uint8_t>, 8> queueSizeForTid;

  for (const auto& mpdu : *PeekPointer (psdu))
    {
      WifiMacHeader& hdr = mpdu->GetHeader ();

      if (hdr.IsQosData ())
        {
          uint8_t tid = hdr.GetQosTid ();
          Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

          if (m_mac->GetTypeOfStation () == STA
              && (m_setQosQueueSize || hdr.IsQosEosp ()))
            {
              // set the Queue Size subfield of the QoS Control field
              if (!queueSizeForTid[tid].has_value ())
                {
                  queueSizeForTid[tid] = edca->GetQosQueueSize (tid, hdr.GetAddr1 ());
                }

              hdr.SetQosEosp ();
              hdr.SetQosQueueSize (queueSizeForTid[tid].value ());
            }

          if (hdr.HasData ())
            {
              edca->CompleteMpduTx (mpdu);
            }
        }
    }
}

void
HtFrameExchangeManager::DequeuePsdu (Ptr<const WifiPsdu> psdu)
{
  NS_LOG_DEBUG (this << psdu);

  for (const auto& mpdu : *PeekPointer (psdu))
    {
      DequeueMpdu (mpdu);
    }
}

void
HtFrameExchangeManager::ForwardPsduDown (Ptr<const WifiPsdu> psdu, WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << psdu << txVector);

  NS_LOG_DEBUG ("Transmitting a PSDU: " << *psdu << " TXVECTOR: " << txVector);
  NotifyTxToEdca (psdu);

  if (psdu->IsAggregate ())
    {
      txVector.SetAggregation (true);
    }

  m_phy->Send (psdu, txVector);
}

bool
HtFrameExchangeManager::IsWithinLimitsIfAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                                 const WifiTxParameters& txParams,
                                                 Time ppduDurationLimit) const
{
  NS_ASSERT (mpdu != 0);
  NS_LOG_FUNCTION (this << *mpdu << &txParams << ppduDurationLimit);

  Mac48Address receiver = mpdu->GetHeader ().GetAddr1 ();
  uint32_t ampduSize = txParams.GetSizeIfAddMpdu (mpdu);

  if (txParams.GetSize (receiver) > 0)
    {
      // we are attempting to perform A-MPDU aggregation, hence we have to check
      // that we meet the limit on the max A-MPDU size
      uint8_t tid;
      const WifiTxParameters::PsduInfo* info;

      if (mpdu->GetHeader ().IsQosData ())
        {
          tid = mpdu->GetHeader ().GetQosTid ();
        }
      else if ((info = txParams.GetPsduInfo (receiver)) && !info->seqNumbers.empty ())
        {
          tid = info->seqNumbers.begin ()->first;
        }
      else
        {
          NS_ABORT_MSG ("Cannot aggregate a non-QoS data frame to an A-MPDU that does"
                        " not contain any QoS data frame");
        }

      WifiModulationClass modulation = txParams.m_txVector.GetModulationClass ();

      if (!IsWithinAmpduSizeLimit (ampduSize, receiver, tid, modulation))
        {
          return false;
        }
    }

  return IsWithinSizeAndTimeLimits (ampduSize, receiver, txParams, ppduDurationLimit);
}

bool
HtFrameExchangeManager::IsWithinAmpduSizeLimit (uint32_t ampduSize, Mac48Address receiver, uint8_t tid,
                                                WifiModulationClass modulation) const
{
  NS_LOG_FUNCTION (this << ampduSize << receiver << +tid << modulation);

  uint32_t maxAmpduSize = m_mpduAggregator->GetMaxAmpduSize (receiver, tid, modulation);

  if (maxAmpduSize == 0)
    {
      NS_LOG_DEBUG ("A-MPDU aggregation disabled");
      return false;
    }

  if (ampduSize > maxAmpduSize)
    {
      NS_LOG_DEBUG ("the frame does not meet the constraint on max A-MPDU size ("
                    << maxAmpduSize << ")");
      return false;
    }
  return true;
}

bool
HtFrameExchangeManager::TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu, WifiTxParameters& txParams,
                                          Time availableTime) const
{
  NS_ASSERT (msdu != 0 && msdu->GetHeader ().IsQosData ());
  NS_LOG_FUNCTION (this << *msdu << &txParams << availableTime);

  // check if aggregating the given MSDU requires a different protection method
  NS_ASSERT (txParams.m_protection);
  Time protectionTime = txParams.m_protection->protectionTime;

  std::unique_ptr<WifiProtection> protection;
  protection = GetProtectionManager ()->TryAggregateMsdu (msdu, txParams);
  bool protectionSwapped = false;

  if (protection)
    {
      // the protection method has changed, calculate the new protection time
      CalculateProtectionTime (protection.get ());
      protectionTime = protection->protectionTime;
      // swap unique pointers, so that the txParams that is passed to the next
      // call to IsWithinLimitsIfAggregateMsdu is the most updated one
      txParams.m_protection.swap (protection);
      protectionSwapped = true;
    }
  NS_ASSERT (protectionTime != Time::Min ());

  // check if aggregating the given MSDU requires a different acknowledgment method
  NS_ASSERT (txParams.m_acknowledgment);
  Time acknowledgmentTime = txParams.m_acknowledgment->acknowledgmentTime;

  std::unique_ptr<WifiAcknowledgment> acknowledgment;
  acknowledgment = GetAckManager ()->TryAggregateMsdu (msdu, txParams);
  bool acknowledgmentSwapped = false;

  if (acknowledgment)
    {
      // the acknowledgment method has changed, calculate the new acknowledgment time
      CalculateAcknowledgmentTime (acknowledgment.get ());
      acknowledgmentTime = acknowledgment->acknowledgmentTime;
      // swap unique pointers, so that the txParams that is passed to the next
      // call to IsWithinLimitsIfAggregateMsdu is the most updated one
      txParams.m_acknowledgment.swap (acknowledgment);
      acknowledgmentSwapped = true;
    }
  NS_ASSERT (acknowledgmentTime != Time::Min ());

  Time ppduDurationLimit = Time::Min ();
  if (availableTime != Time::Min ())
    {
      ppduDurationLimit = availableTime - protectionTime - acknowledgmentTime;
    }

  if (!IsWithinLimitsIfAggregateMsdu (msdu, txParams, ppduDurationLimit))
    {
      // adding MPDU failed, restore protection and acknowledgment methods
      // if they were swapped
      if (protectionSwapped)
        {
          txParams.m_protection.swap (protection);
        }
      if (acknowledgmentSwapped)
        {
          txParams.m_acknowledgment.swap (acknowledgment);
        }
      return false;
    }

  // the given MPDU can be added, hence update the txParams
  txParams.AggregateMsdu (msdu);
  UpdateTxDuration (msdu->GetHeader ().GetAddr1 (), txParams);

  return true;
}

bool
HtFrameExchangeManager::IsWithinLimitsIfAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                                       const WifiTxParameters& txParams,
                                                       Time ppduDurationLimit) const
{
  NS_ASSERT (msdu != 0 && msdu->GetHeader ().IsQosData ());
  NS_LOG_FUNCTION (this << *msdu << &txParams << ppduDurationLimit);

  std::pair<uint16_t, uint32_t> ret = txParams.GetSizeIfAggregateMsdu (msdu);
  Mac48Address receiver = msdu->GetHeader ().GetAddr1 ();
  uint8_t tid = msdu->GetHeader ().GetQosTid ();
  WifiModulationClass modulation = txParams.m_txVector.GetModulationClass ();

  // Check that the limit on A-MSDU size is met
  uint16_t maxAmsduSize = m_msduAggregator->GetMaxAmsduSize (receiver, tid, modulation);

  if (maxAmsduSize == 0)
    {
      NS_LOG_DEBUG ("A-MSDU aggregation disabled");
      return false;
    }

  if (ret.first > maxAmsduSize)
    {
      NS_LOG_DEBUG ("No other MSDU can be aggregated: maximum A-MSDU size ("
                    << maxAmsduSize << ") reached ");
      return false;
    }

  const WifiTxParameters::PsduInfo* info = txParams.GetPsduInfo (msdu->GetHeader ().GetAddr1 ());
  NS_ASSERT (info != nullptr);

  if (info->ampduSize > 0)
    {
      // the A-MSDU being built is aggregated to other MPDUs in an A-MPDU.
      // Check that the limit on A-MPDU size is met.
      if (!IsWithinAmpduSizeLimit (ret.second, receiver, tid, modulation))
        {
          return false;
        }
    }

  return IsWithinSizeAndTimeLimits (ret.second, receiver, txParams, ppduDurationLimit);
}

void
HtFrameExchangeManager::BlockAckTimeout (Ptr<WifiPsdu> psdu, const WifiTxVector& txVector)
{
  NS_LOG_FUNCTION (this << *psdu << txVector);

  m_mac->GetWifiRemoteStationManager ()->ReportDataFailed (*psdu->begin ());

  bool resetCw;
  MissedBlockAck (psdu, txVector, resetCw);

  NS_ASSERT (m_edca != 0);

  if (resetCw)
    {
      m_edca->ResetCw ();
    }
  else
    {
      m_edca->UpdateFailedCw ();
    }

  m_psdu = 0;
  TransmissionFailed ();
}

void
HtFrameExchangeManager::MissedBlockAck (Ptr<WifiPsdu> psdu, const WifiTxVector& txVector, bool& resetCw)
{
  NS_LOG_FUNCTION (this << psdu << txVector << resetCw);

  Mac48Address recipient = psdu->GetAddr1 ();
  bool isBar;
  uint8_t tid;

  if (psdu->GetNMpdus () == 1 && psdu->GetHeader (0).IsBlockAckReq ())
    {
      isBar = true;
      CtrlBAckRequestHeader baReqHdr;
      psdu->GetPayload (0)->PeekHeader (baReqHdr);
      tid = baReqHdr.GetTidInfo ();
    }
  else
    {
      isBar = false;
      m_mac->GetWifiRemoteStationManager ()->ReportAmpduTxStatus (recipient, 0, psdu->GetNMpdus (),
                                                                  0, 0, txVector);
      std::set<uint8_t> tids = psdu->GetTids ();
      NS_ABORT_MSG_IF (tids.size () > 1, "Multi-TID A-MPDUs not handled here");
      NS_ASSERT (!tids.empty ());
      tid = *tids.begin ();
    }

  Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

  if (edca->UseExplicitBarAfterMissedBlockAck () || isBar)
    {
      // we have to send a BlockAckReq, if needed
      if (GetBaManager (tid)->NeedBarRetransmission (tid, recipient))
        {
          NS_LOG_DEBUG ("Missed Block Ack, transmit a BlockAckReq");
          if (isBar)
            {
              psdu->GetHeader (0).SetRetry ();
              edca->ScheduleBar (*psdu->begin ());
            }
          else
            {
              // missed block ack after data frame with Implicit BAR Ack policy
              edca->ScheduleBar (edca->PrepareBlockAckRequest (recipient, tid));
            }
          resetCw = false;
        }
      else
        {
          NS_LOG_DEBUG ("Missed Block Ack, do not transmit a BlockAckReq");
          // if a BA agreement exists, we can get here if there is no outstanding
          // MPDU whose lifetime has not expired yet.
          m_mac->GetWifiRemoteStationManager ()->ReportFinalDataFailed (*psdu->begin ());
          if (GetBaManager (tid)->ExistsAgreementInState (recipient, tid,
                                                          OriginatorBlockAckAgreement::ESTABLISHED))
            {
              // schedule a BlockAckRequest with skipIfNoDataQueued set to true, so that the
              // BlockAckRequest is only sent if there are data frames queued for this recipient.
              edca->ScheduleBar (edca->PrepareBlockAckRequest (recipient, tid), true);
            }
          resetCw = true;
        }
    }
  else
    {
      // we have to retransmit the data frames, if needed
      if (!m_mac->GetWifiRemoteStationManager ()->NeedRetransmission (*psdu->begin ()))
        {
          NS_LOG_DEBUG ("Missed Block Ack, do not retransmit the data frames");
          m_mac->GetWifiRemoteStationManager ()->ReportFinalDataFailed (*psdu->begin ());
          for (const auto& mpdu : *PeekPointer (psdu))
            {
              NotifyPacketDiscarded (mpdu);
              DequeueMpdu (mpdu);
            }
          resetCw = true;
        }
      else
        {
          NS_LOG_DEBUG ("Missed Block Ack, retransmit data frames");
          GetBaManager (tid)->NotifyMissedBlockAck (recipient, tid);
          resetCw = false;
        }
    }
}

void
HtFrameExchangeManager::SendBlockAck (const RecipientBlockAckAgreement& agreement, Time durationId,
                                      WifiTxVector& blockAckTxVector, double rxSnr)
{
  NS_LOG_FUNCTION (this << durationId << blockAckTxVector << rxSnr);

  WifiMacHeader hdr;
  hdr.SetType (WIFI_MAC_CTL_BACKRESP);
  hdr.SetAddr1 (agreement.GetPeer ());
  hdr.SetAddr2 (m_self);
  hdr.SetDsNotFrom ();
  hdr.SetDsNotTo ();

  CtrlBAckResponseHeader blockAck;
  blockAck.SetType (agreement.GetBlockAckType ());
  blockAck.SetTidInfo (agreement.GetTid ());
  agreement.FillBlockAckBitmap (&blockAck);

  Ptr<Packet> packet = Create<Packet> ();
  packet->AddHeader (blockAck);
  Ptr<WifiPsdu> psdu = GetWifiPsdu (Create<WifiMacQueueItem> (packet, hdr), blockAckTxVector);

  // 802.11-2016, Section 9.2.5.7: In a BlockAck frame transmitted in response
  // to a BlockAckReq frame or transmitted in response to a frame containing an
  // implicit block ack request, the Duration/ID field is set to the value obtained
  // from the Duration/ ID field of the frame that elicited the response minus the
  // time, in microseconds between the end of the PPDU carrying the frame that
  // elicited the response and the end of the PPDU carrying the BlockAck frame.
  Time baDurationId = durationId - m_phy->GetSifs ()
                      - m_phy->CalculateTxDuration (psdu, blockAckTxVector, m_phy->GetPhyBand ());
  // The TXOP holder may exceed the TXOP limit in some situations (Sec. 10.22.2.8 of 802.11-2016)
  if (baDurationId.IsStrictlyNegative ())
    {
      baDurationId = Seconds (0);
    }
  psdu->GetHeader (0).SetDuration (baDurationId);

  SnrTag tag;
  tag.Set (rxSnr);
  psdu->GetPayload (0)->AddPacketTag (tag);

  ForwardPsduDown (psdu, blockAckTxVector);
}

bool
HtFrameExchangeManager::GetBaAgreementEstablished (Mac48Address originator, uint8_t tid) const
{
  return (m_agreements.find ({originator, tid}) != m_agreements.end ());
}

BlockAckType
HtFrameExchangeManager::GetBlockAckType (Mac48Address originator, uint8_t tid) const
{
  auto it = m_agreements.find ({originator, tid});
  NS_ABORT_MSG_IF (it == m_agreements.end (), "No established Block Ack agreement");
  return it->second.GetBlockAckType ();
}

void
HtFrameExchangeManager::ReceiveMpdu (Ptr<WifiMacQueueItem> mpdu, RxSignalInfo rxSignalInfo,
                                     const WifiTxVector& txVector, bool inAmpdu)
{
  // The received MPDU is either broadcast or addressed to this station
  NS_ASSERT (mpdu->GetHeader ().GetAddr1 ().IsGroup ()
             || mpdu->GetHeader ().GetAddr1 () == m_self);

  double rxSnr = rxSignalInfo.snr;
  const WifiMacHeader& hdr = mpdu->GetHeader ();

  if (hdr.IsCtl ())
    {
      if (hdr.IsCts () && m_txTimer.IsRunning () && m_txTimer.GetReason () == WifiTxTimer::WAIT_CTS
          && m_psdu != 0)
        {
          NS_ABORT_MSG_IF (inAmpdu, "Received CTS as part of an A-MPDU");
          NS_ASSERT (hdr.GetAddr1 () == m_self);

          Mac48Address sender = m_psdu->GetAddr1 ();
          NS_LOG_DEBUG ("Received CTS from=" << sender);

          SnrTag tag;
          mpdu->GetPacket ()->PeekPacketTag (tag);
          m_mac->GetWifiRemoteStationManager ()->ReportRxOk (sender, rxSignalInfo, txVector);
          m_mac->GetWifiRemoteStationManager ()->ReportRtsOk (m_psdu->GetHeader (0),
                                                              rxSnr, txVector.GetMode (), tag.Get ());

          m_txTimer.Cancel ();
          m_channelAccessManager->NotifyCtsTimeoutResetNow ();
          Simulator::Schedule (m_phy->GetSifs (), &HtFrameExchangeManager::SendPsdu, this);
        }
      else if (hdr.IsBlockAck () && m_txTimer.IsRunning ()
               && m_txTimer.GetReason () == WifiTxTimer::WAIT_BLOCK_ACK
               && hdr.GetAddr1 () == m_self)
        {
          Mac48Address sender = hdr.GetAddr2 ();
          NS_LOG_DEBUG ("Received BlockAck from=" << sender);

          SnrTag tag;
          mpdu->GetPacket ()->PeekPacketTag (tag);

          // notify the Block Ack Manager
          CtrlBAckResponseHeader blockAck;
          mpdu->GetPacket ()->PeekHeader (blockAck);
          uint8_t tid = blockAck.GetTidInfo ();
          std::pair<uint16_t,uint16_t> ret = GetBaManager (tid)->NotifyGotBlockAck (blockAck, hdr.GetAddr2 (), {tid});
          m_mac->GetWifiRemoteStationManager ()->ReportAmpduTxStatus (hdr.GetAddr2 (), ret.first, ret.second,
                                                                      rxSnr, tag.Get (),  m_txParams.m_txVector);

          // cancel the timer
          m_txTimer.Cancel ();
          m_channelAccessManager->NotifyAckTimeoutResetNow ();

          // Reset the CW
          m_edca->ResetCw ();

          m_psdu = 0;
          TransmissionSucceeded ();
        }
      else if (hdr.IsBlockAckReq ())
        {
          NS_ASSERT (hdr.GetAddr1 () == m_self);
          NS_ABORT_MSG_IF (inAmpdu, "BlockAckReq in A-MPDU is not supported");

          Mac48Address sender = hdr.GetAddr2 ();
          NS_LOG_DEBUG ("Received BlockAckReq from=" << sender);

          CtrlBAckRequestHeader blockAckReq;
          mpdu->GetPacket ()->PeekHeader (blockAckReq);
          NS_ABORT_MSG_IF (blockAckReq.IsMultiTid (), "Multi-TID BlockAckReq not supported");
          uint8_t tid = blockAckReq.GetTidInfo ();

          auto agreementIt = m_agreements.find ({sender, tid});

          if (agreementIt == m_agreements.end ())
            {
              NS_LOG_DEBUG ("There's not a valid agreement for this BlockAckReq");
              return;
            }

          agreementIt->second.NotifyReceivedBar (blockAckReq.GetStartingSequence ());

          NS_LOG_DEBUG ("Schedule Block Ack");
          Simulator::Schedule (m_phy->GetSifs (), &HtFrameExchangeManager::SendBlockAck, this,
                               agreementIt->second, hdr.GetDuration (),
                               m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (sender, txVector),
                               rxSnr);
        }
      else
        {
          // the received control frame cannot be handled here
          QosFrameExchangeManager::ReceiveMpdu (mpdu, rxSignalInfo, txVector, inAmpdu);
        }
      return;
    }

  if (hdr.IsQosData () && hdr.HasData () && hdr.GetAddr1 () == m_self)
    {
      uint8_t tid = hdr.GetQosTid ();

      auto agreementIt = m_agreements.find ({hdr.GetAddr2 (), tid});
      if (agreementIt != m_agreements.end ())
        {
          // a Block Ack agreement has been established
          NS_LOG_DEBUG ("Received from=" << hdr.GetAddr2 ()
                        << " (" << *mpdu << ")");

          agreementIt->second.NotifyReceivedMpdu (mpdu);

          if (!inAmpdu && hdr.GetQosAckPolicy () == WifiMacHeader::NORMAL_ACK)
            {
              NS_LOG_DEBUG ("Schedule Normal Ack");
              Simulator::Schedule (m_phy->GetSifs (), &HtFrameExchangeManager::SendNormalAck,
                                   this, hdr, txVector, rxSnr);
            }
          return;
        }
      // We let the QosFrameExchangeManager handle QoS data frame not belonging
      // to a Block Ack agreement
    }

  QosFrameExchangeManager::ReceiveMpdu (mpdu, rxSignalInfo, txVector, inAmpdu);
}

void
HtFrameExchangeManager::EndReceiveAmpdu (Ptr<const WifiPsdu> psdu, const RxSignalInfo& rxSignalInfo,
                                         const WifiTxVector& txVector, const std::vector<bool>& perMpduStatus)
{
  std::set<uint8_t> tids = psdu->GetTids ();

  // Multi-TID A-MPDUs are not supported yet
  if (tids.size () == 1)
    {
      uint8_t tid = *tids.begin ();
      WifiMacHeader::QosAckPolicy ackPolicy = psdu->GetAckPolicyForTid (tid);
      NS_ASSERT (psdu->GetNMpdus () > 1);

      if (ackPolicy == WifiMacHeader::NORMAL_ACK)
        {
          // Normal Ack or Implicit Block Ack Request
          NS_LOG_DEBUG ("Schedule Block Ack");
          auto agreementIt = m_agreements.find ({psdu->GetAddr2 (), tid});
          NS_ASSERT (agreementIt != m_agreements.end ());

          Simulator::Schedule (m_phy->GetSifs (), &HtFrameExchangeManager::SendBlockAck, this,
                               agreementIt->second, psdu->GetDuration (),
                               m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (psdu->GetAddr2 (), txVector),
                               rxSignalInfo.snr);
        }
    }
}

} //namespace ns3
