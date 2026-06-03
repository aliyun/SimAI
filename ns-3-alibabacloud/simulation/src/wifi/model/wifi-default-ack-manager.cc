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
#include "wifi-default-ack-manager.h"
#include "wifi-tx-parameters.h"
#include "wifi-mac-queue-item.h"
#include "qos-utils.h"
#include "wifi-mac-queue.h"
#include "wifi-protection.h"
#include "ap-wifi-mac.h"
#include "ctrl-headers.h"
#include "ns3/he-phy.h"
#include "ns3/he-frame-exchange-manager.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiDefaultAckManager");

NS_OBJECT_ENSURE_REGISTERED (WifiDefaultAckManager);

TypeId
WifiDefaultAckManager::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WifiDefaultAckManager")
    .SetParent<WifiAckManager> ()
    .SetGroupName ("Wifi")
    .AddConstructor<WifiDefaultAckManager> ()
    .AddAttribute ("UseExplicitBar",
                   "Specify whether to send Block Ack Requests (if true) or use"
                   " Implicit Block Ack Request ack policy (if false).",
                   BooleanValue (false),
                   MakeBooleanAccessor (&WifiDefaultAckManager::m_useExplicitBar),
                   MakeBooleanChecker ())
    .AddAttribute ("BaThreshold",
                   "Immediate acknowledgment is requested upon transmission of a frame "
                   "whose sequence number is distant at least BaThreshold multiplied "
                   "by the transmit window size from the starting sequence number of "
                   "the transmit window. Set to zero to request a response for every "
                   "transmitted frame.",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&WifiDefaultAckManager::m_baThreshold),
                   MakeDoubleChecker<double> (0.0, 1.0))
    .AddAttribute ("DlMuAckSequenceType",
                   "Type of the acknowledgment sequence for DL MU PPDUs.",
                   EnumValue (WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE),
                   MakeEnumAccessor (&WifiDefaultAckManager::m_dlMuAckType),
                   MakeEnumChecker (WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE, "DL_MU_BAR_BA_SEQUENCE",
                                    WifiAcknowledgment::DL_MU_TF_MU_BAR, "DL_MU_TF_MU_BAR",
                                    WifiAcknowledgment::DL_MU_AGGREGATE_TF, "DL_MU_AGGREGATE_TF"))
    .AddAttribute ("MaxBlockAckMcs",
                   "The MCS used to send a BlockAck in a TB PPDU is the minimum between "
                   "the MCS used for the PSDU sent in the preceding DL MU PPDU and the "
                   "value of this attribute.",
                   UintegerValue (5),
                   MakeUintegerAccessor (&WifiDefaultAckManager::m_maxMcsForBlockAckInTbPpdu),
                   MakeUintegerChecker<uint8_t> (0, 11))
  ;
  return tid;
}

WifiDefaultAckManager::WifiDefaultAckManager ()
{
  NS_LOG_FUNCTION (this);
}

WifiDefaultAckManager::~WifiDefaultAckManager ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

uint16_t
WifiDefaultAckManager::GetMaxDistFromStartingSeq (Ptr<const WifiMacQueueItem> mpdu,
                                                  const WifiTxParameters& txParams) const
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);

  const WifiMacHeader& hdr = mpdu->GetHeader ();
  Mac48Address receiver = hdr.GetAddr1 ();

  uint8_t tid = hdr.GetQosTid ();
  Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);
  NS_ABORT_MSG_IF (!edca->GetBaAgreementEstablished (receiver, tid),
                   "An established Block Ack agreement is required");

  uint16_t startingSeq = edca->GetBaStartingSequence (receiver, tid);
  uint16_t maxDistFromStartingSeq = (mpdu->GetHeader ().GetSequenceNumber () - startingSeq + SEQNO_SPACE_SIZE) % SEQNO_SPACE_SIZE;
  NS_ABORT_MSG_IF (maxDistFromStartingSeq >= SEQNO_SPACE_HALF_SIZE,
                   "The given QoS data frame is too old");

  const WifiTxParameters::PsduInfo* psduInfo = txParams.GetPsduInfo (receiver);

  if (psduInfo == nullptr || psduInfo->seqNumbers.find (tid) == psduInfo->seqNumbers.end ())
    {
      // there are no aggregated MPDUs (so far)
      return maxDistFromStartingSeq;
    }

  for (const auto& seqNumber : psduInfo->seqNumbers.at (tid))
    {
      if (!QosUtilsIsOldPacket (startingSeq, seqNumber))
        {
          uint16_t currDistToStartingSeq = (seqNumber - startingSeq + SEQNO_SPACE_SIZE) % SEQNO_SPACE_SIZE;

          if (currDistToStartingSeq > maxDistFromStartingSeq)
            {
              maxDistFromStartingSeq = currDistToStartingSeq;
            }
        }
    }

  NS_LOG_DEBUG ("Returning " << maxDistFromStartingSeq);
  return maxDistFromStartingSeq;
}

bool
WifiDefaultAckManager::IsResponseNeeded (Ptr<const WifiMacQueueItem> mpdu,
                                         const WifiTxParameters& txParams) const
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);

  uint8_t tid = mpdu->GetHeader ().GetQosTid ();
  Mac48Address receiver = mpdu->GetHeader ().GetAddr1 ();
  Ptr<QosTxop> edca = m_mac->GetQosTxop (tid);

  // An immediate response (Ack or Block Ack) is needed if any of the following holds:
  // * the maximum distance between the sequence number of an MPDU to transmit
  //   and the starting sequence number of the transmit window is greater than
  //   or equal to the window size multiplied by the BaThreshold
  // * no other frame belonging to this BA agreement is queued (because, in such
  //   a case, a Block Ack is not going to be requested anytime soon)
  // * this is the initial frame of a transmission opportunity and it is not
  //   protected by RTS/CTS (see Annex G.3 of IEEE 802.11-2016)
  if (m_baThreshold > 0
      && GetMaxDistFromStartingSeq (mpdu, txParams) < m_baThreshold * edca->GetBaBufferSize (receiver, tid)
      && (edca->GetWifiMacQueue ()->GetNPackets (tid, receiver)
          - edca->GetBaManager ()->GetNBufferedPackets (receiver, tid) > 1)
      && !(edca->GetTxopLimit ().IsStrictlyPositive ()
           && edca->GetRemainingTxop () == edca->GetTxopLimit ()
           && !(txParams.m_protection && txParams.m_protection->method == WifiProtection::RTS_CTS)))
    {
      return false;
    }

  return true;
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::TryAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                   const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);

  // If the TXVECTOR indicates a DL MU PPDU, call a separate method
  if (txParams.m_txVector.IsDlMu ())
    {
      switch (m_dlMuAckType)
        {
          case WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE:
            return GetAckInfoIfBarBaSequence (mpdu, txParams);
          case WifiAcknowledgment::DL_MU_TF_MU_BAR:
            return GetAckInfoIfTfMuBar (mpdu, txParams);
          case WifiAcknowledgment::DL_MU_AGGREGATE_TF:
            return GetAckInfoIfAggregatedMuBar (mpdu, txParams);
          default:
            NS_ABORT_MSG ("Unknown DL acknowledgment method");
            return nullptr;
        }
    }

  const WifiMacHeader& hdr = mpdu->GetHeader ();
  Mac48Address receiver = hdr.GetAddr1 ();

  // Acknowledgment for TB PPDUs
  if (txParams.m_txVector.IsUlMu ())
    {
      if (hdr.IsQosData () && !hdr.HasData ())
        {
          // QoS Null frame
          WifiNoAck* acknowledgment = nullptr;

          if (txParams.m_acknowledgment)
            {
              NS_ASSERT (txParams.m_acknowledgment->method == WifiAcknowledgment::NONE);
              acknowledgment = static_cast<WifiNoAck*> (txParams.m_acknowledgment.get ());
              acknowledgment = new WifiNoAck (*acknowledgment);
            }
          else
            {
              acknowledgment = new WifiNoAck;
            }
          acknowledgment->SetQosAckPolicy (receiver, hdr.GetQosTid (), WifiMacHeader::NO_ACK);
          return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
        }

      if (txParams.m_acknowledgment)
        {
          NS_ASSERT (txParams.m_acknowledgment->method == WifiAcknowledgment::ACK_AFTER_TB_PPDU);
          return nullptr;
        }

      WifiAckAfterTbPpdu* acknowledgment = new WifiAckAfterTbPpdu;
      if (hdr.IsQosData ())
        {
          acknowledgment->SetQosAckPolicy (receiver, hdr.GetQosTid (), WifiMacHeader::NORMAL_ACK);
        }
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  // if this is a Trigger Frame, call a separate method
  if (hdr.IsTrigger ())
    {
      return TryUlMuTransmission (mpdu, txParams);
    }

  // if the current protection method (if any) is already BLOCK_ACK or BAR_BLOCK_ACK,
  // it will not change by adding an MPDU
  if (txParams.m_acknowledgment
      && (txParams.m_acknowledgment->method == WifiAcknowledgment::BLOCK_ACK
          || txParams.m_acknowledgment->method == WifiAcknowledgment::BAR_BLOCK_ACK))
    {
      return nullptr;
    }

  if (receiver.IsGroup ())
    {
      NS_ABORT_MSG_IF (txParams.GetSize (receiver) > 0,
                       "Unicast frames only can be aggregated");
      WifiNoAck* acknowledgment = new WifiNoAck;
      if (hdr.IsQosData ())
        {
          acknowledgment->SetQosAckPolicy (receiver, hdr.GetQosTid (),
                                           WifiMacHeader::NO_ACK);
        }
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  if ((!hdr.IsQosData ()
       || !m_mac->GetQosTxop (hdr.GetQosTid ())->GetBaAgreementEstablished (receiver, hdr.GetQosTid ()))
      && !hdr.IsBlockAckReq ())
    {
      NS_LOG_DEBUG ("Non-QoS data frame or Block Ack agreement not established, request Normal Ack");
      WifiNormalAck* acknowledgment = new WifiNormalAck;
      acknowledgment->ackTxVector = m_mac->GetWifiRemoteStationManager ()->GetAckTxVector (receiver, txParams.m_txVector);
      if (hdr.IsQosData ())
        {
          acknowledgment->SetQosAckPolicy (receiver, hdr.GetQosTid (), WifiMacHeader::NORMAL_ACK);
        }
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  // we get here if mpdu is a QoS data frame related to an established Block Ack agreement
  // or mpdu is a BlockAckReq frame
  if (!hdr.IsBlockAckReq () && !IsResponseNeeded (mpdu, txParams))
    {
      NS_LOG_DEBUG ("A response is not needed: no ack for now, use Block Ack policy");
      if (txParams.m_acknowledgment
          && txParams.m_acknowledgment->method == WifiAcknowledgment::NONE)
        {
          // no change if the ack method is already NONE
          return nullptr;
        }

      WifiNoAck* acknowledgment = new WifiNoAck;
      if (hdr.IsQosData ())
        {
          acknowledgment->SetQosAckPolicy (receiver, hdr.GetQosTid (),
                                           WifiMacHeader::BLOCK_ACK);
        }
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  // we get here if a response is needed
  uint8_t tid = GetTid (mpdu->GetPacket (), hdr);
  if (!hdr.IsBlockAckReq ()
      && txParams.GetSize (receiver) == 0
      && hdr.GetSequenceNumber ()
         == m_mac->GetQosTxop (tid)->GetBaStartingSequence (receiver, tid))
    {
      NS_LOG_DEBUG ("Sending a single MPDU, no previous frame to ack: request Normal Ack");
      WifiNormalAck* acknowledgment = new WifiNormalAck;
      acknowledgment->ackTxVector = m_mac->GetWifiRemoteStationManager ()->GetAckTxVector (receiver, txParams.m_txVector);
      acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::NORMAL_ACK);
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  // we get here if multiple MPDUs are being/have been sent
  if (!hdr.IsBlockAckReq ()
      && (txParams.GetSize (receiver) == 0 || m_useExplicitBar))
    {
      // in case of single MPDU, there are previous unacknowledged frames, thus
      // we cannot use Implicit Block Ack Request policy, otherwise we get a
      // normal ack as response
      NS_LOG_DEBUG ("Request to schedule a Block Ack Request");

      WifiBarBlockAck* acknowledgment = new WifiBarBlockAck;
      acknowledgment->blockAckReqTxVector = m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (receiver, txParams.m_txVector);
      acknowledgment->blockAckTxVector = acknowledgment->blockAckReqTxVector;
      acknowledgment->barType = m_mac->GetQosTxop (tid)->GetBlockAckReqType (receiver, tid);
      acknowledgment->baType = m_mac->GetQosTxop (tid)->GetBlockAckType (receiver, tid);
      acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::BLOCK_ACK);
      return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
    }

  NS_LOG_DEBUG ("A-MPDU using Implicit Block Ack Request policy or BlockAckReq, request Block Ack");
  WifiBlockAck* acknowledgment = new WifiBlockAck;
  acknowledgment->blockAckTxVector = m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (receiver, txParams.m_txVector);
  acknowledgment->baType = m_mac->GetQosTxop (tid)->GetBlockAckType (receiver, tid);
  acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::NORMAL_ACK);
  return std::unique_ptr<WifiAcknowledgment> (acknowledgment);
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                         const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *msdu << &txParams);

  // Aggregating an MSDU does not change the acknowledgment method
  return nullptr;
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::GetAckInfoIfBarBaSequence (Ptr<const WifiMacQueueItem> mpdu,
                                                  const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);
  NS_ASSERT (txParams.m_txVector.IsDlMu ());
  NS_ASSERT (m_dlMuAckType == WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE);

  const WifiMacHeader& hdr = mpdu->GetHeader ();
  Mac48Address receiver = hdr.GetAddr1 ();

  const WifiTxParameters::PsduInfo* psduInfo = txParams.GetPsduInfo (receiver);

  NS_ABORT_MSG_IF (!hdr.IsQosData (),
                    "QoS data frames only can be aggregated when transmitting a "
                    "DL MU PPDU acknowledged via a sequence of BAR and BA frames");
  uint8_t tid = hdr.GetQosTid ();
  Ptr<QosTxop> edca = m_mac->GetQosTxop (QosUtilsMapTidToAc (tid));

  NS_ASSERT (!txParams.m_acknowledgment
             || txParams.m_acknowledgment->method == WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE);

  WifiDlMuBarBaSequence* acknowledgment = nullptr;
  if (txParams.m_acknowledgment)
    {
      acknowledgment = static_cast<WifiDlMuBarBaSequence*> (txParams.m_acknowledgment.get ());
    }

  if (psduInfo != nullptr)
    {
      // an MPDU addressed to the same receiver has been already added
      NS_ASSERT (acknowledgment != nullptr);

      if ((acknowledgment->stationsSendBlockAckReqTo.find (receiver)
            != acknowledgment->stationsSendBlockAckReqTo.end ())
          || (acknowledgment->stationsReplyingWithBlockAck.find (receiver)
              != acknowledgment->stationsReplyingWithBlockAck.end ()))
        {
          // the receiver either is already listed among the stations that will
          // receive a BlockAckReq frame or is the station that will immediately
          // respond with a BlockAck frame, hence no change is needed
          return nullptr;
        }

      // the receiver was scheduled for responding immediately with a Normal Ack.
      // Given that we are adding an MPDU, the receiver must be scheduled for
      // responding immediately with a Block Ack
      NS_ASSERT (acknowledgment->stationsReplyingWithNormalAck.size () == 1
                  && acknowledgment->stationsReplyingWithNormalAck.begin ()->first == receiver);


      // acknowledgment points to the m_acknowledgment member of txParams, which is
      // passed as const reference because it must not be modified. Therefore, we
      // make a copy of the object pointed to by acknowledgment and make changes to
      // the copy
      acknowledgment = new WifiDlMuBarBaSequence (*acknowledgment);
      acknowledgment->stationsReplyingWithNormalAck.clear ();

      acknowledgment->stationsReplyingWithBlockAck.emplace
        (receiver,
          WifiDlMuBarBaSequence::BlockAckInfo
          {
            m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (receiver, txParams.m_txVector),
            edca->GetBlockAckType (receiver, tid)
          });
      return std::unique_ptr<WifiDlMuBarBaSequence> (acknowledgment);
    }

  // we get here if this is the first MPDU for this receiver
  if (edca->GetBaManager ()->GetBar (true, tid, receiver) != 0
      || (acknowledgment != nullptr
          && (!acknowledgment->stationsReplyingWithNormalAck.empty ()
              || !acknowledgment->stationsReplyingWithBlockAck.empty ())))
    {
      // there is a pending BlockAckReq for this receiver or another receiver
      // was selected for immediate response.
      // Add this receiver to the list of stations receiving a BlockAckReq.
      if (acknowledgment != nullptr)
        {
          // txParams.m_acknowledgment points to an existing WifiDlMuBarBaSequence object.
          // We have to return a copy of this object including the needed changes
          acknowledgment = new WifiDlMuBarBaSequence (*acknowledgment);
        }
      else
        {
          // we have to create a new WifiDlMuBarBaSequence object
          acknowledgment = new WifiDlMuBarBaSequence;
        }

      NS_LOG_DEBUG ("Adding STA " << receiver << " to the list of stations receiving a BlockAckReq");
      acknowledgment->stationsSendBlockAckReqTo.emplace
        (receiver,
          WifiDlMuBarBaSequence::BlockAckReqInfo
          {
            m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (receiver, txParams.m_txVector),
            edca->GetBlockAckReqType (receiver, tid),
            m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (receiver, txParams.m_txVector),
            edca->GetBlockAckType (receiver, tid)
          });

      acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::BLOCK_ACK);
      return std::unique_ptr<WifiDlMuBarBaSequence> (acknowledgment);
    }

  // Add the receiver as the station that will immediately reply with a Normal Ack
  if (acknowledgment != nullptr)
    {
      // txParams.m_acknowledgment points to an existing WifiDlMuBarBaSequence object.
      // We have to return a copy of this object including the needed changes
      acknowledgment = new WifiDlMuBarBaSequence (*acknowledgment);
    }
  else
    {
      // we have to create a new WifiDlMuBarBaSequence object
      acknowledgment = new WifiDlMuBarBaSequence;
    }

  NS_LOG_DEBUG ("Adding STA " << receiver << " as the station that will immediately reply with a Normal Ack");
  acknowledgment->stationsReplyingWithNormalAck.emplace
    (receiver,
      WifiDlMuBarBaSequence::AckInfo
      {
        m_mac->GetWifiRemoteStationManager ()->GetAckTxVector (receiver, txParams.m_txVector)
      });

  acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::NORMAL_ACK);
  return std::unique_ptr<WifiDlMuBarBaSequence> (acknowledgment);
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::GetAckInfoIfTfMuBar (Ptr<const WifiMacQueueItem> mpdu,
                                            const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);
  NS_ASSERT (txParams.m_txVector.IsDlMu ());
  NS_ASSERT (m_dlMuAckType == WifiAcknowledgment::DL_MU_TF_MU_BAR);

  const WifiMacHeader& hdr = mpdu->GetHeader ();
  Mac48Address receiver = hdr.GetAddr1 ();

  const WifiTxParameters::PsduInfo* psduInfo = txParams.GetPsduInfo (receiver);

  NS_ASSERT (!txParams.m_acknowledgment
             || txParams.m_acknowledgment->method == WifiAcknowledgment::DL_MU_TF_MU_BAR);

  WifiDlMuTfMuBar* acknowledgment = nullptr;
  if (txParams.m_acknowledgment)
    {
      acknowledgment = static_cast<WifiDlMuTfMuBar*> (txParams.m_acknowledgment.get ());
    }

  if (psduInfo == nullptr)
    {
      // we get here if this is the first MPDU for this receiver.
      Ptr<ApWifiMac> apMac = DynamicCast<ApWifiMac> (m_mac);
      NS_ABORT_MSG_IF (apMac == 0, "HE APs only can send DL MU PPDUs");
      uint16_t staId = apMac->GetAssociationId (receiver);

      NS_ABORT_MSG_IF (!hdr.IsQosData (),
                      "QoS data frames only can be aggregated when transmitting a "
                      "DL MU PPDU acknowledged via a MU-BAR sent as SU frame");
      uint8_t tid = hdr.GetQosTid ();

      // Add the receiver to the list of stations that will reply with a Block Ack
      if (acknowledgment != nullptr)
        {
          // txParams.m_acknowledgment points to an existing WifiDlMuTfMuBar object.
          // We have to return a copy of this object including the needed changes
          acknowledgment = new WifiDlMuTfMuBar (*acknowledgment);
        }
      else
        {
          // we have to create a new WifiDlMuTfMuBar object
          acknowledgment = new WifiDlMuTfMuBar;
        }

      // determine the TX vector used to send the BlockAck frame
      WifiTxVector blockAckTxVector;
      blockAckTxVector.SetPreambleType (WifiPreamble::WIFI_PREAMBLE_HE_TB);
      blockAckTxVector.SetChannelWidth (txParams.m_txVector.GetChannelWidth ());
      blockAckTxVector.SetGuardInterval (txParams.m_txVector.GetGuardInterval ());
      const auto& userInfo = txParams.m_txVector.GetHeMuUserInfo (staId);
      blockAckTxVector.SetHeMuUserInfo (staId, {userInfo.ru,
                                                HePhy::GetHeMcs (std::min (userInfo.mcs.GetMcsValue (),
                                                                           m_maxMcsForBlockAckInTbPpdu)),
                                                userInfo.nss});

      NS_LOG_DEBUG ("Adding STA " << receiver << " to the list of stations that will be solicited by the MU-BAR");
      Ptr<QosTxop> edca = m_mac->GetQosTxop (QosUtilsMapTidToAc (tid));
      acknowledgment->stationsReplyingWithBlockAck.emplace
        (receiver,
          WifiDlMuTfMuBar::BlockAckInfo
          {
            edca->GetBaManager ()->GetBlockAckReqHeader (receiver, tid),
            blockAckTxVector,
            edca->GetBlockAckType (receiver, tid)
          });

      acknowledgment->barTypes.push_back (edca->GetBlockAckReqType (receiver, tid));
      acknowledgment->muBarTxVector = m_mac->GetWifiRemoteStationManager ()->GetRtsTxVector (receiver);
      acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::BLOCK_ACK);
      return std::unique_ptr<WifiDlMuTfMuBar> (acknowledgment);
    }

  // an MPDU addressed to the same receiver has been already added
  NS_ASSERT (acknowledgment != nullptr);
  NS_ABORT_MSG_IF (!hdr.IsQosData (),
                    "QoS data frames only can be aggregated when transmitting a DL MU PPDU");

  // no change is needed
  return nullptr;
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::GetAckInfoIfAggregatedMuBar (Ptr<const WifiMacQueueItem> mpdu,
                                                    const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);
  NS_ASSERT (txParams.m_txVector.IsDlMu ());
  NS_ASSERT (m_dlMuAckType == WifiAcknowledgment::DL_MU_AGGREGATE_TF);

  const WifiMacHeader& hdr = mpdu->GetHeader ();
  Mac48Address receiver = hdr.GetAddr1 ();

  const WifiTxParameters::PsduInfo* psduInfo = txParams.GetPsduInfo (receiver);

  NS_ASSERT (!txParams.m_acknowledgment
             || txParams.m_acknowledgment->method == WifiAcknowledgment::DL_MU_AGGREGATE_TF);

  WifiDlMuAggregateTf* acknowledgment = nullptr;
  if (txParams.m_acknowledgment)
    {
      acknowledgment = static_cast<WifiDlMuAggregateTf*> (txParams.m_acknowledgment.get ());
    }

  if (psduInfo == nullptr)
    {
      // we get here if this is the first MPDU for this receiver.
      Ptr<ApWifiMac> apMac = DynamicCast<ApWifiMac> (m_mac);
      NS_ABORT_MSG_IF (apMac == 0, "HE APs only can send DL MU PPDUs");
      uint16_t staId = apMac->GetAssociationId (receiver);

      NS_ABORT_MSG_IF (!hdr.IsQosData (),
                      "QoS data frames only can be aggregated when transmitting a "
                      "DL MU PPDU acknowledged via a sequence of BAR and BA frames");
      uint8_t tid = hdr.GetQosTid ();

      // Add the receiver to the list of stations that will reply with a Block Ack
      if (acknowledgment != nullptr)
        {
          // txParams.m_acknowledgment points to an existing WifiDlMuAggregateTf object.
          // We have to return a copy of this object including the needed changes
          acknowledgment = new WifiDlMuAggregateTf (*acknowledgment);
        }
      else
        {
          // we have to create a new WifiDlMuAggregateTf object
          acknowledgment = new WifiDlMuAggregateTf;
        }

      // determine the TX vector used to send the BlockAck frame
      WifiTxVector blockAckTxVector;
      blockAckTxVector.SetPreambleType (WifiPreamble::WIFI_PREAMBLE_HE_TB);
      blockAckTxVector.SetChannelWidth (txParams.m_txVector.GetChannelWidth ());
      blockAckTxVector.SetGuardInterval (txParams.m_txVector.GetGuardInterval ());
      const auto& userInfo = txParams.m_txVector.GetHeMuUserInfo (staId);
      blockAckTxVector.SetHeMuUserInfo (staId, {userInfo.ru,
                                                HePhy::GetHeMcs (std::min (userInfo.mcs.GetMcsValue (),
                                                                           m_maxMcsForBlockAckInTbPpdu)),
                                                userInfo.nss});

      NS_LOG_DEBUG ("Adding STA " << receiver << " to the list of stations that will reply with a Block Ack");
      Ptr<QosTxop> edca = m_mac->GetQosTxop (QosUtilsMapTidToAc (tid));
      acknowledgment->stationsReplyingWithBlockAck.emplace
        (receiver,
          WifiDlMuAggregateTf::BlockAckInfo
          {
            GetMuBarSize ({edca->GetBlockAckReqType (receiver, tid)}),
            edca->GetBaManager ()->GetBlockAckReqHeader (receiver, tid),
            blockAckTxVector,
            edca->GetBlockAckType (receiver, tid)
          });

      acknowledgment->SetQosAckPolicy (receiver, tid, WifiMacHeader::NO_EXPLICIT_ACK);
      return std::unique_ptr<WifiDlMuAggregateTf> (acknowledgment);
    }

  // an MPDU addressed to the same receiver has been already added
  NS_ASSERT (acknowledgment != nullptr);
  NS_ABORT_MSG_IF (!hdr.IsQosData (),
                   "QoS data and MU-BAR Trigger frames only can be aggregated when transmitting a DL MU PPDU");

  // no change is needed
  return nullptr;
}

std::unique_ptr<WifiAcknowledgment>
WifiDefaultAckManager::TryUlMuTransmission (Ptr<const WifiMacQueueItem> mpdu,
                                            const WifiTxParameters& txParams)
{
  NS_LOG_FUNCTION (this << *mpdu << &txParams);
  NS_ASSERT (mpdu->GetHeader ().IsTrigger ());

  Ptr<ApWifiMac> apMac = DynamicCast<ApWifiMac> (m_mac);
  NS_ABORT_MSG_IF (apMac == nullptr, "HE APs only can send Trigger Frames");

  Ptr<HeFrameExchangeManager> heFem = DynamicCast<HeFrameExchangeManager> (m_mac->GetFrameExchangeManager ());
  NS_ABORT_MSG_IF (heFem == nullptr, "HE APs only can send Trigger Frames");

  CtrlTriggerHeader trigger;
  mpdu->GetPacket ()->PeekHeader (trigger);

  if (trigger.IsBasic ())
    {
      // the only supported ack method for now is through a multi-STA BlockAck frame
      WifiUlMuMultiStaBa* acknowledgment = new WifiUlMuMultiStaBa;

      for (const auto& userInfo : trigger)
        {
          uint16_t aid12 = userInfo.GetAid12 ();

          if (aid12 == 2046)
            {
              NS_LOG_INFO ("Unallocated RU");
              continue;
            }
          NS_ABORT_MSG_IF (aid12 == 0 || aid12 > 2007, "Allocation of RA-RUs is not supported");

          NS_ASSERT (apMac->GetStaList ().find (aid12) != apMac->GetStaList ().end ());
          Mac48Address staAddress = apMac->GetStaList ().find (aid12)->second;

          // find a TID for which a BA agreement exists with the given originator
          uint8_t tid = 0;
          while (tid < 8 && !heFem->GetBaAgreementEstablished (staAddress, tid))
            {
              tid++;
            }
          NS_ASSERT_MSG (tid < 8, "No Block Ack agreement established with originator " << staAddress);

          std::size_t index = acknowledgment->baType.m_bitmapLen.size ();
          acknowledgment->stationsReceivingMultiStaBa.emplace (std::make_pair (staAddress, tid), index);

          // we assume the Block Acknowledgment context is used for the multi-STA BlockAck frame
          // (since it requires the longest TX time due to the presence of a bitmap)
          acknowledgment->baType.m_bitmapLen.push_back (heFem->GetBlockAckType (staAddress, tid).m_bitmapLen.at (0));
        }

      uint16_t staId = trigger.begin ()->GetAid12 ();
      acknowledgment->tbPpduTxVector = trigger.GetHeTbTxVector (staId);
      acknowledgment->multiStaBaTxVector = m_mac->GetWifiRemoteStationManager ()->GetBlockAckTxVector (apMac->GetStaList ().find (staId)->second,
                                                                                                       acknowledgment->tbPpduTxVector);
      return std::unique_ptr<WifiUlMuMultiStaBa> (acknowledgment);
    }
  else if (trigger.IsBsrp ())
    {
      return std::unique_ptr<WifiAcknowledgment> (new WifiNoAck);
    }

  return nullptr;
}

} //namespace ns3
