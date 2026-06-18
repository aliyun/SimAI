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

#ifndef HT_FRAME_EXCHANGE_MANAGER_H
#define HT_FRAME_EXCHANGE_MANAGER_H

#include "ns3/qos-frame-exchange-manager.h"
#include "ns3/mpdu-aggregator.h"
#include "ns3/msdu-aggregator.h"
#include "ns3/wifi-psdu.h"

class AmpduAggregationTest;
class TwoLevelAggregationTest;
class HeAggregationTest;

namespace ns3 {

class MgtAddBaResponseHeader;
class RecipientBlockAckAgreement;

/**
 * \ingroup wifi
 *
 * HtFrameExchangeManager handles the frame exchange sequences
 * for HT stations.
 */
class HtFrameExchangeManager : public QosFrameExchangeManager
{
public:
  /// allow AmpduAggregationTest class access
  friend class ::AmpduAggregationTest;
  /// allow TwoLevelAggregationTest class access
  friend class ::TwoLevelAggregationTest;
  /// allow HeAggregationTest class access
  friend class ::HeAggregationTest;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  HtFrameExchangeManager ();
  virtual ~HtFrameExchangeManager ();

  bool StartFrameExchange (Ptr<QosTxop> edca, Time availableTime, bool initialFrame) override;
  void SetWifiMac (const Ptr<WifiMac> mac) override;
  void CalculateAcknowledgmentTime (WifiAcknowledgment* acknowledgment) const override;

  /**
   * Returns the aggregator used to construct A-MSDU subframes.
   *
   * \return the aggregator used to construct A-MSDU subframes.
   */
  Ptr<MsduAggregator> GetMsduAggregator (void) const;
  /**
   * Returns the aggregator used to construct A-MPDU subframes.
   *
   * \return the aggregator used to construct A-MPDU subframes.
   */
  Ptr<MpduAggregator> GetMpduAggregator (void) const;

  /**
   * Check if the PSDU obtained by aggregating the given MPDU to the PSDU specified
   * by the given TX parameters meets the constraints on the maximum A-MPDU size
   * and its transmission time does not exceed the given PPDU duration limit (if
   * different than Time::Min()).
   *
   * \param mpdu the given MPDU
   * \param txParams the TX parameters
   * \param ppduDurationLimit the constraint on the PPDU transmission time
   * \return true if the size and time constraints are met, false otherwise
   */
  bool IsWithinLimitsIfAddMpdu (Ptr<const WifiMacQueueItem> mpdu,
                                const WifiTxParameters& txParams,
                                Time ppduDurationLimit) const override;

  /**
   * Check whether an A-MPDU of the given size meets the constraint on the maximum
   * size for A-MPDUs sent to the given receiver, belonging to the given TID and
   * transmitted using the given modulation class.
   *
   * \param ampduSize the size in bytes of the A-MPDU.
   * \param receiver the address of the station that is the receiver of the A-MPDU
   * \param tid the TID of the A-MPDU
   * \param modulation the modulation class used to transmit the A-MPDU
   * \return true if the constraint on the max A-MPDU size is met.
   */
  virtual bool IsWithinAmpduSizeLimit (uint32_t ampduSize, Mac48Address receiver, uint8_t tid,
                                       WifiModulationClass modulation) const;

  /**
   * Check if aggregating an MSDU to the current MPDU (as specified by the given
   * TX parameters) does not violate the size and time constraints, while taking
   * into account the possibly updated protection and acknowledgment methods. If
   * size and time constraints are met, the TX parameters are modified with the
   * updated protection and acknowledgment methods.
   *
   * \param msdu the given MSDU
   * \param txParams the TX parameters
   * \param availableTime the constraint on the TX time of the PSDU, if different
   *        than Time::Min()
   * \return true if aggregating an MSDU to the current PSDU does not violate the
   *         size and time constraints
   */
  virtual bool TryAggregateMsdu (Ptr<const WifiMacQueueItem> msdu, WifiTxParameters& txParams,
                                 Time availableTime) const;

  /**
   * Check if the PSDU obtained by aggregating the given MSDU to the PSDU specified
   * by the given TX parameters meets the constraints on the maximum A-MSDU size
   * and its transmission time does not exceed the given PPDU duration limit (if
   * different than Time::Min()).
   *
   * \param msdu the given MSDU
   * \param txParams the TX parameters
   * \param ppduDurationLimit the constraint on the PPDU transmission time
   * \return true if the size and time constraints are met, false otherwise
   */
  virtual bool IsWithinLimitsIfAggregateMsdu (Ptr<const WifiMacQueueItem> msdu,
                                              const WifiTxParameters& txParams,
                                              Time ppduDurationLimit) const;

  /**
   * \param respHdr Add block ack response from originator (action
   *        frame).
   * \param originator Address of peer station involved in block ack
   *        mechanism.
   * \param startingSeq Sequence number of the first MPDU of all
   *        packets for which block ack was negotiated.
   *
   * This function is typically invoked only by ns3::WifiMac
   * when the STA (which may be non-AP in ESS, or in an IBSS) has
   * received an ADDBA Request frame and is transmitting an ADDBA
   * Response frame. At this point the frame exchange manager must
   * allocate buffers to collect all correctly received packets belonging
   * to the category for which block ack was negotiated.
   */
  void CreateBlockAckAgreement (const MgtAddBaResponseHeader *respHdr,
                                Mac48Address originator, uint16_t startingSeq);
  /**
   * Destroy a Block Ack agreement.
   *
   * \param originator the originator MAC address
   * \param tid the TID associated with the Block Ack agreement
   */
  void DestroyBlockAckAgreement (Mac48Address originator, uint8_t tid);
  /**
   * This method can be called to accept a received ADDBA Request. An
   * ADDBA Response will be constructed and queued for transmission.
   *
   * \param reqHdr a pointer to the received ADDBA Request header.
   * \param originator the MAC address of the originator.
   */
  void SendAddBaResponse (const MgtAddBaRequestHeader *reqHdr,
                          Mac48Address originator);
  /**
   * Get the maximum supported buffer size for a Block Ack agreement. This value
   * is typically included in ADDBA Response frames.
   *
   * \return the maximum supported buffer size for a Block Ack agreement
   */
  virtual uint16_t GetSupportedBaBufferSize (void) const;

  /**
   * Return true if a Block Ack agreement has been established with the given
   * originator for the given TID.
   *
   * \param originator the MAC address of the given originator
   * \param tid the Traffic ID
   * \return true if a Block Ack agreement has been established with the given
   *         originator for the given TID
   */
  bool GetBaAgreementEstablished (Mac48Address originator, uint8_t tid) const;

  /**
   * Get the type of BlockAck frames sent to the given originator.
   *
   * \param originator the MAC address of the given originator
   * \param tid the Traffic ID
   * \return the type of BlockAck frames sent to the given originator
   */
  BlockAckType GetBlockAckType (Mac48Address originator, uint8_t tid) const;

  /**
   * Sends DELBA frame to cancel a block ack agreement with STA
   * addressed by <i>addr</i> for TID <i>tid</i>.
   *
   * \param addr address of the recipient.
   * \param tid traffic ID.
   * \param byOriginator flag to indicate whether this is set by the originator.
   */
  void SendDelbaFrame (Mac48Address addr, uint8_t tid, bool byOriginator);

protected:
  void DoDispose () override;

  void ReceiveMpdu (Ptr<WifiMacQueueItem> mpdu, RxSignalInfo rxSignalInfo,
                    const WifiTxVector& txVector, bool inAmpdu) override;
  void EndReceiveAmpdu (Ptr<const WifiPsdu> psdu, const RxSignalInfo& rxSignalInfo,
                        const WifiTxVector& txVector, const std::vector<bool>& perMpduStatus) override;
  void NotifyReceivedNormalAck (Ptr<WifiMacQueueItem> mpdu) override;
  void NotifyPacketDiscarded (Ptr<const WifiMacQueueItem> mpdu) override;
  void RetransmitMpduAfterMissedAck (Ptr<WifiMacQueueItem> mpdu) const override;
  void ReleaseSequenceNumber (Ptr<WifiMacQueueItem> mpdu) const override;
  void ForwardMpduDown (Ptr<WifiMacQueueItem> mpdu, WifiTxVector& txVector) override;
  void CtsTimeout (Ptr<WifiMacQueueItem> rts, const WifiTxVector& txVector) override;
  void TransmissionSucceeded (void) override;

  /**
   * Get a PSDU containing the given MPDU
   *
   * \param mpdu the given MPDU
   * \param txVector the TXVECTOR to use to send the MPDU
   * \return a PSDU containing the given MPDU
   */
  virtual Ptr<WifiPsdu> GetWifiPsdu (Ptr<WifiMacQueueItem> mpdu, const WifiTxVector& txVector) const;

  /**
   * Get the Block Ack Manager handling the given TID.
   *
   * \param tid the given TID
   * \return the Block Ack Manager handling the given TID
   */
  Ptr<BlockAckManager> GetBaManager (uint8_t tid) const;

  /**
   * Compute how to set the Duration/ID field of PSDUs that do not include fragments.
   *
   * \param txDuration the duration of the PSDU transmission
   * \param txParams the TX parameters used to send the PSDU
   * \return the value for the Duration/ID field
   */
  virtual Time GetPsduDurationId (Time txDuration, const WifiTxParameters& txParams) const;

  /**
   * Send a PSDU (A-MPDU or BlockAckReq frame) requesting a BlockAck frame or
   * a BlockAckReq frame followed by a BlockAck frame for the acknowledgment.
   * Note that <i>txParams</i> is moved to m_txParams and hence is left in an
   * undefined state.
   *
   * \param psdu the PSDU to send
   * \param txParams the TX parameters to use to transmit the PSDU
   */
  void SendPsduWithProtection (Ptr<WifiPsdu> psdu, WifiTxParameters& txParams);

  /**
   * Notify the transmission of the given PSDU to the EDCAF associated with the
   * AC the PSDU belongs to.
   *
   * \param psdu the PSDU to transmit
   */
  virtual void NotifyTxToEdca (Ptr<const WifiPsdu> psdu) const;

  /**
   * Forward a PSDU down to the PHY layer. Also, notify the Block Ack Manager
   * of the transmission of the constituent MPDUs.
   *
   * \param psdu the PSDU to forward down
   * \param txVector the TXVECTOR used to transmit the MPDU
   */
  virtual void ForwardPsduDown (Ptr<const WifiPsdu> psdu, WifiTxVector& txVector);

  /**
   * Dequeue the MPDUs of the given PSDU from the queue in which they are stored.
   *
   * \param psdu the given PSDU
   */
  void DequeuePsdu (Ptr<const WifiPsdu> psdu);

  /**
   * If the Block Ack Manager associated with the given EDCA has a BlockAckReq frame
   * to transmit (the duration of which plus the response fits within the given
   * available time, if the latter is not Time::Min() and this is not the initial
   * frame of a TXOP), transmit the frame and return true. Otherwise, return false.
   *
   * \param edca the EDCAF which has been granted the opportunity to transmit
   * \param availableTime the amount of time allowed for the frame exchange. Equals
   *                      Time::Min() in case the TXOP limit is null
   * \param initialFrame true if the frame being transmitted is the initial frame
   *                     of the TXOP. This is used to determine whether the TXOP
   *                     limit can be exceeded
   * \return true if frame is transmitted, false otherwise
   */
  virtual bool SendMpduFromBaManager (Ptr<QosTxop> edca, Time availableTime, bool initialFrame);

  /**
   * Given a non-broadcast QoS data frame, prepare the PSDU to transmit by attempting
   * A-MSDU and A-MPDU aggregation (if enabled), while making sure that the frame
   * exchange (possibly including protection and acknowledgment) is completed within
   * the given available time.
   *
   * \param peekedItem the given non-broadcast QoS data frame
   * \param availableTime the amount of time allowed for the frame exchange. Equals
   *                      Time::Min() in case the TXOP limit is null
   * \param initialFrame true if the frame being transmitted is the initial frame
   *                     of the TXOP. This is used to determine whether the TXOP
   *                     limit can be exceeded
   * \return true if frame is transmitted, false otherwise
   */
  virtual bool SendDataFrame (Ptr<const WifiMacQueueItem> peekedItem,
                              Time availableTime, bool initialFrame);

  /**
   * A Block Ack agreement needs to be established with the given recipient for the
   * given TID if it does not already exist (or exists and is in state RESET) and:
   *
   * - the number of packets in the queue reaches the BlockAckThreshold value OR
   * - MPDU aggregation is enabled and there is more than one packet in the queue OR
   * - the station is a VHT station
   *
   * \param recipient address of the recipient.
   * \param tid traffic ID.
   * \return true if a Block Ack agreement needs to be established, false otherwise.
   */
  virtual bool NeedSetupBlockAck (Mac48Address recipient, uint8_t tid);

  /**
   * Sends an ADDBA Request to establish a block ack agreement with STA
   * addressed by <i>recipient</i> for TID <i>tid</i>.
   *
   * \param recipient address of the recipient.
   * \param tid traffic ID.
   * \param startingSeq the BA agreement starting sequence number
   * \param timeout timeout value.
   * \param immediateBAck flag to indicate whether immediate BlockAck is used.
   */
  void SendAddBaRequest (Mac48Address recipient, uint8_t tid, uint16_t startingSeq,
                         uint16_t timeout, bool immediateBAck);

  /**
   * Create a BlockAck frame with header equal to <i>blockAck</i> and start its transmission.
   *
   * \param agreement the agreement the Block Ack response belongs to
   * \param durationId the Duration/ID of the frame soliciting this Block Ack response
   * \param blockAckTxVector the transmit vector for the Block Ack response
   * \param rxSnr the receive SNR
   */
  void SendBlockAck (const RecipientBlockAckAgreement& agreement, Time durationId,
                     WifiTxVector& blockAckTxVector, double rxSnr);

  /**
   * Called when the BlockAck timeout expires.
   *
   * \param psdu the PSDU (BlockAckReq or A-MPDU) that solicited a BlockAck response
   * \param txVector the TXVECTOR used to send the PSDU that solicited a BlockAck response
   */
  virtual void BlockAckTimeout (Ptr<WifiPsdu> psdu, const WifiTxVector& txVector);

  /**
   * Take necessary actions when a BlockAck is missed, such as scheduling a
   * BlockAckReq frame or the retransmission of the unacknowledged frames.
   *
   * \param psdu the PSDU (BlockAckReq or A-MPDU) that solicited a BlockAck response
   * \param txVector the TXVECTOR used to send the PSDU that solicited a BlockAck response
   * \param[out] resetCw true if we shall stop retransmitting the PSDU
   */
  virtual void MissedBlockAck (Ptr<WifiPsdu> psdu, const WifiTxVector& txVector, bool& resetCw);

  /// agreement key typedef (MAC address and TID)
  typedef std::pair<Mac48Address, uint8_t> AgreementKey;

  std::map<AgreementKey, RecipientBlockAckAgreement> m_agreements; //!< agreements
  Ptr<MsduAggregator> m_msduAggregator;                            //!< A-MSDU aggregator
  Ptr<MpduAggregator> m_mpduAggregator;                            //!< A-MPDU aggregator

private:
  /**
   * Send the current PSDU, which can be acknowledged by a BlockAck frame or
   * followed by a BlockAckReq frame and a BlockAck frame.
   */
  void SendPsdu (void);

  Ptr<WifiPsdu> m_psdu;                               //!< the A-MPDU being transmitted
  WifiTxParameters m_txParams;                        //!< the TX parameters for the current frame
};

} //namespace ns3

#endif /* HT_FRAME_EXCHANGE_MANAGER_H */
