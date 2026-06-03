/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#ifndef LTE_RLC_AM_H
#define LTE_RLC_AM_H

#include <ns3/event-id.h>
#include <ns3/lte-rlc-sequence-number.h>
#include <ns3/lte-rlc.h>

#include <vector>
#include <map>

namespace ns3 {

/**
 * LTE RLC Acknowledged Mode (AM), see 3GPP TS 36.322
 */
class LteRlcAm : public LteRlc
{
public:
  LteRlcAm ();
  virtual ~LteRlcAm ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  /**
   * RLC SAP
   *
   * \param p packet
   */
  virtual void DoTransmitPdcpPdu (Ptr<Packet> p);

  /**
   * MAC SAP
   *
   * \param txOpParams the LteMacSapUser::TxOpportunityParameters
   */
  virtual void DoNotifyTxOpportunity (LteMacSapUser::TxOpportunityParameters txOpParams);
    /**
   * Notify HARQ delivery failure
   */
  virtual void DoNotifyHarqDeliveryFailure ();
  virtual void DoReceivePdu (LteMacSapUser::ReceivePduParameters rxPduParams);

private:
  /**
   * This method will schedule a timeout at WaitReplyTimeout interval
   * in the future, unless a timer is already running for the cache,
   * in which case this method does nothing.
   */
  void ExpireReorderingTimer (void);
  /// Expire poll retransmitter 
  void ExpirePollRetransmitTimer (void);
  /// Expire RBS timer
  void ExpireRbsTimer (void);

  /** 
   * method called when the T_status_prohibit timer expires
   * 
   */
  void ExpireStatusProhibitTimer (void);

  /** 
   * method called when the T_status_prohibit timer expires
   * 
   * \param seqNumber SequenceNumber10
   * \returns true is inside receivign window
   */
  bool IsInsideReceivingWindow (SequenceNumber10 seqNumber);
// 
//   void ReassembleOutsideWindow (void);
//   void ReassembleSnLessThan (uint16_t seqNumber);
// 

  /** 
   * Reassemble and deliver
   * 
   * \param packet the packet
   */
  void ReassembleAndDeliver (Ptr<Packet> packet);

  /** 
   * Report buffer status
   */
  void DoReportBufferStatus ();

private:
  /**
   * \brief Store an incoming (from layer above us) PDU, waiting to transmit it
   */
  struct TxPdu
  {
    /**
     * \brief TxPdu default constructor
     * \param pdu the PDU
     * \param time the arrival time
     */
    TxPdu (const Ptr<Packet> &pdu, const Time &time) :
      m_pdu (pdu),
      m_waitingSince (time)
    { }

    TxPdu () = delete;

    Ptr<Packet> m_pdu;           ///< PDU
    Time        m_waitingSince;  ///< Layer arrival time
  };

  std::vector < TxPdu > m_txonBuffer; ///< Transmission buffer

  /// RetxPdu structure
  struct RetxPdu
  {
    Ptr<Packet> m_pdu; ///< PDU
    uint16_t    m_retxCount; ///< retransmit count
    Time        m_waitingSince; ///< Layer arrival time
  };

  std::vector <RetxPdu> m_txedBuffer;  ///< Buffer for transmitted and retransmitted PDUs 
                                       ///< that have not been acked but are not considered 
                                       ///< for retransmission 
  std::vector <RetxPdu> m_retxBuffer;  ///< Buffer for PDUs considered for retransmission

    uint32_t m_maxTxBufferSize; ///< maximum transmission buffer size
    uint32_t m_txonBufferSize; ///< transmit on buffer size
    uint32_t m_retxBufferSize; ///< retransmit buffer size
    uint32_t m_txedBufferSize; ///< transmit ed buffer size

    bool     m_statusPduRequested; ///< status PDU requested
    uint32_t m_statusPduBufferSize; ///< status PDU buffer size

    /// PduBuffer structure
    struct PduBuffer
    {
      SequenceNumber10  m_seqNumber; ///< sequence number
      std::list < Ptr<Packet> >  m_byteSegments; ///< byte segments

      bool      m_pduComplete; ///< PDU complete?
    };

    std::map <uint16_t, PduBuffer > m_rxonBuffer; ///< Reception buffer

    Ptr<Packet> m_controlPduBuffer;               ///< Control PDU buffer (just one PDU)

    // SDU reassembly
//   std::vector < Ptr<Packet> > m_reasBuffer;     // Reassembling buffer
// 
    std::list < Ptr<Packet> > m_sdusBuffer;       ///< List of SDUs in a packet (PDU)

  /**
   * State variables. See section 7.1 in TS 36.322
   */
  // Transmitting side
  SequenceNumber10 m_vtA;                   ///< VT(A)
  SequenceNumber10 m_vtMs;                  ///< VT(MS)
  SequenceNumber10 m_vtS;                   ///< VT(S)
  SequenceNumber10 m_pollSn;                ///< POLL_SN

  // Receiving side
  SequenceNumber10 m_vrR;                   ///< VR(R)
  SequenceNumber10 m_vrMr;                  ///< VR(MR)
  SequenceNumber10 m_vrX;                   ///< VR(X)
  SequenceNumber10 m_vrMs;                  ///< VR(MS)
  SequenceNumber10 m_vrH;                   ///< VR(H)

  /**
   * Counters. See section 7.1 in TS 36.322
   */
  uint32_t m_pduWithoutPoll; ///< PDU without poll
  uint32_t m_byteWithoutPoll; ///< byte without poll

  /**
   * Constants. See section 7.2 in TS 36.322
   */
  uint16_t m_windowSize;

  /**
   * Timers. See section 7.3 in TS 36.322
   */
  EventId m_pollRetransmitTimer; ///< poll retransmit timer
  Time    m_pollRetransmitTimerValue; ///< poll retransmit time value
  EventId m_reorderingTimer; ///< reordering timer
  Time    m_reorderingTimerValue; ///< reordering timer value
  EventId m_statusProhibitTimer; ///< status prohibit timer
  Time    m_statusProhibitTimerValue; ///< status prohibit timer value
  EventId m_rbsTimer; ///< RBS timer
  Time    m_rbsTimerValue; ///< RBS timer value

  /**
   * Configurable parameters. See section 7.4 in TS 36.322
   */
  uint16_t m_maxRetxThreshold;  ///< \todo How these parameters are configured???
  uint16_t m_pollPdu; ///< poll PDU
  uint16_t m_pollByte; ///< poll byte
  
  bool m_txOpportunityForRetxAlwaysBigEnough; ///< transmit opportunity for retransmit? 
  bool m_pollRetransmitTimerJustExpired; ///< poll retransmit timer just expired?

  /**
   * SDU Reassembling state
   */
  typedef enum { NONE            = 0,
                 WAITING_S0_FULL = 1,
                 WAITING_SI_SF   = 2 } ReassemblingState_t;
  ReassemblingState_t m_reassemblingState; ///< reassembling state
  Ptr<Packet> m_keepS0; ///< keep S0

  /**
   * Expected Sequence Number
   */
  SequenceNumber10 m_expectedSeqNumber;

};


} // namespace ns3

#endif // LTE_RLC_AM_H
