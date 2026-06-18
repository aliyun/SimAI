/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
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
 * Author: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#ifndef TCPYEAH_H
#define TCPYEAH_H

#include "ns3/tcp-scalable.h"
#include "ns3/tcp-recovery-ops.h"

namespace ns3 {

/**
 * \ingroup congestionOps
 *
 * \brief An implementation of TCP YeAH
 *
 *  YeAH-TCP (Yet Another HighSpeed TCP) is a heuristic designed to balance various
 *  requirements of a state-of-the-art congestion control algorithm:
 *  1) fully exploit the link capacity of high BDP networks while inducing a small
 *  number of congestion events
 *  2) compete friendly with Reno flows
 *  3) achieve intra and RTT fairness
 *  4) robust to random losses
 *  5) achieve high performance regardless of buffer size
 *
 * YeAH operates between 2 modes: Fast and Slow mode.  In the Fast mode when the
 * queue occupancy is small and the network congestion level is low, YeAH
 * increments its congestion window according to the aggressive STCP rule.
 * When the number of packets in the queue grows beyond a threshold and the
 * network congestion level is high, YeAH enters its Slow mode, acting as Reno
 * with a decongestion algorithm.  YeAH employs Vegas' mechanism for calculating
 * the backlog as in Equation (1).  The estimation of the network congestion
 * level is shown in Equation (2).
 *
 *                      Q = (RTT - BaseRTT) (cwnd / RTT)    (1)
 *                      L = (RTT - BaseRTT) / BaseRTT       (2)
 *
 * To ensure TCP friendliness, YeAH also implements an algorithm to detect the
 * presence of legacy Reno flows. Upon the receipt of 3 duplicate ACKs,
 * YeAH decreases its slow start threshold according to Equation (3) if
 * it's not competing with Reno flows.  Otherwise,  the ssthresh is halved
 * as in Reno.
 *
 *                      ssthresh = min{max{cwnd/8, Q}, cwnd/2}
 *
 * More information: http://www.csc.lsu.edu/~sjpark/cs7601/4-YeAH_TCP.pdf
 */

class TcpYeah : public TcpNewReno
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Create an unbound tcp socket.
   */
  TcpYeah (void);

  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpYeah (const TcpYeah& sock);
  virtual ~TcpYeah (void);

  virtual std::string GetName () const;

  /**
   * \brief Compute RTTs needed to execute YeAH algorithm
   *
   * The function filters RTT samples from the last RTT to find
   * the current smallest propagation delay + queueing delay (minRtt).
   * We take the minimum to avoid the effects of delayed ACKs.
   *
   * The function also min-filters all RTT measurements seen to find the
   * propagation delay (baseRtt).
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments ACKed
   * \param rtt last RTT
   *
   */
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                          const Time& rtt);

  /**
   * \brief Enable/disable YeAH algorithm depending on the congestion state
   *
   * We only start a YeAH cycle when we are in normal congestion state (CA_OPEN state).
   *
   * \param tcb internal congestion state
   * \param newState new congestion state to which the TCP is going to switch
   */
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState);

  /**
   * \brief Adjust cwnd following YeAH dual-mode algorithm
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments ACKed
   */
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

  /**
   * \brief Get slow start threshold upon the receipt of 3 dupACKs
   *
   * \param tcb internal congestion state
   * \param bytesInFlight number of outstanding bytes
   *
   * \return the slow start threshold value
   */
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual Ptr<TcpCongestionOps> Fork ();

protected:
private:
  /**
   * \brief Enable YeAH algorithm to start taking YeAH samples
   *
   * YeAH algorithm is enabled in the following situations:
   * 1. at the establishment of a connection
   * 2. after an RTO
   * 3. after fast recovery
   * 4. when an idle connection is restarted
   *
   * \param nextTxSequence Sequence to transmit next
   */
  void EnableYeah (const SequenceNumber32 &nextTxSequence);

  /**
   * \brief Stop taking YeAH samples
   */
  void DisableYeah ();

private:
  uint32_t m_alpha;   //!< Maximum backlog allowed at the bottleneck queue; Q_max in the paper
  uint32_t m_gamma;   //!< Fraction of queue to be removed per RTT when precautionary decongestion executed
  uint32_t m_delta;   //!< Log minimum fraction of cwnd to be removed on loss
  uint32_t m_epsilon; //!< Log maximum fraction to be removed on early decongestion
  uint32_t m_phy;     //!< Maximum delta from base
  uint32_t m_rho;     //!< Minimum number of consecutive RTT to consider competition with Reno flows on loss
  uint32_t m_zeta;    //!< Minimum number of state switches to reset m_renoCount

  uint32_t m_stcpAiFactor;  //!< STCP additive increase parameter
  Ptr<TcpScalable> m_stcp;  //!< TcpScalable object
  Time m_baseRtt;           //!< Minimum of all YeAH RTT measurements seen during connection
  Time m_minRtt;            //!< Minimum of all RTTs measured within last RTT
  uint32_t m_cntRtt;        //!< Number of RTT measurements during last RTT
  bool m_doingYeahNow;          //!< If true, do YeAH for this RTT
  SequenceNumber32 m_begSndNxt; //!< Right edge during last RTT
  uint32_t m_lastQ;             //!< Last number of packets in the bottleneck queue
  uint32_t m_doingRenoNow;      //!< Number of RTTs in "Slow" mode
  uint32_t m_renoCount;         //!< Estimated cwnd of competing Reno flow
  uint32_t m_fastCount;         //!< Number of RTTs in "Fast" mode
};

} // namespace ns3

#endif // TCPYEAH_H
