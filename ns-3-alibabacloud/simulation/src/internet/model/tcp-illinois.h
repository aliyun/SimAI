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
 * Author: Keerthi Ganta <keerthiganta@ku.edu>
 *         Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#ifndef TCPILLINOIS_H
#define TCPILLINOIS_H

#include "tcp-congestion-ops.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup congestionOps
 *
 * \brief An implementation of TCP Illinois algorithm
 *
 * TCP Illinois is a hybrid congestion control algorithm designed for
 * high-speed networks.  Illinois implements a Concave-AIMD (or C-AIMD)
 * algorithm that uses packet loss as the primary congestion signal to
 * determine the direction of window update and queueing delay as the
 * secondary congestion signal to determine the amount of change.
 *
 * The additive increase and multiplicative decrease factors (denoted as
 * alpha and beta, respectively) are functions of the current average queueing
 * delay da as shown in Equations (1) and (2).  To improve the protocol
 * robustness against sudden fluctuations in its delay sampling,
 * Illinois allows the increment of alpha to alphaMax
 * only if da stays below d1 for a some (theta) amount of time.
 *
 *                              / alphaMax          if da <= d1
 *                      alpha =                                         (1)
 *                              \ k1 / (k2 + da)    otherwise
 *
 *                             / betaMin            if da <= d2
 *                      beta =   k3 + k4da          if d2 < da < d3      (2)
 *                             \ betaMax            otherwise
 *
 *  where the calculations of k1, k2, k3, and k4 are shown in Equations (3), (4),
 *  (5), and (6).
 *
 *           k1 = (dm - d1)(alphaMin)alphaMax / (alphaMax - alphaMin)    (3)
 *
 *           k2 = ((dm - d1)alphaMin / (alphaMax - alphaMin)) - d1       (4)
 *
 *           k3 = ((alphaMin)d3 - (alphaMax)d2) / (d3 - d2)              (5)
 *
 *           k4 = (alphaMax - alphaMin) / (d3 - d2)                      (6)
 *
 *  with da the current average queueing delay calculated in Equation  (7) where:
 *  Ta is the average RTT (sumRtt / cntRtt in the implementation) and
 *  Tmin (baseRtt in the implementation) is the minimum RTT ever seen
 *  dm the maximum (average) queueing delay calculated in Equation (8) where
 *  Tmax (maxRtt in the implementation) is the maximum RTT ever seen
 *
 *           da = Ta - Tmin          (7)
 *
 *           dm = Tmax - Tmin         (8)
 *
 * di (i = 1,2,3) are calculated in Equation (9) (0 <= eta_1 < 1, and
 * 0 <= eta_2 <= eta_3 <=1)
 *
 *           di = (eta_i)dm            (9)
 *
 * Illinois only executes its adaptation of alpha and beta when cwnd exceeds a
 * threshold called winThresh.  Otherwise, it sets alpha and beta to the base
 * values of 1 and 0.5, respectively.
 *
 * Following the implementation of Illinois in the Linux kernel, we use the following
 * default parameter settings:
 *
 *       alphaMin = 0.3      (0.1 in the Illinois paper)
 *       alphaMax = 10.0
 *       betaMin = 0.125
 *       betaMax = 0.5
 *       winThresh = 15      (10 in the Illinois paper)
 *       theta = 5
 *       eta1 = 0.01
 *       eta2 = 0.1
 *       eta3 = 0.8
 *
 * More information: http://www.doi.org/10.1145/1190095.1190166
 */
class TcpIllinois : public TcpNewReno
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
  TcpIllinois (void);

  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpIllinois (const TcpIllinois& sock);
  virtual ~TcpIllinois (void);

  virtual std::string GetName () const;

  /**
   * \brief Get slow start threshold after congestion event
   *
   * \param tcb internal congestion state
   * \param bytesInFlight bytes in flight
   *
   * \return the slow start threshold value
   */
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual Ptr<TcpCongestionOps> Fork ();

  /**
    * \brief Reset Illinois parameters to default values upon a loss
    *
    * \param tcb internal congestion state
    * \param newState new congestion state to which the TCP is going to switch
    */
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState);

  /**
   * \brief Adjust cwnd following Illinois congestion avoidance algorithm
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments ACKed
   */
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);

  /**
   * \brief Measure RTT for each ACK
   * Keep track of min and max RTT
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments ACKed
   * \param rtt last RTT
   */
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                          const Time& rtt);

protected:
private:
  /**
   * \brief Recalculate alpha and beta every RTT
   *
   * \param cWnd current Cwnd (in bytes)
   */
  void RecalcParam (uint32_t cWnd);

  /**
   * \brief Calculate additive increase factor alpha
   *
   * If average queueing delay is at minimum, then alpha is set to alphaMax.
   * Otherwise, alpha is a decreasing function of average queueing delay.
   *
   * \param da current average queueing delay
   * \param dm maximum average queueing delay
   *
   */
  void CalculateAlpha (double da, double dm);

  /**
   * \brief Calculate multiplicative decrease factor beta
   *
   * If the current average queueing delay is <= 10% of max. (average) queueing delay,
   * beta is set to betaMin, which equals to 1/8 by default.
   * If the current average queueing delay is >= 80% of max. (average) queueing delay,
   * beta is set to betaMax, which equals to 1/2 by default.
   * Otherwise, beta is an increasing function of average queueing delay.
   *
   * \param da current average queueing delay
   * \param dm maximum average queueing delay
   *
   */
  void CalculateBeta (double da, double dm);

  /**
   * \brief Calculate average queueing delay
   *
   * \return average queueing delay da
   */
  Time CalculateAvgDelay () const;

  /**
   * \brief Calculate maximum queueing delay
   *
   * \return maximum average queueing delay dm
   */
  Time CalculateMaxDelay () const;

  /**
   * \brief Reset Illinois parameters
   *
   * \param nextTxSequence Next sequence to transmit
   */
  void Reset (const SequenceNumber32 &nextTxSequence);

private:
  Time m_sumRtt;             //!< Sum of all RTT measurements during last RTT
  uint32_t m_cntRtt;         //!< Number of RTT measurements during last RTT
  Time m_baseRtt;            //!< Minimum of all RTT measurements
  Time m_maxRtt;             //!< Maximum of all RTT measurements
  SequenceNumber32 m_endSeq; //!< Right edge of current RTT
  bool m_rttAbove;           //!< True when da > d1
  uint8_t m_rttLow;          //!< Number of RTTs da has stayed below d1
  double m_alphaMin;         //!< Minimum alpha threshold
  double m_alphaMax;         //!< Maximum alpha threshold
  double m_alphaBase;        //!< Base value of alpha for standard AIMD
  double m_alpha;            //!< Additive increase factor
  double m_betaMin;          //!< Minimum beta threshold
  double m_betaMax;          //!< Maximum beta threshold
  double m_betaBase;         //!< Base value of beta for standard AIMD
  double m_beta;             //!< Multiplicative decrease factor
  uint32_t m_winThresh;      //!< Window threshold for adaptive sizing
  uint32_t m_theta;          //!< Number of RTTs required before setting alpha to its max
  uint32_t m_ackCnt;         //!< Number of received ACK

};

} // namespace ns3

#endif // TCPILLINOIS_H
