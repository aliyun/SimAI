/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright © 2011 Marcos Talau
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
 * Author: Marcos Talau (talau@users.sourceforge.net)
 *
 * Thanks to: Duy Nguyen<duy@soe.ucsc.edu> by RED efforts in NS3
 *
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 * Copyright (c) 1990-1997 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor of the Laboratory may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * PORT NOTE: This code was ported from ns-2 (queue/red.h).  Almost all
 * comments also been ported from NS-2.
 * This implementation aims to be close to the results cited in [0]
 * [0] S.Floyd, K.Fall http://icir.org/floyd/papers/redsims.ps
 */

#ifndef RED_QUEUE_DISC_H
#define RED_QUEUE_DISC_H

#include "ns3/queue-disc.h"
#include "ns3/nstime.h"
#include "ns3/boolean.h"
#include "ns3/data-rate.h"
#include "ns3/random-variable-stream.h"

namespace ns3 {

class TraceContainer;

/**
 * \ingroup traffic-control
 *
 * \brief A RED packet queue disc
 */
class RedQueueDisc : public QueueDisc
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief RedQueueDisc Constructor
   *
   * Create a RED queue disc
   */
  RedQueueDisc ();

  /**
   * \brief Destructor
   *
   * Destructor
   */ 
  virtual ~RedQueueDisc ();

  /**
   * \brief Used in Feng's Adaptive RED
   */
  enum FengStatus
    {
      Above,        //!< When m_qAvg > m_maxTh
      Between,      //!< When m_maxTh < m_qAvg < m_minTh
      Below,        //!< When m_qAvg < m_minTh
    };

  /** 
   * \brief Drop types
   */
  enum
  {
    DTYPE_NONE,        //!< Ok, no drop
    DTYPE_FORCED,      //!< A "forced" drop
    DTYPE_UNFORCED,    //!< An "unforced" (random) drop
  };

   /**
    * \brief Set the alpha value to adapt m_curMaxP.
    *
    * \param alpha The value of alpha to adapt m_curMaxP.
    */
   void SetAredAlpha (double alpha);

   /**
    * \brief Get the alpha value to adapt m_curMaxP.
    *
    * \returns The alpha value to adapt m_curMaxP.
    */
   double GetAredAlpha (void);

   /**
    * \brief Set the beta value to adapt m_curMaxP.
    *
    * \param beta The value of beta to adapt m_curMaxP.
    */
   void SetAredBeta (double beta);

   /**
    * \brief Get the beta value to adapt m_curMaxP.
    *
    * \returns The beta value to adapt m_curMaxP.
    */
   double GetAredBeta (void);

   /**
    * \brief Set the alpha value to adapt m_curMaxP in Feng's Adaptive RED.
    *
    * \param a The value of alpha to adapt m_curMaxP in Feng's Adaptive RED.
    */
   void SetFengAdaptiveA (double a);

   /**
    * \brief Get the alpha value to adapt m_curMaxP in Feng's Adaptive RED.
    *
    * \returns The alpha value to adapt m_curMaxP in Feng's Adaptive RED.
    */
   double GetFengAdaptiveA (void);

   /**
    * \brief Set the beta value to adapt m_curMaxP in Feng's Adaptive RED.
    *
    * \param b The value of beta to adapt m_curMaxP in Feng's Adaptive RED.
    */
   void SetFengAdaptiveB (double b);

   /**
    * \brief Get the beta value to adapt m_curMaxP in Feng's Adaptive RED.
    *
    * \returns The beta value to adapt m_curMaxP in Feng's Adaptive RED.
    */
   double GetFengAdaptiveB (void);

  /**
   * \brief Set the thresh limits of RED.
   *
   * \param minTh Minimum thresh in bytes or packets.
   * \param maxTh Maximum thresh in bytes or packets.
   */
  void SetTh (double minTh, double maxTh);

 /**
  * Assign a fixed random variable stream number to the random variables
  * used by this model.  Return the number of streams (possibly zero) that
  * have been assigned.
  *
  * \param stream first stream index to use
  * \return the number of stream indices assigned by this model
  */
  int64_t AssignStreams (int64_t stream);

  // Reasons for dropping packets
  static constexpr const char* UNFORCED_DROP = "Unforced drop";  //!< Early probability drops
  static constexpr const char* FORCED_DROP = "Forced drop";      //!< Forced drops, m_qAvg > m_maxTh
  // Reasons for marking packets
  static constexpr const char* UNFORCED_MARK = "Unforced mark";  //!< Early probability marks
  static constexpr const char* FORCED_MARK = "Forced mark";      //!< Forced marks, m_qAvg > m_maxTh

protected:
  /**
   * \brief Dispose of the object
   */
  virtual void DoDispose (void);

private:
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual Ptr<const QueueDiscItem> DoPeek (void);
  virtual bool CheckConfig (void);

  /**
   * \brief Initialize the queue parameters.
   *
   * Note: if the link bandwidth changes in the course of the
   * simulation, the bandwidth-dependent RED parameters do not change.
   * This should be fixed, but it would require some extra parameters,
   * and didn't seem worth the trouble...
   */
  virtual void InitializeParams (void);
  /**
   * \brief Compute the average queue size
   * \param nQueued number of queued packets
   * \param m simulated number of packets arrival during idle period
   * \param qAvg average queue size
   * \param qW queue weight given to cur q size sample
   * \returns new average queue size
   */
  double Estimator (uint32_t nQueued, uint32_t m, double qAvg, double qW);
   /**
    * \brief Update m_curMaxP
    * \param newAve new average queue length
    */
  void UpdateMaxP (double newAve);
   /**
    * \brief Update m_curMaxP based on Feng's Adaptive RED
    * \param newAve new average queue length
    */
  void UpdateMaxPFeng (double newAve);
  /**
   * \brief Check if a packet needs to be dropped due to probability mark
   * \param item queue item
   * \param qSize queue size
   * \returns 0 for no drop/mark, 1 for drop
   */
  uint32_t DropEarly (Ptr<QueueDiscItem> item, uint32_t qSize);
  /**
   * \brief Returns a probability using these function parameters for the DropEarly function
   * \returns Prob. of packet drop before "count"
   */
  double CalculatePNew (void);
  /**
   * \brief Returns a probability using these function parameters for the DropEarly function
   * \param p Prob. of packet drop before "count"
   * \param size packet size
   * \returns Prob. of packet drop
   */
  double ModifyP (double p, uint32_t size);

  // ** Variables supplied by user
  uint32_t m_meanPktSize;   //!< Avg pkt size
  uint32_t m_idlePktSize;   //!< Avg pkt size used during idle times
  bool m_isWait;            //!< True for waiting between dropped packets
  bool m_isGentle;          //!< True to increase dropping prob. slowly when m_qAvg exceeds m_maxTh
  bool m_isARED;            //!< True to enable Adaptive RED
  bool m_isAdaptMaxP;       //!< True to adapt m_curMaxP
  double m_minTh;           //!< Minimum threshold for m_qAvg (bytes or packets)
  double m_maxTh;           //!< Maximum threshold for m_qAvg (bytes or packets), should be >= 2 * m_minTh
  double m_qW;              //!< Queue weight given to cur queue size sample
  double m_lInterm;         //!< The max probability of dropping a packet
  Time m_targetDelay;       //!< Target average queuing delay in ARED
  Time m_interval;          //!< Time interval to update m_curMaxP
  double m_top;             //!< Upper bound for m_curMaxP in ARED
  double m_bottom;          //!< Lower bound for m_curMaxP in ARED
  double m_alpha;           //!< Increment parameter for m_curMaxP in ARED
  double m_beta;            //!< Decrement parameter for m_curMaxP in ARED
  Time m_rtt;               //!< Rtt to be considered while automatically setting m_bottom in ARED
  bool m_isFengAdaptive;    //!< True to enable Feng's Adaptive RED
  bool m_isNonlinear;       //!< True to enable Nonlinear RED
  double m_b;               //!< Increment parameter for m_curMaxP in Feng's Adaptive RED
  double m_a;               //!< Decrement parameter for m_curMaxP in Feng's Adaptive RED
  bool m_isNs1Compat;       //!< Ns-1 compatibility
  DataRate m_linkBandwidth; //!< Link bandwidth
  Time m_linkDelay;         //!< Link delay
  bool m_useEcn;            //!< True if ECN is used (packets are marked instead of being dropped)
  bool m_useHardDrop;       //!< True if packets are always dropped above max threshold

  // ** Variables maintained by RED
  double m_vA;              //!< 1.0 / (m_maxTh - m_minTh)
  double m_vB;              //!< -m_minTh / (m_maxTh - m_minTh)
  double m_vC;              //!< (1.0 - m_curMaxP) / m_maxTh - used in "gentle" mode
  double m_vD;              //!< 2.0 * m_curMaxP - 1.0 - used in "gentle" mode
  double m_curMaxP;         //!< Current max_p
  Time m_lastSet;           //!< Last time m_curMaxP was updated
  double m_vProb;           //!< Prob. of packet drop
  uint32_t m_countBytes;    //!< Number of bytes since last drop
  uint32_t m_old;           //!< 0 when average queue first exceeds threshold
  uint32_t m_idle;          //!< 0/1 idle status
  double m_ptc;             //!< packet time constant in packets/second
  double m_qAvg;            //!< Average queue length
  uint32_t m_count;         //!< Number of packets since last random number generation
  FengStatus m_fengStatus;  //!< For use in Feng's Adaptive RED
  /**
   * 0 for default RED
   * 1 experimental (see red-queue-disc.cc)
   * 2 experimental (see red-queue-disc.cc)
   * 3 use Idle packet size in the ptc
   */
  uint32_t m_cautious;
  Time m_idleTime;          //!< Start of current idle period

  Ptr<UniformRandomVariable> m_uv;  //!< rng stream
};

}; // namespace ns3

#endif // RED_QUEUE_DISC_H
