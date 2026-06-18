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
 * Authors: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *          Keerthi Ganta <keerthig@ittc.ku.edu>
 *          Md Moshfequr Rahman <moshfequr@ittc.ku.edu>
 *          Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#ifndef TCPSCALABLE_H
#define TCPSCALABLE_H

#include "tcp-congestion-ops.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup congestionOps
 *
 * \brief An implementation of TCP Scalable
 *
 * Scalable improves TCP performance to better utilize the available bandwidth
 * of a highspeed wide area network by altering NewReno congestion window
 * adjustment algorithm.
 *
 * When congestion has not been detected, for each ACK received in an RTT,
 * Scalable increases its cwnd per:
 *
 *         cwnd = cwnd + 0.01                  (1)
 *
 * Following Linux implementation of Scalable, we use 50 instead of 100 to
 * account for delayed ACK.
 *
 * On the first detection of congestion in a given RTT, cwnd is reduced based
 * on the following equation:
 *
 *         cwnd = cwnd - ceil(0.125 * cwnd)       (2)
 *
 * More information: http://doi.acm.org/10.1145/956981.956989
 */

class TcpScalable : public TcpNewReno
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
  TcpScalable (void);

  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpScalable (const TcpScalable& sock);
  virtual ~TcpScalable (void);

  virtual std::string GetName () const;

  /**
   * \brief Get slow start threshold following Scalable principle (Equation 2)
   *
   * \param tcb internal congestion state
   * \param bytesInFlight bytes in flight
   *
   * \return the slow start threshold value
   */
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual Ptr<TcpCongestionOps> Fork ();

protected:
  /**
   * \brief Congestion avoidance of TcpScalable (Equation 1)
   *
   * \param tcb internal congestion state
   * \param segmentsAcked count of segments acked
   */
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb,
                                    uint32_t segmentsAcked);

private:
  uint32_t m_ackCnt;               //!< Number of received ACK
  uint32_t m_aiFactor;             //!< Additive increase factor
  double m_mdFactor;               //!< Multiplicative decrease factor
};

} // namespace ns3

#endif // TCPSCALABLE_H
