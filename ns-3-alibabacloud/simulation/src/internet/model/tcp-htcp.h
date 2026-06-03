/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 ResiliNets, ITTC, University of Kansas
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
 * by: Amir Modarresi <amodarresi@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#ifndef TCP_HTCP_H
#define TCP_HTCP_H

#include "tcp-congestion-ops.h"

namespace ns3 {

class TcpSocketState;

/**
 * \ingroup congestionOps
 *
 * \brief An implementation of the H-TCP variant of TCP.
 *
 * This class contains the H-TCP implementation of TCP, according to
 * Internet-Draft draft-leith-tcp-htcp-03 and its related paper,
 * "H-TCP: TCP for high-speed and long-distance networks"
 * H-TCP is a congestion control protocol suitable for high bandwidth-delay
 * product networks. It is fair to similar flows present in the network and 
 * also friendly with conventional TCP. It also makes use of free 
 * bandwidth when it is available.
 */
class TcpHtcp : public TcpNewReno
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
  TcpHtcp (void);
  /**
   * \brief Copy constructor
   * \param sock the object to copy
   */
  TcpHtcp (const TcpHtcp& sock);
  virtual ~TcpHtcp (void);
  virtual std::string GetName () const;
  virtual Ptr<TcpCongestionOps> Fork ();
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                          const Time &rtt);

protected:
  virtual void CongestionAvoidance (Ptr<TcpSocketState> tcb,
                                    uint32_t segmentsAcked);

private:
  /**
   * \brief Updates the additive increase parameter for H-TCP
   */
  void UpdateAlpha (void);

  /**
   * \brief Updates the multiplicative decrease factor beta for H-TCP
   */
  void UpdateBeta (void);

  // h-tcp variables
  double m_alpha;                    //!< AIMD additive increase parameter
  double m_beta;                     //!< AIMD multiplicative decrease factor
  double m_defaultBackoff;           //!< default value when throughput ratio less than default
  double m_throughputRatio;          //!< ratio of two consequence throughput
  Time m_delta;                      //!< Time in second that has elapsed since the
                                     //last congestion event experienced by a flow
  Time m_deltaL;                     //!< Threshold for switching between standard and new increase function
  Time m_lastCon;                    //!< Time of the last congestion for the flow
  Time m_minRtt;                     //!< Minimum RTT in each congestion period
  Time m_maxRtt;                     //!< Maximum RTT in each congestion period
  uint32_t m_throughput;             //!< Current throughput since last congestion
  uint32_t m_lastThroughput;         //!< Throughput in last congestion period
  uint32_t m_dataSent;               //!< Current amount of data sent since last congestion

};

} // namespace ns3

#endif /* TCP_HTCP_H */
